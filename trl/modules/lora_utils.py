"""Low-Rank Adaptation (LoRA) utilities for SONIC fine-tuning.

Variants supported:
  - vanilla LoRA   : output = base(x) + B (A x) * (alpha / rank)
  - rsLoRA         : same shape, but scaling = alpha / sqrt(rank); more stable at large rank
  - DoRA           : weight = m * normalize(W0 + B A); learns magnitude + direction
                     (Liu et al. 2024, https://arxiv.org/abs/2402.09353)

Per-adapter options:
  - learnable gate (scalar) — useful for warmstart and ablation
  - per-layer rank/alpha (dict) — different ranks per Linear inside an MLP
  - target_layer_filter — callable(idx, layer)->bool to pick which Linears get adapters

Helpers:
  - save_lora_state_dict / load_lora_state_dict — small (MB-scale) ckpts containing
    only LoRA + adapter-bridge params
  - merge_lora_weights — bake adapters back into nn.Linear for deployment
  - print_lora_summary — pretty-print which modules were touched and trainable %.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== Adapter classes ==============================


class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear with a low-rank adapter (vanilla or rsLoRA).

    output = frozen_linear(x) + (B (A x)) * scaling [* gate]

    The original linear weights are frozen; only A, B (and optionally gate)
    are trainable. A is initialized via Kaiming-uniform on a virtual fan_in=rank
    (PEFT convention); B is zero so the adapter starts as a no-op.

    Args:
        original_linear: nn.Linear to wrap (frozen in place).
        rank: rank of the low-rank decomposition.
        alpha: scaling factor numerator.
        dropout: dropout probability on adapter input.
        rslora: if True, use alpha/sqrt(rank) instead of alpha/rank.
        use_gate: if True, multiply the adapter contribution by a learnable
            scalar (init 1.0). Useful for ablations and warmstart.
        init_strategy: "kaiming" (default, PEFT-compatible) or "normal_0.02".
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        rslora: bool = False,
        use_gate: bool = False,
        init_strategy: str = "kaiming",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / (math.sqrt(rank) if rslora else rank)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        if init_strategy == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif init_strategy == "normal_0.02":
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if use_gate:
            self.gate = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("gate", None)

        # Inference fast-path: when forward runs under no_grad, fold the LoRA
        # delta into a single (W + scaling * B A) matmul instead of base + AB
        # extras. A/B don't change during rollout, so caching is safe; we mark
        # the cache dirty in every grad-enabled forward (cheap, one bool set)
        # and lazily refresh on the next no_grad forward.
        self._inference_merge_enabled: bool = True
        self._cache_dirty: bool = True
        self._cached_merged_weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._inference_merge_enabled and not torch.is_grad_enabled():
            if self._cache_dirty or self._cached_merged_weight is None:
                with torch.no_grad():
                    self._cached_merged_weight = self.merged_weight().detach()
                    self._cache_dirty = False
            return F.linear(x, self._cached_merged_weight, self.original.bias)

        # Training path — mark cache dirty so the next rollout refreshes it.
        self._cache_dirty = True
        base_out = self.original(x)
        delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        if self.gate is not None:
            delta = delta * self.gate
        return base_out + delta

    def merged_weight(self) -> torch.Tensor:
        """Return W0 + B A * scaling [* gate] for deployment merge."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        if self.gate is not None:
            delta = delta * self.gate
        return self.original.weight + delta

    def refresh_merged_cache(self) -> None:
        """Recompute (W + scaling * B A) cache. Idempotent. Cheap (one small matmul)."""
        with torch.no_grad():
            self._cached_merged_weight = self.merged_weight().detach()
            self._cache_dirty = False

    def extra_repr(self) -> str:
        gate_str = ", gated" if self.gate is not None else ""
        return (
            f"in={self.original.in_features}, out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}{gate_str}"
        )


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) wrapper.

    Decomposes pretrained weight W0 into magnitude m (per output unit) and
    direction V0 = W0 / ||W0||_col, then learns:
        W' = m * (V0 + B A * scaling) / ||V0 + B A * scaling||_col

    where ||.||_col is the column-wise (per output unit) L2 norm.

    DoRA gives the optimizer separate control over each output unit's magnitude
    versus direction, often improving fine-tuning quality at the same param count
    as LoRA. Liu et al. 2024 (https://arxiv.org/abs/2402.09353).

    Args:
        original_linear: nn.Linear to wrap (frozen).
        rank, alpha, dropout, rslora: same as LoRALinear.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        rslora: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"DoRA rank must be positive, got {rank}")
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / (math.sqrt(rank) if rslora else rank)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Magnitude vector (per output unit). Initialized from frozen weight's
        # column norms so DoRA starts as identity.
        with torch.no_grad():
            magnitude = original_linear.weight.norm(p=2, dim=1)
        self.magnitude = nn.Parameter(magnitude.clone())

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # See LoRALinear for the inference-merge cache rationale.
        self._inference_merge_enabled: bool = True
        self._cache_dirty: bool = True
        self._cached_merged_weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._inference_merge_enabled and not torch.is_grad_enabled():
            if self._cache_dirty or self._cached_merged_weight is None:
                with torch.no_grad():
                    self._cached_merged_weight = self.merged_weight().detach()
                    self._cache_dirty = False
            return F.linear(x, self._cached_merged_weight, self.original.bias)

        self._cache_dirty = True
        # Combined adapter weight before normalization
        weight = self.original.weight + (self.lora_B @ self.lora_A) * self.scaling
        # Per-row L2 norm (along input-feature axis)
        norm = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        # Re-normalize then re-scale per output unit
        weight_normed = weight / norm * self.magnitude.unsqueeze(1)
        return F.linear(self.dropout(x), weight_normed, self.original.bias)

    def merged_weight(self) -> torch.Tensor:
        """Return the fully combined weight, ready to overwrite a Linear."""
        weight = self.original.weight + (self.lora_B @ self.lora_A) * self.scaling
        norm = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return weight / norm * self.magnitude.unsqueeze(1)

    def refresh_merged_cache(self) -> None:
        """Recompute the normalized merged weight cache."""
        with torch.no_grad():
            self._cached_merged_weight = self.merged_weight().detach()
            self._cache_dirty = False

    def extra_repr(self) -> str:
        return (
            f"in={self.original.in_features}, out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}, dora=True"
        )


# ============================== Injection helpers ==============================


def _make_adapter(
    layer: nn.Linear,
    rank: int,
    alpha: float,
    dropout: float,
    variant: str,
    rslora: bool,
    use_gate: bool,
    init_strategy: str,
) -> nn.Module:
    if variant == "lora":
        return LoRALinear(
            layer,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            rslora=rslora,
            use_gate=use_gate,
            init_strategy=init_strategy,
        )
    if variant == "dora":
        return DoRALinear(
            layer,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            rslora=rslora,
        )
    raise ValueError(f"Unknown LoRA variant: {variant!r}. Expected 'lora' or 'dora'.")


def apply_lora_to_sequential(
    module: nn.Sequential,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    skip_last: bool = False,
    *,
    variant: str = "lora",
    rslora: bool = False,
    use_gate: bool = False,
    init_strategy: str = "kaiming",
    per_layer_ranks: Optional[dict] = None,
    per_layer_alphas: Optional[dict] = None,
    target_layer_filter: Optional[Callable[[int, nn.Module], bool]] = None,
    linear_active_indices: Optional[list] = None,
) -> nn.Sequential:
    """Inject LoRA / DoRA adapters into all (or selected) nn.Linear layers.

    Args:
        module: nn.Sequential modified in place.
        rank, alpha, dropout: defaults for layers without per-layer overrides.
        skip_last: if True, do not adapt the final Linear.
        variant: "lora" or "dora".
        rslora: if True, scale by alpha/sqrt(rank). Ignored for DoRA-internal
            scaling decision (still honored on the inner low-rank component).
        use_gate: per-adapter learnable scalar (LoRA only).
        init_strategy: A-matrix init for LoRA ("kaiming" | "normal_0.02").
        per_layer_ranks: optional dict {seq_index: rank_override}.
        per_layer_alphas: optional dict {seq_index: alpha_override}.
        target_layer_filter: optional callable(seq_index, layer) -> bool. If
            provided, only layers returning True are adapted; ``skip_last`` and
            per-layer dicts are still respected for the survivors.
        linear_active_indices: optional list of Linear-positional indices
            (0-based, counting only nn.Linear sublayers). Negative indices
            are Python-style (``-1`` = last Linear). When set, ONLY these
            Linears receive LoRA; everything else stays as a plain (frozen)
            Linear. Combines with the other filters by intersection.

            Memory note: concentrating LoRA on the tail of the Sequential
            lets autograd short-circuit the backward through the frozen
            prefix when nothing upstream needs grad — those layers' inputs
            are then NEVER cached.

    Returns:
        The same module with adapters injected.
    """
    linear_indices = [i for i, layer in enumerate(module) if isinstance(layer, nn.Linear)]
    num_linears = len(linear_indices)
    if linear_active_indices is not None:
        resolved = set()
        for raw in linear_active_indices:
            i = int(raw)
            if i < 0:
                i += num_linears
            if 0 <= i < num_linears:
                resolved.add(i)
        linear_indices = [linear_indices[i] for i in sorted(resolved)]
    if skip_last and linear_indices:
        linear_indices = linear_indices[:-1]
    if target_layer_filter is not None:
        linear_indices = [i for i in linear_indices if target_layer_filter(i, module[i])]

    per_layer_ranks = per_layer_ranks or {}
    per_layer_alphas = per_layer_alphas or {}

    for idx in linear_indices:
        original_linear = module[idx]
        layer_rank = int(per_layer_ranks.get(idx, rank))
        layer_alpha = float(per_layer_alphas.get(idx, alpha))
        module[idx] = _make_adapter(
            original_linear,
            rank=layer_rank,
            alpha=layer_alpha,
            dropout=dropout,
            variant=variant,
            rslora=rslora,
            use_gate=use_gate,
            init_strategy=init_strategy,
        )

    return module


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def cast_frozen_to_dtype(module: nn.Module, dtype: torch.dtype) -> int:
    """Cast every ``requires_grad=False`` fp32 parameter and fp32 buffer to ``dtype``.

    LoRA-specific lever: the frozen backbone never receives gradients, so we
    can store its weights in lower precision (typically bf16) without the
    fp32 master-weight that full fine-tuning would need. Skips parameters
    that already require grad (LoRA A/B, gates, magnitudes, proprio_proj).

    Returns the number of tensors touched (for logging).
    """
    n = 0
    for p in module.parameters():
        if (not p.requires_grad) and p.dtype == torch.float32:
            p.data = p.data.to(dtype)
            n += 1
    for b in module.buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(dtype)
            n += 1
    return n


def set_lora_inference_merge(module: nn.Module, enabled: bool) -> None:
    """Toggle the merged-weight inference fast-path on every LoRA/DoRA adapter.

    When enabled (default), no-grad forward passes use a cached
    ``(W + scaling * B A)`` and run a single F.linear instead of base + AB
    extras. Disable only if you suspect a numerical mismatch — autograd-side
    forward is unaffected either way.
    """
    for m in module.modules():
        if isinstance(m, (LoRALinear, DoRALinear)):
            m._inference_merge_enabled = enabled


def refresh_lora_merged_cache(module: nn.Module) -> None:
    """Eagerly recompute the merged-weight cache on every LoRA/DoRA adapter.

    Normally unnecessary: the cache is auto-refreshed on the first no-grad
    forward after any training-mode forward. Provided for callers that want
    to amortize the recompute outside the hot rollout loop.
    """
    for m in module.modules():
        if isinstance(m, (LoRALinear, DoRALinear)):
            m.refresh_merged_cache()


def count_parameters(module: nn.Module):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ============================== Save / load / merge ==============================


def is_lora_param_name(name: str) -> bool:
    """True if a state-dict key belongs to a LoRA/DoRA adapter (or its gate/magnitude)."""
    # Match anywhere in the dotted path: lora_A, lora_B, gate, magnitude (DoRA),
    # plus the small bridge modules we save alongside (proprio_proj, dof_bridge).
    patterns = (
        r"\.lora_A$",
        r"\.lora_B$",
        r"\.gate$",
        r"\.magnitude$",
        r"^(?:.*\.)?proprio_proj\.",
        r"^(?:.*\.)?dof_bridge\.",
    )
    return any(re.search(p, name) for p in patterns)


def save_lora_state_dict(module: nn.Module, path: str) -> dict:
    """Save only LoRA / adapter / bridge parameters to ``path``.

    Returns the dict that was written. Resulting checkpoint is small (MB-scale),
    intended for sharing fine-tuned adapters without the frozen backbone.
    """
    full_sd = module.state_dict()
    lora_sd = {k: v.detach().cpu() for k, v in full_sd.items() if is_lora_param_name(k)}
    torch.save(lora_sd, path)
    return lora_sd


def load_lora_state_dict(module: nn.Module, path: str, strict: bool = False) -> tuple:
    """Load a LoRA-only state_dict produced by :func:`save_lora_state_dict`.

    Returns ``(missing_keys, unexpected_keys)`` from the underlying load.
    Non-LoRA params in ``module`` are left untouched.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(sd, strict=strict)
    return missing, unexpected


def merge_lora_weights(module: nn.Module) -> nn.Module:
    """Replace every LoRALinear / DoRALinear in ``module`` with a plain nn.Linear.

    The merged Linear's weight = adapter.merged_weight(). Useful for export
    (ONNX, C++ deploy) once training is done — removes adapter overhead.

    Modifies ``module`` in-place and returns it.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRALinear, DoRALinear)):
            base = child.original
            merged = nn.Linear(
                base.in_features, base.out_features, bias=base.bias is not None
            ).to(base.weight.device, base.weight.dtype)
            with torch.no_grad():
                merged.weight.copy_(child.merged_weight())
                if base.bias is not None:
                    merged.bias.copy_(base.bias)
            setattr(module, name, merged)
        else:
            merge_lora_weights(child)
    return module


# ============================== Diagnostics ==============================


def print_lora_summary(module: nn.Module, root_name: str = "model") -> None:
    """Pretty-print every LoRA/DoRA adapter, its parent path, rank, scaling, params."""
    rows = []
    for name, child in module.named_modules():
        if isinstance(child, (LoRALinear, DoRALinear)):
            inferred_rank = child.rank
            scaling = child.scaling
            n_params = sum(
                p.numel() for p in child.parameters() if p.requires_grad
            )
            kind = "DoRA" if isinstance(child, DoRALinear) else "LoRA"
            gate = (
                "gate=on" if isinstance(child, LoRALinear) and child.gate is not None
                else "gate=off"
            )
            rows.append(
                (name, kind, child.original.in_features, child.original.out_features,
                 inferred_rank, scaling, n_params, gate)
            )
    if not rows:
        print(f"[{root_name}] no LoRA/DoRA adapters found")  # noqa: T201
        return

    total_trainable = sum(r[6] for r in rows)
    total_module, trainable_module = count_parameters(module)
    print(f"[{root_name}] {len(rows)} adapter(s), "  # noqa: T201
          f"{total_trainable:,} adapter params, "
          f"{trainable_module:,}/{total_module:,} ({100*trainable_module/max(1,total_module):.2f}%) trainable overall")
    for name, kind, in_f, out_f, r, s, n, g in rows:
        print(f"  {kind:4s}  {in_f:5d} → {out_f:5d}  rank={r:3d}  scaling={s:.3f}  "  # noqa: T201
              f"params={n:6d}  {g}  @ {name}")
