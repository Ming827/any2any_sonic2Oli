"""One-shot GPU memory breakdown.

Hooks ``on_step_end`` to dump a categorized memory snapshot after a fixed
iteration, then exits. Use to answer "where does my GPU memory go?".

Usage (one-off profiling):
    Add to defaults in your exp yaml:
        - /callbacks/memory_profile@callbacks.memory_profile

    Or wire inline:
        callbacks:
          memory_profile:
            _target_: gear_sonic.trl.callbacks.memory_profile_callback.MemoryProfileCallback
            dump_at_iter: 3
            exit_after_dump: false

Categories:
  - Actor backbone params (frozen + trainable Linear/Norm/etc, excluding LoRA)
  - Actor LoRA params (lora_A / lora_B / gate / DoRA magnitude)
  - Actor wrapper extras (DofBridge / proprio_proj / hip-decomp params)
  - Critic backbone params (frozen Linear/Norm)
  - Critic LoRA params
  - Other params (anything we couldn't classify — print names for follow-up)
  - Gradient buffers (sum of ``p.grad.numel() * p.element_size()``)
  - Adam state (sum of ``optimizer.state[p]`` tensors)
  - Rollout buffer estimate (num_envs × num_steps × obs_dim × 4B)
  - "Other PyTorch allocations" (= allocated - sum-of-above; activation cache,
    NCCL staging, motion lib tensors, ObservationManager cache, ...)
  - PyTorch reserved-but-unallocated (allocator fragmentation)
  - Non-PyTorch (= nvidia-smi process used - torch reserved; Isaac Sim physics,
    CUDA libs, NCCL buffers, kernel cache, sibling-rank contexts)
"""

from __future__ import annotations

import subprocess
import sys

import torch
from loguru import logger
from transformers import TrainerCallback


def _query_nvidia_smi_used_for_pid(pid: int) -> int:
    """Return the GPU memory (bytes) reported by nvidia-smi for THIS process.

    Falls back to total GPU 0 used if per-process lookup fails. Multi-GPU
    setups call once per rank; each rank only owns one GPU after the
    CUDA_VISIBLE_DEVICES isolation we set in train_agent_trl.py.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True, timeout=5,
        )
        for line in out.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return int(parts[1]) * 1024 * 1024  # MiB → bytes
    except Exception:  # noqa: BLE001
        pass
    # Fallback: total used on the visible GPU (may be inflated by other procs).
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return int(out.stdout.strip().split("\n")[0]) * 1024 * 1024
    except Exception:  # noqa: BLE001
        return 0


def _classify_param(name: str) -> str:
    """Map a named_parameter path to one of our categories."""
    n = name.lower()
    is_lora = ("lora_a" in n) or ("lora_b" in n) or (".gate" == n[-5:]) or ("magnitude" in n)
    is_wrapper_extra = (
        "dof_bridge" in n
        or "proprio_proj" in n
        or "head_default" in n
    )
    is_critic = ("value_model" in n) or (".critic" in n) or n.startswith("critic")
    is_actor = ("policy" in n) or (".actor" in n) or n.startswith("actor")
    if is_critic:
        return "critic_lora" if is_lora else "critic_backbone"
    if is_actor:
        if is_wrapper_extra:
            return "actor_wrapper_extras"
        return "actor_lora" if is_lora else "actor_backbone"
    return "other_params"


class MemoryProfileCallback(TrainerCallback):
    """Dump a categorized GPU memory breakdown after a chosen PPO iteration."""

    def __init__(
        self,
        dump_at_iter: int = 3,
        exit_after_dump: bool = False,
        rollout_obs_dim_hint: int | None = None,
    ):
        self.dump_at_iter = dump_at_iter
        self.exit_after_dump = exit_after_dump
        self.rollout_obs_dim_hint = rollout_obs_dim_hint
        self._dumped = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._dumped:
            return
        if state.global_step < self.dump_at_iter:
            return
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        env = kwargs.get("env")
        if model is None or optimizer is None:
            return

        try:
            self._dump(model, optimizer, env, state)
        finally:
            self._dumped = True
            if self.exit_after_dump:
                logger.warning("MemoryProfileCallback: exit_after_dump=True, terminating.")
                sys.exit(0)

    def _dump(self, model, optimizer, env, state):
        device = next(model.parameters()).device
        cats = {
            "actor_backbone": 0,
            "actor_lora": 0,
            "actor_wrapper_extras": 0,
            "critic_backbone": 0,
            "critic_lora": 0,
            "other_params": 0,
        }
        unclassified_names = []

        param_grad_bytes = 0
        adam_state_bytes = 0

        for name, p in model.named_parameters():
            sz = p.numel() * p.element_size()
            cat = _classify_param(name)
            cats[cat] += sz
            if cat == "other_params":
                unclassified_names.append(name)
            if p.grad is not None:
                param_grad_bytes += p.grad.numel() * p.grad.element_size()
            # Adam state for this param
            opt_state = optimizer.state.get(p, {})
            for v in opt_state.values():
                if torch.is_tensor(v):
                    adam_state_bytes += v.numel() * v.element_size()

        # Rollout buffer rough estimate
        rollout_bytes = 0
        try:
            num_envs = int(getattr(env, "num_envs", 0))
            # Pull from trainer if available
            num_steps = int(state.__dict__.get("num_steps_per_env", 24))
            obs_dim = self.rollout_obs_dim_hint
            if obs_dim is None:
                # Try to infer from env
                obs_dim = 0
                try:
                    for k, t in env.obs_buf_dict.items():
                        obs_dim += t.shape[-1] if t.ndim >= 2 else t.shape[0]
                except Exception:  # noqa: BLE001
                    pass
            if num_envs and num_steps and obs_dim:
                # obs + actions + reward + value + log_prob, ~obs_dim + ~70 dim of bookkeeping
                rollout_bytes = num_envs * num_steps * (obs_dim + 100) * 4
        except Exception:  # noqa: BLE001
            pass

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_alloc = torch.cuda.max_memory_allocated(device)

        # Process's view of GPU
        import os as _os
        proc_used = _query_nvidia_smi_used_for_pid(_os.getpid())
        if proc_used == 0:
            proc_used = reserved  # fallback so percentages still compute

        # Activation / other allocator = allocated minus params/grad/adam
        param_total = sum(cats.values())
        accounted = param_total + param_grad_bytes + adam_state_bytes + rollout_bytes
        activations_etc = max(0, allocated - accounted)
        fragmentation = max(0, reserved - allocated)
        non_pytorch = max(0, proc_used - reserved)

        def pct(b):
            return 100 * b / proc_used if proc_used else 0.0

        def mb(b):
            return b / (1024 * 1024)

        def line(label, b, depth=0):
            return f"{'  ' * depth}{label:32s} {mb(b):10.1f} MB    {pct(b):6.2f}%"

        out = []
        out.append("\n" + "=" * 70)
        out.append(f"  GPU MEMORY BREAKDOWN @ iter {state.global_step}")
        out.append(f"  total process used: {mb(proc_used):.1f} MB  (peak alloc: {mb(max_alloc):.1f} MB)")
        out.append("=" * 70)
        out.append(line("PyTorch reserved (pool):", reserved))
        out.append(line("├─ Allocated (in-use):", allocated, 0))
        out.append(line("│  ├─ Params:", param_total, 0))
        for k, v in cats.items():
            if v > 0:
                out.append(line(f"│  │  ├─ {k}:", v, 0))
        out.append(line("│  ├─ Gradients:", param_grad_bytes, 0))
        out.append(line("│  ├─ Adam optim state:", adam_state_bytes, 0))
        if rollout_bytes > 0:
            out.append(line("│  ├─ Rollout buffer (est):", rollout_bytes, 0))
        out.append(line("│  └─ Activations / NCCL / etc:", activations_etc, 0))
        out.append(line("└─ Fragmentation (reserved-unalloc):", fragmentation))
        out.append(line("Non-PyTorch (Isaac Sim / CUDA libs / NCCL):", non_pytorch))
        if unclassified_names:
            out.append("")
            out.append("Unclassified param names (first 10):")
            for n in unclassified_names[:10]:
                out.append(f"  {n}")
        out.append("=" * 70)
        msg = "\n".join(out)
        print(msg, flush=True)  # noqa: T201
        logger.info(msg)
