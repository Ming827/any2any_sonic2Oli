"""Oli LoRA wrapper for SONIC's UniversalTokenModule.

Wraps a pretrained G1-based UniversalTokenModule to work with the Oli 31-DOF
robot by:
  1. Freezing the entire pretrained model
  2. Injecting LoRA adapters into the G1 encoder and g1_dyn decoder
  3. Bridging DOF dimensions (31↔29) via learned linear projections
  4. Mapping action output from 29→31 DOF

The proprioception bridging uses a small learned linear layer to project
Oli's proprioception features into the G1 proprioception space, since
actor_obs contains more than just joint positions (velocities, gravity,
commands, etc.) and its total dimension changes with DOF count.
"""

from __future__ import annotations

from loguru import logger
import torch
import torch.nn as nn

from gear_sonic.trl.modules.dof_mapping import (
    DofBridge,
    G1_DOF,
    OLI_HEAD_INDICES,
    compute_g1_critic_obs_dim,
    hip_pitch_g1_to_oli_inplace,
    hip_pitch_oli_to_g1_inplace,
    slice_command_multi_future_to_g1,
    slice_oli_actor_obs_to_g1,
    slice_oli_critic_obs_to_g1,
)
from gear_sonic.trl.modules.lora_utils import (
    apply_lora_to_sequential,
    cast_frozen_to_dtype,
    count_parameters,
    freeze_module,
    load_lora_state_dict,
    merge_lora_weights,
    print_lora_summary,
    save_lora_state_dict,
)
import contextlib
import numpy as np


class _NullCtx(contextlib.AbstractContextManager):
    """Cheap no-op context manager used in the rollout fast path when bf16 is off."""
    def __exit__(self, *exc):  # noqa: D401
        return None


class OliLoRAWrapper(nn.Module):
    """Wrap a pretrained G1 UniversalTokenModule for Oli 31-DOF LoRA fine-tuning.

    This module sits in place of the original UniversalTokenModule inside the
    Actor. It:
      - Freezes the entire pretrained backbone
      - Injects LoRA into the G1 encoder MLP and g1_dyn decoder MLP
      - Projects Oli proprioception (variable dim) → G1 proprioception (930)
      - Maps G1 29-DOF action output → Oli 31-DOF via DofBridge

    The only trainable parameters are:
      - LoRA A/B matrices in the G1 encoder
      - LoRA A/B matrices in the g1_dyn decoder
      - Proprioception projection layer
      - DofBridge head_default (2 params for head yaw/pitch)

    Args:
        pretrained_module: The pretrained UniversalTokenModule (G1 29-DOF).
        oli_proprioception_dim: Dimension of Oli's actor_obs.
        lora_rank: Rank of LoRA decomposition.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout for LoRA input path.
        lora_skip_last: If True, don't apply LoRA to the last Linear layer
            in each MLP (the output projection).
        learnable_head: If True, learn default head joint positions in DofBridge.
    """

    def __init__(
        self,
        pretrained_module: nn.Module,
        oli_proprioception_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_skip_last: bool = False,
        learnable_head: bool = True,
        # ---- New (all default to vanilla LoRA behavior) ----
        variant: str = "lora",                 # "lora" | "dora"
        rslora: bool = False,                  # alpha/sqrt(rank) scaling
        use_gate: bool = False,                # learnable scalar gate per adapter
        init_strategy: str = "kaiming",        # "kaiming" | "normal_0.02"
        encoder_per_layer_ranks: dict | None = None,  # {idx: rank} for g1 encoder
        decoder_per_layer_ranks: dict | None = None,  # {idx: rank} for g1_dyn decoder
        encoder_rank: int | None = None,       # if set, overrides lora_rank for encoder
        decoder_rank: int | None = None,       # if set, overrides lora_rank for decoder
        encoder_alpha: float | None = None,
        decoder_alpha: float | None = None,
        # Restrict LoRA injection to a subset of Linear sublayers (Linear-
        # positional indices, negatives Python-style). None → adapt all
        # Linears (legacy behavior). Concentrating on the tail of each
        # Sequential lets autograd skip caching the frozen-prefix
        # activations — see apply_lora_to_sequential docstring.
        encoder_active_linears: list | None = None,
        decoder_active_linears: list | None = None,
        # ---- Proprioception 31→29 mapping ----
        # "linear" : learnable nn.Linear (default; matches the original wrapper behavior)
        # "slice"  : hard name-based slicer that drops the 2 head DOFs from each
        #            per-DOF segment of actor_obs (joint_pos / joint_vel / actions).
        #            No trainable params on this path; inputs are reshaped to drop
        #            head and reordered to G1 IsaacLab DOF order. Choose this when
        #            you want the pretrained 29-DOF G1 features fed exactly the
        #            joints they were trained on.
        proprio_mode: str = "linear",
        proprio_actor_history_length: int = 10,
        proprio_action_history_length: int = 10,
        # ---- Full fine-tune escape hatch ----
        # When True, the entire pretrained backbone stays trainable (no
        # freeze). Combined with encoder_rank=decoder_rank=0 this turns the
        # wrapper into a "full fine-tune from G1 ckpt" mode that still keeps
        # the proprio bridge / DofBridge / hip_pitch decomp paths intact.
        unfreeze_backbone: bool = False,
        # ---- bfloat16 autocast for forward (memory + speed) ----
        # When True, wrap forward in torch.autocast(dtype=bfloat16). Weights
        # stay fp32 in storage so ckpt format is unchanged; only intermediate
        # activations are bf16. Halves activation cache (~3-6 GB at 16384
        # envs) and speeds up matmul ~30-50% on Ampere/Hopper.
        # bf16 chosen over fp16 because it has the same exponent range as
        # fp32 — no loss scaling needed, no NaN risk in PPO advantage scale.
        use_bf16: bool = False,
    ):
        super().__init__()

        self.pretrained = pretrained_module
        self.lora_variant = variant
        self.rslora = rslora
        self.unfreeze_backbone = unfreeze_backbone
        self.use_bf16 = use_bf16

        # --- Step 1: Freeze (or keep trainable) the pretrained backbone ---
        if unfreeze_backbone:
            for p in self.pretrained.parameters():
                p.requires_grad = True
            logger.warning(
                "OliLoRAWrapper: unfreeze_backbone=True — entire pretrained "
                "backbone is TRAINABLE (full fine-tune mode)"
            )
        else:
            freeze_module(self.pretrained)
            logger.info("Froze all pretrained module parameters")

        # --- Step 2: Inject LoRA into G1 encoder (skip when rank<=0) ---
        enc_rank = encoder_rank if encoder_rank is not None else lora_rank
        enc_alpha = encoder_alpha if encoder_alpha is not None else lora_alpha
        g1_encoder = self.pretrained.encoders["g1"]
        if not hasattr(g1_encoder, "module") or not isinstance(g1_encoder.module, nn.Sequential):
            raise RuntimeError(
                "G1 encoder does not have an nn.Sequential .module attribute. "
                f"Got: {type(g1_encoder)}"
            )
        if enc_rank is not None and int(enc_rank) > 0:
            apply_lora_to_sequential(
                g1_encoder.module,
                rank=enc_rank,
                alpha=enc_alpha,
                dropout=lora_dropout,
                skip_last=lora_skip_last,
                variant=variant,
                rslora=rslora,
                use_gate=use_gate,
                init_strategy=init_strategy,
                per_layer_ranks=encoder_per_layer_ranks,
                linear_active_indices=encoder_active_linears,
            )
            total, trainable = count_parameters(g1_encoder)
            logger.info(
                f"{variant.upper()} injected into G1 encoder "
                f"(rank={enc_rank}, alpha={enc_alpha}, rslora={rslora}, gate={use_gate}): "
                f"{trainable:,} trainable / {total:,} total params"
            )
        else:
            # encoder_rank=0 (or None+lora_rank=0): skip injection entirely.
            # The encoder stays fully frozen (Step 1 already called freeze_module
            # on the whole pretrained backbone). This is useful when the
            # cross-DOF adaptation is dominated by the decoder + FSQ codebook
            # and you want to keep the encoder rigid (G1's G1-encoded latents).
            total, trainable = count_parameters(g1_encoder)
            logger.info(
                f"G1 encoder {'TRAINABLE' if unfreeze_backbone else 'FROZEN'} "
                f"(no LoRA, encoder_rank={enc_rank}): "
                f"{trainable:,} trainable / {total:,} total params"
            )

        # --- Step 3: Inject LoRA into g1_dyn decoder ---
        dec_rank = decoder_rank if decoder_rank is not None else lora_rank
        dec_alpha = decoder_alpha if decoder_alpha is not None else lora_alpha
        g1_dyn_decoder = self.pretrained.decoders["g1_dyn"]
        if not (hasattr(g1_dyn_decoder, "module") and isinstance(g1_dyn_decoder.module, nn.Sequential)):
            raise RuntimeError(
                "g1_dyn decoder does not have an nn.Sequential .module attribute. "
                f"Got: {type(g1_dyn_decoder)}"
            )
        if dec_rank is not None and int(dec_rank) > 0:
            apply_lora_to_sequential(
                g1_dyn_decoder.module,
                rank=dec_rank,
                alpha=dec_alpha,
                dropout=lora_dropout,
                skip_last=lora_skip_last,
                variant=variant,
                rslora=rslora,
                use_gate=use_gate,
                init_strategy=init_strategy,
                per_layer_ranks=decoder_per_layer_ranks,
                linear_active_indices=decoder_active_linears,
            )
            total, trainable = count_parameters(g1_dyn_decoder)
            logger.info(
                f"{variant.upper()} injected into g1_dyn decoder "
                f"(rank={dec_rank}, alpha={dec_alpha}, rslora={rslora}, gate={use_gate}): "
                f"{trainable:,} trainable / {total:,} total params"
            )
        else:
            # decoder_rank=0: no LoRA injection. The decoder is either fully
            # frozen (unfreeze_backbone=False) or fully trainable
            # (unfreeze_backbone=True) — Step 1 already set requires_grad.
            total, trainable = count_parameters(g1_dyn_decoder)
            logger.info(
                f"g1_dyn decoder {'TRAINABLE' if unfreeze_backbone else 'FROZEN'} "
                f"(no LoRA, decoder_rank={dec_rank}): "
                f"{trainable:,} trainable / {total:,} total params"
            )

        # --- Step 4: Proprioception 31→29 mapping ---
        g1_proprioception_dim = self.pretrained.decoder_feature_dims_map["proprioception"]
        self.proprio_mode = proprio_mode
        self.proprio_actor_history_length = proprio_actor_history_length
        self.proprio_action_history_length = proprio_action_history_length

        if proprio_mode == "linear":
            self.proprio_proj = nn.Linear(oli_proprioception_dim, g1_proprioception_dim)
            nn.init.zeros_(self.proprio_proj.bias)
            with torch.no_grad():
                nn.init.zeros_(self.proprio_proj.weight)
                copy_dim = min(oli_proprioception_dim, g1_proprioception_dim)
                self.proprio_proj.weight[:copy_dim, :copy_dim] = torch.eye(copy_dim)
            logger.info(
                f"Proprioception (mode=linear): {oli_proprioception_dim} → {g1_proprioception_dim} "
                f"(learnable nn.Linear, identity-init)"
            )
        elif proprio_mode == "slice":
            # No learnable params on the proprio path — hard name-based slicing.
            self.proprio_proj = None
            # Sanity-check that the layout dims match expectations.
            from gear_sonic.trl.modules.dof_mapping import G1_DOF, OLI_DOF
            expected_oli = (
                3 * proprio_actor_history_length              # gravity_dir
                + 3 * proprio_actor_history_length            # base_ang_vel
                + OLI_DOF * proprio_actor_history_length      # joint_pos
                + OLI_DOF * proprio_actor_history_length      # joint_vel
                + OLI_DOF * proprio_action_history_length     # actions
            )
            expected_g1 = (
                3 * proprio_actor_history_length
                + 3 * proprio_actor_history_length
                + G1_DOF * proprio_actor_history_length
                + G1_DOF * proprio_actor_history_length
                + G1_DOF * proprio_action_history_length
            )
            if oli_proprioception_dim != expected_oli or g1_proprioception_dim != expected_g1:
                raise ValueError(
                    f"proprio_mode='slice' assumes the local_dir_hist layout "
                    f"(gravity+ang_vel+joint_pos+joint_vel+actions × histories). "
                    f"Got oli_proprio_dim={oli_proprioception_dim} (expected {expected_oli}) "
                    f"and g1_proprio_dim={g1_proprioception_dim} (expected {expected_g1}). "
                    f"If the obs layout changed, fall back to proprio_mode='linear' or "
                    f"adjust prop/action history lengths."
                )
            logger.info(
                f"Proprioception (mode=slice): {oli_proprioception_dim} → {g1_proprioception_dim} "
                f"(no learnable params; head DOFs dropped by name, "
                f"prop_hist={proprio_actor_history_length}, "
                f"action_hist={proprio_action_history_length})"
            )
        else:
            raise ValueError(f"Unknown proprio_mode: {proprio_mode!r} (expected 'linear' or 'slice')")

        # --- Step 5: DOF bridge for action output (29 → 31) ---
        self.dof_bridge = DofBridge(learnable_head=learnable_head)
        logger.info("DOF bridge initialized (29→31 with learnable head defaults)")

        # --- Expose attributes needed by Actor and trainer ---
        # Forward config attributes from the pretrained module
        self.actions_dim = 31  # Oli has 31 DOF
        self.env_config = self.pretrained.env_config
        self.algo_config = self.pretrained.algo_config
        self.encoders_cfg = self.pretrained.encoders_cfg
        self.decoders_cfg = self.pretrained.decoders_cfg
        self.token_dim = self.pretrained.token_dim
        self.max_num_tokens = self.pretrained.max_num_tokens
        self.token_total_dim = self.pretrained.token_total_dim
        self.tokenizer_obs_dims = self.pretrained.tokenizer_obs_dims
        self.tokenizer_obs_names = self.pretrained.tokenizer_obs_names
        self.proprioception_features = self.pretrained.proprioception_features
        self.obs_dim_dict = self.pretrained.obs_dim_dict
        self.encoders = self.pretrained.encoders
        self.decoders = self.pretrained.decoders
        self.aux_loss_func = self.pretrained.aux_loss_func
        self.aux_loss_coef = self.pretrained.aux_loss_coef

        # Log summary
        total_all, trainable_all = count_parameters(self)
        logger.info(
            f"OliLoRAWrapper summary: {trainable_all:,} trainable / {total_all:,} total params "
            f"({100.0 * trainable_all / total_all:.2f}% trainable) | "
            f"use_bf16={self.use_bf16} | proprio_mode={self.proprio_mode}"
        )

        # ---- LoRA-specific A: cast frozen backbone weights to bf16 ----
        # The pretrained backbone has requires_grad=False (encoder fully
        # frozen; decoder Linears wrapped by LoRA have base weight frozen
        # but A/B trainable). Casting only the frozen tensors to bf16:
        #   - halves their on-GPU storage (~58 MB → ~29 MB for actor)
        #   - skips the just-in-time fp32→bf16 cast that autocast issues
        #     on every matmul during rollout / training
        # LoRA A/B / proprio_proj / dof_bridge stay fp32 (they need grad
        # and Adam math in fp32). Gated on use_bf16 — fp32 path is unchanged.
        if self.use_bf16:
            n_cast = cast_frozen_to_dtype(self.pretrained, torch.bfloat16)
            logger.info(
                f"[LoRA-A] Cast {n_cast} frozen tensors in pretrained backbone "
                f"to bf16 storage (LoRA-only optimization)"
            )

        # ---- LoRA-specific D: pre-cache tokenizer slice plan + action key ----
        # parse_tokenizer_obs builds the same dict every forward by looping
        # over tokenizer_obs_names with np.prod. Cache the (start, end, dims)
        # plan once so the rollout fast path can do plain slice + view.
        self._tokenizer_slice_plan: list[tuple[str, int, int, tuple[int, ...]]] = []
        _idx = 0
        for _name in self.pretrained.tokenizer_obs_names:
            _dims = tuple(self.pretrained.tokenizer_obs_dims[_name])
            _flat = int(np.prod(_dims))
            self._tokenizer_slice_plan.append((_name, _idx, _idx + _flat, _dims))
            _idx += _flat
        self._tokenizer_total_dim = _idx

        # Whether the active-encoders / decoders match the single-path
        # "rollout fast" assumption. If not, fall back to general forward.
        self._fast_rollout_ok = (
            list(self.pretrained.encoders_to_iterate) == ["g1"]
            and "g1_dyn" in self.pretrained.decoders
            and proprio_mode in ("slice", "linear")
        )
        # Probe which action-key g1_dyn outputs (cached after first real call).
        # Most configs return "action"; some return "body_action" / "meta_action".
        self._cached_action_key: str | None = None

    def parse_tokenizer_obs(self, input_data):
        """Delegate to pretrained module."""
        return self.pretrained.parse_tokenizer_obs(input_data)

    def forward(
        self,
        input_data,
        compute_aux_loss=False,
        return_dict=False,
        latent_residual=None,
        latent_residual_mode="post_quantization",
        **kwargs,
    ):
        """Run forward with Oli DOF bridging.

        The flow:
          1. Parse tokenizer obs (same as pretrained)
          2. Encode with LoRA'd G1 encoder → FSQ → tokens
          3. Project Oli proprioception → G1 proprioception dim
          4. Decode with LoRA'd g1_dyn decoder → 29-DOF action
          5. Bridge 29-DOF action → 31-DOF via DofBridge

        We override the proprioception path by intercepting the decode step.

        Rollout fast path (LoRA-specific D): when grad is disabled and the
        caller doesn't ask for aux loss / dict output / latent residual, we
        skip parse_tokenizer_obs's per-call dict construction, the encoder
        loop, assemble_all_tokens, and the action-key probing — all of
        which have known shapes / single-element collections in the current
        config. Saves a few hundred µs of Python overhead per step.
        """
        if (
            self._fast_rollout_ok
            and not torch.is_grad_enabled()
            and not compute_aux_loss
            and not return_dict
            and latent_residual is None
        ):
            return self._rollout_forward(input_data)

        if self.use_bf16 and input_data["actor_obs"].is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._forward_impl(
                    input_data,
                    compute_aux_loss=compute_aux_loss,
                    return_dict=return_dict,
                    latent_residual=latent_residual,
                    latent_residual_mode=latent_residual_mode,
                    **kwargs,
                )
        return self._forward_impl(
            input_data,
            compute_aux_loss=compute_aux_loss,
            return_dict=return_dict,
            latent_residual=latent_residual,
            latent_residual_mode=latent_residual_mode,
            **kwargs,
        )

    @torch.no_grad()
    def _rollout_forward(self, input_data):
        """Streamlined no-grad forward used only during PPO rollout.

        Pre-conditions verified by ``_fast_rollout_ok`` at init time:
          - encoders_to_iterate == ["g1"]
          - g1_dyn is in decoders
          - proprio_mode is "slice" or "linear"
          - caller is no-grad (rollout context) and only needs action_mean

        Differences vs ``_forward_impl``:
          - Inlined tokenizer-obs split using cached slice plan (no np.prod,
            no dict-of-names loop overhead)
          - No encoder_masks (we know it's a single masked-None encoder)
          - No assemble_all_tokens (single encoder → direct view)
          - Action-key resolved once and cached
          - No aux_loss / return_dict branching
        """
        bf16_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bf16 and input_data["actor_obs"].is_cuda
            else _NullCtx()
        )
        with bf16_ctx:
            actor_obs = input_data["actor_obs"]
            batch_size, seq_len = actor_obs.shape[:2]

            # --- Inlined parse_tokenizer_obs using cached slice plan ---
            tokenizer_flat = input_data["tokenizer"]
            tokenizer_obs = {}
            base_shape = tokenizer_flat.shape[:-1]
            for _name, _s, _e, _dims in self._tokenizer_slice_plan:
                tokenizer_obs[_name] = tokenizer_flat[..., _s:_e].reshape(base_shape + _dims)

            # --- Boundary slice on the motion command (Oli 31 → G1 29) ---
            cmd = tokenizer_obs.get("command_multi_future_nonflat")
            if cmd is not None:
                tokenizer_obs["command_multi_future_nonflat"] = slice_command_multi_future_to_g1(
                    cmd,
                    num_future_frames=self.pretrained.num_future_frames,
                    apply_hip_pitch_decomp=True,
                )

            # --- Proprio path (slice or linear) ---
            prop_in = torch.cat(
                [input_data[k] for k in self.pretrained.proprioception_features], dim=-1
            )
            if self.proprio_mode == "slice":
                prop_proj = slice_oli_actor_obs_to_g1(
                    prop_in,
                    prop_hist=self.proprio_actor_history_length,
                    action_hist=self.proprio_action_history_length,
                    apply_hip_pitch_decomp=True,
                )
            else:
                prop_proj = self.proprio_proj(prop_in)

            # --- Single-encoder shortcut: skip encoder_masks dict + assemble ---
            frame_mask, token_mask = self.pretrained._create_frame_and_token_masks(tokenizer_obs)
            enc_tok, _enc_lat = self.pretrained.encode(
                "g1", tokenizer_obs, None, frame_mask=frame_mask
            )
            # Single encoder → tokens already (B*S, n_tok, d); reshape directly
            all_tokens = enc_tok.view(batch_size, seq_len, *enc_tok.shape[1:])

            # --- Decode g1_dyn with our projected proprio ---
            decode_input = {
                "token": all_tokens,
                "token_flattened": all_tokens.view(*all_tokens.shape[:-2], -1),
                "proprioception": prop_proj,
            }
            decode_input.update(tokenizer_obs)
            g1_dyn_out = self.pretrained.decode("g1_dyn", decode_input, token_mask=token_mask)

            # --- Action-key cache to skip the elif chain on every call ---
            ak = self._cached_action_key
            if ak is None or ak not in g1_dyn_out:
                if "body_action" in g1_dyn_out:
                    ak = "body_action"
                elif "meta_action" in g1_dyn_out:
                    ak = "meta_action"
                else:
                    ak = "action"
                self._cached_action_key = ak
            action_29 = g1_dyn_out[ak]

            action_mean = self.dof_bridge.g1_to_oli(action_29).contiguous()
            hip_pitch_g1_to_oli_inplace(action_mean)
            return action_mean

    def _forward_impl(
        self,
        input_data,
        compute_aux_loss=False,
        return_dict=False,
        latent_residual=None,
        latent_residual_mode="post_quantization",
        **kwargs,
    ):
        """Actual forward body, wrapped above with optional bf16 autocast."""
        batch_size, seq_len = input_data["actor_obs"].shape[:2]
        tokenizer_obs = self.pretrained.parse_tokenizer_obs(input_data)

        # Slice motion command from Oli 31-DOF to G1 29-DOF before it hits the
        # frozen G1 encoder. The env feeds 31-DOF Oli motion data so that
        # ``commands.py`` reset / reward see all 31 joints (matches the Oli
        # robot). The pretrained encoder only saw 29-DOF (G1) in its training,
        # so its first Linear has input dim 2*N_fut*29 + ori_dim. We drop the
        # 2 head DOFs from each future-frame's joint_pos and joint_vel here.
        if "command_multi_future_nonflat" in tokenizer_obs:
            tokenizer_obs["command_multi_future_nonflat"] = (
                slice_command_multi_future_to_g1(
                    tokenizer_obs["command_multi_future_nonflat"],
                    num_future_frames=self.pretrained.num_future_frames,
                    # Decompose Oli's tilted hip_pitch axis into G1's pure
                    # Y component — backbone trained on G1 frame.
                    apply_hip_pitch_decomp=True,
                )
            )

        # Build proprioception from Oli's actor_obs and bridge 31→29 according to mode
        proprioception_input = torch.cat(
            [input_data[key] for key in self.pretrained.proprioception_features], dim=-1
        )
        if self.proprio_mode == "slice":
            proprioception_projected = slice_oli_actor_obs_to_g1(
                proprioception_input,
                prop_hist=self.proprio_actor_history_length,
                action_hist=self.proprio_action_history_length,
                # Same hip_pitch axis decomposition as the motion command —
                # joint_pos / joint_vel / actions all need the transform.
                apply_hip_pitch_decomp=True,
            )
        else:
            proprioception_projected = self.proprio_proj(proprioception_input)

        # Encode (uses LoRA'd G1 encoder)
        residual_reshaped = None
        if latent_residual is not None:
            residual_reshaped = latent_residual.view(
                batch_size, 1, self.pretrained.max_num_tokens, self.pretrained.token_dim
            )

        frame_mask, token_mask = self.pretrained._create_frame_and_token_masks(tokenizer_obs)
        encoder_masks = self.pretrained.create_encoder_masks(tokenizer_obs)
        encoded_tokens = {}
        encoded_latents = {}

        for encoder_name in self.pretrained.encoders_to_iterate:
            encoded_tokens[encoder_name], encoded_latents[encoder_name] = self.pretrained.encode(
                encoder_name,
                tokenizer_obs,
                encoder_masks[encoder_name],
                frame_mask=frame_mask,
            )

        all_tokens = self.pretrained.assemble_all_tokens(
            encoded_tokens, encoder_masks, batch_size, seq_len
        )

        if latent_residual is not None and latent_residual_mode == "post_quantization":
            all_tokens = all_tokens + residual_reshaped

        # Decode with projected proprioception
        decode_input_dict = {
            "token": all_tokens,
            "token_flattened": all_tokens.view(*all_tokens.shape[:-2], -1),
            "proprioception": proprioception_projected,  # Use projected dim
        }
        decode_input_dict.update(tokenizer_obs)

        # Only run g1_dyn decoder (the action decoder)
        g1_dyn_out = self.pretrained.decode("g1_dyn", decode_input_dict, token_mask=token_mask)

        # Extract 29-DOF action
        if "body_action" in g1_dyn_out:
            action_29 = g1_dyn_out["body_action"]
        elif "meta_action" in g1_dyn_out:
            action_29 = g1_dyn_out["meta_action"]
        else:
            action_29 = g1_dyn_out["action"]

        # Bridge 29-DOF → 31-DOF
        action_mean = self.dof_bridge.g1_to_oli(action_29)
        # Backbone outputs in G1 hip-axis convention; convert to Oli's tilted
        # hip_pitch axis (inverse of the obs-side decomposition).
        action_mean = action_mean.contiguous()
        hip_pitch_g1_to_oli_inplace(action_mean)

        if compute_aux_loss or return_dict:
            decoded_outputs = {"g1_dyn": g1_dyn_out}
            output = {
                "action_mean": action_mean,
                "aux_losses": {},
                "aux_loss_coef": self.pretrained.aux_loss_coef,
                "decoded_outputs": decoded_outputs,
                "tokenizer_obs": tokenizer_obs,
                "encoder_masks": encoder_masks,
                "encoded_tokens": encoded_tokens,
                "encoded_latents": encoded_latents,
                "encoders_cfg": self.pretrained.encoders_cfg,
                "decoders_cfg": self.pretrained.decoders_cfg,
            }
        else:
            output = action_mean
        return output

    def get_token_info(self):
        """Delegate to pretrained module."""
        return self.pretrained.get_token_info()

    # ---------- LoRA-only save / load / merge / inspect ----------

    def save_adapters(self, path: str):
        """Save only LoRA + bridge params (proprio_proj, dof_bridge) to ``path``."""
        sd = save_lora_state_dict(self, path)
        logger.info(f"Saved {len(sd)} adapter tensors to {path}")
        return sd

    def load_adapters(self, path: str, strict: bool = False):
        """Load a previously-saved adapter-only state_dict; backbone stays frozen."""
        missing, unexpected = load_lora_state_dict(self, path, strict=strict)
        if missing:
            logger.warning(f"Adapter load: missing keys = {missing}")
        if unexpected:
            logger.warning(f"Adapter load: unexpected keys = {unexpected}")
        return missing, unexpected

    def merge_and_freeze(self) -> nn.Module:
        """Bake LoRA into base weights and return a clean module suitable for export.

        After this call, in-place LoRA layers are gone — useful right before
        ONNX export or copying state to a deploy-side network. Note: this
        modifies the wrapper's pretrained module, so don't call mid-training.
        """
        merge_lora_weights(self.pretrained)
        logger.info("Merged LoRA adapters into base weights")
        return self.pretrained

    def print_summary(self):
        """Print every LoRA/DoRA adapter, its rank, scaling, and trainable param count."""
        print_lora_summary(self, root_name="OliLoRAWrapper")


def load_oli_lora_module(
    pretrained_checkpoint_path: str,
    pretrained_module: nn.Module,
    oli_proprioception_dim: int,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_skip_last: bool = False,
    learnable_head: bool = True,
    device: str = "cpu",
    *,
    variant: str = "lora",
    rslora: bool = False,
    use_gate: bool = False,
    init_strategy: str = "kaiming",
    encoder_rank: int | None = None,
    decoder_rank: int | None = None,
    encoder_alpha: float | None = None,
    decoder_alpha: float | None = None,
    encoder_per_layer_ranks: dict | None = None,
    decoder_per_layer_ranks: dict | None = None,
    encoder_active_linears: list | None = None,
    decoder_active_linears: list | None = None,
    adapter_checkpoint: str | None = None,
    proprio_mode: str = "linear",
    proprio_actor_history_length: int = 10,
    proprio_action_history_length: int = 10,
    unfreeze_backbone: bool = False,
) -> OliLoRAWrapper:
    """Load pretrained weights and create an OliLoRAWrapper.

    Args:
        pretrained_checkpoint_path: Path to the pretrained SONIC checkpoint (.pt).
        pretrained_module: An already-instantiated UniversalTokenModule
            (with correct config for the pretrained model).
        oli_proprioception_dim: Dimension of Oli's actor_obs.
        lora_rank: LoRA rank.
        lora_alpha: LoRA scaling.
        lora_dropout: LoRA dropout.
        lora_skip_last: Skip LoRA on last Linear layer.
        learnable_head: Learn default head positions in DofBridge.
        device: Device to load checkpoint to.

    Returns:
        OliLoRAWrapper with pretrained weights loaded and LoRA injected.
    """
    # Load checkpoint
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)

    # Extract policy state dict (actor_module.* keys)
    if "policy_state_dict" in checkpoint:
        policy_sd = checkpoint["policy_state_dict"]
    else:
        policy_sd = checkpoint

    # Filter to actor_module keys and strip prefix
    prefix = "actor_module."
    module_sd = {}
    for k, v in policy_sd.items():
        if k.startswith(prefix):
            module_sd[k[len(prefix):]] = v

    # Load into pretrained module
    missing, unexpected = pretrained_module.load_state_dict(module_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading pretrained weights: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading pretrained weights: {unexpected}")
    logger.info(
        f"Loaded pretrained weights from {pretrained_checkpoint_path} "
        f"({len(module_sd)} keys)"
    )

    # Create wrapper (freezes + injects LoRA / DoRA)
    wrapper = OliLoRAWrapper(
        pretrained_module=pretrained_module,
        oli_proprioception_dim=oli_proprioception_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_skip_last=lora_skip_last,
        learnable_head=learnable_head,
        variant=variant,
        rslora=rslora,
        use_gate=use_gate,
        init_strategy=init_strategy,
        encoder_rank=encoder_rank,
        decoder_rank=decoder_rank,
        encoder_alpha=encoder_alpha,
        decoder_alpha=decoder_alpha,
        encoder_per_layer_ranks=encoder_per_layer_ranks,
        decoder_per_layer_ranks=decoder_per_layer_ranks,
        encoder_active_linears=encoder_active_linears,
        decoder_active_linears=decoder_active_linears,
        proprio_mode=proprio_mode,
        proprio_actor_history_length=proprio_actor_history_length,
        proprio_action_history_length=proprio_action_history_length,
        unfreeze_backbone=unfreeze_backbone,
    )

    if adapter_checkpoint is not None:
        wrapper.load_adapters(adapter_checkpoint)

    return wrapper



class CriticLoRAWrapper(nn.Module):
    """Wrap a pretrained critic backbone for LoRA fine-tuning + Oli→G1 obs slicing.

    Used to adapt a G1-trained value function to the Oli env. Symmetric to
    ``OliLoRAWrapper`` but for the critic side:
      1. Freeze the critic backbone (the pretrained MLP)
      2. Inject LoRA adapters into the backbone's ``.module`` Sequential
      3. At forward, name-keyed slice ``obs_dict["critic_obs"]`` from the
         Oli view (DOF=31, num_bodies=42) down to the G1 view (DOF=29,
         num_bodies=30) before feeding it into the frozen backbone

    Critic outputs a scalar value per env, so no DOF-bridge on the output.
    The host ``Critic`` module's ``running_mean_std`` (if any) keeps running
    on the Oli side — its stats are NOT loaded from the G1 ckpt because dims
    differ; they accumulate fresh during training.

    Args:
        pretrained_critic_backbone: ``value_model.critic_module`` after
            ``load_state_dict`` from ckpt's ``value_state_dict``. Must
            expose ``.module`` as the inner ``nn.Sequential``.
        group_obs_names_critic: per-segment names from
            ``env.config["obs"]["group_obs_names"]["critic"]``.
        group_obs_dims_critic: aligned per-segment dims from
            ``env.config["obs"]["group_obs_dims"]["critic"]``.
        expected_g1_critic_obs_dim: ckpt's first Linear ``in_features``
            (i.e. the dim the frozen backbone expects after slicing).
            Wrapper raises if our computed post-slice dim disagrees.
    """

    def __init__(
        self,
        pretrained_critic_backbone: nn.Module,
        group_obs_names_critic: list[str],
        group_obs_dims_critic: list[int],
        expected_g1_critic_obs_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_skip_last: bool = False,
        variant: str = "lora",
        rslora: bool = False,
        use_gate: bool = False,
        init_strategy: str = "kaiming",
        per_layer_ranks: dict | None = None,
        # See OliLoRAWrapper.{encoder,decoder}_active_linears — same idea for
        # the critic Sequential. None → all critic Linears get LoRA.
        linear_active_indices: list | None = None,
        # See OliLoRAWrapper.use_bf16 — same semantics for the critic forward.
        use_bf16: bool = False,
    ):
        super().__init__()
        self.pretrained = pretrained_critic_backbone
        self.group_obs_names = list(group_obs_names_critic)
        self.group_obs_dims = list(group_obs_dims_critic)
        self.use_bf16 = use_bf16

        # Verify the slice plan produces exactly ckpt's expected G1 dim.
        oli_total = sum(self.group_obs_dims)
        g1_total = compute_g1_critic_obs_dim(self.group_obs_names, self.group_obs_dims)
        if g1_total != expected_g1_critic_obs_dim:
            seg_str = ", ".join(
                f"{n}={d}" for n, d in zip(self.group_obs_names, self.group_obs_dims)
            )
            raise ValueError(
                f"CriticLoRAWrapper: post-slice dim {g1_total} doesn't match ckpt's "
                f"critic first Linear in_features {expected_g1_critic_obs_dim}.\n"
                f"  Oli critic_obs total = {oli_total}\n"
                f"  Per-segment (name=oli_dim): {seg_str}\n"
                f"  Likely the env's obs layout / history lengths / num bodies "
                f"differ from what the G1 ckpt was trained with. Either change the "
                f"yaml to match, or skip critic LoRA."
            )
        logger.info(
            f"CriticLoRAWrapper: Oli critic_obs ({oli_total}) → G1 ({g1_total}) "
            f"by name-keyed slice; matches ckpt expected {expected_g1_critic_obs_dim}"
        )

        # Freeze the backbone — only LoRA adapters will train.
        freeze_module(self.pretrained)

        # Inject LoRA on the inner Sequential.
        seq = getattr(self.pretrained, "module", None)
        if not isinstance(seq, nn.Sequential):
            raise RuntimeError(
                f"critic backbone has no nn.Sequential .module attribute "
                f"(got: {type(seq).__name__})"
            )
        apply_lora_to_sequential(
            seq,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            skip_last=lora_skip_last,
            variant=variant,
            rslora=rslora,
            use_gate=use_gate,
            init_strategy=init_strategy,
            per_layer_ranks=per_layer_ranks,
            linear_active_indices=linear_active_indices,
        )
        total, trainable = count_parameters(self.pretrained)
        logger.info(
            f"{variant.upper()} injected into critic_module "
            f"(rank={lora_rank}, alpha={lora_alpha}, rslora={rslora}, gate={use_gate}): "
            f"{trainable:,} trainable / {total:,} total params | "
            f"use_bf16={self.use_bf16}"
        )

        # ---- LoRA-specific A: bf16 storage for frozen critic backbone ----
        # Same rationale as OliLoRAWrapper. Only frozen params are touched;
        # LoRA A/B added by apply_lora_to_sequential remain fp32 trainables.
        if self.use_bf16:
            n_cast = cast_frozen_to_dtype(self.pretrained, torch.bfloat16)
            logger.info(
                f"[LoRA-A] Cast {n_cast} frozen tensors in critic backbone "
                f"to bf16 storage"
            )

    def forward(self, obs_dict, **kwargs):
        # Don't mutate caller's dict; only swap the critic_obs entry.
        obs_dict = dict(obs_dict)
        if self.use_bf16 and obs_dict["critic_obs"].is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                obs_dict["critic_obs"] = slice_oli_critic_obs_to_g1(
                    obs_dict["critic_obs"],
                    self.group_obs_names,
                    self.group_obs_dims,
                )
                return self.pretrained(obs_dict, **kwargs)
        obs_dict["critic_obs"] = slice_oli_critic_obs_to_g1(
            obs_dict["critic_obs"],
            self.group_obs_names,
            self.group_obs_dims,
        )
        return self.pretrained(obs_dict, **kwargs)

    def save_adapters(self, path: str):
        """Save only LoRA adapter params on the critic side."""
        sd = save_lora_state_dict(self, path)
        logger.info(f"Saved {len(sd)} critic adapter tensors to {path}")
        return sd

    def load_adapters(self, path: str, strict: bool = False):
        """Load a previously-saved critic adapter-only state_dict."""
        missing, unexpected = load_lora_state_dict(self, path, strict=strict)
        if missing:
            logger.warning(f"Critic adapter load: missing keys = {missing}")
        if unexpected:
            logger.warning(f"Critic adapter load: unexpected keys = {unexpected}")
        return missing, unexpected

    def merge_and_freeze(self) -> nn.Module:
        merge_lora_weights(self.pretrained)
        logger.info("Merged critic LoRA adapters into base weights")
        return self.pretrained

    def print_summary(self):
        print_lora_summary(self, root_name="CriticLoRAWrapper")
