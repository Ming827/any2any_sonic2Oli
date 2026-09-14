#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# sys.path + gear_sonic alias bootstrap: see train_agent_trl.py for the full
# rationale. Strip non-canonical entries pointing at this folder, prepend the
# repo root, and register a runtime ``gear_sonic`` alias if the folder is
# named differently (e.g. ``sonic_WBT``).
import os as _os
import sys as _sys
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.dirname(_script_dir)
_bad_norm = {
    _os.path.normpath(_script_dir),
    _os.path.normpath(_os.getcwd()) if _os.getcwd() == _script_dir else None,
    "",
    ".",
}
_bad_norm.discard(None)
_sys.path[:] = [
    p for p in _sys.path
    if (_os.path.normpath(p) if p not in ("", ".") else p) not in _bad_norm
]
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
_trl_mod = _sys.modules.get("trl")
if _trl_mod is not None:
    _trl_file = getattr(_trl_mod, "__file__", "") or ""
    if _trl_file and _os.path.normpath(_trl_file).startswith(_os.path.normpath(_script_dir)):
        del _sys.modules["trl"]
if _os.path.basename(_script_dir) != "gear_sonic" and "gear_sonic" not in _sys.modules:
    import types as _types
    _alias = _types.ModuleType("gear_sonic")
    _alias.__path__ = [_script_dir]
    _alias.__file__ = _os.path.join(_script_dir, "__init__.py")
    _sys.modules["gear_sonic"] = _alias

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for evaluation but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    import sys
    sys.exit(1)

import filelock  # noqa: I001
import json
import os
import shutil
import subprocess
import sys

sys.path.append(os.getcwd())
import logging
from pathlib import Path

import easydict
import hydra
from hydra import utils
from hydra.core import hydra_config
from loguru import logger
import omegaconf
import yaml

from gear_sonic import train_agent_trl
from gear_sonic.trl.utils import common as trl_utils_common
from gear_sonic.trl.utils import scheduler
from gear_sonic.utils import common as rl_utils_common
from gear_sonic.utils import config_utils, obs_utils

config_utils.register_rl_resolvers()


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: omegaconf.OmegaConf):

    hydra_log_path = os.path.join(hydra_config.HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gear_sonic.utils import logging as utils_logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(utils_logging.HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                raw = file.read()
            # Backward compatibility: rewrite internal repo module paths to release repo paths
            raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
            raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
            raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
            raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
            raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
            raw = raw.replace("groot/rl/data/", "gear_sonic/data/")
            raw = raw.replace("assets/bm/unitree_description/", "assets/robot_description/")
            raw = raw.replace("1215_bones_seed_filtered", "bones_seed_smpl")
            import io
            train_config = omegaconf.OmegaConf.load(io.StringIO(raw))

            if train_config.eval_overrides is not None:
                train_config = omegaconf.OmegaConf.merge(train_config, train_config.eval_overrides)

            config = omegaconf.OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    elif override_config.eval_overrides is not None:
        config = override_config.copy()
        eval_overrides = omegaconf.OmegaConf.to_container(config.eval_overrides, resolve=True)
        for arg in sys.argv[1:]:
            if not arg.startswith("+"):
                key = arg.split("=")[0]
                if key in eval_overrides:
                    del eval_overrides[key]
        config.eval_overrides = omegaconf.OmegaConf.create(eval_overrides)
        config = omegaconf.OmegaConf.merge(config, eval_overrides)
    else:
        config = override_config

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))  # noqa: SIM115
        if config.get("wandb", None) is not None and meta.get("wandb_run"):
            config.wandb.wandb_id = meta["wandb_run"]
            print(f"resume wandb from run: {config.wandb.wandb_id}")  # noqa: T201

    with omegaconf.open_dict(config):
        for event in config.manager_env.config.get("train_only_events", []):
            if event in config.manager_env.events:
                config.manager_env.events.pop(event)
            remove_schedule_keys = []
            for key in config.trainer.get("schedule_dict", {}):
                if event in key:
                    remove_schedule_keys.append(key)
            for key in remove_schedule_keys:
                config.trainer.schedule_dict.pop(key)

        for termination in config.manager_env.config.get("train_only_terminations", []):
            if termination in config.manager_env.terminations:
                config.manager_env.terminations.pop(termination)

    use_encoder = config.get("use_encoder", None)
    if use_encoder is not None:
        encoder_sample_probs = config.manager_env.commands.motion.encoder_sample_probs
        if encoder_sample_probs is not None:
            for encoder in encoder_sample_probs:
                if encoder != use_encoder:
                    encoder_sample_probs[encoder] = 0.0
            print(f"Using encoder: {use_encoder}")  # noqa: T201
            print(f"Encoder sample probs: {encoder_sample_probs}")  # noqa: T201

    simulator_type = "IsaacSim"
    env_config = config.manager_env

    import datetime as dt

    import accelerate
    import torch  # noqa: E402, RUF100

    kwargs = accelerate.InitProcessGroupKwargs(timeout=dt.timedelta(seconds=6000))
    accelerator = accelerate.Accelerator(kwargs_handlers=[kwargs])

    device = str(accelerator.device)
    if accelerator.device.type == "cuda":
        try:
            torch.cuda.set_device(accelerator.local_process_index)
        except Exception:  # noqa: S110, BLE001
            pass

    device = str(accelerator.device)
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    rl_utils_common.seeding(config.seed)

    def _pick_display_gpu_index(default_idx: int = 0) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,display_active,name", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    idx, active = int(parts[0]), parts[1].lower()
                    if active.startswith("enabled") or active.startswith("on"):
                        return idx
        except Exception:  # noqa: S110, BLE001
            pass
        return default_idx

    render_gpu_idx = _pick_display_gpu_index(default_idx=0)

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2
        import argparse

        parser = argparse.ArgumentParser(description="Evaluate an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args  # noqa: RUF005
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = env_config.config.get(
            "render_results", False
        ) or env_config.config.get("enable_cameras", False)

        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        base_kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )
        if args_cli.headless:
            args_cli.kit_args = base_kit_args + " --no-window"
        else:
            args_cli.kit_args = base_kit_args + f" --/renderer/activeGpu={render_gpu_idx}"

        _lock_path = "/tmp/isaaclab_app_launcher.lock"  # noqa: S108
        with filelock.FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    unresolved_conf = omegaconf.OmegaConf.to_container(config, resolve=False)  # noqa: F841
    os.chdir(hydra.utils.get_original_cwd())

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]

    if env_config.config.get("save_rendering_dir", None) is None:
        env_config.config.save_rendering_dir = str(
            checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
        )

    metrics_file = config.get("metrics_file", None)
    if metrics_file is not None:
        metrics_file = Path(metrics_file)
        assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist"
        if metrics_file.exists():
            metrics = json.load(open(metrics_file))  # noqa: SIM115
            all_dict = metrics["eval/all_metrics_dict"]

            # Check if this is grab evaluation (has success_lift)
            has_obj_metrics = "obj_pos_error" in all_dict
            if "success_lift" in all_dict:
                # Grab evaluation: prioritize failed grasps (not lifted) and terminated trajectories
                motion_keys = all_dict["motion_keys"]
                terminated = all_dict["terminated"]
                success_lift = all_dict["success_lift"]
                progress = all_dict.get("progress", [1.0] * len(motion_keys))
                obj_pos_errors = all_dict.get("obj_pos_error", [0.0] * len(motion_keys))

                pairs = []
                for i in range(len(motion_keys)):
                    term = bool(terminated[i]) if i < len(terminated) else False
                    lifted = bool(success_lift[i]) if i < len(success_lift) else False
                    prog = progress[i] if i < len(progress) else 1.0
                    obj_err = obj_pos_errors[i] if i < len(obj_pos_errors) else 0.0
                    priority = 0 if not lifted else (1 if term else 2)
                    pairs.append((motion_keys[i], term, lifted, prog, obj_err, priority))

                pairs_sorted = sorted(pairs, key=lambda x: (x[5], x[3]))
                if len(pairs_sorted) > config.num_envs:
                    pairs_sorted = pairs_sorted[: config.num_envs]

                render_info = []
                for pair in pairs_sorted:
                    motion_key, term, lifted, prog, obj_err, _ = pair
                    status = "FAILED" if not lifted else ("TERMINATED" if term else "SUCCESS")
                    info = [
                        f"{motion_key}",
                        f"lifted: {lifted}",
                        f"progress: {prog:.3f}",
                        f"status: {status}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {obj_err:.4f}m")
                    render_info.append(tuple(info))

                filter_keys = [pair[0] for pair in pairs_sorted]

                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(render_info)
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys
            else:
                # Imitation evaluation: use MPJPE-based sorting
                obj_pos_errors = all_dict.get("obj_pos_error", None)
                success_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        True,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if not all_dict["terminated"][i]
                ]
                render_sort_by = config.get("render_sort_by", "mpjpe_l")
                sort_idx = 4 if render_sort_by == "obj_pos_error" else 1
                success_pair_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
                failed_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        False,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if all_dict["terminated"][i]
                ]
                failed_pair_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)
                all_pair = failed_pair_sorted + success_pair_sorted
                if len(all_pair) > config.num_envs:
                    all_pair = all_pair[: config.num_envs]
                render_info = []
                for pair in all_pair:
                    info = [
                        f"{pair[0]}",
                        f"mpjpe_l: {pair[1]:.2f}",
                        f"mpjpe_g: {pair[2]:.2f}",
                        f"success: {pair[3]}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {pair[4]:.4f}m")
                    render_info.append(tuple(info))
                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(all_pair)
                filter_keys = [pair[0] for pair in all_pair]
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys

    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy_backbone_kwargs = {}
    critic_backbone_kwargs = {}
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = (
                obs_utils.get_group_term_obs_shape(example_obs, key)
            )
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)

    if not getattr(config.algo.config, "distill_only", False):
        value_model = trl_utils_common.custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=critic_backbone_kwargs,
            _resolve=False,
        ).to(device)

    accelerator.wait_for_everyone()

    args = easydict.EasyDict()
    args.is_main_process = accelerator.is_main_process
    args.global_rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    state = easydict.EasyDict()

    from gear_sonic.trl.trainer import ppo_trainer

    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)

    checkpoint_path = str(config.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)

    # Load policy state dict with backward compatibility for std/log_std
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = None

    if state_dict is not None:
        model_uses_std = "std" in model.policy.state_dict()
        checkpoint_has_std = "std" in state_dict
        checkpoint_has_log_std = "log_std" in state_dict

        logger.info(f"Model parameterization: {'std' if model_uses_std else 'log_std'}")
        logger.info(
            f"Checkpoint parameterization: {'std' if checkpoint_has_std else 'log_std' if checkpoint_has_log_std else 'unknown'}"  # noqa: E501
        )

        if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
            logger.info("Transforming 'log_std' -> 'std' (applying exp) for backward compatibility")
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
            logger.info("Transforming 'std' -> 'log_std' (applying log) for backward compatibility")
            state_dict["log_std"] = torch.log(state_dict.pop("std"))

        # ---- LoRA wrap reconstruction ----
        # Train flow does (build → resize g1_dyn → load pretrained → wrap → save).
        # Eval ckpt has the wrapped structure (keys like
        # `actor_module.pretrained.decoders.g1_dyn.module.0.lora_A` /
        # `.original.weight`). We must mirror the resize + wrap before
        # load_state_dict, otherwise the bare UniversalTokenModule keys won't
        # line up with the ckpt's wrapped keys.
        has_lora_keys = any(
            ".pretrained." in k or ".lora_A" in k or ".lora_B" in k
            for k in state_dict.keys()
        )
        lora_cfg = (
            config.algo.config.get("lora", None)
            if hasattr(config, "algo") and config.algo is not None
            else None
        )
        if has_lora_keys and lora_cfg is not None:
            logger.info("Detected LoRA-wrapped checkpoint; rebuilding wrapper before load.")

            actor = model.policy.actor_module

            # Resize g1 encoder first Linear to match ckpt's .original.weight
            # shape. Without this, the env-built encoder Linear has the Oli
            # input dim (e.g. 680 = 2*N_fut*31 + 6*N_fut) while the wrapped
            # ckpt was saved at the G1 input dim (640 = 2*N_fut*29 + 6*N_fut).
            # Mirror the train-time resize so strict-load below succeeds.
            if hasattr(actor, "encoders") and "g1" in actor.encoders:
                g1_enc = actor.encoders["g1"]
                enc_seq = getattr(g1_enc, "module", None)
                if isinstance(enc_seq, torch.nn.Sequential):
                    enc_linear_idxs = [
                        i for i, l in enumerate(enc_seq) if isinstance(l, torch.nn.Linear)
                    ]
                    if enc_linear_idxs:
                        first_enc_idx = enc_linear_idxs[0]
                        # When the encoder was LoRA-wrapped at training time
                        # (encoder_rank > 0), the original Linear weight is
                        # stored under ``...module.{i}.original.weight``.
                        # When encoder_rank=0 was used (plain frozen Linear,
                        # no LoRA injection), the weight stays at the bare
                        # ``...module.{i}.weight`` key. Try both so we resize
                        # regardless of which mode the ckpt used.
                        fk_enc_lora = (
                            f"actor_module.pretrained.encoders.g1.module.{first_enc_idx}.original.weight"
                        )
                        fk_enc_plain = (
                            f"actor_module.pretrained.encoders.g1.module.{first_enc_idx}.weight"
                        )
                        fk_enc = None
                        if fk_enc_lora in state_dict:
                            fk_enc = fk_enc_lora
                        elif fk_enc_plain in state_dict:
                            fk_enc = fk_enc_plain
                        if fk_enc is not None:
                            ckpt_enc_w = state_dict[fk_enc]
                            cur_enc = enc_seq[first_enc_idx]
                            if (
                                cur_enc.in_features != ckpt_enc_w.shape[1]
                                or cur_enc.out_features != ckpt_enc_w.shape[0]
                            ):
                                new_enc = torch.nn.Linear(
                                    ckpt_enc_w.shape[1],
                                    ckpt_enc_w.shape[0],
                                    bias=cur_enc.bias is not None,
                                ).to(cur_enc.weight.device, cur_enc.weight.dtype)
                                enc_seq[first_enc_idx] = new_enc
                                # Patch the encoder module's input_dim used by
                                # base_module.forward to reshape inputs.
                                old_input_dim = getattr(g1_enc, "input_dim", None)
                                if old_input_dim is not None:
                                    ratio = ckpt_enc_w.shape[1] / cur_enc.in_features
                                    g1_enc.input_dim = int(old_input_dim * ratio)
                                logger.info(
                                    f"Resized encoders.g1.module[{first_enc_idx}] "
                                    f"({cur_enc.in_features}→{ckpt_enc_w.shape[1]} in, "
                                    f"{cur_enc.out_features}→{ckpt_enc_w.shape[0]} out); "
                                    f"updated g1_enc.input_dim {old_input_dim}→"
                                    f"{getattr(g1_enc, 'input_dim', None)}"
                                )

            # Resize g1_dyn first/last Linear to match ckpt's .original.weight shapes.
            if hasattr(actor, "decoders") and "g1_dyn" in actor.decoders:
                g1_dyn = actor.decoders["g1_dyn"]
                seq = getattr(g1_dyn, "module", None)
                if isinstance(seq, torch.nn.Sequential):
                    linear_idxs = [
                        i for i, l in enumerate(seq) if isinstance(l, torch.nn.Linear)
                    ]
                    if linear_idxs:
                        first_idx, last_idx = linear_idxs[0], linear_idxs[-1]
                        # Wrapped ckpt stores Linear weights under .original.weight
                        first_key = (
                            f"actor_module.pretrained.decoders.g1_dyn.module.{first_idx}.original.weight"
                        )
                        last_key = (
                            f"actor_module.pretrained.decoders.g1_dyn.module.{last_idx}.original.weight"
                        )
                        if first_key in state_dict and last_key in state_dict:
                            ckpt_first_w = state_dict[first_key]
                            ckpt_last_w = state_dict[last_key]
                            cur_first = seq[first_idx]
                            cur_last = seq[last_idx]
                            if (
                                cur_first.in_features != ckpt_first_w.shape[1]
                                or cur_first.out_features != ckpt_first_w.shape[0]
                            ):
                                new_first = torch.nn.Linear(
                                    ckpt_first_w.shape[1],
                                    ckpt_first_w.shape[0],
                                    bias=cur_first.bias is not None,
                                ).to(cur_first.weight.device, cur_first.weight.dtype)
                                seq[first_idx] = new_first
                                logger.info(
                                    f"Resized decoders.g1_dyn.module[{first_idx}] "
                                    f"({cur_first.in_features}→{ckpt_first_w.shape[1]} in, "
                                    f"{cur_first.out_features}→{ckpt_first_w.shape[0]} out)"
                                )
                            if (
                                cur_last.in_features != ckpt_last_w.shape[1]
                                or cur_last.out_features != ckpt_last_w.shape[0]
                            ):
                                new_last = torch.nn.Linear(
                                    ckpt_last_w.shape[1],
                                    ckpt_last_w.shape[0],
                                    bias=cur_last.bias is not None,
                                ).to(cur_last.weight.device, cur_last.weight.dtype)
                                seq[last_idx] = new_last
                                logger.info(
                                    f"Resized decoders.g1_dyn.module[{last_idx}] "
                                    f"({cur_last.in_features}→{ckpt_last_w.shape[1]} in, "
                                    f"{cur_last.out_features}→{ckpt_last_w.shape[0]} out)"
                                )
                            # Sync dim maps
                            token_total = actor.token_total_dim
                            new_proprio_dim = ckpt_first_w.shape[1] - token_total
                            actor.decoder_feature_dims_map["proprioception"] = new_proprio_dim
                            out_dims = actor.decoder_output_feature_dims.get("g1_dyn")
                            if out_dims is not None:
                                keys = list(out_dims.keys())
                                head_sum = sum(out_dims[k] for k in keys[:-1])
                                out_dims[keys[-1]] = ckpt_last_w.shape[0] - head_sum

            # Wrap with OliLoRAWrapper using the same kwargs as train.
            from gear_sonic.trl.modules.oli_lora_wrapper import OliLoRAWrapper

            oli_proprio_dim = env.config["robot"]["algo_obs_dim_dict"]["actor_obs"]

            def _eval_int_list(key):
                v = lora_cfg.get(key, None)
                if v is None:
                    return None
                return [int(i) for i in v]

            wrapper = OliLoRAWrapper(
                pretrained_module=model.policy.actor_module,
                oli_proprioception_dim=oli_proprio_dim,
                lora_rank=lora_cfg.get("rank", 8),
                lora_alpha=lora_cfg.get("alpha", 16.0),
                lora_dropout=lora_cfg.get("dropout", 0.0),
                lora_skip_last=lora_cfg.get("skip_last", False),
                learnable_head=lora_cfg.get("learnable_head", True),
                variant=lora_cfg.get("variant", "lora"),
                rslora=lora_cfg.get("rslora", False),
                use_gate=lora_cfg.get("use_gate", False),
                init_strategy=lora_cfg.get("init_strategy", "kaiming"),
                encoder_rank=lora_cfg.get("encoder_rank", None),
                decoder_rank=lora_cfg.get("decoder_rank", None),
                encoder_alpha=lora_cfg.get("encoder_alpha", None),
                decoder_alpha=lora_cfg.get("decoder_alpha", None),
                encoder_active_linears=_eval_int_list("encoder_active_linears"),
                decoder_active_linears=_eval_int_list("decoder_active_linears"),
                proprio_mode=lora_cfg.get("proprio_mode", "linear"),
                proprio_actor_history_length=int(
                    lora_cfg.get("proprio_actor_history_length")
                    or config.get("actor_prop_history_length", 10)
                ),
                proprio_action_history_length=int(
                    lora_cfg.get("proprio_action_history_length")
                    or config.get("actor_actions_history_length", 10)
                ),
                unfreeze_backbone=lora_cfg.get("unfreeze_backbone", False),
            )
            model.policy.actor_module = wrapper.to(accelerator.device)
            logger.info("Wrapper rebuilt; loading eval state_dict next.")

        # strict=False because the wrapper's `self.encoders = self.pretrained.encoders`
        # alias produces duplicated keys; load_state_dict still copies the values
        # correctly, but strict mode flags the dups as unexpected.
        missing, unexpected = model.policy.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"load_state_dict missing keys: {len(missing)} ({missing[:3]}...)")
        if unexpected:
            logger.info(f"load_state_dict unexpected keys: {len(unexpected)} ({unexpected[:3]}...)")
        logger.info("Successfully loaded policy state dict")

    state.global_step = checkpoint["state"].global_step

    schedule_wrapper = easydict.EasyDict(env=env, model=model)
    if "schedule_dict" in config.trainer:
        scheduled_params_dict = scheduler.update_scheduled_params(  # noqa: F841
            schedule_wrapper, config.trainer.schedule_dict, state.global_step
        )
    env.reinit_dr()

    global_step = checkpoint["state"].global_step
    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_{global_step}.onnx"
    new_cp_path = f"{os.path.dirname(config.checkpoint)}/model_{global_step}.pt"
    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if config.get("export_onnx_only", False):

        def get_example_obs():
            obs_dict = env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].cpu()
            return obs_dict

        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gear_sonic.utils import inference_helpers

        example_obs_dict = get_example_obs()

        # Check if actor has universal-token encoder structure
        has_actor_module = hasattr(model.policy, "actor_module")
        has_encoders = has_actor_module and hasattr(
            model.policy.actor_module, "encoders_to_iterate"
        )
        # OliLoRAWrapper hides the universal-token module under .pretrained
        # (no .encoders_to_iterate on the wrapper itself). For LoRA ckpts we
        # want a single self-contained ONNX (prop slice + encoder + LoRA
        # decoder + DofBridge + hip-pitch decomp all baked in).
        is_oli_lora = has_actor_module and hasattr(
            model.policy.actor_module, "pretrained"
        ) and hasattr(model.policy.actor_module, "dof_bridge")

        if is_oli_lora and "tokenizer" in example_obs_dict:
            inference_helpers.export_oli_lora_as_onnx(
                model.policy,
                path=exported_policy_path,
                exported_model_name=exported_onnx_name,
                batch_size=1,
            )
        elif "tokenizer" in example_obs_dict and has_encoders:

            # Only export the encoders that are actually active for this run.
            # LoRA configs (e.g. sonic_oli_lora) drop smpl / teleop entirely,
            # so unconditionally trying to export them would crash.
            active_encoders = list(
                config.algo.config.actor.backbone.get("active_encoders", ["g1"])
            )
            for enc_name in active_encoders:
                inference_helpers.export_universal_token_module_as_onnx(
                    model.policy.actor_module,
                    encoder_name=enc_name,
                    decoder_name="g1_dyn",
                    path=exported_policy_path,
                    exported_model_name=exported_onnx_name.replace(
                        ".onnx", f"_{enc_name}.onnx"
                    ),
                    batch_size=1,
                )

            inference_helpers.export_universal_token_encoders_as_onnx(
                model.policy.actor_module,
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_encoder.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_decoder_as_onnx(
                model.policy.actor_module,
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_decoder.onnx"),
                batch_size=1,
            )
            print(  # noqa: T201
                f'Exported encoders ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_encoder.onnx"))}'  # noqa: E501
            )
            print(  # noqa: T201
                f'Exported decoder ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_decoder.onnx"))}'  # noqa: E501
            )

        else:
            inference_helpers.export_policy_as_onnx(
                {"actor": model.policy}, exported_policy_path, exported_onnx_name, example_obs_dict
            )

        logger.info(f"Exported policy as onnx to: {os.path.join(exported_policy_path)}")

        # Export configs to YAML
        export_config = {
            "env_config": omegaconf.OmegaConf.to_container(env.config, resolve=True),
            "algo_config": omegaconf.OmegaConf.to_container(config.algo.config, resolve=True),
        }
        config_yaml_path = os.path.join(os.path.dirname(config.checkpoint), "model_config.yaml")
        with open(config_yaml_path, "w") as f:
            yaml.dump(export_config, f, default_flow_style=False)
        logger.info(f"Exported config to: {config_yaml_path}")
        exit()  # noqa: PLR1722

    eval_callbacks = config.get("eval_callbacks", [])
    if isinstance(eval_callbacks, str):
        eval_callbacks = [eval_callbacks]

    callbacks = {}
    for callback_name in eval_callbacks:
        if callback_name == "im_eval":
            with omegaconf.open_dict(config.callbacks.im_eval):
                config.callbacks.im_eval.eval_only = True
                config.callbacks.im_eval.eval_frequency = 1
                config.callbacks.im_eval.output_dir = config.get("eval_output_dir", None)
                config.callbacks.im_eval.log_keys = config.get("log_keys", None)
        if callback_name not in config.callbacks:
            raise ValueError(f"Callback {callback_name} not found")
        callbacks[callback_name] = utils.instantiate(config.callbacks[callback_name])

    for callback_name, callback in callbacks.items():  # noqa: B007
        if hasattr(callback, "model") and callback.model is None:
            callback.model = model

    for callback_name, callback in callbacks.items():  # noqa: B007
        callback.on_step_end(args, state, None, env=env, model=model, accelerator=accelerator)

    if config.get("run_eval_loop", True):
        env.set_is_evaluating(True)
        obs_dict = env.reset_all()
        model.eval()
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        eval_step_callbacks = {
            name: cb
            for name, cb in callbacks.items()
            if hasattr(cb, "eval_step") and callable(getattr(cb, "eval_step"))  # noqa: B009
        }
        if eval_step_callbacks:
            logger.info(f"Eval step callbacks enabled: {list(eval_step_callbacks.keys())}")

        step_count = 0
        max_render_steps = config.get("max_render_steps", 0)

        run_once = config.get("run_once", False)
        envs_completed = torch.zeros(config.num_envs, dtype=torch.bool, device=device)

        with torch.no_grad():
            while True:
                policy_model = model.policy
                value_model = model.value_model
                policy_model.init_rollout()

                actor_state = {}
                actions = policy_model.rollout(obs_dict=obs_dict)
                actor_state["actions"] = policy_model.action_mean.detach()
                actor_state["obs_dict"] = actions["obs_dict"]

                step_count += 1

                if max_render_steps > 0 and step_count >= max_render_steps:
                    logger.info(f"Reached max_render_steps={max_render_steps}. Exiting.")
                    if hasattr(env, "end_render_results"):
                        env.end_render_results()
                    break

                results = env.step(actor_state)
                obs_dict, rewards, dones, infos = (
                    results[0],
                    results[1],
                    results[2],
                    results[3],
                )  # noqa: F841

                if eval_step_callbacks:
                    all_want_exit = all(
                        cb.eval_step(env, results) for cb in eval_step_callbacks.values()
                    )
                    if all_want_exit:
                        logger.info("All eval step callbacks signaled exit. Exiting evaluation loop.")
                        break

                if run_once:
                    envs_completed = (
                        envs_completed | dones.squeeze(-1)
                        if dones.dim() > 1
                        else envs_completed | dones
                    )
                    if envs_completed.all():
                        logger.info("All environments completed one episode. Exiting (run_once=True).")
                        if hasattr(env, "end_render_results"):
                            env.end_render_results()
                        break

                for obs_key in obs_dict.keys():  # noqa: SIM118
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":
    main()
