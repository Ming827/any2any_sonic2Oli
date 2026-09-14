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

# Always show full Hydra tracebacks — without this, a config-time exception
# is collapsed to the bottom 3 frames and you can't see the real cause.
# Set before importing hydra; setting it later still works but may miss the
# very-early Hydra config errors.
import os
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# Reduce CUDA memory fragmentation. With LoRA wrapper + many envs (e.g.
# 16384), PyTorch's default allocator leaves multi-GB "reserved but
# unallocated" gaps that trigger OOM despite enough total free memory.
# `expandable_segments:True` lets allocations grow into freed gaps instead
# of needing contiguous reserved blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Disable wandb by default. Even in `offline` mode wandb keeps a per-iter
# growing in-memory metric cache (~2.5 MB/iter for our setup), which
# slows down long PPO runs by 30-40% and causes a linear RSS leak.
# TensorBoard (set up by the trainer separately) covers the same logging
# need without the leak. Users who really want wandb can opt back in by
# setting `WANDB_MODE=online` (or `offline`) explicitly in the env.
os.environ.setdefault("WANDB_MODE", "disabled")

# --- NCCL multi-GPU networking defaults ---
# On company training servers with multiple NICs (mgmt / business / k8s
# overlay), NCCL's auto-pick can land on an unreachable interface and crash
# at the first `accelerator.wait_for_everyone()` with:
#   ncclSystemError: socketStartConnect ... Software caused connection abort
# Force loopback for single-node multi-GPU and disable IB (containers often
# lack a working IB stack). `setdefault` so the user can override at runtime
# if they have a real high-speed fabric:
#     NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 python train_agent_trl.py ...
# FORCE-set (not setdefault) — company task scheduler may inject NCCL_*
# env vars pointing at unreachable mgmt-network IPs. We override.
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

# Sanity beacon — proves this exact version is running. Look for
# [NCCL-FIX-V3] in the task log; if missing, server is on stale code.
print(  # noqa: T201
    f"[NCCL-FIX-V3] pid={os.getpid()} rank={os.environ.get('RANK', 'parent')} "
    f"MASTER_ADDR={os.environ.get('MASTER_ADDR', '<unset>')} "
    f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')} "
    f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE')} "
    f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}",
    flush=True,
)

# Fix sys.path: when running as `python gear_sonic/train_agent_trl.py` (or with
# cwd == gear_sonic/), Python adds gear_sonic/ to sys.path, causing
# `from trl import ...` to resolve to our local gear_sonic/trl/ instead of the
# HuggingFace trl package. Strip ANY entry that normalizes to gear_sonic/, ''
# or '.' (cwd shortcuts), then prepend the repo root so `from gear_sonic.X import Y`
# still works.
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
_bad_norm = {
    os.path.normpath(_script_dir),
    os.path.normpath(os.getcwd()) if os.getcwd() == _script_dir else None,
    "",
    ".",
}
_bad_norm.discard(None)
sys.path[:] = [
    p for p in sys.path
    if (os.path.normpath(p) if p not in ("", ".") else p) not in _bad_norm
]
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# If a stale `trl` module already got loaded as our local gear_sonic/trl/,
# evict it so the next `import trl` reaches the installed HuggingFace package.
_trl_mod = sys.modules.get("trl")
if _trl_mod is not None:
    _trl_file = getattr(_trl_mod, "__file__", "") or ""
    if _trl_file and os.path.normpath(_trl_file).startswith(os.path.normpath(_script_dir)):
        del sys.modules["trl"]

# When this repo is cloned under a non-canonical folder name (e.g.
# ``sonic_WBT`` instead of ``gear_sonic``), all the ``from gear_sonic.X``
# imports fail. Register a runtime alias: a fake ``gear_sonic`` module whose
# __path__ points at this directory, so submodule lookup
# (``from gear_sonic.utils.X import Y``) is forwarded here verbatim.
# No-op when the folder is already named ``gear_sonic``.
if os.path.basename(_script_dir) != "gear_sonic" and "gear_sonic" not in sys.modules:
    import types as _types
    _alias = _types.ModuleType("gear_sonic")
    _alias.__path__ = [_script_dir]
    _alias.__file__ = os.path.join(_script_dir, "__init__.py")
    sys.modules["gear_sonic"] = _alias


# --- Multi-GPU launcher (no torchrun / accelerate launch needed) ---
# Usage:
#   python train_agent_trl.py +exp=... nproc_per_node=8 [master_port=29500]
# When ``nproc_per_node`` > 1 and we're the parent process (no RANK env var),
# we spawn N child processes via subprocess, each with RANK / LOCAL_RANK /
# WORLD_SIZE / MASTER_ADDR / MASTER_PORT env vars set. Accelerate inside the
# children auto-detects the distributed env and runs DDP via NCCL. The parent
# never imports isaaclab — only the children do.
def _maybe_spawn_distributed_children():
    if os.environ.get("RANK") is not None:
        return  # already a child
    nproc = None
    master_port = None
    keep_argv = []
    for arg in sys.argv:
        if arg.startswith("nproc_per_node="):
            nproc = int(arg.split("=", 1)[1])
            continue
        if arg.startswith("master_port="):
            master_port = arg.split("=", 1)[1]
            continue
        keep_argv.append(arg)
    if nproc is None or nproc <= 1:
        return
    if master_port is None:
        master_port = os.environ.get("MASTER_PORT", "29500")

    import signal as _signal
    import subprocess as _sp

    procs = []
    try:
        for i in range(nproc):
            env = os.environ.copy()
            # Force loopback for MASTER_ADDR: this launcher is single-node
            # only (no node_rank/nnodes handling), so respecting an inherited
            # MASTER_ADDR from a task scheduler is wrong — that IP often
            # belongs to a different node / unreachable mgmt interface and
            # leads to NCCL ncclSystemError: socketStartConnect ... aborted.
            # Override the env value, don't fall back to it.
            # Pin each child to a single GPU via CUDA_VISIBLE_DEVICES so it
            # builds a CUDA context on ONE device only. Without this, every
            # rank initializes CUDA on all 4 GPUs (~500 MB context each) and
            # ends up wasting ~1.5 GB per GPU on sibling-process contexts —
            # the leftover memory we saw in OOM logs as "Process 27X has
            # 518.00 MiB memory in use". Only set when not already set
            # (e.g. user / task scheduler may have set it intentionally).
            child_visible = env.get("CUDA_VISIBLE_DEVICES")
            if child_visible is None or child_visible == "":
                env["CUDA_VISIBLE_DEVICES"] = str(i)
            env.update(
                {
                    "RANK": str(i),
                    "LOCAL_RANK": "0",   # rank-local GPU index → 0 since CUDA_VISIBLE_DEVICES isolates one
                    "WORLD_SIZE": str(nproc),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": master_port,
                }
            )
            print(  # noqa: T201
                f"[launcher] spawning child rank={i}/{nproc} "
                f"(MASTER_PORT={master_port})"
            )
            procs.append(_sp.Popen([sys.executable] + keep_argv, env=env))
        exit_code = 0
        for p in procs:
            p.wait()
            if p.returncode != 0:
                exit_code = p.returncode
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("[launcher] KeyboardInterrupt — terminating children")  # noqa: T201
        for p in procs:
            try:
                p.send_signal(_signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        for p in procs:
            try:
                p.wait(timeout=30)
            except Exception:  # noqa: BLE001
                p.kill()
        sys.exit(130)


_maybe_spawn_distributed_children()


try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for training but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    sys.exit(1)

import glob
import logging
import os
from pathlib import Path
import re
import sys

from filelock import FileLock
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml

from gear_sonic.trl.utils.common import (
    custom_instantiate,
    get_filtered_state_dict,
    materialize_lazy_params,
    wandb_run_exists,
)
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def resume_training(config):
    if config.get("checkpoint", None) is not None:
        last_existing_checkpoint = config.checkpoint
    elif config.get("experiment_dir", None) is not None:
        last_existing_checkpoint = os.path.join(config.experiment_dir, "last.pt")
    else:
        # Use experiment_dir to find the checkpoint, rather than reconstructing
        # from config.project_name which can differ from the actual filesystem path.
        experiment_dir_base = re.sub(r"-\d{8}_\d{6}$", "", config.experiment_dir)
        checkpoints = sorted(glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt")))
        if not checkpoints:
            print(f"No checkpoint found matching {experiment_dir_base}-*/last.pt, starting fresh")
            return
        last_existing_checkpoint = checkpoints[-1]
    experiment_dir = os.path.dirname(last_existing_checkpoint)
    config.experiment_dir = experiment_dir
    config.checkpoint = last_existing_checkpoint
    print(f"Resuming training from {last_existing_checkpoint}")


def resume_checkpoint(config):
    config.checkpoint = config.checkpoint


def create_manager_env(config, device, args_cli):

    # import wandb

    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )

    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper

    env_instance_cfg = custom_instantiate(config.manager_env)

    # Iteratively check the difference in attribute of env_instance_cfg1 and env_instance_cfg, print out the difference
    def compare_attrs(obj1, obj2, prefix=""):
        # Only compare attributes that do not start with '__' and are not methods
        attrs1 = set(dir(obj1))
        attrs2 = set(dir(obj2))
        common_attrs = attrs1 & attrs2
        for attr in sorted(common_attrs):
            if (
                attr.startswith("__")
                or callable(getattr(obj1, attr))
                or callable(getattr(obj2, attr))
            ):
                continue
            try:
                val1 = getattr(obj1, attr)
                val2 = getattr(obj2, attr)
            except Exception:
                continue
            # Recursively compare if both are objects with __dict__ or are dicts
            if isinstance(val1, dict | DictConfig) and isinstance(val2, dict | DictConfig):
                compare_attrs(val1, val2, prefix + attr + ".")
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                compare_attrs(val1, val2, prefix + attr + ".")
            else:
                if isinstance(val1, list):
                    val1 = tuple(val1)
                if isinstance(val2, list):
                    val2 = tuple(val2)
                if val1 != val2:
                    print(
                        f"\nDifference found at '{prefix}{attr}':\n"
                        f"  - env_instance_cfg1: {val1!r}\n"
                        f"  - env_instance_cfg : {val2!r}\n"
                    )

    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = args_cli.headless
    env = ManagerBasedRLEnv(
        cfg=env_instance_cfg, render_mode="rgb_array" if not args_cli.headless else None
    )

    env = ManagerEnvWrapper(env, env_instance_cfg.config)
    return env


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = "IsaacSim"
    env_config = config.manager_env
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    # Setup model components
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

    if config.get("resume", False):
        resume_training(config)
    elif config.get("checkpoint", None) is not None:
        resume_checkpoint(config)

    config.algo.trl.output_dir = str(Path(config.experiment_dir))

    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    # Add exp_name from main config to training_args
    training_args.exp_name = config.experiment_name

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    import torch  # noqa: E402

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.use_wandb and accelerator.is_main_process:
        project_name = f"{config.project_name}"
        run_name = config.experiment_dir.replace(f"{config.base_dir}/{project_name}/", "")
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        # Auto-fallback to offline mode when no API key is available and the
        # user hasn't explicitly set WANDB_MODE. Avoids the no-tty UsageError
        # on remote / containerized boxes where interactive login is impossible.
        if (
            "WANDB_MODE" not in os.environ
            and not os.environ.get("WANDB_API_KEY")
            and not (Path.home() / ".netrc").exists()
        ):
            os.environ["WANDB_MODE"] = "offline"
            logger.info(
                "[wandb] No API key detected and WANDB_MODE not set — "
                "forcing offline mode. Run `wandb sync <dir>` later to upload."
            )
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="allow",
        )

    # Setup simulator similar to train_agent.py

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

            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        ######################################################### ZL: fix isaacsim 4.5 rendering #########################################################
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        # Enable cameras if enable_cameras, render_results, render_ego, or overview_camera is True
        args_cli.enable_cameras = (
            env_config.config.get("enable_cameras", False)
            or env_config.config.get("render_results", False)
            or env_config.config.get("render_ego", False)
            or env_config.config.get("overview_camera", False)
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Base kit args (quiet logs)
        args_cli.kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )

        # AppLauncher can't handle multiple processes creating it at the same time so we need a lock
        _lock_path = "/tmp/isaaclab_app_launcher.lock"
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        with FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)

        simulation_app = app_launcher.app

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from gear_sonic.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # Setup wandb if enabled
    os.chdir(hydra.utils.get_original_cwd())

    # Save config and meta BEFORE env creation so eval jobs can postprocess
    # checkpoint configs even if training crashes during env init.
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": wandb.run.id if wandb_run_exists() else None}
        meta["max_train_steps"] = config.algo.config.num_learning_iterations
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    # Initialize environment
    env_config.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env_config.config.experiment_dir = str(Path(config.experiment_dir))

    env = create_manager_env(config, device, args_cli)
    if config.get("replay", False):
        _save_video_path = config.get("replay_save_video", None)
        env.run_replay(
            start_time_step=-1,
            loop=config.get("replay_loop_num", True),
            save_video_path=_save_video_path,
            grid_spacing=config.get("replay_grid_spacing", 2.0),
        )
        os._exit(0)
    if config.get("vplanner_replay", False):
        vplanner_checkpoint = config.get("vplanner_checkpoint", None)
        if vplanner_checkpoint is None:
            raise ValueError("vplanner_checkpoint must be specified for vplanner_replay")
        env.run_vplanner_replay(
            checkpoint_path=vplanner_checkpoint,
            max_frames=config.get("vplanner_max_frames", 500),
            replan_interval=config.get("vplanner_replan_interval", 0),
            speed=config.get("vplanner_speed", 1.0),
            loop=config.get("vplanner_loop", True),
            save_images=config.get("vplanner_save_images", False),
            output_dir=config.get("vplanner_output_dir", None),
            dof_noise=config.get("vplanner_dof_noise", 0.0),
            dof_vel_noise=config.get("vplanner_dof_vel_noise", 0.0),
            quat_noise=config.get("vplanner_quat_noise", 0.0),
        )
        os._exit(0)

    ref_model = None
    value_model = None
    disc_model = None
    # import ipdb; ipdb.set_trace()

    if config.algo.config.get("use_new_actor_critic", False):
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
                group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
                    example_obs, key
                )
                env.config["obs"]["group_obs_dims"][key] = group_obs_dims
                env.config["obs"]["group_obs_names"][key] = group_obs_names
                env.config["obs"]["obs_dims"][key] = group_obs_total_dim
                env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
        if config.manager_env.config.get("meta_action_dim", None) is not None:
            env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
        else:
            env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=policy_backbone_kwargs,
            _resolve=False,
        ).to(device)

        if getattr(config.algo.config, "use_dagger", False):
            # Get teacher input key from config or default to "teacher"
            teacher_input_key = config.algo.config.get("teacher_input_key", "teacher")
            ref_model = custom_instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
                input_key=teacher_input_key,
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False):
            value_model = custom_instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                backbone_kwargs=critic_backbone_kwargs,
                _resolve=False,
            ).to(device)
        if config.algo.config.get("use_amp", False):
            disc_model = custom_instantiate(
                config.algo.config.disc,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            ).to(device)
    else:
        raise ValueError("No longer supported")

    materialize_lazy_params(policy, env)

    if config.algo.config.get("pretrained_model", None) is not None:
        pretrained_cfg = config.algo.config.pretrained_model
        sd_key = pretrained_cfg.get("state_dict_key", "state_dict")
        strict = pretrained_cfg.get("strict", True)

        # Older SONIC checkpoints were pickled against trl versions that
        # exposed classes (OnlineTrainerState, legacy PPOConfig, ...) which the
        # current trl 0.28.0 + the gear_sonic.trl shadow package no longer
        # provide. We don't actually need those non-tensor objects — only the
        # model state_dict. Iteratively install shim modules / stub classes
        # for each missing name until torch.load succeeds.
        import types as _types

        class _LegacyStub:  # noqa: D401 — pickle placeholder, discarded after load
            def __setstate__(self, state):
                self.__dict__.update(state if isinstance(state, dict) else {})

        _max_shim_iters = 16
        ckpt = None
        for _attempt in range(_max_shim_iters):
            try:
                ckpt = torch.load(
                    pretrained_cfg.path, map_location=device, weights_only=False
                )
                break
            except (ModuleNotFoundError, AttributeError) as e:
                msg = str(e)
                # ModuleNotFoundError: "No module named 'trl.trainer.utils'"
                if isinstance(e, ModuleNotFoundError) and "'" in msg:
                    missing_mod = msg.split("'")[1]
                    if missing_mod in sys.modules:
                        raise  # already shimmed but still missing — real bug
                    sys.modules[missing_mod] = _types.ModuleType(missing_mod)
                    logger.info(f"Shimmed missing module for legacy ckpt: {missing_mod}")
                    continue
                # AttributeError: "Can't get attribute 'Foo' on <module 'bar'...>"
                if isinstance(e, AttributeError):
                    import re as _re
                    m = _re.search(r"attribute '([^']+)'.*module '([^']+)'", msg)
                    if m is not None:
                        attr_name, mod_name = m.group(1), m.group(2)
                        mod = sys.modules.get(mod_name) or _types.ModuleType(mod_name)
                        sys.modules[mod_name] = mod
                        setattr(mod, attr_name, _LegacyStub)
                        logger.info(
                            f"Shimmed missing class for legacy ckpt: {mod_name}.{attr_name}"
                        )
                        continue
                raise
        else:
            raise RuntimeError(
                f"Could not load {pretrained_cfg.path} after {_max_shim_iters} shim attempts"
            )

        state_dict = ckpt[sd_key]

        # ---- Shape-align g1 encoder first Linear for cross-DOF LoRA ----
        # The env feeds Oli 31-DOF motion data so commands.py reset/reward see
        # all 31 joints. But the G1 ckpt's encoder.g1 first Linear was sized for
        # 29-DOF tokenizer obs (input dim = 2*N_fut*29 + ori_dim = 640 for
        # N_fut=10). With Oli 31-DOF the env-built encoder has input dim 680,
        # which can't load ckpt's 640. Shrink the encoder's first Linear to
        # match ckpt (640) — at runtime the OliLoRAWrapper slices
        # ``command_multi_future_nonflat`` from 31-DOF→29-DOF before feeding the
        # encoder (see slice_command_multi_future_to_g1).
        if (
            config.algo.config.get("lora", None) is not None
            and hasattr(policy, "actor_module")
            and hasattr(policy.actor_module, "encoders")
            and "g1" in policy.actor_module.encoders
        ):
            g1_enc = policy.actor_module.encoders["g1"]
            enc_seq = getattr(g1_enc, "module", None)
            if isinstance(enc_seq, torch.nn.Sequential):
                enc_linear_idxs = [
                    i for i, layer in enumerate(enc_seq) if isinstance(layer, torch.nn.Linear)
                ]
                if enc_linear_idxs:
                    first_enc_idx = enc_linear_idxs[0]
                    prefix_in_ckpt = pretrained_cfg.module_mapping.get(
                        "policy.actor_module", "actor_module."
                    )
                    fk_enc = f"{prefix_in_ckpt}encoders.g1.module.{first_enc_idx}.weight"
                    if fk_enc in state_dict:
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
                            # Also patch the encoder module's own input_dim (used
                            # by base_module.forward at line 554 to reshape the
                            # incoming tensor before the Linear). Without this,
                            # the layer would accept 640 but reshape still tries
                            # to enforce 680 → RuntimeError at runtime.
                            old_input_dim = getattr(g1_enc, "input_dim", None)
                            if old_input_dim is not None:
                                # input_dim is `feat_dim * num_input_temporal_dims`;
                                # only the feat part changed (DOF count from 31 to 29).
                                # Compute the scale factor: (new_feat / old_feat).
                                ratio_in_features = ckpt_enc_w.shape[1] / cur_enc.in_features
                                g1_enc.input_dim = int(old_input_dim * ratio_in_features)
                            logger.info(
                                f"Resized encoders.g1.module[{first_enc_idx}] "
                                f"({cur_enc.in_features}→{ckpt_enc_w.shape[1]} in, "
                                f"{cur_enc.out_features}→{ckpt_enc_w.shape[0]} out) to match ckpt; "
                                f"updated g1_enc.input_dim {old_input_dim}→{getattr(g1_enc, 'input_dim', None)}"
                            )

        # ---- Shape-align g1_dyn for cross-DOF LoRA (Oli env, G1 29-DOF ckpt) ----
        # When a 29-DOF G1 ckpt is loaded into a 31-DOF Oli model, the g1_dyn
        # decoder's first Linear (proprio+token in) and last Linear (action out)
        # have different shapes than the ckpt. The OliLoRAWrapper's design
        # assumes the wrapped module is already G1-shaped, so we resize those
        # two layers to match the ckpt before strict-loading. The runtime
        # dim translation (Oli 990 actor_obs → G1 930) is then handled by the
        # wrapper's `proprio_mode=slice` (or learned `linear`); the action 29→31
        # is handled by `DofBridge.g1_to_oli`.
        if (
            config.algo.config.get("lora", None) is not None
            and hasattr(policy, "actor_module")
            and hasattr(policy.actor_module, "decoders")
            and "g1_dyn" in policy.actor_module.decoders
        ):
            g1_dyn = policy.actor_module.decoders["g1_dyn"]
            seq = getattr(g1_dyn, "module", None)
            if isinstance(seq, torch.nn.Sequential):
                linear_idxs = [
                    i for i, layer in enumerate(seq) if isinstance(layer, torch.nn.Linear)
                ]
                if linear_idxs:
                    first_idx, last_idx = linear_idxs[0], linear_idxs[-1]
                    first_key = f"decoders.g1_dyn.module.{first_idx}.weight"
                    last_key = f"decoders.g1_dyn.module.{last_idx}.weight"
                    # Need to look the keys up under the actor_module prefix
                    prefix_in_ckpt = pretrained_cfg.module_mapping.get(
                        "policy.actor_module", "actor_module."
                    )
                    fk = prefix_in_ckpt + first_key
                    lk = prefix_in_ckpt + last_key
                    if fk in state_dict and lk in state_dict:
                        ckpt_first_w = state_dict[fk]
                        ckpt_last_w = state_dict[lk]
                        cur_first = seq[first_idx]
                        cur_last = seq[last_idx]

                        # First Linear: in dim shrinks 1054 → 994 (drop 2 head DOF × 30)
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
                                f"{cur_first.out_features}→{ckpt_first_w.shape[0]} out) to match ckpt"
                            )

                        # Last Linear: out dim shrinks 31 → 29 (drop head DOFs)
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
                                f"{cur_last.out_features}→{ckpt_last_w.shape[0]} out) to match ckpt"
                            )

                        # Keep the cached dim map in sync so OliLoRAWrapper builds
                        # proprio_proj with the right output dim (G1 930, not Oli 990).
                        token_total = policy.actor_module.token_total_dim
                        new_proprio_dim = ckpt_first_w.shape[1] - token_total
                        if (
                            policy.actor_module.decoder_feature_dims_map.get("proprioception")
                            != new_proprio_dim
                        ):
                            old = policy.actor_module.decoder_feature_dims_map["proprioception"]
                            policy.actor_module.decoder_feature_dims_map["proprioception"] = (
                                new_proprio_dim
                            )
                            logger.info(
                                f"Updated decoder_feature_dims_map['proprioception']: {old} → {new_proprio_dim}"
                            )

                        # Also resync the per-decoder output feature dims used by
                        # decode() (universal_token_modules.py:734), otherwise the
                        # internal `index == output.shape[-1]` assertion fires
                        # (index would still expect Oli 31, output is now ckpt 29).
                        new_action_out_dim = ckpt_last_w.shape[0]
                        out_dims = policy.actor_module.decoder_output_feature_dims.get("g1_dyn")
                        if out_dims is not None:
                            old_out_dims = dict(out_dims)
                            # Distribute the new total over the existing output keys
                            # in order, keeping all but the last at their declared
                            # size and absorbing the remainder into the last key.
                            keys = list(out_dims.keys())
                            head_sum = sum(out_dims[k] for k in keys[:-1])
                            out_dims[keys[-1]] = new_action_out_dim - head_sum
                            if dict(out_dims) != old_out_dims:
                                logger.info(
                                    f"Updated decoder_output_feature_dims['g1_dyn']: "
                                    f"{old_out_dims} → {dict(out_dims)}"
                                )

        for (
            module_name,
            state_dict_key,
        ) in pretrained_cfg.module_mapping.items():
            module = eval(module_name)
            filtered_state_dict = get_filtered_state_dict(state_dict, state_dict_key)

            # Drop ckpt keys that don't exist in the current model — happens when
            # `active_encoders` / `active_decoders` is a subset of what the ckpt
            # was saved with (e.g. only g1 + g1_dyn instantiated, but ckpt has
            # teleop/smpl encoders + g1_kin decoder). Without this filter,
            # `unexpected keys` log spams 30+ lines per LoRA load.
            cur_keys = set(module.state_dict().keys())
            dropped = [k for k in filtered_state_dict if k not in cur_keys]
            if dropped:
                # Group by top-level module path for compact log
                from collections import defaultdict
                by_mod = defaultdict(int)
                for k in dropped:
                    by_mod[".".join(k.split(".")[:2])] += 1
                logger.info(
                    f"Pretrained loading '{module_name}': skipping {len(dropped)} ckpt keys for inactive submodules: "
                    f"{dict(by_mod)}"
                )
                filtered_state_dict = {
                    k: v for k, v in filtered_state_dict.items() if k in cur_keys
                }

            missing, unexpected = module.load_state_dict(filtered_state_dict, strict=strict)
            if missing:
                logger.info(f"Pretrained loading '{module_name}': missing keys: {missing}")
            if unexpected:
                logger.info(f"Pretrained loading '{module_name}': unexpected keys: {unexpected}")

    # --- Oli LoRA wrapping (after pretrained weights are loaded) ---
    lora_cfg = config.algo.config.get("lora", None)
    if lora_cfg is not None:
        from gear_sonic.trl.modules.oli_lora_wrapper import OliLoRAWrapper
        from gear_sonic.trl.modules.lora_utils import count_parameters

        oli_proprio_dim = env.config["robot"]["algo_obs_dim_dict"]["actor_obs"]
        # Optional dict-typed fields are read with .get and forwarded only if set,
        # so old yamls without the new keys keep working unchanged.
        def _get(key, default=None):
            v = lora_cfg.get(key, default)
            # OmegaConf may return DictConfig/ListConfig; convert lazily to dict/list
            if hasattr(v, "items") and not isinstance(v, dict):
                v = {k: int(vv) for k, vv in v.items()}
            return v

        def _int_list(key):
            v = lora_cfg.get(key, None)
            if v is None:
                return None
            return [int(i) for i in v]

        wrapper = OliLoRAWrapper(
            pretrained_module=policy.actor_module,
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
            encoder_per_layer_ranks=_get("encoder_per_layer_ranks", None),
            decoder_per_layer_ranks=_get("decoder_per_layer_ranks", None),
            encoder_active_linears=_int_list("encoder_active_linears"),
            decoder_active_linears=_int_list("decoder_active_linears"),
            proprio_mode=lora_cfg.get("proprio_mode", "linear"),
            # `.get("...", default)` returns None (not default) when the key
            # exists with value null in yaml, so coalesce explicitly.
            proprio_actor_history_length=int(
                lora_cfg.get("proprio_actor_history_length")
                or config.get("actor_prop_history_length", 10)
            ),
            proprio_action_history_length=int(
                lora_cfg.get("proprio_action_history_length")
                or config.get("actor_actions_history_length", 10)
            ),
            unfreeze_backbone=lora_cfg.get("unfreeze_backbone", False),
            use_bf16=lora_cfg.get("use_bf16", False),
        )
        # Optional: load saved adapter-only state on top of fresh injection
        adapter_ckpt = lora_cfg.get("adapter_checkpoint", None)
        if adapter_ckpt:
            wrapper.load_adapters(adapter_ckpt)
        wrapper.print_summary()

        # Post-unfreeze selective re-freeze. When `unfreeze_backbone=true`
        # the wrapper makes EVERY pretrained param trainable, including the
        # decoder and FSQ quantizer. These flags let configs that want
        # "full fine-tune encoder only" re-freeze just those sub-modules
        # without touching encoder grad state.
        if lora_cfg.get("freeze_decoder_post_unfreeze", False):
            n_decoder = 0
            for p in wrapper.pretrained.decoders.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    n_decoder += p.numel()
            logger.info(f"[LoRA] freeze_decoder_post_unfreeze: re-froze {n_decoder:,} decoder params")
        if lora_cfg.get("freeze_quantizer_post_unfreeze", False):
            n_quant = 0
            quant = getattr(wrapper.pretrained, "quantizer", None)
            if quant is not None:
                for p in quant.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        n_quant += p.numel()
            logger.info(f"[LoRA] freeze_quantizer_post_unfreeze: re-froze {n_quant:,} quantizer params")

        policy.actor_module = wrapper

        # Warm-start from a previously-trained LoRA ckpt.
        # The trainer's pretrained_model loading path is designed for BASE
        # ckpts (bare UTM) and doesn't always fire decoder resize correctly
        # when given a wrapper-shaped LoRA ckpt. To work around that:
        #   1. Keep `pretrained_model.path` pointing at the BASE ckpt
        #      (so resize + the standard load path work)
        #   2. Set `lora.warmstart_lora_ckpt` to the LoRA ckpt path; we load
        #      its state_dict into the (already-wrapped) policy here.
        warmstart_path = lora_cfg.get("warmstart_lora_ckpt", None)
        if warmstart_path:
            import os as _os
            if not _os.path.isabs(warmstart_path):
                try:
                    from gear_sonic.utils.config_utils import _repo_root
                    warmstart_path = _os.path.join(_repo_root(), warmstart_path)
                except Exception:
                    pass
            logger.info(f"[LoRA] warmstart_lora_ckpt: loading {warmstart_path}")
            ws_ckpt = torch.load(warmstart_path, map_location="cpu", weights_only=False)
            ws_sd = (
                ws_ckpt.get("policy_state_dict")
                or ws_ckpt.get("model_state_dict")
                or ws_ckpt.get("state_dict")
                or ws_ckpt
            )
            # The LoRA ckpt's state_dict keys are at the policy level
            # ('actor_module.*', 'std', etc.). Drop any 'policy.' prefix
            # if present, then load with strict=False — missing/unexpected
            # are tolerated (e.g. our wrapper may not have lora_A/lora_B
            # if encoder_rank=0 and the ckpt does, and that's fine).
            #
            # ALSO drop any keys whose shape doesn't match the current
            # model — strict=False ignores name mismatches but raises on
            # shape mismatches. This is what catches a mismatched LoRA
            # rank between yaml and ckpt: rather than crashing, we skip
            # those tensors and warn so the user can fix the yaml.
            cleaned = {}
            for k, v in ws_sd.items():
                if k.startswith("policy."):
                    k = k[len("policy."):]
                cleaned[k] = v
            cur_sd = policy.state_dict()
            ws_skipped_shape = []
            for k in list(cleaned.keys()):
                if k in cur_sd and hasattr(cleaned[k], "shape") and hasattr(cur_sd[k], "shape"):
                    if cleaned[k].shape != cur_sd[k].shape:
                        ws_skipped_shape.append((k, tuple(cleaned[k].shape), tuple(cur_sd[k].shape)))
                        del cleaned[k]
            if ws_skipped_shape:
                logger.warning(
                    f"[LoRA] warmstart: dropping {len(ws_skipped_shape)} keys with size mismatch "
                    f"(possible LoRA rank / alpha mismatch with ckpt — first 3: {ws_skipped_shape[:3]})"
                )
            ws_miss, ws_unexp = policy.load_state_dict(cleaned, strict=False)
            logger.info(
                f"[LoRA] warmstart loaded: missing={len(ws_miss)} unexpected={len(ws_unexp)}"
            )
            if ws_miss[:3]:
                logger.info(f"[LoRA] warmstart missing (sample): {ws_miss[:3]}")
            if ws_unexp[:3]:
                logger.info(f"[LoRA] warmstart unexpected (sample): {ws_unexp[:3]}")
        # Update std to 31-DOF for Oli
        old_std = policy.std.data if hasattr(policy, "std") else None
        policy.num_actions = 31
        if old_std is not None:
            new_std = torch.zeros(31, device=old_std.device, dtype=old_std.dtype)
            from gear_sonic.trl.modules.dof_mapping import OLI_TO_G1_INDICES
            for g1_idx, oli_idx in enumerate(OLI_TO_G1_INDICES):
                new_std[oli_idx] = old_std[g1_idx]
            # Head joints: deterministic-effective. Pinned at the global
            # std_clamp_min (1e-3 in sonic_oli_lora.yaml) so PPO's in-place
            # clamp doesn't pull it back up. Combined with `learnable_head:
            # false` (head_default frozen at 0), head action ≈ 0 always.
            head_std = float(config.algo.config.get("std_clamp_min", 1e-3))
            from gear_sonic.trl.modules.dof_mapping import OLI_HEAD_INDICES
            for h_idx in OLI_HEAD_INDICES:
                new_std[h_idx] = head_std
            policy.std = torch.nn.Parameter(new_std)
        total, trainable = count_parameters(policy)
        logger.info(
            f"[LoRA] Policy after wrapping: {trainable:,} trainable / {total:,} total "
            f"({100.0 * trainable / total:.2f}%)"
        )

        # ================ Critic LoRA ================
        # Symmetric to actor: load value_state_dict from ckpt, freeze critic
        # backbone, inject LoRA into critic_module.module Sequential, and
        # wrap with a runtime slicer that converts Oli critic_obs (DOF=31,
        # bodies=42) into the G1 view (DOF=29, bodies=30) the ckpt expects.
        if value_model is not None and "value_state_dict" in ckpt:
            from gear_sonic.trl.modules.oli_lora_wrapper import CriticLoRAWrapper

            value_sd = ckpt["value_state_dict"]
            # Filter to critic_module.* keys; drop running_mean_std stats
            # (they're for the G1-shape obs, our env's obs has different dim).
            critic_sd = {k: v for k, v in value_sd.items() if k.startswith("critic_module.")}
            ckpt_first_w = critic_sd.get("critic_module.module.0.weight")
            if ckpt_first_w is None:
                logger.warning(
                    "[LoRA] value_state_dict has no critic_module.module.0.weight; "
                    "skipping critic LoRA."
                )
            else:
                g1_critic_obs_dim = ckpt_first_w.shape[1]

                # Shrink critic first Linear to ckpt shape (Oli critic_obs >
                # G1 critic_obs because Oli has 2 head DOFs × 3 segments × hist
                # extra dims). The runtime slicer (CriticLoRAWrapper) will feed
                # it a G1-shape view at forward time.
                critic_seq = value_model.critic_module.module
                cur_first = critic_seq[0]
                if (
                    cur_first.in_features != ckpt_first_w.shape[1]
                    or cur_first.out_features != ckpt_first_w.shape[0]
                ):
                    new_first = torch.nn.Linear(
                        ckpt_first_w.shape[1],
                        ckpt_first_w.shape[0],
                        bias=cur_first.bias is not None,
                    ).to(cur_first.weight.device, cur_first.weight.dtype)
                    critic_seq[0] = new_first
                    logger.info(
                        f"[LoRA] Resized critic_module.module[0] "
                        f"({cur_first.in_features}→{ckpt_first_w.shape[1]} in, "
                        f"{cur_first.out_features}→{ckpt_first_w.shape[0]} out) "
                        f"to match ckpt"
                    )

                # Move ckpt tensors onto the same device as the critic.
                critic_device = next(value_model.parameters()).device
                critic_sd = {k: v.to(critic_device) for k, v in critic_sd.items()}
                missing, unexpected = value_model.load_state_dict(critic_sd, strict=False)
                if unexpected:
                    logger.info(
                        f"[LoRA] Critic load: {len(unexpected)} unexpected ckpt keys"
                    )
                # `missing` is expected (running_mean_std + anything else outside
                # critic_module.*); only log the count, not the full list.
                logger.info(
                    f"[LoRA] Loaded {len(critic_sd)} critic weight keys; "
                    f"{len(missing)} target keys absent from ckpt (expected for "
                    f"running_mean_std)"
                )

                # Critic group obs (segment names + per-segment dims) is not
                # cached in env.config["obs"]["group_obs_*"] (line ~368 skips
                # "policy"/"critic"); the wrapper-side example_obs has
                # already-flattened ``critic_obs`` tensor too, so we read the
                # per-term breakdown straight from the IsaacLab observation
                # manager.
                _om = env.env.observation_manager
                critic_group_names = list(_om._group_obs_term_names["critic"])
                _term_dims_raw = _om._group_obs_term_dim["critic"]
                # Each entry is a tuple shape (e.g. (10, 31)); flat dim is the product.
                import numpy as _np
                critic_group_dims = [int(_np.prod(d)) for d in _term_dims_raw]
                logger.info(
                    f"[LoRA] Critic group obs segments: "
                    f"{list(zip(critic_group_names, critic_group_dims))} "
                    f"(total {sum(critic_group_dims)})"
                )

                critic_wrapper = CriticLoRAWrapper(
                    pretrained_critic_backbone=value_model.critic_module,
                    group_obs_names_critic=critic_group_names,
                    group_obs_dims_critic=critic_group_dims,
                    expected_g1_critic_obs_dim=g1_critic_obs_dim,
                    lora_rank=lora_cfg.get("rank", 8),
                    lora_alpha=lora_cfg.get("alpha", 16.0),
                    lora_dropout=lora_cfg.get("dropout", 0.0),
                    lora_skip_last=lora_cfg.get("skip_last", False),
                    variant=lora_cfg.get("variant", "lora"),
                    rslora=lora_cfg.get("rslora", False),
                    use_gate=lora_cfg.get("use_gate", False),
                    init_strategy=lora_cfg.get("init_strategy", "kaiming"),
                    linear_active_indices=_int_list("critic_active_linears"),
                    use_bf16=lora_cfg.get("use_bf16", False),
                ).to(critic_device)
                value_model.critic_module = critic_wrapper

                v_total, v_trainable = count_parameters(value_model)
                logger.info(
                    f"[LoRA] Critic after wrapping: {v_trainable:,} trainable / "
                    f"{v_total:,} total ({100.0 * v_trainable / v_total:.2f}%)"
                )

                # ---- LoRA-specific C: torch.compile on the INNER critic backbone ----
                # We compile critic_wrapper.pretrained (the MLP after the
                # Oli→G1 critic_obs slice), NOT the wrapper itself. Putting
                # `slice_oli_critic_obs_to_g1` inside the compile boundary
                # produces a symbolic shape that Dynamo can't connect to
                # the static weight's in_features (1645) → meta-tensor
                # RuntimeError. Compiling the inner module gives Inductor
                # a clean (B, 1645) input every call, fully static feature
                # dim, dynamic only across batch.
                compile_mode = lora_cfg.get("compile_critic", False)
                if compile_mode and compile_mode != "false":
                    try:
                        mode_str = compile_mode if isinstance(compile_mode, str) else "default"
                        critic_wrapper.pretrained = torch.compile(
                            critic_wrapper.pretrained,
                            mode=mode_str,
                            fullgraph=False,
                            dynamic=(mode_str != "reduce-overhead"),
                        )
                        logger.info(
                            f"[LoRA-C] torch.compile applied to critic_module.pretrained "
                            f"(inner backbone, mode={mode_str!r}, "
                            f"dynamic={mode_str != 'reduce-overhead'}) — "
                            f"slice op stays in eager"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[LoRA-C] torch.compile failed, falling back to eager: {e}"
                        )

    accelerator.wait_for_everyone()

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    ################
    # Training
    ################
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        disc_model=disc_model,
        value_model=value_model,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        resume=config.get("resume", False),
        local_seed=config.seed,
        log_dir=experiment_save_dir,
        accelerator=accelerator,
        _resolve=False,
    )

    # Training loop
    trainer.train()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":

    main()
