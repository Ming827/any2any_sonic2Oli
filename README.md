# Any2Any — SONIC G1 → Oli (31-DoF)

This is the code for **[Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking](https://arxiv.org/abs/2605.23733)**. Project page: [any2any.top](https://any2any.top/)

Any2Any transfers an existing whole-body tracking (WBT) specialist to a new humanoid with only a small fraction of the data and compute needed to train from scratch: first kinematic alignment of the input/output spaces, then dynamics adaptation via lightweight parameter-efficient fine-tuning (PEFT) on the dynamics-sensitive modules.

This repository is the concrete instance of that recipe: it adapts the pretrained **SONIC** G1 (29-DoF) tracking policy to the **Oli** humanoid (31-DoF).

Built on the SONIC training stack in [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (the `gear_sonic/` subproject). Upstream docs: [nvlabs.github.io/GR00T-WholeBodyControl](https://nvlabs.github.io/GR00T-WholeBodyControl/) · SONIC paper: [arXiv:2511.07820](https://arxiv.org/abs/2511.07820)

Two training entry points share the same environment, reward, and PPO stack:

| Mode | Experiment config | Starts from |
| --- | --- | --- |
| **LoRA** (Any2Any) | `sonic_oli_lora` | Pretrained SONIC G1 checkpoint |
| **Native** (baseline) | `sonic_oli_native` | Random init |

---

## 1. Prerequisites

Matching the upstream [Installation (Training)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html) requirements:

| | |
| --- | --- |
| GPU | NVIDIA GPU with CUDA 12.x (L40 recommended) |
| OS | Ubuntu 22.04+ |
| Python | 3.11 (required by Isaac Lab; this package declares `>=3.10`) |
| Isaac Lab | 2.3+ |
| VRAM | ≥ 24 GB; 48 GB+ for the default `num_envs=2048` |

### Directory name matters

The code imports itself as an absolute package (`from gear_sonic.trl.utils.common import ...`), matching the upstream monorepo layout where the training stack lives in `gear_sonic/`. The repository directory **must be named `gear_sonic`**:

```
<workspace>/
└── gear_sonic/          ← this repository (name is required)
    ├── train_agent_trl.py
    ├── checkpoint/
    └── config/
```

If you cloned it under a different name, rename it:

```bash
mv gear_sonic_v2 gear_sonic
```

---

## 2. Environment setup

### 2.1 Install Isaac Lab

SONIC training uses Isaac Lab for physics simulation. Follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then verify:

```bash
python -c "import isaaclab; print(isaaclab.__version__)"
```

Everything below runs **inside the Isaac Lab Python environment**. If you manage it with conda:

```bash
conda create -n sonic python=3.11 -y
conda activate sonic
# ... install Isaac Sim + Isaac Lab into this env ...
```

### 2.2 Install this package

From the parent of the repository directory (same form as upstream's `pip install -e "gear_sonic/[training]"`):

```bash
cd <workspace>
pip install -e "gear_sonic/[training]"
```

This adds `hydra-core==1.3.2`, `trl==0.28.0`, `transformers`, `accelerate`, `tensorboard`, and `wandb` on top of Isaac Lab.

Verify:

```bash
python -c "import hydra, omegaconf; print(hydra.__version__, omegaconf.__version__)"
# expected: 1.3.2 2.3.0
```

> **Git LFS.** If you also clone the upstream monorepo, run `sudo apt install git-lfs && git lfs install && git lfs pull` — without it you get pointer stubs instead of meshes and ONNX models, which fail silently.

---

## 3. Download the pretrained SONIC checkpoint

LoRA mode fine-tunes on top of the released SONIC G1 policy, published on Hugging Face at [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC). The repo is public — no token required.

The PyTorch training checkpoint lives under `sonic_release/` on the Hub:

| Hub file | Purpose |
| --- | --- |
| `sonic_release/last.pt` | Training checkpoint (eval / fine-tuning) |
| `sonic_release/config.yaml` | The config it was produced with |
| `config.json` | Release manifest — include it when downloading selected files |

### 3.1 Download

```bash
pip install -U "huggingface_hub[cli]"

hf download nvidia/GEAR-SONIC \
    config.json \
    sonic_release/last.pt \
    sonic_release/config.yaml \
    --local-dir /tmp/gear_sonic_hf
```

Or via Python:

```python
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/GEAR-SONIC"
hf_hub_download(repo_id=REPO_ID, filename="config.json")
ckpt = hf_hub_download(repo_id=REPO_ID, filename="sonic_release/last.pt")
cfg = hf_hub_download(repo_id=REPO_ID, filename="sonic_release/config.yaml")
print(ckpt, cfg, sep="\n")
```

If you have the upstream monorepo checked out, its helper script does the same plus the SMPL dataset:

```bash
python download_from_hf.py --training            # checkpoint + SMPL data (~30 GB)
python download_from_hf.py --training --no-smpl  # checkpoint only
python download_from_hf.py --sample              # sample motion data (~4 MB)
```

### 3.2 Place it under `checkpoint/`

This repo resolves the pretrained weights at `checkpoint/last.pt`:

```bash
cd <workspace>/gear_sonic
mkdir -p checkpoint
cp /tmp/gear_sonic_hf/sonic_release/last.pt     checkpoint/
cp /tmp/gear_sonic_hf/sonic_release/config.yaml checkpoint/
```

Final layout:

```
gear_sonic/
└── checkpoint/
    ├── last.pt          # SONIC G1 policy weights (~450 MB)
    └── config.yaml      # the config the checkpoint was trained with
```

The path comes from [config/exp/manager/universal_token/all_modes/sonic_oli_lora.yaml](config/exp/manager/universal_token/all_modes/sonic_oli_lora.yaml):

```yaml
pretrained_model:
  path: ${repo_root:}/checkpoint/last.pt
  state_dict_key: policy_state_dict
  strict: false
  module_mapping:
    policy.actor_module: actor_module.
```

`${repo_root:}` is a custom OmegaConf resolver registered in [utils/config_utils.py](utils/config_utils.py), so it works regardless of your current working directory.

> Model weights are **not tracked in Git** (`checkpoint/` and `*.pt` are ignored). A 450 MB blob also exceeds GitHub's 100 MB per-file limit.

### 3.3 Verify

```bash
cd <workspace>/gear_sonic
python - <<'PY'
import torch
from pathlib import Path

ckpt = Path("checkpoint/last.pt")
assert ckpt.exists(), "checkpoint/last.pt not found"
assert Path("checkpoint/config.yaml").exists(), "checkpoint/config.yaml not found"

sd = torch.load(ckpt, map_location="cpu", weights_only=False)
print("top-level keys:", list(sd.keys()))
assert "policy_state_dict" in sd, "missing 'policy_state_dict' — wrong checkpoint format"

actor = [k for k in sd["policy_state_dict"] if k.startswith("policy.actor_module")]
print(f"actor tensors: {len(actor)}")
print(f"size: {ckpt.stat().st_size / 1e6:.0f} MB")
PY
```

`policy_state_dict` must contain `policy.actor_module.*` tensors — that prefix is what `module_mapping` rewrites to `actor_module.` at load time. If the assert fails you downloaded the ONNX deployment artifacts (`model_encoder.onnx` / `model_decoder.onnx`) instead of the training checkpoint.

---

## 4. Prepare motion data

Retargeted Oli motions live under `data/oli_motion/`:

```
data/oli_motion/
└── atom_motions_v6_selected_fix10n11_100hz/   # .npy motion clips
```

Point the motion library at the directory containing your clips. If you pass a parent directory holding several sub-folders, either keep the default `include_subdirs` whitelist or disable it with `include_subdirs=null`.

For upstream G1 data (Bones-SEED, 142K motions), see [Training Data](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training_data.html) and the conversion scripts in `data_process/`.

---

## 5. Single-GPU training

Single GPU is the **default** — [config/base.yaml](config/base.yaml) ships with `num_gpus: 1` and `multi_gpu: False`. No launcher (`accelerate launch` / `torchrun`) is needed; just run the script.

### 5.1 Smoke test first

Confirm the whole stack starts before committing to a long run (upstream's verification recipe):

```bash
cd <workspace>/gear_sonic
conda activate sonic

CUDA_VISIBLE_DEVICES=0 python train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_oli_native \
  num_envs=16 headless=True \
  ++algo.config.num_learning_iterations=5
```

After about a minute of initialization you should see reward and tracking-error metrics printing to the console.

### 5.2 LoRA fine-tuning (recommended)

```bash
CUDA_VISIBLE_DEVICES=0 python train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_oli_lora \
  num_envs=2048 headless=True \
  manager_env.commands.motion.motion_lib_cfg.motion_file="${PWD}/data/oli_motion/atom_motions_v6_selected_fix10n11_100hz"
```

Outputs go to `logs/rsl_rl/LoRA/<timestamp>/`.

### 5.3 Native training from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_oli_native \
  num_envs=2048 headless=True \
  manager_env.commands.motion.motion_lib_cfg.motion_file="${PWD}/data/oli_motion/atom_motions_v6_selected_fix10n11_100hz"
```

Outputs go to `logs/rsl_rl/Native/<timestamp>/`. No pretrained checkpoint is required.

> Upstream trains full SONIC on 64+ GPUs. On a single GPU expect a much longer wall clock — start from LoRA with a reduced `num_envs`.

### 5.4 LoRA vs. Native

Both configs are aligned on every shared hyperparameter (`num_envs`, `num_steps_per_env`, learning rates, adaptive-LR bounds, entropy schedule, rewards, terminations, observations, domain randomization). Only the following differ by design:

| Key | LoRA | Native |
| --- | --- | --- |
| `algo.config.pretrained_model` | present | absent |
| `algo.config.lora.*` | present | absent |
| `...actor.backbone.freeze_quantizer` | `true` | `false` |

The FSQ quantizer stays frozen under LoRA (it belongs to the pretrained tokenizer) and is trainable in Native mode.

---

## 6. Useful overrides

All Hydra overrides are `key=value` on the command line. See the upstream [Configuration Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/configuration.html).

```bash
# Smaller GPU — halve the parallel envs
num_envs=1024

# Watch the sim (disables headless; much slower)
headless=false

# Reproducibility
seed=42

# Pin the output root elsewhere
base_dir=/data/sonic_runs

# Enable Weights & Biases (the training script sets WANDB_MODE=disabled by default)
WANDB_MODE=online python train_agent_trl.py use_wandb=true ...

# Point at a different pretrained checkpoint (LoRA only)
algo.config.pretrained_model.path=/abs/path/to/last.pt

# Disable the motion sub-directory whitelist
manager_env.commands.motion.motion_lib_cfg.include_subdirs=null

# Disable observation noise (useful when debugging tracking quality)
++manager_env.observations.policy.enable_corruption=False
++manager_env.observations.tokenizer.enable_corruption=False
```

Print the fully composed config without launching the simulator:

```bash
python train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_oli_lora --cfg job
```

---

## Citation

```bibtex
@article{any2any,
    title={Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking},
    author={Anonymous},
    journal={arXiv preprint arXiv:2605.23733},
    year={2026}
}
```

Please also cite SONIC, the pretrained backbone this work transfers from:

```bibtex
@article{luo2025sonic,
    title={SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
    author={Luo, Zhengyi and Yuan, Ye and Wang, Tingwu and Li, Chenran and Chen, Sirui
            and Casta\~neda, Fernando and Cao, Zi-Ang and Li, Jiefeng and Minor, David
            and Ben, Qingwei and Da, Xingye and Ding, Runyu and Hogg, Cyrus and Song, Lina
            and Lim, Edy and Jeong, Eugene and He, Tairan and Xue, Haoru and Xiao, Wenli
            and Wang, Zi and Yuen, Simon and Kautz, Jan and Chang, Yan and Iqbal, Umar
            and Fan, Linxi and Zhu, Yuke},
    journal={arXiv preprint arXiv:2511.07820},
    year={2025}
}
```

## License

Dual licensing, following upstream:

- **Source code** — Apache License 2.0
- **Model weights** — downloaded SONIC checkpoints are covered by the [NVIDIA Open Model License](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/LICENSE)
