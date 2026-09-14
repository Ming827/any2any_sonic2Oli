#!/usr/bin/env python3
"""Phase 1 BC: distill 0.1s-lookahead encoder → 0.02s-lookahead encoder.

E_old: frozen, takes obs at 10 frames × 0.1s = 1.0s lookahead (current LoRA-trained g1 enc).
E_new: full finetune, identical architecture, takes obs at 10 frames × 0.02s = 0.2s lookahead.

Loss: MSE on pre-FSQ latent. Same motion at same t, just different sampling stride.

  obs_sparse stride =  5 motion frames  (target_fps=50Hz → 0.1s)
  obs_dense  stride =  1 motion frame   (target_fps=50Hz → 0.02s)

Per-frame obs features for both encoders:
  joint_pos(29) | joint_vel(29) | motion_anchor_ori_6d(6) = 64 dims
  10 frames × 64 = 640 (matches base ckpt encoders.g1.module.0.weight shape (2048, 640))

No env, no PPO, no rollout — pure supervised distillation.

Usage:
  cd /home/limx/GR00T-WholeBodyControl/gear_sonic
  conda activate sonic
  PYTHONPATH=. python scripts/train_dense_encoder_bc.py \\
      --motion_dir data/motion_lib_bones_seed/robot_filtered \\
      --output_dir /tmp/dense_encoder_bc \\
      --max_iters 20000 --batch_size 256
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import os.path as osp
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

TARGET_FPS         = 50
NUM_FUTURE_FRAMES  = 10
FRAME_SKIP_OLD     = 5      # 5 motion-lib frames @ 50fps = 0.10s spacing
FRAME_SKIP_NEW     = 1      # 1 motion-lib frame  @ 50fps = 0.02s spacing
MAX_FUTURE_OFFSET  = (NUM_FUTURE_FRAMES - 1) * FRAME_SKIP_OLD   # 45 frames
G1_DOF             = 29
FEAT_PER_FRAME     = G1_DOF * 2 + 6   # 64


# ============================================================================
# Quaternion helpers (xyzw convention — what pkl stores; convert to wxyz only
# when feeding into torch ops that need wxyz)
# ============================================================================

def quat_xyzw_to_wxyz(q):
    return torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)

def quat_inv_wxyz(q):
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out

def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def matrix_from_quat_wxyz(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    M[..., 0, 0] = 1 - 2*(y*y + z*z); M[..., 0, 1] = 2*(x*y - z*w); M[..., 0, 2] = 2*(x*z + y*w)
    M[..., 1, 0] = 2*(x*y + z*w); M[..., 1, 1] = 1 - 2*(x*x + z*z); M[..., 1, 2] = 2*(y*z - x*w)
    M[..., 2, 0] = 2*(x*z - y*w); M[..., 2, 1] = 2*(y*z + x*w); M[..., 2, 2] = 1 - 2*(x*x + y*y)
    return M

def quat_to_6d_wxyz(q):
    return matrix_from_quat_wxyz(q)[..., :, :2].reshape(*q.shape[:-1], 6)


# ============================================================================
# Motion dataset (load pkl files, resample to TARGET_FPS, cache tensors)
# ============================================================================

def _slerp(q0, q1, t):
    """SLERP between (T,4) wxyz quats at scalar interp t."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1e-6)
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def _resample(arr_np, src_fps, target_fps, is_quat=False):
    """Resample (T_src, D) → (T_tgt, D) via linear interp (or SLERP for quats)."""
    if abs(src_fps - target_fps) < 1e-6:
        return torch.from_numpy(arr_np).float()
    T_src = arr_np.shape[0]
    T_tgt = max(1, int(round(T_src * target_fps / src_fps)))
    src_idx = np.linspace(0, T_src - 1, T_tgt)
    i0 = np.floor(src_idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, T_src - 1)
    frac = (src_idx - i0).astype(np.float32)
    a = torch.from_numpy(arr_np[i0]).float()
    b = torch.from_numpy(arr_np[i1]).float()
    t = torch.from_numpy(frac).unsqueeze(-1)
    if is_quat:
        return _slerp(a, b, t)
    return a + (b - a) * t


def load_motion_dataset(motion_dir: str) -> list[dict]:
    """Load all .pkl files in motion_dir, resample each to TARGET_FPS, build
    per-motion tensors (dof, dof_vel, root_quat_wxyz). Keep only motions long
    enough for a full sparse future window."""
    pkls = sorted(glob.glob(osp.join(motion_dir, "*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"no .pkl files under {motion_dir}")

    motions = []
    for path in pkls:
        d = joblib.load(path)
        # Outer key wraps a single motion; if there are multiple, iterate.
        records = d.items() if all(isinstance(v, dict) for v in d.values()) else [(osp.basename(path), d)]
        for name, inner in records:
            if "dof" not in inner or "root_rot" not in inner:
                continue
            src_fps = int(inner.get("fps", TARGET_FPS))
            dof = _resample(np.asarray(inner["dof"]), src_fps, TARGET_FPS)            # (T, 29)
            root_rot_xyzw = _resample(np.asarray(inner["root_rot"]), src_fps, TARGET_FPS, is_quat=False)
            root_rot_xyzw = root_rot_xyzw / root_rot_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw)                          # (T, 4) wxyz

            T = dof.shape[0]
            if T <= MAX_FUTURE_OFFSET + 1:
                continue   # too short for full sparse window

            # dof_vel via finite diff
            dof_vel = torch.zeros_like(dof)
            dof_vel[:-1] = (dof[1:] - dof[:-1]) * TARGET_FPS
            dof_vel[-1] = dof_vel[-2]

            motions.append({
                "name": name,
                "T": T,
                "dof": dof,                  # (T, 29)
                "dof_vel": dof_vel,          # (T, 29)
                "root_quat": root_rot_wxyz,  # (T, 4) wxyz
            })

    if not motions:
        raise RuntimeError(f"no usable motions (need length > {MAX_FUTURE_OFFSET})")
    total_frames = sum(m["T"] for m in motions)
    print(f"[data] loaded {len(motions)} motions, total {total_frames} frames @ {TARGET_FPS}Hz "
          f"(avg {total_frames/len(motions):.0f} frames/clip)")
    return motions


def sample_batch_indices(motions, batch_size: int, rng: np.random.Generator):
    """Sample (motion_idx, t_start) for a batch. t_start ∈ [0, T - MAX_FUTURE_OFFSET - 1]."""
    weights = np.array([m["T"] - MAX_FUTURE_OFFSET for m in motions], dtype=np.float64)
    weights /= weights.sum()
    motion_idx = rng.choice(len(motions), size=batch_size, p=weights)
    t_max = np.array([motions[i]["T"] - MAX_FUTURE_OFFSET - 1 for i in motion_idx])
    t_start = rng.integers(0, t_max + 1)
    return motion_idx, t_start


def build_obs_batch(motions, motion_idx, t_start, frame_skip, device):
    """Build a batch of encoder inputs.

    Returns:
        obs: (B, 10 * 64) = (B, 640) in MLP-ready flat form
    """
    B = len(motion_idx)
    offsets = torch.arange(NUM_FUTURE_FRAMES, dtype=torch.long) * frame_skip   # (10,)

    # Per-sample gather: t_start[b] + offsets → (B, 10)
    t_start_t = torch.as_tensor(t_start, dtype=torch.long)
    fut_idx = (t_start_t.unsqueeze(1) + offsets.unsqueeze(0))                   # (B, 10)

    jp_list, jv_list, q_list = [], [], []
    anchor_q_list = []
    for b in range(B):
        m = motions[motion_idx[b]]
        idx = fut_idx[b]
        jp_list.append(m["dof"][idx])                  # (10, 29)
        jv_list.append(m["dof_vel"][idx])              # (10, 29)
        q_list.append(m["root_quat"][idx])             # (10, 4) wxyz
        anchor_q_list.append(m["root_quat"][t_start[b]])  # (4,) — anchor at current t

    jp = torch.stack(jp_list).to(device)               # (B, 10, 29)
    jv = torch.stack(jv_list).to(device)               # (B, 10, 29)
    q_future = torch.stack(q_list).to(device)          # (B, 10, 4)
    q_anchor = torch.stack(anchor_q_list).to(device)   # (B, 4)

    # motion_anchor_ori_b_mf_nonflat = quat_inv(anchor) ⊗ q_future, then 6D
    q_anchor_inv = quat_inv_wxyz(q_anchor).unsqueeze(1).expand(-1, NUM_FUTURE_FRAMES, -1)
    q_rel = quat_mul_wxyz(q_anchor_inv, q_future)      # (B, 10, 4)
    ori6d = quat_to_6d_wxyz(q_rel)                     # (B, 10, 6)

    # Per-frame concat: [jp, jv, ori6d] → (B, 10, 64).
    # BaseModule.forward expects (..., temporal, feature) and flattens internally.
    obs = torch.cat([jp, jv, ori6d], dim=-1)            # (B, 10, 64)
    return obs


# ============================================================================
# Encoder build + weight load (reuse export_smpl_lora_onnx infrastructure)
# ============================================================================

def build_g1_encoder():
    """Construct UTM + OliLoRAWrapper (just to get a properly-shaped + LoRA-injected
    g1 encoder), return wrapper.pretrained.encoders["g1"]."""
    sys.path.insert(0, "/home/limx/Beyondmimic_Deploy_Oli_Sonic")
    from export_smpl_lora_onnx import build_lora_wrapper, load_weights
    wrapper = build_lora_wrapper()
    load_weights(wrapper)
    enc = wrapper.pretrained.encoders["g1"]
    enc.eval()
    return enc


# ============================================================================
# Main training loop
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", default="/home/limx/GR00T-WholeBodyControl/gear_sonic/data/motion_lib_bones_seed/robot_filtered")
    ap.add_argument("--output_dir", default="/tmp/dense_encoder_bc")
    ap.add_argument("--max_iters", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1) Build encoders
    print("[build] constructing g1 encoder (base + LoRA delta merged in)...")
    E_old = build_g1_encoder().to(args.device)
    for p in E_old.parameters():
        p.requires_grad = False
    E_new = copy.deepcopy(E_old).to(args.device)
    for p in E_new.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in E_new.parameters() if p.requires_grad)
    print(f"[build] E_new trainable params: {n_params:,}")

    # 2) Data
    print(f"[data] loading motions from {args.motion_dir}")
    motions = load_motion_dataset(args.motion_dir)

    # 3) Optimizer
    optimizer = torch.optim.AdamW(E_new.parameters(), lr=args.lr)

    # 4) Train
    loss_ema = None
    t0 = time.time()
    for it in range(args.max_iters):
        E_new.train()
        m_idx, t_start = sample_batch_indices(motions, args.batch_size, rng)

        obs_sparse = build_obs_batch(motions, m_idx, t_start, FRAME_SKIP_OLD, args.device)
        obs_dense  = build_obs_batch(motions, m_idx, t_start, FRAME_SKIP_NEW, args.device)

        with torch.no_grad():
            latent_old = E_old(obs_sparse)   # (B, ?, ?)

        latent_new = E_new(obs_dense)
        loss = F.mse_loss(latent_new, latent_old.detach())

        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(E_new.parameters(), max_norm=10.0)
        optimizer.step()

        loss_val = float(loss.item())
        loss_ema = loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * loss_val)

        if it % args.log_every == 0:
            iters_per_sec = (it + 1) / (time.time() - t0)
            print(f"[iter {it:6d}] loss={loss_val:.6f}  ema={loss_ema:.6f}  "
                  f"|grad|={gn.item():.3f}  {iters_per_sec:.1f} it/s")

        if (it + 1) % args.save_every == 0:
            ckpt_path = osp.join(args.output_dir, f"dense_encoder_step_{it+1}.pt")
            torch.save({
                "encoder_state_dict": E_new.state_dict(),
                "iter": it + 1,
                "loss_ema": loss_ema,
                "config": {
                    "target_fps": TARGET_FPS,
                    "num_future_frames": NUM_FUTURE_FRAMES,
                    "frame_skip_new": FRAME_SKIP_NEW,
                    "frame_skip_old_teacher": FRAME_SKIP_OLD,
                },
            }, ckpt_path)
            print(f"[save] {ckpt_path}")

    final_path = osp.join(args.output_dir, "dense_encoder_final.pt")
    torch.save({
        "encoder_state_dict": E_new.state_dict(),
        "iter": args.max_iters,
        "loss_ema": loss_ema,
    }, final_path)
    print(f"[done] final ckpt: {final_path}  loss_ema={loss_ema:.6f}")


if __name__ == "__main__":
    main()
