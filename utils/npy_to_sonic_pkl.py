#!/usr/bin/env python3
"""Convert wbt_lite_mlp .npy motion files to SONIC .pkl format.

wbt_lite_mlp npy format (Oli 31-DOF, IsaacLab order):
    dof_pos_vel:  (T, 31, 2)    joint positions and velocities
    body_states:  (T, 32, 13)   per-body: pos(3) + quat(4) + linvel(3) + angvel(3)
    dof_names:    list[str]      joint names
    body_names:   list[str]      body names

SONIC pkl format (G1 29-DOF, MuJoCo order):
    root_trans_offset:  (T, 3)       root position
    root_rot:           (T, 4)       root quaternion (x, y, z, w)
    pose_aa:            (T, 30, 3)   axis-angle per body (including root)
    dof:                (T, 29)      joint angles in MuJoCo order
    smpl_joints:        (T, 24, 3)   SMPL joints (zeros if unavailable)
    fps:                int          frame rate

Usage:
    python npy_to_sonic_pkl.py --input_dir /path/to/npy/files --output_dir /path/to/pkl/output
    python npy_to_sonic_pkl.py --input_file /path/to/motion.npy --output_dir /path/to/pkl/output
"""

import argparse
import glob
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


# G1 joint axes in MuJoCo order (from g1_29dof_with_hand.xml)
# Each entry: (axis_x, axis_y, axis_z)
G1_MUJOCO_JOINT_AXES = np.array([
    [0, 1, 0],  # left_hip_pitch
    [1, 0, 0],  # left_hip_roll
    [0, 0, 1],  # left_hip_yaw
    [0, 1, 0],  # left_knee
    [0, 1, 0],  # left_ankle_pitch
    [1, 0, 0],  # left_ankle_roll
    [0, 1, 0],  # right_hip_pitch
    [1, 0, 0],  # right_hip_roll
    [0, 0, 1],  # right_hip_yaw
    [0, 1, 0],  # right_knee
    [0, 1, 0],  # right_ankle_pitch
    [1, 0, 0],  # right_ankle_roll
    [0, 0, 1],  # waist_yaw
    [1, 0, 0],  # waist_roll
    [0, 1, 0],  # waist_pitch
    [0, 1, 0],  # left_shoulder_pitch
    [1, 0, 0],  # left_shoulder_roll
    [0, 0, 1],  # left_shoulder_yaw
    [0, 1, 0],  # left_elbow
    [1, 0, 0],  # left_wrist_roll
    [0, 1, 0],  # left_wrist_pitch
    [0, 0, 1],  # left_wrist_yaw
    [0, 1, 0],  # right_shoulder_pitch
    [1, 0, 0],  # right_shoulder_roll
    [0, 0, 1],  # right_shoulder_yaw
    [0, 1, 0],  # right_elbow
    [1, 0, 0],  # right_wrist_roll
    [0, 1, 0],  # right_wrist_pitch
    [0, 0, 1],  # right_wrist_yaw
], dtype=np.float32)

# Oli IsaacLab indices that map to G1 joints (skip head at 15, 16)
# Fallback when dof_names is not present in the npy.
OLI_TO_G1_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# G1 IsaacLab → MuJoCo reordering
G1_ISAACLAB_TO_MUJOCO = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]

# G1 MuJoCo joint order (names). dof_names from npy are resolved against this
# list to build a per-file dynamic mapping. Keeps the converter robust to
# differences in how wbt_lite_mlp orders its DOFs.
G1_MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _build_npy_to_g1_mujoco_indices(npy_dof_names: list) -> list:
    """Build a (29,) index list: for each G1 MuJoCo joint, return its index in the npy dof array.

    Drops head joints (not in G1) and reorders IsaacLab→MuJoCo in one step.
    Raises if any required G1 joint is missing from npy_dof_names.
    """
    indices = []
    for g1_name in G1_MUJOCO_JOINT_NAMES:
        if g1_name not in npy_dof_names:
            raise ValueError(
                f"G1 joint '{g1_name}' not found in npy dof_names. "
                f"Available: {npy_dof_names}"
            )
        indices.append(npy_dof_names.index(g1_name))
    return indices


# Body-name aliases when remapping from Oli npy body_names to G1 body_names.
# Oli has the same physical body but different name vs G1 in two places.
# (Mapping direction: G1 name → Oli npy name.)
G1_TO_OLI_NPY_BODY_ALIAS = {
    "pelvis": "base_link",
    "torso_link": "waist_pitch_link",
}


def _build_npy_to_g1_body_indices(npy_body_names: list, g1_body_names: list) -> list:
    """For each G1 body name, find the same-named (or aliased) entry in npy_body_names.

    Used to scatter Oli npy body_states (which carry Oli's actual recorded
    body positions) into G1-shape body slots so motion_lib's skip_fk path
    works for the LoRA pipeline.
    """
    indices = []
    for g1_name in g1_body_names:
        target = G1_TO_OLI_NPY_BODY_ALIAS.get(g1_name, g1_name)
        if target not in npy_body_names:
            raise ValueError(
                f"G1 body '{g1_name}' (looking for '{target}') not found in npy body_names. "
                f"Available: {npy_body_names}"
            )
        indices.append(npy_body_names.index(target))
    return indices


def convert_npy_to_dict(npy_path: str, target_fps: int = None, source_fps: int = None) -> dict:
    """Convert a wbt_lite_mlp .npy file to SONIC motion dict (in-memory).

    This is the core conversion logic shared by both the CLI converter
    and the on-the-fly motion lib loader. It does NOT resample frames —
    motion_lib's fk_batch/interpolation handles target_fps retiming based
    on the returned ``fps`` field, so we just pass the raw frames through.

    Args:
        npy_path: Path to input .npy file.
        target_fps: Deprecated / kept for CLI compatibility. Ignored for
            frame resampling; the returned dict's ``fps`` always reflects
            the source data's fps so motion_lib handles retiming itself.
        source_fps: Source FPS of the npy data. If None, read from the npy
            file's ``fps`` field (falling back to 30Hz if absent).

    Returns:
        Dict with keys: root_trans_offset, root_rot, pose_aa, dof,
        smpl_joints, fps — matching SONIC pkl format.
    """
    data = np.load(npy_path, allow_pickle=True).item()

    dof_pos_vel = data["dof_pos_vel"]   # (T, 31, 2)
    body_states = data["body_states"]   # (T, 32, 13)
    T = dof_pos_vel.shape[0]

    if source_fps is None:
        # Same convention as convert_npy_to_dict_oli: wbt_lite_mlp Oli npy
        # often omits/None-s the fps key. Trust target_fps when missing,
        # else use the file's value. The data we've seen is consistently
        # 50 Hz; hardcoding 30 here would make motion_lib's FK path
        # spuriously upsample by 50/30 and trip the adp_samp assertion.
        fps_val = data.get("fps")
        if fps_val is None:
            source_fps = int(target_fps) if target_fps is not None else 50
        else:
            source_fps = int(fps_val)

    # --- Extract root pose ---
    root_trans = body_states[:, 0, 0:3].astype(np.float32)

    # Quaternion: npy stores as (w, x, y, z), SONIC expects (x, y, z, w)
    root_quat_wxyz = body_states[:, 0, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)

    # --- Map DOF: Oli (npy order) → G1 29 (MuJoCo order) ---
    if "dof_names" in data:
        npy_dof_names = list(data["dof_names"])
        npy_to_g1_mujoco = _build_npy_to_g1_mujoco_indices(npy_dof_names)
        dof_g1_mujoco = dof_pos_vel[:, npy_to_g1_mujoco, 0].astype(np.float32)
    else:
        # Fallback: assume canonical Oli IsaacLab order
        dof_g1_isaaclab = dof_pos_vel[:, OLI_TO_G1_INDICES, 0]  # (T, 29)
        dof_g1_mujoco = dof_g1_isaaclab[:, G1_ISAACLAB_TO_MUJOCO].astype(np.float32)

    # --- Build pose_aa ---
    num_bodies = 30  # root + 29 joints for G1
    pose_aa = np.zeros((T, num_bodies, 3), dtype=np.float32)

    root_rot = Rotation.from_quat(root_quat_xyzw)
    pose_aa[:, 0, :] = root_rot.as_rotvec().astype(np.float32)

    for j in range(29):
        pose_aa[:, j + 1, :] = dof_g1_mujoco[:, j:j+1] * G1_MUJOCO_JOINT_AXES[j:j+1]

    smpl_joints = np.zeros((T, 24, 3), dtype=np.float32)

    # ---- Body data (skip_fk path) ----
    # Read body_states from npy and remap into G1 30-body order via name
    # lookup. Aliases pelvis↔base_link, torso_link↔waist_pitch_link bridge
    # the two name spaces. After remap, motion_lib's skip_fk branch fires
    # (avoids running G1 skeleton FK, which would have produced G1
    # articulation body positions instead of the actual Oli-recorded ones).
    from gear_sonic.envs.manager_env.robots.g1 import G1_ISAACLAB_JOINTS

    out = {
        "root_trans_offset": root_trans,
        "root_rot": root_quat_xyzw,
        "pose_aa": pose_aa,
        "dof": dof_g1_mujoco,
        "smpl_joints": smpl_joints,
        "fps": source_fps,
    }

    if "body_names" not in data:
        # Without body_names we cannot remap Oli body slots → G1 slots; the
        # motion lib would silently fall back to G1 FK, which yields G1's
        # *proximal* wrist position for the wrist_roll_link slot (G1 chain
        # is roll→pitch→yaw) — mismatched against Oli's terminal wrist body
        # at reward time. Better to fail loudly than to train with a hidden
        # constant offset on the end-effector tracking signal.
        raise ValueError(
            f"npy file {npy_path} is missing 'body_names'; the LoRA path "
            "requires body data to bypass G1 FK and use Oli's actually-recorded "
            "body positions. Re-export the npy with body_names+body_states or "
            "use the Oli-native path (convert_npy_to_dict_oli)."
        )
    npy_body_names = list(data["body_names"])
    npy_to_g1_body = _build_npy_to_g1_body_indices(
        npy_body_names, G1_ISAACLAB_JOINTS
    )
    body_states_g1 = body_states[:, npy_to_g1_body, :]   # (T, 30, 13)

    # Decompose: pos(3) + quat(4, xyzw) + linvel(3) + angvel(3).
    # Quaternion is xyzw (verified for wbt_lite_mlp Oli npy); motion_lib
    # consumes xyzw downstream.
    body_pos_w = body_states_g1[:, :, 0:3].astype(np.float32)
    body_quat_w = body_states_g1[:, :, 3:7].astype(np.float32)
    body_lin_vel_w = body_states_g1[:, :, 7:10].astype(np.float32)
    body_ang_vel_w = body_states_g1[:, :, 10:13].astype(np.float32)

    # dof_vel (G1 MuJoCo order, paired with dof_g1_mujoco above).
    if "dof_names" in data:
        dof_g1_mujoco_vel = dof_pos_vel[:, npy_to_g1_mujoco, 1].astype(np.float32)
    else:
        dof_g1_isaaclab_vel = dof_pos_vel[:, OLI_TO_G1_INDICES, 1]
        dof_g1_mujoco_vel = dof_g1_isaaclab_vel[:, G1_ISAACLAB_TO_MUJOCO].astype(np.float32)

    out.update({
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "dof_vel": dof_g1_mujoco_vel,
    })

    # If target_fps differs from source we cannot short-circuit; FK path
    # handles resampling internally. Drop body fields so caller falls
    # back to FK in that case.
    if target_fps is not None and int(target_fps) != int(source_fps):
        for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "dof_vel"):
            out.pop(k, None)

    return out


# ====================== Oli-native 31-DOF conversion ======================
# Used by the from-scratch sonic_oli_native task. Preserves Oli's canonical
# IsaacLab ordering for both DOFs (31) and bodies (32) throughout. Emits
# the npy's body_states directly so motion_lib can short-circuit forward
# kinematics — body poses and velocities come from IsaacLab simulation
# rather than approximate FK against a surrogate MJCF skeleton.

# Target order for the native Oli converter — matches the runtime IsaacLab
# articulation DFS order (verified 2026-04-28 via robot.data.joint_names /
# robot.data.body_names from a live env). The converter reorders the
# incoming npy to this order so motion-lib storage and the downstream env
# share a single canonical ordering.
OLI_ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",        # 0
    "right_hip_pitch_joint",       # 1
    "waist_yaw_joint",             # 2
    "left_hip_roll_joint",         # 3
    "right_hip_roll_joint",        # 4
    "waist_roll_joint",            # 5
    "left_hip_yaw_joint",          # 6
    "right_hip_yaw_joint",         # 7
    "waist_pitch_joint",           # 8
    "left_knee_joint",             # 9
    "right_knee_joint",            # 10
    "head_yaw_joint",              # 11
    "left_shoulder_pitch_joint",   # 12
    "right_shoulder_pitch_joint",  # 13
    "left_ankle_pitch_joint",      # 14
    "right_ankle_pitch_joint",     # 15
    "head_pitch_joint",            # 16
    "left_shoulder_roll_joint",    # 17
    "right_shoulder_roll_joint",   # 18
    "left_ankle_roll_joint",       # 19
    "right_ankle_roll_joint",      # 20
    "left_shoulder_yaw_joint",     # 21
    "right_shoulder_yaw_joint",    # 22
    "left_elbow_joint",            # 23
    "right_elbow_joint",           # 24
    "left_wrist_yaw_joint",        # 25
    "right_wrist_yaw_joint",       # 26
    "left_wrist_pitch_joint",      # 27
    "right_wrist_pitch_joint",     # 28
    "left_wrist_roll_joint",       # 29
    "right_wrist_roll_joint",      # 30
]

OLI_ISAACLAB_BODY_NAMES = [
    "base_link",                   # 0
    "left_hip_pitch_link",         # 1
    "right_hip_pitch_link",        # 2
    "waist_yaw_link",              # 3
    "left_hip_roll_link",          # 4
    "right_hip_roll_link",         # 5
    "waist_roll_link",             # 6
    "left_hip_yaw_link",           # 7
    "right_hip_yaw_link",          # 8
    "waist_pitch_link",            # 9
    "left_knee_link",              # 10
    "right_knee_link",             # 11
    "head_yaw_link",               # 12
    "left_shoulder_pitch_link",    # 13
    "right_shoulder_pitch_link",   # 14
    "left_ankle_pitch_link",       # 15
    "right_ankle_pitch_link",      # 16
    "head_pitch_link",             # 17
    "left_shoulder_roll_link",     # 18
    "right_shoulder_roll_link",    # 19
    "left_ankle_roll_link",        # 20
    "right_ankle_roll_link",       # 21
    "left_shoulder_yaw_link",      # 22
    "right_shoulder_yaw_link",     # 23
    "left_elbow_link",             # 24
    "right_elbow_link",            # 25
    "left_wrist_yaw_link",         # 26
    "right_wrist_yaw_link",        # 27
    "left_wrist_pitch_link",       # 28
    "right_wrist_pitch_link",      # 29
    "left_wrist_roll_link",        # 30
    "right_wrist_roll_link",       # 31
]


def _build_reorder(src_names: list, dst_names: list) -> list:
    """For each dst name, return its index in src. Raises on missing."""
    idx = []
    for n in dst_names:
        if n not in src_names:
            raise ValueError(f"Name '{n}' missing from source list. Available: {src_names}")
        idx.append(src_names.index(n))
    return idx


def convert_npy_to_dict_oli(
    npy_path: str, target_fps: int = None, source_fps: int = None
) -> dict:
    """Convert a wbt_lite_mlp Oli 31-DOF .npy to SONIC motion dict (native, FK-free).

    Reads the npy's precomputed body_states (from IsaacLab sim) and emits
    them directly so motion_lib can bypass forward kinematics.

    DOF + body are in Oli canonical IsaacLab order (OLI_ISAACLAB_JOINT_NAMES /
    OLI_ISAACLAB_BODY_NAMES). If ``target_fps`` differs from source fps the
    body/dof fields are NOT emitted and the caller falls back to the FK path.

    Returned dict (shapes; T' = frame count after possible resampling):
      - root_trans_offset:  (T', 3)
      - root_rot:           (T', 4)       xyzw
      - pose_aa:            (T', 32, 3)   axis-angle (root + zero joints; placeholder for FK fallback)
      - dof:                (T', 31)      joint angles
      - dof_vel:            (T', 31)      joint velocities
      - body_pos_w:         (T', 32, 3)   per-body world pos
      - body_quat_w:        (T', 32, 4)   xyzw
      - body_lin_vel_w:     (T', 32, 3)
      - body_ang_vel_w:     (T', 32, 3)
      - smpl_joints:        (T', 24, 3)   zeros
      - fps:                int
    """
    data = np.load(npy_path, allow_pickle=True).item()

    dof_pos_vel = data["dof_pos_vel"]   # (T, 31, 2)
    body_states = data["body_states"]   # (T, 32, 13)

    if source_fps is None:
        # Ignore npy's own `fps` field: the wbt_lite_mlp Oli pipeline often
        # omits it (or leaves it None), and the data we've seen is always
        # written at 50 Hz matching the target. Trust the caller's target_fps
        # and fall back to 50 Hz — this keeps skip_fk in play and avoids
        # silently mis-retiming motions.
        source_fps = int(target_fps) if target_fps is not None else 50

    # --- DOF reorder to Oli IsaacLab canonical (identity if npy is already there) ---
    if "dof_names" in data:
        npy_dof_names = list(data["dof_names"])
        npy_to_oli = _build_reorder(npy_dof_names, OLI_ISAACLAB_JOINT_NAMES)
        dof_pos = dof_pos_vel[:, npy_to_oli, 0].astype(np.float32)
        dof_vel = dof_pos_vel[:, npy_to_oli, 1].astype(np.float32)
    else:
        dof_pos = dof_pos_vel[:, :, 0].astype(np.float32)
        dof_vel = dof_pos_vel[:, :, 1].astype(np.float32)

    # --- Body reorder to Oli IsaacLab canonical ---
    if "body_names" in data:
        npy_body_names = list(data["body_names"])
        npy_to_oli_body = _build_reorder(npy_body_names, OLI_ISAACLAB_BODY_NAMES)
        body_states = body_states[:, npy_to_oli_body, :]

    # Decompose body_states: pos(3) + quat(4, xyzw) + linvel(3) + angvel(3).
    # The wbt_lite_mlp Oli npy stores quaternions in **xyzw** order (the same
    # convention IsaacLab's Articulation tensors use, and what wbt_lite_mlp's
    # play_motion.py implicitly assumes — its [6,3,4,5]→[3,4,5,6] reorder only
    # makes sense if the source is xyzw). motion_lib downstream consumes
    # body_quat_w as xyzw too (`rotations.xyzw_to_wxyz` is applied at
    # motion_lib_base.py:1615 before handing it to IsaacLab). So we pass the
    # quat through unchanged. Earlier this routine treated the slice as wxyz
    # and reordered to "xyzw" — that mis-mapped near-identity quaternions to
    # 180° about Z, which showed up as the robot reference being rotated 180°
    # relative to the actual robot in eval/training.
    body_pos_w = body_states[:, :, 0:3].astype(np.float32)
    body_quat_w = body_states[:, :, 3:7].astype(np.float32)
    body_lin_vel_w = body_states[:, :, 7:10].astype(np.float32)
    body_ang_vel_w = body_states[:, :, 10:13].astype(np.float32)

    # Root pose taken from body 0 (base_link).
    root_trans = body_pos_w[:, 0, :].copy()
    root_quat_xyzw = body_quat_w[:, 0, :].copy()

    T = dof_pos.shape[0]
    pose_aa = np.zeros((T, 32, 3), dtype=np.float32)  # placeholder (FK is skipped)
    smpl_joints = np.zeros((T, 24, 3), dtype=np.float32)

    out = {
        "root_trans_offset": root_trans,
        "root_rot": root_quat_xyzw,
        "pose_aa": pose_aa,
        "dof": dof_pos,
        "dof_vel": dof_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "smpl_joints": smpl_joints,
        "fps": source_fps,
    }

    # If the loader requested a different target_fps we cannot short-circuit
    # (FK path handles resampling internally). Drop the body fields so the
    # caller falls back to FK. Source-== target is the common case for
    # wbt_lite_mlp data at its native rate.
    if target_fps is not None and int(target_fps) != int(source_fps):
        for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "dof_vel"):
            out.pop(k, None)

    return out


def convert_npy_to_pkl(npy_path: str, output_dir: str, target_fps: int = 30) -> str:
    """Convert a single wbt_lite_mlp .npy file to SONIC .pkl format.

    Args:
        npy_path: Path to input .npy file.
        output_dir: Directory for output .pkl file.
        target_fps: Target FPS for the output. If source is 50Hz and target
            is 30Hz, frames are downsampled.

    Returns:
        Path to the created .pkl file.
    """
    motion_dict = convert_npy_to_dict(npy_path, target_fps=target_fps)
    motion_name = os.path.splitext(os.path.basename(npy_path))[0]
    motion_data = {motion_name: motion_dict}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{motion_name}.pkl")
    joblib.dump(motion_data, output_path)
    return output_path


def convert_directory(input_dir: str, output_dir: str, target_fps: int = 30):
    """Convert all .npy files in a directory tree to SONIC .pkl format."""
    npy_files = glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)
    print(f"Found {len(npy_files)} .npy files in {input_dir}")

    converted = 0
    errors = 0
    for npy_path in sorted(npy_files):
        # Preserve subdirectory structure
        rel_path = os.path.relpath(os.path.dirname(npy_path), input_dir)
        sub_output_dir = os.path.join(output_dir, rel_path)

        try:
            out_path = convert_npy_to_pkl(npy_path, sub_output_dir, target_fps)
            converted += 1
            if converted % 100 == 0:
                print(f"  Converted {converted}/{len(npy_files)}...")
        except Exception as e:
            print(f"  ERROR converting {npy_path}: {e}")
            errors += 1

    print(f"Done. Converted: {converted}, Errors: {errors}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert wbt_lite_mlp npy to SONIC pkl")
    parser.add_argument("--input_dir", type=str, help="Directory of .npy files to convert")
    parser.add_argument("--input_file", type=str, help="Single .npy file to convert")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pkl files")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    args = parser.parse_args()

    if args.input_file:
        out = convert_npy_to_pkl(args.input_file, args.output_dir, args.fps)
        print(f"Converted: {out}")
    elif args.input_dir:
        convert_directory(args.input_dir, args.output_dir, args.fps)
    else:
        parser.error("Either --input_dir or --input_file must be specified")
