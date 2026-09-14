"""Oli 31-DOF robot configuration for SONIC.

References the HU_D04_01 robot asset (from wbt_finetune).
Joint order matches the IsaacLab convention derived from the USD.
"""

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils


# ======================== Oli joint / body names (IsaacLab order) ========================

# Joint names in IsaacLab runtime articulation order (verified 2026-04-28
# via robot.data.joint_names). DFS-interleaved across left/right + waist +
# (head_yaw, head_pitch are split apart, NOT contiguous).
OLI_JOINT_NAMES = [
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
    "head_yaw_joint",              # 11   ← head_yaw, Oli-only
    "left_shoulder_pitch_joint",   # 12
    "right_shoulder_pitch_joint",  # 13
    "left_ankle_pitch_joint",      # 14
    "right_ankle_pitch_joint",     # 15
    "head_pitch_joint",            # 16   ← head_pitch, Oli-only
    "left_shoulder_roll_joint",    # 17
    "right_shoulder_roll_joint",   # 18
    "left_ankle_roll_joint",       # 19
    "right_ankle_roll_joint",      # 20
    "left_shoulder_yaw_joint",     # 21
    "right_shoulder_yaw_joint",    # 22
    "left_elbow_joint",            # 23
    "right_elbow_joint",           # 24
    "left_wrist_yaw_joint",        # 25   ← Oli wrist sub-order: yaw / pitch / roll
    "right_wrist_yaw_joint",       # 26
    "left_wrist_pitch_joint",      # 27
    "right_wrist_pitch_joint",     # 28
    "left_wrist_roll_joint",       # 29
    "right_wrist_roll_joint",      # 30
]

# Body names of the 32 kinematic bodies (root + 31 link bodies) in DFS
# articulation order. The full Oli articulation has 42 bodies; the 10 extras
# are 6 foot contact sites and 4 hand sites — those are NOT in motion data
# and are excluded from this canonical list.
OLI_BODY_NAMES = [
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

# ======================== DOF reordering ========================
# Motion data for Oli LoRA training is in G1 29-DOF MuJoCo order (converted
# from Oli npy via npy_to_sonic_pkl.py). The motion_lib therefore needs G1's
# 29-DOF MuJoCo↔IsaacLab mapping, not an Oli-specific 31-DOF mapping.
#
# Downstream (commands.py) uses joint_utils.get_body_joint_indices to map
# the 29 motion-lib DOFs to their positions in Oli's 31-DOF joint tensor.

from gear_sonic.envs.manager_env.robots.g1 import (
    G1_ISAACLAB_JOINTS,
    G1_ISAACLAB_TO_MUJOCO_BODY,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_TO_ISAACLAB_BODY,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)

# Indices of head joints in the runtime DFS order (head is split apart at
# [11, 16], not contiguous like wbt_lite_mlp's grouped order would suggest).
OLI_HEAD_INDICES = [
    OLI_JOINT_NAMES.index("head_yaw_joint"),
    OLI_JOINT_NAMES.index("head_pitch_joint"),
]
assert OLI_HEAD_INDICES == [11, 16], OLI_HEAD_INDICES

# Indices of non-head joints (maps 1:1 to G1's 29 joints by name).
OLI_NON_HEAD_INDICES = [i for i in range(31) if i not in OLI_HEAD_INDICES]

# Motion-lib mappings reuse G1's 29-DOF / 30-body mappings so the motion data
# (G1 29-DOF MuJoCo order) is correctly reordered to G1 IsaacLab order for
# downstream body_joint_indices scatter.
OLI_ISAACLAB_TO_MUJOCO_MAPPING = {
    "num_dof": 29,  # motion-lib DOF count (G1)
    "isaaclab_joints": G1_ISAACLAB_JOINTS,  # G1 body names for motion-lib
    "isaaclab_to_mujoco_dof": G1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": G1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": G1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": G1_MUJOCO_TO_ISAACLAB_BODY,
}

# =================== Native 31-DOF motion-lib mapping ===================
# Used by the from-scratch sonic_oli_native task. Motion data is written
# directly in Oli's canonical IsaacLab order (OLI_JOINT_NAMES / OLI_BODY_NAMES),
# and forward kinematics is bypassed in favour of the npy's own body_states.
#
# Because converter output is already in Oli IsaacLab order, both DOF and
# body reorders are identity — motion_lib's post-load permutation is a no-op.

OLI_NATIVE_ISAACLAB_TO_MUJOCO_DOF = list(range(31))
OLI_NATIVE_MUJOCO_TO_ISAACLAB_DOF = list(range(31))
OLI_NATIVE_ISAACLAB_TO_MUJOCO_BODY = list(range(32))
OLI_NATIVE_MUJOCO_TO_ISAACLAB_BODY = list(range(32))

OLI_NATIVE_ISAACLAB_TO_MUJOCO_MAPPING = {
    "num_dof": 31,
    "isaaclab_joints": OLI_BODY_NAMES,  # canonical 32-body Oli order
    "isaaclab_to_mujoco_dof": OLI_NATIVE_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": OLI_NATIVE_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": OLI_NATIVE_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": OLI_NATIVE_MUJOCO_TO_ISAACLAB_BODY,
}

# ======================== Robot ArticulationCfg ========================

# USD path — resolved relative to the gear_sonic package root (parents[3] of
# this file: robots → manager_env → envs → gear_sonic).
OLI_USD_PATH = str(
    Path(__file__).resolve().parents[3] / "HU_D04_description" / "usd" / "HU_D04_01.usd"
)

OLI_31DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=OLI_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            fix_root_link=False,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            "left_hip_pitch_joint": -0.15,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": -0.05,
            "left_knee_joint": 0.28,
            "left_ankle_pitch_joint": -0.16,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.15,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.05,
            "right_knee_joint": 0.28,
            "right_ankle_pitch_joint": -0.16,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": -0.063,
            "left_shoulder_roll_joint": 0.206,
            "left_shoulder_yaw_joint": -0.297,
            "left_elbow_joint": -0.086,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": -0.063,
            "right_shoulder_roll_joint": -0.206,
            "right_shoulder_yaw_joint": 0.297,
            "right_elbow_joint": -0.086,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=139.41,
            damping=17.75,
            armature=0.14125,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=93.0,
            velocity_limit_sim=37.0,
            stiffness=93.65,
            damping=11.92,
            armature=0.1845504,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=93.0,
            velocity_limit_sim=37.0,
            stiffness=93.65,
            damping=11.92,
            armature=0.1845504,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=87.0,
            velocity_limit_sim=37.0,
            stiffness=87.51,
            damping=11.14,
            armature=0.0886706,
        ),
        "head_wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_pitch_joint",
                "head_yaw_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=15.0,
            velocity_limit_sim=22.0,
            stiffness=15.12,
            damping=1.93,
            armature=0.0153218,
        ),
    },
)

# ======================== Action scale ========================

# OLI_ACTION_SCALE = {}
# for a in OLI_31DOF_CFG.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     names = a.joint_names_expr
#     if not isinstance(e, dict):
#         e = dict.fromkeys(names, e)
#     if not isinstance(s, dict):
#         s = dict.fromkeys(names, s)
#     for n in names:
#         if n in e and n in s and s[n]:
#             OLI_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
import re as _re

_OLI_ACTION_SCALE_REGEX = {
    ".*_hip_pitch.*":  0.2511,
    ".*_hip_roll.*":   0.2511,
    ".*_hip_yaw.*":    0.2511,
    ".*_knee.*":       0.2511,
    ".*_ankle.*":      0.1121,
    "waist_.*":        0.1121,
    "head_.*":         0.3141,
    ".*_shoulder.*":   0.1200,
    ".*_elbow.*":      0.1200,
    ".*_wrist.*":      0.3141,
}

OLI_ACTION_SCALE = {}
for _name in OLI_JOINT_NAMES:
    for _pat, _val in _OLI_ACTION_SCALE_REGEX.items():
        if _re.fullmatch(_pat, _name):
            OLI_ACTION_SCALE[_name] = _val
            break
    else:
        raise ValueError(f"OLI_ACTION_SCALE: joint '{_name}' did not match any regex")
del _name, _pat, _val