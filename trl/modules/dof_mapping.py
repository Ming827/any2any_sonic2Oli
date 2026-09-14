"""DOF mapping between G1 29-DOF and Oli 31-DOF robots.

Both lists below come from the **runtime IsaacLab articulation** —
``robot.data.joint_names`` after ``JointPositionActionCfg(joint_names=[".*"])``
resolves the regex against the loaded USD/URDF. Verified by printing
``robot.data.joint_names`` from a live env on 2026-04-28.

This is *neither* URDF declaration order *nor* wbt_lite_mlp's grouped
``JOINT_NAMES`` order — IsaacLab walks the articulation prim tree depth-first
and interleaves left/right sides + waist segments + (Oli-only) head joints.

Three non-trivial differences between G1 and Oli:

  1. **Wrist sub-order is reversed.** G1 wrist is ``(roll, pitch, yaw)``;
     Oli wrist is ``(yaw, pitch, roll)``. Name-keyed mapping handles this
     automatically.

  2. **Head joints (Oli-only) sit at indices 11 and 16**, not at the end and
     not contiguous — they're tucked in between knees/shoulders and between
     ankle_pitch/shoulder_roll respectively. Those slots are filled by
     ``DofBridge.head_default`` when expanding G1 → Oli.

  3. **Indices 0–10 are identity** (legs + waist + knees). After Oli's
     ``head_yaw`` at index 11, every subsequent G1 index sits one slot
     earlier than its Oli counterpart; after Oli's ``head_pitch`` at 16,
     it sits two slots earlier — until the wrist block, where the
     reverse-order shows up on top of that.
"""

import torch
import torch.nn as nn

# Oli 31-DOF joint names in IsaacLab runtime articulation order.
# Source: robot.data.joint_names from a live env (2026-04-28).
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
assert len(OLI_JOINT_NAMES) == 31

# G1 29-DOF joint names in IsaacLab runtime articulation order.
# Source: robot.data.joint_names from a live env (2026-04-28).
G1_ISAACLAB_DOF_NAMES = [
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
    "left_shoulder_pitch_joint",   # 11
    "right_shoulder_pitch_joint",  # 12
    "left_ankle_pitch_joint",      # 13
    "right_ankle_pitch_joint",     # 14
    "left_shoulder_roll_joint",    # 15
    "right_shoulder_roll_joint",   # 16
    "left_ankle_roll_joint",       # 17
    "right_ankle_roll_joint",      # 18
    "left_shoulder_yaw_joint",     # 19
    "right_shoulder_yaw_joint",    # 20
    "left_elbow_joint",            # 21
    "right_elbow_joint",           # 22
    "left_wrist_roll_joint",       # 23   ← G1 wrist sub-order: roll / pitch / yaw
    "right_wrist_roll_joint",      # 24
    "left_wrist_pitch_joint",      # 25
    "right_wrist_pitch_joint",     # 26
    "left_wrist_yaw_joint",        # 27
    "right_wrist_yaw_joint",       # 28
]
assert len(G1_ISAACLAB_DOF_NAMES) == 29
# Backward-compatible alias (some old code reads G1_JOINT_NAMES).
G1_JOINT_NAMES = G1_ISAACLAB_DOF_NAMES

# ---------- Name-keyed 29 ↔ 31 index table ----------
# For each G1 DOF position, the Oli DOF index that holds the same-named joint.
G1_TO_OLI_INDICES = [OLI_JOINT_NAMES.index(n) for n in G1_ISAACLAB_DOF_NAMES]
assert len(G1_TO_OLI_INDICES) == 29
# Concrete values (regression-protect; matches the ordering documented above):
_EXPECTED = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 13, 14, 15,
    17, 18, 19, 20, 21, 22, 23, 24,
    29, 30, 27, 28, 25, 26,
]
assert G1_TO_OLI_INDICES == _EXPECTED, (G1_TO_OLI_INDICES, _EXPECTED)

# Legacy name kept for callers that read it (e.g. train_agent_trl.py's std
# remap). Same semantics: ``new_oli[OLI_TO_G1_INDICES[g1_idx]] = old_g1[g1_idx]``.
OLI_TO_G1_INDICES = G1_TO_OLI_INDICES

# Oli slots NOT covered by G1 (head_yaw, head_pitch). Filled with
# ``DofBridge.head_default`` on the 29 → 31 path.
OLI_HEAD_INDICES = [
    OLI_JOINT_NAMES.index("head_yaw_joint"),
    OLI_JOINT_NAMES.index("head_pitch_joint"),
]
assert OLI_HEAD_INDICES == [11, 16], OLI_HEAD_INDICES

G1_DOF = 29
OLI_DOF = 31


# ============================ Hip-pitch axis decomposition ============================
# Oli's hip_pitch joint axis is tilted 25° in the Y-Z plane:
#   Left  hip_pitch axis: (0, +cos25°, -sin25°) = (0, 0.90631, -0.42262)
#   Right hip_pitch axis: (0, +cos25°, +sin25°) = (0, 0.90631, +0.42262)
# G1's hip_pitch is pure-Y. So an Oli hip_pitch joint angle θ_oli, in
# small-angle approximation, decomposes into two G1 joint values:
#   hp_g1   = cos25° · θ_oli            (the Y component)
#   yaw_g1 += ∓sin25° · θ_oli           (the Z component; sign mirrors L/R)
#
# When the LoRA backbone (trained on G1) sees Oli hip_pitch values, it
# misinterprets them. We transform on both sides:
#   - obs (Oli → G1 convention): before feeding the backbone
#   - action (G1 → Oli convention): after the backbone outputs
#
# Indices below are in the **G1 IsaacLab DOF order** (post-slice view):
import math as _math

HIP_PITCH_TILT_DEG = 25.0
_C25 = _math.cos(_math.radians(HIP_PITCH_TILT_DEG))   # 0.90631
_S25 = _math.sin(_math.radians(HIP_PITCH_TILT_DEG))   # 0.42262

# G1 IsaacLab DOF indices:
G1_LEFT_HIP_PITCH_IDX = 0   # G1_ISAACLAB_DOF_NAMES[0] = "left_hip_pitch_joint"
G1_RIGHT_HIP_PITCH_IDX = 1  # G1_ISAACLAB_DOF_NAMES[1] = "right_hip_pitch_joint"
G1_LEFT_HIP_YAW_IDX = 6     # G1_ISAACLAB_DOF_NAMES[6] = "left_hip_yaw_joint"
G1_RIGHT_HIP_YAW_IDX = 7    # G1_ISAACLAB_DOF_NAMES[7] = "right_hip_yaw_joint"
assert G1_ISAACLAB_DOF_NAMES[G1_LEFT_HIP_PITCH_IDX] == "left_hip_pitch_joint"
assert G1_ISAACLAB_DOF_NAMES[G1_RIGHT_HIP_PITCH_IDX] == "right_hip_pitch_joint"
assert G1_ISAACLAB_DOF_NAMES[G1_LEFT_HIP_YAW_IDX] == "left_hip_yaw_joint"
assert G1_ISAACLAB_DOF_NAMES[G1_RIGHT_HIP_YAW_IDX] == "right_hip_yaw_joint"


def hip_pitch_oli_to_g1_inplace(x: torch.Tensor) -> torch.Tensor:
    """Convert Oli hip_pitch/hip_yaw values (last dim, G1-DOF view) to G1 frame.

    Modifies the last-dim slice **in place** for hip_pitch (idx 0,1) and
    hip_yaw (idx 6,7). Operates on tensors of shape ``(..., G1_DOF)`` where
    the values at hip_pitch slots are Oli's tilted-axis joint angles.

    Linearization for small angles:
      hp_oli (around Oli's tilted axis) = hp_g1 (Y) + yaw (Z, signed)
      ⇒ hp_g1   ← cos25° · hp_oli
        yaw_g1 += ∓sin25° · hp_oli   (left: -, right: +)

    Note: applies to both joint_pos and joint_vel (linear transform).
    For action history, also linear since values are scaled identically L/R.
    """
    if x.shape[-1] != G1_DOF:
        raise ValueError(
            f"hip_pitch_oli_to_g1_inplace: expected last dim {G1_DOF}, got {x.shape[-1]}"
        )
    hp_l = x[..., G1_LEFT_HIP_PITCH_IDX].clone()
    hp_r = x[..., G1_RIGHT_HIP_PITCH_IDX].clone()
    x[..., G1_LEFT_HIP_PITCH_IDX] = _C25 * hp_l
    x[..., G1_RIGHT_HIP_PITCH_IDX] = _C25 * hp_r
    x[..., G1_LEFT_HIP_YAW_IDX] = x[..., G1_LEFT_HIP_YAW_IDX] - _S25 * hp_l
    x[..., G1_RIGHT_HIP_YAW_IDX] = x[..., G1_RIGHT_HIP_YAW_IDX] + _S25 * hp_r
    return x


def hip_pitch_g1_to_oli_inplace(x: torch.Tensor, *, oli_dof: bool = True) -> torch.Tensor:
    """Inverse of ``hip_pitch_oli_to_g1_inplace``: G1-frame → Oli-frame.

    Applied to the action output **after** ``DofBridge.g1_to_oli`` produces a
    31-DOF Oli action. The hip_pitch / hip_yaw channels in that output are
    still in G1 convention (because backbone was trained on G1). We
    decompose to Oli's tilted-axis convention so the action when applied
    to Oli's actuator produces the body motion the backbone intended.

    Inverse transform (assuming the input has G1's pure-Y hp + pure-Z yaw):
      hp_oli  = hp_g1 / cos25°
      yaw_oli = yaw_g1 + ∓ tan25° · hp_g1   (left: +, right: -;
                                              opposite sign of forward)

    When ``oli_dof=True``, indices into the 31-DOF Oli ordering are used:
      OLI: left_hip_pitch=0, right_hip_pitch=1, left_hip_yaw=6, right_hip_yaw=7
    The Oli IsaacLab order matches G1's for these four indices (legs come
    first), so the same idx constants apply.
    """
    if oli_dof and x.shape[-1] != OLI_DOF:
        raise ValueError(
            f"hip_pitch_g1_to_oli_inplace: expected last dim {OLI_DOF}, got {x.shape[-1]}"
        )
    if not oli_dof and x.shape[-1] != G1_DOF:
        raise ValueError(
            f"hip_pitch_g1_to_oli_inplace: expected last dim {G1_DOF}, got {x.shape[-1]}"
        )
    # Indices for the four channels are the same in both Oli and G1 IsaacLab
    # orderings (legs come first; head doesn't touch these positions).
    hp_l_g1 = x[..., G1_LEFT_HIP_PITCH_IDX].clone()
    hp_r_g1 = x[..., G1_RIGHT_HIP_PITCH_IDX].clone()
    inv_c = 1.0 / _C25
    tan25 = _S25 / _C25
    x[..., G1_LEFT_HIP_PITCH_IDX] = inv_c * hp_l_g1
    x[..., G1_RIGHT_HIP_PITCH_IDX] = inv_c * hp_r_g1
    x[..., G1_LEFT_HIP_YAW_IDX] = x[..., G1_LEFT_HIP_YAW_IDX] + tan25 * hp_l_g1
    x[..., G1_RIGHT_HIP_YAW_IDX] = x[..., G1_RIGHT_HIP_YAW_IDX] - tan25 * hp_r_g1
    return x


# ============================ Body name mapping ============================
# Body names in IsaacLab runtime articulation order (verified 2026-04-28 by
# printing robot.data.body_names from a live env).
#
# G1 has 30 bodies (root + 29 link bodies — same count as DOFs + 1 root).
# Oli has 42: 32 kinematic links (incl. head_yaw_link, head_pitch_link) plus
# 6 foot contact sites and 4 hand sites. Of those 42, only 30 have a G1
# counterpart by name (with two aliases: pelvis↔base_link, torso_link↔
# waist_pitch_link); the other 12 are Oli-specific.

G1_BODY_NAMES = [
    "pelvis",                       # 0
    "left_hip_pitch_link",          # 1
    "right_hip_pitch_link",         # 2
    "waist_yaw_link",               # 3
    "left_hip_roll_link",           # 4
    "right_hip_roll_link",          # 5
    "waist_roll_link",              # 6
    "left_hip_yaw_link",            # 7
    "right_hip_yaw_link",           # 8
    "torso_link",                   # 9   ← alias to Oli waist_pitch_link
    "left_knee_link",               # 10
    "right_knee_link",              # 11
    "left_shoulder_pitch_link",     # 12
    "right_shoulder_pitch_link",    # 13
    "left_ankle_pitch_link",        # 14
    "right_ankle_pitch_link",       # 15
    "left_shoulder_roll_link",      # 16
    "right_shoulder_roll_link",     # 17
    "left_ankle_roll_link",         # 18
    "right_ankle_roll_link",        # 19
    "left_shoulder_yaw_link",       # 20
    "right_shoulder_yaw_link",      # 21
    "left_elbow_link",              # 22
    "right_elbow_link",             # 23
    "left_wrist_roll_link",         # 24    ← G1 wrist body sub-order: roll/pitch/yaw
    "right_wrist_roll_link",        # 25
    "left_wrist_pitch_link",        # 26
    "right_wrist_pitch_link",       # 27
    "left_wrist_yaw_link",          # 28
    "right_wrist_yaw_link",         # 29
]
assert len(G1_BODY_NAMES) == 30

OLI_BODY_NAMES = [
    "base_link",                    # 0    ← alias from G1 pelvis
    "left_hip_pitch_link",          # 1
    "right_hip_pitch_link",         # 2
    "waist_yaw_link",               # 3
    "left_hip_roll_link",           # 4
    "right_hip_roll_link",          # 5
    "waist_roll_link",              # 6
    "left_hip_yaw_link",            # 7
    "right_hip_yaw_link",           # 8
    "waist_pitch_link",             # 9    ← alias from G1 torso_link
    "left_knee_link",               # 10
    "right_knee_link",              # 11
    "head_yaw_link",                # 12   ← Oli-only
    "left_shoulder_pitch_link",     # 13
    "right_shoulder_pitch_link",    # 14
    "left_ankle_pitch_link",        # 15
    "right_ankle_pitch_link",       # 16
    "head_pitch_link",              # 17   ← Oli-only
    "left_shoulder_roll_link",      # 18
    "right_shoulder_roll_link",     # 19
    "left_ankle_roll_link",         # 20
    "right_ankle_roll_link",        # 21
    "left_shoulder_yaw_link",       # 22
    "right_shoulder_yaw_link",      # 23
    "contact_foot_center_L",        # 24   ← Oli-only (foot contact site)
    "contact_foot_heel_L",          # 25
    "contact_foot_tip_L",           # 26
    "contact_foot_center_R",        # 27
    "contact_foot_heel_R",          # 28
    "contact_foot_tip_R",           # 29
    "left_elbow_link",              # 30
    "right_elbow_link",             # 31
    "left_wrist_yaw_link",          # 32   ← Oli wrist body sub-order: yaw/pitch/roll
    "right_wrist_yaw_link",         # 33
    "left_wrist_pitch_link",        # 34
    "right_wrist_pitch_link",       # 35
    "left_wrist_roll_link",         # 36
    "right_wrist_roll_link",        # 37
    "left_hand_contact",            # 38   ← Oli-only (hand site)
    "left_hand_manip",              # 39
    "right_hand_contact",           # 40
    "right_hand_manip",             # 41
]
assert len(OLI_BODY_NAMES) == 42

# G1 body name → Oli body name aliases (for the two cases where the same
# physical body is named differently between robots).
G1_TO_OLI_BODY_NAME_ALIAS = {
    "pelvis": "base_link",
    "torso_link": "waist_pitch_link",
}


def _resolve_oli_body_idx(g1_name: str) -> int:
    target = G1_TO_OLI_BODY_NAME_ALIAS.get(g1_name, g1_name)
    return OLI_BODY_NAMES.index(target)


# For each G1 body position, the Oli body index (with name aliases applied).
G1_BODY_TO_OLI_BODY_INDICES = [_resolve_oli_body_idx(n) for n in G1_BODY_NAMES]
assert len(G1_BODY_TO_OLI_BODY_INDICES) == 30
_EXPECTED_BODY = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    13, 14, 15, 16,
    18, 19, 20, 21, 22, 23,
    30, 31,
    36, 37, 34, 35, 32, 33,
]
assert G1_BODY_TO_OLI_BODY_INDICES == _EXPECTED_BODY, (G1_BODY_TO_OLI_BODY_INDICES, _EXPECTED_BODY)

# Oli body slots NOT covered by G1 (head links, foot contacts, hand sites).
OLI_EXTRA_BODY_INDICES = [
    i for i in range(len(OLI_BODY_NAMES)) if i not in set(G1_BODY_TO_OLI_BODY_INDICES)
]
assert OLI_EXTRA_BODY_INDICES == [12, 17, 24, 25, 26, 27, 28, 29, 38, 39, 40, 41]

G1_NUM_BODIES = 30
OLI_NUM_BODIES = 42


def build_oli_to_g1_matrix() -> torch.Tensor:
    """Build (29, 31) selection matrix mapping Oli 31-DOF → G1 29-DOF.

    Each G1 row picks the Oli column whose joint name matches G1[row].
    Head columns (Oli 11, 16) are not selected — they have no G1 counterpart.

    Usage: ``g1_dof = oli_dof @ M.T``  or  ``g1_dof = M @ oli_dof`` (per-sample)
    """
    M = torch.zeros(G1_DOF, OLI_DOF)
    for g1_idx, oli_idx in enumerate(G1_TO_OLI_INDICES):
        M[g1_idx, oli_idx] = 1.0
    return M


def build_g1_to_oli_matrix() -> torch.Tensor:
    """Build (31, 29) selection matrix mapping G1 29-DOF → Oli 31-DOF.

    Each non-head Oli row picks its G1 column by joint name. Head rows
    (11, 16) stay zero and get filled by ``DofBridge.head_default``.

    Usage: ``oli_dof = g1_dof @ M.T``  or  ``oli_dof = M @ g1_dof`` (per-sample)
    """
    M = torch.zeros(OLI_DOF, G1_DOF)
    for g1_idx, oli_idx in enumerate(G1_TO_OLI_INDICES):
        M[oli_idx, g1_idx] = 1.0
    return M


# ---------- Convenience: name-keyed slicers (no torch dep, plain Python) ----------


def oli_to_g1_indices_for(joint_names: list[str]) -> list[int]:
    """Given a custom list of joint names (must all be in OLI_JOINT_NAMES),
    return the Oli indices that select those joints.

    Example
    -------
    >>> oli_to_g1_indices_for(G1_ISAACLAB_DOF_NAMES) == G1_TO_OLI_INDICES
    True
    """
    return [OLI_JOINT_NAMES.index(n) for n in joint_names]


def slice_oli_to_g1(x: torch.Tensor) -> torch.Tensor:
    """Drop Oli's head joints and reorder wrists into G1 IsaacLab DOF order.

    Last dim must be 31. Output shape: ``(..., 29)``. Pure advanced index —
    no allocation of a Linear / matrix multiply. Differentiable.
    """
    if x.shape[-1] != OLI_DOF:
        raise ValueError(f"Expected last dim {OLI_DOF}, got {x.shape[-1]}")
    return x[..., G1_TO_OLI_INDICES]


def slice_oli_actor_obs_to_g1(
    actor_obs: torch.Tensor,
    prop_hist: int = 10,
    action_hist: int = 10,
    gravity_dim: int = 3,
    ang_vel_dim: int = 3,
    apply_hip_pitch_decomp: bool = False,
) -> torch.Tensor:
    """Convert Oli's flat ``actor_obs`` (31-DOF) to G1-compatible (29-DOF) by
    dropping head joints and reordering wrists in each per-DOF segment.

    The actual flat layout is determined by ``PolicyCfg`` dataclass attribute
    declaration order in ``mdp/observations.py`` (NOT the yaml ``defaults``
    list order). For ``local_dir_hist.yaml`` the runtime layout, verified
    via DUMP_ORDERS=1, is::

        [base_ang_vel × prop_hist  ]   (ang_vel_dim × prop_hist)
        [joint_pos    × prop_hist  ]   (OLI_DOF     × prop_hist)
        [joint_vel    × prop_hist  ]   (OLI_DOF     × prop_hist)
        [actions      × action_hist]   (OLI_DOF     × action_hist)
        [gravity_dir  × prop_hist  ]   (gravity_dim × prop_hist)

    Each per-DOF segment is reshaped ``(..., hist, OLI_DOF)``, name-sliced via
    ``G1_TO_OLI_INDICES`` to ``(..., hist, G1_DOF)``, and flattened back.
    Non-DOF segments (ang_vel, gravity) pass through unchanged.

    Output last dim:
        ang_vel_dim * prop_hist + gravity_dim * prop_hist
        + G1_DOF * (2 * prop_hist + action_hist)
    For the canonical ``hist=10`` layout this is 930 — matching the G1 release ckpt.

    Args:
        actor_obs: ``(..., 990)`` for the canonical Oli layout.
        prop_hist: history length used for the per-step proprio terms.
        action_hist: history length used for the action term.
        gravity_dim, ang_vel_dim: per-step dims of the non-DOF segments.

    Returns:
        ``(..., 930)`` G1-compatible tensor in the same segment order as the
        input (so the G1 backbone, also trained on the env's natural layout,
        receives a layout it recognizes).
    """
    ang_sz = ang_vel_dim * prop_hist
    jp_sz = OLI_DOF * prop_hist
    jv_sz = OLI_DOF * prop_hist
    act_sz = OLI_DOF * action_hist
    grav_sz = gravity_dim * prop_hist
    expected = ang_sz + jp_sz + jv_sz + act_sz + grav_sz
    if actor_obs.shape[-1] != expected:
        raise ValueError(
            f"slice_oli_actor_obs_to_g1: expected last dim {expected} "
            f"(prop_hist={prop_hist}, action_hist={action_hist}), got {actor_obs.shape[-1]}"
        )

    # Split in the actual env layout order: ang_vel, joint_pos, joint_vel,
    # actions, gravity_dir.
    ang, jp, jv, act, grav = torch.split(
        actor_obs, [ang_sz, jp_sz, jv_sz, act_sz, grav_sz], dim=-1
    )
    leading_shape = jp.shape[:-1]
    jp_g1 = jp.reshape(*leading_shape, prop_hist, OLI_DOF)[..., G1_TO_OLI_INDICES]
    jv_g1 = jv.reshape(*leading_shape, prop_hist, OLI_DOF)[..., G1_TO_OLI_INDICES]
    act_g1 = act.reshape(*leading_shape, action_hist, OLI_DOF)[..., G1_TO_OLI_INDICES]
    # Optional: apply hip-pitch axis decomposition on each per-step DOF view
    # so the values look "G1-frame" to the backbone (Oli's hip_pitch axis is
    # tilted 25°, G1's is pure Y).
    if apply_hip_pitch_decomp:
        jp_g1 = jp_g1.contiguous()
        jv_g1 = jv_g1.contiguous()
        act_g1 = act_g1.contiguous()
        hip_pitch_oli_to_g1_inplace(jp_g1)
        hip_pitch_oli_to_g1_inplace(jv_g1)
        hip_pitch_oli_to_g1_inplace(act_g1)
    return torch.cat(
        [
            ang,
            jp_g1.reshape(*leading_shape, prop_hist * G1_DOF),
            jv_g1.reshape(*leading_shape, prop_hist * G1_DOF),
            act_g1.reshape(*leading_shape, action_hist * G1_DOF),
            grav,
        ],
        dim=-1,
    )


def slice_command_multi_future_to_g1(
    nonflat: torch.Tensor,
    num_future_frames: int,
    oli_dof: int = OLI_DOF,
    apply_hip_pitch_decomp: bool = False,
) -> torch.Tensor:
    """Slice motion ``command_multi_future_nonflat`` from Oli 31-DOF to G1 29-DOF.

    The env feeds tokenizer obs in Oli native shape because env-level reset
    needs all 31 joints. The G1 backbone only saw 29 DOFs in pretraining, so
    this helper drops the 2 head DOFs from each future-frame's joint_pos and
    joint_vel before the data hits the encoder's first Linear.

    Underlying ``command_multi_future`` is built by
    ``cat([joint_pos_multi_future, joint_vel_multi_future], dim=1)`` —
    flat layout ``[pos block of N*D | vel block of N*D]`` where each block is
    t-major. The "nonflat" reshape is just ``(..., N_fut, 2 * D)`` over the
    same flat tensor, so we collapse the trailing two dims, expose the
    ``(2, N, D)`` structure, slice the DOF axis, and reshape back.

    Args:
        nonflat: ``(..., N_fut, 2 * oli_dof)`` Oli view.
        num_future_frames: ``N_fut`` (must match the trailing-second dim).
        oli_dof: Oli DOF count (default 31).

    Returns:
        ``(..., N_fut, 2 * G1_DOF)`` G1-compatible view.
    """
    expected_last = 2 * oli_dof
    if nonflat.shape[-1] != expected_last or nonflat.shape[-2] != num_future_frames:
        raise ValueError(
            f"slice_command_multi_future_to_g1: expected (..., {num_future_frames}, "
            f"{expected_last}), got {tuple(nonflat.shape[-2:])}"
        )
    leading = nonflat.shape[:-2]
    flat = nonflat.reshape(*leading, 2 * num_future_frames * oli_dof)
    exposed = flat.reshape(*leading, 2, num_future_frames, oli_dof)
    sliced = exposed[..., G1_TO_OLI_INDICES]  # (..., 2, N, G1_DOF)
    if apply_hip_pitch_decomp:
        sliced = sliced.contiguous()
        # ``sliced`` has shape (..., 2, N_fut, G1_DOF). hip_pitch helper
        # operates on the last dim (G1_DOF) — applies to both pos block (dim
        # -3 == 0) and vel block (dim -3 == 1) since the linear transform is
        # the same for position and velocity.
        hip_pitch_oli_to_g1_inplace(sliced)
    g1_dof = len(G1_TO_OLI_INDICES)
    out_flat = sliced.reshape(*leading, 2 * num_future_frames * g1_dof)
    return out_flat.reshape(*leading, num_future_frames, 2 * g1_dof)


class DofBridge(nn.Module):
    """Differentiable DOF bridge between G1 29-DOF and Oli 31-DOF.

    Provides both directions:
      - ``oli_to_g1()``: ``(*, 31) → (*, 29)``  drops head joints, reorders wrists
      - ``g1_to_oli()``: ``(*, 29) → (*, 31)``  inserts head_default at [11, 16]

    The ``head_default`` parameter allows learning a default head pose during training.
    """

    def __init__(self, learnable_head: bool = True):
        super().__init__()
        self.register_buffer("oli_to_g1_mat", build_oli_to_g1_matrix())  # (29, 31)
        self.register_buffer("g1_to_oli_mat", build_g1_to_oli_matrix())  # (31, 29)

        if learnable_head:
            # Learnable default for head joints (yaw, pitch) when expanding 29→31
            self.head_default = nn.Parameter(torch.zeros(len(OLI_HEAD_INDICES)))
        else:
            self.register_buffer("head_default", torch.zeros(len(OLI_HEAD_INDICES)))

        # One-hot scatter matrix for head_default → 31-DOF additive vector.
        # Shape ``(len(OLI_HEAD_INDICES), 31)`` — each row picks one Oli head
        # slot. Used by ``g1_to_oli`` instead of fancy-indexed in-place add,
        # because ``out[..., idx] = out[..., idx] + head_default`` doesn't
        # trace cleanly to ONNX (produces empty-input Add nodes that break
        # graph load). The matmul form is mathematically identical and
        # exports without issue.
        head_one_hot = torch.zeros(len(OLI_HEAD_INDICES), 31)
        for i, idx in enumerate(OLI_HEAD_INDICES):
            head_one_hot[i, idx] = 1.0
        self.register_buffer("_head_one_hot", head_one_hot)

    def oli_to_g1(self, x: torch.Tensor) -> torch.Tensor:
        """Map Oli 31-DOF → G1 29-DOF. Last dim must be 31."""
        return x @ self.oli_to_g1_mat.to(x.dtype).T

    def g1_to_oli(self, x: torch.Tensor) -> torch.Tensor:
        """Map G1 29-DOF → Oli 31-DOF. Last dim must be 29.

        Head joints are filled with ``self.head_default`` (learnable or zero).
        Casts the bridge buffers and head_default to ``x.dtype`` so this works
        under bfloat16 / float16 autocast.

        Uses ``head_default @ _head_one_hot`` (a fixed 2×31 scatter matrix)
        instead of ``out[..., OLI_HEAD_INDICES] += head_default`` — the latter
        breaks ONNX export with empty-input Add nodes.
        """
        out = x @ self.g1_to_oli_mat.to(x.dtype).T
        head_addition = self.head_default.to(out.dtype) @ self._head_one_hot.to(out.dtype)
        return out + head_addition

    def oli_to_g1_batch_joints(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        """Convenience: map both joint_pos and joint_vel from Oli→G1."""
        return self.oli_to_g1(joint_pos), self.oli_to_g1(joint_vel)


# ============================ Critic obs slicer ============================


# Segment-name → slice rule. Names match those in
# ``env.config["obs"]["group_obs_names"]["critic"]``.
#
# Note: ``body_pos`` / ``body_ori`` are NOT in any slice set — both G1 and
# Oli envs index them by a fixed list of named bodies from the yaml
# ``manager_env.commands.motion.body_names`` (14 reward-point bodies in the
# current LoRA config), with name aliases (pelvis↔base_link, torso↔waist_pitch)
# resolving the two robots to the same set. So body segments are identical-size
# between G1 and Oli — pass through as identity.
_DOF_SEGMENT_NAMES = frozenset({"joint_pos", "joint_vel", "actions"})
_BODY_SEGMENT_NAMES = frozenset()  # intentionally empty, see comment above
# Segments whose flat dim is ``2 * N_fut * num_dof`` ([pos block | vel block]),
# with N_fut inferred from ``seg_dim // (2 * oli_dof)``. These need the same
# DOF-axis slice as ``joint_pos`` etc. but reshape to ``(2, N_fut, oli_dof)``
# rather than ``(N_fut, oli_dof)``.
_COMMAND_DOF_SEGMENT_NAMES = frozenset({"command_multi_future"})


def compute_g1_critic_obs_dim(
    group_obs_names: list[str],
    group_obs_dims: list[int],
    *,
    oli_dof: int = OLI_DOF,
    g1_dof: int = G1_DOF,
    oli_num_bodies: int = OLI_NUM_BODIES,
    g1_num_bodies: int = G1_NUM_BODIES,
) -> int:
    """Compute the post-slice total dim of critic_obs (G1 view).

    Per-segment rule:
      - ``joint_pos`` / ``joint_vel`` / ``actions``: per-DOF, 31→29
      - ``body_pos`` / ``body_ori``: per-body, 42→30
      - else: identity

    Raises ``ValueError`` if a DOF/body segment's reported dim is not
    divisible by ``oli_dof`` / ``oli_num_bodies`` (means our segment
    classification is wrong for that name in the current env).
    """
    if len(group_obs_names) != len(group_obs_dims):
        raise ValueError(
            f"length mismatch: names={len(group_obs_names)}, dims={len(group_obs_dims)}"
        )
    total = 0
    for name, seg_dim in zip(group_obs_names, group_obs_dims):
        if name in _DOF_SEGMENT_NAMES:
            if seg_dim % oli_dof != 0:
                raise ValueError(
                    f"DOF segment '{name}' dim {seg_dim} not divisible by oli_dof={oli_dof}"
                )
            total += (seg_dim // oli_dof) * g1_dof
        elif name in _COMMAND_DOF_SEGMENT_NAMES:
            # Layout: 2 * N_fut * oli_dof (pos block + vel block, t-major).
            two_oli_dof = 2 * oli_dof
            if seg_dim % two_oli_dof != 0:
                raise ValueError(
                    f"Command segment '{name}' dim {seg_dim} not divisible by "
                    f"2*oli_dof={two_oli_dof}"
                )
            n_fut = seg_dim // two_oli_dof
            total += 2 * n_fut * g1_dof
        elif name in _BODY_SEGMENT_NAMES:
            if seg_dim % oli_num_bodies != 0:
                raise ValueError(
                    f"Body segment '{name}' dim {seg_dim} not divisible by "
                    f"oli_num_bodies={oli_num_bodies}"
                )
            per_body = seg_dim // oli_num_bodies
            total += g1_num_bodies * per_body
        else:
            total += seg_dim
    return total


def slice_oli_critic_obs_to_g1(
    critic_obs: torch.Tensor,
    group_obs_names: list[str],
    group_obs_dims: list[int],
    *,
    oli_dof: int = OLI_DOF,
    g1_dof: int = G1_DOF,
    oli_num_bodies: int = OLI_NUM_BODIES,
    g1_num_bodies: int = G1_NUM_BODIES,
) -> torch.Tensor:
    """Slice an Oli-side ``critic_obs`` into a G1-shape view, name-keyed.

    Drives entirely off ``group_obs_names`` / ``group_obs_dims`` from the
    env config so it survives obs-layout config changes (as long as segment
    names are stable). Each segment is split off the flat tensor in order,
    transformed independently, and concatenated back.

    Per-segment rule:
      - ``joint_pos`` / ``joint_vel`` / ``actions``: reshape to
        ``(..., n_steps, oli_dof)``, advanced-index with
        ``G1_TO_OLI_INDICES``, flatten back to ``(..., n_steps * g1_dof)``.
      - ``body_pos`` / ``body_ori``: reshape to
        ``(..., oli_num_bodies, per_body)``, advanced-index axis -2 with
        ``G1_BODY_TO_OLI_BODY_INDICES``, flatten back.
      - everything else: pass through unchanged (motion command, anchor
        pos/ori, base lin/ang vel — none are DOF/body indexed).

    Raises if the input flat dim disagrees with ``sum(group_obs_dims)``.
    """
    expected_in = sum(group_obs_dims)
    if critic_obs.shape[-1] != expected_in:
        raise ValueError(
            f"slice_oli_critic_obs_to_g1: expected last dim {expected_in} "
            f"(sum of group_obs_dims), got {critic_obs.shape[-1]}"
        )

    out_segments = []
    cursor = 0
    leading_shape = critic_obs.shape[:-1]
    for name, seg_dim in zip(group_obs_names, group_obs_dims):
        seg = critic_obs[..., cursor : cursor + seg_dim]
        cursor += seg_dim

        if name in _DOF_SEGMENT_NAMES:
            n_steps = seg_dim // oli_dof
            seg_g1 = seg.reshape(*leading_shape, n_steps, oli_dof)[..., G1_TO_OLI_INDICES]
            out_segments.append(seg_g1.reshape(*leading_shape, n_steps * g1_dof))
        elif name in _COMMAND_DOF_SEGMENT_NAMES:
            # Layout: 2 * N_fut * oli_dof (pos block + vel block, t-major).
            n_fut = seg_dim // (2 * oli_dof)
            seg_g1 = seg.reshape(*leading_shape, 2, n_fut, oli_dof)[..., G1_TO_OLI_INDICES]
            out_segments.append(seg_g1.reshape(*leading_shape, 2 * n_fut * g1_dof))
        elif name in _BODY_SEGMENT_NAMES:
            per_body = seg_dim // oli_num_bodies
            seg_g1 = seg.reshape(*leading_shape, oli_num_bodies, per_body)[
                ..., G1_BODY_TO_OLI_BODY_INDICES, :
            ]
            out_segments.append(seg_g1.reshape(*leading_shape, g1_num_bodies * per_body))
        else:
            out_segments.append(seg)

    return torch.cat(out_segments, dim=-1)
