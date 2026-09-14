"""Replay motion **through motion_lib** (the same path the env uses to feed the
policy), in the exact joint order the policy obs sees.

Differs from ``play_motion.py``:
  - ``play_motion.py`` reads the npy directly, reorders to ``OLI_JOINT_NAMES``,
    and writes that to the articulation. Bypasses motion_lib entirely.
  - This script initializes a real ``MotionLibRobot`` with the **same config
    used at training time** (mujoco↔isaaclab mapping, body_indexes,
    skip_fk = True, etc.), queries ``get_dof_pos``/``get_root_pos_w``/
    ``get_root_quat_w`` per frame, and writes to the articulation. Whatever
    the policy sees, you see.

If wrist sub-order or head index is wrong in our mapping tables, the wrists or
head will visibly mis-track here — that's the diagnostic.

Usage:
    python gear_sonic/scripts/play_motion_via_lib.py \\
        --motion_file data/oli_motion/atom_motions_v6_selected_fix10n11_100hz \\
        [--motion_idx 0] [--fps 50]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay an Oli motion through motion_lib in policy-obs order."
)
parser.add_argument(
    "--motion_file",
    type=str,
    required=True,
    help="Directory of .npy / .pkl motions, or a single file (same as training motion_file).",
)
parser.add_argument(
    "--motion_idx",
    type=int,
    default=0,
    help="Which loaded motion clip to replay (0..N-1).",
)
parser.add_argument(
    "--fps",
    type=int,
    default=50,
    help="Replay rate in Hz (motion_lib resamples to this; default 50).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from gear_sonic.envs.manager_env.robots.oli import (
    OLI_31DOF_CFG,
    OLI_JOINT_NAMES,
    OLI_BODY_NAMES,
    OLI_NATIVE_ISAACLAB_TO_MUJOCO_MAPPING,
)
from gear_sonic.utils.motion_lib import motion_lib_robot


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot = OLI_31DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _build_motion_lib_cfg(motion_file: str, device: str):
    """Mirror commands.py's motion_lib config but with the augmentations off."""
    # Use the native Oli mapping (everything in DFS / IsaacLab runtime order
    # after the 2026-04-29 unification).
    mapping = OLI_NATIVE_ISAACLAB_TO_MUJOCO_MAPPING

    # Lower joints by name (legs: hip + knee + ankle, both sides) — used by
    # motion_lib when cat_upper_body_poses is enabled. We turn that off
    # below, but the field is required by motion_lib_cfg.
    lower_keywords = ("_hip_", "_knee", "_ankle")
    lower_indices = [
        i
        for i, n in enumerate(OLI_JOINT_NAMES)
        if any(k in n for k in lower_keywords)
    ]

    # Body indexes: identity-ish — motion_lib's get_body_*_w returns the bodies
    # in the order of mapping["isaaclab_joints"] (= OLI_BODY_NAMES, 32 kinematic
    # bodies in DFS order). For the no-FK / direct-replay path we just need
    # the per-frame root pose; body_indexes is only used by some advanced
    # downstream paths. Pass list(range(32)) as a placeholder (full bodies).
    body_indexes = list(range(len(OLI_BODY_NAMES)))
    body_indexes_data = list(range(len(OLI_BODY_NAMES)))

    from easydict import EasyDict

    cfg = EasyDict({
        "motion_file": motion_file,
        "smpl_motion_file": None,
        "asset": EasyDict({
            # Oli MJCF (matches sonic_oli_native.yaml). motion_lib loads this
            # only for SkeletonTree / Humanoid_Batch init bookkeeping; FK is
            # bypassed when the npy carries body_states. Resolve relative to
            # this script's location so the path works regardless of cwd.
            "assetRoot": str(
                Path(__file__).resolve().parents[1] / "HU_D04_description" / "xml"
            ),
            "assetFileName": "HU_D04_01_vis.xml",
            "urdfFileName": "",
        }),
        "extend_config": [],
        "target_fps": int(args_cli.fps),
        "multi_thread": False,
        "filter_motion_keys": None,
        # Mapping (unified DFS for native Oli)
        "mujoco_to_isaaclab_dof": mapping["mujoco_to_isaaclab_dof"],
        "isaaclab_to_mujoco_dof": mapping["isaaclab_to_mujoco_dof"],
        "mujoco_to_isaaclab_body": mapping["mujoco_to_isaaclab_body"],
        "isaaclab_to_mujoco_body": mapping["isaaclab_to_mujoco_body"],
        "num_dof": mapping["num_dof"],
        "body_indexes": torch.tensor(body_indexes, dtype=torch.long, device=device),
        "body_indexes_data": body_indexes_data,
        "lower_joint_indices_mujoco": lower_indices,
        # Disable every augmentation so we see the raw motion.
        "cat_upper_body_poses": False,
        "cat_upper_body_poses_prob": 0.0,
        "randomize_heading": False,
        "freeze_frame_aug": False,
        "freeze_frame_aug_prob": 0.0,
        "randomize_wrist_poses": False,
        "randomize_wrist_prob": 0.0,
        "randomize_wrist_std": 0.0,
    })
    return cfg


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    device = sim.device

    # ---- Init motion_lib ----
    cfg = _build_motion_lib_cfg(args_cli.motion_file, device)
    mlib = motion_lib_robot.MotionLibRobot(cfg, num_envs=1, device=device)
    # Load all unique motions. Same call commands.py uses on init.
    mlib.load_motions(
        random_sample=False, num_motions_to_load=mlib._num_unique_motions  # noqa: SLF001
    )
    n_motions = mlib._num_unique_motions  # noqa: SLF001
    print(f"[play_motion_via_lib] loaded {n_motions} motion(s); replaying #{args_cli.motion_idx}")

    motion_id = torch.tensor([args_cli.motion_idx % n_motions], dtype=torch.long, device=device)
    n_frames_total = int(mlib.get_time_step_total(motion_id).item())
    print(f"[play_motion_via_lib] motion has {n_frames_total} frames at {args_cli.fps} Hz")

    # The articulation native joint order matches OLI_JOINT_NAMES (DFS) after
    # the 2026-04-29 unification, so a direct identity write works. Verify
    # cheaply at startup:
    art_names = list(robot.data.joint_names)
    if art_names != OLI_JOINT_NAMES:
        # Fallback: build a remap table by name, with a clear log.
        print("[play_motion_via_lib] WARNING: articulation joint order != OLI_JOINT_NAMES; using name-keyed remap.")
        remap = [art_names.index(n) for n in OLI_JOINT_NAMES]
    else:
        remap = list(range(len(OLI_JOINT_NAMES)))
    remap_t = torch.tensor(remap, device=device, dtype=torch.long)

    # Replay sub-step gating so frame advances at fps regardless of sim dt.
    steps_per_frame = max(1, int(round((1.0 / args_cli.fps) / sim_dt)))
    frame_idx = 0
    substep = 0

    while simulation_app.is_running():
        step_t = torch.tensor([frame_idx], dtype=torch.long, device=device)

        # ---- Query motion_lib in policy-obs order ----
        dof_pos = mlib.get_dof_pos(motion_id, step_t)            # (1, num_dof)
        dof_vel = mlib.get_dof_vel(motion_id, step_t)            # (1, num_dof)
        root_pos = mlib.get_root_pos_w(motion_id, step_t)        # (1, 3)
        # motion_lib_base.py:1615 already converts body_quat_w xyzw→wxyz
        # before storing, so get_root_quat_w returns wxyz directly.
        root_quat_wxyz = mlib.get_root_quat_w(motion_id, step_t) # (1, 4)  wxyz
        root_lin = mlib.get_root_lin_vel_w(motion_id, step_t)    # (1, 3)
        root_ang = mlib.get_root_ang_vel_w(motion_id, step_t)    # (1, 3)

        # ---- Write root state ----
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = root_pos + scene.env_origins
        root_state[:, 3:7] = root_quat_wxyz
        root_state[:, 7:10] = root_lin
        root_state[:, 10:13] = root_ang
        robot.write_root_state_to_sim(root_state)

        # ---- Write joint state ----
        # dof_pos is in motion-lib "isaaclab" order (= DFS = OLI_JOINT_NAMES).
        # remap_t handles the rare case where articulation order disagrees.
        robot.write_joint_state_to_sim(
            dof_pos[:, remap_t] if not torch.equal(remap_t, torch.arange(len(OLI_JOINT_NAMES), device=device)) else dof_pos,
            dof_vel[:, remap_t] if not torch.equal(remap_t, torch.arange(len(OLI_JOINT_NAMES), device=device)) else dof_vel,
        )

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_state[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        substep += 1
        if substep >= steps_per_frame:
            substep = 0
            frame_idx = (frame_idx + 1) % n_frames_total


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 200.0  # sim tick rate; frame advance gated by --fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
