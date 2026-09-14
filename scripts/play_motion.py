"""Replay a wbt_lite_mlp-style .npy motion on Oli (HU_D04) in Isaac Lab.

    python gear_sonic/scripts/play_motion.py \
        --motion_path data/oli_motion/some_clip.npy

The npy must contain ``body_states: (T, 32, 13)`` (pos3 + quat_xyzw4 + linvel3
+ angvel3) and ``dof_pos_vel: (T, 31, 2)``, in Oli canonical IsaacLab order
(see OLI_JOINT_NAMES / OLI_BODY_NAMES). Joints are reordered on the fly via
``robot.find_joints(..., preserve_order=True)`` so the npy's own ``dof_names``
order is not assumed to match the articulation's internal order.

Adapted from wbt_lite_mlp/scripts/rsl_rl/play_motion.py.
"""

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay an Oli 31-DOF .npy motion.")
parser.add_argument("--motion_path", type=str, required=True, help="Absolute path to the .npy motion file.")
parser.add_argument("--fps", type=int, default=50, help="Replay rate in Hz (should match npy source fps).")
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

from gear_sonic.envs.manager_env.robots.oli import OLI_31DOF_CFG, OLI_JOINT_NAMES


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot = OLI_31DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = np.load(args_cli.motion_path, allow_pickle=True).item()
    if "body_states" not in motion or "dof_pos_vel" not in motion:
        raise ValueError("Motion npy must contain 'body_states' and 'dof_pos_vel' keys")

    # body_states quaternion stored as (x, y, z, w); IsaacLab wants (w, x, y, z).
    # Reorder indices [3,4,5,6] ← [6,3,4,5] so the 4 quaternion components become wxyz.
    motion["body_states"][:, :, [3, 4, 5, 6]] = motion["body_states"][:, :, [6, 3, 4, 5]]

    body_states_np = motion["body_states"]   # (T, 32, 13)
    dof_pos_vel_np = motion["dof_pos_vel"]   # (T, 31, 2)
    T = body_states_np.shape[0]
    print(f"[play_motion] Loaded {args_cli.motion_path}: T={T} frames, fps={motion.get('fps', args_cli.fps)}")

    body_states = torch.from_numpy(body_states_np).to(device=sim.device, dtype=torch.float)
    dof_pos = torch.from_numpy(dof_pos_vel_np[..., 0]).to(device=sim.device, dtype=torch.float)
    dof_vel = torch.from_numpy(dof_pos_vel_np[..., 1]).to(device=sim.device, dtype=torch.float)

    # If the npy carries dof_names, use them to build a per-file mapping into
    # OLI_JOINT_NAMES; otherwise assume it's already in canonical Oli order.
    if "dof_names" in motion:
        npy_dof_names = list(motion["dof_names"])
        dof_reorder = [npy_dof_names.index(n) for n in OLI_JOINT_NAMES]
        dof_reorder = torch.tensor(dof_reorder, device=sim.device, dtype=torch.long)
        dof_pos = dof_pos[:, dof_reorder]
        dof_vel = dof_vel[:, dof_reorder]

    # Robot's own articulation joint order may differ from OLI_JOINT_NAMES;
    # find_joints resolves the canonical-names → articulation-index mapping.
    joint_order = robot.find_joints(OLI_JOINT_NAMES, preserve_order=True)[0]
    joint_order = torch.tensor(joint_order, device=sim.device, dtype=torch.long)

    # Advance frame at target fps regardless of sim dt. Substeps per frame:
    steps_per_frame = max(1, int(round((1.0 / args_cli.fps) / sim_dt)))
    frame_idx = 0
    substep = 0
    while simulation_app.is_running():
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = body_states[frame_idx, 0, 0:3] + scene.env_origins
        root_states[:, 3:7] = body_states[frame_idx, 0, 3:7]
        root_states[:, 7:10] = body_states[frame_idx, 0, 7:10]
        root_states[:, 10:13] = body_states[frame_idx, 0, 10:13]
        robot.write_root_state_to_sim(root_states)

        robot.write_joint_state_to_sim(
            dof_pos[frame_idx].unsqueeze(0),
            dof_vel[frame_idx].unsqueeze(0),
            joint_ids=joint_order,
        )

        scene.write_data_to_sim()
        sim.render()   # no sim.step() — we're kinematically replaying, not simulating dynamics
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        substep += 1
        if substep >= steps_per_frame:
            substep = 0
            frame_idx = (frame_idx + 1) % T


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 200.0   # sim tick rate; frame advance is gated by --fps above
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
