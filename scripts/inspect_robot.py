"""Spawn a robot in Isaac Lab and print joint / body names + axes in the
articulation's actual runtime DFS order — this is the order the simulator
uses (and matches what ``OLI_JOINT_NAMES`` / ``G1_ISAACLAB_DOF_NAMES`` were
captured from).

Usage:
    python scripts/inspect_robot.py [--robot oli|g1] [--axes] [--limits] [--headless]

Examples:
    # Default: Oli, joints + bodies + axes in IsaacLab order
    python scripts/inspect_robot.py --headless

    # G1 instead, with joint limits
    python scripts/inspect_robot.py --robot g1 --headless --limits

What's printed for each JOINT (in articulation DOF order):
    index, name, axis (xyz, world frame at zero pose), type, limits

What's printed for each BODY (in articulation body order):
    index, name, parent

Notes:
    - "Axis" is read from the URDF/MJCF the robot was spawned from, then
      re-mapped to articulation order. Values are in the joint's local
      frame (URDF <axis xyz=...>). For Oli's tilted hip_pitch you should
      see ``(0, 0.906, ±0.423)``.
    - Run with ``--headless`` to skip rendering.
"""

import argparse


# ---------- AppLauncher must run before any IsaacLab import ----------

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument(
    "--robot",
    choices=["oli", "g1"],
    default="oli",
    help="Which robot to inspect (default: oli).",
)
parser.add_argument(
    "--axes",
    action="store_true",
    default=True,
    help="Print joint axes (default: on).",
)
parser.add_argument("--no-axes", dest="axes", action="store_false")
parser.add_argument(
    "--limits",
    action="store_true",
    help="Print joint position / velocity limits.",
)
parser.add_argument(
    "--from-cfg",
    action="store_true",
    default=True,
    help=(
        "Fast path: read joint/body order from the Python constants in "
        "dof_mapping.py / robots/{oli,g1}.py (these are dumps of the actual "
        "runtime articulation order). Skips the URDF→USD spawn entirely. "
        "Default: on. Use ``--spawn`` to actually spawn the robot in IsaacLab."
    ),
)
parser.add_argument(
    "--spawn",
    dest="from_cfg",
    action="store_false",
    help="Force spawn-based path (slow, takes 30-60s for URDF→USD conversion).",
)

# In ``--from-cfg`` mode we don't even need IsaacLab. Parse args first to
# decide whether to load the heavy AppLauncher.
import sys as _sys
_pre_args, _unknown = parser.parse_known_args()
if _pre_args.from_cfg:
    # Silently ignore IsaacLab-only args (e.g. --headless, --device) so users
    # can pass the same flags they'd use with --spawn without errors.
    args, _ = parser.parse_known_args()
    simulation_app = None  # not used
else:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app


# ---------- IsaacLab imports (only when --spawn mode) ----------

if not _pre_args.from_cfg:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _get_robot_cfg(name):
    if name == "oli":
        from gear_sonic.envs.manager_env.robots.oli import OLI_31DOF_CFG
        return OLI_31DOF_CFG
    if name == "g1":
        from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG
        return G1_CYLINDER_MODEL_12_DEX_CFG
    raise ValueError(f"unknown robot: {name}")


if not _pre_args.from_cfg:
    @configclass
    class InspectSceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 2.0
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=500.0),
        )
        # robot is filled in at runtime


def _fmt_vec(v, prec=4):
    return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in v) + "]"


def _read_axes_from_urdf(urdf_path):
    """Return ``{joint_name: (ax, ay, az)}`` parsed from a URDF file."""
    import os
    import xml.etree.ElementTree as ET

    if not urdf_path or not os.path.exists(urdf_path):
        return {}
    tree = ET.parse(urdf_path)
    out = {}
    for joint in tree.iter("joint"):
        name = joint.get("name")
        ax_el = joint.find("axis")
        if ax_el is None:
            continue
        xyz_str = ax_el.get("xyz")
        if xyz_str is None:
            continue
        try:
            ax = tuple(float(x) for x in xyz_str.split())
        except ValueError:
            continue
        out[name] = ax
    return out


def _say(msg=""):
    """Print and flush — IsaacLab can hold stdout in a buffer that gets dropped on exit."""
    import sys

    print(msg, flush=True)
    sys.stdout.flush()


def _read_axes_from_urdf_safe(urdf_path):
    """Wrap _read_axes_from_urdf with a graceful fallback when the URDF is missing."""
    import os

    if urdf_path and os.path.exists(urdf_path):
        return _read_axes_from_urdf(urdf_path)
    return {}


def main_from_cfg():
    """Fast path: dump joint/body lists from the Python constants.

    These constants in ``trl/modules/dof_mapping.py`` and the per-robot
    config files were captured from a live env's ``robot.joint_names`` /
    ``robot.body_names`` (see comments at the top of dof_mapping.py), so
    they ARE the simulator's runtime articulation order. No need to spawn
    IsaacLab to verify.
    """
    import os

    # All constants live in dof_mapping.py which only imports torch — no
    # IsaacLab/USD chain. Avoid touching ``robots/g1.py`` or ``robots/oli.py``
    # which pull in the full IsaacLab stack.
    if args.robot == "oli":
        from gear_sonic.trl.modules.dof_mapping import (
            OLI_JOINT_NAMES,
            OLI_BODY_NAMES,
        )
        joint_names = OLI_JOINT_NAMES
        body_names = OLI_BODY_NAMES
        urdf_rel = "HU_D04_description/HU_D04_01_template.urdf"
    elif args.robot == "g1":
        from gear_sonic.trl.modules.dof_mapping import (
            G1_ISAACLAB_DOF_NAMES,
            G1_BODY_NAMES,
        )
        joint_names = G1_ISAACLAB_DOF_NAMES
        body_names = G1_BODY_NAMES
        urdf_rel = "data/assets/robot_description/urdf/g1/main.urdf"
    else:
        raise ValueError(f"unknown robot: {args.robot}")

    # Try to find the URDF for axis lookup (relative to gear_sonic root)
    gs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(gs_root, urdf_rel)
    axis_map = _read_axes_from_urdf_safe(urdf_path) if args.axes else {}

    _say("=" * 78)
    _say(f"  Robot: {args.robot}  (from-cfg mode — Python constants)")
    _say(f"  num_joints = {len(joint_names)}, num_bodies = {len(body_names)}")
    if args.axes:
        if axis_map:
            _say(f"  Axes from: {urdf_path}")
        else:
            _say(f"  Axes: URDF not found at {urdf_path} — printing without axes")
    _say("=" * 78)

    _say("\n--- Joints (articulation DOF order) ---")
    for i, jname in enumerate(joint_names):
        parts = [f"  [{i:2d}] {jname:<35}"]
        if args.axes:
            ax = axis_map.get(jname)
            ax_s = _fmt_vec(ax) if ax is not None else "(none)"
            parts.append(f"axis={ax_s}")
        _say("  ".join(parts))

    _say("\n--- Bodies (articulation body order) ---")
    for i, bname in enumerate(body_names):
        _say(f"  [{i:2d}] {bname}")

    _say("\n--- Copy-pasteable Python lists ---")
    _say(f"\n{args.robot.upper()}_JOINT_NAMES = [")
    for jname in joint_names:
        _say(f"    {jname!r},")
    _say("]")
    _say(f"\n{args.robot.upper()}_BODY_NAMES = [")
    for bname in body_names:
        _say(f"    {bname!r},")
    _say("]")

    out_path = f"/tmp/inspect_robot_{args.robot}.txt"
    with open(out_path, "w") as f:
        f.write(f"# Robot: {args.robot}  (from-cfg mode)\n")
        f.write(f"# num_joints={len(joint_names)}  num_bodies={len(body_names)}\n\n")
        f.write(f"{args.robot.upper()}_JOINT_NAMES = [\n")
        for jname in joint_names:
            ax = axis_map.get(jname) if args.axes else None
            ax_s = _fmt_vec(ax) if ax is not None else "(none)"
            f.write(f"    {jname!r:<40},  # axis={ax_s}\n")
        f.write("]\n\n")
        f.write(f"{args.robot.upper()}_BODY_NAMES = [\n")
        for bname in body_names:
            f.write(f"    {bname!r},\n")
        f.write("]\n")
    _say(f"\n[inspect] Also wrote backup to {out_path}")


def main():
    _say(f"\n[inspect] Building sim for robot={args.robot} ...")
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60)
    sim_cfg.device = args.device
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.5])

    _say("[inspect] Configuring scene ...")
    scene_cfg = InspectSceneCfg(num_envs=1, env_spacing=2.0)
    # InteractiveScene expects ``{ENV_REGEX_NS}`` so it can replicate per env.
    raw_robot_cfg = _get_robot_cfg(args.robot)
    scene_cfg.robot = raw_robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # The robot configs use relative paths like ``gear_sonic/data/assets/...``
    # (e.g. ``g1.py:ASSET_DIR``). When this script runs from inside gear_sonic
    # those resolve to a doubled path (``gear_sonic/gear_sonic/...``) and the
    # URDF→USD conversion silently fails. Resolve to absolute path here so
    # the script works regardless of cwd.
    import os as _os
    _gs_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _gs_parent = _os.path.dirname(_gs_root)
    spawn_obj = scene_cfg.robot.spawn
    for _attr in ("asset_path", "usd_path"):
        _p = getattr(spawn_obj, _attr, None)
        if _p and not _os.path.isabs(_p) and not _os.path.exists(_p):
            # Try resolving against gear_sonic parent (where ``gear_sonic/...`` works)
            _candidate = _os.path.join(_gs_parent, _p)
            if _os.path.exists(_candidate):
                setattr(spawn_obj, _attr, _candidate)
                _say(f"[inspect] {_attr}: {_p}  →  {_candidate}")
                continue
            # Try resolving against gear_sonic itself (strip leading gear_sonic/)
            _stripped = _p
            if _p.startswith("gear_sonic/"):
                _stripped = _p[len("gear_sonic/"):]
            _candidate = _os.path.join(_gs_root, _stripped)
            if _os.path.exists(_candidate):
                setattr(spawn_obj, _attr, _candidate)
                _say(f"[inspect] {_attr}: {_p}  →  {_candidate}")
                continue
            _say(f"[inspect] WARNING: cannot resolve {_attr}={_p!r}")
    scene = InteractiveScene(scene_cfg)
    _say("[inspect] Resetting sim (this may take a few seconds for URDF→USD)...")
    sim.reset()
    robot: Articulation = scene["robot"]

    # Try to find the source URDF for axis lookup
    spawn_cfg = robot.cfg.spawn
    urdf_path = getattr(spawn_cfg, "asset_path", None) or getattr(
        spawn_cfg, "usd_path", None
    )
    axis_map = {}
    if urdf_path and urdf_path.endswith(".urdf"):
        axis_map = _read_axes_from_urdf(urdf_path)

    _say()
    _say("=" * 78)
    _say(f"  Robot: {args.robot}")
    if urdf_path:
        _say(f"  Source: {urdf_path}")
    _say(f"  num_joints = {robot.num_joints}, num_bodies = {len(robot.body_names)}")
    _say("=" * 78)

    # ---- Joints in articulation DOF order ----
    _say("\n--- Joints (articulation DOF order) ---")
    soft_lim = (
        robot.data.soft_joint_pos_limits[0].cpu().numpy() if args.limits else None
    )
    for i, jname in enumerate(robot.joint_names):
        parts = [f"  [{i:2d}] {jname:<35}"]
        if args.axes:
            ax = axis_map.get(jname)
            ax_s = _fmt_vec(ax) if ax is not None else "(none)"
            parts.append(f"axis={ax_s}")
        if args.limits and soft_lim is not None:
            lo, hi = soft_lim[i]
            parts.append(f"limits=[{lo:+.3f}, {hi:+.3f}]")
        _say("  ".join(parts))

    # ---- Bodies in articulation body order ----
    _say("\n--- Bodies (articulation body order) ---")
    for i, bname in enumerate(robot.body_names):
        _say(f"  [{i:2d}] {bname}")

    # ---- Comparison hook: print as Python list literal so it can be pasted ----
    _say("\n--- Copy-pasteable Python lists ---")
    _say(f"\n{args.robot.upper()}_JOINT_NAMES = [")
    for jname in robot.joint_names:
        _say(f"    {jname!r},")
    _say("]")
    _say(f"\n{args.robot.upper()}_BODY_NAMES = [")
    for bname in robot.body_names:
        _say(f"    {bname!r},")
    _say("]")

    # Also write to a file as a backup in case stdout is being eaten.
    out_dir = "/tmp"
    out_path = f"{out_dir}/inspect_robot_{args.robot}.txt"
    with open(out_path, "w") as f:
        f.write(f"# Robot: {args.robot}\n")
        f.write(f"# num_joints={robot.num_joints}  num_bodies={len(robot.body_names)}\n\n")
        f.write(f"{args.robot.upper()}_JOINT_NAMES = [\n")
        for jname in robot.joint_names:
            ax = axis_map.get(jname)
            ax_s = _fmt_vec(ax) if ax is not None else "(none)"
            f.write(f"    {jname!r:<40},  # axis={ax_s}\n")
        f.write("]\n\n")
        f.write(f"{args.robot.upper()}_BODY_NAMES = [\n")
        for bname in robot.body_names:
            f.write(f"    {bname!r},\n")
        f.write("]\n")
    _say(f"\n[inspect] Also wrote backup to {out_path}")


if args.from_cfg:
    main_from_cfg()
else:
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
