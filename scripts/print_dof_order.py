"""Print G1 / Oli joint orders from URDF and the name-keyed G1↔Oli bridge.

Standalone: parses URDFs only, no IsaacLab required. URDF joint definition
order matches IsaacLab's `articulation.data.joint_names` for both robots.

Usage:
    python gear_sonic/scripts/print_dof_order.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
G1_URDF = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
OLI_URDF = REPO_ROOT / "gear_sonic/HU_D04_description/urdf/HU_D04_01.urdf"


def urdf_joint_names(path: Path) -> list[str]:
    tree = ET.parse(path)
    return [
        j.attrib["name"]
        for j in tree.getroot().findall("joint")
        if j.attrib.get("type") != "fixed"
    ]


def main() -> None:
    g1 = urdf_joint_names(G1_URDF)
    oli = urdf_joint_names(OLI_URDF)

    print(f"G1  ({len(g1)} joints, from {G1_URDF.name}):")
    for i, n in enumerate(g1):
        print(f"  {i:2d}  {n}")
    print()

    print(f"Oli ({len(oli)} joints, from {OLI_URDF.name}):")
    for i, n in enumerate(oli):
        print(f"  {i:2d}  {n}")
    print()

    # Name-keyed bridge: for each G1 dof, find its Oli index by joint name.
    g1_to_oli: list[int] = []
    missing_in_oli: list[str] = []
    for n in g1:
        try:
            g1_to_oli.append(oli.index(n))
        except ValueError:
            missing_in_oli.append(n)
            g1_to_oli.append(-1)

    oli_head_idx = [i for i, n in enumerate(oli) if n not in g1]

    print("G1 → Oli (by joint name):")
    for g1_idx, (g1_name, oli_idx) in enumerate(zip(g1, g1_to_oli)):
        oli_name = oli[oli_idx] if oli_idx >= 0 else "<MISSING>"
        marker = "" if g1_idx == oli_idx else "  ←  reorder"
        print(f"  G1[{g1_idx:2d}] {g1_name:30s} → Oli[{oli_idx:2d}] {oli_name}{marker}")
    print()

    print(f"Oli slots NOT covered by G1 (fill with head_default): {oli_head_idx}")
    for i in oli_head_idx:
        print(f"  Oli[{i:2d}] {oli[i]}")
    print()

    if missing_in_oli:
        print(f"WARNING: G1 joints not found in Oli: {missing_in_oli}")
        return

    print("=" * 64)
    print("Drop-in tables for gear_sonic/trl/modules/dof_mapping.py")
    print("=" * 64)
    print()
    print("# Oli URDF order (gear_sonic env actually emits in this order):")
    print("OLI_JOINT_NAMES = [")
    for n in oli:
        print(f'    "{n}",')
    print("]")
    print()
    print("# G1 URDF order (pretrained ckpt was trained against this):")
    print("G1_ISAACLAB_DOF_NAMES = [")
    for n in g1:
        print(f'    "{n}",')
    print("]")
    print()
    print("# G1[i] → Oli[G1_TO_OLI_INDICES[i]]  (name-keyed)")
    print(f"G1_TO_OLI_INDICES = {g1_to_oli}")
    print()
    print(f"OLI_HEAD_INDICES = {oli_head_idx}")


if __name__ == "__main__":
    main()
