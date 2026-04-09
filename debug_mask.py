"""
Diagnostic: print per-link camera-frame positions and projected pixel coords.
Usage:
    python debug_mask.py --h5 /path/to/file.h5 --frame 200
"""
import argparse, sys
import numpy as np
import h5py
import pybullet
import pybullet_data
from scipy.spatial.transform import Rotation

T_CAM_TO_BASE = np.array([
    [-0.02199727, -0.80581615,  0.59175708,  0.20403467],
    [-0.99905014,  0.03998766,  0.01731508, -0.25486327],
    [-0.03761575, -0.59081411, -0.80593036,  0.43379187],
    [ 0.,          0.,          0.,          1.        ]
])
T_BASE_TO_CAM = np.linalg.inv(T_CAM_TO_BASE)

CAM_K = np.array([
    [266.5086044, 0.0,         320.0],
    [0.0,         266.5086044, 240.0],
    [0.0,         0.0,         1.0  ],
])

ARM_JOINT_NAMES  = [f"fer_joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",    required=True)
    parser.add_argument("--urdf",  default="/home/hudela/pandaorca_description/urdf/fer_orcahand_right_extended.urdf")
    parser.add_argument("--frame", type=int, default=200)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        qpos_arm  = f["observations/qpos_arm"][args.frame]
        qpos_hand = f["observations/qpos_hand"][args.frame]

    print(f"qpos_arm  = {np.round(qpos_arm, 3)}")
    print(f"qpos_hand = {np.round(qpos_hand, 3)}")

    client = pybullet.connect(pybullet.DIRECT)
    pybullet.setGravity(0, 0, 0, physicsClientId=client)
    robot = pybullet.loadURDF(
        args.urdf, useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )

    n = pybullet.getNumJoints(robot, physicsClientId=client)
    joint_name_to_idx = {}
    link_name_to_idx  = {}
    print(f"\n--- All {n} joints in URDF ---")
    for j in range(n):
        info  = pybullet.getJointInfo(robot, j, physicsClientId=client)
        jname = info[1].decode()
        lname = info[12].decode()
        joint_name_to_idx[jname] = j
        link_name_to_idx[lname]  = j
        jtype = info[2]
        print(f"  j{j:3d}  joint={jname:40s}  link={lname:40s}  type={jtype}")

    # Check all expected joint names exist
    print("\n--- Joint name lookup ---")
    for n in ARM_JOINT_NAMES + HAND_JOINT_NAMES:
        found = n in joint_name_to_idx
        print(f"  {'OK' if found else 'MISSING':7s}  {n}")

    # Set joints
    for idx, angle in zip([joint_name_to_idx[n] for n in ARM_JOINT_NAMES], qpos_arm):
        pybullet.resetJointState(robot, idx, angle, physicsClientId=client)
    for idx, angle in zip([joint_name_to_idx[n] for n in HAND_JOINT_NAMES], qpos_hand):
        pybullet.resetJointState(robot, idx, angle, physicsClientId=client)

    fx, fy = CAM_K[0,0], CAM_K[1,1]
    cx, cy = CAM_K[0,2], CAM_K[1,2]

    print("\n--- Hand link positions ---")
    hand_links = [ln for ln in link_name_to_idx if "right_" in ln or "orca" in ln]
    for lname in sorted(hand_links):
        lidx  = link_name_to_idx[lname]
        state = pybullet.getLinkState(robot, lidx, computeForwardKinematics=True, physicsClientId=client)
        pos_b = np.array(state[4])
        pos_h = np.append(pos_b, 1.0)
        pos_c = T_BASE_TO_CAM @ pos_h
        z = pos_c[2]
        if z > 0.01:
            u = pos_c[0] / z * fx + cx
            v = pos_c[1] / z * fy + cy
            print(f"  {lname:40s}  base={np.round(pos_b,3)}  cam_z={z:.3f}  uv=({u:.0f},{v:.0f})")
        else:
            print(f"  {lname:40s}  base={np.round(pos_b,3)}  cam_z={z:.3f}  BEHIND CAMERA")

    pybullet.disconnect(client)

if __name__ == "__main__":
    main()
