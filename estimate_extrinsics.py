"""
Estimate T_cam_to_tower: fixed transform from right_tower link frame to the
Aria camera optical frame.  The camera moves with the arm, so:
    T_base_to_cam(t) = inv( T_tower_in_base(t) @ T_cam_to_tower )

Constraints used
----------------
1. Early static frames (duck not yet picked up): duck observed at duck_cam(i)
   must all map to the SAME base-frame position with z ≈ DUCK_Z and (x,y)
   within the robot workspace.
2. Frame FRAME_HAND: right_palm (known in base frame from FK) must project to
   approximately the pixel PALM_UV in the image (upper-right area).

Usage
-----
    python estimate_extrinsics.py \
        --h5  /work/courses/3dv/team22/hdf5/20250804_104715.h5 \
        --urdf /home/hudela/pandaorca_description/urdf/fer_orcahand_right_extended.urdf \
        --poses_dir /work/courses/3dv/team22/foundationpose/debug/duck_vda_hand/ob_in_cam \
        [--palm_uv 560 80]   # pixel where the palm appears at --frame_hand
"""
import argparse, os
import numpy as np
import h5py
import pybullet
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

# ── Camera intrinsics ────────────────────────────────────────────────────────
CAM_K = np.array([
    [266.5086044, 0.0,         320.0],
    [0.0,         266.5086044, 240.0],
    [0.0,         0.0,         1.0  ],
])
IMG_W, IMG_H = 640, 480
fx, fy = CAM_K[0, 0], CAM_K[1, 1]
cx, cy = CAM_K[0, 2], CAM_K[1, 2]

# Duck approximate height in robot base frame (m)
DUCK_Z = 0.03

ARM_JOINT_NAMES  = [f"fer_joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pid",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

# Visual mesh origin from URDF (right hand):
# <origin xyz="0.041 0.053 -0.020" rpy="-3.066 -1.081 -1.676"/>
TOWER_CAM_XYZ = np.array([0.04141561887638722,
                            0.052825801794652016,
                           -0.019856742052109723])
TOWER_CAM_RPY = np.array([-3.066013393320364,
                           -1.0811766886677918,
                           -1.6762365509261419])


def make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def pose6_to_T(v):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    T[:3,  3] = v[3:]
    return T


def T_to_pose6(T):
    v = np.zeros(6)
    v[:3] = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    v[3:] = T[:3, 3]
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",        required=True)
    parser.add_argument("--urdf",      required=True)
    parser.add_argument("--poses_dir", required=True,
        help="Directory with ob_in_cam .txt files (FoundationPose output)")
    parser.add_argument("--n_static",  type=int, default=30,
        help="Number of early frames where duck is static on table")
    parser.add_argument("--frame_hand", type=int, default=200,
        help="Frame where the hand is visible for pixel-level verification")
    parser.add_argument("--palm_uv",   type=float, nargs=2, default=[560, 80],
        help="Approximate pixel (u v) where the palm appears at --frame_hand")
    args = parser.parse_args()

    # ── PyBullet setup ───────────────────────────────────────────────────────
    client = pybullet.connect(pybullet.DIRECT)
    pybullet.setGravity(0, 0, 0, physicsClientId=client)
    robot  = pybullet.loadURDF(
        args.urdf, useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    n = pybullet.getNumJoints(robot, physicsClientId=client)
    jname2idx = {}
    lname2idx = {}
    for j in range(n):
        info = pybullet.getJointInfo(robot, j, physicsClientId=client)
        jname2idx[info[1].decode()] = j
        lname2idx[info[12].decode()] = j

    arm_idxs  = [jname2idx[nm] for nm in ARM_JOINT_NAMES if nm in jname2idx]
    hand_idxs = [jname2idx[nm] for nm in HAND_JOINT_NAMES if nm in jname2idx]
    tower_idx = lname2idx.get("right_tower")

    with h5py.File(args.h5, "r") as f:
        qpos_arm_all  = f["observations/qpos_arm"][()]
        qpos_hand_all = f["observations/qpos_hand"][()]

    def set_joints(frame):
        for idx, a in zip(arm_idxs, qpos_arm_all[frame, :len(arm_idxs)]):
            pybullet.resetJointState(robot, idx, a, physicsClientId=client)
        for idx, a in zip(hand_idxs, qpos_hand_all[frame, :len(hand_idxs)]):
            pybullet.resetJointState(robot, idx, a, physicsClientId=client)

    def get_T_tower(frame):
        set_joints(frame)
        state = pybullet.getLinkState(robot, tower_idx,
                                       computeForwardKinematics=True,
                                       physicsClientId=client)
        R = Rotation.from_quat(state[5]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = np.array(state[4])
        return T

    def get_link_pos(frame, lname):
        set_joints(frame)
        if lname not in lname2idx:
            return None
        state = pybullet.getLinkState(robot, lname2idx[lname],
                                       computeForwardKinematics=True,
                                       physicsClientId=client)
        return np.array(state[4])

    # ── Load duck camera-frame positions for early static frames ─────────────
    print(f"Loading duck poses (first {args.n_static} frames) …")
    duck_cam_list  = []
    tower_T_list   = []
    for i in range(args.n_static):
        fn = os.path.join(args.poses_dir, f"{i:06d}.txt")
        if not os.path.exists(fn):
            continue
        M = np.loadtxt(fn).reshape(4, 4)
        duck_cam_list.append(M[:3, 3])
        tower_T_list.append(get_T_tower(i))

    duck_cam = np.array(duck_cam_list)   # (N, 3)
    print(f"  {len(duck_cam)} frames loaded")
    print(f"  duck_cam mean: {duck_cam.mean(0).round(4)}")

    # ── Palm position in base frame at frame_hand ────────────────────────────
    palm_base = get_link_pos(args.frame_hand, "right_palm")
    T_tower_hand = get_T_tower(args.frame_hand)
    print(f"\nFrame {args.frame_hand}: palm base = {palm_base.round(4)}")
    print(f"Frame {args.frame_hand}: tower base pos = {T_tower_hand[:3,3].round(4)}")

    # ── Initial guess: URDF camera visual mesh origin ────────────────────────
    T0 = make_T(TOWER_CAM_XYZ, TOWER_CAM_RPY)
    v0 = T_to_pose6(T0)

    # ── Cost function ────────────────────────────────────────────────────────
    PALM_U, PALM_V = args.palm_uv

    def cost(v):
        T_ct = pose6_to_T(v)          # T_cam_to_tower (what we're solving for)
        c = 0.0

        # 1. Duck consistency: all early frames map duck to same base-frame point
        duck_base_pts = []
        for i, (dc, T_tb) in enumerate(zip(duck_cam_list, tower_T_list)):
            T_cam_in_base = T_tb @ T_ct
            p_base = (T_cam_in_base @ np.append(dc, 1))[:3]
            duck_base_pts.append(p_base)
        duck_base_pts = np.array(duck_base_pts)

        # Penalise variance in duck position across frames
        duck_mean = duck_base_pts.mean(0)
        c += 50.0 * ((duck_base_pts - duck_mean)**2).sum()

        # Duck z ≈ DUCK_Z
        c += 200.0 * (duck_mean[2] - DUCK_Z)**2

        # Duck x,y in workspace
        c += 1.0 * max(0, np.abs(duck_mean[:2]).max() - 0.9)**2

        # 2. Palm pixel constraint at frame_hand
        T_cam_in_base_hand = T_tower_hand @ T_ct
        T_base_to_cam_hand = np.linalg.inv(T_cam_in_base_hand)
        palm_cam = (T_base_to_cam_hand @ np.append(palm_base, 1))[:3]
        z = palm_cam[2]
        if z > 0.01:
            u = palm_cam[0] / z * fx + cx
            v_px = palm_cam[1] / z * fy + cy
            c += 10.0 * ((u - PALM_U)**2 + (v_px - PALM_V)**2) / 100.0
        else:
            c += 500.0  # palm behind camera

        return c

    c0 = cost(v0)
    print(f"\nInitial cost (URDF visual origin): {c0:.4f}")
    print("Optimising …")

    res = minimize(cost, v0, method='Nelder-Mead',
                   options={'maxiter': 100000, 'xatol': 1e-7, 'fatol': 1e-7,
                            'adaptive': True})
    print(f"Final cost: {res.fun:.4f}  success={res.success}")

    T_opt = pose6_to_T(res.x)

    print(f"\n=== Optimised T_cam_to_tower ===")
    print(np.array2string(T_opt, precision=8, suppress_small=True))
    rpy = Rotation.from_matrix(T_opt[:3, :3]).as_euler('xyz', degrees=True)
    print(f"xyz = {T_opt[:3,3].round(5)}")
    print(f"rpy (deg) = {rpy.round(3)}")

    # ── Verify ───────────────────────────────────────────────────────────────
    print(f"\n=== Verification ===")
    duck_base_pts = []
    for dc, T_tb in zip(duck_cam_list, tower_T_list):
        T_cam_in_base = T_tb @ T_opt
        p_base = (T_cam_in_base @ np.append(dc, 1))[:3]
        duck_base_pts.append(p_base)
    duck_base_arr = np.array(duck_base_pts)
    print(f"Duck base mean:  {duck_base_arr.mean(0).round(4)}")
    print(f"Duck base std:   {duck_base_arr.std(0).round(4)}")

    T_base_to_cam = np.linalg.inv(T_tower_hand @ T_opt)
    palm_cam = (T_base_to_cam @ np.append(palm_base, 1))[:3]
    z = palm_cam[2]
    if z > 0.01:
        u = palm_cam[0] / z * fx + cx
        vp = palm_cam[1] / z * fy + cy
        print(f"Palm: cam_z={z:.3f}  uv=({u:.0f},{vp:.0f})  target=({PALM_U:.0f},{PALM_V:.0f})")
    else:
        print(f"Palm still behind camera: cam_z={z:.3f}")

    pybullet.disconnect(client)


if __name__ == "__main__":
    main()
