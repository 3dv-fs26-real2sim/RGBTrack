"""
Estimate T_cam_to_base from duck FoundationPose poses + robot FK.

Method:
  The duck is static for the first N_STATIC frames.
  Duck in camera frame: known from FoundationPose (poses_dir).
  Duck in base frame: unknown x_d, y_d, but z_d ≈ DUCK_Z.

  We also know the EE (fer_link7) position in base frame from FK.
  In camera frame, the EE is unknown BUT it must project onto the image.

  To get T_cam_to_base we use a different trick:
  Since the camera is FIXED and the duck is FIXED (early frames),
  the mean duck_cam position gives us ONE point pair:
    p_duck_cam (known) ↔ p_duck_base (unknown x_d, y_d, z_d=DUCK_Z)

  We need a second independent point pair.  We use the world origin
  (robot base, link0 is at origin) and assume it projects to a known
  approximate pixel.  OR we search over a range of T_cam_to_base candidates
  and pick the one that is consistent with:
    (a) duck at z_base ≈ DUCK_Z
    (b) hand links at *positive* cam_z at frame 200 (where they're visible)
    (c) hand links project to the upper-right quadrant of the image

  We parameterise T_cam_to_base by 6 DOF and do a brute-force / gradient
  search to minimise the residual.

Usage:
    python estimate_extrinsics.py \
        --h5  /work/courses/3dv/team22/hdf5/20250804_104715.h5 \
        --urdf /home/hudela/pandaorca_description/urdf/fer_orcahand_right_extended.urdf \
        --poses_dir /work/courses/3dv/team22/foundationpose/debug/duck_vda_hand/ob_in_cam
"""
import argparse, glob, os
import numpy as np
import h5py
import pybullet
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

CAM_K = np.array([
    [266.5086044, 0.0,         320.0],
    [0.0,         266.5086044, 240.0],
    [0.0,         0.0,         1.0  ],
])
IMG_W, IMG_H = 640, 480
fx, fy = CAM_K[0,0], CAM_K[1,1]
cx, cy = CAM_K[0,2], CAM_K[1,2]

# How many early frames to treat duck as static
N_STATIC  = 30
# Duck approximate height above table in base frame (m) — half the duck model
DUCK_Z    = 0.03

ARM_JOINT_NAMES = [f"fer_joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]


def pose6_to_T(v):
    """(rx,ry,rz, tx,ty,tz) → 4x4 matrix, rotation as axis-angle."""
    rot  = Rotation.from_rotvec(v[:3]).as_matrix()
    T    = np.eye(4)
    T[:3,:3] = rot
    T[:3, 3] = v[3:]
    return T

def T_to_pose6(T):
    v = np.zeros(6)
    v[:3] = Rotation.from_matrix(T[:3,:3]).as_rotvec()
    v[3:] = T[:3, 3]
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",       required=True)
    parser.add_argument("--urdf",     required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--frame_hand", type=int, default=200,
        help="Frame at which the hand is clearly visible in upper-right (pixel ~540,100)")
    args = parser.parse_args()

    # ── Search for existing calibration files ──────────────────────────
    print("=== Searching for calibration files ===")
    search_roots = [
        os.path.dirname(args.h5),
        os.path.dirname(os.path.dirname(args.h5)),
        os.path.dirname(args.poses_dir),
        os.path.dirname(os.path.dirname(args.poses_dir)),
        "/work/courses/3dv/team22",
    ]
    found_any = False
    for root in search_roots:
        for pat in ["**/*.json","**/*.yaml","**/*.yml","**/*.npy","**/*.npz","**/*.txt"]:
            for p in glob.glob(os.path.join(root, pat), recursive=True):
                bn = os.path.basename(p).lower()
                if any(k in bn for k in ["calib","extrinsic","T_cam","cam_to","base_to","transform","aria"]):
                    print(f"  CANDIDATE: {p}")
                    found_any = True
    if not found_any:
        print("  (none found)")

    # ── Load duck poses in camera frame (early static frames) ───────────
    print("\n=== Loading duck poses (first", N_STATIC, "frames) ===")
    duck_cam_list = []
    for i in range(N_STATIC):
        fn = os.path.join(args.poses_dir, f"{i:06d}.txt")
        if os.path.exists(fn):
            M  = np.loadtxt(fn).reshape(4,4)
            duck_cam_list.append(M[:3, 3])
    if not duck_cam_list:
        print("ERROR: no pose files found in", args.poses_dir)
        return
    duck_cam = np.array(duck_cam_list)
    p_duck_cam = duck_cam.mean(0)
    print(f"  Duck cam (mean over {len(duck_cam)} frames): {p_duck_cam.round(4)}")
    print(f"  Duck cam (std): {duck_cam.std(0).round(4)}")

    # ── PyBullet FK setup ───────────────────────────────────────────────
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
    arm_idxs  = [jname2idx[n] for n in ARM_JOINT_NAMES]
    hand_idxs = [jname2idx[n] for n in HAND_JOINT_NAMES]

    with h5py.File(args.h5, "r") as f:
        qpos_arm_all  = f["observations/qpos_arm"][()]
        qpos_hand_all = f["observations/qpos_hand"][()]

    def set_joints(frame):
        for idx, a in zip(arm_idxs,  qpos_arm_all[frame]):
            pybullet.resetJointState(robot, idx, a, physicsClientId=client)
        for idx, a in zip(hand_idxs, qpos_hand_all[frame]):
            pybullet.resetJointState(robot, idx, a, physicsClientId=client)

    def get_link_pos(lname):
        if lname not in lname2idx: return None
        s = pybullet.getLinkState(robot, lname2idx[lname],
                                   computeForwardKinematics=True,
                                   physicsClientId=client)
        return np.array(s[4])

    set_joints(args.frame_hand)
    hand_links_base = {}
    for ln in ["right_palm","right_tower","right_wrist_jointbody"]:
        p = get_link_pos(ln)
        if p is not None:
            hand_links_base[ln] = p
    print(f"\n=== Hand link positions in base frame (frame {args.frame_hand}) ===")
    for ln, p in hand_links_base.items():
        print(f"  {ln}: {p.round(4)}")

    # ── Optimise T_cam_to_base ───────────────────────────────────────────
    # Decision variables: 6-DOF pose of cam in base frame
    # (i.e. T_cam_to_base: p_base = R @ p_cam + t)
    #
    # Cost terms:
    # 1. Duck_base.z should be ≈ DUCK_Z  (strong)
    # 2. Duck_base.x, .y should be in reachable workspace [−0.8, 0.8]  (weak)
    # 3. At frame_hand, EACH hand link should project to UPPER-RIGHT quadrant:
    #    u ∈ [IMG_W*0.5, IMG_W]  v ∈ [0, IMG_H*0.4]   (strong for palm/wrist)
    # 4. All hand link cam_z should be > 0  (strong)

    # Pixel target for palm at frame_hand (approximate from image)
    PALM_U_TARGET = 540
    PALM_V_TARGET = 100

    def cost(v):
        T = pose6_to_T(v)  # T_cam_to_base: p_base = T @ p_cam
        T_bc = np.linalg.inv(T)  # T_base_to_cam: p_cam = T_bc @ p_base

        c = 0.0

        # 1. Duck z in base
        p_duck_base_z = (T @ np.append(p_duck_cam, 1))[2]
        c += 100.0 * (p_duck_base_z - DUCK_Z)**2

        # 2. Duck x,y in workspace  (soft)
        p_duck_base_xy = (T @ np.append(p_duck_cam, 1))[:2]
        c += 0.1 * max(0, np.abs(p_duck_base_xy).max() - 0.8)**2

        # 3 & 4. Hand links
        for ln, p_base in hand_links_base.items():
            p_cam = (T_bc @ np.append(p_base, 1))[:3]
            z = p_cam[2]
            # Penalty for negative z (behind camera)
            c += 50.0 * max(0, 0.1 - z)**2
            if z > 0.01:
                u = p_cam[0]/z * fx + cx
                v = p_cam[1]/z * fy + cy
                if ln == "right_palm":
                    # Palm pixel target
                    c += 5.0 * ((u - PALM_U_TARGET)**2 + (v - PALM_V_TARGET)**2) / 100.0
                else:
                    # Other hand links: stay in upper-right  (soft)
                    c += 1.0 * max(0, IMG_W*0.5 - u)**2 / 100.0
                    c += 1.0 * max(0, v - IMG_H*0.6)**2 / 100.0

        return c

    # Initial guess: current hardcoded matrix
    T0 = np.array([
        [-0.02199727, -0.80581615,  0.59175708,  0.20403467],
        [-0.99905014,  0.03998766,  0.01731508, -0.25486327],
        [-0.03761575, -0.59081411, -0.80593036,  0.43379187],
        [ 0.,          0.,          0.,          1.        ]
    ])
    v0 = T_to_pose6(T0)
    c0 = cost(v0)
    print(f"\n=== Optimising T_cam_to_base (initial cost: {c0:.2f}) ===")

    res = minimize(cost, v0, method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-6, 'fatol': 1e-6})
    print(f"  Final cost: {res.fun:.4f}  success: {res.success}  msg: {res.message}")

    T_opt = pose6_to_T(res.x)
    T_bc_opt = np.linalg.inv(T_opt)

    print(f"\n=== Optimised T_cam_to_base ===")
    print(np.array2string(T_opt, precision=8, suppress_small=True))

    # Check result
    print(f"\n=== Verification ===")
    p_duck_base = (T_opt @ np.append(p_duck_cam, 1))[:3]
    print(f"Duck in base: {p_duck_base.round(4)}  (want z≈{DUCK_Z})")

    set_joints(args.frame_hand)
    for ln, p_base in hand_links_base.items():
        p_cam = (T_bc_opt @ np.append(p_base, 1))[:3]
        z = p_cam[2]
        if z > 0.01:
            u = p_cam[0]/z * fx + cx
            v = p_cam[1]/z * fy + cy
            print(f"  {ln}: cam_z={z:.3f}  uv=({u:.0f},{v:.0f})")
        else:
            print(f"  {ln}: cam_z={z:.3f}  BEHIND CAMERA")

    pybullet.disconnect(client)


if __name__ == "__main__":
    main()
