"""
FK-based hand mask renderer.

Uses PyBullet for FK (joint angles → link transforms in robot base frame),
trimesh to load STL meshes, then projects + rasterizes to a binary mask.

Only hand links (orcahand) are rendered — arm links are far from the duck crop.

Joint ordering (must match H5 qpos arrays):
  qpos_arm  (7,)  : fer_joint1 .. fer_joint7
  qpos_hand (17,) : right_wrist, right_thumb_mcp, right_thumb_abd,
                    right_thumb_pip, right_thumb_dip,
                    right_index_abd, right_index_mcp, right_index_pip,
                    right_middle_abd, right_middle_mcp, right_middle_pip,
                    right_ring_abd, right_ring_mcp, right_ring_pip,
                    right_pinky_abd, right_pinky_mcp, right_pinky_pip
"""

import os
import re
import numpy as np
import trimesh
import cv2
import pybullet
import pybullet_data
from scipy.spatial.transform import Rotation


# Joint names in H5 order
ARM_JOINT_NAMES = [f"fer_joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]
ALL_JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES


class HandMaskRenderer:
    def __init__(self, urdf_path, cam_K, img_w=640, img_h=480, dilate_px=5):
        """
        Args:
            urdf_path : path to fer_orcahand_right_extended.urdf
            cam_K     : (3,3) camera intrinsics
            img_w/h   : image dimensions
            dilate_px : morphological dilation of final mask (pixels)

        The camera-to-base transform is NOT hardcoded — the Aria camera is
        mounted on the right_tower link (moves with the arm).  T_base_to_cam
        is recomputed each render() call from the right_tower FK.
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir  = os.path.dirname(self.urdf_path)
        self.K         = cam_K.astype(np.float64)
        self.T_base_to_cam = np.eye(4)   # updated each frame from FK
        self.W, self.H = img_w, img_h
        self.dilate_px = dilate_px

        # Start PyBullet in DIRECT (headless) mode
        self.client = pybullet.connect(pybullet.DIRECT)
        pybullet.setGravity(0, 0, 0, physicsClientId=self.client)
        self.robot = pybullet.loadURDF(
            self.urdf_path,
            useFixedBase=True,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )

        # Build joint name → index map
        n_joints = pybullet.getNumJoints(self.robot, physicsClientId=self.client)
        self.joint_name_to_idx = {}
        self.link_name_to_idx  = {}
        for j in range(n_joints):
            info = pybullet.getJointInfo(self.robot, j, physicsClientId=self.client)
            jname = info[1].decode()
            lname = info[12].decode()
            self.joint_name_to_idx[jname] = j
            self.link_name_to_idx[lname]  = j  # link index == joint index in PyBullet

        # Indices for arm and hand joints
        self.arm_idxs  = [self.joint_name_to_idx[n] for n in ARM_JOINT_NAMES]
        self.hand_idxs = [self.joint_name_to_idx[n] for n in HAND_JOINT_NAMES]

        # Load visual meshes for hand links only
        self.link_meshes = self._load_hand_meshes()
        print(f"[HandMaskRenderer] loaded {len(self.link_meshes)} hand link meshes")

    # ──────────────────────────────────────────────────────────────────────────
    def _load_hand_meshes(self):
        """Parse URDF to get visual mesh for each hand link, load with trimesh."""
        # Parse URDF XML
        with open(self.urdf_path, "r") as f:
            content = f.read()

        # Find all <link name="..."> blocks and their visual meshes
        link_mesh_map = {}
        # Match link blocks
        link_blocks = re.findall(
            r'<link name="([^"]+)">(.*?)</link>', content, re.DOTALL
        )
        # Links to skip: structural / camera housing / non-hand geometry
        SKIP_LINKS = {"right_tower", "world2right_tower_fixed_jointbody"}

        for link_name, block in link_blocks:
            # Only process orcahand finger/palm/wrist links
            if "right_" not in link_name and "orca" not in link_name:
                continue
            if link_name in SKIP_LINKS:
                continue
            # Find visual mesh (first one, not collision)
            visual_match = re.search(
                r'<visual[^>]*>.*?<mesh filename="([^"]+)"', block, re.DOTALL
            )
            if visual_match:
                rel_path = visual_match.group(1)
                abs_path = os.path.normpath(
                    os.path.join(self.urdf_dir, rel_path)
                )
                if os.path.exists(abs_path):
                    try:
                        mesh = trimesh.load(abs_path, force="mesh")
                        link_mesh_map[link_name] = mesh
                    except Exception as e:
                        print(f"  Warning: could not load {abs_path}: {e}")

        return link_mesh_map

    # ──────────────────────────────────────────────────────────────────────────
    def _set_joints(self, qpos_arm, qpos_hand):
        """Set joint angles in PyBullet."""
        for idx, angle in zip(self.arm_idxs, qpos_arm):
            pybullet.resetJointState(
                self.robot, idx, angle, physicsClientId=self.client
            )
        for idx, angle in zip(self.hand_idxs, qpos_hand):
            pybullet.resetJointState(
                self.robot, idx, angle, physicsClientId=self.client
            )

    def _update_camera_from_tower(self):
        """Recompute T_base_to_cam from right_tower FK (camera moves with arm)."""
        tower_idx = self.link_name_to_idx.get("right_tower")
        if tower_idx is None:
            return
        state = pybullet.getLinkState(
            self.robot, tower_idx,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos  = np.array(state[4])
        quat = np.array(state[5])
        R = Rotation.from_quat(quat).as_matrix()
        T_tower_in_base = np.eye(4)
        T_tower_in_base[:3, :3] = R
        T_tower_in_base[:3,  3] = pos
        self.T_base_to_cam = np.linalg.inv(T_tower_in_base)

    def _get_link_transform(self, link_name):
        """Returns 4x4 transform of link in robot base frame."""
        if link_name not in self.link_name_to_idx:
            return None
        link_idx = self.link_name_to_idx[link_name]
        state = pybullet.getLinkState(
            self.robot, link_idx,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos  = np.array(state[4])  # worldLinkFramePosition
        quat = np.array(state[5])  # worldLinkFrameOrientation (x,y,z,w)
        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = pos
        return T

    # ──────────────────────────────────────────────────────────────────────────
    def render(self, qpos_arm, qpos_hand):
        """
        Render binary hand mask for given joint angles.

        Returns:
            mask : (H, W) uint8, 255 = hand pixel, 0 = background
        """
        self._set_joints(qpos_arm, qpos_hand)
        self._update_camera_from_tower()

        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        for link_name, mesh in self.link_meshes.items():
            T_link_in_base = self._get_link_transform(link_name)
            if T_link_in_base is None:
                continue

            # Transform: base → camera
            T_link_in_cam = self.T_base_to_cam @ T_link_in_base

            # Transform mesh vertices to camera frame
            verts = np.array(mesh.vertices)  # (N, 3)
            verts_h = np.hstack([verts, np.ones((len(verts), 1))])  # (N, 4)
            verts_cam = (T_link_in_cam @ verts_h.T).T[:, :3]  # (N, 3)

            # Keep only vertices in front of camera
            in_front = verts_cam[:, 2] > 0.01
            if not np.any(in_front):
                continue
            verts_cam = verts_cam[in_front]

            # Project to pixel coordinates (all front-facing vertices)
            u_f = verts_cam[:, 0] / verts_cam[:, 2] * fx + cx
            v_f = verts_cam[:, 1] / verts_cam[:, 2] * fy + cy

            # Skip link if all verts project far outside image
            MARGIN = 500
            near = ((u_f >= -MARGIN) & (u_f < self.W + MARGIN) &
                    (v_f >= -MARGIN) & (v_f < self.H + MARGIN))
            if not np.any(near):
                continue

            # Clip to prevent int32 overflow in OpenCV
            u = np.clip(u_f, -10000, 10000).astype(np.int32)
            v = np.clip(v_f, -10000, 10000).astype(np.int32)

            pts = np.stack([u, v], axis=1)

            # Convex hull of all projected verts → OpenCV clips to image bounds
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.reshape(-1, 1, 2))
                cv2.fillConvexPoly(mask, hull, 255)
            else:
                for p in pts:
                    if 0 <= p[0] < self.W and 0 <= p[1] < self.H:
                        cv2.circle(mask, tuple(p), 2, 255, -1)

        # Dilate to cover boundary pixels
        if self.dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.dilate_px * 2 + 1,) * 2
            )
            mask = cv2.dilate(mask, kernel)

        return mask

    def close(self):
        pybullet.disconnect(self.client)
