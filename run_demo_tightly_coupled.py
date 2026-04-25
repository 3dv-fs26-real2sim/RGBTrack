"""
FoundationPose duck tracker — Stage 1: anomaly detection & Mask Vis.

Per-frame:
  1. Predict T_pred from a 5-frame velocity buffer.
  2. Run FP track_one to get raw pose T_raw.
  3. Anomaly check: ‖t_raw - t_pred‖ > TR_THRESH_M or angle(R_raw, R_pred) > ROT_THRESH_DEG
  4. Render Expected Silhouette (Green) at T_pred.
  5. Load SAM2VP Raw Mask (Red).
  6. Save Overlay to debug_dir/mask_vis/ to visually verify Stage 2 thresholds.
"""
import argparse
import os
import time
from collections import deque
import logging

import cv2
import numpy as np

from estimater import *
from datareader import *
from tools import render_cad_mask, draw_posed_3d_box

# ── Anomaly thresholds ────────────────────────────────────────────────────────
TR_THRESH_M    = 0.05    # 5 cm/frame translation jump
ROT_THRESH_DEG = 5.0     # 5°/frame rotation jump
LOG_INTERVAL   = 5

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, required=True)
    parser.add_argument('--test_scene_dir', type=str, required=True)
    parser.add_argument('--debug_dir', type=str, required=True)
    parser.add_argument('--est_refine_iter', type=int, default=2)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    args = parser.parse_args()

    debug_dir = args.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    mask_vis_dir = f"{debug_dir}/mask_vis"
    os.makedirs(mask_vis_dir, exist_ok=True)

    # Init Reader & Estimator
    reader = get_dataset_reader(args.test_scene_dir)
    est = FoundationPose(
        model_pts=np.loadtxt(args.mesh_file.replace('.obj', '.txt')),
        model_normals=np.loadtxt(args.mesh_file.replace('.obj', '_normals.txt')),
        mesh=trimesh.load(args.mesh_file),
        debug=args.debug,
        debug_dir=debug_dir
    )

    history = deque(maxlen=5)
    n_anom = 0

    for i in range(len(reader.color_paths)):
        t1 = time.time()
        color = cv2.imread(reader.color_paths[i])
        depth = cv2.imread(reader.depth_paths[i], cv2.IMREAD_UNCHANGED)
        d_scaled = depth * reader.depth_scale

        # Load Raw SAM2VP mask (assuming standard datareader path)
        mask_path = os.path.join(args.test_scene_dir, 'masks', f'{reader.id_strs[i]}.png')
        if os.path.exists(mask_path):
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if raw_mask is not None and len(raw_mask.shape) == 3:
                raw_mask = raw_mask[:, :, 0]
            raw_mask_bin = (raw_mask > 0).astype(np.uint8)
        else:
            raw_mask_bin = np.zeros((color.shape[0], color.shape[1]), dtype=np.uint8)

        # --- INIT FRAME ---
        if i == 0:
            pose = est.register(K=reader.K, rgb=color, depth=d_scaled, ob_mask=raw_mask_bin)
            history.append(pose.copy())
            logging.info(f"[frame {i}] INIT")
            continue

        # --- PREDICT T_pred (Motion Prior) ---
        if len(history) >= 2:
            t_last = history[-1][:3, 3]
            velocities = [history[j][:3, 3] - history[j-1][:3, 3] for j in range(1, len(history))]
            mean_v = np.mean(velocities, axis=0)
            
            T_pred = history[-1].copy()
            T_pred[:3, 3] = t_last + mean_v
            # Freeze rotation: R_pred = R_last
        else:
            T_pred = history[-1].copy()

        # --- RAW TRACKING ---
        T_raw = est.track_one(rgb=color, depth=d_scaled, K=reader.K, iteration=args.track_refine_iter)

        # --- ANOMALY DETECTION ---
        # 1. Translation jump
        tr_jump = np.linalg.norm(T_raw[:3, 3] - T_pred[:3, 3])
        
        # 2. Rotation jump
        R_raw = T_raw[:3, :3]
        R_pred = T_pred[:3, :3]
        rot_vec, _ = cv2.Rodrigues(R_raw @ R_pred.T)
        rt_jump = np.degrees(np.linalg.norm(rot_vec))

        anomaly = (tr_jump > TR_THRESH_M) or (rt_jump > ROT_THRESH_DEG)
        if anomaly:
            n_anom += 1
            tag = "ANOM"
            logging.info(f"[frame {i}] ANOMALY  Δt={tr_jump*100:.1f}cm  Δθ={rt_jump:.1f}°")
        else:
            tag = "OK"

        # --- MASK VISUALIZATION (STAGE 2 PREVIEW) ---
        # Render the Expected Silhouette using T_pred
        expected_mask = render_cad_mask(T_pred, est.mesh, reader.K, w=color.shape[1], h=color.shape[0])

        overlay = color.copy()
        
        # Draw Raw SAM2VP Mask in RED
        overlay[raw_mask_bin > 0] = overlay[raw_mask_bin > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        # Draw Expected Mask in GREEN
        if expected_mask is not None:
            overlay[expected_mask > 0] = overlay[expected_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

        # Save the visualization
        cv2.imwrite(f"{mask_vis_dir}/{reader.id_strs[i]}.png", overlay)

        # Stage 1: Accept raw pose regardless
        pose = T_raw.copy()
        history.append(pose.copy())

        # --- STANDARD DEBUG & LOGGING ---
        if i % LOG_INTERVAL == 0:
            logging.info(f"[frame {i}] {tag}  anomalies_so_far={n_anom}")

        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.debug >= 1:
            t2 = time.time()
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {tag} a{n_anom}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255) if tag == "ANOM" else (255, 0, 0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=pose, bbox=est.bbox)
            
            track_vis_dir = f"{debug_dir}/track_vis"
            os.makedirs(track_vis_dir, exist_ok=True)
            cv2.imwrite(f"{track_vis_dir}/{reader.id_strs[i]}.png", vis)