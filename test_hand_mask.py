"""
Quick test: render hand mask for a few frames and overlay on RGB.
Run locally before deploying to cluster.

Usage:
    python test_hand_mask.py --h5 /path/to/20250804_104715.h5 --out_dir /tmp/mask_test
"""
import argparse, os
import numpy as np
import cv2
import h5py
from hand_mask_renderer import HandMaskRenderer

T_CAM_TO_BASE = np.array([
    [-0.02199727, -0.80581615,  0.59175708,  0.20403467],
    [-0.99905014,  0.03998766,  0.01731508, -0.25486327],
    [-0.03761575, -0.59081411, -0.80593036,  0.43379187],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

CAM_K = np.array([
    [266.5086044, 0.0,         320.0],
    [0.0,         266.5086044, 240.0],
    [0.0,         0.0,         1.0  ],
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",      type=str, required=True)
    parser.add_argument("--urdf",    type=str,
        default="/home/hudela/pandaorca_description/urdf/fer_orcahand_right_extended.urdf")
    parser.add_argument("--out_dir", type=str, default="/tmp/mask_test")
    parser.add_argument("--frames",  type=int, nargs="+", default=[0, 100, 300, 600, 900])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    renderer = HandMaskRenderer(
        urdf_path=args.urdf,
        cam_K=CAM_K,
        T_cam_to_base=T_CAM_TO_BASE,
    )

    with h5py.File(args.h5, "r") as f:
        qpos_arm  = f["observations/qpos_arm"][()]
        qpos_hand = f["observations/qpos_hand"][()]
        rgb_data  = f["observations/images/aria_rgb_cam/color"][()]

    for i in args.frames:
        mask = renderer.render(qpos_arm[i], qpos_hand[i])
        rgb  = cv2.cvtColor(rgb_data[i], cv2.COLOR_RGB2BGR)

        print(f"Frame {i}  mask px: {(mask > 0).sum()}")

        # Also save raw mask for inspection
        mask_path = os.path.join(args.out_dir, f"mask_{i:06d}.png")
        cv2.imwrite(mask_path, mask)

        # Overlay mask in red
        overlay = rgb.copy()
        overlay[mask > 0] = (0, 0, 200)
        vis = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

        out_path = os.path.join(args.out_dir, f"frame_{i:06d}.png")
        cv2.imwrite(out_path, vis)
        print(f"  Saved {out_path}")

    renderer.close()
    print("Done.")

if __name__ == "__main__":
    main()
