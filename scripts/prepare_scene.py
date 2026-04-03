# prepare_scene.py
import cv2, numpy as np, os
from pathlib import Path

VIDEO     = "/work/courses/3dv/team22/foundationpose/data/videos/20250804_104715_aria_rgb_cam.mp4"
SCENE_DIR = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"

rgb_dir = Path(SCENE_DIR) / "rgb"
rgb_dir.mkdir(parents=True, exist_ok=True)

# --- extract RGB frames ---
cap = cv2.VideoCapture(VIDEO)
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), frame)
    i += 1
cap.release()
print(f"Extracted {i} RGB frames")


# --- verify cam_K.txt exists ---
cam_K_path = Path(SCENE_DIR) / "cam_K.txt"
assert cam_K_path.exists(), f"cam_K.txt not found at {cam_K_path}"
K = np.loadtxt(cam_K_path).reshape(3, 3)
print(f"Loaded cam_K.txt:\n{K}")