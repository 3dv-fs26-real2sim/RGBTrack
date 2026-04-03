# generate_mask.py
import numpy as np, cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

SCENE_DIR    = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
CHECKPOINT   = "/work/courses/3dv/team22/foundationpose/weights/sam_vit_h_4b8939.pth.1"
MODEL_TYPE   = "vit_h"

# Load frame 0
frame0 = cv2.imread(f"{SCENE_DIR}/rgb/000000.png")
frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

# Load SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.cuda()
predictor = SamPredictor(sam)
predictor.set_image(frame0_rgb)

# Provide a point ON the object (click-to-get coordinates from the image)
# To find x, y: open frame 0 in any image viewer, note the pixel coords of the object center
point_x, point_y = 320, 240   # TODO: replace with actual object center pixel
input_point = np.array([[point_x, point_y]])
input_label = np.array([1])   # 1 = foreground

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Pick the highest-scoring mask
best = masks[np.argmax(scores)]  # (H, W) bool

# Save as 0/255 uint8 PNG
import os
os.makedirs(f"{SCENE_DIR}/masks", exist_ok=True)
cv2.imwrite(f"{SCENE_DIR}/masks/000000.png", (best * 255).astype(np.uint8))
print(f"Mask saved. Nonzero pixels: {best.sum()}")

# Quick visual check
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(frame0_rgb); plt.title("Frame 0")
plt.subplot(1, 2, 2); plt.imshow(best, cmap='gray'); plt.title("Mask")
plt.savefig(f"{SCENE_DIR}/masks/000000_check.png")
print("Saved visual check to masks/000000_check.png")