#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_frame1_mask_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_frame1_mask_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack

/work/courses/3dv/team22/py310_env/bin/python - <<'EOF'
import cv2
import numpy as np
import trimesh
import torch
import os

from estimater import *
from datareader import *
from tools import *

set_seed(0)
set_logging_format()

SCENE    = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
MESH     = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
POSES    = "/work/courses/3dv/team22/foundationpose/debug/duck_vda_palm_rot/ob_in_cam"
OUT_DIR  = "/work/courses/3dv/team22/foundationpose/debug/frame1_test"
os.makedirs(OUT_DIR, exist_ok=True)

mesh = trimesh.load(MESH)
mesh.apply_scale(0.001)

scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir=OUT_DIR, glctx=glctx)

reader = YcbineoatReader(video_dir=SCENE, shorter_side=None, zfar=np.inf)

# Load depth scale from frame 0 (same as tracking runs)
mask0 = cv2.imread(f"{SCENE}/masks/000000.png", cv2.IMREAD_GRAYSCALE)
mask0 = (mask0 > 127).astype(np.uint8)
depth0 = cv2.imread(f"{SCENE}/depth/000000.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
pose0  = np.loadtxt(f"{POSES}/000000.txt").reshape(4, 4)
bsd_z  = float(pose0[2, 3])
vda_z  = depth0[mask0 > 0].mean()
depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
logging.info(f"Depth scale: {depth_scale:.3f}")

# Frame 1
FRAME = 1
id_str = reader.id_strs[FRAME]
color  = reader.get_color(FRAME)
depth  = cv2.imread(f"{SCENE}/depth/{id_str}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 * depth_scale

# Seed pose_last from duck_vda_palm_rot
seed_pose = np.loadtxt(f"{POSES}/{id_str}.txt").reshape(4, 4)
est.pose_last = torch.from_numpy(seed_pose).float().cuda()
logging.info(f"Seed pose:\n{seed_pose}")

# Refine with many iterations — multiple CAD comparisons from good starting point
mask1 = cv2.imread(f"{SCENE}/masks/{id_str}.png", cv2.IMREAD_GRAYSCALE)
mask1 = (mask1 > 127).astype(np.uint8)
clean_depth = depth * (mask1 > 0)

refined_pose = est.track_one(rgb=color, depth=clean_depth, K=reader.K, iteration=10)
logging.info(f"Refined pose:\n{refined_pose}")

# Render CAD mask from refined pose
h, w = color.shape[:2]
cad_mask = render_cad_mask(refined_pose, mesh, reader.K, w=w, h=h)

# Save outputs
cv2.imwrite(f"{OUT_DIR}/mask_cad_frame1.png",  (cad_mask * 255).astype(np.uint8) if cad_mask is not None else np.zeros((h,w), np.uint8))
cv2.imwrite(f"{OUT_DIR}/mask_sam2_frame1.png", mask1 * 255)

# Overlay comparison
overlay = cv2.cvtColor(color, cv2.COLOR_RGB2BGR).copy()
if cad_mask is not None:
    overlay[cad_mask > 0] = (overlay[cad_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
overlay[mask1 > 0] = (overlay[mask1 > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
cv2.imwrite(f"{OUT_DIR}/overlay_frame1.png", overlay)
logging.info(f"Saved to {OUT_DIR}")
EOF
