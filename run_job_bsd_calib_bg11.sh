#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_calib_bg11_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_calib_bg11_%j.err

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
import os
from estimater import *
from datareader import *
from tools import *

set_seed(0)
set_logging_format()

SCENE     = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
MESH      = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
RGB_DIR   = os.path.join(SCENE, "rgb_bg11")
DEPTH_DIR = os.path.join(SCENE, "depth_vda_bg11")
MASK_PATH = os.path.join(SCENE, "masks", "000000.png")

mesh = trimesh.load(MESH)
mesh.apply_scale(0.001)

scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir="/tmp", glctx=glctx)

reader = YcbineoatReader(video_dir=SCENE, shorter_side=None, zfar=np.inf)

color = cv2.cvtColor(cv2.imread(os.path.join(RGB_DIR, "000000.png")), cv2.COLOR_BGR2RGB)
mask0 = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
mask0 = (mask0 > 127).astype(bool)

pose = binary_search_depth(est, mesh, color, mask0, reader.K, debug=True)
fp_z = pose[2, 3]
logging.info(f"BSD pose Z (metric): {fp_z:.4f} m")

# Read VDA raw depth at duck centroid
depth_raw = cv2.imread(os.path.join(DEPTH_DIR, "000000.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
ys, xs = np.where(mask0)
cx, cy = int(xs.mean()), int(ys.mean())
vda_mm = depth_raw[cy, cx]
vda_m  = vda_mm / 1000.0
logging.info(f"VDA raw depth at duck centroid ({cx},{cy}): {vda_mm:.1f} mm = {vda_m:.4f} m")

if vda_m > 0:
    scale = fp_z / vda_m
    logging.info(f">>> depth_scale = fp_z / vda_m = {fp_z:.4f} / {vda_m:.4f} = {scale:.4f}")
else:
    logging.warning("VDA depth at centroid is 0, cannot compute scale")
EOF
