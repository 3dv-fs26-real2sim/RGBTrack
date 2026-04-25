#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_frame0_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_frame0_%j.err

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

SCENE = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
MESH  = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
OUT   = os.path.join(SCENE, "masks_sb", "000000.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

mesh = trimesh.load(MESH)
mesh.apply_scale(0.001)

scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir="/tmp", glctx=glctx)

reader   = YcbineoatReader(video_dir=SCENE, shorter_side=None, zfar=np.inf)
color_sb = cv2.cvtColor(
    cv2.imread(os.path.join(SCENE, "rgb_sb", "000000.png")),
    cv2.COLOR_BGR2RGB)

# Use existing clean frame-0 mask as the prompt
mask0 = cv2.imread(os.path.join(SCENE, "masks", "000000.png"), cv2.IMREAD_GRAYSCALE)
mask0 = (mask0 > 127).astype(bool)

pose = binary_search_depth(est, mesh, color_sb, mask0, reader.K, debug=True)
logging.info(f"BSD pose on shadow-busted frame 0:\n{pose}")

# Render the new mask from the BSD pose
from tools import render_cad_mask
h, w = color_sb.shape[:2]
new_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
if new_mask is not None:
    cv2.imwrite(OUT, (new_mask * 255).astype(np.uint8))
    logging.info(f"Saved new frame-0 mask to {OUT}")
else:
    logging.warning("render_cad_mask returned None, saving original mask")
    cv2.imwrite(OUT, (mask0 * 255).astype(np.uint8))
EOF
