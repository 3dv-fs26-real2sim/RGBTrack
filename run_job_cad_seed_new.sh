#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_cad_seed_new_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_cad_seed_new_%j.err

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

/work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import cv2, os, numpy as np, trimesh
from estimater import *
from datareader import *
from tools import *

set_seed(0)
set_logging_format()

DATA = "/work/courses/3dv/team22/foundationpose/data"
MESH = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
SCENES = ["20250804_113654", "20250804_124203", "20250806_102854"]

mesh = trimesh.load(MESH)
mesh.apply_scale(0.001)

scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir="/tmp", glctx=glctx)

for scene in SCENES:
    SCENE_DIR = f"{DATA}/{scene}"
    seed_path = f"{SCENE_DIR}/masks/000000.png"
    if not os.path.exists(seed_path):
        print(f"[SKIP] {scene}: no seed at {seed_path}"); continue

    reader = YcbineoatReader(video_dir=SCENE_DIR, shorter_side=None, zfar=np.inf)
    color  = cv2.cvtColor(cv2.imread(f"{SCENE_DIR}/rgb/000000.png"), cv2.COLOR_BGR2RGB)
    mask0  = (cv2.imread(seed_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
    if not mask0.any():
        print(f"[SKIP] {scene}: empty seed mask"); continue

    pose = binary_search_depth(est, mesh, color, mask0, reader.K, debug=False)
    print(f"[{scene}] BSD pose Z={pose[2,3]:.3f}m")

    h, w = color.shape[:2]
    new_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
    if new_mask is None:
        print(f"[{scene}] render returned None — keeping seed"); continue
    new_mask_u8 = (new_mask * 255).astype(np.uint8)
    cv2.imwrite(seed_path, new_mask_u8)
    print(f"[{scene}] CAD mask saved ({(new_mask>0).sum()} px)")
PYEOF
