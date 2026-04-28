#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_cad_seed_new_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_cad_seed_new_%j.err

# Usage:
#   sbatch run_job_cad_seed_new.sh <SCENE_NAME>             # write rendered mask
#   sbatch run_job_cad_seed_new.sh <SCENE_NAME> --check     # render but DON'T overwrite; save preview overlay
SCENE=${1:?need scene name}
MODE=${2:-write}

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

SCENE_NAME=$SCENE MODE=$MODE /work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import cv2, os, numpy as np, trimesh
from estimater import *
from datareader import *
from tools import *

set_seed(0)
set_logging_format()

DATA  = "/work/courses/3dv/team22/foundationpose/data"
DEBUG = "/work/courses/3dv/team22/foundationpose/debug"
MESH  = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
scene = os.environ["SCENE_NAME"]
mode  = os.environ.get("MODE", "write")
check = mode in ("--check", "check")

mesh = trimesh.load(MESH)
mesh.apply_scale(0.001)

scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir="/tmp", glctx=glctx)

SCENE_DIR = f"{DATA}/{scene}"
seed_path = f"{SCENE_DIR}/masks/000000.png"
assert os.path.exists(seed_path), f"no seed at {seed_path}"

reader = YcbineoatReader(video_dir=SCENE_DIR, shorter_side=None, zfar=np.inf)
color  = cv2.cvtColor(cv2.imread(f"{SCENE_DIR}/rgb/000000.png"), cv2.COLOR_BGR2RGB)
mask0  = (cv2.imread(seed_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
assert mask0.any(), "empty seed"

pose = binary_search_depth(est, mesh, color, mask0, reader.K, debug=False)
print(f"[{scene}] BSD pose Z={pose[2,3]:.3f}m")

h, w = color.shape[:2]
new_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
assert new_mask is not None, "render returned None"
new_u8 = (new_mask * 255).astype(np.uint8)
n_px   = int((new_mask > 0).sum())

# Always save a preview overlay so we can verify visually
preview = cv2.cvtColor(color, cv2.COLOR_RGB2BGR).copy()
preview[new_mask > 0] = (0.4 * preview[new_mask > 0] +
                         0.6 * np.array([0, 0, 255])).astype(np.uint8)
os.makedirs(DEBUG, exist_ok=True)
prev_path = f"{DEBUG}/cad_seed_{scene}.png"
cv2.imwrite(prev_path, preview)
print(f"[{scene}] preview saved -> {prev_path}")

if check:
    print(f"[{scene}] CHECK mode — NOT overwriting seed. rendered={n_px} px")
else:
    cv2.imwrite(seed_path, new_u8)
    print(f"[{scene}] CAD mask saved ({n_px} px)")
PYEOF
