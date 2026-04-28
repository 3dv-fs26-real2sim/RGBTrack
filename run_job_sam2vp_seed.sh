#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sam2vp_seed_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sam2vp_seed_%j.err

# SAM2VP click-prompted seed mask for ONE scene.
# Click coordinates are pre-defined per scene below.
#
# Usage:
#   sbatch run_job_sam2vp_seed.sh <SCENE_NAME>
# Example:
#   sbatch run_job_sam2vp_seed.sh 20250804_113654
SCENE=${1:?need scene name}

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

SCENE=$SCENE /work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import cv2, os, glob, torch, numpy as np, trimesh
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor
from estimater import *
from datareader import *
from tools import *

set_seed(0)
set_logging_format()

DATA  = "/work/courses/3dv/team22/foundationpose/data"
DEBUG = "/work/courses/3dv/team22/foundationpose/debug"
MESH  = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Original click positions (top-left coords) per scene
CLICKS = {
    "20250804_113654": (345, 278),
    "20250804_124203": (320, 240),
    "20250806_102854": (310, 295),
}

scene = os.environ["SCENE"]
assert scene in CLICKS, f"no click defined for {scene}"
cx, cy = CLICKS[scene]

SCENE_DIR = f"{DATA}/{scene}"
JPG_DIR   = f"{SCENE_DIR}/seed_jpg"   # tiny dir, only frame 0 — avoids OOM on long videos
out_path  = f"{SCENE_DIR}/masks/000000.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
os.makedirs(JPG_DIR, exist_ok=True)
os.makedirs(DEBUG, exist_ok=True)

# Only need frame 0 for the seed mask
png_files = sorted(glob.glob(f"{SCENE_DIR}/rgb/*.png"))
out0 = f"{JPG_DIR}/000000.jpg"
if not os.path.exists(out0):
    cv2.imwrite(out0, cv2.imread(png_files[0]), [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"[{scene}] total frames in scene: {len(png_files)}  click=({cx},{cy})")

predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=JPG_DIR,
                                  offload_video_to_cpu=True,
                                  offload_state_to_cpu=True)
    _, _, mlogits = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=np.array([[cx, cy]], dtype=np.float32),
        labels=np.array([1],        dtype=np.int32),
    )
    m = (mlogits[0] > 0.0).cpu().numpy()
    if m.ndim == 3: m = m[0]
    m = binary_fill_holes(m).astype(np.uint8) * 255

sam_mask = m > 127
print(f"[{scene}] SAM2VP seed: {int(sam_mask.sum())} px")

del predictor, state
torch.cuda.empty_cache()

# ── BSD + CAD render: refine to full duck silhouette ──────────────────────
mesh = trimesh.load(MESH)
mesh.apply_scale(0.001)

scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir="/tmp", glctx=glctx)

reader = YcbineoatReader(video_dir=SCENE_DIR, shorter_side=None, zfar=np.inf)
color0 = cv2.imread(png_files[0])
rgb0   = cv2.cvtColor(color0, cv2.COLOR_BGR2RGB)
h, w   = rgb0.shape[:2]

pose = binary_search_depth(est, mesh, rgb0, sam_mask, reader.K, debug=False)
print(f"[{scene}] BSD pose Z={pose[2,3]:.3f}m")

cad = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
assert cad is not None, "CAD render failed"
cad_u8 = (cad.astype(np.uint8)) * 255
print(f"[{scene}] CAD mask: {int((cad>0).sum())} px")

cv2.imwrite(out_path, cad_u8)
print(f"[{scene}] saved CAD seed -> {out_path}")

# Preview overlay (red = final CAD, green dot = click)
img = color0.copy()
img[cad > 0] = (0.4 * img[cad > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
cv2.imwrite(f"{DEBUG}/sam2vp_seed_{scene}.png", img)
print(f"[{scene}] preview -> {DEBUG}/sam2vp_seed_{scene}.png")
PYEOF
