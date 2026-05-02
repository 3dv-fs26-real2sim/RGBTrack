#!/bin/bash

#SBATCH --account=3dv
#SBATCH --time=12:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/pipeline_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/pipeline_%j.err

# Modular pipeline: 1=frames  2=seed_mask  3=mask_propagation  4=depth  5=fp
#
# Usage:
#   sbatch run_pipeline.sh --scene <name> [--steps 1,2,3,4,5]
#                          [--video <path> --click_x <x> --click_y <y>]
#                          [--mesh <obj>] [--max_frames N] [--fp_debug_dir <name>]
#
# --steps defaults to all (1,2,3,4,5). Pass any subset to run only those stages.
#   step 1 needs --video; step 2 needs --click_x/--click_y; step 5 writes to fp/ unless
#   --fp_debug_dir is given (useful to A/B compare e.g. fp vs fp_tex).
#
# Example — full run:
#   sbatch run_pipeline.sh --video .../20250804_113654.mp4 --scene 20250804_113654 \
#     --click_x 345 --click_y 278
# Example — FP only on already-prepped scene, separate output:
#   sbatch run_pipeline.sh --scene 20250804_124203 --steps 5 --fp_debug_dir fp_tex

MAX_FRAMES=1200
VIDEO=""; SCENE=""; CX=""; CY=""
STEPS="1,2,3,4,5"
FP_DEBUG_DIR="fp"
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj  # default object
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)         VIDEO=$2;         shift 2 ;;
        --scene)         SCENE=$2;         shift 2 ;;
        --click_x)       CX=$2;            shift 2 ;;
        --click_y)       CY=$2;            shift 2 ;;
        --mesh)          MESH=$2;          shift 2 ;;
        --max_frames)    MAX_FRAMES=$2;    shift 2 ;;
        --steps)         STEPS=$2;         shift 2 ;;
        --fp_debug_dir)  FP_DEBUG_DIR=$2;  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[ -z "$SCENE" ] && \
    echo "Usage: sbatch run_pipeline.sh --scene <s> [--steps 1,2,3,4,5] [--video <v> --click_x <x> --click_y <y>] [--mesh <obj>] [--max_frames N] [--fp_debug_dir <name>]" && exit 1

run_step() { [[ ",$STEPS," == *",$1,"* ]]; }

# All outputs in one isolated run directory in scratch
RUN_DIR=/work/scratch/hudela/$SCENE
SCENE_DIR=$RUN_DIR                    # YcbineoatReader expects cam_K + rgb here
PY=/work/courses/3dv/team22/py310_env/bin/python

mkdir -p $RUN_DIR/rgb $RUN_DIR/masks $RUN_DIR/depth_vda $RUN_DIR/fp

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
mkdir -p logs $SCENE_DIR/rgb $SCENE_DIR/masks

echo "=================================================="
echo " PIPELINE  scene=$SCENE  steps=$STEPS  max_frames=$MAX_FRAMES"
echo "=================================================="

# ── 1. Extract frames (up to MAX_FRAMES) ────────────────────────────────────
if ! run_step 1; then
    echo "[1/5] skipped (not in --steps)"
else
N_RGB=$(ls $SCENE_DIR/rgb/*.png 2>/dev/null | wc -l)
if [ "$N_RGB" -gt 0 ]; then
    echo "[1/5] frames already extracted ($N_RGB), skipping"
else
    echo "[1/5] Extracting frames → $SCENE_DIR/rgb"
    $PY -c "
import cv2, os
cap = cv2.VideoCapture('$VIDEO')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
limit = min(total, $MAX_FRAMES)
print(f'Video: {total} frames, extracting {limit}')
for i in range(limit):
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(f'$SCENE_DIR/rgb/{i:06d}.png', frame)
cap.release()
print(f'Extracted {limit} frames')
"
fi
fi
[ -f $RUN_DIR/cam_K.txt ] || cp /work/courses/3dv/team22/foundationpose/data/20250804_104715/cam_K.txt $RUN_DIR/cam_K.txt

# ── 2. Seed mask: SAM2VP click → dilated mask → BSD → CAD render ─────────────
if ! run_step 2; then
    echo "[2/5] skipped (not in --steps)"
elif [ -f $SCENE_DIR/masks/000000.png ]; then
    echo "[2/5] seed mask already exists, skipping"
else
echo "[2/5] Generating seed mask (CAD-anchored)"
SCENE=$SCENE CX=$CX CY=$CY RUN_DIR=$RUN_DIR MESH_FILE=$MESH \
$PY - <<'PYEOF'
import cv2, os, glob, torch, numpy as np, trimesh
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor
from estimater import *
from datareader import *
from tools import *
set_seed(0); set_logging_format()

SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

cx, cy    = int(os.environ["CX"]), int(os.environ["CY"])
SCENE_DIR = os.environ["RUN_DIR"]
MESH_PATH = os.environ["MESH_FILE"]
seed_path = f"{SCENE_DIR}/masks/000000.png"
JPG_DIR   = f"{SCENE_DIR}/seed_jpg"
os.makedirs(JPG_DIR, exist_ok=True)

png_files = sorted(glob.glob(f"{SCENE_DIR}/rgb/*.png"))
cv2.imwrite(f"{JPG_DIR}/000000.jpg", cv2.imread(png_files[0]), [cv2.IMWRITE_JPEG_QUALITY, 95])

img_bgr = cv2.imread(png_files[0])
h, w = img_bgr.shape[:2]
rgb0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# SAM2VP click → seed
predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=JPG_DIR, offload_video_to_cpu=True, offload_state_to_cpu=True)
    _, _, mlogits = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=np.array([[cx, cy]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32))
    m = (mlogits[0] > 0.0).cpu().numpy()
    if m.ndim == 3: m = m[0]
    sam_mask = binary_fill_holes(m)
del predictor, state; torch.cuda.empty_cache()
print(f"SAM2VP seed: {int(sam_mask.sum())} px")

# Dilate to cover full duck
dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
dilated = cv2.dilate(sam_mask.astype(np.uint8) * 255, dil_k) > 127

# BSD + nvdiffrast render
mesh = trimesh.load(MESH_PATH); mesh.apply_scale(0.001)
scorer = ScorePredictor(); refiner = PoseRefinePredictor()
try:
    glctx = dr.RasterizeCudaContext()
    use_rast = True
except Exception as e:
    print(f"nvdiffrast unavailable ({e}), using convex hull fallback")
    glctx = None; use_rast = False

est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner, debug=0, debug_dir="/tmp", glctx=glctx)
reader = YcbineoatReader(video_dir=SCENE_DIR, shorter_side=None, zfar=np.inf)
pose = binary_search_depth(est, mesh, rgb0, dilated, reader.K, debug=False)
print(f"BSD Z={pose[2,3]:.3f}m")
if use_rast:
    _, _, mask_r = est.render_rgbd(mesh, pose[None], reader.K, w, h)
    mask_np = mask_r[0].cpu().numpy() if mask_r.ndim == 3 else mask_r.cpu().numpy()
    cad_u8 = (mask_np > 0.5).astype(np.uint8) * 255
else:
    verts = np.array(mesh.vertices)
    verts_cam = (pose @ np.hstack([verts, np.ones((len(verts),1))]).T).T[:,:3]
    proj = (reader.K @ verts_cam.T).T
    proj = proj[:,:2] / proj[:,2:3]
    hull = cv2.convexHull(np.int32(proj).reshape(-1,1,2))
    cad_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cad_u8, [hull], 255)
cv2.imwrite(seed_path, cad_u8)
print(f"CAD seed saved: {int((cad_u8>127).sum())} px → {seed_path}")
PYEOF
fi

# ── 3. SAM2VP mask propagation (chunked) ────────────────────────────────────
if ! run_step 3; then
    echo "[3/5] skipped (not in --steps)"
else
N_MASKS=$(ls $SCENE_DIR/masks/*.png 2>/dev/null | wc -l)
N_RGB=$(ls $SCENE_DIR/rgb/*.png 2>/dev/null | wc -l)
if [ "$N_MASKS" -ge "$N_RGB" ] && [ "$N_MASKS" -gt 1 ]; then
    echo "[3/5] masks already complete ($N_MASKS), skipping"
else
echo "[3/5] SAM2VP mask propagation"
$PY - <<PYEOF
import glob, os, torch, cv2, shutil
import numpy as np
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

SCENE_DIR = "$SCENE_DIR"
JPG_DIR   = f"{SCENE_DIR}/rgb_jpg"
OUT_DIR   = f"{SCENE_DIR}/masks"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHUNK     = 1000

os.makedirs(JPG_DIR, exist_ok=True)
frame_files = sorted(glob.glob(f"{SCENE_DIR}/rgb/*.png"))
N = len(frame_files)
for p in frame_files:
    out = f"{JPG_DIR}/" + os.path.splitext(os.path.basename(p))[0] + ".jpg"
    if not os.path.exists(out):
        cv2.imwrite(out, cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 95])

duck_mask0 = (cv2.imread(f"{OUT_DIR}/000000.png", cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
chunk_start = 0
while chunk_start < N:
    chunk_end = min(chunk_start + CHUNK, N)
    chunk_dir = f"{SCENE_DIR}/rgb_jpg_chunk"
    os.makedirs(chunk_dir, exist_ok=True)
    for p in frame_files[chunk_start:chunk_end]:
        dst = f"{chunk_dir}/" + os.path.splitext(os.path.basename(p))[0] + ".jpg"
        src = f"{JPG_DIR}/" + os.path.splitext(os.path.basename(p))[0] + ".jpg"
        if not os.path.exists(dst): os.symlink(src, dst)
    cur_seed = duck_mask0 if chunk_start == 0 else \
        (cv2.imread(f"{OUT_DIR}/" + os.path.splitext(os.path.basename(frame_files[chunk_start - 1]))[0] + ".png",
                    cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=chunk_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=cur_seed)
        for fi, _, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0] > 0.0).cpu().numpy()
            if m.ndim == 3: m = m[0]
            m = binary_fill_holes(m).astype(np.uint8) * 255
            name = os.path.splitext(os.path.basename(frame_files[chunk_start + fi]))[0]
            cv2.imwrite(f"{OUT_DIR}/{name}.png", m)
        predictor.reset_state(state); del state; torch.cuda.empty_cache()
    shutil.rmtree(chunk_dir)
    chunk_start = chunk_end
    print(f"  chunk done, total: {len(os.listdir(OUT_DIR))}")
print(f"Masks done -> {OUT_DIR}")
PYEOF
fi
fi

# ── 4. VDA depth estimation ──────────────────────────────────────────────────
if ! run_step 4; then
    echo "[4/5] skipped (not in --steps)"
else
echo "[4/5] VDA depth estimation"
N_DEPTH=$(ls $SCENE_DIR/depth_vda/*.png 2>/dev/null | wc -l)
N_RGB=$(ls $SCENE_DIR/rgb/*.png 2>/dev/null | wc -l)
if [ "$N_DEPTH" -eq "$N_RGB" ] && [ "$N_DEPTH" -gt 0 ]; then
    echo "depth_vda already complete ($N_DEPTH frames), skipping"
else
    $PY generate_vda_streaming_maps.py \
        --scene_dir $SCENE_DIR \
        --vda_repo  /work/courses/3dv/team22/Video-Depth-Anything-Internal \
        --rgb_dir   $SCENE_DIR/rgb \
        --out_dir   $SCENE_DIR/depth_vda
fi
fi

# ── 5. FoundationPose tracking — run_demo_bsd_raw_depth (BSD init + raw depth,
#       same recipe that produced duck_bsd_raw). VDA depth is metric in unit but
#       its absolute scale is per-scene off, so we calibrate via BSD on frame 0
#       (depth_scale = bsd_z / vda_mean_in_mask) and write depth_vda_scaled/.
if ! run_step 5; then
    echo "[5/5] skipped (not in --steps)"
else
SCALED_DIR=$SCENE_DIR/depth_vda_scaled
N_SCALED=$(ls $SCALED_DIR/*.png 2>/dev/null | wc -l)
N_RGB=$(ls $SCENE_DIR/rgb/*.png 2>/dev/null | wc -l)
if [ "$N_SCALED" -eq "$N_RGB" ] && [ "$N_SCALED" -gt 0 ]; then
    echo "[5a/5] depth_vda_scaled already complete ($N_SCALED), skipping"
else
echo "[5a/5] BSD-anchored depth scaling → $SCALED_DIR"
SCENE_DIR=$SCENE_DIR MESH_FILE=$MESH $PY - <<'PYEOF'
import os, glob, cv2, numpy as np, trimesh
from estimater import *
from datareader import *
from tools import *
set_seed(0); set_logging_format()

SCENE_DIR = os.environ["SCENE_DIR"]
MESH_PATH = os.environ["MESH_FILE"]
RGB_DIR   = f"{SCENE_DIR}/rgb"
DVDA_DIR  = f"{SCENE_DIR}/depth_vda"
OUT_DIR   = f"{SCENE_DIR}/depth_vda_scaled"
os.makedirs(OUT_DIR, exist_ok=True)

mesh = trimesh.load(MESH_PATH); mesh.apply_scale(0.001)
scorer = ScorePredictor(); refiner = PoseRefinePredictor()
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner,
                     debug=0, debug_dir="/tmp", glctx=glctx)
reader = YcbineoatReader(video_dir=SCENE_DIR, shorter_side=None, zfar=np.inf)

# frame 0: BSD on the seed mask → metric Z
rgb0 = cv2.cvtColor(cv2.imread(f"{RGB_DIR}/000000.png"), cv2.COLOR_BGR2RGB)
m0   = (cv2.imread(f"{SCENE_DIR}/masks/000000.png", cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
pose0 = binary_search_depth(est, mesh, rgb0, m0, reader.K, debug=False)
bsd_z = float(pose0[2, 3])
d0 = cv2.imread(f"{DVDA_DIR}/000000.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
vda_z = float(d0[m0].mean()) if m0.any() else 1.0
scale = bsd_z / vda_z if vda_z > 0 else 1.0
print(f"BSD z={bsd_z:.3f}  VDA z={vda_z:.3f}  scale={scale:.3f}")

# apply to every frame, save as uint16 mm
for p in sorted(glob.glob(f"{DVDA_DIR}/*.png")):
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    d_scaled = (d * scale * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(f"{OUT_DIR}/{os.path.basename(p)}", d_scaled)
print(f"Wrote {len(os.listdir(OUT_DIR))} scaled depth PNGs")
PYEOF
fi

echo "[5b/5] FoundationPose tracking → $RUN_DIR/$FP_DEBUG_DIR"
$PY run_demo_bsd_raw_depth.py \
    --mesh_file        $MESH \
    --test_scene_dir   $SCENE_DIR \
    --pred_depth_dir   $SCENE_DIR/depth_vda_scaled \
    --debug_dir        $RUN_DIR/$FP_DEBUG_DIR \
    --est_refine_iter  2 \
    --track_refine_iter 2 \
    --debug 2
fi

echo "=================================================="
echo " PIPELINE DONE — scene=$SCENE"
echo "=================================================="
