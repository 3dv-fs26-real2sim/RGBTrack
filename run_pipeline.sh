#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=12:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/pipeline_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/pipeline_%j.err

# Full pipeline: video → frames → seed mask (CAD) → SAM2VP masks → VDA depth → FP++
#
# Usage:
#   sbatch run_pipeline.sh --video <path> --scene <name> --click_x <x> --click_y <y>
#   Optional: --max_frames 1200 (default)
#
# Example:
#   sbatch run_pipeline.sh \
#     --video  /work/courses/3dv/team22/RGBTrack/NEWVIDS/duck/20250804_113654.mp4 \
#     --scene  20250804_113654 \
#     --click_x 345 --click_y 278

MAX_FRAMES=1200
VIDEO=""; SCENE=""; CX=""; CY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)      VIDEO=$2;      shift 2 ;;
        --scene)      SCENE=$2;      shift 2 ;;
        --click_x)    CX=$2;         shift 2 ;;
        --click_y)    CY=$2;         shift 2 ;;
        --max_frames) MAX_FRAMES=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[ -z "$VIDEO" ] || [ -z "$SCENE" ] || [ -z "$CX" ] || [ -z "$CY" ] && \
    echo "Usage: sbatch run_pipeline.sh --video <v> --scene <s> --click_x <x> --click_y <y>" && exit 1

SCENE_DIR=/work/courses/3dv/team22/foundationpose/data/$SCENE
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj
SCRATCH=/work/scratch/hudela
PY=/work/courses/3dv/team22/py310_env/bin/python

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
echo " PIPELINE  scene=$SCENE  max_frames=$MAX_FRAMES"
echo "=================================================="

# ── 1. Extract frames (up to MAX_FRAMES) ────────────────────────────────────
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
cp /work/courses/3dv/team22/foundationpose/data/20250804_104715/cam_K.txt $SCENE_DIR/cam_K.txt

# ── 2. Seed mask: SAM2VP click → dilated mask → BSD → CAD render ─────────────
echo "[2/5] Generating seed mask (CAD-anchored)"
SCENE=$SCENE CX=$CX CY=$CY \
$PY - <<'PYEOF'
import cv2, os, glob, torch, numpy as np, trimesh
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor
from estimater import *
from datareader import *
from tools import *
set_seed(0); set_logging_format()

DATA  = "/work/courses/3dv/team22/foundationpose/data"
MESH  = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

scene  = os.environ["SCENE"]
cx, cy = int(os.environ["CX"]), int(os.environ["CY"])
SCENE_DIR = f"{DATA}/{scene}"
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
mesh = trimesh.load(MESH); mesh.apply_scale(0.001)
scorer = ScorePredictor(); refiner = PoseRefinePredictor(); glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner, debug=0, debug_dir="/tmp", glctx=glctx)
reader = YcbineoatReader(video_dir=SCENE_DIR, shorter_side=None, zfar=np.inf)
pose = binary_search_depth(est, mesh, rgb0, dilated, reader.K, debug=False)
print(f"BSD Z={pose[2,3]:.3f}m")
_, _, mask_r = est.render_rgbd(mesh, pose[None], reader.K, w, h)
mask_np = mask_r[0].cpu().numpy() if mask_r.ndim == 3 else mask_r.cpu().numpy()
cad_u8 = (mask_np > 0.5).astype(np.uint8) * 255
cv2.imwrite(seed_path, cad_u8)
print(f"CAD seed saved: {int((cad_u8>127).sum())} px → {seed_path}")
PYEOF

# ── 3. SAM2VP mask propagation (chunked) ────────────────────────────────────
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
        (cv2.imread(f"{OUT_DIR}/" + os.path.splitext(os.path.basename(frame_files[chunk_start]))[0] + ".png",
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

# ── 4. VDA depth estimation ──────────────────────────────────────────────────
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

# ── 5. FoundationPose++ (run_demo_vda_hand — same as duck_vda_palm_rot, no ScoreNet)
echo "[5/5] FoundationPose++ tracking"
$PY run_demo_vda_hand.py \
    --mesh_file        $MESH \
    --test_scene_dir   $SCENE_DIR \
    --depth_dir        $SCENE_DIR/depth_vda \
    --masks_dir        $SCENE_DIR/masks \
    --debug_dir        $SCRATCH/duck_fp_$SCENE \
    --est_refine_iter  2 \
    --track_refine_iter 2 \
    --debug 2

echo "=================================================="
echo " PIPELINE DONE — scene=$SCENE"
echo "=================================================="
