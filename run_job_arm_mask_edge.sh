#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_arm_mask_edge_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_arm_mask_edge_%j.err

# Arm mask via SAM2VP click → edge-refined per-frame flood fill.
# Uses Canny on RGB as the edge barrier (no separate edge dir needed).
# Usage: sbatch run_job_arm_mask_edge.sh <SCENE> <CLICK_X> <CLICK_Y>
SCENE=${1:?need scene}
CX=${2:?need click x}
CY=${3:?need click y}

# Source RGB lives in the team22 data dir for the regular scene
SCENE_DIR=/work/courses/3dv/team22/foundationpose/data/$SCENE
OUT_DIR=/work/scratch/hudela/${SCENE}_arm/masks_arm
DBG_DIR=/work/scratch/hudela/${SCENE}_arm/dbg
mkdir -p $OUT_DIR $DBG_DIR

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

SCENE_DIR=$SCENE_DIR OUT_DIR=$OUT_DIR DBG_DIR=$DBG_DIR CX=$CX CY=$CY \
/work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import os, glob, cv2, torch, numpy as np
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

SCENE_DIR = os.environ["SCENE_DIR"]
OUT_DIR   = os.environ["OUT_DIR"]
DBG_DIR   = os.environ["DBG_DIR"]
cx, cy    = int(os.environ["CX"]), int(os.environ["CY"])

SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

# JPGs for SAM2VP
RGB_DIR = f"{SCENE_DIR}/rgb"
JPG_DIR = f"/work/scratch/hudela/{os.path.basename(os.path.dirname(OUT_DIR))}/rgb_jpg"
os.makedirs(JPG_DIR, exist_ok=True)
rgb_pngs = sorted(glob.glob(f"{RGB_DIR}/*.png"))
N = len(rgb_pngs)
for p in rgb_pngs:
    j = f"{JPG_DIR}/{os.path.splitext(os.path.basename(p))[0]}.jpg"
    if not os.path.exists(j):
        cv2.imwrite(j, cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"jpgs ready: {N}")

# 1) SAM2VP arm mask track (chunked)
CHUNK = 1000
predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
sam_dir = f"/work/scratch/hudela/{os.path.basename(os.path.dirname(OUT_DIR))}/masks_sam_arm"
os.makedirs(sam_dir, exist_ok=True)

start = 0
import shutil
while start < N:
    end = min(start + CHUNK, N)
    chunk = f"{JPG_DIR}_chunk"
    os.makedirs(chunk, exist_ok=True)
    for p in rgb_pngs[start:end]:
        n = os.path.splitext(os.path.basename(p))[0]
        dst = f"{chunk}/{n}.jpg"; src = f"{JPG_DIR}/{n}.jpg"
        if not os.path.exists(dst): os.symlink(src, dst)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=chunk,
                                     offload_video_to_cpu=True, offload_state_to_cpu=True)
        if start == 0:
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=np.array([[cx, cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32))
        else:
            seed_name = os.path.splitext(os.path.basename(rgb_pngs[start-1]))[0]
            seed = (cv2.imread(f"{sam_dir}/{seed_name}.png", cv2.IMREAD_GRAYSCALE) > 127)
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
        for fi, _, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0] > 0.0).cpu().numpy()
            if m.ndim == 3: m = m[0]
            m = binary_fill_holes(m).astype(np.uint8) * 255
            name = os.path.splitext(os.path.basename(rgb_pngs[start + fi]))[0]
            cv2.imwrite(f"{sam_dir}/{name}.png", m)
        predictor.reset_state(state); del state; torch.cuda.empty_cache()
    shutil.rmtree(chunk)
    start = end
    print(f"  SAM2VP arm chunk done, total: {len(os.listdir(sam_dir))}")

# 2) Edge-refined per-frame: flood-fill SAM mask out to Canny edges
def refine(sam_mask, rgb):
    gray  = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    barrier = cv2.dilate(edges, np.ones((3,3), np.uint8), 1).astype(bool)
    seed = sam_mask.astype(bool) & ~barrier
    if not seed.any():
        return sam_mask
    # flood fill from each connected component of seed, respecting barrier
    h, w = sam_mask.shape
    canvas = np.zeros((h, w), np.uint8)
    canvas[seed] = 1
    n_lbl, lbl = cv2.connectedComponents(canvas)
    out = np.zeros_like(canvas)
    ff_mask_template = np.zeros((h+2, w+2), np.uint8)
    ff_mask_template[1:-1, 1:-1] = barrier.astype(np.uint8)
    for c in range(1, n_lbl):
        ys, xs = np.where(lbl == c)
        if len(ys) == 0: continue
        # use centroid as seed
        sx, sy = int(xs.mean()), int(ys.mean())
        if barrier[sy, sx]:           # centroid hits an edge — pick first non-edge pixel
            sy, sx = int(ys[0]), int(xs[0])
        canvas_copy = np.zeros_like(canvas)
        ff_mask = ff_mask_template.copy()
        cv2.floodFill(canvas_copy, ff_mask, (sx, sy), 1, 0, 0,
                      flags=4 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY)
        out |= (ff_mask[1:-1, 1:-1] & ~barrier).astype(np.uint8)
    out = (out > 0).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return out

for i, p in enumerate(rgb_pngs):
    name = os.path.splitext(os.path.basename(p))[0]
    rgb  = cv2.imread(p)
    sam  = cv2.imread(f"{sam_dir}/{name}.png", cv2.IMREAD_GRAYSCALE)
    if sam is None: continue
    refined = refine(sam, rgb)
    cv2.imwrite(f"{OUT_DIR}/{name}.png", refined)
    if i % 100 == 0:
        # debug overlay every 100 frames
        ov = rgb.copy()
        ov[refined > 0] = (0.5*rgb[refined>0] + 0.5*np.array([0,255,0])).astype(np.uint8)
        cv2.imwrite(f"{DBG_DIR}/{name}.png", ov)
        print(f"  refine {i}/{len(rgb_pngs)}")

print(f"Arm masks done -> {OUT_DIR}")
PYEOF
