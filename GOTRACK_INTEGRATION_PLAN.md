# GoTrack Integration Plan

## Overview

Add **GoTrack** (Meta, CV4MR 2025) as a refinement stage on top of FoundationPose, run only on frames where the duck is heavily occluded. GoTrack's visibility-aware optical flow + PnP-RANSAC handles occlusion architecturally — it explicitly excludes occluder pixels from pose computation.

- Repo: https://github.com/facebookresearch/gotrack
- Paper: *GoTrack: Generic 6DoF Object Pose Refinement and Tracking* (Van Nguyen Nguyen et al., CV4MR 2025)
- Built on **frozen DINOv2** features
- Model-to-frame refinement only in the public release (no frame-to-frame tracking) — which is what we want anyway

## What we already have

- Working FoundationPose pipeline (RGBTrack-tuned) producing 6D poses per frame.
- SAM2VP duck masks per frame, mask propagation in pipeline.
- Per-frame VDA depth, BSD-anchored scaling at frame 0.
- `nvdiffrast` rendering plumbed into the FP wrapper (estimater).
- Scaled CAD meshes for ball, duck, fish, grape, shovel.
- FK rotation lock during grasp phases (palm-anchored — see palm layer below).
- Camera intrinsics constant across scenes.

## Integration strategy

Per frame:

1. Run FoundationPose (existing) → coarse pose `T_fp`.
2. Compute **visibility ratio** = SAM2VP mask area / rendered CAD silhouette area at `T_fp`.
3. If `visibility_ratio < 0.4` → run GoTrack refinement init'd with `T_fp` → `T_refined`.
4. Else use `T_fp` directly.
5. Apply FK / palm rotation lock if grasp detected (separate layer).

## Steps

### Step 1 — install GoTrack on the cluster

Separate conda env to avoid breaking the FoundationPose env.

```bash
cd /work/courses/3dv/team22
git clone --recurse-submodules https://github.com/facebookresearch/gotrack.git
cd gotrack
conda env create -f environment.yml
conda activate gotrack
bash scripts/env.sh
```

Update `configs/user/default.yaml` with `root_dir` pointing to a working directory.

### Step 2 — run their LMO demo to verify install

```bash
export DATASET_NAME=lmo
python -m scripts.inference_pose_estimation \
    dataset_name=$DATASET_NAME mode=localization \
    model.use_default_detections=true
```

If this writes pose outputs without error, the install is good.

### Step 3 — understand their inference code

Read:
- `scripts/inference_pose_estimation.py` (main entry)
- The refinement stage
- Model loading
- Pose refinement function

Goal: identify the standalone refinement function with signature roughly
`refine(rgb, K, mesh, init_pose, n_iter) -> refined_pose`.

### Step 4 — standalone wrapper

Create `gotrack_refiner.py` exposing:

```python
class GoTrackRefiner:
    def __init__(self, ckpt_path, device="cuda"):
        ...
    def refine(self, rgb_uint8, K, mesh, init_pose_4x4, n_iter=3) -> np.ndarray:
        ...
```

Adapts to whatever GoTrack's actual API is once read.

### Step 5 — integrate into pipeline

```python
fp_pose = est.track_one(rgb, depth, K, ...)
visibility = compute_visibility_ratio(sam2vp_mask, render_silhouette(mesh, fp_pose, K))
final_pose = gotrack.refine(rgb, K, mesh, fp_pose) if visibility < 0.4 else fp_pose
if grasp_detected_at_frame(i):
    final_pose = apply_palm_rotation_lock(final_pose, i)
```

### Step 6 — visibility ratio

Use the existing `nvdiffrast` context to render the CAD silhouette at `fp_pose`. Ratio = visible-mask-pixels / rendered-silhouette-pixels.

### Step 7 — threshold tuning

Run at `0.0`, `0.4`, `0.9` thresholds on a test sequence. Pick best.

### Step 8 — sanity on visible frames

Run with threshold `0.9` (GoTrack runs almost everywhere). Verify visible-frame poses don't degrade vs FP alone.

### Step 9 — pipeline timing target

For 1200 frames, ~20% occluded:
- FP: ~1 min
- GoTrack on occluded: ~1–2 min
- Total: 2–3 min per video.

## Watchpoints

- **DINOv2 weights** — pulled from torch hub. Cluster may need internet at first run, or pre-staged weights.
- **Mesh format** — GoTrack may expect `.ply` with normals; convert `.obj` if needed.
- **VRAM** — DINOv2 + GoTrack + FP in one process risks OOM on 16 GB. Better: **two-pass** — FP writes poses to disk, then GoTrack refinement pass on occluded frames only.
- **Init quality** — GoTrack needs a reasonable init pose. If FP fails wildly, GoTrack may not recover.
- **Two envs** — easier to script two sequential `sbatch` jobs than to share envs in one process.

## Recommended order

1. Install GoTrack.
2. Run their LMO demo.
3. Read their inference code.
4. Minimal standalone refinement script (RGB + pose + CAD → refined pose).
5. Test on ONE occluded frame from our data — verify output beats FP alone.
6. Then integrate into the pipeline as a post-FP pass.
7. Tune threshold + measure timing.

## Success criteria

- Pose error on occluded frames decreases vs FP alone.
- Pose error on visible frames does NOT increase.
- Total processing time under 5 minutes per 1199-frame video.
- No new failure modes introduced.

## Pipeline shape after this layer is added

```
[1] frame extraction
[2] seed mask (SAM2VP click + CAD)
[3] SAM2VP mask propagation
[4] VDA depth
[5] FoundationPose tracking (existing run_demo_vda_hand)
[6] palm-rotation override on grasp frames (run_demo_palm_anchored logic)
[7] GoTrack refinement on heavily-occluded frames
   ↓
final ob_in_cam/
```
