# run_pipeline.sh

End-to-end zero-shot 6-DoF object tracking from a single RGB video. One SLURM
job, five modular stages, all outputs isolated under
`/work/scratch/hudela/<SCENE>/`.

## What it does

```
video.mp4 ──► [1] frames ──► [2] seed mask (CAD-anchored)
                                       │
                                       ▼
                              [3] SAM2VP propagation ──► masks/
                                       │
                  rgb/  ──────────────► [4] VDA depth ──► depth_vda/
                                       │
                  mesh/ ──────────────► [5] FoundationPose++ ──► fp/
```

| # | Stage              | Tool                               | Output                              |
|---|--------------------|------------------------------------|-------------------------------------|
| 1 | Frame extraction   | OpenCV                             | `<RUN>/rgb/000000.png …`            |
| 2 | Seed mask          | SAM2VP click + dilation + BSD + nvdiffrast CAD render | `<RUN>/masks/000000.png` |
| 3 | Mask propagation   | SAM2VP video predictor (chunked)   | `<RUN>/masks/NNNNNN.png`            |
| 4 | Depth              | Video-Depth-Anything (streaming)   | `<RUN>/depth_vda/NNNNNN.png`        |
| 5 | 6-DoF tracking     | FoundationPose++ (`run_demo_vda_hand.py`, ScoreNet off) | `<RUN>/fp/{ob_in_cam,track_vis}/`   |

`<RUN>` = `/work/scratch/hudela/<SCENE>`.

## Inputs

| Required | Flag         | Notes                                                       |
|----------|--------------|-------------------------------------------------------------|
| ✓        | `--scene`    | Tag used for the run directory and naming. Free-form string. |
| step 1   | `--video`    | Path to source `.mp4`/`.mov`. Only needed if running step 1. |
| step 2   | `--click_x`, `--click_y` | Pixel coordinates of any point on the object in frame 0. Used as the SAM2VP seed. |

| Optional | Flag             | Default                                                                          |
|----------|------------------|----------------------------------------------------------------------------------|
|          | `--mesh`         | `/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj`              |
|          | `--max_frames`   | `1200`                                                                           |
|          | `--steps`        | `1,2,3,4,5` (any comma-separated subset)                                         |
|          | `--fp_debug_dir` | `fp` — name of the FP output folder under `<RUN>/`. Useful for A/B comparisons.  |

`cam_K.txt` is copied from `20250804_104715` automatically (shared camera across
all duck scenes).

## Outputs

```
/work/scratch/hudela/<SCENE>/
├── cam_K.txt
├── rgb/             000000.png …            # extracted frames
├── masks/           000000.png …            # binary 0/255 PNGs
├── depth_vda/       000000.png …            # uint16 depth in mm
└── fp/                                      # (or whatever --fp_debug_dir is)
    ├── ob_in_cam/   000000.txt …            # 4×4 pose, world←camera
    └── track_vis/   000000.png …            # overlay frames for visualization
```

SLURM logs: `/work/courses/3dv/team22/RGBTrack/logs/pipeline_<JOBID>.{out,err}`.

## How to use

Submit with `sbatch` from `/work/courses/3dv/team22/RGBTrack`:

```bash
cd /work/courses/3dv/team22/RGBTrack
sbatch run_pipeline.sh --scene <name> [flags…]
```

The job uses `account=3dv`, GPU runtime, 12h wall-clock.

### Picking the SAM click

Open frame 0 (or the first frame after extraction) in any viewer and read the
pixel coordinates of any clearly-visible point on the object. SAM2VP only
needs one positive click — dilation (61 px ellipse) covers the rest before
BSD locks in the metric pose.

Example clicks already used:

| Scene              | (x, y)     |
|--------------------|------------|
| 20250804_113654    | 345, 278   |
| 20250804_124203    | 320, 240   |
| 20250806_102854    | 310, 295   |

### Picking the mesh

Default is the textured duck (`duck.obj` + `duck.mtl` + `duck.png`, 4096²
texture). Any `.obj` with a sibling `.mtl` referencing a `map_Kd` texture
file works — trimesh auto-resolves them. Currently available under
`/work/courses/3dv/team22/foundationpose/data/object/`:

```
duck/   grape/   ball/   shovel/   fish/
```

Pass with `--mesh /path/to/object.obj`.

## Step combinations

`--steps` selects which stages run. Stages still have a skip-if-output-exists
safety net inside, so re-running a full pipeline on a finished scene is a
no-op (except step 5, which always overwrites unless you redirect with
`--fp_debug_dir`).

### Full run (new video)

```bash
sbatch run_pipeline.sh \
  --video /work/courses/3dv/team22/RGBTrack/NEWVIDS/duck/20250806_102854.mp4 \
  --scene 20250806_102854 \
  --click_x 310 --click_y 295
```

### FP only — re-run tracking on a prepped scene

Common case: try a different mesh (e.g. textured vs untextured), tweak FP
hyperparameters, or recover from a crashed step 5. Output to a separate
folder so you don't overwrite the previous run.

```bash
sbatch run_pipeline.sh --scene 20250804_124203 --steps 5 --fp_debug_dir fp_tex
sbatch run_pipeline.sh --scene 20250804_124203 --steps 5 \
  --mesh /path/to/another.obj --fp_debug_dir fp_alt
```

### Re-do masks + depth + FP (keep frames)

```bash
rm -rf /work/scratch/hudela/<SCENE>/masks /work/scratch/hudela/<SCENE>/depth_vda
sbatch run_pipeline.sh --scene <SCENE> --steps 2,3,4,5 \
  --click_x <x> --click_y <y>
```

### Re-do only the seed mask (e.g. wrong click)

```bash
rm /work/scratch/hudela/<SCENE>/masks/000000.png
sbatch run_pipeline.sh --scene <SCENE> --steps 2,3,5 --click_x <x> --click_y <y>
# step 4 is skipped because depth_vda is already complete from before
```

### Just regenerate depth

```bash
rm -rf /work/scratch/hudela/<SCENE>/depth_vda
sbatch run_pipeline.sh --scene <SCENE> --steps 4
```

### Extract frames only (preview before committing)

```bash
sbatch run_pipeline.sh --scene <SCENE> --video <path> --steps 1 --max_frames 60
```

## Visualizing results

```bash
SCENE=<scene>
/work/courses/3dv/team22/py310_env/bin/python - <<PYEOF
import cv2, glob
src = sorted(glob.glob(f"/work/scratch/hudela/$SCENE/fp/track_vis/*.png"))
out = f"/work/scratch/hudela/${SCENE}_fp.mp4"
h, w = cv2.imread(src[0]).shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for p in src: vw.write(cv2.imread(p))
vw.release()
print(out, len(src))
PYEOF
scp hudela@student-cluster.inf.ethz.ch:/work/scratch/hudela/${SCENE}_fp.mp4 .
```

## Notes & caveats

- **No `set -e`** — pipeline intentionally continues on stage failure so you
  see what produced output and what didn't. Check the log tail.
- **nvdiffrast on RTX 5060 Ti (sm_120)** can fail with CUDA error 209; step 2
  has a convex-hull fallback that still produces a usable seed.
- **Long videos**: SAM2VP step 3 chunks at 1000 frames to avoid OOM.
- **`apply_scale(0.001)`** is applied to the mesh — your `.obj` should be in
  millimetres (standard for Artec scans). If your mesh is already in metres,
  edit the script.
- **ScoreNet is off** in `run_demo_vda_hand.py` — FP accepts the refiner's
  rotation each frame. Use `run_demo_fp_freeze.py` (separate job:
  `run_job_fp_freeze.sh`) if you want anomaly-detection + freeze-rotation on
  top of the same VDA pipeline.
