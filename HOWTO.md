# RGBTrack on ETH Student Cluster — Full How-To

## Overview

Depth-free 6D pose estimation pipeline. No depth camera needed — only RGB video + CAD mesh in mm. Outputs one 4×4 pose matrix per frame.

---

## 1. Connect to the Cluster

**Requires ETH VPN to be active.**

```bash
ssh <your_nethz_id>@student-cluster.inf.ethz.ch
```

Enter your ETH password. You land on a login node — **no GPU here**, all GPU work goes through SLURM.

---

## 2. Key Paths

| Path | What it is |
|------|-----------|
| `/work/courses/3dv/team22/RGBTrack/` | Tracking code (this repo) |
| `/work/courses/3dv/team22/foundationpose/data/` | Scene data, meshes, SAM weights |
| `/work/courses/3dv/team22/foundationpose/weights/` | Refiner + scorer model weights (symlinked into RGBTrack) |
| `/work/courses/3dv/team22/foundationpose_env/` | Python environment |

---

## 3. Set Up Python (login node)

No `activate` script — use alias or full path:

```bash
alias python=/work/courses/3dv/team22/foundationpose_env/bin/python
alias pip=/work/courses/3dv/team22/foundationpose_env/bin/pip
```

> Aliases don't work in SLURM batch jobs — those use full paths already.

---

## 4. First-Time Setup (only once)

### 4a. Build C++/CUDA extensions

```bash
sed -i 's/\r//' /work/courses/3dv/team22/RGBTrack/build_all_conda.sh
sbatch /work/courses/3dv/team22/RGBTrack/run_build.sh
```

Wait for it to finish (`squeue -u $USER`), then check:
```bash
cat /work/courses/3dv/team22/RGBTrack/logs/build_*.out | tail -5
# Should end with: Successfully installed common-0.0.0
```

### 4b. Install extra Python packages

```bash
pip install filterpy warp-lang
```

### 4c. XMem weights

```bash
mkdir -p /work/courses/3dv/team22/RGBTrack/XMem/saves
wget -c -P /work/courses/3dv/team22/RGBTrack/XMem/saves \
    https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth
```

### 4d. Add to Python path

```bash
echo "/work/courses/3dv/team22/RGBTrack/XMem" >> /work/courses/3dv/team22/foundationpose_env/lib/python3.9/site-packages/xmem.pth
echo "/work/courses/3dv/team22/RGBTrack" >> /work/courses/3dv/team22/foundationpose_env/lib/python3.9/site-packages/rgbtrack.pth
```

### 4e. Create tmp directory

```bash
mkdir -p /work/courses/3dv/team22/RGBTrack/tmp
mkdir -p /work/courses/3dv/team22/RGBTrack/logs
```

---

## 5. Pipeline Overview

```
MP4 video  →  prepare_scene.py  →  generate_mask.py  →  run_job.sh  →  pose .txt files
```

---

## 6. Step 1 — Extract RGB Frames

Run once per video:

```bash
python /work/courses/3dv/team22/RGBTrack/scripts/prepare_scene.py
```

Output: 1199 PNG frames in `/work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb/`

---

## 7. Step 2 — Generate Frame 0 Mask with SAM

**First**, find the object pixel coordinates. Download frame 0 locally:

```bash
# On your LOCAL machine:
scp <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb/000000.png .
```

Open it, find pixel (x, y) of object center. Image is 640×480, origin **top-left**.

**Then** on the cluster, update coordinates and submit:

```bash
sed -i 's/point_x, point_y = .*/point_x, point_y = <X>, <Y>/' \
    /work/courses/3dv/team22/RGBTrack/scripts/generate_mask.py
sbatch /work/courses/3dv/team22/RGBTrack/run_mask.sh
```

Verify the mask:
```bash
# On your LOCAL machine:
scp <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/data/20250804_104715/masks/000000_check.png .
```

Open `000000_check.png` — left panel is frame 0, right panel is the mask. Should tightly outline the object.

---

## 8. Step 3 — Run the Tracker

Update `--mesh_file` and `--debug_dir` in `run_job.sh` for the object you want to track:

```bash
nano /work/courses/3dv/team22/RGBTrack/run_job.sh
```

Object mesh paths:
| Object | `--mesh_file` |
|--------|--------------|
| duck   | `/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj` |
| ball   | `/work/courses/3dv/team22/foundationpose/data/object/ball/ball.obj` |
| fish   | `/work/courses/3dv/team22/foundationpose/data/object/fish/fish.obj` |
| grape  | `/work/courses/3dv/team22/foundationpose/data/object/grape/grape.obj` |
| shovel | `/work/courses/3dv/team22/foundationpose/data/object/shovel/shovel.obj` |

> Use a **different `--debug_dir` per object** so poses don't overwrite each other:
> e.g. `--debug_dir /work/courses/3dv/team22/foundationpose/debug/duck`

> **Mesh units:** all meshes are in **mm**. The code applies `×0.001` at load time — do not rescale the `.obj` files.

Submit:
```bash
sbatch /work/courses/3dv/team22/RGBTrack/run_job.sh
```

Monitor:
```bash
squeue -u $USER
tail -f /work/courses/3dv/team22/RGBTrack/logs/job_<ID>.out
```

---

## 9. Output

Pose files written to `--debug_dir/ob_in_cam/`, one per frame:
```
000000.txt  000001.txt  ...  001198.txt
```

Each is a 4×4 homogeneous transform (object → camera):
```
R  R  R  tx
R  R  R  ty
R  R  R  tz
0  0  0   1
```
Translation in **meters**.

Load in Python:
```python
import numpy as np, glob
from scipy.spatial.transform import Rotation

pose_files = sorted(glob.glob("/path/to/debug/duck/ob_in_cam/*.txt"))
poses = np.stack([np.loadtxt(f).reshape(4, 4) for f in pose_files])  # (N, 4, 4)

R = poses[i, :3, :3]   # rotation matrix
t = poses[i, :3,  3]   # translation in meters
euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)  # roll, pitch, yaw
```

Download locally:
```bash
# On your LOCAL machine:
scp -r <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/debug/duck/ob_in_cam ./poses_duck
```

---

## 10. Optional — Visualization

Saves overlay images (3D bounding box + XYZ axes) to `--debug_dir/track_vis/`.

Add `--debug 2` to `run_job.sh`. To limit to first N frames (faster), temporarily edit `run_demo_without_depth.py`:

```python
# Change:
for i in range(len(reader.color_files)):
# To:
for i in range(min(50, len(reader.color_files))):
```

Download result:
```bash
# On your LOCAL machine:
scp -r <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/debug/duck/track_vis ./track_vis_duck
```

---

## 11. Running on a New Video

### 11a. Upload the video

```bash
# On your LOCAL machine:
scp /path/to/your/video.mp4 <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/data/videos/
```

Also upload `cam_K.txt` (3×3 camera intrinsics, one row per line) for your camera if it differs from the existing one:
```bash
scp cam_K.txt <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/data/<your_scene_name>/
```

### 11b. Edit the two Python scripts (top of file only)

**`scripts/prepare_scene.py`** — lines 5–6:
```python
VIDEO     = "/work/courses/3dv/team22/foundationpose/data/videos/<your_video>.mp4"
SCENE_DIR = "/work/courses/3dv/team22/foundationpose/data/<your_scene_name>"
```

**`scripts/generate_mask.py`** — line 6:
```python
SCENE_DIR = "/work/courses/3dv/team22/foundationpose/data/<your_scene_name>"
```

Edit on the cluster:
```bash
nano /work/courses/3dv/team22/RGBTrack/scripts/prepare_scene.py
nano /work/courses/3dv/team22/RGBTrack/scripts/generate_mask.py
```

### 11c. Extract frames

Run on the login node (no GPU needed):
```bash
python /work/courses/3dv/team22/RGBTrack/scripts/prepare_scene.py
```

Copy `cam_K.txt` into the scene dir if you haven't already:
```bash
cp /path/to/cam_K.txt /work/courses/3dv/team22/foundationpose/data/<your_scene_name>/cam_K.txt
```

### 11d. Generate mask

Find the object center pixel in frame 0:
```bash
# On your LOCAL machine:
scp <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/data/<your_scene_name>/rgb/000000.png .
```
Open it, note pixel (x, y) of the object center. Then update `generate_mask.py` line 22:
```python
point_x, point_y = <X>, <Y>
```

Submit:
```bash
sbatch /work/courses/3dv/team22/RGBTrack/run_mask.sh
```

Verify:
```bash
# On your LOCAL machine:
scp <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/data/<your_scene_name>/masks/000000_check.png .
```

### 11e. Run the tracker

Edit the three lines in `run_job.sh`:
```bash
nano /work/courses/3dv/team22/RGBTrack/run_job.sh
```

Change:
```bash
--mesh_file  /work/courses/3dv/team22/foundationpose/data/object/<object>/.<ext>
--test_scene_dir  /work/courses/3dv/team22/foundationpose/data/<your_scene_name>
--debug_dir  /work/courses/3dv/team22/foundationpose/debug/<object>_<your_scene_name>
```

Submit:
```bash
sbatch /work/courses/3dv/team22/RGBTrack/run_job.sh
```

### 11f. Visualization

Edit `run_job.sh` to add `--debug 2`:
```bash
nano /work/courses/3dv/team22/RGBTrack/run_job.sh
# add --debug 2 to the python call
```

To limit to first 50 frames (faster), edit on the cluster:
```bash
nano /work/courses/3dv/team22/RGBTrack/run_demo_without_depth.py
# Change:
#   for i in range(len(reader.color_files)):
# To:
#   for i in range(min(50, len(reader.color_files))):
```

Resubmit, then download:
```bash
# On your LOCAL machine:
scp -r <nethz_id>@student-cluster.inf.ethz.ch:/work/courses/3dv/team22/foundationpose/debug/<object>_<your_scene_name>/track_vis ./track_vis
```

---

## 12. Known Issues & Fixes Already Applied

All fixes below are already in the repo. This table is for reference when debugging new setups.

| Issue | Symptom | Fix |
|-------|---------|-----|
| Windows line endings | `$'\r': command not found` in SLURM | `sed -i 's/\r//'` on `.sh` files |
| SLURM time format | `02:00` = 2 min not 2 hours | Use `HH:MM:SS` e.g. `06:00:00` |
| No `bin/activate` | `source .../activate: No such file or directory` | `export PATH=.../foundationpose_env/bin:$PATH` |
| `tensorrt` not installed | `ModuleNotFoundError: No module named 'tensorrt'` | Import wrapped in `try/except` in `tensorrt_models.py` |
| `bop_toolkit_lib` not installed | `ModuleNotFoundError: No module named 'bop_toolkit_lib'` | Unused imports removed from `tools.py` |
| `filterpy` not installed | `ModuleNotFoundError: No module named 'filterpy'` | `pip install filterpy` (see step 4b) |
| `warp` not installed | `NameError: erode_depth not defined` | `pip install warp-lang` (see step 4b) |
| XMem has no `setup.py` | `pip install -e` fails | Added `.pth` file to site-packages (see step 4d) |
| SAM checkpoint corrupted | `PytorchStreamReader failed reading zip archive` | Use `sam_vit_h_4b8939.pth.1` (2.4GB) not `.pth` (35MB) |
| No GPU on login node | `RuntimeError: No CUDA GPUs are available` | Always submit GPU jobs via `sbatch` |
| No display on cluster | `qt.qpa.xcb: could not connect to display` | `cv2.imshow`/`waitKey` disabled in `run_demo_without_depth.py` |
| OOM during pose estimation | `torch.OutOfMemoryError: CUDA out of memory` | `bs` reduced 512/1024 → 64 in `predict_pose_refine.py` and `predict_score.py` |
| Missing `tmp/` directory | `FileNotFoundError: tmp/debug_1.25.png` | `mkdir -p .../RGBTrack/tmp` (see step 4e) |
| `toOpen3dCloud` on zero depth | `ValueError: zero-size array to reduction` | Guarded with `if valid.any():` in `estimater.py` |
| Mesh in millimeters | Pose translation at -14m (nonsense) | `mesh.apply_scale(0.001)` in `run_demo_without_depth.py` |
| Weights not found | `FileNotFoundError: weights/2024-.../config.yml` | `ln -s .../foundationpose/weights .../RGBTrack/weights` on cluster |
| `mycpp` not built | `AttributeError: 'NoneType' object has no attribute 'cluster_poses'` | Run `run_build.sh` once (see step 4a) |
| Deprecated CUDA API | Compile error with PyTorch 2.x | `.type()` → `.scalar_type()` in `bundlesdf/mycuda/common.cu` |
| C++ standard too old | Build errors with newer GCC | `c++14` → `c++17` in `bundlesdf/mycuda/setup.py` |
| Eigen not found during build | `fatal error: Eigen/Core: No such file or directory` | Cluster eigen path added to `bundlesdf/mycuda/setup.py` |
