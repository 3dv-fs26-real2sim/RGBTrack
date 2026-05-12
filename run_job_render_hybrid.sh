#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=00:20:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_render_hybrid_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_render_hybrid_%j.err

# Render a hybrid track_vis stream: FP base poses outside [start..end],
# alternative (e.g. GoTrack) poses inside the window. Writes per-frame
# overlay PNGs + a stitched 50 fps mp4.
#
# CPU-only — no GPU needed (uses opencv + numpy + trimesh).
#
# Usage:
#   sbatch run_job_render_hybrid.sh \
#       --scene_dir   <RUN>                     # has rgb/ + cam_K.txt
#       --base_dir    <BASE>/ob_in_cam          # primary pose source
#       --gotrack_dir <ALT>/ob_in_cam           # override inside the window
#       --out_dir     <OUT>                     # per-frame PNGs land here
#       --out_mp4     <PATH>                    # mp4 path
#       [--start 220]  [--end 400]              # frame range to override
#       [--mesh <obj>]                          # default: duck
#       [--fps  50]
#
# Example:
#   sbatch run_job_render_hybrid.sh \
#     --scene_dir   /work/scratch/hudela/20250804_104715_full \
#     --base_dir    /work/scratch/hudela/20250804_104715_full/fp_no_freeze/ob_in_cam \
#     --gotrack_dir /work/scratch/hudela/20250804_104715_full/fp_gotrack/ob_in_cam \
#     --out_dir     /work/scratch/hudela/20250804_104715_full/fp_hybrid \
#     --out_mp4     /work/scratch/hudela/20250804_104715_full_hybrid.mp4

SCENE_DIR=""
BASE_DIR=""
GOTRACK_DIR=""
OUT_DIR=""
OUT_MP4=""
START=220
END=400
FPS=50
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj

while [[ $# -gt 0 ]]; do
    case $1 in
        --scene_dir)   SCENE_DIR=$2;   shift 2 ;;
        --base_dir)    BASE_DIR=$2;    shift 2 ;;
        --gotrack_dir) GOTRACK_DIR=$2; shift 2 ;;
        --out_dir)     OUT_DIR=$2;     shift 2 ;;
        --out_mp4)     OUT_MP4=$2;     shift 2 ;;
        --start)       START=$2;       shift 2 ;;
        --end)         END=$2;         shift 2 ;;
        --fps)         FPS=$2;         shift 2 ;;
        --mesh)        MESH=$2;        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[ -z "$SCENE_DIR"   ] && echo "ERROR: --scene_dir is required"   && exit 1
[ -z "$BASE_DIR"    ] && echo "ERROR: --base_dir is required"    && exit 1
[ -z "$GOTRACK_DIR" ] && echo "ERROR: --gotrack_dir is required" && exit 1
[ -z "$OUT_DIR"     ] && echo "ERROR: --out_dir is required"     && exit 1
[ -z "$OUT_MP4"     ] && echo "ERROR: --out_mp4 is required"     && exit 1

. /etc/profile.d/modules.sh
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH

cd /work/courses/3dv/team22/RGBTrack
git pull
mkdir -p logs

echo "=================================================="
echo " hybrid render"
echo "  scene_dir   = $SCENE_DIR"
echo "  base_dir    = $BASE_DIR"
echo "  gotrack_dir = $GOTRACK_DIR"
echo "  range       = [$START..$END]   fps=$FPS"
echo "  out_dir     = $OUT_DIR"
echo "  out_mp4     = $OUT_MP4"
echo "=================================================="

/work/courses/3dv/team22/py310_env/bin/python render_hybrid.py \
    --scene_dir   $SCENE_DIR \
    --base_dir    $BASE_DIR \
    --gotrack_dir $GOTRACK_DIR \
    --mesh        $MESH \
    --start       $START \
    --end         $END \
    --out_dir     $OUT_DIR \
    --out_mp4     $OUT_MP4 \
    --fps         $FPS

echo "=================================================="
echo " hybrid render DONE → $OUT_MP4"
echo "=================================================="
