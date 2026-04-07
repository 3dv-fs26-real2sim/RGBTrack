#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_vis_depth_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_vis_depth_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

SCENE=/work/courses/3dv/team22/foundationpose/data/20250804_104715
OUT=/work/courses/3dv/team22/foundationpose/debug/depth_comparison
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj

mkdir -p $OUT/vda $OUT/metric3d $OUT/depth_pro

# VDA (calibrated)
/work/courses/3dv/team22/py310_env/bin/python visualize_depth.py \
    --source vda \
    --test_scene_dir $SCENE \
    --mesh_file $MESH \
    --calibrate \
    --out_video $OUT/vda/depth_vda.avi

# Metric3D (calibrated)
/work/courses/3dv/team22/py310_env/bin/python visualize_depth.py \
    --source depth_pro_maps \
    --test_scene_dir $SCENE \
    --depth_pro_maps_dir $SCENE/depth_metric3d \
    --mesh_file $MESH \
    --calibrate \
    --out_video $OUT/metric3d/depth_metric3d.avi

# Depth Pro (calibrated)
/work/courses/3dv/team22/py310_env/bin/python visualize_depth.py \
    --source depth_pro_maps \
    --test_scene_dir $SCENE \
    --depth_pro_maps_dir $SCENE/depth_pro \
    --mesh_file $MESH \
    --calibrate \
    --out_video $OUT/depth_pro/depth_depth_pro.avi
