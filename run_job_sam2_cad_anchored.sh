#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sam2_cad_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sam2_cad_%j.err

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
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python run_sam2_cad_anchored.py \
    --scene_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --ob_in_cam   /work/courses/3dv/team22/foundationpose/debug/duck_vda_palm_rot/ob_in_cam \
    --mesh_file   /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --mask_out_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_cad_anchored \
    --anchor_every 1
