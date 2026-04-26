#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/compile_gdino_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/compile_gdino_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /work/courses/3dv/team22/FoundationPose-plus-plus/sam-hq/seginw/GroundingDINO
pip install -e . --no-deps
