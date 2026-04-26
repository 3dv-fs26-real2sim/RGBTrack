#!/bin/bash
# Run this after: conda activate gsam_env
# Sets up GroundedSAM2 environment from scratch

. /etc/profile.d/modules.sh
module load cuda/12.8
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install supervision pycocotools transformers==4.30.2 ninja
pip install opencv-python-headless Pillow numpy

# GroundingDINO — patch and install
cd /work/courses/3dv/team22/GroundingDINO
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation

# SAM2
pip install -e /work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time --no-build-isolation

echo "Done — test with: python -c \"from groundingdino.util.inference import load_model; print('ok')\""
