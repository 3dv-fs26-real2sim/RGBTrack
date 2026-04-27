#!/bin/bash
# Usage: ./extract_frames.sh <video.mp4> <SCENE_NAME>
# Extracts PNGs to /work/courses/3dv/team22/foundationpose/data/<SCENE_NAME>/rgb/
# Copies cam_K.txt from the existing reference scene.

VIDEO=$1
SCENE_NAME=$2
REF_SCENE=/work/courses/3dv/team22/foundationpose/data/20250804_104715
OUT=/work/courses/3dv/team22/foundationpose/data/$SCENE_NAME

if [ -z "$VIDEO" ] || [ -z "$SCENE_NAME" ]; then
    echo "Usage: $0 <video.mp4> <SCENE_NAME>"
    exit 1
fi

mkdir -p $OUT/rgb
ffmpeg -y -i "$VIDEO" -vf "fps=50" -start_number 0 "$OUT/rgb/%06d.png"

cp $REF_SCENE/cam_K.txt $OUT/cam_K.txt
echo "Extracted $(ls $OUT/rgb | wc -l) frames -> $OUT/rgb"
echo "Now create $OUT/masks/000000.png as the seed mask before running pipeline"
