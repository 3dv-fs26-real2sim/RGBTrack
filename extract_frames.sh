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
/work/courses/3dv/team22/py310_env/bin/python -c "
import cv2, os
cap = cv2.VideoCapture('$VIDEO')
i = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(f'$OUT/rgb/{i:06d}.png', frame)
    i += 1
cap.release()
print('extracted', i)
"

cp $REF_SCENE/cam_K.txt $OUT/cam_K.txt
echo "Extracted $(ls $OUT/rgb | wc -l) frames -> $OUT/rgb"
echo "Now create $OUT/masks/000000.png as the seed mask before running pipeline"
