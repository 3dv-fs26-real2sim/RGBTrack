#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_stitch_clean_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_stitch_clean_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/py310_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python -c "
import cv2, glob
paths = sorted(glob.glob('/work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb_clean/*.png'))
h, w = cv2.imread(paths[0]).shape[:2]
w_ = cv2.VideoWriter('/work/courses/3dv/team22/foundationpose/debug/rgb_clean.mp4',
                     cv2.VideoWriter_fourcc(*'mp4v'), 50, (w, h))
for p in paths: w_.write(cv2.imread(p))
w_.release()
print('done', len(paths))
"
