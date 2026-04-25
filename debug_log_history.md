1. Base rgbtrack
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/1bddd4da95395efd24857e78961ef07f69a36b8c
2. Failed recovery strategy/IoU mask check + re-register after loss)
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/0184b0654d3a6510721d78ce6213d502086b5c7b
3. SAM2 addition. No longer losing track after release
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/7f5b467e5c37eb93cedcf869d21018c7639cc70d
4. SAM2 but smoothed
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/b66257adc229981ca5b30c2fd9efa545ce2fb036
5. SAM2VP (SAM2 Video Prediction). Takes into account previous frames now. Improved tracking smoother transitions.
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/0b246131e9548b0dba43bb9769a98dfab7f87e7f
6. ADD DEPTH (RGBTrack --> FoundationPose)!!!!!!!!!!!!!!!!
   Metric 3d pipeline !!!!(great overestimation of depth)!!!!
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/fffd817833bd63e89a8db164fc5928024953e485
7. Switch to VDA
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/2d82bd0ff3009fd43268551d293489a91910d61f

(In between this two track_one is modded to track_one_new, doubts on this, almost no effect though) have also tested in the end, tbd.

8. GREAT STEP (Binary search to scale depth)
https://github.com/3dv-fs26-real2sim/RGBTrack/commit/c048bea1399fc4aadab67bb702931be1a7a5ef26
