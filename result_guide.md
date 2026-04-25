Pretty easy to navigate.

duck_hand -> video file of actual last result. no hand tracking (ironically). scorenet to validate rotations.

duck_pipe -> mediapipe hand tracking.

depth -> raw depth map
    - See absolute value differences

depth_clip -> clip with some margin.
    - Useful to actually appreciate resolution / boundary layer accuracy

depth_scaled -> same as clipped but has been previously scaled with binary search as is done when processing the videos. scaling, then clip.
    - Same as previous but can now compare in a real scenario. (Still big differences between models)

depth_scaled_v2 -> proper clip, more visual for what happens in table
    - Better appreciate close-up