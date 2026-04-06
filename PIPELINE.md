# RGBTrack Pipeline — Techniques Overview

## 1. Depth Estimation: Video Depth Anything (VDA)

**What:** Monocular depth estimator trained on video sequences. Produces temporally consistent metric depth maps.

**Why:** FoundationPose needs a depth map per frame for accurate 3D translation tracking. VDA provides this without a physical depth sensor.

**How used:**
- Run offline on all RGB frames → uint16 PNG depth maps (mm) in `test_scene_dir/depth/`
- At frame 0: depth scale correction computed as `depth_scale = bsd_z / vda_z`, where `bsd_z` is the Z from `binary_search_depth` and `vda_z` is the mean VDA depth over the duck mask pixels. This aligns VDA's relative depth scale to metric space.
- All subsequent frames: `depth * depth_scale` passed to `track_one_new`
- VDA depth is only used when duck is **not occluded**. During occlusion the depth over the duck region is unreliable (hand surface dominates), so rotation tracking falls back to other methods.

---

## 2. Object Segmentation: SAM2 Video Predictor (SAM2VP)

**What:** Offline bidirectional video object segmentation. Seeded with one mask, propagates forward through the full video.

**Why:** FoundationPose requires a per-frame binary mask of the tracked object. SAM2VP produces smooth, temporally consistent masks without per-frame prompting.

**How used:**
- **Duck masks**: seeded from a hand-drawn frame-0 mask → saved to `test_scene_dir/masks/`
  - Used every frame as the primary tracking mask for `track_one_new`
  - Used for occlusion detection (pixel count vs frame-0 baseline)
  - Used for recovery re-init (`binary_search_depth`)
  - Used for MediaPipe grasp detection (`on_mask` check)
- **Hand masks**: seeded from `HAND_SEED_FRAME=90` (frame where palm is fully visible) → saved to `test_scene_dir/hand_masks/`
  - Previously used for FoundationPose hand tracking (now removed)
  - Still generated but currently unused in the main pipeline

---

## 3. 6DoF Pose Estimation: FoundationPose

**What:** Model-based 6DoF object pose estimator. Uses a textured 3D mesh + RGB image + depth map + binary mask.

**Modes:**
- `binary_search_depth(est, mesh, color, mask, K)`: initialization — searches for correct Z depth via binary search, then refines full 6DoF pose. Used at frame 0 and after recovery re-init.
- `track_one_new(rgb, depth, K, mask)`: per-frame tracking — refines pose from previous frame estimate using RGB + depth + mask.
- `track_one_new_without_depth(rgb, K, mask)`: per-frame tracking without depth — RGB + mask only. Previously used during active grasp to avoid depth contamination from hand surface (currently not used).

**Mask used per situation:**
| Situation | Mask passed to FoundationPose |
|---|---|
| Frame 0 init | Duck mask only |
| Normal tracking | Duck mask only |
| Occluded (active grasp) | Duck mask **minus** hand mask (rotation overridden by MediaPipe) |
| Occluded (post-release) | Duck mask only (rotation clipped/averaged) |
| Recovery re-init | Duck mask only |

**Single instance** — only one FoundationPose estimator (`est`) for the duck. The second hand estimator (`est_hand`) was removed when MediaPipe replaced it.

---

## 4. Hand Tracking: MediaPipe Hands

**What:** CPU-based hand landmark detector. Returns 21 2D image landmarks and 21 3D world landmarks per frame. Part of Google MediaPipe 0.10.9.

**Why:** Provides clean palm orientation deltas every frame on CPU, and pixel-precise fingertip/bone positions for grasp detection. Replaces the FoundationPose hand estimator which was slow and noisy.

**Two uses:**

### 4a. Palm Rotation Delta
- Palm 3D frame built from 4 world landmarks: wrist (0), index MCP (5), middle MCP (9), pinky MCP (17)
- Axes: `y = middle_mcp - wrist`, `x = pinky_mcp - index_mcp`, `z = cross(x,y)` (re-orthogonalised)
- Frame-to-frame delta: `R_delta = R_new @ R_prev.T`
- Applied to duck rotation during active grasp: `new_rot = R_delta @ last_good_duck_rot`
- Then clipped to 2°/frame via `clip_rotation_consistent`

### 4b. Grasp Detection via Mask Overlap (`on_mask`)
- 21 landmark points projected to image pixels and checked against duck mask
- All 23 bone segments (finger bones + palm knuckles) sampled at 15 points each, also checked against duck mask
- Returns `True` if **any** point or segment falls on the duck mask
- `grasp_entered = True` once `on_mask` first returns True
- Release: `on_mask` returns False for `RELEASE_CONSEC=5` consecutive frames

---

## 5. Occlusion Detection

**What:** Pixel-count based detection using the duck's SAM2VP mask.

**Logic:**
- `frame0_mask_area` = duck mask pixel count at frame 0 (baseline for a fully visible duck)
- `occluded = mask_area < 0.90 * frame0_mask_area`
- `recovered = mask_area >= 0.95 * frame0_mask_area`
- Hysteresis gap (0.90 lock / 0.95 unlock) prevents flickering around the threshold

---

## 6. One-Shot Grasp State Machine

**What:** Controls which rotation source is used during occlusion. Fires exactly once per video.

**States:**
| State variable | Meaning |
|---|---|
| `grasp_entered` | Hand landmarks have been detected on duck mask (confirmed contact) |
| `hand_released` | Hand has been off mask for 5+ consecutive frames; waiting for mask recovery |
| `grasp_done` | Re-init after release complete; hand logic permanently disabled |
| `off_mask_count` | Counter of consecutive frames with hand off mask (resets if hand returns) |

**Per-frame logic:**
```
if not grasp_done and not hand_released:
    if hand_on_mask:
        grasp_entered = True
        off_mask_count = 0
    elif grasp_entered:
        off_mask_count += 1
    if grasp_entered and off_mask_count >= 5:
        hand_released = True
```

**After mask recovery (`mask_area >= 0.95 * frame0`):**
- `binary_search_depth` re-init with duck mask
- If `hand_released`: set `grasp_done = True` → hand logic off forever

---

## 7. Rotation Source Per State

| Condition | Rotation source | Clip |
|---|---|---|
| Not occluded | FoundationPose `track_one_new` (full 6DoF) | None |
| Occluded, `grasp_entered`, `hand_rot_delta` available | MediaPipe palm delta applied to `last_good_duck_rot` | 2°/frame |
| Occluded, no grasp / no MediaPipe data | 3-frame averaged raw FoundationPose rotation | 5°/frame |
| Occluded, `hand_released` or `grasp_done` | 3-frame averaged raw FoundationPose rotation | 5°/frame |
| Partially recovered (`was_occluded`, mask not yet at 0.95) | 3-frame averaged raw FoundationPose rotation | 5°/frame |
| Recovery re-init frame | `binary_search_depth` result | None |

---

## 8. Rotation Filtering

### 3-Frame Averaging
- Raw rotation matrices from `track_one_new` buffered in `raw_rot_buffer` (max 3)
- Average computed via **rotvec mean** relative to `last_good_duck_rot`:
  ```python
  vecs = [ScipyR.from_matrix(R_ref.T @ R).as_rotvec() for R in R_list]
  avg = R_ref @ ScipyR.from_rotvec(mean(vecs)).as_matrix()
  ```
- Cancels random frame-to-frame noise; opposing spikes cancel in rotvec space

### Clip + Consistency Check (`clip_rotation_consistent`)
- Computes rotation delta as rotvec between `R_prev` and `R_new`
- **Consistency check**: compares new delta axis against rolling mean of last 3 rotvecs
  - If `dot(new_rotvec, trend) < 0.7` → sudden axis change → allow only 1°/frame
  - Otherwise → allow `max_deg` (2° for grasp, 5° for post-release)
- Clips by scaling the rotvec magnitude: `rotvec * (max_rad / angle)`
- Updates `vel_history` for next frame's consistency check

---

## 9. Recovery Re-init

**What:** Re-initializes duck pose after occlusion clears using `binary_search_depth`.

**Why:** After a long occlusion (especially with hand rotation applied), the tracked pose may have accumulated drift. Re-init using the clean duck mask snaps back to ground truth.

**When it fires:**
- `was_occluded=True` AND `mask_area >= 0.95 * frame0_mask_area`
- On re-init: clears `raw_rot_buffer` and `vel_history` (fresh start)
- If `hand_released` was True: sets `grasp_done=True` after re-init

---

## Summary Table

| Component | Purpose | Mask used | Runs on |
|---|---|---|---|
| Video Depth Anything | Metric depth per frame | — | GPU (offline) |
| SAM2 Video Predictor | Duck + hand masks | — | GPU (offline) |
| FoundationPose | 6DoF duck pose tracking | Duck mask | GPU (per frame) |
| MediaPipe Hands | Palm delta + grasp detection | Duck mask (overlap check) | CPU (per frame) |
| Occlusion detector | Mask area threshold | Duck mask | CPU |
| One-shot grasp state machine | Gate hand rotation | Duck mask (via on_mask) | CPU |
| 3-frame rotation averaging | Smooth raw tracker noise | — | CPU |
| Clip + consistency filter | Cap per-frame rotation change | — | CPU |
| Recovery re-init | Snap back after occlusion | Duck mask | GPU (on trigger) |
