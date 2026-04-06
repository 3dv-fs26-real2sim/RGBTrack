"""
MediaPipe Hands wrapper for palm orientation tracking and grasp detection.

- update(rgb): returns frame-to-frame palm rotation delta (3x3) or None
- on_mask(mask): returns True if any landmark or bone segment overlaps the mask
"""
import numpy as np
import mediapipe as mp

# Landmark indices
WRIST      = 0
INDEX_MCP  = 5
MIDDLE_MCP = 9
PINKY_MCP  = 17

# Finger bone segments (pairs of landmark indices)
BONES = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm knuckles
]

LINE_SAMPLES = 15  # points sampled per bone segment for mask overlap


def _palm_rotation(world_landmarks):
    """3x3 rotation matrix from MediaPipe world landmarks."""
    lm = world_landmarks.landmark
    wrist      = np.array([lm[WRIST].x,      lm[WRIST].y,      lm[WRIST].z])
    index_mcp  = np.array([lm[INDEX_MCP].x,  lm[INDEX_MCP].y,  lm[INDEX_MCP].z])
    middle_mcp = np.array([lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y, lm[MIDDLE_MCP].z])
    pinky_mcp  = np.array([lm[PINKY_MCP].x,  lm[PINKY_MCP].y,  lm[PINKY_MCP].z])

    y = middle_mcp - wrist
    y /= np.linalg.norm(y) + 1e-9
    x = pinky_mcp - index_mcp
    x /= np.linalg.norm(x) + 1e-9
    z = np.cross(x, y)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-9
    return np.stack([x, y, z], axis=1)


class MediaPipeHandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._R_last = None
        self._last_results = None

    def update(self, rgb_frame):
        """
        Process one RGB frame.
        Returns rotation delta (3x3) from last frame, or None if hand not detected.
        """
        self._last_results = self._hands.process(rgb_frame)

        if not self._last_results.multi_hand_world_landmarks:
            self._R_last = None
            return None

        R_new = _palm_rotation(self._last_results.multi_hand_world_landmarks[0])

        if self._R_last is None:
            self._R_last = R_new
            return None

        delta = R_new @ self._R_last.T
        self._R_last = R_new
        return delta

    def on_mask(self, mask):
        """
        Returns True if any landmark point or bone segment overlaps the duck mask.
        mask: HxW uint8 (>0 = duck).
        """
        if self._last_results is None or not self._last_results.multi_hand_landmarks:
            return False

        h, w = mask.shape[:2]
        lm_img = self._last_results.multi_hand_landmarks[0].landmark

        def px(lm):
            return int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1))

        # Check individual landmark points
        for lm in lm_img:
            x, y = px(lm)
            if mask[y, x] > 0:
                return True

        # Check bone segments (sampled points)
        for a_idx, b_idx in BONES:
            ax, ay = px(lm_img[a_idx])
            bx, by = px(lm_img[b_idx])
            for t in np.linspace(0, 1, LINE_SAMPLES):
                sx = int(ax + t * (bx - ax))
                sy = int(ay + t * (by - ay))
                sx = np.clip(sx, 0, w - 1)
                sy = np.clip(sy, 0, h - 1)
                if mask[sy, sx] > 0:
                    return True

        return False

    def reset(self):
        self._R_last = None

    def close(self):
        self._hands.close()
