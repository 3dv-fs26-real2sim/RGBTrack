"""
MediaPipe Hands wrapper for palm orientation tracking.

Computes a stable 3x3 rotation matrix from hand world landmarks each frame,
and exposes the frame-to-frame rotation delta for use as a duck rotation proxy
during occlusion.
"""
import numpy as np
import mediapipe as mp


# MediaPipe landmark indices used for palm frame
WRIST      = 0
INDEX_MCP  = 5
MIDDLE_MCP = 9
PINKY_MCP  = 17


def _palm_rotation(world_landmarks):
    """
    Build a 3x3 rotation matrix from MediaPipe world landmarks.
    Frame: x = index→pinky (side), y = wrist→middle (forward), z = palm normal.
    """
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

    # Re-orthogonalise
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-9

    return np.stack([x, y, z], axis=1)  # columns = axes → 3x3


class MediaPipeHandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._R_last = None

    def update(self, rgb_frame):
        """
        Process one RGB frame. Returns rotation delta (3x3) from last frame,
        or None if hand not detected this frame.
        """
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_world_landmarks:
            self._R_last = None
            return None

        R_new = _palm_rotation(results.multi_hand_world_landmarks[0])

        if self._R_last is None:
            self._R_last = R_new
            return None  # first detection — no delta yet

        delta = R_new @ self._R_last.T
        self._R_last = R_new
        return delta

    def reset(self):
        self._R_last = None

    def close(self):
        self._hands.close()
