import torch
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor

class SAM2Wrapper:
    def __init__(self, checkpoint_path, config_path, device="cuda"):
        """
        Initializes the SAM2 real-time predictor.
        """
        self.device = device
        # Load the SAM-2 model
        self.predictor = build_sam2_camera_predictor(config_path, checkpoint_path, device=device)
        self.predictor_state = None

    def initialize(self, first_frame, first_mask):
        """
        Initializes the tracking state with the first frame and the initial object mask.
        (Mirrors how XMem starts tracking).
        """
        # SAM-2 requires the frame to be a numpy array (H, W, 3) in RGB format
        self.predictor.load_first_frame(first_frame)
        
        # Add the target mask to the predictor
        # ann_obj_id determines which object we are tracking (useful if multi-object later)
        ann_obj_id = 1
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            frame_idx=0,
            obj_id=ann_obj_id,
            mask=first_mask
        )
        return out_mask_logits

    def track(self, next_frame):
        """
        Processes the next frame and returns the predicted mask.
        """
        # Step the predictor forward
        out_obj_ids, out_mask_logits = self.predictor.track(next_frame)
        
        # Convert logits back to a binary mask (H, W)
        # Assuming single object tracking for now (index 0)
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
        
        return mask