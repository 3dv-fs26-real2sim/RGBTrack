"""
GoTrack 6D pose refiner — wrapper around facebookresearch/gotrack.

API:
    refiner = GoTrackRefiner(
        ckpt_path = "/path/to/gotrack_checkpoint.pt",
        mesh_path = "/path/to/duck.obj",
        gotrack_root = "/work/courses/3dv/team22/gotrack",
        obj_id = 1,
        device = "cuda",
    )
    refined_pose = refiner.refine(rgb_uint8, K, init_pose_4x4, n_iter=3)

`refine()` takes a single frame and returns a 4x4 cam_from_model pose.
Internally GoTrack expects BOP-style data structures (Collection of objects,
images, cameras), so we build a tiny stub dataset and the required Collections
on the fly.

This file is a skeleton. The first time it's run end-to-end with a real
checkpoint, expect to fix small mismatches in the Camera struct and template
mesh path. The TODOs flag where to look.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh


class GoTrackRefiner:
    def __init__(
        self,
        ckpt_path: str,
        mesh_path: str,
        gotrack_root: str = "/work/courses/3dv/team22/gotrack",
        obj_id: int = 1,
        device: str = "cuda",
    ) -> None:
        self.ckpt_path = ckpt_path
        self.mesh_path = mesh_path
        self.gotrack_root = gotrack_root
        self.obj_id = obj_id
        self.device = device

        # 1. Make GoTrack importable.
        sys.path.insert(0, gotrack_root)
        sys.path.insert(0, str(Path(gotrack_root) / "external" / "bop_toolkit"))
        sys.path.insert(0, str(Path(gotrack_root) / "external" / "dinov2"))

        # 2. Convert mesh to .ply at the BOP-style template path.
        self._models_dir = Path("/tmp/gotrack_models")
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._ply_path = self._models_dir / f"obj_{obj_id:06d}.ply"
        self._mesh = trimesh.load(mesh_path, force="mesh")
        # Heuristic: if mesh extents are large (>1m), assume mm and scale.
        if float(self._mesh.extents.max()) > 1.0:
            self._mesh.apply_scale(0.001)
        print(f"mesh extents (m): {self._mesh.extents}")
        self._mesh.export(str(self._ply_path))

        # 3. Build minimal stub dataset for set_renderer().
        self._stub_dataset = _StubDataset(
            models_dir=self._models_dir,
            mesh=self._mesh,
            obj_id=obj_id,
        )

        # 4. Instantiate model + load checkpoint.
        from model.gotrack import GoTrack
        from model.config import GoTrackOpts
        from utils import net_util

        # GoTrackOpts is a NamedTuple — immutable, use _replace.
        opts = GoTrackOpts()._replace(num_iterations_test=3)
        self.model = GoTrack(opts=opts).to(device)
        self.model.eval()
        net_util.load_checkpoint(
            model=self.model,
            checkpoint_path=ckpt_path,
            checkpoint_key="model_state_dict",
            prefix="models.1.",
        )

        # 5. Set the renderer (GoTrack uses pyrender).
        self.model.set_renderer(self._stub_dataset)

        # 6. Result_dir is asserted by forward_pipeline; satisfy it with a tmp dir.
        self.model.result_dir = Path("/tmp/gotrack_results")
        self.model.result_dir.mkdir(parents=True, exist_ok=True)
        self.model.result_file_name = "smoke_test"

    @torch.no_grad()
    def refine(
        self,
        rgb_uint8: np.ndarray,         # H, W, 3 uint8
        K: np.ndarray,                 # 3x3 intrinsics
        init_pose_4x4: np.ndarray,     # 4x4 cam_from_model
        n_iter: int = 3,
    ) -> np.ndarray:
        """Refine a single-frame init pose; returns refined 4x4 cam_from_model."""
        from utils import structs

        # opts is a NamedTuple — rebind to override n_iter for this call.
        self.model.opts = self.model.opts._replace(num_iterations_test=n_iter)

        H, W = rgb_uint8.shape[:2]

        # Build images Collection.
        images = structs.Collection(
            bitmaps=torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0)
                    .permute(2, 0, 1).unsqueeze(0).to(self.device),       # 1,3,H,W
            cameras=[_build_camera(K, W, H)],                              # list of len 1
            scene_ids=torch.tensor([0], dtype=torch.long),
            im_ids=torch.tensor([0], dtype=torch.long),
            times=torch.tensor([0.0]),
        )

        # Build objects Collection (single object referencing frame 0).
        T = torch.from_numpy(init_pose_4x4.astype(np.float32)).unsqueeze(0).to(self.device)
        objects = structs.Collection(
            labels=torch.tensor([self.obj_id], dtype=torch.long),
            frame_ids=torch.tensor([0], dtype=torch.long),
            poses_cam_from_model=T,
            poses_world_from_model=T.clone(),     # world = cam if no extrinsic
            pose_scores=torch.zeros(1),
        )

        inputs = {"images": images, "objects": objects}
        outputs = self.model.forward_pipeline(inputs, batch_idx=0)

        refined = outputs["objects"].poses_cam_from_model[0].cpu().numpy()
        return refined.astype(np.float64)


class _StubDataset:
    """Minimal dataset stub satisfying GoTrack.set_renderer requirements."""
    def __init__(self, models_dir: Path, mesh: trimesh.Trimesh, obj_id: int) -> None:
        self.dp_model = {"model_tpath": str(models_dir / "obj_{obj_id:06d}.ply")}
        # numpy (not torch) — gotrack's approximate_bounding_sphere uses .max(axis=0)
        # which on torch returns a NamedTuple, not just values.
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        self.models_vertices = {obj_id: verts}
        self.models = {obj_id: {"diameter": float(np.linalg.norm(mesh.extents))}}


def _build_camera(K: np.ndarray, W: int, H: int):
    """Construct a GoTrack PinholePlaneCameraModel from a 3×3 K + (W, H)."""
    from utils.structs import PinholePlaneCameraModel
    return PinholePlaneCameraModel(
        width=W,
        height=H,
        f=(float(K[0, 0]), float(K[1, 1])),
        c=(float(K[0, 2]), float(K[1, 2])),
        T_world_from_eye=np.eye(4, dtype=np.float64),
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--rgb", required=True)
    p.add_argument("--init_pose", required=True, help="4x4 .txt file")
    p.add_argument("--cam_K", required=True, help="3x3 .txt file")
    args = p.parse_args()

    import cv2
    rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
    K = np.loadtxt(args.cam_K).reshape(3, 3)
    init = np.loadtxt(args.init_pose).reshape(4, 4)

    refiner = GoTrackRefiner(ckpt_path=args.ckpt, mesh_path=args.mesh)
    refined = refiner.refine(rgb, K, init, n_iter=3)
    print("Initial pose:\n", init)
    print("Refined pose:\n", refined)
