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

        # 2. Convert mesh to .ply. The renderer (utils/renderer.py:126) ALWAYS
        # divides loaded vertices by 1000 (BOP mm→m). So we export the .ply at
        # mm scale (raw / unscaled) — the renderer's /1000 then yields metres.
        # However the models_vertices passed via _StubDataset must be in metres
        # so the bbox-sphere cropping logic works correctly.
        self._models_dir = Path("/work/scratch/hudela/gotrack_models")
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._ply_path = self._models_dir / f"obj_{obj_id:06d}.ply"

        mesh_mm = trimesh.load(mesh_path, force="mesh")  # assume mm, don't scale
        # Heuristic: if extents already <1, mesh is in metres; convert to mm.
        if float(mesh_mm.extents.max()) < 1.0:
            mesh_mm.apply_scale(1000.0)
        print(f"mesh (mm): verts={len(mesh_mm.vertices)} faces={len(mesh_mm.faces)} "
              f"extents={mesh_mm.extents}")
        mesh_mm.export(str(self._ply_path))
        ply_size = self._ply_path.stat().st_size
        print(f"ply written (mm-scale): {self._ply_path} ({ply_size/1e6:.1f} MB)")

        # Metres-scale copy for cropping/bbox logic.
        self._mesh = mesh_mm.copy()
        self._mesh.apply_scale(0.001)

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
        # debug=False skips the visualization path (Pillow 10 broke font.getsize).
        opts = GoTrackOpts()._replace(num_iterations_test=3, debug=False)
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
        self.model.opts = self.model.opts._replace(num_iterations_test=n_iter, debug=False)

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

        # DEBUG: dump the rgbs_template + rgbs_query + flow magnitude for the
        # last iter so we can verify the renderer is producing a visible duck.
        try:
            import cv2 as _cv2
            dbg = Path("/work/scratch/hudela/gotrack_debug")
            dbg.mkdir(parents=True, exist_ok=True)
            gi = inputs.get("gotrack_inputs")
            if gi is not None:
                qry = gi.crop_rgbs[0].cpu().numpy().transpose(1, 2, 0)
                _cv2.imwrite(str(dbg / "query.png"),
                             (qry * 255).clip(0, 255).astype("uint8")[..., ::-1])
                tpls = gi.templates
                # templates is a Collection; rgbs field is [B, 3, H, W]
                if hasattr(tpls, "rgbs"):
                    tpl = tpls.rgbs[0].cpu().numpy().transpose(1, 2, 0)
                    _cv2.imwrite(str(dbg / "template.png"),
                                 (tpl * 255).clip(0, 255).astype("uint8")[..., ::-1])
                    print(f"DEBUG: template max={tpl.max():.3f} mean={tpl.mean():.3f}")
                if hasattr(tpls, "masks"):
                    msk = tpls.masks[0].cpu().numpy()
                    print(f"DEBUG: template mask sum={int((msk>0).sum())}px (>0)")
                print(f"DEBUG template + query dumped to {dbg}")
        except Exception as e:
            print(f"DEBUG dump skipped: {e}")

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
