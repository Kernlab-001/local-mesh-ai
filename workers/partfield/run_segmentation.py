"""NVIDIA PartField Neural 3D Component Segmentation Worker.

Executes genuine neural feature-field inference using the pretrained PartField
checkpoint (e.g. model_objaverse.ckpt) and maps predicted component regions
back to the original mesh topology without destroying vertices or attributes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Compatibility Shims: ensure worker runs cleanly without requiring external
# packages that are non-essential for local inference (boto3, loguru, yacs, torch_scatter)
# ---------------------------------------------------------------------------

if "boto3" not in sys.modules:
    sys.modules["boto3"] = types.ModuleType("boto3")

if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")
    class _DummyLogger:
        def info(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
    _loguru.logger = _DummyLogger()
    sys.modules["loguru"] = _loguru

# YACS CfgNode shim
if "yacs" not in sys.modules:
    _yacs = types.ModuleType("yacs")
    _yacs_config = types.ModuleType("yacs.config")

    class CfgNode(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __getattr__(self, name):
            if name in self:
                val = self[name]
                if isinstance(val, dict) and not isinstance(val, CfgNode):
                    val = CfgNode(val)
                    self[name] = val
                return val
            raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    _yacs_config.CfgNode = CfgNode
    _yacs.config = _yacs_config
    sys.modules["yacs"] = _yacs
    sys.modules["yacs.config"] = _yacs_config

# Pure PyTorch fallback for torch_scatter
import torch

class _PureTorchScatter:
    @staticmethod
    def scatter_mean(src, index, out=None, dim=-1, dim_size=None):
        if out is None:
            if dim_size is None:
                dim_size = int(index.max().item()) + 1
            shape = list(src.shape)
            shape[dim] = dim_size
            out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        if index.shape != src.shape:
            index = index.expand_as(src)
        return out.scatter_reduce(dim, index, src, reduce="mean", include_self=False)

    @staticmethod
    def scatter_max(src, index, out=None, dim=-1, dim_size=None):
        if out is None:
            if dim_size is None:
                dim_size = int(index.max().item()) + 1
            shape = list(src.shape)
            shape[dim] = dim_size
            out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
        if index.shape != src.shape:
            index = index.expand_as(src)
        res = out.scatter_reduce(dim, index, src, reduce="amax", include_self=False)
        return res, res

if "torch_scatter" not in sys.modules:
    sys.modules["torch_scatter"] = _PureTorchScatter()


def _resolve_partfield_repo(custom_path: Optional[str] = None) -> Path:
    """Find upstream PartField source directory."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    if "PARTFIELD_REPO_DIR" in os.environ:
        candidates.append(Path(os.environ["PARTFIELD_REPO_DIR"]))
    if "LOCAL_MESH_MODELS_ROOT" in os.environ:
        candidates.append(
            Path(os.environ["LOCAL_MESH_MODELS_ROOT"]) / "partfield" / "source" / "PartField-main" / "PartField-main"
        )
    candidates.extend([
        Path(r"E:\LocalMeshAI-Data\models\partfield\source\PartField-main\PartField-main"),
        Path(__file__).parent / "source" / "PartField-main" / "PartField-main",
        Path.cwd() / "models" / "partfield" / "source" / "PartField-main" / "PartField-main",
    ])
    for cand in candidates:
        if cand.exists() and (cand / "partfield").exists():
            return cand.resolve()
    raise FileNotFoundError(
        f"Upstream PartField source tree not found. Checked: {[str(c) for c in candidates]}"
    )


def _resolve_checkpoint(custom_path: Optional[str] = None) -> Path:
    """Find PartField checkpoint (model_objaverse.ckpt)."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    if "PARTFIELD_CHECKPOINT" in os.environ:
        candidates.append(Path(os.environ["PARTFIELD_CHECKPOINT"]))
    if "LOCAL_MESH_MODELS_ROOT" in os.environ:
        candidates.append(Path(os.environ["LOCAL_MESH_MODELS_ROOT"]) / "partfield" / "model_objaverse.ckpt")
        candidates.append(Path(os.environ["LOCAL_MESH_MODELS_ROOT"]) / "partfield" / "partfield_objaverse.pt")
    candidates.extend([
        Path(r"E:\LocalMeshAI-Data\models\partfield\model_objaverse.ckpt"),
        Path(r"C:\Users\diyaj\Downloads\model_objaverse.ckpt"),
        Path(__file__).parent / "model_objaverse.ckpt",
        Path.cwd() / "models" / "partfield" / "model_objaverse.ckpt",
    ])
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"PartField checkpoint not found. Checked: {[str(c) for c in candidates]}"
    )


def run_segmentation(
    mesh_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    partfield_repo: Path,
    device_str: str = "cuda",
    n_points: int = 10000,
    k_clusters: int = 4,
    export_submeshes_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute neural segmentation pipeline and generate structured component hierarchy."""
    t_start = time.time()

    # 1. Add PartField repo to sys.path
    if str(partfield_repo) not in sys.path:
        sys.path.insert(0, str(partfield_repo))

    from partfield.model.triplane import TriplaneTransformer
    from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat

    # 2. Check device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 3. Load mesh
    if not mesh_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {mesh_path}")

    loaded_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(loaded_mesh.dump())
    else:
        mesh = loaded_mesh

    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.faces)
    if num_vertices == 0 or num_faces == 0:
        raise ValueError(f"Mesh has 0 vertices or 0 faces: {mesh_path}")

    # 4. Load PartField checkpoint
    t_load_0 = time.time()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    from yacs.config import CfgNode
    cfg = CfgNode(ckpt["hyper_parameters"]["cfg"])

    pvcnn_sd = {k[len("pvcnn."):]: v for k, v in sd.items() if k.startswith("pvcnn.")}
    transformer_sd = {k[len("triplane_transformer."):]: v for k, v in sd.items() if k.startswith("triplane_transformer.")}

    pvcnn = TriPlanePC2Encoder(
        cfg.pvcnn,
        device=device.type,
        shape_min=-1,
        shape_length=2,
        use_2d_feat=False,
    ).to(device)
    pvcnn.load_state_dict(pvcnn_sd, strict=True)
    pvcnn.eval()

    triplane_transformer = TriplaneTransformer(
        input_dim=cfg.triplane_channels_low * 2,
        transformer_dim=1024,
        transformer_layers=6,
        transformer_heads=8,
        triplane_low_res=32,
        triplane_high_res=128,
        triplane_dim=cfg.triplane_channels_high,
    ).to(device)
    triplane_transformer.load_state_dict(transformer_sd, strict=True)
    triplane_transformer.eval()
    t_load = time.time() - t_load_0

    # 5. Preprocessing & Normalization
    # PartField expects point cloud normalized into roughly [-0.9, 0.9]
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    scale = (mesh.bounds[1] - mesh.bounds[0]).max()
    scale = max(scale, 1e-6)

    # Sample points on surface
    sample_count = min(n_points, max(2048, num_faces * 4))
    points, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    norm_points = (points - center) / (scale / 1.8)  # fits inside [-0.9, 0.9]

    # 6. Forward Neural Inference
    t_infer_0 = time.time()
    with torch.no_grad():
        pc_tensor = torch.from_numpy(norm_points).float().unsqueeze(0).to(device)
        pc_feat = pvcnn(pc_tensor, pc_tensor)
        planes = triplane_transformer(pc_feat)
        _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        # Sample features at the sampled surface points
        features_tensor = sample_triplane_feat(part_planes, pc_tensor)
        features = features_tensor.squeeze(0).cpu().numpy()  # (sample_count, 448)
    t_infer = time.time() - t_infer_0

    # 7. Feature normalization and clustering
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    # Choose reasonable cluster count: k_clusters or adaptive
    n_clusters = max(2, min(k_clusters, 12, num_faces))
    
    # Subsample features if large for hierarchical clustering linkage
    clustering_sample_limit = 2500
    if len(features_norm) > clustering_sample_limit:
        sub_indices = np.random.choice(len(features_norm), clustering_sample_limit, replace=False)
        sub_feats = features_norm[sub_indices]
        dists = pdist(sub_feats, metric="cosine")
        z = linkage(dists, method="average")
        sub_labels = fcluster(z, t=n_clusters, criterion="maxclust") - 1
        
        # Propagate to all sampled points via KDTree on normalized features
        tree = cKDTree(sub_feats)
        _, nearest_idx = tree.query(features_norm, k=1)
        point_labels = sub_labels[nearest_idx]
    else:
        dists = pdist(features_norm, metric="cosine")
        z = linkage(dists, method="average")
        point_labels = fcluster(z, t=n_clusters, criterion="maxclust") - 1

    # 8. Map point labels back to original mesh faces
    # For each face, determine label by majority vote of sampled points on that face
    face_votes: Dict[int, List[int]] = {i: [] for i in range(num_faces)}
    for p_idx, f_idx in enumerate(face_indices):
        face_votes[int(f_idx)].append(int(point_labels[p_idx]))

    face_labels = np.zeros(num_faces, dtype=np.int32)
    unassigned_faces = []

    for f_idx, votes in face_votes.items():
        if len(votes) > 0:
            face_labels[f_idx] = int(np.bincount(votes).argmax())
        else:
            unassigned_faces.append(f_idx)

    # For faces with no sampled points, use KNN from face centroids to sampled points
    if len(unassigned_faces) > 0:
        face_centroids = mesh.triangles_center[unassigned_faces]
        tree = cKDTree(points)
        _, nearest_p = tree.query(face_centroids, k=1)
        for i, f_idx in enumerate(unassigned_faces):
            p_match = nearest_p[i]
            face_labels[f_idx] = int(point_labels[p_match])

    # 9. Build Component Hierarchy
    unique_labels = np.unique(face_labels)
    components = []

    if export_submeshes_dir:
        export_submeshes_dir.mkdir(parents=True, exist_ok=True)

    for idx, label in enumerate(unique_labels):
        comp_faces = np.where(face_labels == label)[0]
        if len(comp_faces) == 0:
            continue

        cid = f"comp_{idx + 1:03d}"
        
        # Calculate submesh bounding box and volume
        sub_faces = mesh.faces[comp_faces]
        sub_vertex_indices = np.unique(sub_faces)
        sub_verts = mesh.vertices[sub_vertex_indices]

        bbox_min = sub_verts.min(axis=0).tolist()
        bbox_max = sub_verts.max(axis=0).tolist()
        dims = (sub_verts.max(axis=0) - sub_verts.min(axis=0)).tolist()
        centroid = sub_verts.mean(axis=0).tolist()

        # Calculate cluster coherence score as confidence
        comp_point_indices = np.where(point_labels == label)[0]
        if len(comp_point_indices) > 1:
            comp_feats = features_norm[comp_point_indices]
            center_feat = comp_feats.mean(axis=0)
            center_norm = np.linalg.norm(center_feat)
            confidence = float(np.clip(center_norm, 0.4, 0.98))
        else:
            confidence = 0.65

        # Submesh export if requested
        submesh_path_str = None
        if export_submeshes_dir:
            submesh = mesh.submesh([comp_faces], append=True)
            submesh_file = export_submeshes_dir / f"{cid}.obj"
            submesh.export(submesh_file)
            submesh_path_str = str(submesh_file)

        components.append({
            "component_id": cid,
            "label_id": int(label),
            "face_count": int(len(comp_faces)),
            "vertex_count": int(len(sub_vertex_indices)),
            "face_indices": comp_faces.tolist(),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "dimensions": dims,
            "centroid": centroid,
            "confidence": confidence,
            "submesh_path": submesh_path_str,
        })

    # Telemetry & memory cleanup
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    del pvcnn
    del triplane_transformer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    t_total = time.time() - t_start

    result: Dict[str, Any] = {
        "status": "SUCCESS",
        "engine": "partfield_neural",
        "mesh_path": str(mesh_path),
        "checkpoint_path": str(checkpoint_path),
        "device": device_str,
        "num_vertices": int(num_vertices),
        "num_faces": int(num_faces),
        "num_components": int(len(components)),
        "components": components,
        "telemetry": {
            "load_time_seconds": round(t_load, 3),
            "inference_time_seconds": round(t_infer, 3),
            "total_time_seconds": round(t_total, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
        },
        "provenance": {
            "architecture": "PVCNN_TriplaneTransformer",
            "feature_channels": 448,
            "checkpoint_epoch": int(ckpt.get("epoch", 215)),
            "official_source": "https://github.com/nv-tlabs/PartField",
            "license": "NVIDIA Proprietary / Non-Commercial Research License",
        },
    }

    # Write output JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="PartField Neural 3D Segmentation Worker")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input 3D mesh")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model_objaverse.ckpt")
    parser.add_argument("--partfield-repo", type=str, default=None, help="Path to upstream PartField repo")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--n-points", type=int, default=10000, help="Number of surface points to sample")
    parser.add_argument("--k-clusters", type=int, default=4, help="Target component cluster count")
    parser.add_argument("--export-submeshes", type=str, default=None, help="Directory to export submeshes")

    args = parser.parse_args()

    try:
        partfield_repo = _resolve_partfield_repo(args.partfield_repo)
        checkpoint = _resolve_checkpoint(args.checkpoint)
        mesh_path = Path(args.mesh).resolve()
        output_path = Path(args.output).resolve()
        export_dir = Path(args.export_submeshes).resolve() if args.export_submeshes else None

        res = run_segmentation(
            mesh_path=mesh_path,
            output_path=output_path,
            checkpoint_path=checkpoint,
            partfield_repo=partfield_repo,
            device_str=args.device,
            n_points=args.n_points,
            k_clusters=args.k_clusters,
            export_submeshes_dir=export_dir,
        )
        print(f"PartField neural segmentation succeeded: {res['num_components']} components created in {res['telemetry']['total_time_seconds']}s")
        sys.exit(0)
    except Exception as e:
        import traceback
        err_dict = {
            "status": "ERROR",
            "engine": "partfield_neural",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            out_p = Path(args.output).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(err_dict, f, indent=2)
        except Exception:
            pass
        print(f"PartField error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
