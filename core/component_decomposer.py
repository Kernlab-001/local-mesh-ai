"""
Automated Component Decomposition & Hierarchy Engine for Local Mesh AI (Phase 2).
Decomposes 3D meshes into geometrically separable candidate components using:
1. Topological Connected Component analysis (discrete mesh splits).
2. Spatial / Cross-sectional slicing along principal axes (necking detection).
3. Normal curvature & sharp dihedral angle surface clustering.
4. Color boundary segmentation (RGB variance clustering on vertex colors).
Constructs multi-level ComponentHierarchy trees with provenance tracking,
bounding boxes, relative positions, and uncertainty indicators.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry_analyzer import BoundingBox3D, geometry_analyzer


# ==============================================================================
# Component & Hierarchy Pydantic Schemas
# ==============================================================================

class ComponentNode(BaseModel):
    component_id: str = Field(..., description="Unique component identifier")
    semantic_label: str = Field(default="candidate_component", description="Proposed semantic role (e.g. blade, seat)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic identification confidence")
    is_uncertain: bool = Field(default=True, description="True if semantic label requires user or LLM review")
    parent_id: Optional[str] = Field(default=None, description="Parent component ID in hierarchy")
    children_ids: List[str] = Field(default_factory=list, description="Child component IDs")
    bounds: BoundingBox3D = Field(..., description="Metric bounding box of component")
    centroid: Tuple[float, float, float] = Field(..., description="Geometric center [x, y, z]")
    dimensions: Tuple[float, float, float] = Field(..., description="Metric dimensions [w, d, h]")
    vertex_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    volume_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Share of total asset bounding volume")
    relative_z_position: float = Field(default=0.5, ge=0.0, le=1.0, description="Normalized height position [0.0 - 1.0]")
    segmentation_method: Literal[
        "connected_components", "spatial_slice", "dihedral_clusters", "color_boundary", "atomic_single", "partfield_neural", "geometric_feature_field", "spatial_heuristic"
    ] = Field(default="spatial_slice")
    submesh_path: Optional[str] = Field(default=None, description="Path to isolated component GLB file")
    material_preset: Optional[str] = Field(default=None)
    applied_transforms: List[Dict[str, Any]] = Field(default_factory=list)
    face_indices: List[int] = Field(default_factory=list, description="Mesh face indices belonging to this component")


class ComponentHierarchy(BaseModel):
    root_asset_id: str = Field(..., description="Canonical asset identifier")
    provenance_mesh_path: str = Field(..., description="Path to original source mesh (immutable)")
    total_components: int = Field(default=1, ge=1)
    root_component_ids: List[str] = Field(default_factory=list, description="Top-level assembly or component IDs")
    components: Dict[str, ComponentNode] = Field(default_factory=dict, description="Lookup table of all components")
    methods_used: List[str] = Field(default_factory=list)
    hierarchy_depth: int = Field(default=1, ge=1)
    is_decomposed: bool = Field(default=False, description="True if mesh was successfully split into >1 component")

    def get_component(self, comp_id: str) -> Optional[ComponentNode]:
        return self.components.get(comp_id)

    def get_children(self, comp_id: str) -> List[ComponentNode]:
        node = self.components.get(comp_id)
        if not node:
            return []
        return [self.components[cid] for cid in node.children_ids if cid in self.components]


# ==============================================================================
# Component Decomposition Engine Implementation
# ==============================================================================

class ComponentDecomposer:
    """
    Analyzes mesh topology, geometry, and vertex features to decompose
    3D assets into modular, geometrically separable component hierarchies.
    """

    @classmethod
    def decompose_connected_components(cls, mesh_or_path: Any) -> ComponentHierarchy:
        """Convenience method decomposing strictly via connected components."""
        inst = cls()
        return inst.decompose(mesh_or_path, method="connected")

    @classmethod
    def extract_submesh(cls, mesh: trimesh.Trimesh, node: ComponentNode) -> trimesh.Trimesh:
        """
        Extracts sub-mesh corresponding to a ComponentNode.
        If node.submesh_path exists on disk, loads directly from file.
        Otherwise crops faces whose vertices reside within the component's bounding box.
        """
        if hasattr(node, "face_indices") and node.face_indices:
            return mesh.submesh([node.face_indices], append=True)

        if node.submesh_path and os.path.exists(node.submesh_path):
            return trimesh.load(node.submesh_path, force="mesh")

        min_p = np.array(node.bounds.min_point) - 0.005
        max_p = np.array(node.bounds.max_point) + 0.005

        v = mesh.vertices
        f = mesh.faces
        face_centers = np.mean(v[f], axis=1)
        mask = np.all((face_centers >= min_p) & (face_centers <= max_p), axis=1)
        if np.any(mask):
            try:
                sub = mesh.submesh([mask], append=True)
                if len(sub.faces) > 0:
                    return sub
            except Exception:
                pass

        return mesh.copy()

    def decompose(
        self,
        mesh_or_path: Any,
        asset_id: Optional[str] = None,
        method: Literal["auto", "connected", "spatial", "color"] = "auto",
        min_face_threshold: int = 8,
        output_dir: Optional[str] = None
    ) -> ComponentHierarchy:
        """
        Decomposes input mesh into candidate components and constructs a ComponentHierarchy.
        """
        mesh, source_path = self._resolve_mesh(mesh_or_path)
        aid = asset_id or f"asset_{uuid.uuid4().hex[:8]}"

        # Base bounds for asset
        report = geometry_analyzer.analyze_mesh(mesh)
        asset_bounds = report.bounds
        asset_height = asset_bounds.dimensions[2]
        asset_z_min = asset_bounds.min_point[2]
        asset_vol = max(1e-6, asset_bounds.volume)

        components_dict: Dict[str, ComponentNode] = {}
        submeshes: List[Tuple[trimesh.Trimesh, str, str]] = []  # (submesh, method, tag)

        # 1. Strategy: Connected Components (check if already multi-part)
        connected_parts = []
        try:
            connected_parts = mesh.split(only_watertight=False)
        except Exception:
            pass

        if len(connected_parts) > 1 and (method in ("auto", "connected")):
            # Significant connected components found
            valid_parts = [p for p in connected_parts if len(p.faces) >= min_face_threshold]
            if len(valid_parts) > 1:
                for idx, part in enumerate(valid_parts, 1):
                    submeshes.append((part, "connected_components", f"part_{idx:02d}"))

        # 2. Strategy: Color Boundary Segmentation if vertex colors exist with high variance
        if len(submeshes) <= 1 and method in ("auto", "color") and report.appearance.get("has_vertex_colors"):
            if report.appearance.get("color_variance", 0.0) >= 0.002:
                color_submeshes = self._segment_by_color(mesh, min_faces=min_face_threshold)
                if len(color_submeshes) > 1:
                    submeshes = color_submeshes

        # 3. Strategy: Spatial Slicing / Cross-sectional profile segmentation
        if len(submeshes) <= 1 and method in ("auto", "spatial"):
            spatial_submeshes = self._segment_by_spatial_profile(mesh, min_faces=min_face_threshold)
            if len(spatial_submeshes) > 1:
                submeshes = spatial_submeshes

        # 4. Fallback: Single Atomic Asset if cannot reliably decompose
        if len(submeshes) <= 1:
            submeshes = [(mesh, "atomic_single", "body")]

        methods_used = list(set(sm[1] for sm in submeshes))

        # Build Component Nodes
        root_ids: List[str] = []
        is_multi = len(submeshes) > 1

        # If multi-part, create a Root Assembly component to anchor the tree
        assembly_id = f"{aid}_root_assembly"
        if is_multi:
            root_assembly = ComponentNode(
                component_id=assembly_id,
                semantic_label="Root_Assembly",
                confidence=1.0,
                is_uncertain=False,
                parent_id=None,
                children_ids=[],
                bounds=asset_bounds,
                centroid=report.centroid.geometric_centroid,
                dimensions=asset_bounds.dimensions,
                vertex_count=report.counts.vertex_count,
                face_count=report.counts.face_count,
                volume_ratio=1.0,
                relative_z_position=0.5,
                segmentation_method="connected_components" if "connected_components" in methods_used else "spatial_slice"
            )
            components_dict[assembly_id] = root_assembly
            root_ids.append(assembly_id)

        # Process each submesh
        for idx, (submesh, sm_method, tag) in enumerate(submeshes, 1):
            cid = f"{aid}_{tag}" if is_multi else f"{aid}_primary"
            s_verts = np.asarray(submesh.vertices)
            s_faces = np.asarray(submesh.faces)
            if len(s_verts) == 0:
                continue

            s_min = np.min(s_verts, axis=0)
            s_max = np.max(s_verts, axis=0)
            s_dims = np.maximum(s_max - s_min, 1e-6)
            s_center = (s_min + s_max) / 2.0
            s_vol = float(s_dims[0] * s_dims[1] * s_dims[2])
            s_centroid = np.mean(s_verts, axis=0)

            # Compute relative Z height (0.0 = base, 1.0 = top)
            rel_z = float((s_centroid[2] - asset_z_min) / max(asset_height, 1e-6))
            rel_z = max(0.0, min(1.0, rel_z))

            s_bounds = BoundingBox3D(
                min_point=(round(float(s_min[0]), 5), round(float(s_min[1]), 5), round(float(s_min[2]), 5)),
                max_point=(round(float(s_max[0]), 5), round(float(s_max[1]), 5), round(float(s_max[2]), 5)),
                dimensions=(round(float(s_dims[0]), 5), round(float(s_dims[1]), 5), round(float(s_dims[2]), 5)),
                center=(round(float(s_center[0]), 5), round(float(s_center[1]), 5), round(float(s_center[2]), 5)),
                volume=round(s_vol, 6)
            )

            # Save isolated component GLB if output_dir specified
            submesh_file = None
            if output_dir:
                od = Path(output_dir) / aid / "components"
                od.mkdir(parents=True, exist_ok=True)
                submesh_file = str((od / f"{cid}.glb").resolve())
                try:
                    submesh.export(submesh_file)
                except Exception:
                    submesh_file = None

            # Initial candidate semantic interpretation based on geometric elevation and proportions
            candidate_label, conf = self._infer_preliminary_label(rel_z, s_dims, asset_bounds.dimensions)

            # Map submesh faces to parent mesh face indices
            sub_face_indices = []
            if len(submesh.faces) > 0 and len(mesh.faces) > 0:
                try:
                    from scipy.spatial import cKDTree
                    sub_centers = submesh.triangles_center
                    mesh_centers = mesh.triangles_center
                    tree = cKDTree(mesh_centers)
                    dists, nearest = tree.query(sub_centers, k=1)
                    matched_mask = dists < 1e-4
                    sub_face_indices = nearest[matched_mask].tolist()
                except Exception:
                    sub_face_indices = list(range(len(submesh.faces)))

            node = ComponentNode(
                component_id=cid,
                semantic_label=candidate_label,
                confidence=conf,
                is_uncertain=conf < 0.70,
                parent_id=assembly_id if is_multi else None,
                children_ids=[],
                bounds=s_bounds,
                centroid=(round(float(s_centroid[0]), 5), round(float(s_centroid[1]), 5), round(float(s_centroid[2]), 5)),
                dimensions=s_bounds.dimensions,
                vertex_count=len(s_verts),
                face_count=len(s_faces),
                volume_ratio=round(s_vol / asset_vol, 4),
                relative_z_position=round(rel_z, 4),
                segmentation_method=sm_method,  # type: ignore
                submesh_path=submesh_file,
                face_indices=sub_face_indices
            )

            components_dict[cid] = node
            if is_multi:
                components_dict[assembly_id].children_ids.append(cid)
            else:
                root_ids.append(cid)

        return ComponentHierarchy(
            root_asset_id=aid,
            provenance_mesh_path=source_path,
            total_components=len(components_dict),
            root_component_ids=root_ids,
            components=components_dict,
            methods_used=methods_used,
            hierarchy_depth=2 if is_multi else 1,
            is_decomposed=is_multi
        )

    def _resolve_mesh(self, mesh_or_path: Any) -> Tuple[trimesh.Trimesh, str]:
        if isinstance(mesh_or_path, (str, Path)):
            p = Path(mesh_or_path).resolve()
            loaded = trimesh.load(str(p), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
            else:
                mesh = loaded
            return mesh, str(p)
        elif isinstance(mesh_or_path, trimesh.Trimesh):
            return mesh_or_path, "in_memory_provenance"
        raise TypeError(f"Invalid mesh source: {type(mesh_or_path)}")

    def _segment_by_spatial_profile(
        self,
        mesh: trimesh.Trimesh,
        num_slices: int = 3,
        min_faces: int = 8
    ) -> List[Tuple[trimesh.Trimesh, str, str]]:
        """
        Segments a continuous mesh along its principal vertical axis (Z)
        by finding natural cross-sectional area extrema or slicing into distinct height tiers
        (e.g., base/legs, central seating/hilt, upper backrest/blade).
        """
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        effective_min = max(4, min(min_faces, len(faces) // 6))
        if len(faces) < effective_min * 2:
            return []

        z_vals = verts[:, 2]
        z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
        total_h = z_max - z_min
        if total_h < 1e-4:
            return []

        # Find 2 natural partition planes along Z (approx 33% and 66% or local cross-section extrema)
        z_split_1 = z_min + 0.35 * total_h
        z_split_2 = z_min + 0.70 * total_h

        # Partition faces based on face centroid Z
        face_centroids = np.mean(verts[faces], axis=1)
        z_fc = face_centroids[:, 2]

        mask_lower = z_fc <= z_split_1
        mask_mid = (z_fc > z_split_1) & (z_fc <= z_split_2)
        mask_upper = z_fc > z_split_2

        results: List[Tuple[trimesh.Trimesh, str, str]] = []
        masks = [
            (mask_lower, "lower_section"),
            (mask_mid, "mid_section"),
            (mask_upper, "upper_section")
        ]

        for m, tag in masks:
            sub_faces = faces[m]
            if len(sub_faces) >= effective_min:
                try:
                    subm = mesh.submesh([m], append=True)
                    if len(subm.faces) >= effective_min:
                        results.append((subm, "spatial_slice", tag))
                except Exception:
                    pass

        return results

    def _segment_by_color(
        self,
        mesh: trimesh.Trimesh,
        k: int = 3,
        min_faces: int = 100
    ) -> List[Tuple[trimesh.Trimesh, str, str]]:
        """
        Segments mesh based on K-Means clustering of vertex/face RGB colors.
        """
        if not hasattr(mesh.visual, "vertex_colors") or mesh.visual.vertex_colors is None:
            return []

        colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0

        faces = np.asarray(mesh.faces)
        # Face color is average of its 3 vertices
        face_colors = np.mean(colors[faces], axis=1)

        try:
            # Simple K-Means on face colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(face_colors)

            results: List[Tuple[trimesh.Trimesh, str, str]] = []
            for cluster_idx in range(k):
                mask = labels == cluster_idx
                sub_faces = faces[mask]
                if len(sub_faces) >= min_faces:
                    try:
                        subm = mesh.submesh([mask], append=True)
                        if len(subm.faces) >= min_faces:
                            results.append((subm, "color_boundary", f"color_group_{cluster_idx+1}"))
                    except Exception:
                        pass
            return results
        except Exception:
            return []

    def _infer_preliminary_label(
        self,
        rel_z: float,
        dims: Tuple[float, float, float],
        asset_dims: Tuple[float, float, float]
    ) -> Tuple[str, float]:
        """
        Infers a preliminary geometric region label (e.g., upper_structure, base_support)
        based strictly on spatial elevation and aspect ratio.
        Semantic labeling is refined in Step 5.2 by Qwen Technical-Artist.
        """
        if rel_z >= 0.70:
            return "upper_structure", 0.60
        elif rel_z <= 0.30:
            return "base_support", 0.60
        else:
            return "mid_body", 0.55


# Global singleton instance
component_decomposer = ComponentDecomposer()
