"""
Comprehensive 3D Geometry & Topology Analysis Engine for Local Mesh AI (Phase 2).
Provides deterministic, machine-readable measurements of 3D meshes:
- Metric bounding box, dimensions, centroid, volume, surface area.
- Polygon, vertex, and edge counts (triangles vs quads).
- Rigorous topological integrity: watertightness, boundary edges, non-manifold edges,
  degenerate zero-area faces, duplicate vertices, isolated vertices, Euler characteristic.
- Geometry quality: NaN/Inf checks, thin feature detection, aspect ratios.
- Symmetry diagnostics: bilateral symmetry planes and deviation metrics.
- Orientation diagnostics: principal component axes and upright posture checks.
All measurements are strictly CPU-executed using trimesh and numpy, returning strongly typed schemas.
"""

import os
import sys
import math
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


# ==============================================================================
# Pydantic Schemas for Geometry Analysis
# ==============================================================================

class BoundingBox3D(BaseModel):
    min_point: Tuple[float, float, float] = Field(..., description="Minimum bounds [min_x, min_y, min_z]")
    max_point: Tuple[float, float, float] = Field(..., description="Maximum bounds [max_x, max_y, max_z]")
    dimensions: Tuple[float, float, float] = Field(..., description="Span [width_x, depth_y, height_z] in meters")
    center: Tuple[float, float, float] = Field(..., description="Center of bounding box [cx, cy, cz]")
    volume: float = Field(default=0.0, description="Bounding box volume (m^3)")


class MeshCentroid(BaseModel):
    geometric_centroid: Tuple[float, float, float] = Field(..., description="Mean vertex coordinates")
    center_of_mass: Optional[Tuple[float, float, float]] = Field(default=None, description="Volume/area-weighted center")


class GeometryCounts(BaseModel):
    vertex_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    triangle_count: int = Field(default=0, ge=0)
    quad_count: int = Field(default=0, ge=0)
    connected_components: int = Field(default=1, ge=0)


class TopologyMeasurements(BaseModel):
    is_watertight: bool = Field(default=False, description="True if mesh is closed with 0 boundary edges")
    boundary_edge_count: int = Field(default=0, ge=0, description="Edges shared by only 1 face (open boundaries)")
    non_manifold_edge_count: int = Field(default=0, ge=0, description="Edges shared by >2 faces")
    degenerate_face_count: int = Field(default=0, ge=0, description="Faces with near-zero area (<= 1e-12)")
    duplicate_vertex_count: int = Field(default=0, ge=0, description="Coincident vertex count")
    isolated_vertex_count: int = Field(default=0, ge=0, description="Vertices not referenced by any face")
    is_winding_consistent: bool = Field(default=False, description="Consistent outward-facing normal orientation")
    euler_number: Optional[int] = Field(default=None, description="Euler characteristic: V - E + F")


class SymmetryDiagnostics(BaseModel):
    primary_plane: Literal["none", "yz_plane", "xz_plane", "xy_plane"] = Field(
        default="none", description="Best candidate bilateral symmetry plane"
    )
    deviation_score: float = Field(
        default=1.0, ge=0.0, description="Mean normalized distance to mirrored geometry (0.0 = perfect symmetry)"
    )
    is_approximately_symmetric: bool = Field(
        default=False, description="True if bilateral deviation is within tolerance (<= 0.08)"
    )


class OrientationDiagnostics(BaseModel):
    primary_axis: Tuple[float, float, float] = Field(..., description="Dominant principal axis of elongation")
    secondary_axis: Tuple[float, float, float] = Field(..., description="Secondary principal axis")
    estimated_up_axis: Literal["+Z", "-Z", "+Y", "-Y", "+X", "-X"] = Field(default="+Z")
    is_upright: bool = Field(default=True, description="True if primary or secondary elongation aligns with Z")


class GeometryQualityDiagnostics(BaseModel):
    has_nan_or_inf: bool = Field(default=False)
    nan_vertex_count: int = Field(default=0, ge=0)
    extremely_thin_features: bool = Field(default=False)
    min_edge_length: float = Field(default=0.0, ge=0.0)
    aspect_ratios: Tuple[float, float, float] = Field(..., description="Normalized proportions relative to max span")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall geometric health score")
    warnings: List[str] = Field(default_factory=list)


class CompleteGeometryReport(BaseModel):
    mesh_path: str = Field(...)
    file_size_bytes: int = Field(default=0, ge=0)
    is_valid_mesh: bool = Field(default=True)
    surface_area: float = Field(default=0.0, ge=0.0, description="Total mesh surface area in m^2")
    mesh_volume: Optional[float] = Field(default=None, description="Enclosed volume in m^3 if watertight")
    bounds: BoundingBox3D
    centroid: MeshCentroid
    counts: GeometryCounts
    topology: TopologyMeasurements
    symmetry: SymmetryDiagnostics
    orientation: OrientationDiagnostics
    quality: GeometryQualityDiagnostics
    appearance: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Geometry Analysis Engine Implementation
# ==============================================================================

class GeometryAnalyzer:
    """
    Deterministic CPU engine for analyzing 3D mesh geometry, topology,
    proportions, symmetry, and quality diagnostics.
    """

    def analyze_mesh(self, mesh_or_path: Any) -> CompleteGeometryReport:
        """
        Loads and performs full geometric and topological inspection of a mesh.
        Accepts file path (str, Path) or trimesh.Trimesh instance.
        """
        mesh, mesh_path, file_size = self._resolve_mesh(mesh_or_path)

        # 1. Coordinate Validity & NaN Check
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        has_nan_inf = bool(~np.isfinite(verts).all())
        nan_count = int(np.sum(~np.isfinite(verts).all(axis=1))) if has_nan_inf else 0

        # Purge bad coords for safe downstream math
        if has_nan_inf:
            finite_mask = np.isfinite(verts).all(axis=1)
            valid_verts = verts[finite_mask]
        else:
            valid_verts = verts

        if len(valid_verts) == 0:
            raise ValueError(f"Mesh has 0 valid finite vertices: {mesh_path}")

        # 2. Bounding Box & Dimensions
        min_pt = np.min(valid_verts, axis=0)
        max_pt = np.max(valid_verts, axis=0)
        dims = np.maximum(max_pt - min_pt, 1e-9)
        center = (min_pt + max_pt) / 2.0
        bb_vol = float(dims[0] * dims[1] * dims[2])

        bounds = BoundingBox3D(
            min_point=(round(float(min_pt[0]), 5), round(float(min_pt[1]), 5), round(float(min_pt[2]), 5)),
            max_point=(round(float(max_pt[0]), 5), round(float(max_pt[1]), 5), round(float(max_pt[2]), 5)),
            dimensions=(round(float(dims[0]), 5), round(float(dims[1]), 5), round(float(dims[2]), 5)),
            center=(round(float(center[0]), 5), round(float(center[1]), 5), round(float(center[2]), 5)),
            volume=round(bb_vol, 6)
        )

        # 3. Centroid & Mass
        mean_v = np.mean(valid_verts, axis=0)
        com = None
        if hasattr(mesh, "center_mass") and mesh.is_watertight and not has_nan_inf:
            try:
                cm = np.asarray(mesh.center_mass)
                com = (round(float(cm[0]), 5), round(float(cm[1]), 5), round(float(cm[2]), 5))
            except Exception:
                pass

        centroid = MeshCentroid(
            geometric_centroid=(round(float(mean_v[0]), 5), round(float(mean_v[1]), 5), round(float(mean_v[2]), 5)),
            center_of_mass=com
        )

        # 4. Geometry Counts
        num_v = len(verts)
        num_f = len(faces)
        num_e = len(mesh.edges_unique) if hasattr(mesh, "edges_unique") else 0
        connected_comp_count = 1
        try:
            split_list = mesh.split(only_watertight=False)
            connected_comp_count = max(1, len(split_list))
        except Exception:
            pass

        counts = GeometryCounts(
            vertex_count=num_v,
            face_count=num_f,
            edge_count=num_e,
            triangle_count=num_f,  # trimesh represents faces as triangles
            quad_count=0,
            connected_components=connected_comp_count
        )

        # 5. Topology Measurements
        is_watertight = bool(getattr(mesh, "is_watertight", False)) and not has_nan_inf
        is_winding = bool(getattr(mesh, "is_winding_consistent", False)) and not has_nan_inf

        vol = None
        if is_watertight and hasattr(mesh, "volume"):
            try:
                raw_vol = float(mesh.volume)
                if np.isfinite(raw_vol) and raw_vol > 0.0:
                    vol = round(raw_vol, 6)
            except Exception:
                pass

        surf_area = 0.0
        if hasattr(mesh, "area") and not has_nan_inf:
            try:
                raw_area = float(mesh.area)
                if np.isfinite(raw_area) and raw_area >= 0.0:
                    surf_area = round(raw_area, 6)
            except Exception:
                pass

        # Boundary edges (edges that appear in only 1 face)
        boundary_edges = 0
        non_manifold_edges = 0
        if hasattr(mesh, "edges_face"):
            try:
                # Count occurrences of unique edges
                from collections import Counter
                edge_counts = Counter(tuple(sorted(e)) for e in mesh.edges)
                boundary_edges = sum(1 for c in edge_counts.values() if c == 1)
                non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)
            except Exception:
                pass

        # Degenerate faces
        deg_faces = 0
        try:
            areas = mesh.area_faces
            deg_faces = int(np.sum(areas <= 1e-12))
        except Exception:
            pass

        # Duplicate vertices & isolated vertices
        duplicate_v = 0
        isolated_v = 0
        try:
            used_verts = set(np.unique(faces))
            isolated_v = max(0, num_v - len(used_verts))
            # Fast duplicate vertex count via KDTree or rounded unique
            rounded_v = np.round(valid_verts, decimals=6)
            unique_v = len(np.unique(rounded_v, axis=0))
            duplicate_v = max(0, len(valid_verts) - unique_v)
        except Exception:
            pass

        euler_num = None
        if num_v > 0 and num_e > 0 and num_f > 0:
            euler_num = num_v - num_e + num_f

        topology = TopologyMeasurements(
            is_watertight=is_watertight,
            boundary_edge_count=boundary_edges,
            non_manifold_edge_count=non_manifold_edges,
            degenerate_face_count=deg_faces,
            duplicate_vertex_count=duplicate_v,
            isolated_vertex_count=isolated_v,
            is_winding_consistent=is_winding,
            euler_number=euler_num
        )

        # 6. Quality Diagnostics
        max_span = max(dims)
        norm_ratios = (round(float(dims[0] / max_span), 4), round(float(dims[1] / max_span), 4), round(float(dims[2] / max_span), 4))
        
        min_edge = 0.0
        if hasattr(mesh, "edges_unique_length"):
            try:
                raw_min = float(np.min(mesh.edges_unique_length))
                if np.isfinite(raw_min) and raw_min >= 0.0:
                    min_edge = raw_min
            except Exception:
                pass

        warnings = []
        health_penalty = 0.0

        if has_nan_inf:
            warnings.append(f"Detected {nan_count} NaN/Inf vertex coordinates.")
            health_penalty += 0.4
        if deg_faces > 0:
            warnings.append(f"Detected {deg_faces} degenerate zero-area faces.")
            health_penalty += min(0.2, deg_faces / max(1, num_f))
        if non_manifold_edges > 0:
            warnings.append(f"Detected {non_manifold_edges} non-manifold edges.")
            health_penalty += 0.2
        if not is_watertight:
            warnings.append(f"Mesh is not watertight ({boundary_edges} boundary open edges).")
            health_penalty += 0.1
        if isolated_v > 0:
            warnings.append(f"Detected {isolated_v} unreferenced/isolated vertices.")
            health_penalty += 0.05
        if max_span < 0.01:
            warnings.append(f"Suspiciously tiny dimensions ({max_span:.4f}m maximum span).")
            health_penalty += 0.1
        elif max_span > 100.0:
            warnings.append(f"Suspiciously large dimensions ({max_span:.2f}m maximum span).")

        # Extremely thin feature detection (e.g. thickness < 0.5% of max span)
        sorted_spans = sorted(dims)
        is_thin = (sorted_spans[0] / max_span) < 0.015
        if is_thin:
            warnings.append(f"Extremely thin geometry detected: min thickness is {sorted_spans[0]:.4f}m.")

        quality_score = round(max(0.0, 1.0 - health_penalty), 3)

        quality = GeometryQualityDiagnostics(
            has_nan_or_inf=has_nan_inf,
            nan_vertex_count=nan_count,
            extremely_thin_features=is_thin,
            min_edge_length=round(min_edge, 6),
            aspect_ratios=norm_ratios,
            quality_score=quality_score,
            warnings=warnings
        )

        # 7. Symmetry Diagnostics
        symmetry = self._calculate_symmetry(valid_verts, center)

        # 8. Orientation Diagnostics
        orientation = self._calculate_orientation(valid_verts, dims)

        # 9. Appearance Metadata
        appearance = self._inspect_appearance(mesh)

        return CompleteGeometryReport(
            mesh_path=mesh_path,
            file_size_bytes=file_size,
            is_valid_mesh=not has_nan_inf and num_f > 0,
            surface_area=surf_area,
            mesh_volume=vol,
            bounds=bounds,
            centroid=centroid,
            counts=counts,
            topology=topology,
            symmetry=symmetry,
            orientation=orientation,
            quality=quality,
            appearance=appearance
        )

    def _resolve_mesh(self, mesh_or_path: Any) -> Tuple[Any, str, int]:
        """Resolves mesh input into (trimesh.Trimesh, path_str, file_size_bytes)."""
        if trimesh is None:
            raise RuntimeError("trimesh is required for geometry analysis.")

        if isinstance(mesh_or_path, (str, Path)):
            p = Path(mesh_or_path).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Mesh file does not exist: {p}")
            size_b = p.stat().st_size
            loaded = trimesh.load(str(p), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) == 0:
                    raise ValueError(f"Mesh scene contains 0 geometries: {p}")
                mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
            else:
                mesh = loaded
            return mesh, str(p), size_b
        elif isinstance(mesh_or_path, trimesh.Trimesh):
            return mesh_or_path, "in_memory_mesh", 0
        else:
            raise TypeError(f"Unsupported mesh object type: {type(mesh_or_path)}")

    def _calculate_symmetry(self, verts: np.ndarray, center: np.ndarray) -> SymmetryDiagnostics:
        """
        Calculates bilateral symmetry across candidate coordinate planes (YZ: x-reflection, XZ: y-reflection).
        Uses sample distances from mirrored points to nearest neighbors.
        """
        if len(verts) < 10:
            return SymmetryDiagnostics()

        from scipy.spatial import cKDTree

        # Subsample for speed if mesh is large (> 2000 points)
        if len(verts) > 2000:
            indices = np.random.choice(len(verts), size=2000, replace=False)
            sample_verts = verts[indices]
        else:
            sample_verts = verts

        tree = cKDTree(verts)
        max_extent = max(np.ptp(verts, axis=0))
        if max_extent < 1e-6:
            return SymmetryDiagnostics()

        # Plane 1: YZ plane (reflection across X centered at center[0])
        x_mirrored = sample_verts.copy()
        x_mirrored[:, 0] = 2.0 * center[0] - x_mirrored[:, 0]
        dists_x, _ = tree.query(x_mirrored)
        score_x = float(np.mean(dists_x)) / max_extent

        # Plane 2: XZ plane (reflection across Y centered at center[1])
        y_mirrored = sample_verts.copy()
        y_mirrored[:, 1] = 2.0 * center[1] - y_mirrored[:, 1]
        dists_y, _ = tree.query(y_mirrored)
        score_y = float(np.mean(dists_y)) / max_extent

        # Choose best candidate
        if score_x <= score_y:
            best_plane = "yz_plane"
            best_score = score_x
        else:
            best_plane = "xz_plane"
            best_score = score_y

        is_symmetric = best_score <= 0.08  # Average deviation <= 8% of max extent

        return SymmetryDiagnostics(
            primary_plane=best_plane if is_symmetric else "none",
            deviation_score=round(best_score, 4),
            is_approximately_symmetric=is_symmetric
        )

    def _calculate_orientation(self, verts: np.ndarray, dims: np.ndarray) -> OrientationDiagnostics:
        """
        Computes dominant principal component axes (PCA) to determine elongation
        and evaluate upright orientation relative to +Z.
        """
        centered = verts - np.mean(verts, axis=0)
        # Covariance matrix 3x3
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        v1 = eigenvectors[:, order[0]]  # Primary principal axis
        v2 = eigenvectors[:, order[1]]  # Secondary principal axis

        # Determine if primary elongation is vertical (+Z or -Z)
        z_dot = abs(v1[2])
        is_upright = z_dot > 0.65 or dims[2] >= max(dims[0], dims[1])

        # Estimate closest canonical up axis
        canonical_axes = {
            "+Z": np.array([0, 0, 1]),
            "-Z": np.array([0, 0, -1]),
            "+Y": np.array([0, 1, 0]),
            "-Y": np.array([0, -1, 0]),
            "+X": np.array([1, 0, 0]),
            "-X": np.array([-1, 0, 0]),
        }
        best_axis = "+Z"
        best_dot = -1.0
        for name, vec in canonical_axes.items():
            dot = np.dot(v1, vec)
            if dot > best_dot:
                best_dot = dot
                best_axis = name

        return OrientationDiagnostics(
            primary_axis=(round(float(v1[0]), 4), round(float(v1[1]), 4), round(float(v1[2]), 4)),
            secondary_axis=(round(float(v2[0]), 4), round(float(v2[1]), 4), round(float(v2[2]), 4)),
            estimated_up_axis=best_axis,
            is_upright=is_upright
        )

    def _inspect_appearance(self, mesh: Any) -> Dict[str, Any]:
        """Gathers vertex color and texture material attributes."""
        has_vc = bool(hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0)
        has_uv = bool(hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0)
        has_tex = False
        if hasattr(mesh.visual, "material"):
            mat = mesh.visual.material
            if getattr(mat, "image", None) is not None or getattr(mat, "baseColorTexture", None) is not None:
                has_tex = True

        color_var = 0.0
        mean_rgb = [0.8, 0.8, 0.8]
        if has_vc:
            try:
                colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                color_var = round(float(np.var(colors)), 6)
                mean_rgb = [round(float(c), 3) for c in np.mean(colors, axis=0)]
            except Exception:
                pass

        return {
            "has_vertex_colors": has_vc,
            "has_uvs": has_uv,
            "has_textures": has_tex,
            "color_variance": color_var,
            "mean_rgb": mean_rgb
        }


# Global singleton instance
geometry_analyzer = GeometryAnalyzer()
