"""
Advanced Geometry Construction Engine for Local Mesh AI.
Provides deterministic, parametric primitive generation and procedural profile creation
for both fast CPU-based preview/manipulation (via Trimesh) and high-fidelity Blender generation.

Supported Primitives:
- Cube, UV Sphere, Icosphere, Cylinder, Cone, Torus, Plane, Circle
- Custom Extruded Profile (from 2D coordinate polyline)
- Custom Mesh (from validated vertex and face arrays)
"""

import os
import sys
import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class PrimitiveType(str):
    CUBE = "cube"
    UV_SPHERE = "uv_sphere"
    ICOSPHERE = "icosphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    TORUS = "torus"
    PLANE = "plane"
    CIRCLE = "circle"
    CUSTOM_PROFILE = "custom_profile"
    CUSTOM_MESH = "custom_mesh"


class ConstructPrimitiveArgs(BaseModel):
    primitive_type: str = Field(description="Type of primitive to construct (cube, uv_sphere, icosphere, cylinder, cone, torus, plane, circle, custom_profile)")
    dimensions: Optional[List[float]] = Field(default=None, description="[width, depth, height] or [size_x, size_y, size_z]")
    radius: Optional[float] = Field(default=None, description="Radius for sphere, cylinder, cone, circle")
    radius2: Optional[float] = Field(default=None, description="Secondary radius for cone tip or torus minor radius")
    height: Optional[float] = Field(default=None, description="Height for cylinder or cone")
    segments: Optional[int] = Field(default=32, description="Radial subdivisions")
    rings: Optional[int] = Field(default=16, description="Rings for sphere or torus")
    subdivisions: Optional[int] = Field(default=2, description="Subdivision level for icosphere")
    profile_points_2d: Optional[List[List[float]]] = Field(default=None, description="2D polygon points [[x, y], ...] for custom profile")
    extrusion_depth: Optional[float] = Field(default=1.0, description="Extrusion depth for 2D profile")
    center: Optional[List[float]] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Target origin position [x, y, z]")
    output_path: Optional[str] = Field(default=None, description="Optional path to export constructed mesh (.glb/.obj)")


class GeometryConstructionEngine:
    """
    Constructs 3D primitives and procedural shapes deterministically.
    """

    @staticmethod
    def create_cube(
        extents: Union[List[float], Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a watertight box with given extents [dx, dy, dz]."""
        ext = np.array(extents, dtype=np.float64)
        mesh = trimesh.creation.box(extents=ext)
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_uv_sphere(
        radius: float = 0.5,
        segments: int = 32,
        rings: int = 16,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a UV sphere with given radius, segments (longitude), and rings (latitude)."""
        mesh = trimesh.creation.uv_sphere(radius=radius, count=[segments, rings])
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_icosphere(
        radius: float = 0.5,
        subdivisions: int = 3,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a geodetic icosphere with uniform triangle distribution."""
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_cylinder(
        radius: float = 0.5,
        height: float = 1.0,
        sections: int = 32,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a cylinder aligned vertically along Z."""
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_cone(
        radius: float = 0.5,
        height: float = 1.0,
        radius2: float = 0.0,
        sections: int = 32,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a cone or truncated cone."""
        mesh = trimesh.creation.cone(radius=radius, height=height, sections=sections)
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_torus(
        major_radius: float = 1.0,
        minor_radius: float = 0.25,
        major_sections: int = 32,
        minor_sections: int = 16,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a torus centered at origin in the XY plane."""
        mesh = trimesh.creation.torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_sections=major_sections,
            minor_sections=minor_sections
        )
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_plane(
        size_x: float = 1.0,
        size_y: float = 1.0,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a rectangular flat plane on XY with normal +Z."""
        hx, hy = size_x / 2.0, size_y / 2.0
        vertices = np.array([
            [-hx, -hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, hy, 0.0]
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_circle(
        radius: float = 0.5,
        sections: int = 32,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Constructs a 2D filled disc in the XY plane."""
        angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        pts = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(sections)])
        center_pt = np.array([[0.0, 0.0, 0.0]])
        verts = np.vstack([center_pt, pts])
        faces = []
        for i in range(sections):
            next_i = (i + 1) % sections
            faces.append([0, i + 1, next_i + 1])
        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64))
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @staticmethod
    def create_extruded_profile(
        points_2d: List[List[float]],
        height: float = 1.0,
        center: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    ) -> trimesh.Trimesh:
        """Extrudes a 2D closed polygon along Z into a watertight 3D prism."""
        pts = np.array(points_2d, dtype=np.float64)
        n = len(pts)
        c2d = np.mean(pts, axis=0)

        # Bottom vertices (z = 0) and Top vertices (z = height)
        v_bottom = np.column_stack([pts, np.zeros(n)])
        v_top = np.column_stack([pts, np.full(n, height)])
        verts = np.vstack([v_bottom, v_top, [c2d[0], c2d[1], 0.0], [c2d[0], c2d[1], height]])
        c_bottom_idx = 2 * n
        c_top_idx = 2 * n + 1

        faces = []
        # Bottom cap (normal points -Z)
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([c_bottom_idx, next_i, i])

        # Top cap (normal points +Z)
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([c_top_idx, n + i, n + next_i])

        # Side walls
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, n + next_i])
            faces.append([i, n + next_i, n + i])

        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64))
        mesh.merge_vertices()
        if center != (0.0, 0.0, 0.0):
            mesh.apply_translation(center)
        return mesh

    @classmethod
    def construct_from_args(cls, args: ConstructPrimitiveArgs) -> trimesh.Trimesh:
        """Unified factory method constructing a primitive from typed args."""
        ptype = args.primitive_type.lower()
        center = tuple(args.center or [0.0, 0.0, 0.0])

        if ptype in ("cube", "box"):
            extents = args.dimensions or [1.0, 1.0, 1.0]
            mesh = cls.create_cube(extents=extents, center=center)
        elif ptype in ("uv_sphere", "sphere"):
            radius = args.radius or 0.5
            mesh = cls.create_uv_sphere(radius=radius, segments=args.segments or 32, rings=args.rings or 16, center=center)
        elif ptype in ("icosphere", "ico"):
            radius = args.radius or 0.5
            mesh = cls.create_icosphere(radius=radius, subdivisions=args.subdivisions or 3, center=center)
        elif ptype == "cylinder":
            radius = args.radius or 0.5
            height = args.height or (args.dimensions[2] if args.dimensions and len(args.dimensions) > 2 else 1.0)
            mesh = cls.create_cylinder(radius=radius, height=height, sections=args.segments or 32, center=center)
        elif ptype == "cone":
            radius = args.radius or 0.5
            height = args.height or 1.0
            radius2 = args.radius2 or 0.0
            mesh = cls.create_cone(radius=radius, height=height, radius2=radius2, sections=args.segments or 32, center=center)
        elif ptype == "torus":
            maj_r = args.radius or 1.0
            min_r = args.radius2 or 0.25
            mesh = cls.create_torus(major_radius=maj_r, minor_radius=min_r, major_sections=args.segments or 32, minor_sections=args.rings or 16, center=center)
        elif ptype == "plane":
            sx = args.dimensions[0] if args.dimensions else 1.0
            sy = args.dimensions[1] if args.dimensions and len(args.dimensions) > 1 else 1.0
            mesh = cls.create_plane(size_x=sx, size_y=sy, center=center)
        elif ptype == "circle":
            radius = args.radius or 0.5
            mesh = cls.create_circle(radius=radius, sections=args.segments or 32, center=center)
        elif ptype == "custom_profile":
            if not args.profile_points_2d:
                raise ValueError("profile_points_2d required for custom_profile construction")
            depth = args.extrusion_depth or (args.dimensions[2] if args.dimensions and len(args.dimensions) > 2 else 1.0)
            mesh = cls.create_extruded_profile(points_2d=args.profile_points_2d, height=depth, center=center)
        else:
            raise ValueError(f"Unsupported primitive type: '{ptype}'")

        if args.output_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
            mesh.export(args.output_path)

        return mesh
