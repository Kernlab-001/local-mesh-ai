"""
Appearance Analyzer for Local Mesh AI.
Extracts semantic material guidance, color palettes, and surface physical properties
from text prompts and reference artwork using deterministic NLP keyword matching
and color clustering. Works in both standard Python and Blender's bundled Python.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from scipy.cluster.vq import kmeans
except ImportError:
    kmeans = None

# Standard PBR Physical Material Presets
MATERIAL_PRESETS = {
    "wood": {
        "roughness": 0.35,
        "metallic": 0.0,
        "texture_type": "wood_grain",
        "default_color": [61, 38, 25]  # Dark Walnut Brown
    },
    "fabric": {
        "roughness": 0.85,
        "metallic": 0.0,
        "texture_type": "woven_fabric",
        "default_color": [242, 235, 224]  # Warm Ivory Cream
    },
    "metal": {
        "roughness": 0.22,
        "metallic": 0.90,
        "texture_type": "brushed_metal",
        "default_color": [180, 185, 190]  # Steel / Silver
    },
    "gold": {
        "roughness": 0.25,
        "metallic": 0.95,
        "texture_type": "brushed_metal",
        "default_color": [212, 175, 55]  # Warm Gold
    },
    "ceramic": {
        "roughness": 0.15,
        "metallic": 0.0,
        "texture_type": "ceramic_gloss",
        "default_color": [34, 139, 34]  # Forest Green / Glazed Ceramic
    },
    "car_paint": {
        "roughness": 0.18,
        "metallic": 0.65,
        "texture_type": "car_paint",
        "default_color": [210, 25, 25]  # Crimson Red Gloss
    },
    "rubber": {
        "roughness": 0.80,
        "metallic": 0.0,
        "texture_type": "rubber_matte",
        "default_color": [30, 30, 32]  # Dark Tire Rubber
    },
    "glass": {
        "roughness": 0.05,
        "metallic": 0.05,
        "texture_type": "glass_tint",
        "default_color": [60, 80, 100]  # Tinted Glass
    },
    "leather": {
        "roughness": 0.50,
        "metallic": 0.0,
        "texture_type": "leather_grain",
        "default_color": [80, 45, 25]  # Saddle Brown
    },
    "plastic": {
        "roughness": 0.40,
        "metallic": 0.0,
        "texture_type": "custom",
        "default_color": [120, 120, 120]
    }
}

# Standard Color Mapping for Natural Language Prompts
COLOR_KEYWORDS = {
    "walnut": [61, 38, 25],
    "dark walnut": [50, 30, 18],
    "brown": [85, 55, 35],
    "ivory": [245, 242, 235],
    "cream": [242, 235, 220],
    "ivory-cream": [245, 240, 228],
    "white": [240, 240, 240],
    "black": [25, 25, 25],
    "red": [205, 30, 30],
    "crimson": [180, 20, 30],
    "green": [34, 139, 34],
    "emerald": [20, 140, 70],
    "blue": [30, 90, 180],
    "navy": [20, 40, 90],
    "gold": [212, 175, 55],
    "golden": [218, 165, 32],
    "silver": [192, 192, 192],
    "gray": [128, 128, 128],
    "grey": [128, 128, 128],
    "yellow": [220, 200, 30],
    "orange": [220, 110, 25],
    "purple": [128, 30, 128],
    "cyan": [30, 180, 190],
    "bronze": [140, 95, 45],
    "copper": [184, 115, 51]
}

def simple_kmeans_numpy(data: np.ndarray, k: int, max_iter: int = 15) -> np.ndarray:
    """Pure NumPy K-Means clustering fallback with zero dependencies."""
    if len(data) <= k:
        return data
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices].astype(float)
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1.0):
            break
        centroids = new_centroids
    return centroids

class MaterialSlotConfig:
    def __init__(
        self,
        name: str,
        color_rgb: List[int],
        roughness: float,
        metallic: float,
        texture_type: str,
        region_id: str,
        z_min_norm: float = 0.0,
        z_max_norm: float = 1.0,
        description: str = ""
    ):
        self.name = name
        self.color_rgb = [int(c) for c in color_rgb]
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.texture_type = texture_type
        self.region_id = region_id
        self.z_min_norm = float(z_min_norm)
        self.z_max_norm = float(z_max_norm)
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "color_rgb": self.color_rgb,
            "color_normalized": [c / 255.0 for c in self.color_rgb],
            "roughness": self.roughness,
            "metallic": self.metallic,
            "texture_type": self.texture_type,
            "region_id": self.region_id,
            "z_min_norm": self.z_min_norm,
            "z_max_norm": self.z_max_norm,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaterialSlotConfig":
        return cls(
            name=data.get("name", "M_Surface_PBR"),
            color_rgb=data.get("color_rgb", [180, 180, 180]),
            roughness=float(data.get("roughness", 0.5)),
            metallic=float(data.get("metallic", 0.0)),
            texture_type=data.get("texture_type", "custom"),
            region_id=data.get("region_id", "body"),
            z_min_norm=float(data.get("z_min_norm", 0.0)),
            z_max_norm=float(data.get("z_max_norm", 1.0)),
            description=data.get("description", "")
        )

class AppearanceProfile:
    def __init__(self, category: str, materials: List[MaterialSlotConfig], dominant_palette: List[List[int]]):
        self.category = category
        self.materials = materials
        self.dominant_palette = dominant_palette

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "materials": [m.to_dict() for m in self.materials],
            "dominant_palette": self.dominant_palette
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppearanceProfile":
        mats = [MaterialSlotConfig.from_dict(m) for m in data.get("materials", [])]
        return cls(
            category=data.get("category", "generic"),
            materials=mats,
            dominant_palette=data.get("dominant_palette", [[180, 180, 180]])
        )

class AppearanceAnalyzer:
    def __init__(self):
        pass

    def extract_image_palette(self, image_path: Optional[str], k: int = 3) -> List[List[int]]:
        """Extracts dominant color palette from reference image."""
        if not image_path or not os.path.exists(image_path):
            return [[180, 180, 180], [60, 60, 60], [220, 220, 220]]

        try:
            rgb = None
            if Image is not None:
                with Image.open(image_path) as img:
                    arr = np.array(img.convert("RGBA"))
                raw_rgb = arr[..., :3].reshape(-1, 3)
                alpha = arr[..., 3].reshape(-1)
                mask = alpha > 40
                rgb = raw_rgb[mask] if mask.sum() >= 200 else raw_rgb
            elif 'bpy' in sys.modules:
                import bpy
                b_img = bpy.data.images.load(image_path)
                w, h = b_img.size
                pixels = np.empty(w * h * 4, dtype=np.float32)
                b_img.pixels.foreach_get(pixels)
                bpy.data.images.remove(b_img)
                arr = (pixels.reshape(-1, 4) * 255.0).astype(np.uint8)
                raw_rgb = arr[..., :3]
                alpha = arr[..., 3]
                mask = alpha > 40
                rgb = raw_rgb[mask] if mask.sum() >= 200 else raw_rgb

            if rgb is None or len(rgb) == 0:
                return [[180, 180, 180], [60, 60, 60], [220, 220, 220]]

            if len(rgb) > 20000:
                idx = np.random.choice(len(rgb), 20000, replace=False)
                fg = rgb[idx]
            else:
                fg = rgb

            if kmeans is not None:
                centroids, _ = kmeans(fg.astype(float), k)
            else:
                centroids = simple_kmeans_numpy(fg.astype(float), k)

            palette = [[int(np.clip(c, 0, 255)) for c in row] for row in centroids]
            return palette
        except Exception:
            return [[180, 180, 180], [60, 60, 60], [220, 220, 220]]

    def analyze_appearance(
        self,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> AppearanceProfile:
        """
        Analyzes prompt and reference image to produce an appearance profile
        with distinct material slots, colors, and PBR parameters.
        """
        prompt_lower = (prompt or "").lower()
        palette = self.extract_image_palette(image_path, k=4)

        # ---------------------------------------------------------------------
        # Case A: Chair / Seating Multi-Material Partitioning
        # ---------------------------------------------------------------------
        if "chair" in prompt_lower or "armchair" in prompt_lower or "stool" in prompt_lower or "sofa" in prompt_lower:
            wood_color = [61, 38, 25]  # Dark walnut default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower and ("wood" in kw or "walnut" in kw or "brown" in kw or "frame" in prompt_lower):
                    wood_color = rgb
                    break

            cushion_color = [242, 235, 220]  # Ivory cream default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower and (kw in ("ivory", "cream", "white", "ivory-cream", "beige")):
                    cushion_color = rgb
                    break

            if palette and len(palette) >= 2:
                # Darker centroid for frame/legs, lighter centroid for cushion
                sorted_by_lum = sorted(palette, key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
                if not any(k in prompt_lower for k in ("walnut", "brown", "dark")):
                    wood_color = sorted_by_lum[0]
                if not any(k in prompt_lower for k in ("ivory", "cream", "white")):
                    cushion_color = sorted_by_lum[-1]

            mat_frame = MaterialSlotConfig(
                name="M_Wood_Frame",
                color_rgb=wood_color,
                roughness=0.35,
                metallic=0.0,
                texture_type="wood_grain",
                region_id="legs_and_frame",
                z_min_norm=0.0,
                z_max_norm=0.45,
                description="Dark walnut-brown wooden frame and legs with realistic wood grain"
            )

            mat_cushion = MaterialSlotConfig(
                name="M_Fabric_Cushion",
                color_rgb=cushion_color,
                roughness=0.85,
                metallic=0.0,
                texture_type="woven_fabric",
                region_id="seat_cushion",
                z_min_norm=0.45,
                z_max_norm=0.60,
                description="Ivory-cream upholstered seat cushion with woven upholstery fabric"
            )

            mat_backrest = MaterialSlotConfig(
                name="M_Fabric_Backrest",
                color_rgb=cushion_color,
                roughness=0.82,
                metallic=0.0,
                texture_type="woven_fabric",
                region_id="backrest",
                z_min_norm=0.60,
                z_max_norm=1.0,
                description="Matching cream tufted backrest"
            )

            return AppearanceProfile(
                category="chair",
                materials=[mat_frame, mat_cushion, mat_backrest],
                dominant_palette=palette
            )

        # ---------------------------------------------------------------------
        # Case B: Vehicle / Car Multi-Material Partitioning
        # ---------------------------------------------------------------------
        elif "car" in prompt_lower or "vehicle" in prompt_lower or "automobile" in prompt_lower:
            body_color = [210, 25, 25]  # Red default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower:
                    body_color = rgb
                    break
            if palette and len(palette) > 0 and not any(k in prompt_lower for k in COLOR_KEYWORDS):
                body_color = palette[0]

            mat_body = MaterialSlotConfig(
                name="M_Car_Paint",
                color_rgb=body_color,
                roughness=0.18,
                metallic=0.65,
                texture_type="car_paint",
                region_id="body",
                z_min_norm=0.25,
                z_max_norm=0.75,
                description="Glossy reflective automotive paint"
            )
            mat_wheels = MaterialSlotConfig(
                name="M_Tire_Rubber",
                color_rgb=[25, 25, 28],
                roughness=0.82,
                metallic=0.0,
                texture_type="rubber_matte",
                region_id="wheels",
                z_min_norm=0.0,
                z_max_norm=0.25,
                description="Matte rubber tires"
            )
            mat_glass = MaterialSlotConfig(
                name="M_Window_Glass",
                color_rgb=[50, 70, 90],
                roughness=0.08,
                metallic=0.1,
                texture_type="glass_tint",
                region_id="cabin",
                z_min_norm=0.75,
                z_max_norm=1.0,
                description="Tinted cockpit glass"
            )
            return AppearanceProfile(category="vehicle", materials=[mat_body, mat_wheels, mat_glass], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case C: Robot / Sci-Fi / Hard-Surface
        # ---------------------------------------------------------------------
        elif "robot" in prompt_lower or "droid" in prompt_lower or "mech" in prompt_lower or "cyber" in prompt_lower:
            primary_color = [40, 110, 200]  # Blue default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower:
                    primary_color = rgb
                    break

            mat_armor = MaterialSlotConfig(
                name="M_Armor_Plates",
                color_rgb=primary_color,
                roughness=0.25,
                metallic=0.85,
                texture_type="brushed_metal",
                region_id="armor",
                z_min_norm=0.4,
                z_max_norm=1.0,
                description="Chassis metal armor plates"
            )
            mat_joints = MaterialSlotConfig(
                name="M_Joint_Mechanics",
                color_rgb=[50, 52, 55],
                roughness=0.45,
                metallic=0.90,
                texture_type="brushed_metal",
                region_id="joints",
                z_min_norm=0.0,
                z_max_norm=0.4,
                description="Dark mechanical joints and frame"
            )
            return AppearanceProfile(category="robot", materials=[mat_armor, mat_joints], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case D: Ceramic / Vase / Pottery / Tableware
        # ---------------------------------------------------------------------
        elif "vase" in prompt_lower or "ceramic" in prompt_lower or "pottery" in prompt_lower or "mug" in prompt_lower or "cup" in prompt_lower or "plate" in prompt_lower:
            glaze_color = [34, 139, 34]  # Green glaze default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower:
                    glaze_color = rgb
                    break
            if palette and len(palette) > 0 and not any(k in prompt_lower for k in COLOR_KEYWORDS):
                glaze_color = palette[0]

            mat_glaze = MaterialSlotConfig(
                name="M_Ceramic_Glaze",
                color_rgb=glaze_color,
                roughness=0.15,
                metallic=0.0,
                texture_type="ceramic_gloss",
                region_id="outer_surface",
                z_min_norm=0.1,
                z_max_norm=1.0,
                description="High-gloss glazed ceramic finish"
            )
            mat_base = MaterialSlotConfig(
                name="M_Terracotta_Base",
                color_rgb=[180, 115, 80],
                roughness=0.75,
                metallic=0.0,
                texture_type="custom",
                region_id="foot_ring",
                z_min_norm=0.0,
                z_max_norm=0.1,
                description="Unglazed matte earthenware base foot"
            )
            return AppearanceProfile(category="ceramic", materials=[mat_glaze, mat_base], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case E: Generic Object / Prop / Character
        # ---------------------------------------------------------------------
        else:
            primary_color = [180, 180, 180]
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower:
                    primary_color = rgb
                    break
            if palette and len(palette) > 0 and not any(k in prompt_lower for k in COLOR_KEYWORDS):
                primary_color = palette[0]

            mat_main = MaterialSlotConfig(
                name="M_Surface_PBR",
                color_rgb=primary_color,
                roughness=0.45,
                metallic=0.1,
                texture_type="custom",
                region_id="primary_mesh",
                z_min_norm=0.0,
                z_max_norm=1.0,
                description="Primary surface PBR material"
            )
            return AppearanceProfile(category="generic", materials=[mat_main], dominant_palette=palette)

appearance_analyzer = AppearanceAnalyzer()
