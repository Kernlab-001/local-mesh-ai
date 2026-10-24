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

def has_word(text: str, *words: str) -> bool:
    """Checks for whole word presence using regex word boundaries to avoid substring collisions."""
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
            return True
    return False

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
        if has_word(prompt_lower, "chair", "armchair", "stool", "sofa", "bench", "seating", "cushion", "pillow"):
            is_cushion_only = has_word(prompt_lower, "cushion", "pillow") and not has_word(prompt_lower, "chair", "armchair", "sofa", "stool")
            if is_cushion_only:
                fab_type = "velvet" if "velvet" in prompt_lower else "woven_fabric"
                fab_color = [18, 120, 65] if "emerald" in prompt_lower or "green" in prompt_lower else [220, 215, 205]
                trim_color = [212, 175, 55] if "gold" in prompt_lower or "golden" in prompt_lower else [150, 150, 150]
                mat_cushion = MaterialSlotConfig(
                    name="M_Velvet_Fabric" if "velvet" in prompt_lower else "M_Fabric_Surface",
                    color_rgb=fab_color,
                    roughness=0.75,
                    metallic=0.0,
                    texture_type=fab_type,
                    region_id="body",
                    z_min_norm=0.1,
                    z_max_norm=1.0,
                    description="Luxurious tufted velvet fabric"
                )
                mat_trim = MaterialSlotConfig(
                    name="M_Gold_Trim" if "gold" in prompt_lower else "M_Piping_Cord",
                    color_rgb=trim_color,
                    roughness=0.28,
                    metallic=0.90 if "gold" in prompt_lower else 0.1,
                    texture_type="brushed_metal" if "gold" in prompt_lower else "woven_fabric",
                    region_id="piping",
                    z_min_norm=0.0,
                    z_max_norm=0.1,
                    description="Golden cord piping border"
                )
                return AppearanceProfile(category="furniture", materials=[mat_cushion, mat_trim], dominant_palette=palette)

            # Standard Chair or Armchair
            is_leather = "leather" in prompt_lower
            wood_color = [61, 38, 25]  # Dark walnut default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower and ("wood" in kw or "walnut" in kw or "brown" in kw or "frame" in prompt_lower):
                    wood_color = rgb
                    break

            cushion_color = [80, 45, 25] if is_leather else [242, 235, 220]
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower and (kw in ("ivory", "cream", "white", "ivory-cream", "beige", "brown")):
                    cushion_color = rgb
                    break

            mat_frame = MaterialSlotConfig(
                name="M_Wood_Frame",
                color_rgb=wood_color,
                roughness=0.35,
                metallic=0.0,
                texture_type="wood_grain",
                region_id="legs_and_frame",
                z_min_norm=0.0,
                z_max_norm=0.45,
                description="Walnut wooden frame and legs with realistic wood grain"
            )

            mat_cushion = MaterialSlotConfig(
                name="M_Leather_Upholstery" if is_leather else "M_Fabric_Cushion",
                color_rgb=cushion_color,
                roughness=0.52 if is_leather else 0.85,
                metallic=0.0,
                texture_type="leather" if is_leather else "woven_fabric",
                region_id="seat_cushion",
                z_min_norm=0.45,
                z_max_norm=0.60,
                description="Deep tufted vintage leather upholstery" if is_leather else "Upholstered seat cushion"
            )

            extra_mats = []
            if "brass" in prompt_lower or "rivet" in prompt_lower:
                extra_mats.append(MaterialSlotConfig(
                    name="M_Brass_Rivets",
                    color_rgb=[212, 175, 55],
                    roughness=0.25,
                    metallic=0.92,
                    texture_type="brushed_metal",
                    region_id="rivets",
                    z_min_norm=0.45,
                    z_max_norm=0.75,
                    description="Polished decorative brass rivets"
                ))
            else:
                extra_mats.append(MaterialSlotConfig(
                    name="M_Fabric_Backrest",
                    color_rgb=cushion_color,
                    roughness=0.82,
                    metallic=0.0,
                    texture_type="woven_fabric",
                    region_id="backrest",
                    z_min_norm=0.60,
                    z_max_norm=1.0,
                    description="Matching backrest"
                ))

            return AppearanceProfile(
                category="furniture",
                materials=[mat_frame, mat_cushion] + extra_mats,
                dominant_palette=palette
            )

        # ---------------------------------------------------------------------
        # Case B: Weapon (Battleaxe, Sword, Rifle, Staff, Dagger, Blade)
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "battleaxe", "axe", "sword", "blade", "staff", "rifle", "dagger", "warhammer", "hammer", "weapon", "bow"):
            if "rifle" in prompt_lower:
                casing_mat = MaterialSlotConfig(
                    name="M_Carbon_Fiber_Casing",
                    color_rgb=[32, 34, 36],
                    roughness=0.30,
                    metallic=0.15,
                    texture_type="carbon_fiber",
                    region_id="casing",
                    z_min_norm=0.3,
                    z_max_norm=1.0,
                    description="Carbon fiber chassis casing with cyan emissive accents"
                )
                metal_mat = MaterialSlotConfig(
                    name="M_Receiver_Metal",
                    color_rgb=[80, 85, 90],
                    roughness=0.22,
                    metallic=0.92,
                    texture_type="brushed_metal",
                    region_id="barrel_receiver",
                    z_min_norm=0.0,
                    z_max_norm=0.3,
                    description="Machined weapon steel barrel and receiver"
                )
                return AppearanceProfile(category="weapon", materials=[casing_mat, metal_mat], dominant_palette=palette)

            elif "staff" in prompt_lower:
                staff_wood = MaterialSlotConfig(
                    name="M_Ancient_Wood",
                    color_rgb=[75, 48, 25],
                    roughness=0.78,
                    metallic=0.0,
                    texture_type="wood_grain",
                    region_id="shaft",
                    z_min_norm=0.0,
                    z_max_norm=0.85,
                    description="Twisted gnarled ancient wooden staff shaft"
                )
                crystal_orb = MaterialSlotConfig(
                    name="M_Crystal_Orb",
                    color_rgb=[170, 70, 220],
                    roughness=0.08,
                    metallic=0.1,
                    texture_type="crystal",
                    region_id="orb_head",
                    z_min_norm=0.85,
                    z_max_norm=1.0,
                    description="Glowing purple resonant crystal orb"
                )
                return AppearanceProfile(category="weapon", materials=[staff_wood, crystal_orb], dominant_palette=palette)

            metal_color = [185, 190, 195]  # Steel default
            if "gold" in prompt_lower:
                metal_color = [220, 180, 55]
            elif "iron" in prompt_lower:
                metal_color = [75, 78, 82]
            elif "bronze" in prompt_lower:
                metal_color = [150, 100, 50]

            handle_color = [70, 45, 28]  # Wood / leather handle default
            if "leather" in prompt_lower:
                handle_color = [85, 50, 30]
            elif "black" in prompt_lower:
                handle_color = [30, 30, 32]

            mat_blade = MaterialSlotConfig(
                name="M_Blade_Metal",
                color_rgb=metal_color,
                roughness=0.22,
                metallic=0.92,
                texture_type="brushed_metal",
                region_id="blade_head",
                z_min_norm=0.4,
                z_max_norm=1.0,
                description="Reflective metallic striking edge / barrel"
            )
            mat_handle = MaterialSlotConfig(
                name="M_Handle_Wood" if ("wood" in prompt_lower or "iron" in prompt_lower) else "M_Handle_Grip",
                color_rgb=handle_color,
                roughness=0.65,
                metallic=0.05,
                texture_type="wood_grain",
                region_id="shaft_grip",
                z_min_norm=0.0,
                z_max_norm=0.4,
                description="Carved wood or wrapped handle grip"
            )
            return AppearanceProfile(category="weapon", materials=[mat_blade, mat_handle], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case C: Vehicle / Automobile / Wheel
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "car", "vehicle", "automobile", "truck", "wagon", "kart", "wheel", "tire", "rim"):
            if has_word(prompt_lower, "wheel", "tire", "rim"):
                mat_rim = MaterialSlotConfig(
                    name="M_Alloy_Rim_Aluminum",
                    color_rgb=[210, 215, 220],
                    roughness=0.20,
                    metallic=0.95,
                    texture_type="brushed_metal",
                    region_id="rim",
                    z_min_norm=0.2,
                    z_max_norm=0.8,
                    description="Forged aluminum alloy wheel rim"
                )
                mat_tire = MaterialSlotConfig(
                    name="M_Tire_Rubber",
                    color_rgb=[25, 25, 28],
                    roughness=0.82,
                    metallic=0.0,
                    texture_type="rubber_matte",
                    region_id="tire_tread",
                    z_min_norm=0.0,
                    z_max_norm=1.0,
                    description="Black molded tread rubber tire"
                )
                return AppearanceProfile(category="vehicle", materials=[mat_rim, mat_tire], dominant_palette=palette)

            body_color = [210, 25, 25]  # Red default
            for kw, rgb in COLOR_KEYWORDS.items():
                if kw in prompt_lower:
                    body_color = rgb
                    break

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
        # Case D: Armor / Helm / Shield
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "helm", "helmet", "armor", "gauntlet", "cuirass", "shield", "greaves"):
            steel_color = [190, 195, 200]
            trim_color = [212, 175, 55] if "brass" in prompt_lower or "gold" in prompt_lower else [80, 80, 85]
            mat_steel = MaterialSlotConfig(
                name="M_Armor_Steel",
                color_rgb=steel_color,
                roughness=0.20,
                metallic=0.95,
                texture_type="brushed_metal",
                region_id="plates",
                z_min_norm=0.1,
                z_max_norm=1.0,
                description="Polished gothic steel knight helm plates"
            )
            mat_trim = MaterialSlotConfig(
                name="M_Brass_Trim" if "brass" in prompt_lower else "M_Trim_Metal",
                color_rgb=trim_color,
                roughness=0.26,
                metallic=0.90,
                texture_type="brushed_metal",
                region_id="trim",
                z_min_norm=0.0,
                z_max_norm=0.1,
                description="Engraved brass trim borders"
            )
            return AppearanceProfile(category="armor", materials=[mat_steel, mat_trim], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case E: Furniture (Desk, Table, Cabinet, Shelf, Barrel, Chest, Altar)
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "desk", "table", "cabinet", "shelf", "bookshelf", "altar", "chest", "barrel"):
            if "barrel" in prompt_lower:
                mat_wood = MaterialSlotConfig(
                    name="M_Oak_Wood_Staves",
                    color_rgb=[110, 72, 40],
                    roughness=0.45,
                    metallic=0.0,
                    texture_type="wood_grain",
                    region_id="barrel_body",
                    z_min_norm=0.1,
                    z_max_norm=0.9,
                    description="Weathered oak wooden barrel staves"
                )
                mat_hoops = MaterialSlotConfig(
                    name="M_Iron_Hoops",
                    color_rgb=[45, 45, 48],
                    roughness=0.60,
                    metallic=0.88,
                    texture_type="brushed_metal",
                    region_id="metal_bands",
                    z_min_norm=0.0,
                    z_max_norm=1.0,
                    description="Black rusted iron binding hoops"
                )
                return AppearanceProfile(category="furniture", materials=[mat_wood, mat_hoops], dominant_palette=palette)

            elif "chest" in prompt_lower:
                mat_chest_wood = MaterialSlotConfig(
                    name="M_Chest_Wood",
                    color_rgb=[95, 60, 32],
                    roughness=0.50,
                    metallic=0.0,
                    texture_type="wood_grain",
                    region_id="chest_body",
                    z_min_norm=0.1,
                    z_max_norm=0.9,
                    description="Heavy aged timber treasure chest body"
                )
                mat_iron_bands = MaterialSlotConfig(
                    name="M_Iron_Hardware",
                    color_rgb=[40, 42, 45],
                    roughness=0.55,
                    metallic=0.90,
                    texture_type="brushed_metal",
                    region_id="reinforcements",
                    z_min_norm=0.0,
                    z_max_norm=1.0,
                    description="Ironbound corner brackets and padlock"
                )
                return AppearanceProfile(category="furniture", materials=[mat_chest_wood, mat_iron_bands], dominant_palette=palette)

            elif "altar" in prompt_lower:
                mat_altar = MaterialSlotConfig(
                    name="M_Basalt_Stone",
                    color_rgb=[45, 48, 50],
                    roughness=0.88,
                    metallic=0.0,
                    texture_type="stone",
                    region_id="monolith",
                    z_min_norm=0.0,
                    z_max_norm=1.0,
                    description="Monolithic dark basalt stone sacrificial altar"
                )
                return AppearanceProfile(category="prop", materials=[mat_altar], dominant_palette=palette)

            # Desk / Table
            is_birch = "birch" in prompt_lower or "light" in prompt_lower
            mat_top = MaterialSlotConfig(
                name="M_Birch_Wood" if is_birch else "M_Wood_Surface",
                color_rgb=[215, 195, 160] if is_birch else [120, 80, 45],
                roughness=0.38,
                metallic=0.0,
                texture_type="wood_grain",
                region_id="desktop",
                z_min_norm=0.7,
                z_max_norm=1.0,
                description="Birch wood desktop surface"
            )
            mat_legs = MaterialSlotConfig(
                name="M_Steel_Hairpin_Legs",
                color_rgb=[30, 30, 32],
                roughness=0.45,
                metallic=0.90,
                texture_type="brushed_metal",
                region_id="legs",
                z_min_norm=0.0,
                z_max_norm=0.7,
                description="Matte black steel hairpin legs"
            )
            return AppearanceProfile(category="furniture", materials=[mat_top, mat_legs], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case F: Robot / Android / Character
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "robot", "droid", "android", "mech", "cyber", "humanoid"):
            mat_chassis = MaterialSlotConfig(
                name="M_White_Composite",
                color_rgb=[240, 242, 245],
                roughness=0.25,
                metallic=0.10,
                texture_type="composite",
                region_id="outer_chassis",
                z_min_norm=0.3,
                z_max_norm=1.0,
                description="Sleek white composite humanoid android armor"
            )
            mat_joints = MaterialSlotConfig(
                name="M_Chrome_Joints",
                color_rgb=[220, 225, 230],
                roughness=0.10,
                metallic=0.98,
                texture_type="brushed_metal",
                region_id="mechanical_joints",
                z_min_norm=0.0,
                z_max_norm=0.3,
                description="Chrome metallic mechanical joints"
            )
            return AppearanceProfile(category="character", materials=[mat_chassis, mat_joints], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case G: Ceramic / Teapot / Pot / Chalice
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "teapot", "pot", "vase", "ceramic", "pottery", "mug", "cup", "chalice"):
            if "chalice" in prompt_lower:
                mat_gold = MaterialSlotConfig(
                    name="M_Solid_Gold",
                    color_rgb=[225, 185, 50],
                    roughness=0.18,
                    metallic=0.98,
                    texture_type="brushed_metal",
                    region_id="chalice_cup",
                    z_min_norm=0.0,
                    z_max_norm=1.0,
                    description="Engraved solid ceremonial gold chalice"
                )
                mat_gems = MaterialSlotConfig(
                    name="M_Ruby_Gem",
                    color_rgb=[185, 20, 40],
                    roughness=0.05,
                    metallic=0.05,
                    texture_type="crystal",
                    region_id="gem_studs",
                    z_min_norm=0.4,
                    z_max_norm=0.6,
                    description="Faceted red ruby gemstones"
                )
                return AppearanceProfile(category="prop", materials=[mat_gold, mat_gems], dominant_palette=palette)

            elif "terracotta" in prompt_lower or ("plant" in prompt_lower and "pot" in prompt_lower):
                mat_clay = MaterialSlotConfig(
                    name="M_Terracotta_Clay",
                    color_rgb=[195, 95, 55],
                    roughness=0.85,
                    metallic=0.0,
                    texture_type="clay",
                    region_id="pot_wall",
                    z_min_norm=0.2,
                    z_max_norm=1.0,
                    description="Rough porous orange terracotta clay pot"
                )
                mat_soil = MaterialSlotConfig(
                    name="M_Soil_Earth",
                    color_rgb=[45, 32, 24],
                    roughness=0.95,
                    metallic=0.0,
                    texture_type="soil",
                    region_id="interior_soil",
                    z_min_norm=0.0,
                    z_max_norm=0.2,
                    description="Rich dark organic soil potting mix"
                )
                return AppearanceProfile(category="prop", materials=[mat_clay, mat_soil], dominant_palette=palette)

            glaze_color = [185, 30, 30] if "red" in prompt_lower else [212, 175, 55] if "gold" in prompt_lower else [34, 139, 34]
            mat_glaze = MaterialSlotConfig(
                name="M_Ceramic_Glaze",
                color_rgb=glaze_color,
                roughness=0.12,
                metallic=0.0,
                texture_type="ceramic_gloss",
                region_id="outer_surface",
                z_min_norm=0.0,
                z_max_norm=1.0,
                description="Glossy red glazed ceramic teapot finish"
            )
            return AppearanceProfile(category="prop", materials=[mat_glaze], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case H: Organic / Stone / Gargoyle
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "gargoyle", "statue", "monument", "stone", "sculpture"):
            mat_stone = MaterialSlotConfig(
                name="M_Stone_Carved",
                color_rgb=[125, 128, 130],
                roughness=0.92,
                metallic=0.0,
                texture_type="stone",
                region_id="body",
                z_min_norm=0.0,
                z_max_norm=1.0,
                description="Weathered chiseled gothic stone"
            )
            return AppearanceProfile(category="organic", materials=[mat_stone], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case I: Lantern / Lamp (Brass & Glass)
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "lantern", "lamp", "torch", "candle"):
            mat_brass = MaterialSlotConfig(
                name="M_Victorian_Brass",
                color_rgb=[212, 175, 55],
                roughness=0.25,
                metallic=0.92,
                texture_type="brushed_metal",
                region_id="cage_and_ring",
                z_min_norm=0.0,
                z_max_norm=1.0,
                description="Ornate brass lantern frame and hanging ring"
            )
            mat_glass = MaterialSlotConfig(
                name="M_Clear_Glass",
                color_rgb=[230, 240, 245],
                roughness=0.05,
                metallic=0.05,
                texture_type="glass_tint",
                region_id="chamber",
                z_min_norm=0.25,
                z_max_norm=0.75,
                description="Clear glass oil chamber"
            )
            return AppearanceProfile(category="prop", materials=[mat_brass, mat_glass], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case J: Ship Anchor / Nautical Iron
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "anchor", "nautical", "fluke"):
            mat_anchor = MaterialSlotConfig(
                name="M_Cast_Iron",
                color_rgb=[40, 42, 45],
                roughness=0.72,
                metallic=0.90,
                texture_type="brushed_metal",
                region_id="anchor_body",
                z_min_norm=0.0,
                z_max_norm=1.0,
                description="Heavy black cast iron nautical ship anchor"
            )
            return AppearanceProfile(category="prop", materials=[mat_anchor], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case K: Canister / Container (Sci-Fi Coolant, Metal & Glass)
        # ---------------------------------------------------------------------
        elif has_word(prompt_lower, "canister", "container", "cylinder", "cryo", "tank"):
            mat_metal = MaterialSlotConfig(
                name="M_Pressurized_Metal",
                color_rgb=[140, 145, 150],
                roughness=0.28,
                metallic=0.88,
                texture_type="brushed_metal",
                region_id="canister_casing",
                z_min_norm=0.1,
                z_max_norm=1.0,
                description="Pressurized sci-fi alloy metal canister"
            )
            mat_gauge = MaterialSlotConfig(
                name="M_Gauge_Glass",
                color_rgb=[200, 235, 255],
                roughness=0.08,
                metallic=0.05,
                texture_type="glass_tint",
                region_id="pressure_gauge",
                z_min_norm=0.0,
                z_max_norm=0.2,
                description="Glass coolant chamber and pressure gauge"
            )
            return AppearanceProfile(category="prop", materials=[mat_metal, mat_gauge], dominant_palette=palette)

        # ---------------------------------------------------------------------
        # Case L: Generic Fallback
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
