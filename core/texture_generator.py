"""
Texture Generator for Local Mesh AI.
Generates realistic PBR texture sets (Diffuse Albedo, Tangent-Space Normal Map, Roughness Map)
for multi-material surfaces without external network calls or heavy weights.
Works in both standard Python and Blender's bundled Python.
"""

import os
import zlib
import struct
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

def save_png_array(arr: np.ndarray, filepath: str):
    """Saves uint8 HxWx3 or HxWx4 numpy array to PNG with pure Python zlib fallback."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if Image is not None:
        Image.fromarray(arr).save(filepath, format="PNG")
    else:
        h, w, c = arr.shape
        raw = b''.join(b'\x00' + arr[i].tobytes() for i in range(h))
        compressor = zlib.compressobj()
        idat = compressor.compress(raw) + compressor.flush()
        def chunk(tag, data):
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
        ihdr = struct.pack('>IIBBBBB', w, h, 8, 2 if c == 3 else 6, 0, 0, 0)
        with open(filepath, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b''))

class TextureGenerator:
    def __init__(self):
        pass

    def height_to_normal_map(self, height_map: np.ndarray, strength: float = 2.0) -> np.ndarray:
        """
        Converts a 2D scalar height map [0, 1] into a tangent-space Normal Map RGB [0, 255].
        """
        h, w = height_map.shape
        # Sobel / Central difference gradients
        dx = (np.roll(height_map, -1, axis=1) - np.roll(height_map, 1, axis=1)) * 0.5 * strength
        dy = (np.roll(height_map, -1, axis=0) - np.roll(height_map, 1, axis=0)) * 0.5 * strength
        dz = np.ones((h, w), dtype=np.float32)

        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        nx = -dx / norm
        ny = -dy / norm
        nz = dz / norm

        r = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        g = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        b = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        return np.dstack([r, g, b])

    def generate_wood_pbr(
        self,
        base_rgb: List[int],
        res: int = 1024,
        roughness_base: float = 0.35
    ) -> Dict[str, np.ndarray]:
        """Generates realistic wood grain PBR texture arrays (Albedo, Normal, Roughness)."""
        y, x = np.mgrid[0:res, 0:res]
        
        # Organic wood grain with eccentric ring distortion and cellular noise
        eccentric_x = (x - res * 0.35) * 0.12
        eccentric_y = y * 0.75
        dist = np.sqrt(eccentric_x**2 + eccentric_y**2)
        rings = np.sin(dist * 0.18 + np.sin(y * 0.04) * 2.5) * 0.5 + 0.5
        pores = (np.sin(x * 0.8) * np.cos(y * 0.2)) * 0.15
        wood_pattern = np.clip(rings + pores, 0.0, 1.0)

        # Diffuse Albedo with natural wood tone modulation
        base = np.array(base_rgb, dtype=float)
        diffuse_arr = np.zeros((res, res, 3), dtype=np.uint8)
        for c in range(3):
            # Modulate darkness for ring boundaries
            col = base[c] * (0.78 + 0.32 * wood_pattern)
            diffuse_arr[..., c] = np.clip(col, 0, 255).astype(np.uint8)

        # Tangent-space Normal map from wood fiber depth
        normal_arr = self.height_to_normal_map(wood_pattern, strength=2.2)

        # Roughness map: fiber pores are rougher (higher roughness), polished rings smoother
        rough_arr = np.clip((roughness_base * (0.8 + 0.4 * (1.0 - wood_pattern))) * 255.0, 0, 255).astype(np.uint8)
        roughness_rgb = np.dstack([rough_arr, rough_arr, rough_arr])

        return {
            "diffuse": diffuse_arr,
            "normal": normal_arr,
            "roughness": roughness_rgb
        }

    def generate_fabric_pbr(
        self,
        base_rgb: List[int],
        res: int = 1024,
        roughness_base: float = 0.85
    ) -> Dict[str, np.ndarray]:
        """Generates woven textile/upholstery PBR texture arrays (Albedo, Normal, Roughness)."""
        y, x = np.mgrid[0:res, 0:res]
        # Cross-woven warp and weft yarn pattern
        freq = 0.25
        warp = np.sin(x * freq) * 0.5 + 0.5
        weft = np.cos(y * freq) * 0.5 + 0.5
        weave = (warp * (1.0 - weft) + weft * (1.0 - warp)) * 0.8 + 0.1

        base = np.array(base_rgb, dtype=float)
        diffuse_arr = np.zeros((res, res, 3), dtype=np.uint8)
        for c in range(3):
            col = base[c] * (0.88 + 0.20 * weave)
            diffuse_arr[..., c] = np.clip(col, 0, 255).astype(np.uint8)

        normal_arr = self.height_to_normal_map(weave, strength=2.8)
        rough_arr = np.clip((roughness_base * (0.92 + 0.16 * weave)) * 255.0, 0, 255).astype(np.uint8)
        roughness_rgb = np.dstack([rough_arr, rough_arr, rough_arr])

        return {
            "diffuse": diffuse_arr,
            "normal": normal_arr,
            "roughness": roughness_rgb
        }

    def generate_metal_pbr(
        self,
        base_rgb: List[int],
        res: int = 1024,
        roughness_base: float = 0.22
    ) -> Dict[str, np.ndarray]:
        """Generates brushed anisotropic metal PBR texture arrays."""
        y, x = np.mgrid[0:res, 0:res]
        # Anisotropic linear brush lines
        scratches = np.sin(x * 1.5) * 0.5 + np.sin(x * 3.7 + np.sin(y * 0.02) * 2.0) * 0.5
        scratches = (scratches - scratches.min()) / (scratches.max() - scratches.min() + 1e-6)

        base = np.array(base_rgb, dtype=float)
        diffuse_arr = np.zeros((res, res, 3), dtype=np.uint8)
        for c in range(3):
            col = base[c] * (0.92 + 0.16 * scratches)
            diffuse_arr[..., c] = np.clip(col, 0, 255).astype(np.uint8)

        normal_arr = self.height_to_normal_map(scratches, strength=0.8)
        rough_arr = np.clip((roughness_base * (0.9 + 0.2 * scratches)) * 255.0, 0, 255).astype(np.uint8)
        roughness_rgb = np.dstack([rough_arr, rough_arr, rough_arr])

        return {
            "diffuse": diffuse_arr,
            "normal": normal_arr,
            "roughness": roughness_rgb
        }

    def generate_generic_pbr(
        self,
        base_rgb: List[int],
        texture_type: str = "custom",
        roughness_base: float = 0.40,
        res: int = 1024
    ) -> Dict[str, np.ndarray]:
        """Generates PBR textures based on requested texture type."""
        t_type = (texture_type or "custom").lower()
        if "wood" in t_type:
            return self.generate_wood_pbr(base_rgb, res=res, roughness_base=roughness_base)
        elif "fabric" in t_type or "cloth" in t_type or "upholster" in t_type:
            return self.generate_fabric_pbr(base_rgb, res=res, roughness_base=roughness_base)
        elif "metal" in t_type or "gold" in t_type or "steel" in t_type:
            return self.generate_metal_pbr(base_rgb, res=res, roughness_base=roughness_base)
        else:
            # Smooth subtle organic surface variation
            y, x = np.mgrid[0:res, 0:res]
            surface = np.sin(x * 0.05) * np.cos(y * 0.05) * 0.5 + 0.5
            base = np.array(base_rgb, dtype=float)
            diffuse_arr = np.zeros((res, res, 3), dtype=np.uint8)
            for c in range(3):
                col = base[c] * (0.94 + 0.12 * surface)
                diffuse_arr[..., c] = np.clip(col, 0, 255).astype(np.uint8)

            normal_arr = self.height_to_normal_map(surface, strength=0.6)
            rough_arr = np.clip((roughness_base * np.ones((res, res))) * 255.0, 0, 255).astype(np.uint8)
            roughness_rgb = np.dstack([rough_arr, rough_arr, rough_arr])

            return {
                "diffuse": diffuse_arr,
                "normal": normal_arr,
                "roughness": roughness_rgb
            }

    def save_pbr_textures(
        self,
        textures: Dict[str, Any],
        output_dir: str,
        prefix: str = "mat"
    ) -> Dict[str, str]:
        """Saves generated PBR texture images to output_dir and returns absolute paths."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        for key in ["diffuse", "normal", "roughness"]:
            if key in textures:
                target_path = os.path.join(output_dir, f"{prefix}_{key}.png")
                item = textures[key]
                if isinstance(item, np.ndarray):
                    save_png_array(item, target_path)
                elif Image is not None and isinstance(item, Image.Image):
                    item.save(target_path, format="PNG")
                elif hasattr(item, "save"):
                    item.save(target_path)
                paths[key] = os.path.abspath(target_path)
        return paths

texture_generator = TextureGenerator()
