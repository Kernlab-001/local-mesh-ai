"""
Accurate Color Calibration & Transfer Engine for Local Neural 3D Studio.
Aligns 3D mesh vertex colors and texture maps 1:1 with reference artwork
using perceptual Lab color space distribution matching (Reinhard transfer)
and saturation enhancement, mirroring Tripo AI high-fidelity fidelity.
"""

import os
import trimesh
import numpy as np
from PIL import Image
from typing import Optional
from skimage import color

class ColorCalibrator:
    def __init__(self):
        pass

    def extract_foreground_colors(self, image_path: str) -> np.ndarray:
        """
        Extracts foreground RGB colors from reference image.
        Uses alpha channel if present, otherwise rejects plain white/gray background pixels.
        """
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            arr = np.array(img).astype(np.float32)

        rgb = arr[..., :3] / 255.0
        alpha = arr[..., 3] / 255.0

        # Mask out transparent background
        mask = alpha > 0.2

        # If image had no transparent background, filter out pure white or neutral gray background
        if mask.sum() < 100:
            # Check for white/near-white or neutral gray background
            gray_diff = np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])
            is_white = (rgb[..., 0] > 0.95) & (rgb[..., 1] > 0.95) & (rgb[..., 2] > 0.95)
            is_gray = (gray_diff < 0.05) & (rgb[..., 0] > 0.45) & (rgb[..., 0] < 0.55)
            mask = ~(is_white | is_gray)

        if mask.sum() < 50:
            # Fallback to entire image if mask is too restrictive
            fg = rgb.reshape(-1, 3)
        else:
            fg = rgb[mask]

        return fg

    def transfer_color_distribution(
        self,
        target_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        vibrancy_boost: float = 1.15
    ) -> np.ndarray:
        """
        Transfers color distribution (mean and std in perceptual Lab color space)
        from reference_rgb to target_rgb with optional saturation/vibrancy boost.
        """
        if len(target_rgb) == 0 or len(reference_rgb) == 0:
            return target_rgb

        # Convert to Lab color space
        ref_lab = color.rgb2lab(reference_rgb[None, ...])[0]
        tgt_lab = color.rgb2lab(target_rgb[None, ...])[0]

        # Compute statistics in Lab
        ref_mean = ref_lab.mean(axis=0)
        ref_std = ref_lab.std(axis=0) + 1e-6

        tgt_mean = tgt_lab.mean(axis=0)
        tgt_std = tgt_lab.std(axis=0) + 1e-6

        # Standard Reinhard distribution shift & scale
        calibrated_lab = ((tgt_lab - tgt_mean) * (ref_std / tgt_std)) + ref_mean

        # Optional vibrancy boost on chroma channels (a, b)
        if vibrancy_boost != 1.0:
            calibrated_lab[:, 1] *= vibrancy_boost
            calibrated_lab[:, 2] *= vibrancy_boost

        # Convert back to sRGB and clamp
        calibrated_rgb = color.lab2rgb(calibrated_lab[None, ...])[0]
        return np.clip(calibrated_rgb, 0.0, 1.0)

    def calibrate_mesh_colors(
        self,
        mesh_path: str,
        reference_image_path: str,
        output_path: Optional[str] = None,
        vibrancy_boost: float = 1.15
    ) -> str:
        """
        Calibrates vertex colors of a GLB or OBJ mesh to match the exact palette
        and saturation of the source concept image.
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        if not os.path.exists(reference_image_path):
            return mesh_path

        out_path = output_path or mesh_path
        ref_fg = self.extract_foreground_colors(reference_image_path)
        scene = trimesh.load(mesh_path)

        geometries = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]

        for geom in geometries:
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
                current_colors = geom.visual.vertex_colors
                if len(current_colors) > 0:
                    tgt_rgb = current_colors[:, :3].astype(np.float32) / 255.0
                    calibrated_rgb = self.transfer_color_distribution(
                        tgt_rgb,
                        ref_fg,
                        vibrancy_boost=vibrancy_boost
                    )
                    
                    alpha = current_colors[:, 3:4] if current_colors.shape[1] > 3 else np.ones((len(tgt_rgb), 1), dtype=np.uint8) * 255
                    new_rgba = np.hstack([
                        (calibrated_rgb * 255.0).astype(np.uint8),
                        alpha
                    ])
                    geom.visual.vertex_colors = new_rgba

        scene.export(out_path)
        return out_path

    def calibrate_texture_image(
        self,
        texture_path: str,
        reference_image_path: str,
        output_path: Optional[str] = None,
        vibrancy_boost: float = 1.15
    ) -> str:
        """
        Calibrates a UV texture atlas image to match reference image palette.
        """
        if not os.path.exists(texture_path) or not os.path.exists(reference_image_path):
            return texture_path

        out_path = output_path or texture_path
        ref_fg = self.extract_foreground_colors(reference_image_path)

        with Image.open(texture_path) as img:
            arr = np.array(img.convert("RGBA")).astype(np.float32)

        h, w, c = arr.shape
        flat_rgb = (arr[..., :3] / 255.0).reshape(-1, 3)
        calibrated_rgb = self.transfer_color_distribution(flat_rgb, ref_fg, vibrancy_boost=vibrancy_boost)
        calibrated_arr = (calibrated_rgb.reshape(h, w, 3) * 255.0).astype(np.uint8)

        result_arr = np.dstack([calibrated_arr, arr[..., 3].astype(np.uint8)])
        Image.fromarray(result_arr).save(out_path)
        return out_path

color_calibrator = ColorCalibrator()
