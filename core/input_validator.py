"""
Shared Input & Asset Validation Layer for Local Mesh AI.
Provides strict, deterministic validation for both creation paths:
- Mode A: Text-to-3D (Prompt sanitization & generated concept verification)
- Mode B: Image-to-3D (File inspection, PIL readability, dimension & depth checks)
Ensures untrusted inputs never cause unhandled crashes or path traversal exploits.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_DIMENSION = 64
MAX_DIMENSION = 4096
MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 Megabytes

class ValidationError(ValueError):
    """Raised when an input asset fails validation constraints."""
    pass

class InputValidator:
    """
    Unified validation authority for uploaded photographs and generated concept art.
    """

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Strips path traversal attempts and invalid OS characters."""
        base = os.path.basename(filename)
        # Remove any character not alphanumeric, dot, underscore, or hyphen
        clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
        return clean or "concept_image.png"

    @staticmethod
    def validate_prompt(prompt: Optional[str]) -> str:
        """
        Validates and sanitizes text prompts for Mode A (Text-to-3D).
        """
        if not prompt or not isinstance(prompt, str):
            raise ValidationError("Prompt must be a non-empty string.")

        cleaned = prompt.strip()
        if len(cleaned) < 2:
            raise ValidationError(f"Prompt is too short ({len(cleaned)} chars). Minimum 2 characters required.")
        if len(cleaned) > 1000:
            raise ValidationError(f"Prompt exceeds maximum length of 1000 characters ({len(cleaned)} chars).")

        return cleaned

    @classmethod
    def validate_uploaded_image(
        cls,
        source_path: str,
        destination_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mode B: Image-to-3D validation.
        Validates file existence, actual file headers (ignoring spoofed extensions),
        readability, byte bounds, dimensions, and safely stages a copy into destination_dir.
        """
        if not source_path or not isinstance(source_path, str):
            raise ValidationError("Invalid source image path provided.")

        p = Path(source_path).resolve()
        if not p.exists() or not p.is_file():
            raise ValidationError(f"Uploaded image does not exist or is not a regular file: {p}")

        # 1. Byte Size Check
        file_size = p.stat().st_size
        if file_size == 0:
            raise ValidationError(f"Uploaded file is empty (0 bytes): {p.name}")
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = round(file_size / (1024 * 1024), 2)
            max_mb = round(MAX_FILE_SIZE_BYTES / (1024 * 1024), 2)
            raise ValidationError(f"Uploaded image ({size_mb} MB) exceeds maximum permitted size of {max_mb} MB.")

        # 2. Extension Check
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValidationError(f"Unsupported file extension '{ext}'. Permitted extensions: {sorted(SUPPORTED_EXTENSIONS)}")

        # 3. Content & Readability Check via PIL
        try:
            with Image.open(p) as img:
                img_format = img.format
                width, height = img.size
                mode = img.mode

                if img_format not in SUPPORTED_IMAGE_FORMATS:
                    raise ValidationError(f"File content format '{img_format}' is not supported. Required: {SUPPORTED_IMAGE_FORMATS}")

                if width < MIN_DIMENSION or height < MIN_DIMENSION:
                    raise ValidationError(f"Image resolution {width}x{height} is too small. Minimum resolution is {MIN_DIMENSION}x{MIN_DIMENSION}.")

                if width > MAX_DIMENSION or height > MAX_DIMENSION:
                    raise ValidationError(f"Image resolution {width}x{height} exceeds maximum bounds of {MAX_DIMENSION}x{MAX_DIMENSION}.")

                has_alpha = mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info)

        except UnidentifiedImageError:
            raise ValidationError(f"File '{p.name}' is corrupt or cannot be parsed as a valid image.")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Image validation failed: {str(e)}")

        # 4. Safe Workspace Isolation
        safe_path = str(p)
        if destination_dir:
            os.makedirs(destination_dir, exist_ok=True)
            clean_name = cls.sanitize_filename(p.name)
            target_path = os.path.join(destination_dir, clean_name)
            if str(p) != target_path:
                shutil.copy2(str(p), target_path)
            safe_path = target_path

        return {
            "valid": True,
            "format": img_format,
            "width": width,
            "height": height,
            "mode": mode,
            "has_alpha": has_alpha,
            "size_bytes": file_size,
            "source_path": str(p),
            "staged_path": safe_path
        }

    @classmethod
    def validate_generated_concept(cls, generated_path: str) -> Dict[str, Any]:
        """
        Mode A: Text-to-3D verification.
        Confirms the concept diffusion model produced a genuine, readable image.
        """
        if not generated_path or not os.path.exists(generated_path):
            raise ValidationError(f"Concept image generation failed: file not written to '{generated_path}'.")

        p = Path(generated_path).resolve()
        if p.stat().st_size == 0:
            raise ValidationError(f"Generated concept image is an empty 0-byte file: '{p.name}'.")

        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                w, h = img.size
                return {
                    "valid": True,
                    "format": img.format,
                    "width": w,
                    "height": h,
                    "mode": img.mode,
                    "size_bytes": p.stat().st_size,
                    "path": str(p)
                }
        except Exception as err:
            raise ValidationError(f"Generated concept image is corrupt: {err}")

input_validator = InputValidator()
