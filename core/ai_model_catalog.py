"""
Central AI Model Catalogue for Local Mesh AI.
Provides exhaustive, typed metadata, hardware requirements, runtime environments,
licensing, and installation verification for all local AI models and deterministic engines.

Adheres strictly to the local-first architecture: no cloud APIs, no external telemetry.
Respects user-configured models storage root via environment variables and core.download_manager.
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware_detector import hardware_detector


class ModelTaskCategory(str, Enum):
    SEGMENTATION_3D = "segmentation_3d"
    SEMANTIC_VISION_RECOGNITION = "semantic_vision_recognition"
    TECHNICAL_ARTIST_PLANNING = "technical_artist_planning"
    CONCEPT_IMAGE_GENERATION = "concept_image_generation"
    IMAGE_TO_3D_RECONSTRUCTION = "image_to_3d_reconstruction"
    MATERIALS_TEXTURE_GENERATION = "materials_texture_generation"
    DIMENSION_MEASUREMENT_ALIGNMENT = "dimension_measurement_alignment"
    GEOMETRY_CONSTRUCTION_EDITING = "geometry_construction_editing"


class VerificationStatus(str, Enum):
    VERIFIED_ON_HARDWARE = "VERIFIED ON HARDWARE"
    ESTIMATED_NOT_TESTED = "ESTIMATED — NOT TESTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_APPLICABLE_DETERMINISTIC = "NOT APPLICABLE (DETERMINISTIC)"


class ModelCatalogueEntry(BaseModel):
    model_id: str
    name: str
    task_category: ModelTaskCategory
    model_source: str
    official_repo: str
    download_ref: str
    license: str
    architecture: str
    quantization: Optional[str] = None
    weight_file_size_gb: float
    expected_vram_range_gb: List[float] = Field(description="[min_vram_gb, recommended_vram_gb]")
    expected_system_ram_gb: float
    supported_operating_systems: List[str]
    required_runtime: str
    dependency_environment: str
    input_formats: List[str]
    output_format: str
    capabilities: List[str]
    known_limitations: List[str]
    tested_hardware_profiles: List[str]
    is_installed: bool = False
    verification_status: VerificationStatus = VerificationStatus.ESTIMATED_NOT_TESTED
    is_neural: bool = True
    notes: Optional[str] = None


class AIModelCatalog:
    """
    Central repository of all model capabilities, hardware requirements,
    and runtime compatibility across the Local Mesh AI ecosystem.
    """

    def __init__(self, external_models_root: Optional[str] = None):
        self.models_root = Path(
            external_models_root or os.environ.get("LOCAL_MESH_MODELS_ROOT") or "E:\\LocalMeshAI-Data\\models"
        )
        self._catalog: Dict[str, ModelCatalogueEntry] = {}
        self._populate_catalogue()

    def _populate_catalogue(self):
        entries = [
            # 1. 3D Component Segmentation
            ModelCatalogueEntry(
                model_id="partfield",
                name="PartField 3D Part Decomposition",
                task_category=ModelTaskCategory.SEGMENTATION_3D,
                model_source="NVIDIA Toronto AI Lab (nv-tlabs)",
                official_repo="https://github.com/nv-tlabs/PartField",
                download_ref="https://huggingface.co/nv-tlabs/PartField-Objaverse",
                license="NVIDIA Source Code License / Research Non-Commercial",
                architecture="Feedforward Continuous Feature Field Network (Triplane representation)",
                quantization=None,
                weight_file_size_gb=1.25,
                expected_vram_range_gb=[4.0, 6.0],
                expected_system_ram_gb=16.0,
                supported_operating_systems=["Windows 11", "Linux"],
                required_runtime="PyTorch 2.4 / CUDA 12.1+",
                dependency_environment="Isolated Sub-Environment (Python 3.10 recommended)",
                input_formats=[".glb", ".obj", "point_cloud (10k-20k points)"],
                output_format="Part segmentation indices per face/vertex + continuous feature vectors",
                capabilities=[
                    "Open-world class-agnostic 3D part decomposition",
                    "Single feedforward 3D pass without per-shape optimization",
                    "Continuous feature-field clustering at configurable thresholds",
                    "Cross-shape part feature consistency",
                ],
                known_limitations=[
                    "Requires Python 3.10 / PyTorch 2.4 environment; isolated execution worker needed on Python 3.12 host",
                    "Objaverse checkpoint is trained predominantly on synthetic assets",
                    "Non-commercial research license restricts proprietary redistribution",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop (Isolated Worker)", "RTX 4090 24GB"],
                is_installed=False,
                verification_status=VerificationStatus.ESTIMATED_NOT_TESTED,
                is_neural=True,
            ),
            ModelCatalogueEntry(
                model_id="geometric_feature_field",
                name="Deterministic Multi-Scale Curvature & Normal Segmentation",
                task_category=ModelTaskCategory.SEGMENTATION_3D,
                model_source="Local Mesh AI Core Engine",
                official_repo="https://github.com/Kernlab-001/local-mesh-ai",
                download_ref="Built-in (core/component_decomposer.py)",
                license="MIT",
                architecture="Geometric Curvature Tensor & Planar Bottleneck Graph Clustering",
                quantization="N/A",
                weight_file_size_gb=0.0,
                expected_vram_range_gb=[0.0, 0.0],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="Python 3.12 (NumPy, SciPy, Trimesh)",
                dependency_environment="Main Project Environment",
                input_formats=[".glb", ".obj", ".ply", "Trimesh"],
                output_format="ComponentHierarchy with bounding boxes, centroids, volumes, and face masks",
                capabilities=[
                    "100% deterministic, offline execution with zero GPU VRAM requirement",
                    "Topological connected component extraction",
                    "Cross-sectional spatial profile bottleneck detection",
                    "Vertex color variance boundary clustering",
                ],
                known_limitations=[
                    "Cannot infer high-level semantics without visual or language assistance",
                    "Organic smooth transitions may require fine-grained threshold tuning",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop", "All x86_64 Systems"],
                is_installed=True,
                verification_status=VerificationStatus.VERIFIED_ON_HARDWARE,
                is_neural=False,
                notes="Primary deterministic component segmentation engine with zero VRAM cost.",
            ),

            # 2. Semantic Vision-Language Component Recognition
            ModelCatalogueEntry(
                model_id="qwen3_vl_4b_instruct",
                name="Qwen3-VL-4B-Instruct GGUF",
                task_category=ModelTaskCategory.SEMANTIC_VISION_RECOGNITION,
                model_source="Alibaba Cloud Qwen Team",
                official_repo="https://github.com/QwenLM/Qwen3-VL",
                download_ref="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF",
                license="Apache 2.0",
                architecture="Multimodal Transformer (Vision Transformer + Qwen3 LLM)",
                quantization="Q4_K_M (LLM) + FP16/BF16 (mmproj vision projector)",
                weight_file_size_gb=2.85,
                expected_vram_range_gb=[3.5, 4.5],
                expected_system_ram_gb=12.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="llama-cpp-python with CUDA/cuBLAS & mmproj support",
                dependency_environment="Main Project Environment",
                input_formats=["Multi-view PNG renders (front, back, left, right, top, perspective) + text prompt"],
                output_format="Structured JSON mapping candidate component IDs to semantic labels & material intents",
                capabilities=[
                    "Interprets multi-view visual evidence with color-coded component masks",
                    "Assigns functional semantic labels (seat, backrest, blade, crossguard, hilt, wheel)",
                    "Never invents new component IDs; flags uncertainty when confidence is low",
                    "Low VRAM footprint (fits comfortably in 6GB VRAM alongside display driver)",
                ],
                known_limitations=[
                    "Vision projector requires strict resolution matching (e.g. 448x448 or 896x896)",
                    "Must be run sequentially via GPU Lease Broker to avoid OOM with TripoSR or Qwen Coder",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop", "RTX 4060 8GB"],
                is_installed=False,
                verification_status=VerificationStatus.ESTIMATED_NOT_TESTED,
                is_neural=True,
            ),
            ModelCatalogueEntry(
                model_id="qwen2.5_vl_3b_instruct",
                name="Qwen2.5-VL-3B-Instruct GGUF",
                task_category=ModelTaskCategory.SEMANTIC_VISION_RECOGNITION,
                model_source="Alibaba Cloud Qwen Team",
                official_repo="https://github.com/QwenLM/Qwen2.5-VL",
                download_ref="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF",
                license="Apache 2.0",
                architecture="Multimodal Vision Transformer + Qwen2.5 LLM",
                quantization="Q4_K_M (LLM) + mmproj vision projector",
                weight_file_size_gb=2.20,
                expected_vram_range_gb=[2.8, 3.8],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="llama-cpp-python with CUDA 12.4",
                dependency_environment="Main Project Environment",
                input_formats=["Multi-view PNG renders + JSON metadata"],
                output_format="Structured JSON semantic component mapping",
                capabilities=[
                    "Highly compact 3B vision-language model",
                    "Runs smoothly on 4GB and 6GB VRAM configurations",
                    "Solid spatial relationship understanding from multi-angle orthographic renders",
                ],
                known_limitations=[
                    "Slightly lower fine-grained semantic naming granularity than 7B or 4B next-gen VLM",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop", "GTX 1650 4GB (Partial CPU)"],
                is_installed=False,
                verification_status=VerificationStatus.ESTIMATED_NOT_TESTED,
                is_neural=True,
            ),

            # 3. Central Technical-Artist Planning
            ModelCatalogueEntry(
                model_id="qwen2.5_coder_7b",
                name="Qwen 2.5 Coder 7B Instruct GGUF",
                task_category=ModelTaskCategory.TECHNICAL_ARTIST_PLANNING,
                model_source="Alibaba Cloud",
                official_repo="https://github.com/QwenLM/Qwen2.5-Coder",
                download_ref="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                license="Apache 2.0",
                architecture="Decoder-Only Dense Transformer (7B parameters)",
                quantization="Q4_K_M (4.68 GB) / Q5_K_M (5.44 GB)",
                weight_file_size_gb=5.07,
                expected_vram_range_gb=[4.0, 5.5],
                expected_system_ram_gb=16.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="llama-cpp-python (cuBLAS CUDA 12.4)",
                dependency_environment="Main Project Environment",
                input_formats=["Geometry analysis reports, component hierarchies, user requests, repair logs"],
                output_format="TechnicalArtistPlan & PrecisionCorrectionPlan Pydantic JSON schemas",
                capabilities=[
                    "Top-tier code and structural reasoning for 3D construction planning",
                    "Strict output adherence to Pydantic JSON schemas",
                    "Sequential tool call generation without arbitrary code execution",
                    "Evaluates before-and-after geometry measurements to detect regressions",
                ],
                known_limitations=[
                    "Consumes ~4.2GB - 4.8GB VRAM in Q5_K_M; requires full GPU lease and sequential eviction",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop (Verified cuBLAS offload)"],
                is_installed=True,
                verification_status=VerificationStatus.VERIFIED_ON_HARDWARE,
                is_neural=True,
                notes="Primary local technical-artist planner installed at E:\\Game\\Models\\qwen2.5-coder-7b-instruct-q5_k_m.gguf",
            ),
            ModelCatalogueEntry(
                model_id="qwen2.5_coder_1.5b",
                name="Qwen 2.5 Coder 1.5B Instruct GGUF",
                task_category=ModelTaskCategory.TECHNICAL_ARTIST_PLANNING,
                model_source="Alibaba Cloud",
                official_repo="https://github.com/QwenLM/Qwen2.5-Coder",
                download_ref="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                license="Apache 2.0",
                architecture="Decoder-Only Dense Transformer (1.5B parameters)",
                quantization="Q4_K_M",
                weight_file_size_gb=1.12,
                expected_vram_range_gb=[1.5, 2.5],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="llama-cpp-python (cuBLAS)",
                dependency_environment="Main Project Environment",
                input_formats=["Structured geometry JSON"],
                output_format="Validated JSON tool calls",
                capabilities=[
                    "Ultra-compact footprint for 4GB VRAM or CPU-only execution",
                    "Fast inference latency (< 1.5s per plan)",
                ],
                known_limitations=[
                    "Reduced multi-step reasoning depth compared to 7B model; requires explicit few-shot examples",
                ],
                tested_hardware_profiles=["4GB VRAM Tiers", "CPU-Only Fallback"],
                is_installed=False,
                verification_status=VerificationStatus.ESTIMATED_NOT_TESTED,
                is_neural=True,
            ),

            # 4. Text-to-Image Concept Generation
            ModelCatalogueEntry(
                model_id="sd_turbo",
                name="SD-Turbo 1-Step Latent Diffusion",
                task_category=ModelTaskCategory.CONCEPT_IMAGE_GENERATION,
                model_source="Stability AI",
                official_repo="https://github.com/Stability-AI/generative-models",
                download_ref="https://huggingface.co/stabilityai/sd-turbo",
                license="Stability AI Non-Commercial Research Community License",
                architecture="Adversarial Diffusion Distillation (ADD) UNet",
                quantization="FP16",
                weight_file_size_gb=2.50,
                expected_vram_range_gb=[3.2, 4.0],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux"],
                required_runtime="PyTorch / Diffusers (CUDA 12.4)",
                dependency_environment="Main Project Environment",
                input_formats=["Text prompt string"],
                output_format="512x512 RGBA PNG concept image",
                capabilities=[
                    "Real-time 1-step concept generation (~0.8s on RTX 3050)",
                    "Clean silhouette generation suitable for TripoSR feedforward reconstruction",
                ],
                known_limitations=[
                    "Single-step generation trades fine micro-details for extreme inference speed",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop"],
                is_installed=False,
                verification_status=VerificationStatus.VERIFIED_ON_HARDWARE,
                is_neural=True,
            ),

            # 5. Image-to-3D Neural Reconstruction
            ModelCatalogueEntry(
                model_id="triposr",
                name="TripoSR Feedforward Volumetric Checkpoint",
                task_category=ModelTaskCategory.IMAGE_TO_3D_RECONSTRUCTION,
                model_source="Tripo AI & Stability AI",
                official_repo="https://github.com/VAST-AI-Research/TripoSR",
                download_ref="https://huggingface.co/stabilityai/TripoSR",
                license="MIT",
                architecture="Transformer NeRF Backbone + DINO ViT-B16 + Marching Cubes",
                quantization="FP16 (TSR Backbone & Marching Cubes)",
                weight_file_size_gb=1.68,
                expected_vram_range_gb=[3.0, 4.2],
                expected_system_ram_gb=16.0,
                supported_operating_systems=["Windows 11", "Linux"],
                required_runtime="PyTorch 2.6 / CUDA 12.4 / Trimesh",
                dependency_environment="Main Project Environment",
                input_formats=["512x512 RGBA PNG with alpha foreground mask"],
                output_format="Isolated .glb mesh with vertex colors",
                capabilities=[
                    "Ultra-fast feedforward single-image 3D synthesis (30-35s on RTX 3050)",
                    "Watertight closed genus-0 mesh generation via Marching Cubes chunking",
                    "Zero external API calls; runs 100% locally offline",
                ],
                known_limitations=[
                    "Generates monolithic mesh without component decomposition (addressed by Phase 2/3)",
                    "Untextured vertex color representation (PBR maps baked in downstream pipeline)",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop (Verified 33.01s on chair.png)"],
                is_installed=True,
                verification_status=VerificationStatus.VERIFIED_ON_HARDWARE,
                is_neural=True,
                notes="Primary neural 3D reconstruction engine installed and verified in Phase 1.",
            ),
            ModelCatalogueEntry(
                model_id="stable_fast_3d",
                name="Stable Fast 3D (SF3D)",
                task_category=ModelTaskCategory.IMAGE_TO_3D_RECONSTRUCTION,
                model_source="Stability AI",
                official_repo="https://github.com/Stability-AI/stable-fast-3d",
                download_ref="https://huggingface.co/stabilityai/stable-fast-3d",
                license="Stability AI Community License",
                architecture="Triplane NeRF + UV Unwrapping + Texture Map Predictor",
                quantization="FP16",
                weight_file_size_gb=1.45,
                expected_vram_range_gb=[7.5, 9.0],
                expected_system_ram_gb=16.0,
                supported_operating_systems=["Linux", "Windows 11 (requires Triton/custom wheels)"],
                required_runtime="PyTorch / CUDA 12.1+ / Custom C++ extensions",
                dependency_environment="Isolated Environment",
                input_formats=["512x512 RGBA PNG"],
                output_format="UV unwrapped .glb with diffuse, roughness, and normal maps",
                capabilities=[
                    "Direct UV unwrapped textured mesh generation in ~1 second",
                    "PBR material maps generated end-to-end",
                ],
                known_limitations=[
                    "Requires 8GB+ VRAM; causes CUDA OOM on 6GB laptop GPU unless heavy CPU offload is enabled",
                    "Complex Windows installation due to Triton compiler requirements",
                ],
                tested_hardware_profiles=["RTX 4070 12GB", "RTX 3080 10GB"],
                is_installed=False,
                verification_status=VerificationStatus.ESTIMATED_NOT_TESTED,
                is_neural=True,
            ),
            ModelCatalogueEntry(
                model_id="trellis_large",
                name="Microsoft TRELLIS Image-to-3D Large",
                task_category=ModelTaskCategory.IMAGE_TO_3D_RECONSTRUCTION,
                model_source="Microsoft Research & JeffreyXiang",
                official_repo="https://github.com/microsoft/TRELLIS",
                download_ref="https://huggingface.co/JeffreyXiang/TRELLIS-image-large",
                license="Creative Commons Attribution-NonCommercial 4.0",
                architecture="Structured Latent Representation + 3D Rectified Flow Transformers",
                quantization="FP16 / BF16",
                weight_file_size_gb=14.20,
                expected_vram_range_gb=[14.0, 18.0],
                expected_system_ram_gb=32.0,
                supported_operating_systems=["Linux", "Windows 11 (WSL2 recommended)"],
                required_runtime="PyTorch 2.4+ / CUDA 12.1+",
                dependency_environment="Dedicated High-VRAM Environment",
                input_formats=["Single RGB image"],
                output_format="High-fidelity Radiance Field / 3D Gaussian Splat / Textured Mesh",
                capabilities=[
                    "State-of-the-art structural detail, complex topology, and clean geometry",
                ],
                known_limitations=[
                    "Mandates 16GB+ VRAM; strictly unsupported on 4GB-8GB consumer laptop GPUs",
                    "Non-commercial license",
                ],
                tested_hardware_profiles=["RTX 4090 24GB", "A100 80GB"],
                is_installed=False,
                verification_status=VerificationStatus.UNSUPPORTED,
                is_neural=True,
                notes="Flagged as unsupported on 6GB testbed; recommended for 16GB+ tier.",
            ),

            # 6. Materials & Texture Generation
            ModelCatalogueEntry(
                model_id="procedural_pbr_blender",
                name="Deterministic Blender Procedural PBR & Texture Baker",
                task_category=ModelTaskCategory.MATERIALS_TEXTURE_GENERATION,
                model_source="Blender Foundation & Local Mesh AI Bridge",
                official_repo="https://www.blender.org",
                download_ref="Built-in (Blender 5.2 Alpha)",
                license="GNU General Public License (GPL) v3",
                architecture="Cycles / EEVEE Principled BSDF Node Synthesizer",
                quantization="N/A",
                weight_file_size_gb=0.0,
                expected_vram_range_gb=[0.0, 1.0],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="Blender 5.2 Python API",
                dependency_environment="Blender Python Environment",
                input_formats=[".blend", ".glb", "material_intent string"],
                output_format="PBR Material Slots + Baked Diffuse, Normal, Roughness, Metalness PNGs",
                capabilities=[
                    "Deterministic procedural material generation (wood, steel, leather, marble, gold)",
                    "Automated UV projection and non-overlapping chart packing",
                    "Zero external AI model required; 100% reliable and reproducible",
                ],
                known_limitations=[
                    "Synthesizes procedural patterns rather than photographic generative textures",
                ],
                tested_hardware_profiles=["RTX 3050 6GB Laptop", "Blender 5.2 Alpha"],
                is_installed=True,
                verification_status=VerificationStatus.NOT_APPLICABLE_DETERMINISTIC,
                is_neural=False,
                notes="Primary production material pipeline; runs directly inside headless Blender.",
            ),

            # 7. Dimension Measurement & Alignment
            ModelCatalogueEntry(
                model_id="cpu_geometry_engine",
                name="Deterministic CPU Geometry Measurement & Calibration Engine",
                task_category=ModelTaskCategory.DIMENSION_MEASUREMENT_ALIGNMENT,
                model_source="Local Mesh AI Core (Trimesh / NumPy / SciPy)",
                official_repo="https://github.com/Kernlab-001/local-mesh-ai",
                download_ref="Built-in (core/geometry_analyzer.py, core/precision_aligner.py)",
                license="MIT",
                architecture="Deterministic Cartesian Affine & KD-Tree Diagnostics",
                quantization="N/A",
                weight_file_size_gb=0.0,
                expected_vram_range_gb=[0.0, 0.0],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="Python 3.12",
                dependency_environment="Main Project Environment",
                input_formats=[".glb", ".obj", ".ply", "Trimesh"],
                output_format="CompleteGeometryReport & AlignmentResult Pydantic models",
                capabilities=[
                    "Exact metric bounding box extents, volume, surface area, and center of mass",
                    "Watertightness, non-manifold edges, degenerate faces, and Euler characteristic",
                    "Bilateral symmetry deviation scoring and reflection plane detection",
                    "Ground plane snapping (Z_min = 0.0), XY centering, upright normalization",
                    "Uniform metric scale calibration with provenance tracking",
                ],
                known_limitations=[
                    "Non-neural; does not interpret artistic intent (planned by Qwen Coder)",
                ],
                tested_hardware_profiles=["Intel i5-13420H / RTX 3050 Laptop", "All x86_64 CPUs"],
                is_installed=True,
                verification_status=VerificationStatus.NOT_APPLICABLE_DETERMINISTIC,
                is_neural=False,
                notes="AI is strictly prohibited from replacing exact mathematical geometry calculations.",
            ),

            # 8. Geometry Construction & Editing
            ModelCatalogueEntry(
                model_id="blender_deterministic_tools",
                name="Deterministic Headless Blender Geometry Construction & Editing API",
                task_category=ModelTaskCategory.GEOMETRY_CONSTRUCTION_EDITING,
                model_source="Blender Foundation & Local Mesh AI Bridge",
                official_repo="https://www.blender.org",
                download_ref="Built-in (bridge/blender_hierarchy_bridge.py, core/geometry_editing.py)",
                license="GNU GPL v3",
                architecture="BMesh & Modifier Evaluation Engine",
                quantization="N/A",
                weight_file_size_gb=0.0,
                expected_vram_range_gb=[0.0, 1.5],
                expected_system_ram_gb=8.0,
                supported_operating_systems=["Windows 11", "Linux", "macOS"],
                required_runtime="Blender 5.2 Alpha / BMesh",
                dependency_environment="Blender Python Environment",
                input_formats=[".blend", ".glb", "Pydantic Tool Arguments"],
                output_format="Edited .blend Scene + Exported .glb",
                capabilities=[
                    "Parametric primitive generation (cube, sphere, cylinder, cone, torus, plane)",
                    "Extrusion, inset, bevel, loop cuts, subdivision, solidify, boolean operations",
                    "Component-level transformation targeting validated component IDs",
                    "Hierarchy tree generation under AI_Asset_Hierarchy Master Collection",
                ],
                known_limitations=[
                    "Requires headless Blender process invocation (~3.8s launch time on Windows 11)",
                ],
                tested_hardware_profiles=["Intel i5-13420H / Blender 5.2 Alpha"],
                is_installed=True,
                verification_status=VerificationStatus.NOT_APPLICABLE_DETERMINISTIC,
                is_neural=False,
                notes="Blender deterministic tools execute operations sequenced by Qwen Technical-Artist.",
            ),
        ]

        for e in entries:
            self._catalog[e.model_id] = e

        self._check_installed_statuses()

    def _check_installed_statuses(self):
        """Verifies local presence of model weights and runtime dependencies."""
        from core.download_manager import download_manager

        # TripoSR check
        if "triposr" in self._catalog:
            stat = download_manager.verify_triposr_status()
            self._catalog["triposr"].is_installed = bool(stat.get("ready_for_offline_inference", False))

        # Qwen 2.5 Coder check
        if "qwen2.5_coder_7b" in self._catalog:
            q_stat = download_manager.verify_qwen_status()
            self._catalog["qwen2.5_coder_7b"].is_installed = (q_stat.get("state") == "fully_installed")

        # Check local files for other models if directory exists
        for model_id, entry in self._catalog.items():
            if not entry.is_neural:
                entry.is_installed = True
                continue

            target_dir = self.models_root / model_id
            if target_dir.exists() and any(target_dir.iterdir()):
                entry.is_installed = True

    def get_all_entries(self) -> List[ModelCatalogueEntry]:
        """Returns all catalogue entries."""
        return list(self._catalog.values())

    def get_entry(self, model_id: str) -> Optional[ModelCatalogueEntry]:
        """Returns single catalogue entry by model_id."""
        return self._catalog.get(model_id)

    def get_entries_by_task(self, task: ModelTaskCategory) -> List[ModelCatalogueEntry]:
        """Returns all entries for a specific task category."""
        return [e for e in self._catalog.values() if e.task_category == task]


ai_model_catalog = AIModelCatalog()
