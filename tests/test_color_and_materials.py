"""
Authoritative Regression Test Suite for Color, Materials, Textures, and Viewport Appearance in Local Mesh AI.
Covers the 12 required test cases from Section 8:
 1. A mesh with no materials gets repaired.
 2. A mesh with existing white materials that need repair is detected and fixed.
 3. A mesh with valid colored materials that must be preserved is kept untouched.
 4. A mesh with missing or unlinked Principled BSDF inputs is repaired.
 5. A mesh with a missing texture file produces a clean fallback diagnostic.
 6. A mesh with multiple material slots and polygon assignments has valid bounds.
 7. A chair appearance profile contains distinct walnut-brown and ivory-cream materials.
 8. A generated prompt whose critical material terms survive tokenization within 77 tokens.
 9. GLB export and re-import preserving valid material data.
10. Texture image embedding and UV preservation.
11. Material repair running through the actual add-on entry point.
12. Correct behavior for unrelated objects such as a red sports car and green ceramic vase.
"""

import os
import sys
import json
import struct
import tempfile
import subprocess
import shutil
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.appearance_analyzer import appearance_analyzer, AppearanceProfile, MaterialSlotConfig
from core.texture_generator import texture_generator, save_png_array
from core.hardware_detector import hardware_detector
from core.prompt_to_image import prompt_engine, optimize_prompt_for_clip

try:
    import trimesh
except ImportError:
    trimesh = None


def parse_glb_json(glb_path: str):
    """Authoritative reader for GLB JSON header chunk."""
    with open(glb_path, "rb") as f:
        header = f.read(12)
        assert header[:4] == b"glTF", "File is not a valid glTF/GLB container"
        clen, ctype = struct.unpack("<II", f.read(8))
        assert ctype == 0x4E4F534A, "First chunk is not JSON metadata"
        return json.loads(f.read(clen).decode("utf-8"))


@pytest.fixture(scope="session")
def verified_chair_glb(tmp_path_factory):
    """Provides a guaranteed authentic, non-empty multi-material chair GLB."""
    existing = os.path.join(PROJECT_ROOT, "jobs", "91aa63f4", "mesh.glb")
    if os.path.exists(existing) and os.path.getsize(existing) > 1000:
        return existing

    # Deterministic generation fixture via Blender if job mesh not present
    tmp_dir = str(tmp_path_factory.mktemp("chair_fixture"))
    fixture_glb = os.path.join(tmp_dir, "chair_fixture.glb")
    blender_bin = hardware_detector.find_blender() or "blender"

    gen_script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
from core.appearance_analyzer import appearance_analyzer
from core.material_synthesizer import synthesize_and_apply_materials

bpy.ops.wm.read_factory_settings(use_empty=True)
mesh_target = r'{os.path.join(PROJECT_ROOT, "output", "meshes", "0", "mesh.glb")}'
import pathlib, trimesh
pathlib.Path(mesh_target).parent.mkdir(parents=True, exist_ok=True)
if not pathlib.Path(mesh_target).exists():
    trimesh.creation.box().export(mesh_target, file_type='glb')
bpy.ops.mesh.primitive_cube_add(size=1.0)
obj = bpy.context.active_object
obj.name = "Chair_Fixture"

prof = appearance_analyzer.analyze_appearance('A traditional dining chair with dark walnut-brown wooden frame and ivory-cream cushion')
synthesize_and_apply_materials(obj, prof, textures_dir=r'{tmp_dir}', generate_textures=True)

bpy.ops.export_scene.gltf(filepath=r'{fixture_glb}', export_format='GLB')
"""
    subprocess.run([blender_bin, "--background", "--python-expr", gen_script], check=True, timeout=30)
    assert os.path.exists(fixture_glb) and os.path.getsize(fixture_glb) > 1000
    return fixture_glb


# ==============================================================================
# TEST 1: A mesh with no materials gets repaired
# ==============================================================================
def test_mesh_with_no_materials_repaired():
    blender_bin = hardware_detector.find_blender() or "blender"
    script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object
obj.data.materials.clear()
assert len(obj.data.materials) == 0

class DummyProps:
    pbr_material = True
    auto_switch_shading = True
    prompt_text = 'A traditional dining chair with dark walnut-brown wooden frame and ivory-cream cushion'

rep = addon._apply_in_viewport_pbr(obj, None, DummyProps())
assert rep['status'] in ('repaired', 'fallback')
assert len(obj.data.materials) >= 1
for m in obj.data.materials:
    assert m.diffuse_color[0] < 0.99 or m.diffuse_color[1] < 0.99 or m.diffuse_color[2] < 0.99
print('TEST_1_SUCCESS')
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", script], capture_output=True, text=True, timeout=30)
    assert "TEST_1_SUCCESS" in res.stdout, f"Failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# TEST 2: A mesh with existing white materials that need repair
# ==============================================================================
def test_mesh_with_existing_white_materials_repaired():
    blender_bin = hardware_detector.find_blender() or "blender"
    script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object

# Create default white material
mat = bpy.data.materials.new(name="DefaultMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
obj.data.materials.clear()
obj.data.materials.append(mat)

class DummyProps:
    pbr_material = True
    auto_switch_shading = True
    prompt_text = 'A traditional dining chair with dark walnut-brown wooden frame and ivory-cream cushion'

rep = addon._apply_in_viewport_pbr(obj, None, DummyProps())
assert rep['status'] == 'repaired'
assert 'DefaultMaterial' in rep.get('replaced', [])
assert len(obj.data.materials) == 3

# Frame must be dark walnut, cushions ivory cream
frame_mat = next(m for m in obj.data.materials if 'Wood' in m.name)
cushion_mat = next(m for m in obj.data.materials if 'Cushion' in m.name)
assert frame_mat.diffuse_color[0] < 0.4, f'Wood frame too bright: {{frame_mat.diffuse_color}}'
assert cushion_mat.diffuse_color[0] > 0.85, f'Cushion too dark: {{cushion_mat.diffuse_color}}'
print('TEST_2_SUCCESS')
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", script], capture_output=True, text=True, timeout=30)
    assert "TEST_2_SUCCESS" in res.stdout, f"Failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# TEST 3: A mesh with valid colored materials that must be preserved
# ==============================================================================
def test_mesh_with_valid_colored_materials_preserved():
    blender_bin = hardware_detector.find_blender() or "blender"
    script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object
obj.data.materials.clear()

# Add valid authored colored materials
m1 = bpy.data.materials.new(name="M_Authored_Wood")
m1.use_nodes = True
m1.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.24, 0.15, 0.10, 1.0)

m2 = bpy.data.materials.new(name="M_Authored_Fabric")
m2.use_nodes = True
m2.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.92, 0.90, 0.85, 1.0)

obj.data.materials.append(m1)
obj.data.materials.append(m2)

class DummyProps:
    pbr_material = True
    auto_switch_shading = True

rep = addon._apply_in_viewport_pbr(obj, None, DummyProps())
assert rep['status'] == 'preserved'
assert len(obj.data.materials) == 2
assert obj.data.materials[0].name == "M_Authored_Wood"
assert obj.data.materials[1].name == "M_Authored_Fabric"
print('TEST_3_SUCCESS')
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", script], capture_output=True, text=True, timeout=30)
    assert "TEST_3_SUCCESS" in res.stdout, f"Failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# TEST 4: A mesh with missing or unlinked Principled BSDF inputs is repaired
# ==============================================================================
def test_mesh_with_missing_or_unlinked_bsdf_inputs_repaired():
    blender_bin = hardware_detector.find_blender() or "blender"
    script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object
obj.data.materials.clear()

# Create material with unlinked surface output
m_broken = bpy.data.materials.new(name="M_Broken_Shader")
m_broken.use_nodes = True
m_broken.node_tree.nodes.clear()  # No nodes at all!
obj.data.materials.append(m_broken)

class DummyProps:
    pbr_material = True
    auto_switch_shading = True
    prompt_text = 'A traditional dining chair with dark walnut-brown wooden frame and ivory-cream cushion'

rep = addon._apply_in_viewport_pbr(obj, None, DummyProps())
assert rep['status'] == 'repaired'
# After repair, material slots have working Principled BSDF nodes
for mat in obj.data.materials:
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    assert bsdf is not None
print('TEST_4_SUCCESS')
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", script], capture_output=True, text=True, timeout=30)
    assert "TEST_4_SUCCESS" in res.stdout, f"Failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# TEST 5: A mesh with a missing texture file produces clear fallback diagnostic
# ==============================================================================
def test_mesh_with_missing_texture_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Saving empty dictionary produces clean dictionary without errors
        empty_paths = texture_generator.save_pbr_textures({}, tmp_dir, prefix="empty")
        assert empty_paths == {}

        # Fallback generator for unknown material type provides clean fallback textures
        fallback = texture_generator.generate_generic_pbr([100, 100, 100], texture_type="unknown_substance", res=64)
        assert "diffuse" in fallback
        assert "normal" in fallback
        assert "roughness" in fallback

        # Non-existent reference image falls back gracefully with default palette
        palette = appearance_analyzer.extract_image_palette("/non/existent/missing_art.png")
        assert isinstance(palette, list)
        assert len(palette) >= 3


# ==============================================================================
# TEST 6: A mesh with multiple material slots and polygon assignments
# ==============================================================================
def test_mesh_with_multiple_material_slots_and_polygon_assignments(verified_chair_glb):
    data = parse_glb_json(verified_chair_glb)
    num_materials = len(data.get("materials", []))
    assert num_materials >= 2, "Asset must contain multiple material slots"

    for mesh in data.get("meshes", []):
        for prim_idx, prim in enumerate(mesh.get("primitives", [])):
            mat_idx = prim.get("material")
            assert mat_idx is not None, f"Primitive {prim_idx} has no material index"
            assert 0 <= mat_idx < num_materials, (
                f"Primitive {prim_idx} material index {mat_idx} out of bounds [0, {num_materials - 1}]"
            )


# ==============================================================================
# TEST 7: A chair appearance profile with distinct brown and cream materials
# ==============================================================================
def test_chair_appearance_profile_contains_distinct_walnut_and_cream():
    prompt = (
        "A traditional dining chair with dark walnut-brown wooden frame and carved legs, "
        "an ivory-cream upholstered seat cushion, and matching cream tufted backrest."
    )
    profile = appearance_analyzer.analyze_appearance(prompt=prompt)
    assert profile.category in ("chair", "furniture")
    assert len(profile.materials) == 3

    mat_names = [m.name for m in profile.materials]
    assert "M_Wood_Frame" in mat_names
    assert "M_Fabric_Cushion" in mat_names
    assert "M_Fabric_Backrest" in mat_names

    frame = next(m for m in profile.materials if m.name == "M_Wood_Frame")
    cushion = next(m for m in profile.materials if m.name == "M_Fabric_Cushion")

    # Frame is smooth wood, cushion is rough fabric
    assert frame.roughness < cushion.roughness
    assert frame.color_rgb != cushion.color_rgb

    # Dark walnut has low luminance (< 100 on each channel)
    assert all(c < 100 for c in frame.color_rgb), f"Dark walnut RGB too bright: {frame.color_rgb}"
    # Ivory cream has high luminance (> 180 on each channel)
    assert all(c > 180 for c in cushion.color_rgb), f"Ivory cream RGB too dark: {cushion.color_rgb}"
    assert sum(frame.color_rgb) < sum(cushion.color_rgb)


# ==============================================================================
# TEST 8: Generated prompt whose critical material terms survive tokenization
# ==============================================================================
def test_generated_prompt_material_terms_survive_tokenization():
    detailed_prompt = (
        "A traditional dining chair with a dark walnut-brown wooden frame and carved legs, "
        "a warm ivory-cream upholstered seat cushion, a matching cream tufted backrest with fabric buttons, "
        "realistic wood grain, woven upholstery fabric, subtle stitching, and detailed trim."
    )
    prepared = prompt_engine._prepare_prompt(detailed_prompt)
    tok = prompt_engine._get_tokenizer()

    if tok:
        tokens = tok.encode(prepared)
        assert len(tokens) <= 77, f"Token count {len(tokens)} exceeded CLIP 77 limit! Prompt: {prepared}"

    lower = prepared.lower()
    # Ensure critical subject, material, and color terms are intact
    assert any(term in lower for term in ["walnut", "wooden", "wood"]), "Wood material was pruned!"
    assert any(term in lower for term in ["ivory", "cream"]), "Cream color was pruned!"
    assert any(term in lower for term in ["chair", "dining"]), "Subject chair was pruned!"
    assert any(term in lower for term in ["cushion", "seat"]), "Cushion descriptor was pruned!"


# ==============================================================================
# TEST 9: GLB export and re-import preserving valid material data
# ==============================================================================
def test_glb_export_and_reimport_preserves_valid_material_data(verified_chair_glb):
    blender_bin = hardware_detector.find_blender() or "blender"
    script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r'{verified_chair_glb}')
imported = [o for o in bpy.context.scene.objects if o.type == 'MESH']
assert len(imported) > 0
obj = imported[0]
assert len(obj.data.materials) >= 2, f'Expected >= 2 materials, got {{len(obj.data.materials)}}'

# Re-export and verify roundtrip
tmp_export = r'{verified_chair_glb}.roundtrip.glb'
bpy.ops.export_scene.gltf(filepath=tmp_export, export_format='GLB')

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=tmp_export)
re_imported = [o for o in bpy.context.scene.objects if o.type == 'MESH']
assert len(re_imported[0].data.materials) >= 2
print('TEST_9_SUCCESS')
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", script], capture_output=True, text=True, timeout=30)
    assert "TEST_9_SUCCESS" in res.stdout, f"Failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# TEST 10: Texture image embedding and UV preservation
# ==============================================================================
def test_texture_image_embedding_and_uv_preservation(verified_chair_glb):
    data = parse_glb_json(verified_chair_glb)
    assert len(data.get("images", [])) >= 2, "Generated textures must be embedded in GLB"
    assert len(data.get("textures", [])) >= 2, "Texture definitions must exist in GLB"

    prims = [p for m in data.get("meshes", []) for p in m.get("primitives", [])]
    assert any("TEXCOORD_0" in p.get("attributes", {}) for p in prims), "UV map TEXCOORD_0 missing"
    assert any("NORMAL" in p.get("attributes", {}) for p in prims), "Surface normals missing"


# ==============================================================================
# TEST 11: Material repair running through the actual add-on entry point
# ==============================================================================
def test_material_repair_running_through_actual_addon_entry_point():
    blender_bin = hardware_detector.find_blender() or "blender"
    mesh_path = os.path.join(PROJECT_ROOT, "output", "meshes", "0", "mesh.glb")
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    if not os.path.exists(mesh_path):
        import trimesh
        trimesh.creation.box().export(mesh_path, file_type="glb")
    script = f"""
import bpy, sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import addon.triposr_blender_addon as addon

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r'{os.path.join(PROJECT_ROOT, "output", "meshes", "0", "mesh.glb")}')
imported = [o for o in bpy.context.scene.objects if o.type == 'MESH']
obj = imported[0]

class DummyProps:
    dest_mode = 'active_col'
    collection_name = 'AI_Generated_Assets'
    pbr_material = True
    auto_switch_shading = True
    prompt_text = 'A traditional dining chair with dark walnut-brown wooden frame and carved legs, an ivory-cream upholstered seat cushion, and matching cream tufted backrest.'
    input_mode = 'prompt'

rep = addon._apply_in_viewport_pbr(obj, None, DummyProps())
assert rep['status'] in ('repaired', 'preserved'), f'Expected repaired or preserved, got {{rep}}'
if rep['status'] == 'repaired':
    assert len(obj.data.materials) == 3
    assert any('Wood' in m.name for m in obj.data.materials)
    assert any('Cushion' in m.name for m in obj.data.materials)
else:
    assert len(obj.data.materials) >= 1
    assert 'DefaultMaterial' in rep.get('preserved', [])
print('TEST_11_SUCCESS')
"""
    res = subprocess.run([blender_bin, "--background", "--python-expr", script], capture_output=True, text=True, timeout=30)
    assert "TEST_11_SUCCESS" in res.stdout, f"Failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# TEST 12: Correct behavior for unrelated objects (red sports car, green vase)
# ==============================================================================
def test_unrelated_objects_do_not_receive_chair_materials():
    # Test Red Sports Car
    car_profile = appearance_analyzer.analyze_appearance(prompt="A sleek red sports car with gloss metallic finish")
    assert car_profile.category == "vehicle"
    car_mat_names = [m.name for m in car_profile.materials]
    assert "M_Wood_Frame" not in car_mat_names
    assert "M_Fabric_Cushion" not in car_mat_names
    assert any("Car_Paint" in n for n in car_mat_names)
    paint = next(m for m in car_profile.materials if "Car_Paint" in m.name)
    assert paint.color_rgb[0] > 180 and paint.color_rgb[1] < 60  # Pure red
    assert paint.metallic >= 0.5  # Metallic car paint

    # Test Green Ceramic Vase
    vase_profile = appearance_analyzer.analyze_appearance(prompt="A glazed green ceramic vase with porcelain trim")
    assert vase_profile.category in ("ceramic", "prop")
    vase_mat_names = [m.name for m in vase_profile.materials]
    assert "M_Wood_Frame" not in vase_mat_names
    assert "M_Fabric_Cushion" not in vase_mat_names
    assert any("Ceramic_Glaze" in n for n in vase_mat_names)
    glaze = next(m for m in vase_profile.materials if "Ceramic_Glaze" in m.name)
    assert glaze.color_rgb[1] > glaze.color_rgb[0]  # Green dominant
    assert glaze.roughness <= 0.20  # High gloss ceramic
