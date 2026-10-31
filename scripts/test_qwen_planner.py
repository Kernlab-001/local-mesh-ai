"""
Verification script for Qwen 2.5 Coder Technical-Artist Planner.
Tests generation and validation of both CommandPlan and TechnicalArtistPlan.
"""

import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.qwen_planner import qwen_planner

def main():
    print("=" * 70)
    print("  TESTING QWEN 2.5 CODER TECHNICAL-ARTIST PLANNER")
    print("=" * 70)

    prompt = "Double-edged steel broadsword with brass crossguard and brown leather grip"
    print(f"Prompt: {prompt}")
    print("Generating TechnicalArtistPlan (invoking Qwen under GPU lease)...")

    plan = qwen_planner.generate_technical_artist_plan(prompt)
    print("\n[SUCCESS] Successfully generated and validated TechnicalArtistPlan:")
    print(f"  Asset Name: {plan.asset_name}")
    print(f"  Category: {plan.category}")
    print(f"  Description: {plan.description}")
    print(f"  Components ({len(plan.components)}): {[c.name for c in plan.components]}")
    print(f"  Target Dimensions: {plan.dimensions.model_dump()}")
    print(f"  Materials ({len(plan.materials)}): {[m.material_id + ' (' + m.preset + ')' for m in plan.materials]}")
    print(f"  Topology Target: {plan.topology.target_triangles} tris, Style: {plan.topology.mesh_style}")

    # Convert to Blender CommandPlan
    cmd_plan = plan.to_command_plan()
    print(f"\n[Blender CommandPlan Conversion] Generated {len(cmd_plan.commands)} executable operations:")
    for i, cmd in enumerate(cmd_plan.commands, 1):
        print(f"    {i}. {cmd.op} -> {cmd.model_dump()}")

    print("\n" + "=" * 70)
    print("  ALL QWEN TECHNICAL-ARTIST TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
