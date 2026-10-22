"""
Unit tests for Prompt-to-Image Engine.
"""

import pytest
from core.prompt_to_image import prompt_engine

def test_prompt_preparation():
    raw_prompt = "  cyberpunk hover car  "
    enhanced = prompt_engine._prepare_prompt(raw_prompt)
    assert "cyberpunk hover car" in enhanced
    assert "solid white background" in enhanced
    assert "3d asset" in enhanced
    assert "isometric" in enhanced

def test_prompt_engine_initialization():
    assert prompt_engine.model_id == "stabilityai/sd-turbo"
