"""
Unit tests for Prompt-to-Image Engine.
"""

import pytest
from core.prompt_to_image import prompt_engine

def test_prompt_preparation():
    raw_prompt = "  cyberpunk hover car  "
    enhanced = prompt_engine._prepare_prompt(raw_prompt)
    assert "cyberpunk hover car" in enhanced
    assert "white background" in enhanced
    # Verify strict CLIP token limit adherence
    tok = prompt_engine._get_tokenizer()
    if tok:
        tokens = tok.encode(enhanced)
        assert len(tokens) <= 77, f"Token length {len(tokens)} exceeded CLIP limit 77"

def test_prompt_engine_initialization():
    assert prompt_engine.model_id == "stabilityai/sd-turbo"
