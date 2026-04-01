"""Unit tests for preprocessing functions."""

import numpy as np
import pytest
from PIL import Image

from src.data.preprocessing import (
    get_image_transforms,
    build_vqa_prompt,
    build_mcqa_prompt,
    preprocess_pathvqa_sample,
    preprocess_medmcqa_sample,
)


# ── Image transforms ───────────────────────────────────────────────────────────

def test_image_transform_output_shape():
    transform = get_image_transforms(image_size=224)
    img = Image.new("RGB", (300, 400))
    out = transform(img)
    assert out.shape == (3, 224, 224)

def test_image_transform_converts_grayscale():
    transform = get_image_transforms(image_size=224)
    img = Image.new("L", (100, 100))   # grayscale
    out = transform(img)
    assert out.shape == (3, 224, 224)

def test_image_transform_normalized_range():
    transform = get_image_transforms(image_size=224)
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    out = transform(img)
    # After ImageNet normalization values should be near zero
    assert out.min() > -5.0 and out.max() < 5.0


# ── Prompt builders ────────────────────────────────────────────────────────────

def test_vqa_prompt_no_answer():
    p = build_vqa_prompt("Is this normal?")
    assert "Question: Is this normal?" in p
    assert "Answer:" in p
    assert "<image>" in p

def test_vqa_prompt_with_answer():
    p = build_vqa_prompt("Is this normal?", answer="yes")
    assert p.endswith("yes")

def test_mcqa_prompt_contains_options():
    opts = {"a": "Opt A", "b": "Opt B", "c": "Opt C", "d": "Opt D"}
    p = build_mcqa_prompt("Which is correct?", opts)
    for label in ["A.", "B.", "C.", "D."]:
        assert label in p

def test_mcqa_prompt_with_answer():
    opts = {"a": "Opt A", "b": "Opt B", "c": "Opt C", "d": "Opt D"}
    p = build_mcqa_prompt("Which?", opts, answer_key="b")
    assert p.endswith("B")


# ── Sample preprocessors ───────────────────────────────────────────────────────

def test_preprocess_pathvqa_sample():
    sample = {
        "image": Image.new("RGB", (100, 100)),
        "question": "What is shown?",
        "answer": "yes",
    }
    out = preprocess_pathvqa_sample(sample, image_size=224)
    assert out["pixel_values"].shape == (3, 224, 224)
    assert "Answer:" in out["prompt"]
    assert out["label"] == "yes"
    assert out["source"] == "path-vqa"

def test_preprocess_medmcqa_sample():
    sample = {
        "question": "Which drug?",
        "opa": "Aspirin",
        "opb": "Ibuprofen",
        "opc": "Paracetamol",
        "opd": "Metformin",
        "cop": 2,       # correct option index → "c"
        "exp": "Paracetamol is OTC.",
    }
    out = preprocess_medmcqa_sample(sample)
    assert out["label"] == "c"
    assert "Aspirin" in out["prompt"]
    assert out["source"] == "medmcqa"
