"""
Tests for MedicalCollator.
No model downloads needed — uses a tiny tokenizer (GPT-2) loaded from cache.
"""
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from src.data.collator import MedicalCollator, _build_pathvqa_texts, _build_text_only_texts

CACHE = "artifacts/model_cache"
TOKENIZER_ID = "openai-community/gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, cache_dir=CACHE)
    tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def collator(tokenizer):
    # Identity preprocessor: returns a [3, 224, 224] zero tensor for any PIL image
    dummy_preprocessor = lambda img: torch.zeros(3, 224, 224)
    return MedicalCollator(tokenizer=tokenizer, image_preprocessor=dummy_preprocessor, max_length=128)


@pytest.fixture(scope="module")
def text_collator(tokenizer):
    return MedicalCollator(tokenizer=tokenizer, image_preprocessor=None, max_length=128)


# ── Prompt builders ───────────────────────────────────────────────────────────

def test_pathvqa_prompt_contains_image_token():
    prompt, _ = _build_pathvqa_texts({"question": "Is this normal?", "answer": "yes"})
    assert "<image>" in prompt

def test_pathvqa_full_contains_answer():
    _, full = _build_pathvqa_texts({"question": "Is this normal?", "answer": "yes"})
    assert "yes" in full

def test_text_only_full_contains_label():
    sample = {"prompt": "Question: What drug?\nA. Aspirin\nAnswer:", "label": "A"}
    _, full = _build_text_only_texts(sample)
    assert full.endswith("A")


# ── Label masking ─────────────────────────────────────────────────────────────

def test_labels_have_minus_100_for_prompt(collator):
    sample = {"question": "Is this normal?", "answer": "yes",
              "image": Image.new("RGB", (224, 224))}
    batch = collator([sample])
    labels = batch["labels"][0]
    # At least the first few tokens (prompt) should be masked
    assert (labels == -100).sum() > 0

def test_labels_have_answer_tokens(collator):
    sample = {"question": "Is this normal?", "answer": "yes",
              "image": Image.new("RGB", (224, 224))}
    batch = collator([sample])
    labels = batch["labels"][0]
    # At least some tokens should NOT be -100 (the answer tokens)
    assert (labels != -100).sum() > 0

def test_text_only_labels(text_collator):
    sample = {"prompt": "Question: Which drug?\nA. Aspirin\nAnswer:", "label": "A", "source": "medmcqa"}
    batch = text_collator([sample])
    labels = batch["labels"][0]
    assert (labels == -100).sum() > 0    # prompt masked
    assert (labels != -100).sum() > 0    # answer not masked


# ── Batch shapes ──────────────────────────────────────────────────────────────

def test_multimodal_batch_keys(collator):
    samples = [
        {"question": "Is this normal?", "answer": "yes", "image": Image.new("RGB", (224, 224))},
        {"question": "What do you see?", "answer": "cells", "image": Image.new("RGB", (300, 300))},
    ]
    batch = collator(samples)
    assert "input_ids"      in batch
    assert "attention_mask" in batch
    assert "labels"         in batch
    assert "pixel_values"   in batch

def test_multimodal_pixel_values_shape(collator):
    samples = [
        {"question": "Q1", "answer": "yes", "image": Image.new("RGB", (224, 224))},
        {"question": "Q2", "answer": "no",  "image": Image.new("L",   (100, 100))},
    ]
    batch = collator(samples)
    assert batch["pixel_values"].shape == (2, 3, 224, 224)

def test_text_batch_no_pixel_values(text_collator):
    samples = [
        {"prompt": "Question: A?\nAnswer:", "label": "A", "source": "medmcqa"},
        {"prompt": "Question: B?\nAnswer:", "label": "B", "source": "medmcqa"},
    ]
    batch = text_collator(samples)
    assert "pixel_values" not in batch

def test_batch_input_ids_same_length(collator):
    """Padding must make all sequences the same length."""
    samples = [
        {"question": "Short?", "answer": "yes", "image": Image.new("RGB", (224, 224))},
        {"question": "A much longer question about tissue morphology?",
         "answer": "abnormal", "image": Image.new("RGB", (224, 224))},
    ]
    batch = collator(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape == batch["labels"].shape

def test_grayscale_image_converted_to_rgb(collator):
    """Grayscale images must be converted to RGB before preprocessing."""
    sample = {"question": "Q?", "answer": "yes", "image": Image.new("L", (224, 224))}
    batch = collator([sample])
    assert batch["pixel_values"].shape == (1, 3, 224, 224)

def test_no_preprocessor_raises_on_image(tokenizer):
    collator_no_img = MedicalCollator(tokenizer=tokenizer, image_preprocessor=None)
    sample = {"question": "Q?", "answer": "yes", "image": Image.new("RGB", (224, 224))}
    with pytest.raises(RuntimeError, match="image_preprocessor is None"):
        collator_no_img([sample])
