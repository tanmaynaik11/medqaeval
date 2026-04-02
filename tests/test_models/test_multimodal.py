"""
MedVisionModel integration tests — CPU, uses GPT-2 + CLIP (no large downloads).
"""
import pytest
import torch
from src.models.multimodal import MedVisionModel, MedVisionConfig

CACHE = "artifacts/model_cache"


@pytest.fixture(scope="module")
def model():
    cfg = MedVisionConfig(
        encoder_type="clip",
        vision_hidden_dim=768,
        llm_model_id="openai-community/gpt2",
        llm_hidden_dim=768,
        proj_hidden_dim=256,
        lora_target_modules=["c_attn"],
        cache_dir=CACHE,
    )
    return MedVisionModel(cfg)


# ── Sanity ────────────────────────────────────────────────────────────────────

def test_model_loads(model):
    assert model is not None

def test_image_token_registered(model):
    assert model.image_token_id is not None
    assert model.image_token_id >= 0


# ── Trainable parameters ──────────────────────────────────────────────────────

def test_vision_encoder_frozen(model):
    for n, p in model.vision_encoder.named_parameters():
        assert not p.requires_grad, f"vision_encoder.{n} should be frozen"

def test_projection_trainable(model):
    for n, p in model.projection.named_parameters():
        assert p.requires_grad, f"projection.{n} should be trainable"

def test_only_lora_and_projection_train(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "lora" in name or "projection" in name, (
                f"Unexpected trainable param: {name}"
            )


# ── Text-only forward ─────────────────────────────────────────────────────────

def test_text_only_forward(model):
    tok = model.tokenizer(
        "Question: What is the diagnosis?\nAnswer:",
        return_tensors="pt",
        padding=True,
    )
    out = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
    )
    assert out.logits is not None
    assert out.logits.shape[0] == 1

def test_text_only_loss(model):
    tok = model.tokenizer(
        "Question: What is shown?\nAnswer: cells",
        return_tensors="pt",
    )
    labels = tok["input_ids"].clone()
    out = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        labels=labels,
    )
    assert out.loss is not None
    assert out.loss.item() > 0


# ── Multimodal forward ────────────────────────────────────────────────────────

def test_multimodal_forward(model):
    prompt = "<image>\nQuestion: What is shown?\nAnswer:"
    tok = model.tokenizer(prompt, return_tensors="pt")
    pixel_values = torch.zeros(1, 3, 224, 224)
    out = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        pixel_values=pixel_values,
    )
    assert out.logits is not None

def test_multimodal_sequence_length_grows(model):
    """After merge, sequence length = original - 1 (<image> token) + 49 patches."""
    prompt = "<image>\nQuestion: What?\nAnswer:"
    tok    = model.tokenizer(prompt, return_tensors="pt")
    S      = tok["input_ids"].shape[1]
    pixel_values = torch.zeros(1, 3, 224, 224)
    out = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        pixel_values=pixel_values,
    )
    expected_len = S - 1 + 49   # CLIP base-patch32 = 49 patches
    assert out.logits.shape[1] == expected_len

def test_multimodal_batch(model):
    """Batch of 2 multimodal samples."""
    prompts = ["<image>\nQuestion: What?\nAnswer:", "<image>\nQuestion: Is this normal?\nAnswer:"]
    tok     = model.tokenizer(prompts, return_tensors="pt", padding=True)
    pixel_values = torch.zeros(2, 3, 224, 224)
    out = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        pixel_values=pixel_values,
    )
    assert out.logits.shape[0] == 2
