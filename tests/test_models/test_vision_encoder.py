"""
Tests for vision encoders.
CLIPVisionEncoder runs fully on CPU (fast).
BiomedCLIPEncoder test only validates loading + interface —
actual forward pass runs on CPU but downloads ~330MB on first run.
"""
import pytest
import torch
from src.models.vision_encoder import (
    CLIPVisionEncoder,
    BiomedCLIPEncoder,
    build_vision_encoder,
)

CACHE = "artifacts/model_cache"


# ── CLIPVisionEncoder ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clip_encoder():
    return CLIPVisionEncoder(
        model_id="openai/clip-vit-base-patch32",
        cache_dir=CACHE,
        frozen=True,
    )

def test_clip_patch_shape(clip_encoder):
    x = torch.zeros(2, 3, 224, 224)
    out = clip_encoder.get_patch_embeddings(x)
    assert out.shape == (2, 49, 768)   # base-patch32: 49 patches, hidden=768

def test_clip_frozen(clip_encoder):
    for p in clip_encoder.encoder.parameters():
        assert not p.requires_grad


# ── BiomedCLIPEncoder ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def biomed_encoder():
    return BiomedCLIPEncoder(cache_dir=CACHE, frozen=True)

def test_biomedclip_hidden_dim(biomed_encoder):
    assert biomed_encoder.hidden_dim == 768

def test_biomedclip_patch_shape(biomed_encoder):
    """ViT-B/16: 196 patches, hidden_dim=768."""
    x = torch.zeros(2, 3, 224, 224)
    out = biomed_encoder.get_patch_embeddings(x)
    assert out.shape == (2, 196, 768)

def test_biomedclip_frozen(biomed_encoder):
    for p in biomed_encoder.clip.parameters():
        assert not p.requires_grad

def test_biomedclip_no_grad_on_output(biomed_encoder):
    """Output must not carry gradients — trunk called inside torch.no_grad()."""
    x = torch.zeros(2, 3, 224, 224)
    out = biomed_encoder.get_patch_embeddings(x)
    assert not out.requires_grad

def test_biomedclip_uses_timm_trunk(biomed_encoder):
    """open_clip loads BiomedCLIP via TimmModel — verify .trunk exists."""
    assert hasattr(biomed_encoder.clip.visual, "trunk"), (
        "Expected TimmModel backend with .trunk. "
        "open_clip version may have changed the backend."
    )

def test_biomedclip_trunk_forward_features_shape(biomed_encoder):
    """trunk.forward_features must return [B, 197, 768] — CLS + 196 patches."""
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        all_tokens = biomed_encoder.clip.visual.trunk.forward_features(x)
    assert all_tokens.shape == (1, 197, 768)


# ── Factory ───────────────────────────────────────────────────────────────────

def test_factory_clip():
    enc = build_vision_encoder(
        "clip",
        cache_dir=CACHE,
        model_id="openai/clip-vit-base-patch32",
    )
    assert isinstance(enc, CLIPVisionEncoder)

def test_factory_biomedclip():
    enc = build_vision_encoder("biomedclip", cache_dir=CACHE)
    assert isinstance(enc, BiomedCLIPEncoder)

def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown encoder_type"):
        build_vision_encoder("resnet", cache_dir=CACHE)
