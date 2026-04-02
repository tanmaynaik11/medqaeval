"""Unit tests for VisionProjection — all run on CPU, no downloads needed."""

import pytest
import torch
import torch.nn as nn
from src.models.projection import VisionProjection


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_output_shape_2layer():
    """Standard BiomedCLIP → Qwen2-VL config."""
    proj = VisionProjection(in_dim=768, out_dim=3584, num_layers=2)
    x   = torch.randn(2, 196, 768)   # batch=2, 196 patches, hidden=768
    out = proj(x)
    assert out.shape == (2, 196, 3584)

def test_output_shape_1layer():
    """Linear-only projection (num_layers=1)."""
    proj = VisionProjection(in_dim=768, out_dim=768, num_layers=1)
    x   = torch.randn(1, 49, 768)
    assert proj(x).shape == (1, 49, 768)

def test_output_shape_local_dev():
    """BiomedCLIP → GPT-2 (both hidden_dim=768, used in local CPU tests)."""
    proj = VisionProjection(in_dim=768, out_dim=768, num_layers=2)
    x   = torch.randn(4, 196, 768)
    assert proj(x).shape == (4, 196, 768)

def test_custom_hidden_dim():
    proj = VisionProjection(in_dim=768, out_dim=3584, hidden_dim=1024, num_layers=2)
    x   = torch.randn(1, 196, 768)
    assert proj(x).shape == (1, 196, 3584)


# ── Parameter & gradient tests ────────────────────────────────────────────────

def test_all_params_trainable():
    """Every projection weight must be trainable — nothing frozen here."""
    proj = VisionProjection(in_dim=768, out_dim=768, num_layers=2)
    for name, p in proj.named_parameters():
        assert p.requires_grad, f"{name} should be trainable"

def test_gradients_flow():
    """Backward pass must produce non-zero gradients on all linear layers."""
    proj = VisionProjection(in_dim=768, out_dim=256, num_layers=2)
    x    = torch.randn(2, 10, 768)
    loss = proj(x).mean()
    loss.backward()
    for name, p in proj.named_parameters():
        assert p.grad is not None,          f"{name}: gradient is None"
        assert p.grad.abs().sum() > 0,      f"{name}: gradient is zero"

def test_num_layers_one_has_single_linear():
    proj = VisionProjection(in_dim=64, out_dim=128, num_layers=1)
    linears = [m for m in proj.mlp.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 1

def test_num_layers_two_has_two_linears():
    proj = VisionProjection(in_dim=64, out_dim=128, num_layers=2)
    linears = [m for m in proj.mlp.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2


# ── Init tests ────────────────────────────────────────────────────────────────

def test_bias_initialised_to_zero():
    proj = VisionProjection(in_dim=64, out_dim=128, num_layers=2)
    for m in proj.mlp.modules():
        if isinstance(m, nn.Linear):
            assert torch.all(m.bias == 0), "bias must start at zero"

def test_invalid_num_layers_raises():
    with pytest.raises(AssertionError):
        VisionProjection(in_dim=64, out_dim=128, num_layers=0)
