"""
Vision-to-Language Projection MLP.

Bridges the gap between BiomedCLIP's patch embeddings and the LLM's
embedding space. This is the ONLY component trained from scratch —
BiomedCLIP is frozen, the LLM is updated only via LoRA.

  BiomedCLIP patches                LLM token space
  [B, 196, 768]  ──── Projection ────→  [B, 196, 3584]
                          ↑
                    (this module)

Design: 2-layer MLP with GELU, following LLaVA-1.6 / LLaVA-Next.
  Layer 1: in_dim  → hidden_dim  + GELU + Dropout
  Layer 2: hidden_dim → out_dim

Why 2 layers over a single linear?
  A single linear projection can only rotate/scale the embedding space.
  The 2-layer MLP learns a non-linear mapping, which is important because
  BiomedCLIP's embedding geometry (trained with contrastive loss on medical
  images) differs structurally from the LLM's token embedding space
  (trained with next-token prediction on text).  The non-linearity lets the
  projection learn to reshape one manifold into the other.

Why GELU over ReLU?
  GELU is smoother than ReLU — no hard zero-threshold — which prevents
  dead neurons during early training when the projection weights are random.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VisionProjection(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_dim:     Input dimension — must match vision encoder hidden_dim.
                        BiomedCLIP ViT-B/16 → 768
                        CLIP ViT-B/32       → 768
            out_dim:    Output dimension — must match LLM hidden_dim.
                        Qwen2-VL-7B         → 3584
                        GPT-2 (local dev)   → 768
            hidden_dim: Intermediate MLP width.  Defaults to out_dim.
            num_layers: 1 = linear projection only.
                        2 = standard LLaVA-style MLP (recommended).
            dropout:    Applied after GELU in intermediate layers.
                        Kept at 0.0 by default — projection is small and
                        trains quickly, dropout rarely helps here.
        """
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"

        if hidden_dim is None:
            hidden_dim = out_dim

        layers: list[nn.Module] = []
        current = in_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current = hidden_dim

        layers.append(nn.Linear(current, out_dim))

        self.mlp    = nn.Sequential(*layers)
        self.in_dim  = in_dim
        self.out_dim = out_dim

        self._init_weights()
        logger.info(
            f"VisionProjection: {in_dim}→{'→'.join([str(hidden_dim)]*(num_layers-1))}→{out_dim}"
            f"  layers={num_layers}  params={self._count_params():,}"
        )

    def _init_weights(self) -> None:
        """Xavier uniform init — keeps early gradients in a healthy range."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim]  — patch embeddings from vision encoder
        Returns:
            [B, N, out_dim]    — projected into LLM embedding space
        """
        return self.mlp(x)
