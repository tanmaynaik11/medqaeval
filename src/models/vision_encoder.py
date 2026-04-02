"""
Vision encoders for MedVision-Eval.

Two encoders, same interface — swap via config:

  CLIPVisionEncoder   — general CLIP (openai/clip-vit-base-patch32)
                        Good for local dev / testing on CPU.

  BiomedCLIPEncoder   — Microsoft BiomedCLIP (ViT-B/16)
                        Pretrained on 15M PubMed Central image-text pairs.
                        Use this for production medical fine-tuning.

Both expose:
  .hidden_dim                → int, patch embedding dimension
  .get_patch_embeddings(x)   → [B, N, D]  (no CLS token)
  .preprocessor              → callable that converts PIL → tensor
"""

import logging
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

logger = logging.getLogger(__name__)


# ── Standard CLIP ─────────────────────────────────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    """
    HuggingFace CLIP wrapper.  Useful for local CPU development because
    clip-vit-base-patch32 is only ~340 MB and runs on CPU in seconds.

    Output: patch embeddings [B, num_patches, hidden_dim]
      base-patch32 → 49 patches, hidden=768
      large-patch14 → 256 patches, hidden=1024
    """

    def __init__(self, model_id: str, cache_dir: str, frozen: bool = True):
        super().__init__()
        logger.info(f"Loading CLIP vision encoder: {model_id}")

        self.processor = CLIPImageProcessor.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.encoder = CLIPVisionModel.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.hidden_dim = self.encoder.config.hidden_size
        self.preprocessor = self.processor   # unified API

        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("CLIP encoder frozen.")

    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 224, 224]
        Returns:
            [B, num_patches, hidden_dim]  — CLS token stripped
        """
        out = self.encoder(pixel_values=pixel_values)
        return out.last_hidden_state[:, 1:, :]   # [B, N, D]


# ── BiomedCLIP ────────────────────────────────────────────────────────────────

class BiomedCLIPEncoder(nn.Module):
    """
    Microsoft BiomedCLIP vision encoder.

    WHY IT'S BETTER FOR MEDICAL TASKS
    ──────────────────────────────────
    Standard CLIP was trained on general web images (LAION-400M).
    BiomedCLIP was trained on 15 million PubMed Central (PMC) figure-caption
    pairs using the same contrastive objective — but every image is a medical
    figure (pathology slides, radiology scans, microscopy, etc.) and every
    caption is clinical text written by researchers.

    This means BiomedCLIP's patch embeddings encode medically meaningful
    concepts: cell morphology, tissue structure, imaging artifacts, anatomical
    landmarks.  When we feed these into the LLM, it receives much richer
    medical visual signal than standard CLIP would provide.

    HOW PATCH EXTRACTION WORKS
    ──────────────────────────
    open_clip loads BiomedCLIP via the timm backend (TimmModel), so the
    internal structure is:

      clip.visual              → TimmModel wrapper
      clip.visual.trunk        → timm ViT-B/16
      clip.visual.trunk.forward_features(x) → [B, 197, 768]
                                               (CLS + 196 patches, post-norm)

    We call trunk.forward_features() directly — no hook needed.
    This returns all 197 tokens; we strip index 0 (CLS) to get 196 patch
    embeddings that the LLM projection can attend to spatially.

    Architecture:
      BiomedCLIP ViT-B/16
      Input:      [B, 3, 224, 224]
      Patches:    196  ( (224/16)^2 )
      hidden_dim: 768
    """

    HF_MODEL_ID  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    HIDDEN_DIM   = 768
    NUM_PATCHES  = 196   # (224 / 16) ^ 2

    def __init__(self, cache_dir: str, frozen: bool = True):
        super().__init__()

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for BiomedCLIPEncoder. "
                "Install it with: pip install open-clip-torch"
            )

        logger.info("Loading BiomedCLIP from HuggingFace hub (~330 MB)...")
        self.clip, _, self.preprocessor = open_clip.create_model_and_transforms(
            self.HF_MODEL_ID,
            cache_dir=cache_dir,
        )
        self.hidden_dim = self.HIDDEN_DIM

        if frozen:
            for p in self.clip.parameters():
                p.requires_grad = False
            logger.info("BiomedCLIP encoder frozen — excluded from gradient updates.")

    def _extract_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract all patch tokens via TimmModel's trunk.forward_features().

        WHAT CHANGED FROM THE ORIGINAL DESIGN
        ──────────────────────────────────────
        We originally planned to use a forward hook on `.visual.transformer`,
        but open_clip loads BiomedCLIP through the timm backend (TimmModel),
        so `.visual` has no `.transformer` attribute.

        The actual call stack is:
          clip.encode_image(x)
            → clip.visual(x)           # TimmModel
              → clip.visual.trunk(x)   # timm ViT-B/16
                → patch_embed → pos_embed → blocks → norm → head

        TimmModel.trunk is the raw timm ViT.  Its forward_features() method
        returns the full sequence [CLS + patches] AFTER the final norm layer
        but BEFORE the classification head or contrastive projection — exactly
        the rich spatial features we need.

        No hook required: we call trunk.forward_features() directly.
        This is simpler, more robust, and easier to debug.

        Returns:
            [B, 196, 768] — 196 patch tokens, hidden_dim=768 (ViT-B/16)
        """
        with torch.no_grad():
            # forward_features returns [B, num_tokens, hidden_dim]
            # num_tokens = 1 (CLS) + 196 (patches) = 197
            all_tokens = self.clip.visual.trunk.forward_features(pixel_values)

        return all_tokens[:, 1:, :]   # strip CLS → [B, 196, 768]

    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Public interface matching CLIPVisionEncoder.

        Args:
            pixel_values: [B, 3, 224, 224]  (BiomedCLIP normalisation expected)
        Returns:
            [B, 196, 768]
        """
        return self._extract_patch_embeddings(pixel_values)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_vision_encoder(encoder_type: str, cache_dir: str, **kwargs):
    """
    Instantiate the right encoder from a config string.

    encoder_type:
      "clip"       → CLIPVisionEncoder  (fast, CPU-friendly dev encoder)
      "biomedclip" → BiomedCLIPEncoder  (production medical encoder)
    """
    if encoder_type == "clip":
        return CLIPVisionEncoder(cache_dir=cache_dir, **kwargs)
    elif encoder_type == "biomedclip":
        return BiomedCLIPEncoder(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. Choose 'clip' or 'biomedclip'."
        )
