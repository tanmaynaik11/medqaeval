"""
MedVisionModel — full multimodal model for medical VQA and clinical QA.

Architecture (LLaVA-style):
  BiomedCLIP (frozen)  →  VisionProjection (trained)  →  ┐
                                                           ├→  LLM + LoRA  →  output
  Tokenizer            →  Text Embeddings              →  ┘

Image-text merging:
  The prompt contains a special <image> token as a placeholder.
  During the forward pass we:
    1. Encode the image into 196 patch embeddings via BiomedCLIP
    2. Project patches into the LLM's embedding space
    3. Find the <image> token position in input_ids
    4. Replace the single <image> token with 196 patch embeddings
    5. Forward the merged sequence through the LLM

  Before merge:  [Q, u, e, <image>, t, e, x, t]   length = S
  After merge:   [Q, u, e, p0..p195, t, e, x, t]  length = S - 1 + 196

Text-only path:
  When pixel_values is None (Stage 1 text SFT), vision encoding is skipped
  entirely and input_ids are forwarded directly.

Local dev vs Production:
  Local dev  →  GPT-2 (768-dim, c_attn LoRA target, CPU-friendly)
  Production →  Qwen/Qwen2-7B-Instruct (3584-dim, q/k/v/o_proj LoRA targets)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.models.vision_encoder import build_vision_encoder
from src.models.projection import VisionProjection

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"


@dataclass
class MedVisionConfig:
    # Vision encoder
    encoder_type:     str  = "biomedclip"             # "clip" | "biomedclip"
    vision_hidden_dim: int = 768                       # BiomedCLIP ViT-B/16

    # Language model
    llm_model_id:   str  = "openai-community/gpt2"    # swap → Qwen/Qwen2-7B-Instruct on RunPod
    llm_hidden_dim: int  = 768                        # gpt2=768 | Qwen2-7B=3584

    # Projection MLP
    proj_hidden_dim:  int   = 1024
    proj_num_layers:  int   = 2
    proj_dropout:     float = 0.0

    # LoRA
    lora_r:               int          = 16
    lora_alpha:           int          = 32
    lora_dropout:         float        = 0.05
    lora_target_modules:  list[str]    = field(
        default_factory=lambda: ["c_attn"]   # GPT-2; use ["q_proj","k_proj","v_proj","o_proj"] for Qwen
    )

    # QLoRA — set load_in_4bit=True on RunPod, leave False for local CPU dev
    load_in_4bit:           bool = False
    bnb_4bit_quant_type:    str  = "nf4"           # nf4 = NormalFloat4, best for weights
    bnb_4bit_compute_dtype: str  = "bfloat16"      # compute in bf16 for stability

    cache_dir: str = "artifacts/model_cache"


class MedVisionModel(nn.Module):

    def __init__(self, config: MedVisionConfig):
        super().__init__()
        self.config = config

        # ── 1. Tokenizer ──────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {config.llm_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_id, cache_dir=config.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        logger.info(f"<image> token id: {self.image_token_id}")

        # ── 2. Vision encoder (frozen) ────────────────────────────────────────
        self.vision_encoder = build_vision_encoder(
            encoder_type=config.encoder_type,
            cache_dir=config.cache_dir,
            frozen=True,
            **({"model_id": "openai/clip-vit-base-patch32"} if config.encoder_type == "clip" else {}),
        )

        # ── 3. Projection MLP (trained from scratch) ──────────────────────────
        self.projection = VisionProjection(
            in_dim=config.vision_hidden_dim,
            out_dim=config.llm_hidden_dim,
            hidden_dim=config.proj_hidden_dim,
            num_layers=config.proj_num_layers,
            dropout=config.proj_dropout,
        )

        # ── 4. LLM + LoRA ─────────────────────────────────────────────────────
        logger.info(f"Loading LLM: {config.llm_model_id}")

        if config.load_in_4bit:
            # QLoRA path — RunPod only
            # NF4 quantization: model weights stored in 4-bit NormalFloat format.
            # Forward pass dequantizes to bfloat16 on the fly — no accuracy stored
            # in 4-bit, just the weights. This cuts VRAM from ~14GB to ~4GB for 7B.
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,   # nested quantization: saves ~0.4GB extra
            )
            base_llm = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                quantization_config=bnb_cfg,
                device_map="auto",                # bitsandbytes requires GPU from load time
                cache_dir=config.cache_dir,
            )
            # prepare_model_for_kbit_training: enables gradient checkpointing,
            # casts LayerNorm to fp32 (stable), freezes non-LoRA params
            base_llm = prepare_model_for_kbit_training(
                base_llm, use_gradient_checkpointing=True
            )
            logger.info("LLM loaded in 4-bit NF4 (QLoRA mode).")
        else:
            # Full precision path — local CPU dev
            base_llm = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id, cache_dir=config.cache_dir
            )

        # Resize embeddings to include the new <image> token.
        base_llm.resize_token_embeddings(len(self.tokenizer))

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(base_llm, lora_cfg)
        self.llm.print_trainable_parameters()

        # Move non-LLM components to GPU.
        # The LLM is already on GPU via device_map="auto" (bitsandbytes).
        # projection and vision_encoder stay on CPU by default — move them.
        if torch.cuda.is_available():
            self.projection = self.projection.cuda()

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings from the (LoRA-wrapped) LLM."""
        return self.llm.get_base_model().get_input_embeddings()(input_ids)

    def _merge_image_text(
        self,
        image_embeds: torch.Tensor,    # [B, N_patches, D_llm]
        input_ids:    torch.Tensor,    # [B, S]
        attention_mask: torch.Tensor,  # [B, S]
        labels: Optional[torch.Tensor],# [B, S] or None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Replace the <image> token in each sequence with the N_patches
        image embeddings.  Returns updated (inputs_embeds, attention_mask, labels).

        If a sample has no <image> token (text-only within a multimodal batch),
        image embeddings are prepended — this keeps batch processing uniform.
        """
        B, N, D = image_embeds.shape
        text_embeds = self._get_text_embeddings(input_ids)   # [B, S, D]
        S = text_embeds.shape[1]

        merged_embeds, merged_mask, merged_labels = [], [], []

        for b in range(B):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]

            if len(img_pos) == 0:
                # No <image> token: prepend image patches before text
                pos = 0
                pre_e  = text_embeds[b, :0]          # empty
                post_e = text_embeds[b]
                pre_m  = attention_mask[b, :0]
                post_m = attention_mask[b]
                pre_l  = labels[b, :0]  if labels is not None else None
                post_l = labels[b]      if labels is not None else None
            else:
                pos    = img_pos[0].item()
                pre_e  = text_embeds[b, :pos]
                post_e = text_embeds[b, pos + 1:]    # skip the <image> token
                pre_m  = attention_mask[b, :pos]
                post_m = attention_mask[b, pos + 1:]
                pre_l  = labels[b, :pos]      if labels is not None else None
                post_l = labels[b, pos + 1:]  if labels is not None else None

            # Patch embeddings + mask (all attend) + labels (all ignored)
            img_e = image_embeds[b]                                     # [N, D]
            img_m = torch.ones(N, dtype=attention_mask.dtype, device=attention_mask.device)
            img_l = torch.full((N,), -100, dtype=torch.long, device=input_ids.device) \
                    if labels is not None else None

            merged_embeds.append(torch.cat([pre_e, img_e, post_e], dim=0))
            merged_mask.append(torch.cat([pre_m, img_m, post_m], dim=0))
            if labels is not None:
                merged_labels.append(torch.cat([pre_l, img_l, post_l], dim=0))

        # Pad to same length within batch
        max_len = max(e.shape[0] for e in merged_embeds)

        def _pad_embeds(seq):
            pad = torch.zeros(max_len - seq.shape[0], D, device=seq.device, dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)

        def _pad_1d(seq, pad_val):
            pad = torch.full((max_len - seq.shape[0],), pad_val,
                             device=seq.device, dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)

        embeds_out = torch.stack([_pad_embeds(e) for e in merged_embeds])
        mask_out   = torch.stack([_pad_1d(m, 0) for m in merged_mask])
        labels_out = torch.stack([_pad_1d(l, -100) for l in merged_labels]) \
                     if labels is not None else None

        return embeds_out, mask_out, labels_out

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids:      [B, S]           tokenized prompt
            attention_mask: [B, S]
            pixel_values:   [B, 3, 224, 224] or None for text-only samples
            labels:         [B, S]           -100 on non-answer tokens

        Returns:
            HuggingFace CausalLMOutputWithPast (has .loss and .logits)
        """
        if pixel_values is not None:
            # Multimodal path
            patches      = self.vision_encoder.get_patch_embeddings(pixel_values)
            image_embeds = self.projection(patches)
            inputs_embeds, attention_mask, labels = self._merge_image_text(
                image_embeds, input_ids, attention_mask, labels
            )
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            # Text-only path (Stage 1 SFT — no images)
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

    def trainable_parameters(self) -> list:
        """Returns only the parameters that will receive gradient updates."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
