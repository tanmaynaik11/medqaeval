"""
End-to-end smoke test — Phase 2 validation before RunPod.

Runs the full pipeline on CPU with tiny data:
  Raw dataset → Collator → MedVisionModel → Loss → Backward pass

Uses CLIP + GPT-2 (no large downloads, runs in ~60s on CPU).
On RunPod, swap to BiomedCLIP + Qwen2-7B-Instruct.

Usage:
    python scripts/smoke_test.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.utils.logging import setup_logging
from src.utils.reproducibility import seed_everything
from src.data.ingestion import load_dataset_by_name
from src.data.preprocessing import preprocess_medmcqa_sample
from src.data.dataset import MedVQADataset, MedTextDataset
from src.data.collator import MedicalCollator
from src.models.multimodal import MedVisionModel, MedVisionConfig

setup_logging(level="INFO")
logger = logging.getLogger(__name__)
seed_everything(42)

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    logger.info(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("SMOKE TEST — MedVision-Eval end-to-end pipeline")
logger.info("=" * 60)

# ── Step 1: Load tiny datasets ────────────────────────────────────────────────
logger.info("\n[1/5] Loading datasets (from local cache)...")

pathvqa = load_dataset_by_name("path-vqa", cache_dir="data/raw/pathvqa", max_samples=20)
medmcqa = load_dataset_by_name("medmcqa",  cache_dir="data/raw/medmcqa",  max_samples=20)

check("PathVQA loaded",  len(pathvqa["train"]) > 0, f"{len(pathvqa['train'])} samples")
check("MedMCQA loaded",  len(medmcqa["train"]) > 0, f"{len(medmcqa['train'])} samples")

# ── Step 2: Build model ───────────────────────────────────────────────────────
logger.info("\n[2/5] Building MedVisionModel (CLIP + GPT-2, CPU)...")

cfg = MedVisionConfig(
    encoder_type      = "clip",
    vision_hidden_dim = 768,
    llm_model_id      = "openai-community/gpt2",
    llm_hidden_dim    = 768,
    proj_hidden_dim   = 256,
    proj_num_layers   = 2,
    lora_target_modules = ["c_attn"],
    cache_dir         = "artifacts/model_cache",
)
model = MedVisionModel(cfg)
model.eval()

trainable = model.trainable_param_count()
total     = sum(p.numel() for p in model.parameters())
check("Model built", model is not None)
check("Trainable params < total", trainable < total,
      f"{trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
check("Vision encoder frozen",
      all(not p.requires_grad for p in model.vision_encoder.parameters()))
check("Projection trainable",
      all(p.requires_grad for p in model.projection.parameters()))

# ── Step 3: Build collators ───────────────────────────────────────────────────
logger.info("\n[3/5] Building collators...")

# CLIP preprocessor wraps HF's CLIPImageProcessor to return [3,224,224] tensor
raw_processor = model.vision_encoder.preprocessor
def clip_preprocess(pil_img):
    out = raw_processor(images=pil_img, return_tensors="pt")
    return out["pixel_values"].squeeze(0)   # [3, 224, 224]

mm_collator   = MedicalCollator(
    tokenizer=model.tokenizer,
    image_preprocessor=clip_preprocess,
    max_length=128,
)
text_collator = MedicalCollator(
    tokenizer=model.tokenizer,
    image_preprocessor=None,
    max_length=128,
)
check("Multimodal collator built", mm_collator is not None)
check("Text collator built",       text_collator is not None)

# ── Step 4: Forward pass — multimodal ────────────────────────────────────────
logger.info("\n[4/5] Forward pass with real data...")

# ---- 4a: Multimodal (PathVQA) -----------------------------------------------
mm_samples = [pathvqa["train"][i] for i in range(4)]
mm_batch   = mm_collator(mm_samples)

check("Multimodal batch has pixel_values", "pixel_values" in mm_batch)
check("pixel_values shape",
      mm_batch["pixel_values"].shape == (4, 3, 224, 224),
      str(mm_batch["pixel_values"].shape))
check("Labels have -100 (prompt masked)",
      (mm_batch["labels"] == -100).any().item())
check("Labels have real tokens (answer)",
      (mm_batch["labels"] != -100).any().item())

with torch.no_grad():
    mm_out = model(
        input_ids      = mm_batch["input_ids"],
        attention_mask = mm_batch["attention_mask"],
        pixel_values   = mm_batch["pixel_values"],
        labels         = mm_batch["labels"],
    )
check("Multimodal forward pass runs",  mm_out.logits is not None)
check("Multimodal loss is finite",
      mm_out.loss is not None and torch.isfinite(mm_out.loss),
      f"loss={mm_out.loss.item():.4f}")
check("Logits batch size == 4",        mm_out.logits.shape[0] == 4)

# ---- 4b: Text-only (MedMCQA) ------------------------------------------------
text_samples = [
    preprocess_medmcqa_sample(medmcqa["train"][i])
    for i in range(4)
]
text_batch = text_collator(text_samples)

check("Text batch has no pixel_values", "pixel_values" not in text_batch)

with torch.no_grad():
    text_out = model(
        input_ids      = text_batch["input_ids"],
        attention_mask = text_batch["attention_mask"],
        labels         = text_batch["labels"],
    )
check("Text-only forward pass runs", text_out.logits is not None)
check("Text-only loss is finite",
      text_out.loss is not None and torch.isfinite(text_out.loss),
      f"loss={text_out.loss.item():.4f}")

# ── Step 5: Backward pass — gradient check ────────────────────────────────────
logger.info("\n[5/5] Backward pass — verifying gradient flow...")

model.train()
mm_batch_grad = mm_collator([pathvqa["train"][0]])

out = model(
    input_ids      = mm_batch_grad["input_ids"],
    attention_mask = mm_batch_grad["attention_mask"],
    pixel_values   = mm_batch_grad["pixel_values"],
    labels         = mm_batch_grad["labels"],
)
out.loss.backward()

proj_grads = [p.grad for p in model.projection.parameters() if p.grad is not None]
lora_grads = [p.grad for n, p in model.llm.named_parameters()
              if "lora" in n and p.grad is not None]
clip_grads = [p.grad for p in model.vision_encoder.parameters()
              if p.grad is not None]

check("Projection received gradients",    len(proj_grads) > 0)
check("LoRA layers received gradients",   len(lora_grads) > 0)
check("Vision encoder has NO gradients",  len(clip_grads) == 0,
      "CLIP must stay frozen")

# ── Summary ───────────────────────────────────────────────────────────────────
logger.info("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
logger.info(f"RESULTS: {passed} passed  /  {failed} failed  /  {len(results)} total")

if failed:
    logger.info("\nFailed checks:")
    for name, ok, detail in results:
        if not ok:
            logger.info(f"  [{FAIL}] {name}" + (f" — {detail}" if detail else ""))
    sys.exit(1)
else:
    logger.info("All checks passed — pipeline ready for RunPod.")
    logger.info("=" * 60)
