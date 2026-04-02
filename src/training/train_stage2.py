"""
Stage 2: Multimodal Medical SFT
Loads Stage 1 LoRA checkpoint, trains projection + LoRA on PathVQA.

Usage on RunPod:
    python scripts/train_stage2.py
    python scripts/train_stage2.py --smoke   # 20 samples, 1 epoch
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from peft import PeftModel

from src.utils.logging import setup_logging
from src.utils.env import load_env
from src.utils.reproducibility import seed_everything
from src.data.ingestion import load_dataset_by_name
from src.data.collator import MedicalCollator
from src.models.multimodal import MedVisionModel, MedVisionConfig
from src.training.trainer import SFTTrainer

setup_logging(level="INFO", log_file="logs/stage2_sft.log")
logger = logging.getLogger(__name__)
load_env()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/stage2_sft.yaml")
    p.add_argument("--smoke",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    seed_everything(cfg.training.seed)

    if args.smoke:
        logger.info("SMOKE MODE — 20 samples, 1 epoch")
        cfg.training.num_epochs         = 1
        cfg.data.pathvqa_max            = 20
        cfg.training.save_steps         = 999999
        cfg.training.eval_steps         = 999999
        cfg.training.logging_steps      = 1

    # ── 1. Build model ────────────────────────────────────────────────────────
    logger.info("Building model...")
    model_cfg = MedVisionConfig(
        encoder_type          = cfg.model.encoder_type,
        vision_hidden_dim     = cfg.model.vision_hidden_dim,
        llm_model_id          = cfg.model.llm_model_id,
        llm_hidden_dim        = cfg.model.llm_hidden_dim,
        proj_hidden_dim       = cfg.model.proj_hidden_dim,
        proj_num_layers       = cfg.model.proj_num_layers,
        lora_r                = cfg.model.lora_r,
        lora_alpha            = cfg.model.lora_alpha,
        lora_dropout          = cfg.model.lora_dropout,
        lora_target_modules   = list(cfg.model.lora_target_modules),
        load_in_4bit          = cfg.model.load_in_4bit,
        bnb_4bit_quant_type   = cfg.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype= cfg.model.bnb_4bit_compute_dtype,
        cache_dir             = cfg.model.cache_dir,
    )
    model = MedVisionModel(model_cfg)

    # ── 2. Load Stage 1 LoRA checkpoint ──────────────────────────────────────
    stage1_ckpt = Path(cfg.model.stage1_checkpoint)
    lora_path   = stage1_ckpt / "lora_adapters"

    if lora_path.exists():
        logger.info(f"Loading Stage 1 LoRA weights from {lora_path}")
        model.llm = PeftModel.from_pretrained(
            model.llm.get_base_model(),
            str(lora_path),
            is_trainable=True,
        )
        logger.info("Stage 1 LoRA weights loaded successfully.")
    else:
        logger.warning(
            f"Stage 1 checkpoint not found at {lora_path}. "
            "Training from scratch — results will be weaker."
        )

    # ── 3. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading PathVQA...")
    pathvqa = load_dataset_by_name(
        "path-vqa",
        cache_dir=cfg.data.pathvqa_cache,
        max_samples=cfg.data.pathvqa_max,
    )

    # BiomedCLIP preprocessor for image collation
    biomed_preprocessor = model.vision_encoder.preprocessor

    collator = MedicalCollator(
        tokenizer=model.tokenizer,
        image_preprocessor=biomed_preprocessor,
        max_length=cfg.data.max_length,
    )

    train_loader = DataLoader(
        pathvqa["train"],
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        pathvqa["validation"],
        batch_size=cfg.training.per_device_batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(pathvqa['train']):,}  Val: {len(pathvqa['validation']):,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        output_dir         = cfg.training.output_dir,
        num_epochs         = cfg.training.num_epochs,
        lora_lr            = cfg.training.lora_lr,
        projection_lr      = cfg.training.projection_lr,   # non-zero in stage 2
        weight_decay       = cfg.training.weight_decay,
        gradient_accumulation = cfg.training.gradient_accumulation,
        max_grad_norm      = cfg.training.max_grad_norm,
        warmup_ratio       = cfg.training.warmup_ratio,
        save_steps         = cfg.training.save_steps,
        eval_steps         = cfg.training.eval_steps,
        logging_steps      = cfg.training.logging_steps,
        use_wandb          = not args.smoke,
        wandb_project      = cfg.wandb.project,
        wandb_run_name     = cfg.wandb.run_name,
        seed               = cfg.training.seed,
    )

    trainer.train()
    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
