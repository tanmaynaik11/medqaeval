"""
Stage 1: Medical Text SFT
Trains LoRA adapters on MedMCQA + MedQA-USMLE (no images).

Usage on RunPod:
    python scripts/train_stage1.py
    python scripts/train_stage1.py --smoke   # 50 samples, 1 epoch — validate pipeline
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from src.utils.logging import setup_logging
from src.utils.env import load_env
from src.utils.reproducibility import seed_everything
from src.data.ingestion import load_dataset_by_name
from src.data.preprocessing import preprocess_medmcqa_sample, preprocess_medqausmle_sample
from src.data.dataset import MedTextDataset
from src.data.collator import MedicalCollator
from src.models.multimodal import MedVisionModel, MedVisionConfig
from src.training.trainer import SFTTrainer

setup_logging(level="INFO", log_file="logs/stage1_sft.log")
logger = logging.getLogger(__name__)
load_env()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/stage1_sft.yaml")
    p.add_argument("--smoke",  action="store_true",
                   help="Smoke mode: 50 samples, 1 epoch — fast pipeline check")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    seed_everything(cfg.training.seed)

    if args.smoke:
        logger.info("SMOKE MODE — 50 samples, 1 epoch")
        cfg.training.num_epochs             = 1
        cfg.data.medmcqa_stratified_n       = 40
        cfg.data.medqausmle_max             = 10
        cfg.training.save_steps             = 999999
        cfg.training.eval_steps             = 999999
        cfg.training.logging_steps          = 1
        cfg.training.dataloader_workers     = 0  # avoid multiprocessing deadlocks in containers

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading datasets...")

    medmcqa_ds = load_dataset_by_name(
        "medmcqa",
        cache_dir=cfg.data.medmcqa_cache,
        stratified_n=cfg.data.medmcqa_stratified_n if not args.smoke else None,
        max_samples=40 if args.smoke else None,
    )
    usmle_ds = load_dataset_by_name(
        "medqa-usmle",
        cache_dir=cfg.data.medqausmle_cache,
        max_samples=cfg.data.medqausmle_max,
    )

    # Preprocess and combine
    mcqa_train  = MedTextDataset(medmcqa_ds["train"], preprocess_medmcqa_sample)
    usmle_train = MedTextDataset(usmle_ds["train"],   preprocess_medqausmle_sample)
    mcqa_val    = MedTextDataset(medmcqa_ds["validation"], preprocess_medmcqa_sample)
    usmle_val   = MedTextDataset(usmle_ds["validation"],   preprocess_medqausmle_sample)

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([mcqa_train, usmle_train])
    val_dataset   = ConcatDataset([mcqa_val,   usmle_val])

    logger.info(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    # ── 2. Build model ────────────────────────────────────────────────────────
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

    # ── 3. Collators and DataLoaders ──────────────────────────────────────────
    collator = MedicalCollator(
        tokenizer=model.tokenizer,
        image_preprocessor=None,   # text-only stage
        max_length=cfg.data.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.per_device_batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        output_dir         = cfg.training.output_dir,
        num_epochs         = cfg.training.num_epochs,
        lora_lr            = cfg.training.lora_lr,
        projection_lr      = 0.0,   # projection not trained in stage 1
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
    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
