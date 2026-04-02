"""
SFTTrainer — Supervised Fine-Tuning training loop.

Handles both Stage 1 (text-only) and Stage 2 (multimodal) via the same
loop. The only difference is whether batches contain pixel_values.

Key design decisions:
  - Accelerate for device management (single or multi-GPU transparent)
  - Two parameter groups: LoRA adapters + projection (different LRs)
  - Cosine LR schedule with linear warmup
  - Gradient accumulation to reach effective batch size of 32
  - Checkpoint every N steps + best checkpoint tracked by val loss
  - W&B logging for all metrics
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate.utils import set_seed

logger = logging.getLogger(__name__)


class SFTTrainer:

    def __init__(
        self,
        model,
        train_loader:       DataLoader,
        val_loader:         DataLoader,
        output_dir:         str,
        num_epochs:         int   = 3,
        lora_lr:            float = 2e-4,
        projection_lr:      float = 2e-4,
        weight_decay:       float = 0.01,
        gradient_accumulation: int = 8,
        max_grad_norm:      float = 1.0,
        warmup_ratio:       float = 0.05,
        save_steps:         int   = 500,
        eval_steps:         int   = 500,
        logging_steps:      int   = 25,
        use_wandb:          bool  = True,
        wandb_project:      str   = "medvision-eval",
        wandb_run_name:     str   = "sft-run",
        seed:               int   = 42,
    ):
        self.model              = model
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.output_dir         = Path(output_dir)
        self.num_epochs         = num_epochs
        self.grad_accum         = gradient_accumulation
        self.max_grad_norm      = max_grad_norm
        self.save_steps         = save_steps
        self.eval_steps         = eval_steps
        self.logging_steps      = logging_steps
        self.use_wandb          = use_wandb

        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

        # ── Optimizer — separate LR per parameter group ───────────────────────
        # LoRA adapters and projection have different roles:
        #   LoRA: fine-tuning existing knowledge → moderate LR
        #   Projection: learning from scratch → can use same or slightly higher LR
        # Both excluded from weight decay for bias/LayerNorm params (standard practice)
        lora_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "projection" not in n
        ]
        proj_params = list(model.projection.parameters())

        param_groups = []
        if proj_params and projection_lr > 0:
            param_groups.append({"params": proj_params, "lr": projection_lr})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lora_lr})

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)

        # ── LR Scheduler — cosine decay with warmup ───────────────────────────
        total_steps  = len(train_loader) * num_epochs // gradient_accumulation
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = self._build_scheduler(total_steps, warmup_steps)

        # ── Device ───────────────────────────────────────────────────────────
        # With device_map="auto", layers are spread across devices by bitsandbytes.
        # Input ids must go to the same device as the embedding table.
        # We find that device here and use it to move batches consistently.
        self.device = model.llm.get_base_model().get_input_embeddings().weight.device

        # Store dataloaders without Accelerate wrapping (incompatible with device_map="auto")
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.global_step  = 0
        self.best_val_loss = float("inf")

    def _build_scheduler(self, total_steps: int, warmup_steps: int):
        """Linear warmup then cosine decay — standard for SFT."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        logger.info(f"Starting training: {self.num_epochs} epochs, "
                    f"grad_accum={self.grad_accum}")
        logger.info(f"Trainable params: {self.model.trainable_param_count():,}")

        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1} / {self.num_epochs}")
            logger.info(f"{'='*50}")

            train_loss = self._train_epoch(epoch)
            val_loss   = self._evaluate()

            logger.info(f"Epoch {epoch+1} — train_loss: {train_loss:.4f}  "
                        f"val_loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f"New best val_loss: {val_loss:.4f} — saved best checkpoint.")

        self._save_checkpoint("final")
        logger.info("Training complete. Final checkpoint saved.")

        if self.use_wandb:
            import wandb
            wandb.finish()

    def _move_batch(self, batch: dict) -> dict:
        """Move batch tensors to the model's primary device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss    = 0.0
        num_batches   = 0
        accum_loss    = 0.0

        self.optimizer.zero_grad()

        logger.info(f"  batches={len(self.train_loader)}")
        for step, batch in enumerate(self.train_loader):
            logger.info(f"  step {step} — forward...")
            # device_map="auto" handles inter-layer device movement internally.
            # No manual device movement or autocast — bitsandbytes 4-bit handles
            # bf16 compute internally and torch.amp.autocast triggers CUDA lazy
            # init which hangs on some RunPod driver configurations.
            outputs = self.model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                pixel_values   = batch.get("pixel_values"),
                labels         = batch["labels"],
            )
            # Scale loss by grad_accum so gradients average correctly
            loss = outputs.loss / self.grad_accum

            loss.backward()

            accum_loss  += outputs.loss.detach().item()
            total_loss  += outputs.loss.detach().item()
            num_batches += 1
            self.global_step += 1

            # Optimizer step every grad_accum batches
            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Logging
            if self.global_step % self.logging_steps == 0:
                avg_loss = accum_loss / self.logging_steps
                lr       = self.scheduler.get_last_lr()[0]
                logger.info(f"  step {self.global_step:5d} | "
                            f"loss {avg_loss:.4f} | lr {lr:.2e}")
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr":   lr,
                        "train/step": self.global_step,
                        "train/epoch": epoch,
                    }, step=self.global_step)
                accum_loss = 0.0

            # Mid-epoch checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(f"step-{self.global_step}")

            # Mid-epoch evaluation
            if self.global_step % self.eval_steps == 0:
                val_loss = self._evaluate()
                logger.info(f"  [eval] step {self.global_step} | val_loss {val_loss:.4f}")
                if self.use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss}, step=self.global_step)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
                self.model.train()

        return total_loss / max(num_batches, 1)

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss  = 0.0
        num_batches = 0

        for batch in self.val_loader:
            outputs = self.model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                pixel_values   = batch.get("pixel_values"),
                labels         = batch["labels"],
            )
            total_loss  += outputs.loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str) -> None:
        """
        Save LoRA adapters + projection weights.
        We do NOT save the full model — only the trained parameters.
        To resume: load base model, apply these weights on top.
        """
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.model

        # Save LoRA adapters (PEFT format — can be loaded with PeftModel.from_pretrained)
        unwrapped.llm.save_pretrained(str(ckpt_dir / "lora_adapters"))

        # Save projection weights separately
        torch.save(
            unwrapped.projection.state_dict(),
            ckpt_dir / "projection.pt",
        )

        logger.info(f"Checkpoint saved → {ckpt_dir}")
