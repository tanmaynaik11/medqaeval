"""
Evaluator — runs inference on medical benchmarks and computes metrics.

Supports two model types:
  1. MedVisionModel (ours) — loads base Qwen2-7B + Stage1 LoRA + Stage2 projection
  2. Qwen2-VL          — HuggingFace Qwen2-VL for multimodal baseline comparison

Both are evaluated on the same test sets so results are directly comparable.

Evaluation benchmarks:
  PathVQA test   (~6K)  — multimodal: image + yes/no or open question
  MedQA-USMLE test (~1.3K) — text-only: 4-option MCQ
  MedMCQA val    (~4.2K) — text-only: 4-option MCQ
"""

import logging
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from src.evaluation.metrics import (
    extract_option_letter,
    extract_yes_no,
    compute_accuracy,
    compute_pathvqa_accuracy,
)

logger = logging.getLogger(__name__)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _mcqa_prompt(question: str, options: dict) -> str:
    opts = "\n".join(f"  {k.upper()}. {v}" for k, v in options.items())
    return f"Question: {question}\n{opts}\nAnswer:"


def _vqa_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# ── MedVision evaluator ───────────────────────────────────────────────────────

class MedVisionEvaluator:
    """
    Loads our fine-tuned MedVisionModel and runs greedy inference.
    Handles both text-only (Stage 1) and multimodal (Stage 2) inputs.
    """

    def __init__(
        self,
        stage1_checkpoint: str,
        stage2_checkpoint: Optional[str] = None,
        model_id: str = "Qwen/Qwen2-7B-Instruct",
        cache_dir: str = "artifacts/model_cache",
        max_new_tokens: int = 32,
    ):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from src.models.multimodal import MedVisionModel, MedVisionConfig
        from peft import PeftModel

        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading MedVisionModel from {stage1_checkpoint}...")

        cfg = MedVisionConfig(
            llm_model_id          = model_id,
            llm_hidden_dim        = 3584,
            load_in_4bit          = True,
            lora_target_modules   = ["q_proj", "k_proj", "v_proj", "o_proj"],
            cache_dir             = cache_dir,
        )
        self.model = MedVisionModel(cfg)

        # Load Stage 1 LoRA
        lora_path = Path(stage1_checkpoint) / "lora_adapters"
        if lora_path.exists():
            self.model.llm = PeftModel.from_pretrained(
                self.model.llm.get_base_model(),
                str(lora_path),
                is_trainable=False,
            )
            logger.info("Stage 1 LoRA loaded.")

        # Load Stage 2 projection + LoRA if provided
        if stage2_checkpoint:
            proj_path = Path(stage2_checkpoint) / "projection.pt"
            if proj_path.exists():
                state = torch.load(proj_path, map_location="cpu")
                self.model.projection.load_state_dict(state)
                logger.info("Stage 2 projection weights loaded.")

            lora2_path = Path(stage2_checkpoint) / "lora_adapters"
            if lora2_path.exists():
                self.model.llm = PeftModel.from_pretrained(
                    self.model.llm.get_base_model(),
                    str(lora2_path),
                    is_trainable=False,
                )
                logger.info("Stage 2 LoRA loaded.")

        self.model.eval()
        self.tokenizer = self.model.tokenizer

    @torch.no_grad()
    def generate_text(self, prompt: str) -> str:
        """Run greedy decoding for a text-only prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        out = self.model.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_multimodal(self, prompt: str, image) -> str:
        """Run greedy decoding for an image + text prompt."""
        from PIL import Image as PILImage
        import torch

        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)

        # Preprocess image using BiomedCLIP preprocessor
        preprocessor = self.model.vision_encoder.preprocessor
        pixel_values = preprocessor(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        # Add <image> token to prompt
        full_prompt = f"<image>\n{prompt}"
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Use model.forward to get embeddings then generate
        patches = self.model.vision_encoder.get_patch_embeddings(pixel_values)
        image_embeds = self.model.projection(patches)

        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        inputs_embeds, attention_mask, _ = self.model._merge_image_text(
            image_embeds, input_ids, attention_mask, labels=None
        )

        out = self.model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ── Qwen2-VL evaluator ────────────────────────────────────────────────────────

class Qwen2VLEvaluator:
    """
    Loads Qwen2-VL-7B-Instruct for multimodal baseline comparison.
    Uses HuggingFace transformers with 4-bit quantization.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        cache_dir: str = "artifacts/model_cache",
        max_new_tokens: int = 32,
    ):
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import Qwen2VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Qwen2-VL from {model_id}...")

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            cache_dir=cache_dir,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.eval()
        logger.info("Qwen2-VL loaded.")

    @torch.no_grad()
    def generate_text(self, prompt: str) -> str:
        """Text-only inference."""
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_multimodal(self, prompt: str, image) -> str:
        """Image + text inference."""
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


# ── Benchmark runners ─────────────────────────────────────────────────────────

def evaluate_pathvqa(evaluator, dataset, max_samples: Optional[int] = None) -> dict:
    """
    Evaluate on PathVQA test split.
    Returns accuracy broken down by yes/no and open questions.
    """
    logger.info("Evaluating PathVQA...")

    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    predictions, labels, question_types = [], [], []

    for sample in tqdm(samples, desc="PathVQA"):
        question = sample["question"]
        label    = str(sample["answer"]).lower().strip()
        image    = sample["image"]

        prompt = _vqa_prompt(question)

        if hasattr(evaluator, 'generate_multimodal'):
            raw = evaluator.generate_multimodal(prompt, image)
        else:
            raw = evaluator.generate_text(prompt)

        # Detect question type
        is_yn = label in ("yes", "no")
        question_types.append("yes/no" if is_yn else "open")

        if is_yn:
            pred = extract_yes_no(raw)
        else:
            pred = raw.lower().strip()[:50]  # open: partial match

        predictions.append(pred)
        labels.append(label)

    return compute_pathvqa_accuracy(predictions, labels, question_types)


def evaluate_mcqa(evaluator, dataset, option_fields: dict, answer_field: str,
                  max_samples: Optional[int] = None, desc: str = "MCQ") -> dict:
    """
    Generic MCQ evaluator for MedMCQA and MedQA-USMLE.

    Args:
        option_fields: maps option keys to dataset column names
                       e.g. {"a": "opa", "b": "opb", "c": "opc", "d": "opd"} for MedMCQA
        answer_field:  column name containing the correct answer key
    """
    logger.info(f"Evaluating {desc}...")

    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    predictions, labels = [], []

    for sample in tqdm(samples, desc=desc):
        question = sample["question"]
        options  = {k: sample[v] for k, v in option_fields.items()}
        label    = str(sample[answer_field]).upper().strip()

        prompt = _mcqa_prompt(question, options)
        raw    = evaluator.generate_text(prompt)
        pred   = extract_option_letter(raw)

        predictions.append(pred)
        labels.append(label)

    return compute_accuracy(predictions, labels)
