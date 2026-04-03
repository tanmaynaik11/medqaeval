"""
Preprocessing pipeline for multimodal medical data.
Handles image transforms and text tokenization independently
so each can be swapped without touching the other.
"""

import logging
from typing import Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# ---------- Image Preprocessing ----------

def get_image_transforms(image_size: int = 224):
    """
    Returns a callable that resizes, normalizes, and converts
    a PIL image to a numpy array. No torchvision dependency here
    — keeps preprocessing decoupled from training framework.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def transform(image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)   # HWC → CHW

    return transform


# ---------- Text Preprocessing ----------

def build_vqa_prompt(question: str, answer: Optional[str] = None) -> str:
    """
    Formats a VQA sample into the instruction-following prompt template
    we'll use consistently across SFT and evaluation.
    """
    prompt = (
        f"<image>\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    if answer is not None:
        prompt += f" {answer}"
    return prompt


def build_mcqa_prompt(question: str, options: dict) -> str:
    """
    Formats a multiple-choice medical QA sample into a prompt (no answer).
    The answer is kept separate so the collator can mask only the prompt tokens.
    options: {"a": "...", "b": "...", "c": "...", "d": "..."}
    """
    opts_str = "\n".join(f"  {k.upper()}. {v}" for k, v in options.items())
    return f"Question: {question}\n{opts_str}\nAnswer:"


# ---------- Dataset-specific Mappers ----------

def preprocess_pathvqa_sample(sample: dict, image_size: int = 224) -> dict:
    """Map a raw Path-VQA sample to model-ready format."""
    transform = get_image_transforms(image_size)
    return {
        "pixel_values": transform(sample["image"]),
        "prompt": build_vqa_prompt(sample["question"]),
        "label": str(sample["answer"]),
        "source": "path-vqa",
    }


def preprocess_medqausmle_sample(sample: dict) -> dict:
    """Map a raw MedQA-USMLE sample to model-ready format.

    Raw schema:
        question   : str
        options    : {"A": "...", "B": "...", "C": "...", "D": "..."}
        answer_idx : "A" | "B" | "C" | "D"
        answer     : str  (full text of correct option)
    """
    options = sample.get("options", {})
    answer_key = sample.get("answer_idx", "A").upper()
    options_lower = {k.lower(): v for k, v in options.items()}
    # Label = "A. <full option text>" so model learns the answer content
    answer_text = options.get(answer_key, "")
    label = f"{answer_key}. {answer_text}"
    return {
        "prompt": build_mcqa_prompt(sample["question"], options_lower),
        "label": label,
        "source": "medqa-usmle",
    }


def preprocess_medmcqa_sample(sample: dict) -> dict:
    """Map a raw MedMCQA sample to model-ready format."""
    options = {
        "a": sample.get("opa", ""),
        "b": sample.get("opb", ""),
        "c": sample.get("opc", ""),
        "d": sample.get("opd", ""),
    }
    answer_map = {0: "a", 1: "b", 2: "c", 3: "d"}
    answer_key = answer_map.get(sample.get("cop", -1), "a")
    # Label = "A. <full option text>. <explanation>" so the model learns
    # the correct answer AND the medical reasoning behind it.
    # Explanation is included only when non-empty (~60% of MedMCQA samples).
    answer_text = options.get(answer_key, "")
    explanation = sample.get("exp", "").strip()
    label = f"{answer_key.upper()}. {answer_text}"
    if explanation:
        label += f". {explanation}"
    return {
        "prompt": build_mcqa_prompt(sample["question"], options),
        "label": label,
        "source": "medmcqa",
    }
