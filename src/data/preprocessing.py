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


def build_mcqa_prompt(
    question: str,
    options: dict,
    answer_key: Optional[str] = None,
) -> str:
    """
    Formats a multiple-choice medical QA sample.
    options: {"a": "...", "b": "...", "c": "...", "d": "..."}
    """
    opts_str = "\n".join(f"  {k.upper()}. {v}" for k, v in options.items())
    prompt = f"Question: {question}\n{opts_str}\nAnswer:"
    if answer_key is not None:
        prompt += f" {answer_key.upper()}"
    return prompt


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


def preprocess_medmcqa_sample(sample: dict) -> dict:
    """Map a raw MedMCQA sample to model-ready format."""
    options = {
        "a": sample.get("opa", ""),
        "b": sample.get("opb", ""),
        "c": sample.get("opc", ""),
        "d": sample.get("opd", ""),
    }
    answer_map = {0: "a", 1: "b", 2: "c", 3: "d"}
    answer_key = answer_map.get(sample.get("cop", -1))
    return {
        "prompt": build_mcqa_prompt(sample["question"], options, answer_key),
        "label": answer_key,
        "explanation": sample.get("exp", ""),
        "source": "medmcqa",
    }
