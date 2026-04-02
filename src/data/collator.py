"""
MedicalCollator — bridges raw dataset samples to model-ready tensors.

Responsibilities:
  1. Detect sample type (multimodal vs text-only) from keys
  2. Apply vision preprocessor to PIL images
  3. Build prompt + full-text strings in the right chat format
  4. Tokenize with padding and truncation
  5. Build labels — -100 (ignored) everywhere except the answer tokens
  6. Return a dict of tensors ready for model.forward()

Why label masking matters for SFT:
  We only want the model to learn to produce the answer, not to reproduce
  the question.  Setting labels=-100 on all prompt tokens means the loss
  is computed only on the answer portion.  Without this, the model wastes
  capacity memorising question formatting instead of learning medical reasoning.

Supported sample formats (detected automatically):
  Multimodal (PathVQA):
    {"image": PIL.Image, "question": str, "answer": str}

  Text-only (MedMCQA after preprocess_medmcqa_sample):
    {"prompt": str, "label": str, "source": "medmcqa"}

  Text-only (MedQA-USMLE after preprocess_medqausmle_sample):
    {"prompt": str, "label": str, "source": "medqa-usmle"}

Mixed batches (some images, some not) are NOT supported.
Each DataLoader batch must be homogeneous — enforce this via separate
DataLoaders for Stage 1 (text) and Stage 2 (multimodal).
"""

import logging
from typing import Callable, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_pathvqa_texts(sample: dict) -> tuple[str, str]:
    """
    Returns (prompt_only, full_text).

    prompt_only: what the model sees as input (no answer) — used to
                 compute prompt length for label masking.
    full_text:   prompt + answer — what gets tokenized for training.
    """
    q = sample["question"]
    a = str(sample["answer"])
    prompt    = f"<image>\nQuestion: {q}\nAnswer:"
    full_text = f"<image>\nQuestion: {q}\nAnswer: {a}"
    return prompt, full_text

def _build_text_only_texts(sample: dict) -> tuple[str, str]:
    """
    prompt already formatted by preprocess_medmcqa_sample or
    preprocess_medqausmle_sample (ends with 'Answer:').
    """
    prompt    = sample["prompt"]
    label     = str(sample.get("label") or "")
    full_text = f"{prompt} {label}"
    return prompt, full_text

# ── Collator ──────────────────────────────────────────────────────────────────

class MedicalCollator:

    def __init__(
        self,
        tokenizer,
        image_preprocessor: Optional[Callable] = None,
        max_length: int = 512,
    ):
        """
        Args:
            tokenizer:           HuggingFace tokenizer (from MedVisionModel).
            image_preprocessor:  Callable: PIL.Image → FloatTensor [3, H, W].
                                 For BiomedCLIP this is the torchvision transform
                                 returned by open_clip.create_model_and_transforms().
                                 Pass None for text-only collation.
            max_length:          Truncation length for tokenizer.
        """
        self.tokenizer          = tokenizer
        self.image_preprocessor = image_preprocessor
        self.max_length         = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ── Label masking ─────────────────────────────────────────────────────────

    def _build_labels(
        self,
        prompt: str,
        full_text: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize full_text and mask all prompt tokens with -100.

        Returns:
            input_ids:      [S]   full sequence token ids
            attention_mask: [S]   1 for real tokens, 0 for padding
            labels:         [S]   -100 for prompt tokens, real ids for answer

        How prompt length is found:
          We tokenize the prompt alone (with the same special-token settings)
          and use its length N as the mask boundary.  Tokens 0..N-1 → -100.
          Tokens N..end → copied from input_ids (loss computed on these).
        """
        # Tokenize prompt alone to measure its token length
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # Tokenize full sequence
        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)       # [S]
        attention_mask = full_enc["attention_mask"].squeeze(0)  # [S]

        # Mask prompt tokens in labels
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # If the entire sequence is shorter than prompt (edge case with truncation)
        # ensure at least one token has a real label
        if (labels != -100).sum() == 0:
            labels[-1] = input_ids[-1]

        return input_ids, attention_mask, labels

    # ── Image preprocessing ───────────────────────────────────────────────────

    def _preprocess_image(self, image) -> torch.Tensor:
        """Convert PIL image to tensor [3, 224, 224] using the vision preprocessor."""
        if self.image_preprocessor is None:
            raise RuntimeError(
                "image_preprocessor is None but a multimodal sample was received. "
                "Pass image_preprocessor when constructing MedicalCollator."
            )
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.image_preprocessor(image)

        # BiomedCLIP preprocessor returns [3, H, W] directly
        # HF CLIPImageProcessor returns {"pixel_values": [1, 3, H, W]} — unwrap
        if isinstance(tensor, dict):
            tensor = tensor["pixel_values"].squeeze(0)

        return tensor   # [3, 224, 224]

    # ── Batch assembly ────────────────────────────────────────────────────────

    def _pad_batch(
        self,
        sequences: list[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:
        """Right-pad 1-D tensors to the length of the longest one."""
        max_len = max(s.shape[0] for s in sequences)
        padded  = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(sequences):
            padded[i, :s.shape[0]] = s
        return padded

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate a list of raw dataset samples into a model-ready batch.

        Returns dict with keys: input_ids, attention_mask, labels,
        and optionally pixel_values (for multimodal batches).
        """
        is_multimodal = "image" in samples[0]

        all_input_ids, all_masks, all_labels = [], [], []
        all_pixel_values = [] if is_multimodal else None

        for sample in samples:
            if is_multimodal:
                prompt, full_text = _build_pathvqa_texts(sample)
                pixel = self._preprocess_image(sample["image"])
                all_pixel_values.append(pixel)
            else:
                prompt, full_text = _build_text_only_texts(sample)

            input_ids, mask, labels = self._build_labels(prompt, full_text)
            all_input_ids.append(input_ids)
            all_masks.append(mask)
            all_labels.append(labels)

        pad_id = self.tokenizer.pad_token_id or 0

        batch = {
            "input_ids":      self._pad_batch(all_input_ids, pad_id),
            "attention_mask": self._pad_batch(all_masks,     0),
            "labels":         self._pad_batch(all_labels,   -100),
        }

        if is_multimodal:
            batch["pixel_values"] = torch.stack(all_pixel_values)   # [B, 3, 224, 224]

        return batch
