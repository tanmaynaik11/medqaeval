"""
PyTorch Dataset wrappers for multimodal medical data.
Keeps raw HuggingFace datasets separate from torch-specific logic.
"""

import logging
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MedVQADataset(Dataset):
    """
    Wraps a HuggingFace dataset split for multimodal VQA.
    Applies a preprocessing function lazily (at __getitem__ time).
    """

    def __init__(
        self,
        hf_dataset,
        preprocess_fn: Callable,
        max_samples: Optional[int] = None,
    ):
        self.data = hf_dataset
        self.preprocess_fn = preprocess_fn
        if max_samples:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
        logger.info(f"MedVQADataset ready: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        raw = self.data[idx]
        return self.preprocess_fn(raw)


class MedTextDataset(Dataset):
    """Wraps text-only medical QA datasets."""

    def __init__(self, hf_dataset, preprocess_fn: Callable):
        self.data = hf_dataset
        self.preprocess_fn = preprocess_fn
        logger.info(f"MedTextDataset ready: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.preprocess_fn(self.data[idx])
