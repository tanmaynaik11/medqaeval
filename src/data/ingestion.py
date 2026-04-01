"""
Data ingestion: loads raw datasets from HuggingFace or local disk.
Supports multimodal (image+text) and text-only medical datasets.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


def _split_spec(max_samples: Optional[int], split: str, fraction: float = 1.0) -> str:
    """Build a HuggingFace split-slicing string e.g. 'train[:100]'.
    Slicing is resolved server-side so only the needed parquet rows are fetched.
    """
    if max_samples:
        n = int(max_samples * fraction)
        return f"{split}[:{n}]"
    return split


def load_pathvqa(
    cache_dir: str,
    max_samples: Optional[int] = None,
) -> DatasetDict:
    """Load Path-VQA: pathology images + open/closed Q&A pairs.

    Uses split-slicing so only the requested number of rows are downloaded,
    not the full 477 MB corpus.
    """
    logger.info(f"Loading Path-VQA (max_samples={max_samples}) from HuggingFace...")

    train = load_dataset(
        "flaviagiammarino/path-vqa",
        split=_split_spec(max_samples, "train"),
        cache_dir=cache_dir,
    )
    val = load_dataset(
        "flaviagiammarino/path-vqa",
        split=_split_spec(max_samples, "validation", fraction=0.2),
        cache_dir=cache_dir,
    )
    test = load_dataset(
        "flaviagiammarino/path-vqa",
        split=_split_spec(max_samples, "test", fraction=0.2),
        cache_dir=cache_dir,
    )

    dataset = DatasetDict({"train": train, "validation": val, "test": test})
    logger.info(f"Splits: { {k: len(v) for k, v in dataset.items()} }")
    return dataset


def load_medmcqa(
    cache_dir: str,
    max_samples: Optional[int] = None,
) -> DatasetDict:
    """Load MedMCQA: medical multiple-choice questions (text only)."""
    logger.info(f"Loading MedMCQA (max_samples={max_samples}) from HuggingFace...")

    train = load_dataset(
        "medmcqa",
        split=_split_spec(max_samples, "train"),
        cache_dir=cache_dir,
    )
    val = load_dataset(
        "medmcqa",
        split=_split_spec(max_samples, "validation", fraction=0.2),
        cache_dir=cache_dir,
    )

    dataset = DatasetDict({"train": train, "validation": val})
    logger.info(f"Splits: { {k: len(v) for k, v in dataset.items()} }")
    return dataset


# Registry — add new datasets here as you onboard them
DATASET_REGISTRY = {
    "path-vqa": load_pathvqa,
    "medmcqa": load_medmcqa,
}


def load_dataset_by_name(name: str, cache_dir: str, **kwargs) -> DatasetDict:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    os.makedirs(cache_dir, exist_ok=True)
    return DATASET_REGISTRY[name](cache_dir=cache_dir, **kwargs)
