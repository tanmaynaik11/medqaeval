"""
Data ingestion: loads raw datasets from HuggingFace or local disk.
Supports multimodal (image+text) and text-only medical datasets.

Evaluation data isolation strategy (no leakage):
  PathVQA    — HF train/val/test splits used as-is.
               Stage 2 trains on 'train', monitors 'validation'.
               'test' (~6K samples) held out for final evaluation.

  MedMCQA    — HF has a 'test' split but answers are not publicly released.
               Stage 1 trains on 'train' (stratified 30K), monitors 'validation' (~4K).
               'validation' is used for final evaluation (standard for this dataset).

  MedQA-USMLE — HF only has 'train' (~10K) and 'test' (~1.3K).
               We carve 10% from 'train' as our val_loader during Stage 1.
               The HF 'test' split is never touched during training and is
               used exclusively for final benchmark evaluation.
"""

import os
import math
import random
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

from datasets import load_dataset, DatasetDict, Dataset

logger = logging.getLogger(__name__)


def _stratified_sample(
    dataset: Dataset,
    field: str,
    n_samples: int,
    seed: int = 42,
) -> Dataset:
    """Sample exactly n_samples from a HuggingFace Dataset, stratified by `field`.

    Uses the largest remainder method to guarantee the output is exactly
    n_samples rows while preserving class proportions as closely as possible.
    Plain floor-rounding loses rows when remainders accumulate across classes.

    Args:
        dataset:   HuggingFace Dataset to sample from.
        field:     Column name to stratify on (e.g. "subject_name").
        n_samples: Total samples to return (exact).
        seed:      Random seed for reproducibility.

    Returns:
        A new Dataset with exactly n_samples rows.
    """
    # Group row indices by class
    class_indices: dict[str, list[int]] = defaultdict(list)
    for i, value in enumerate(dataset[field]):
        class_indices[str(value)].append(i)

    total = len(dataset)

    # Compute exact floating-point quota per class
    exact_quotas = {
        cls: (len(idxs) / total) * n_samples
        for cls, idxs in class_indices.items()
    }

    # Floor quotas — this is where rows go missing
    floor_quotas: dict[str, int] = {cls: math.floor(q) for cls, q in exact_quotas.items()}

    # Largest remainder method: give +1 to the classes with the biggest fractional parts
    # until we've distributed all n_samples
    n_remaining = n_samples - sum(floor_quotas.values())
    ranked = sorted(exact_quotas, key=lambda c: exact_quotas[c] - floor_quotas[c], reverse=True)
    for cls in ranked[:n_remaining]:
        floor_quotas[cls] += 1

    # Shuffle each class's indices and take the allocated quota
    selected: list[int] = []
    for cls in sorted(class_indices):
        indices = class_indices[cls]
        n_cls = min(floor_quotas[cls], len(indices))   # safety: never exceed class size
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected.extend(indices[:n_cls])

    # Final global shuffle for training randomness
    random.Random(seed).shuffle(selected)

    logger.info(
        f"Stratified sample: {len(selected)} rows from {total} "
        f"across {len(class_indices)} classes ({field})"
    )
    return dataset.select(selected)


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
    stratified_n: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    """Load MedMCQA: medical multiple-choice questions (text only).

    Two sampling modes — use one, not both:
      max_samples:  fast dev cap via HF split-slicing (no full download).
      stratified_n: download full train set, then sample proportionally
                    across all 21 medical subjects. Use this on RunPod.

    Args:
        cache_dir:    Local cache directory.
        max_samples:  Quick cap for local development.
        stratified_n: Subject-stratified sample size for production (~30_000).
        seed:         Random seed for stratified sampling reproducibility.
    """
    if stratified_n:
        logger.info(
            f"Loading full MedMCQA train for stratified sampling "
            f"(target={stratified_n}) from HuggingFace..."
        )
        train_full = load_dataset("medmcqa", split="train", cache_dir=cache_dir)
        train = _stratified_sample(train_full, field="subject_name",
                                   n_samples=stratified_n, seed=seed)
    else:
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


def load_medqausmle(
    cache_dir: str,
    max_samples: Optional[int] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Load MedQA-USMLE-4-options: clinical USMLE-style MCQ (text only).

    Split strategy to avoid data leakage during evaluation:
      train (90%)    → gradient updates during Stage 1 training
      validation (10% of train) → val_loss monitoring / checkpoint selection
      test           → completely held out; never seen during training or validation

    HuggingFace only has a 'train' and 'test' split for this dataset.
    We carve our own validation from 'train' so that 'test' stays
    untouched for final benchmark evaluation.

    train: ~10,178 | test: ~1,273  (both sourced from HuggingFace)
    """
    logger.info(f"Loading MedQA-USMLE (max_samples={max_samples}) from HuggingFace...")

    if max_samples:
        # For local dev / smoke tests: take a small slice, split it in-memory
        full_slice = load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            split=f"train[:{max_samples}]",
            cache_dir=cache_dir,
        )
        split = full_slice.train_test_split(test_size=val_fraction, seed=seed)
        train = split["train"]
        val   = split["test"]
    else:
        # Production: load the full HF train split, then carve out validation
        full_train = load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            split="train",
            cache_dir=cache_dir,
        )
        split = full_train.train_test_split(test_size=val_fraction, seed=seed)
        train = split["train"]
        val   = split["test"]

    # Load the HF test split — held out completely for final evaluation
    test = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="test",
        cache_dir=cache_dir,
    )

    dataset = DatasetDict({"train": train, "validation": val, "test": test})
    logger.info(f"Splits: { {k: len(v) for k, v in dataset.items()} }")
    return dataset


# Registry — add new datasets here as you onboard them
DATASET_REGISTRY = {
    "path-vqa":    load_pathvqa,
    "medmcqa":     load_medmcqa,
    "medqa-usmle": load_medqausmle,
}


def load_dataset_by_name(name: str, cache_dir: str, **kwargs) -> DatasetDict:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    os.makedirs(cache_dir, exist_ok=True)
    return DATASET_REGISTRY[name](cache_dir=cache_dir, **kwargs)
