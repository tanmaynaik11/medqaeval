"""
Unit tests for the data ingestion layer.
Uses tiny sample sizes so they run fast on CPU with no GPU.
"""

import pytest
from datasets import DatasetDict

from src.data.ingestion import _split_spec, load_dataset_by_name


# ── _split_spec ────────────────────────────────────────────────────────────────

def test_split_spec_with_limit():
    assert _split_spec(100, "train") == "train[:100]"

def test_split_spec_with_fraction():
    assert _split_spec(100, "validation", fraction=0.2) == "validation[:20]"

def test_split_spec_no_limit():
    assert _split_spec(None, "train") == "train"

def test_split_spec_floors_fraction():
    # int(50 * 0.2) = 10, not 10.0
    result = _split_spec(50, "test", fraction=0.2)
    assert result == "test[:10]"


# ── load_dataset_by_name ───────────────────────────────────────────────────────

def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset_by_name("nonexistent", cache_dir="data/raw/tmp")


# ── Path-VQA ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pathvqa_dataset():
    return load_dataset_by_name(
        "path-vqa",
        cache_dir="data/raw/pathvqa",
        max_samples=20,
    )

def test_pathvqa_returns_datasetdict(pathvqa_dataset):
    assert isinstance(pathvqa_dataset, DatasetDict)

def test_pathvqa_has_required_splits(pathvqa_dataset):
    assert "train" in pathvqa_dataset
    assert "validation" in pathvqa_dataset

def test_pathvqa_sample_has_required_fields(pathvqa_dataset):
    sample = pathvqa_dataset["train"][0]
    assert "image" in sample
    assert "question" in sample
    assert "answer" in sample

def test_pathvqa_train_size_capped(pathvqa_dataset):
    assert len(pathvqa_dataset["train"]) <= 20


# ── MedMCQA ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def medmcqa_dataset():
    return load_dataset_by_name(
        "medmcqa",
        cache_dir="data/raw/medmcqa",
        max_samples=20,
    )

def test_medmcqa_returns_datasetdict(medmcqa_dataset):
    assert isinstance(medmcqa_dataset, DatasetDict)

def test_medmcqa_has_required_splits(medmcqa_dataset):
    assert "train" in medmcqa_dataset
    assert "validation" in medmcqa_dataset

def test_medmcqa_sample_has_required_fields(medmcqa_dataset):
    sample = medmcqa_dataset["train"][0]
    assert "question" in sample
    assert "opa" in sample   # option A

def test_medmcqa_train_size_capped(medmcqa_dataset):
    assert len(medmcqa_dataset["train"]) <= 20
