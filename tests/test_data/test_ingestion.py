"""
Unit tests for the data ingestion layer.
Uses tiny sample sizes so they run fast on CPU with no GPU.
"""

import math
import pytest
from collections import Counter
from datasets import DatasetDict

from src.data.ingestion import _split_spec, _stratified_sample, load_dataset_by_name


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


# ── Stratified sampler ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def medmcqa_full_small():
    """Load a slightly larger slice so stratification has something to work with."""
    return load_dataset_by_name(
        "medmcqa",
        cache_dir="data/raw/medmcqa",
        max_samples=300,
    )

def test_stratified_sample_size(medmcqa_full_small):
    ds = medmcqa_full_small["train"]
    result = _stratified_sample(ds, field="subject_name", n_samples=100, seed=42)
    assert len(result) == 100

def test_stratified_sample_preserves_all_classes(medmcqa_full_small):
    ds = medmcqa_full_small["train"]
    original_subjects = set(ds["subject_name"])
    result = _stratified_sample(ds, field="subject_name", n_samples=100, seed=42)
    sampled_subjects = set(result["subject_name"])
    # Every class present in the original should appear at least once
    assert sampled_subjects == original_subjects

def test_stratified_sample_proportions(medmcqa_full_small):
    ds = medmcqa_full_small["train"]
    n = 100
    result = _stratified_sample(ds, field="subject_name", n_samples=n, seed=42)
    orig_counts  = Counter(ds["subject_name"])
    samp_counts  = Counter(result["subject_name"])
    total_orig   = len(ds)
    # Each class proportion in sample should be within 10% of original
    for subj, orig_n in orig_counts.items():
        expected_prop = orig_n / total_orig
        actual_prop   = samp_counts[subj] / n
        assert abs(actual_prop - expected_prop) < 0.10, (
            f"Subject '{subj}' proportion drifted: "
            f"expected~{expected_prop:.2f} got {actual_prop:.2f}"
        )

def test_stratified_sample_reproducible(medmcqa_full_small):
    ds = medmcqa_full_small["train"]
    r1 = _stratified_sample(ds, field="subject_name", n_samples=50, seed=42)
    r2 = _stratified_sample(ds, field="subject_name", n_samples=50, seed=42)
    assert r1["id"] == r2["id"]   # same rows, same order

def test_stratified_sample_different_seeds_differ(medmcqa_full_small):
    ds = medmcqa_full_small["train"]
    r1 = _stratified_sample(ds, field="subject_name", n_samples=50, seed=42)
    r2 = _stratified_sample(ds, field="subject_name", n_samples=50, seed=99)
    assert r1["id"] != r2["id"]


# ── MedQA-USMLE ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def medqausmle_dataset():
    return load_dataset_by_name(
        "medqa-usmle",
        cache_dir="data/raw/medqausmle",
        max_samples=20,
    )

def test_medqausmle_returns_datasetdict(medqausmle_dataset):
    assert isinstance(medqausmle_dataset, DatasetDict)

def test_medqausmle_has_required_splits(medqausmle_dataset):
    assert "train" in medqausmle_dataset
    assert "validation" in medqausmle_dataset

def test_medqausmle_sample_has_required_fields(medqausmle_dataset):
    sample = medqausmle_dataset["train"][0]
    assert "question" in sample
    assert "options" in sample
    assert "answer_idx" in sample

def test_medqausmle_options_has_four_keys(medqausmle_dataset):
    options = medqausmle_dataset["train"][0]["options"]
    assert set(options.keys()) == {"A", "B", "C", "D"}

def test_medqausmle_answer_idx_is_valid(medqausmle_dataset):
    for s in medqausmle_dataset["train"]:
        assert s["answer_idx"] in {"A", "B", "C", "D"}
