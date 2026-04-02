"""
Script to download and validate all configured datasets.
Run this once to populate data/raw/ before training.
Usage: python scripts/download_data.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import load_dataset_by_name
from src.data.preprocessing import preprocess_pathvqa_sample, preprocess_medmcqa_sample
from src.data.dataset import MedVQADataset, MedTextDataset

from src.utils.env import load_env
load_env()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_pathvqa():
    logger.info("=== Testing Path-VQA pipeline ===")
    ds = load_dataset_by_name("path-vqa", cache_dir="data/raw/pathvqa", max_samples=100)
    
    train_ds = MedVQADataset(
        hf_dataset=ds["train"],
        preprocess_fn=preprocess_pathvqa_sample,
    )
    
    sample = train_ds[0]
    logger.info(f"pixel_values shape : {sample['pixel_values'].shape}")
    logger.info(f"prompt             : {sample['prompt'][:80]}...")
    logger.info(f"label              : {sample['label']}")
    logger.info(f"source             : {sample['source']}")
    logger.info("Path-VQA: PASSED\n")


def test_medmcqa():
    logger.info("=== Testing MedMCQA pipeline ===")
    ds = load_dataset_by_name("medmcqa", cache_dir="data/raw/medmcqa", max_samples=200)
    
    train_ds = MedTextDataset(
        hf_dataset=ds["train"],
        preprocess_fn=preprocess_medmcqa_sample,
    )
    
    sample = train_ds[0]
    logger.info(f"prompt  : {sample['prompt'][:120]}...")
    logger.info(f"label   : {sample['label']}")
    logger.info(f"source  : {sample['source']}")
    logger.info("MedMCQA: PASSED\n")


if __name__ == "__main__":
    test_pathvqa()
    test_medmcqa()
    logger.info("All datasets validated successfully.")
