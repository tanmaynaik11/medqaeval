"""
Phase 4: Evaluation Framework
Compares MedVision (fine-tuned) vs Qwen2-VL-7B on medical benchmarks.

Usage:
    # Evaluate MedVision on all benchmarks
    python scripts/evaluate.py --model medvision

    # Evaluate Qwen2-VL baseline
    python scripts/evaluate.py --model qwen2vl

    # Evaluate both and generate comparison report
    python scripts/evaluate.py --model both

    # Quick eval with fewer samples (for smoke testing)
    python scripts/evaluate.py --model both --max_samples 100
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.utils.env import load_env
from src.data.ingestion import load_dataset_by_name
from src.evaluation.evaluator import (
    MedVisionEvaluator,
    Qwen2VLEvaluator,
    evaluate_pathvqa,
    evaluate_mcqa,
)

setup_logging(level="INFO", log_file="logs/evaluation.log")
logger = logging.getLogger(__name__)
load_env()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["medvision", "qwen2vl", "both"], default="both")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per benchmark (None = use full test set)")
    p.add_argument("--stage1_checkpoint", default="artifacts/checkpoints/stage1/best")
    p.add_argument("--stage2_checkpoint", default="artifacts/checkpoints/stage2/best")
    p.add_argument("--cache_dir", default="artifacts/model_cache")
    p.add_argument("--output", default="artifacts/eval_results.json")
    return p.parse_args()


def load_eval_datasets(cache_dir: str) -> dict:
    logger.info("Loading evaluation datasets...")
    base = Path(cache_dir).parent

    pathvqa = load_dataset_by_name("path-vqa",    cache_dir=str(base / "data/raw/pathvqa"))
    medmcqa = load_dataset_by_name("medmcqa",     cache_dir=str(base / "data/raw/medmcqa"))
    usmle   = load_dataset_by_name("medqa-usmle", cache_dir=str(base / "data/raw/medqausmle"))

    return {
        "pathvqa_test": pathvqa["test"],
        "medmcqa_val":  medmcqa["validation"],
        "usmle_test":   usmle["test"],
    }


def run_evaluation(evaluator, datasets: dict, max_samples, model_name: str) -> dict:
    results = {"model": model_name, "timestamp": datetime.now().isoformat()}

    # ── PathVQA ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}\n{model_name} — PathVQA\n{'='*50}")
    results["pathvqa"] = evaluate_pathvqa(
        evaluator, datasets["pathvqa_test"], max_samples=max_samples
    )
    logger.info(f"PathVQA overall: {results['pathvqa']['overall']['accuracy']}%")
    if "yes_no" in results["pathvqa"]:
        logger.info(f"  Yes/No: {results['pathvqa']['yes_no']['accuracy']}%")
    if "open" in results["pathvqa"]:
        logger.info(f"  Open:   {results['pathvqa']['open']['accuracy']}%")

    # ── MedMCQA ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}\n{model_name} — MedMCQA\n{'='*50}")

    # MedMCQA cop field is int (0-3), map to letter for comparison
    import datasets as hf_datasets
    mcqa_ds = datasets["medmcqa_val"]
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    mcqa_ds = mcqa_ds.map(lambda x: {"cop_str": answer_map.get(x["cop"], "A")})

    results["medmcqa"] = evaluate_mcqa(
        evaluator,
        mcqa_ds,
        option_fields={"a": "opa", "b": "opb", "c": "opc", "d": "opd"},
        answer_field="cop_str",
        max_samples=max_samples,
        desc="MedMCQA",
    )
    logger.info(f"MedMCQA: {results['medmcqa']['accuracy']}%")

    # ── MedQA-USMLE ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}\n{model_name} — MedQA-USMLE\n{'='*50}")
    # USMLE options are stored as a nested dict {"A": "...", "B": "..."}
    # Flatten into separate columns so evaluate_mcqa can access them directly
    usmle_ds = datasets["usmle_test"].map(lambda x: {
        "opt_a": x["options"]["A"],
        "opt_b": x["options"]["B"],
        "opt_c": x["options"]["C"],
        "opt_d": x["options"]["D"],
    })
    results["usmle"] = evaluate_mcqa(
        evaluator,
        usmle_ds,
        option_fields={"a": "opt_a", "b": "opt_b", "c": "opt_c", "d": "opt_d"},
        answer_field="answer_idx",
        max_samples=max_samples,
        desc="MedQA-USMLE",
    )
    logger.info(f"MedQA-USMLE: {results['usmle']['accuracy']}%")

    return results


def print_comparison(mv: dict, qvl: dict):
    print("\n" + "="*62)
    print("  EVALUATION RESULTS — MedVision vs Qwen2-VL-7B")
    print("="*62)
    print(f"{'Benchmark':<25} {'MedVision':>12} {'Qwen2-VL':>12} {'Delta':>10}")
    print("-"*62)

    benchmarks = [
        ("PathVQA (overall)",  "pathvqa", "overall"),
        ("PathVQA (yes/no)",   "pathvqa", "yes_no"),
        ("PathVQA (open)",     "pathvqa", "open"),
        ("MedMCQA",            "medmcqa", None),
        ("MedQA-USMLE",        "usmle",   None),
    ]

    for name, key, subkey in benchmarks:
        try:
            mv_acc  = mv[key][subkey]["accuracy"]  if subkey else mv[key]["accuracy"]
            qvl_acc = qvl[key][subkey]["accuracy"] if subkey else qvl[key]["accuracy"]
            delta   = mv_acc - qvl_acc
            delta_s = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            print(f"{name:<25} {mv_acc:>11.1f}% {qvl_acc:>11.1f}% {delta_s:>10}")
        except KeyError:
            print(f"{name:<25} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    print("="*62)


def main():
    args    = parse_args()
    datasets = load_eval_datasets(args.cache_dir)
    all_results = {}

    if args.model in ("medvision", "both"):
        logger.info("Loading MedVision evaluator...")
        ev = MedVisionEvaluator(
            stage1_checkpoint=args.stage1_checkpoint,
            stage2_checkpoint=args.stage2_checkpoint,
            cache_dir=args.cache_dir,
        )
        all_results["medvision"] = run_evaluation(ev, datasets, args.max_samples, "MedVision")
        del ev
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

    if args.model in ("qwen2vl", "both"):
        logger.info("Loading Qwen2-VL evaluator...")
        ev = Qwen2VLEvaluator(cache_dir=args.cache_dir)
        all_results["qwen2vl"] = run_evaluation(ev, datasets, args.max_samples, "Qwen2-VL")
        del ev

    # Save results to JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    # Print comparison table
    if "medvision" in all_results and "qwen2vl" in all_results:
        print_comparison(all_results["medvision"], all_results["qwen2vl"])


if __name__ == "__main__":
    main()
