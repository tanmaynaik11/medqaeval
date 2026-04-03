"""
Evaluation metrics for medical VQA and MCQ benchmarks.

Answer extraction strategy:
  Model outputs free text like "A. Hypertension. Because kidneys..."
  We extract the option letter (A/B/C/D) or yes/no from the beginning.
  Exact match on the extracted answer vs ground truth label.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_option_letter(text: str) -> Optional[str]:
    """
    Extract the answer option letter from model output.

    Handles formats like:
      "A"
      "A."
      "A. Hypertension"
      "The answer is A"
      "Answer: B. Because..."
    """
    if not text:
        return None

    text = text.strip()

    # Direct single letter
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()

    # Starts with letter + period/space
    m = re.match(r"^([A-Da-d])[.\s]", text)
    if m:
        return m.group(1).upper()

    # "The answer is X" or "Answer: X"
    m = re.search(r"(?:answer(?:\s+is)?|answer:)\s*([A-Da-d])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Any standalone letter A-D
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()

    return None


def extract_yes_no(text: str) -> Optional[str]:
    """
    Extract yes/no answer from model output.
    PathVQA has ~80% yes/no questions.
    """
    if not text:
        return None

    text = text.strip().lower()

    if text in ("yes", "no"):
        return text

    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"

    return None


def compute_accuracy(predictions: list[str], labels: list[str]) -> dict:
    """
    Compute exact match accuracy.

    Args:
        predictions: list of extracted answer strings
        labels:      list of ground truth answer strings

    Returns:
        dict with 'accuracy', 'correct', 'total', 'skipped'
    """
    assert len(predictions) == len(labels), "predictions and labels must be same length"

    correct  = 0
    skipped  = 0  # model output couldn't be parsed

    for pred, label in zip(predictions, labels):
        if pred is None:
            skipped += 1
            continue
        if pred.lower().strip() == label.lower().strip():
            correct += 1

    total    = len(predictions)
    answered = total - skipped
    accuracy = correct / answered if answered > 0 else 0.0

    return {
        "accuracy":      round(accuracy * 100, 2),
        "correct":       correct,
        "answered":      answered,
        "total":         total,
        "skipped":       skipped,
    }


def compute_pathvqa_accuracy(
    predictions: list[str],
    labels:      list[str],
    question_types: list[str],   # "yes/no" or "open"
) -> dict:
    """
    PathVQA has two question types:
      - yes/no  (~80%): binary, high baseline (50% random)
      - open    (~20%): free-form, harder

    Compute accuracy separately for each type.
    """
    yn_preds, yn_labels = [], []
    open_preds, open_labels = [], []

    for pred, label, qtype in zip(predictions, labels, question_types):
        if qtype == "yes/no":
            yn_preds.append(pred)
            yn_labels.append(label)
        else:
            open_preds.append(pred)
            open_labels.append(label)

    results = {"overall": compute_accuracy(predictions, labels)}

    if yn_preds:
        results["yes_no"] = compute_accuracy(yn_preds, yn_labels)
    if open_preds:
        results["open"]   = compute_accuracy(open_preds, open_labels)

    return results
