"""Shared utility functions for the CODI-AO pipeline."""

import re
import json
import random
from pathlib import Path

import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_number(text: str) -> float | None:
    """Extract the last number from a string. Returns None if no number found."""
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    return float(numbers[-1])


def format_number(value: float) -> str:
    """Format a number for display: show as int if whole, otherwise float."""
    if value == int(value):
        return str(int(value))
    return str(value)


def save_json(data: list | dict, path: str | Path) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> list | dict:
    """Load data from JSON."""
    with open(path) as f:
        return json.load(f)


def save_activations(data: dict, path: str | Path) -> None:
    """Save activation data as a torch .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_activations(path: str | Path) -> dict:
    """Load activation data from a torch .pt file."""
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().lower()
    answer = answer.rstrip(".")
    return answer


def numeric_match(predicted: str, target: str, tolerance: float = 1e-3) -> bool:
    """Check if predicted and target numbers match within tolerance."""
    pred_num = extract_number(predicted)
    target_num = extract_number(target)
    if pred_num is None or target_num is None:
        return False
    return abs(pred_num - target_num) < tolerance


def binary_match(predicted: str, target: str) -> bool:
    """Check if predicted matches target for Yes/No questions."""
    pred = normalize_answer(predicted)
    tgt = normalize_answer(target)
    # Handle common variants
    yes_variants = {"yes", "true", "correct", "1"}
    no_variants = {"no", "false", "incorrect", "0"}
    pred_is_yes = any(v in pred for v in yes_variants)
    pred_is_no = any(v in pred for v in no_variants)
    tgt_is_yes = tgt in yes_variants
    tgt_is_no = tgt in no_variants
    if pred_is_yes and not pred_is_no:
        return tgt_is_yes
    if pred_is_no and not pred_is_yes:
        return tgt_is_no
    return False
