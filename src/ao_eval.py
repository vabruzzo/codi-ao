"""Evaluation suite for the trained Activation Oracle.

Metrics per category:
- Cat 1: Exact numeric match, stratified by CoT step count
- Cat 2: Binary accuracy + format correctness
- Cat 3: BLEU-4, ROUGE-L
- Cat 4: Mixed exact match + binary accuracy
- Cat 5: Exact match for numbers
- Cat 6: Binary accuracy

Also: comparison to CODI's vocab projection baseline and error localization.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import EvalConfig
from src.ao_dataset import (
    TrainingDataPoint,
    construct_batch,
    get_prompt_only,
)
from src.steering import get_steering_hook, add_hook
from src.utils import extract_number, numeric_match, binary_match, normalize_answer


@torch.no_grad()
def generate_responses(
    eval_data: list[TrainingDataPoint],
    model,
    tokenizer,
    injection_submodule,
    steering_coefficient: float = 1.0,
    batch_size: int = 16,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> list[dict]:
    """Generate responses for evaluation data using the trained AO.

    Returns:
        List of dicts with 'generated', 'target', 'meta_info' keys.
    """
    model.eval()
    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    results = []

    # Process in batches
    for start in tqdm(range(0, len(eval_data), batch_size), desc="Generating"):
        batch_points = eval_data[start:start + batch_size]

        # Get prompt-only versions for generation
        prompt_points = [get_prompt_only(dp) for dp in batch_points]
        batch = construct_batch(prompt_points, tokenizer, device)

        # Set up steering hook
        hook_fn = get_steering_hook(
            vectors=batch.steering_vectors,
            positions=batch.positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
        )

        # Generate with hook
        with add_hook(injection_submodule, hook_fn):
            generated_ids = model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode responses (skip the prompt tokens)
        for i, (gen_ids, dp) in enumerate(zip(generated_ids, batch_points)):
            prompt_len = batch.input_ids.shape[1]
            response_ids = gen_ids[prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            results.append({
                "generated": response_text,
                "target": dp.target_output,
                "ds_label": dp.ds_label,
                "meta_info": dp.meta_info,
            })

    return results


def score_numeric_exact_match(results: list[dict]) -> dict:
    """Score numeric exact match (Cat 1, Cat 4 open-ended, Cat 5)."""
    correct = 0
    total = 0
    for r in results:
        total += 1
        if numeric_match(r["generated"], r["target"]):
            correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


def score_binary(results: list[dict]) -> dict:
    """Score binary Yes/No accuracy (Cat 2, Cat 4 binary, Cat 6)."""
    correct = 0
    total = 0
    format_valid = 0
    for r in results:
        total += 1
        gen = normalize_answer(r["generated"])
        if any(v in gen for v in ("yes", "no", "true", "false")):
            format_valid += 1
        if binary_match(r["generated"], r["target"]):
            correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "format_correctness": format_valid / max(total, 1),
        "correct": correct,
        "total": total,
    }


def score_text_similarity(results: list[dict]) -> dict:
    """Score text similarity for open-ended generation (Cat 3).

    Computes simple token-level overlap metrics.
    For full BLEU/ROUGE, use the nltk or rouge_score packages.
    """
    total_overlap = 0.0
    total = 0

    for r in results:
        total += 1
        gen_tokens = set(r["generated"].lower().split())
        ref_tokens = set(r["target"].lower().split())
        if ref_tokens:
            overlap = len(gen_tokens & ref_tokens) / len(ref_tokens)
            total_overlap += overlap

    return {
        "token_recall": total_overlap / max(total, 1),
        "total": total,
    }


def evaluate(
    eval_data: list[TrainingDataPoint],
    model,
    tokenizer,
    injection_submodule,
    config: EvalConfig,
    device: str = "cuda",
) -> dict:
    """Run full evaluation across all categories.

    Returns:
        Dict of category_name -> metrics.
    """
    # Generate all responses
    results = generate_responses(
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        injection_submodule=injection_submodule,
        steering_coefficient=1.0,
        batch_size=config.eval_batch_size,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    # Group by ds_label (category)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        label = r["ds_label"] or "unknown"
        groups[label].append(r)

    # Score each category
    metrics = {}
    for label, group_results in sorted(groups.items()):
        if label.startswith("cat1"):
            metrics[label] = score_numeric_exact_match(group_results)
        elif label.startswith("cat2"):
            metrics[label] = score_binary(group_results)
        elif label.startswith("cat3"):
            metrics[label] = score_text_similarity(group_results)
        elif label.startswith("cat4"):
            # Cat4 has both open-ended and binary sub-types
            if any(r["target"] in ("Yes", "No") for r in group_results):
                metrics[label] = score_binary(group_results)
            else:
                metrics[label] = score_numeric_exact_match(group_results)
        elif label.startswith("cat5"):
            metrics[label] = score_numeric_exact_match(group_results)
        elif label.startswith("cat6"):
            # Cat6 has both binary and categorical sub-types
            if any(r["target"] in ("Yes", "No") for r in group_results):
                metrics[label] = score_binary(group_results)
            else:
                # Exact string match for "computational"/"transitional"
                correct = sum(
                    1 for r in group_results
                    if normalize_answer(r["generated"]) == normalize_answer(r["target"])
                )
                metrics[label] = {
                    "accuracy": correct / max(len(group_results), 1),
                    "correct": correct,
                    "total": len(group_results),
                }
        else:
            metrics[label] = {"total": len(group_results)}

    # Overall summary
    all_correct = sum(m.get("correct", 0) for m in metrics.values())
    all_total = sum(m.get("total", 0) for m in metrics.values())
    metrics["overall"] = {
        "accuracy": all_correct / max(all_total, 1),
        "correct": all_correct,
        "total": all_total,
    }

    return metrics


def error_localization(
    wrong_answer_data: list[TrainingDataPoint],
    model,
    tokenizer,
    injection_submodule,
    config: EvalConfig,
    device: str = "cuda",
) -> list[dict]:
    """Run error localization on problems where CODI got the wrong answer.

    For each problem:
    1. Ask the AO about each thought's intermediate result
    2. Compare to ground truth
    3. Identify which step diverges

    Args:
        wrong_answer_data: TrainingDataPoints for wrong-answer problems,
                          one per thought per problem (Cat 1 format).

    Returns:
        List of per-problem error localization results.
    """
    results = generate_responses(
        eval_data=wrong_answer_data,
        model=model,
        tokenizer=tokenizer,
        injection_submodule=injection_submodule,
        batch_size=config.eval_batch_size,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    # Group by problem_id
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        pid = r["meta_info"].get("problem_id", "unknown")
        by_problem[pid].append(r)

    localization_results = []
    for pid, problem_results in by_problem.items():
        # Check which thought's claimed result diverges from ground truth
        divergent_step = None
        for r in sorted(problem_results, key=lambda x: x["meta_info"].get("thought_idx", 0)):
            if not numeric_match(r["generated"], r["target"]):
                divergent_step = r["meta_info"].get("thought_idx", -1)
                break

        localization_results.append({
            "problem_id": pid,
            "divergent_step": divergent_step,
            "num_steps": len(problem_results),
            "all_correct": divergent_step is None,
            "details": problem_results,
        })

    return localization_results


def save_results(metrics: dict, output_path: str) -> None:
    """Save evaluation results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {path}")


def print_results(metrics: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for category, scores in sorted(metrics.items()):
        if category == "overall":
            continue
        acc = scores.get("accuracy", scores.get("token_recall", "N/A"))
        total = scores.get("total", 0)
        if isinstance(acc, float):
            print(f"  {category:40s}  acc={acc:.3f}  (n={total})")
        else:
            print(f"  {category:40s}  {acc}  (n={total})")

    if "overall" in metrics:
        overall = metrics["overall"]
        print("-" * 60)
        print(f"  {'OVERALL':40s}  acc={overall['accuracy']:.3f}  (n={overall['total']})")
    print("=" * 60)
