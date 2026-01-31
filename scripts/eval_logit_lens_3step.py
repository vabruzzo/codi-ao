#!/usr/bin/env python3
"""
Logit Lens evaluation for 3-step problems.

Tests Logit Lens's ability to extract:
- Step 1 result from z2 (position 1)
- Step 2 result from z4 (position 3)
- Step 3 / final answer from z6 (position 5)
- Operation type from z2

This provides the baseline comparison for Activation Oracle results.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_codi_model(config_path="configs/default.yaml"):
    """Load the CODI model."""
    from src.codi_wrapper import CODIWrapper
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    codi = CODIWrapper.from_pretrained(
        checkpoint_path=config["model"]["codi_checkpoint"],
        model_name_or_path=config["model"]["codi_base_model"],
        lora_r=config["model"]["codi_lora_r"],
        lora_alpha=config["model"]["codi_lora_alpha"],
        num_latent=config["model"]["codi_num_latent"],
        use_prj=config["model"]["codi_use_prj"],
        device=device,
    )
    return codi


def get_number_token_ids(tokenizer, number: int) -> list[int]:
    """
    Get all possible token IDs that could represent a number.
    Handles various formats: "42", " 42", "42.", etc.
    """
    token_ids = set()
    
    # Various representations
    representations = [
        str(number),           # "42"
        f" {number}",          # " 42"
        f"{number}.",          # "42."
        f" {number}.",         # " 42."
        f"{number},",          # "42,"
        f" {number},",         # " 42,"
    ]
    
    for rep in representations:
        tokens = tokenizer.encode(rep, add_special_tokens=False)
        # Take first token (the number itself)
        if tokens:
            token_ids.add(tokens[0])
    
    return list(token_ids)


def get_operation_token_ids(tokenizer) -> dict[str, list[int]]:
    """Get token IDs for operation-related tokens."""
    ops = {
        "add": ["add", "addition", "plus", "+", "Add", "Addition", "Plus"],
        "sub": ["sub", "subtraction", "minus", "-", "Sub", "Subtraction", "Minus"],
        "mul": ["mul", "multiplication", "times", "*", "×", "Mul", "Multiplication", "Times"],
    }
    
    result = {}
    for op, keywords in ops.items():
        token_ids = set()
        for kw in keywords:
            tokens = tokenizer.encode(kw, add_special_tokens=False)
            if tokens:
                token_ids.add(tokens[0])
            # Also try with space prefix
            tokens = tokenizer.encode(f" {kw}", add_special_tokens=False)
            if tokens:
                token_ids.add(tokens[0])
        result[op] = list(token_ids)
    
    return result


def analyze_latent_for_number(
    codi,
    latent_vector: torch.Tensor,
    target_number: int,
    top_k: int = 10,
) -> dict:
    """
    Analyze a latent vector using Logit Lens to check for a number.
    
    Returns:
    - top_k_tokens: Top K predicted tokens
    - target_in_top_k: Whether target number is in top K
    - top_1_correct: Whether top 1 is the target
    - target_rank: Rank of target (if found)
    - target_prob: Probability of target tokens
    """
    # Get logit lens result (returns LogitLensResult object)
    result = codi.logit_lens(latent_vector, top_k=top_k)
    
    # Extract from result
    layer_result = result.layer_results[0]
    top_tokens = layer_result["top_tokens"]
    top_probs = layer_result["top_probs"]
    top_indices = layer_result["top_indices"]
    
    # Get target token IDs
    target_ids = get_number_token_ids(codi.tokenizer, target_number)
    
    # Check if target is in top K
    top_k_set = set(top_indices)
    target_in_top_k = any(tid in top_k_set for tid in target_ids)
    
    # Get target probability from top-k if present
    target_prob = 0.0
    for idx, prob in zip(top_indices, top_probs):
        if idx in target_ids:
            target_prob += prob
    
    # Find rank of target within top-k
    target_rank = None
    for rank, idx in enumerate(top_indices):
        if idx in target_ids:
            target_rank = rank + 1
            break
    
    # Check top 1
    top_1_correct = top_indices[0] in target_ids
    
    return {
        "top_k_tokens": list(zip(top_tokens, top_probs)),
        "target_in_top_k": target_in_top_k,
        "top_1_correct": top_1_correct,
        "target_rank": target_rank,
        "target_prob": target_prob,
    }


def analyze_latent_for_operation(
    codi,
    latent_vector: torch.Tensor,
    target_op: str,
    top_k: int = 10,
) -> dict:
    """
    Analyze a latent vector using Logit Lens to check for operation.
    """
    # Get logit lens result
    result = codi.logit_lens(latent_vector, top_k=top_k)
    
    # Extract from result
    layer_result = result.layer_results[0]
    top_tokens = layer_result["top_tokens"]
    top_probs = layer_result["top_probs"]
    top_indices = layer_result["top_indices"]
    
    # Get operation token IDs
    op_token_ids = get_operation_token_ids(codi.tokenizer)
    target_ids = op_token_ids.get(target_op, [])
    
    # Check if target is in top K
    top_k_set = set(top_indices)
    target_in_top_k = any(tid in top_k_set for tid in target_ids)
    
    # Get probability for each operation from top-k
    op_probs = {"add": 0.0, "sub": 0.0, "mul": 0.0}
    for idx, prob in zip(top_indices, top_probs):
        for op, ids in op_token_ids.items():
            if idx in ids:
                op_probs[op] += prob
    
    # Predicted operation (highest prob among detected)
    # If no operation tokens in top-k, use "add" as default
    if sum(op_probs.values()) > 0:
        predicted_op = max(op_probs, key=op_probs.get)
    else:
        # Fall back to checking if any op keywords appear in top tokens
        predicted_op = "add"  # Default
        for token in top_tokens:
            token_lower = token.lower().strip()
            if any(kw in token_lower for kw in ["sub", "minus", "-"]):
                predicted_op = "sub"
                break
            elif any(kw in token_lower for kw in ["mul", "times", "*", "×"]):
                predicted_op = "mul"
                break
            elif any(kw in token_lower for kw in ["add", "plus", "+"]):
                predicted_op = "add"
                break
    
    correct = (predicted_op == target_op)
    
    return {
        "top_k_tokens": list(zip(top_tokens, top_probs)),
        "target_in_top_k": target_in_top_k,
        "op_probs": op_probs,
        "predicted_op": predicted_op,
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=str, default="data/synthetic_problems_3step.json")
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/logit_lens_3step.json")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Logit Lens Evaluation for 3-Step Problems")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Top-K: {args.top_k}")
    
    # Load problems
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    all_problems = data["problems"]
    
    # Use last N as test set
    test_problems = all_problems[-args.n_test:]
    print(f"Testing on {len(test_problems)} held-out problems")
    
    # Load CODI
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    # Initialize results
    results = {
        "step1_extraction": {
            "top_k_correct": 0,
            "top_1_correct": 0,
            "total": 0,
            "avg_target_prob": 0,
            "avg_rank": 0,
            "ranks": [],
        },
        "step2_extraction": {
            "top_k_correct": 0,
            "top_1_correct": 0,
            "total": 0,
            "avg_target_prob": 0,
            "avg_rank": 0,
            "ranks": [],
        },
        "step3_extraction_z5": {
            "top_k_correct": 0,
            "top_1_correct": 0,
            "total": 0,
            "avg_target_prob": 0,
            "avg_rank": 0,
            "ranks": [],
        },
        "step3_extraction_z6": {
            "top_k_correct": 0,
            "top_1_correct": 0,
            "total": 0,
            "avg_target_prob": 0,
            "avg_rank": 0,
            "ranks": [],
        },
        "operation_detection": {
            "correct": 0,
            "total": 0,
            "by_op": {},
            "confusion": {"add": {}, "sub": {}, "mul": {}},
        },
        "config": {
            "n_test": args.n_test,
            "top_k": args.top_k,
        },
        "examples": [],
    }
    
    # Run evaluations
    print("\nRunning evaluations...")
    print("-" * 60)
    
    for problem in tqdm(test_problems, desc="Evaluating"):
        # Collect latents
        latent_result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        
        if len(latent_result.latent_vectors) < 6:
            continue
        
        z2 = latent_result.latent_vectors[1]  # Position 1 - step 1
        z4 = latent_result.latent_vectors[3]  # Position 3 - step 2
        z5 = latent_result.latent_vectors[4]  # Position 4 - maybe step 3?
        z6 = latent_result.latent_vectors[5]  # Position 5 - step 3
        
        step1 = problem["step1"]
        step2 = problem["step2"]
        step3 = problem["step3"]
        op = problem["operation"]
        
        # =====================================================================
        # STEP 1 EXTRACTION (from z2)
        # =====================================================================
        analysis = analyze_latent_for_number(codi, z2, step1, args.top_k)
        
        results["step1_extraction"]["total"] += 1
        if analysis["target_in_top_k"]:
            results["step1_extraction"]["top_k_correct"] += 1
        if analysis["top_1_correct"]:
            results["step1_extraction"]["top_1_correct"] += 1
        results["step1_extraction"]["avg_target_prob"] += analysis["target_prob"]
        if analysis["target_rank"]:
            results["step1_extraction"]["ranks"].append(analysis["target_rank"])
        
        # =====================================================================
        # STEP 2 EXTRACTION (from z4)
        # =====================================================================
        analysis = analyze_latent_for_number(codi, z4, step2, args.top_k)
        
        results["step2_extraction"]["total"] += 1
        if analysis["target_in_top_k"]:
            results["step2_extraction"]["top_k_correct"] += 1
        if analysis["top_1_correct"]:
            results["step2_extraction"]["top_1_correct"] += 1
        results["step2_extraction"]["avg_target_prob"] += analysis["target_prob"]
        if analysis["target_rank"]:
            results["step2_extraction"]["ranks"].append(analysis["target_rank"])
        
        # =====================================================================
        # STEP 3 / FINAL ANSWER EXTRACTION (from z5)
        # =====================================================================
        analysis = analyze_latent_for_number(codi, z5, step3, args.top_k)
        
        results["step3_extraction_z5"]["total"] += 1
        if analysis["target_in_top_k"]:
            results["step3_extraction_z5"]["top_k_correct"] += 1
        if analysis["top_1_correct"]:
            results["step3_extraction_z5"]["top_1_correct"] += 1
        results["step3_extraction_z5"]["avg_target_prob"] += analysis["target_prob"]
        if analysis["target_rank"]:
            results["step3_extraction_z5"]["ranks"].append(analysis["target_rank"])
        
        # =====================================================================
        # STEP 3 / FINAL ANSWER EXTRACTION (from z6)
        # =====================================================================
        analysis = analyze_latent_for_number(codi, z6, step3, args.top_k)
        
        results["step3_extraction_z6"]["total"] += 1
        if analysis["target_in_top_k"]:
            results["step3_extraction_z6"]["top_k_correct"] += 1
        if analysis["top_1_correct"]:
            results["step3_extraction_z6"]["top_1_correct"] += 1
        results["step3_extraction_z6"]["avg_target_prob"] += analysis["target_prob"]
        if analysis["target_rank"]:
            results["step3_extraction_z6"]["ranks"].append(analysis["target_rank"])
        
        # =====================================================================
        # OPERATION DETECTION (from z2)
        # =====================================================================
        analysis = analyze_latent_for_operation(codi, z2, op, args.top_k)
        
        results["operation_detection"]["total"] += 1
        if analysis["correct"]:
            results["operation_detection"]["correct"] += 1
        
        # Per-operation stats
        results["operation_detection"]["by_op"].setdefault(op, {"correct": 0, "total": 0})
        results["operation_detection"]["by_op"][op]["total"] += 1
        if analysis["correct"]:
            results["operation_detection"]["by_op"][op]["correct"] += 1
        
        # Confusion matrix
        pred = analysis["predicted_op"]
        results["operation_detection"]["confusion"][op].setdefault(pred, 0)
        results["operation_detection"]["confusion"][op][pred] += 1
    
    # Calculate averages
    for key in ["step1_extraction", "step2_extraction", "step3_extraction_z5", "step3_extraction_z6"]:
        total = results[key]["total"]
        if total > 0:
            results[key]["avg_target_prob"] /= total
            if results[key]["ranks"]:
                results[key]["avg_rank"] = sum(results[key]["ranks"]) / len(results[key]["ranks"])
    
    # =========================================================================
    # PRINT RESULTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("LOGIT LENS RESULTS")
    print("=" * 60)
    
    def pct(correct, total):
        return f"{100 * correct / total:.1f}%" if total > 0 else "N/A"
    
    print(f"\n--- Number Extraction (Top-{args.top_k} Check) ---")
    for key, label in [
        ("step1_extraction", "Step 1 (z2)"),
        ("step2_extraction", "Step 2 (z4)"),
        ("step3_extraction_z5", "Step 3 / Final (z5)"),
        ("step3_extraction_z6", "Step 3 / Final (z6)"),
    ]:
        r = results[key]
        print(f"\n{label}:")
        print(f"  Top-{args.top_k} accuracy: {pct(r['top_k_correct'], r['total'])} ({r['top_k_correct']}/{r['total']})")
        print(f"  Top-1 accuracy:   {pct(r['top_1_correct'], r['total'])} ({r['top_1_correct']}/{r['total']})")
        print(f"  Avg target prob:  {r['avg_target_prob']:.4f}")
        if r['ranks']:
            print(f"  Avg rank (when found): {r['avg_rank']:.1f}")
    
    print(f"\n--- Operation Detection (from z2) ---")
    op_r = results["operation_detection"]
    print(f"Overall: {pct(op_r['correct'], op_r['total'])} ({op_r['correct']}/{op_r['total']})")
    
    for op in ["add", "sub", "mul"]:
        if op in op_r["by_op"]:
            op_data = op_r["by_op"][op]
            print(f"  {op}: {pct(op_data['correct'], op_data['total'])}")
    
    print("\nConfusion matrix (true -> predicted):")
    for true_op in ["add", "sub", "mul"]:
        preds = op_r["confusion"].get(true_op, {})
        pred_str = ", ".join(f"{p}:{c}" for p, c in sorted(preds.items()))
        print(f"  {true_op}: {pred_str}")
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (for comparison with AO)")
    print("=" * 60)
    
    print(f"\n{'Task':<25} {'Top-{} Acc'.format(args.top_k):<15} {'Top-1 Acc':<15} {'Avg Prob':<15}")
    print("-" * 70)
    
    for key, label in [
        ("step1_extraction", "Step 1 (z2)"),
        ("step2_extraction", "Step 2 (z4)"),
        ("step3_extraction_z5", "Step 3 / Final (z5)"),
        ("step3_extraction_z6", "Step 3 / Final (z6)"),
    ]:
        r = results[key]
        top_k_acc = pct(r["top_k_correct"], r["total"])
        top_1_acc = pct(r["top_1_correct"], r["total"])
        avg_prob = f"{r['avg_target_prob']:.4f}"
        print(f"{label:<25} {top_k_acc:<15} {top_1_acc:<15} {avg_prob:<15}")
    
    op_acc = pct(op_r["correct"], op_r["total"])
    print(f"{'Operation (z2)':<25} {op_acc:<15} {op_acc:<15} {'N/A':<15}")
    
    # Save results
    # Remove ranks list for cleaner JSON (can be very large)
    for key in ["step1_extraction", "step2_extraction", "step3_extraction_z5", "step3_extraction_z6"]:
        del results[key]["ranks"]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
