#!/usr/bin/env python3
"""
Logit Lens evaluation on holdout data for direct comparison with AO.

Evaluates:
1. Step 1/2/3 extraction accuracy
2. Operation detection (checking for operation tokens)
3. Performance on novel vs seen values
4. CODI correctness correlation
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codi_wrapper import CODIWrapper


def load_codi():
    """Load CODI model."""
    return CODIWrapper.from_pretrained()


def extract_number(text):
    """Extract first number from text."""
    if text is None:
        return None
    numbers = re.findall(r'-?\d+', str(text))
    return int(numbers[0]) if numbers else None


def analyze_logit_lens_for_number(codi, latent, target_number, top_k=10):
    """
    Use Logit Lens to check if target number is in top-k tokens.
    Returns rank and probability if found, else None.
    """
    result = codi.logit_lens(latent)
    
    target_str = str(target_number)
    
    for rank, (token, prob) in enumerate(zip(result.top_tokens, result.top_probs)):
        token_clean = token.strip()
        if token_clean == target_str or token_clean == f" {target_str}":
            return {
                "found": True,
                "rank": rank + 1,
                "probability": prob,
                "top_token": result.top_tokens[0].strip(),
                "top_prob": result.top_probs[0],
            }
    
    return {
        "found": False,
        "rank": None,
        "probability": 0.0,
        "top_token": result.top_tokens[0].strip() if result.top_tokens else None,
        "top_prob": result.top_probs[0] if result.top_probs else 0.0,
    }


def analyze_logit_lens_for_operation(codi, latent, target_op, top_k=20):
    """
    Check if operation-related tokens appear in top-k.
    """
    result = codi.logit_lens(latent, top_k=top_k)
    
    op_tokens = {
        "add": ["+", "add", "plus", "sum", "added", "adding"],
        "sub": ["-", "sub", "minus", "subtract", "subtracted", "less"],
        "mul": ["*", "×", "x", "mul", "times", "multiply", "multiplied"],
    }
    
    target_tokens = op_tokens.get(target_op, [])
    
    for rank, (token, prob) in enumerate(zip(result.top_tokens, result.top_probs)):
        token_clean = token.strip().lower()
        if token_clean in target_tokens:
            return {
                "found": True,
                "rank": rank + 1,
                "probability": prob,
                "matched_token": token.strip(),
            }
    
    return {
        "found": False,
        "rank": None,
        "probability": 0.0,
        "matched_token": None,
    }


def categorize_rarity(value, freq_dict, rare_threshold=10, common_threshold=30):
    """Categorize value rarity."""
    count = freq_dict.get(str(value), freq_dict.get(value, 0))
    if count <= rare_threshold:
        return "rare"
    elif count >= common_threshold:
        return "common"
    return "medium"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=str, default="data/problems_holdout.json")
    parser.add_argument("--output", type=str, default="results/logit_lens_holdout_eval.json")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Logit Lens Evaluation on Holdout Data")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    
    test_problems = data["test_problems"]
    holdout_info = data["holdout"]
    train_freqs = data.get("train_value_frequencies", {})
    
    print(f"Test problems: {len(test_problems)}")
    
    # Load CODI
    print("\nLoading CODI model...")
    codi = load_codi()
    
    # Initialize results
    results = {
        "config": {
            "problems": args.problems,
            "top_k": args.top_k,
            "n_test": len(test_problems),
        },
        
        # Overall accuracy (top-1 match)
        "overall": {
            "step1_z2": {"correct": 0, "total": 0},
            "step2_z4": {"correct": 0, "total": 0},
            "step3_z6": {"correct": 0, "total": 0},
            "operation_z2": {"correct": 0, "total": 0},
        },
        
        # Top-k match (found anywhere in top-k)
        "top_k_match": {
            "step1_z2": {"found": 0, "total": 0},
            "step2_z4": {"found": 0, "total": 0},
            "step3_z6": {"found": 0, "total": 0},
        },
        
        # Holdout analysis
        "holdout": {
            "by_type": {
                "tuple": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}},
                "value": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}},
                "seen": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}},
            },
            "novel_step1": {"correct": 0, "total": 0},
            "seen_step1": {"correct": 0, "total": 0},
            "novel_step3": {"correct": 0, "total": 0},
            "seen_step3": {"correct": 0, "total": 0},
        },
        
        # CODI correctness
        "codi_analysis": {
            "codi_correct": 0,
            "codi_total": 0,
            "ll_step3_when_codi_correct": {"correct": 0, "total": 0},
            "ll_step3_when_codi_wrong": {"correct": 0, "total": 0},
        },
        
        # Rarity
        "rarity": {
            "step1_rare": {"correct": 0, "total": 0},
            "step1_medium": {"correct": 0, "total": 0},
            "step1_common": {"correct": 0, "total": 0},
            "step3_rare": {"correct": 0, "total": 0},
            "step3_medium": {"correct": 0, "total": 0},
            "step3_common": {"correct": 0, "total": 0},
        },
        
        # By operation
        "by_operation": {
            "add": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op_detect": {"c": 0, "t": 0}},
            "sub": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op_detect": {"c": 0, "t": 0}},
            "mul": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op_detect": {"c": 0, "t": 0}},
        },
        
        # Probability distributions
        "prob_stats": {
            "step1_found_probs": [],
            "step3_found_probs": [],
        },
    }
    
    # Evaluate
    print("\nEvaluating...")
    
    for problem in tqdm(test_problems, desc="Evaluating"):
        # Collect latents
        ground_truth = str(problem["step3"])
        latent_result = codi.collect_latents(
            problem["prompt"],
            ground_truth_answer=ground_truth,
            return_hidden_states=False
        )
        
        if len(latent_result.latent_vectors) < 6:
            continue
        
        latents = latent_result.latent_vectors[:6]
        codi_correct = latent_result.is_correct
        
        step1 = problem["step1"]
        step2 = problem["step2"]
        step3 = problem["step3"]
        op = problem["operation"]
        holdout_type = problem.get("holdout_type", "unknown")
        novel_s1 = problem.get("novel_step1", False)
        novel_s3 = problem.get("novel_step3", False)
        
        z2 = latents[1]
        z4 = latents[3]
        z6 = latents[5]
        
        # === Step 1 (z2) ===
        s1_result = analyze_logit_lens_for_number(codi, z2, step1, args.top_k)
        s1_top1_correct = (s1_result["top_token"] == str(step1))
        
        results["overall"]["step1_z2"]["total"] += 1
        results["top_k_match"]["step1_z2"]["total"] += 1
        if s1_top1_correct:
            results["overall"]["step1_z2"]["correct"] += 1
        if s1_result["found"]:
            results["top_k_match"]["step1_z2"]["found"] += 1
            results["prob_stats"]["step1_found_probs"].append(s1_result["probability"])
        
        # Holdout for step1
        results["holdout"]["by_type"][holdout_type]["step1"]["t"] += 1
        if s1_top1_correct:
            results["holdout"]["by_type"][holdout_type]["step1"]["c"] += 1
        
        if novel_s1:
            results["holdout"]["novel_step1"]["total"] += 1
            if s1_top1_correct:
                results["holdout"]["novel_step1"]["correct"] += 1
        else:
            results["holdout"]["seen_step1"]["total"] += 1
            if s1_top1_correct:
                results["holdout"]["seen_step1"]["correct"] += 1
        
        # Rarity
        s1_rarity = categorize_rarity(step1, train_freqs.get("step1", {}))
        results["rarity"][f"step1_{s1_rarity}"]["total"] += 1
        if s1_top1_correct:
            results["rarity"][f"step1_{s1_rarity}"]["correct"] += 1
        
        # By operation
        results["by_operation"][op]["step1"]["t"] += 1
        if s1_top1_correct:
            results["by_operation"][op]["step1"]["c"] += 1
        
        # === Step 2 (z4) ===
        s2_result = analyze_logit_lens_for_number(codi, z4, step2, args.top_k)
        s2_top1_correct = (s2_result["top_token"] == str(step2))
        
        results["overall"]["step2_z4"]["total"] += 1
        results["top_k_match"]["step2_z4"]["total"] += 1
        if s2_top1_correct:
            results["overall"]["step2_z4"]["correct"] += 1
        if s2_result["found"]:
            results["top_k_match"]["step2_z4"]["found"] += 1
        
        # === Step 3 (z6) ===
        s3_result = analyze_logit_lens_for_number(codi, z6, step3, args.top_k)
        s3_top1_correct = (s3_result["top_token"] == str(step3))
        
        results["overall"]["step3_z6"]["total"] += 1
        results["top_k_match"]["step3_z6"]["total"] += 1
        if s3_top1_correct:
            results["overall"]["step3_z6"]["correct"] += 1
        if s3_result["found"]:
            results["top_k_match"]["step3_z6"]["found"] += 1
            results["prob_stats"]["step3_found_probs"].append(s3_result["probability"])
        
        # Holdout for step3
        results["holdout"]["by_type"][holdout_type]["step3"]["t"] += 1
        if s3_top1_correct:
            results["holdout"]["by_type"][holdout_type]["step3"]["c"] += 1
        
        if novel_s3:
            results["holdout"]["novel_step3"]["total"] += 1
            if s3_top1_correct:
                results["holdout"]["novel_step3"]["correct"] += 1
        else:
            results["holdout"]["seen_step3"]["total"] += 1
            if s3_top1_correct:
                results["holdout"]["seen_step3"]["correct"] += 1
        
        # Rarity
        s3_rarity = categorize_rarity(step3, train_freqs.get("step3", {}))
        results["rarity"][f"step3_{s3_rarity}"]["total"] += 1
        if s3_top1_correct:
            results["rarity"][f"step3_{s3_rarity}"]["correct"] += 1
        
        # By operation
        results["by_operation"][op]["step3"]["t"] += 1
        if s3_top1_correct:
            results["by_operation"][op]["step3"]["c"] += 1
        
        # === Operation detection (z2) ===
        op_result = analyze_logit_lens_for_operation(codi, z2, op, top_k=20)
        
        results["overall"]["operation_z2"]["total"] += 1
        if op_result["found"]:
            results["overall"]["operation_z2"]["correct"] += 1
        
        results["by_operation"][op]["op_detect"]["t"] += 1
        if op_result["found"]:
            results["by_operation"][op]["op_detect"]["c"] += 1
        
        # === CODI correctness ===
        if codi_correct is not None:
            results["codi_analysis"]["codi_total"] += 1
            if codi_correct:
                results["codi_analysis"]["codi_correct"] += 1
                results["codi_analysis"]["ll_step3_when_codi_correct"]["total"] += 1
                if s3_top1_correct:
                    results["codi_analysis"]["ll_step3_when_codi_correct"]["correct"] += 1
            else:
                results["codi_analysis"]["ll_step3_when_codi_wrong"]["total"] += 1
                if s3_top1_correct:
                    results["codi_analysis"]["ll_step3_when_codi_wrong"]["correct"] += 1
    
    # === PRINT RESULTS ===
    print("\n" + "=" * 70)
    print("LOGIT LENS RESULTS")
    print("=" * 70)
    
    def pct(d):
        if d["total"] == 0:
            return "N/A"
        return f'{100 * d["correct"] / d["total"]:.1f}%'
    
    def pct2(d, k1="c", k2="t"):
        if d[k2] == 0:
            return "N/A"
        return f'{100 * d[k1] / d[k2]:.1f}%'
    
    def pct_found(d):
        if d["total"] == 0:
            return "N/A"
        return f'{100 * d["found"] / d["total"]:.1f}%'
    
    print("\n--- Overall Accuracy (Top-1 Match) ---")
    print(f"Step 1 (z2): {pct(results['overall']['step1_z2'])}")
    print(f"Step 2 (z4): {pct(results['overall']['step2_z4'])}")
    print(f"Step 3 (z6): {pct(results['overall']['step3_z6'])}")
    print(f"Operation:   {pct(results['overall']['operation_z2'])}")
    
    print(f"\n--- Top-{args.top_k} Match ---")
    print(f"Step 1 (z2): {pct_found(results['top_k_match']['step1_z2'])}")
    print(f"Step 2 (z4): {pct_found(results['top_k_match']['step2_z4'])}")
    print(f"Step 3 (z6): {pct_found(results['top_k_match']['step3_z6'])}")
    
    print("\n--- Holdout Analysis ---")
    for htype in ["seen", "value", "tuple"]:
        h = results["holdout"]["by_type"][htype]
        print(f"{htype.upper():6s}: step1={pct2(h['step1'])} step3={pct2(h['step3'])}")
    
    print("\n  By specific value novelty:")
    print(f"    Novel step1: {pct(results['holdout']['novel_step1'])} (n={results['holdout']['novel_step1']['total']})")
    print(f"    Seen step1:  {pct(results['holdout']['seen_step1'])} (n={results['holdout']['seen_step1']['total']})")
    print(f"    Novel step3: {pct(results['holdout']['novel_step3'])} (n={results['holdout']['novel_step3']['total']})")
    print(f"    Seen step3:  {pct(results['holdout']['seen_step3'])} (n={results['holdout']['seen_step3']['total']})")
    
    print("\n--- CODI Correctness ---")
    ca = results["codi_analysis"]
    if ca["codi_total"] > 0:
        print(f"CODI accuracy: {ca['codi_correct']}/{ca['codi_total']}")
        print(f"LL step3 when CODI correct: {pct(ca['ll_step3_when_codi_correct'])}")
        print(f"LL step3 when CODI wrong:   {pct(ca['ll_step3_when_codi_wrong'])}")
    
    print("\n--- By Operation ---")
    for op in ["add", "sub", "mul"]:
        o = results["by_operation"][op]
        print(f"  {op}: step1={pct2(o['step1'])} step3={pct2(o['step3'])} detect={pct2(o['op_detect'])}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
