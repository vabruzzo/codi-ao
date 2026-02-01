#!/usr/bin/env python3
"""
Comprehensive evaluation of Activation Oracle with holdout analysis.

Tracks:
1. Overall accuracy for step1, step2, step3, operation
2. Single vs multi-latent performance
3. Novel vs seen value performance (memorization test)
4. Novel tuple vs seen tuple performance
5. CODI correctness correlation
6. Rarity analysis (frequency in training)
7. Operation balance
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
from src.activation_oracle import ActivationOracle, AOConfig


def load_codi():
    """Load CODI model."""
    return CODIWrapper.from_pretrained()


def load_ao(checkpoint_path):
    """Load trained Activation Oracle."""
    config = AOConfig(model_name="meta-llama/Llama-3.2-1B-Instruct")
    ao = ActivationOracle.from_pretrained(config=config, lora_path=checkpoint_path)
    ao.eval_mode()
    return ao


def extract_number(text):
    """Extract first number from text."""
    if text is None:
        return None
    numbers = re.findall(r'-?\d+', str(text))
    return int(numbers[0]) if numbers else None


def extract_operation(text):
    """Extract operation from text."""
    if text is None:
        return None
    text_lower = text.lower()
    if "addition" in text_lower or "add" in text_lower:
        return "add"
    elif "subtraction" in text_lower or "subtract" in text_lower:
        return "sub"
    elif "multiplication" in text_lower or "multipl" in text_lower:
        return "mul"
    return None


def ao_generate(ao, latent_vectors, question, max_new_tokens=32):
    """Generate response from AO."""
    vectors = []
    for v in latent_vectors:
        if isinstance(v, torch.Tensor):
            vectors.append(v)
        elif isinstance(v, list):
            vectors.append(torch.tensor(v, dtype=torch.float32))
        else:
            vectors.append(v)
    
    prompt = ao.create_prompt(question=question, activation_vectors=vectors)
    response = ao.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0)
    return response.strip()


def categorize_rarity(value, freq_dict, rare_threshold=10, common_threshold=30):
    """Categorize value rarity based on training frequency."""
    count = freq_dict.get(str(value), freq_dict.get(value, 0))
    if count <= rare_threshold:
        return "rare"
    elif count >= common_threshold:
        return "common"
    return "medium"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--problems", type=str, default="data/problems_holdout.json")
    parser.add_argument("--output", type=str, default="results/ao_holdout_eval.json")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle latents (sanity check)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Comprehensive AO Evaluation with Holdout Analysis")
    print("=" * 70)
    
    if args.shuffle:
        print("\n*** SHUFFLE MODE: Latents mismatched (sanity check) ***\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    
    test_problems = data["test_problems"]
    holdout_info = data["holdout"]
    train_freqs = data.get("train_value_frequencies", {})
    
    print(f"Test problems: {len(test_problems)}")
    print(f"Held-out step1 values: {holdout_info['step1_values']}")
    print(f"Held-out tuples: {len(holdout_info['tuples'])}")
    
    # Load models
    print("\nLoading CODI model...")
    codi = load_codi()
    
    print(f"Loading AO from {args.checkpoint}...")
    ao = load_ao(args.checkpoint)
    
    # Initialize comprehensive results
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "problems": args.problems,
            "shuffle": args.shuffle,
            "n_test": len(test_problems),
        },
        
        # === OVERALL ACCURACY ===
        "overall": {
            "step1_single": {"correct": 0, "total": 0},
            "step1_multi": {"correct": 0, "total": 0},
            "step2_single": {"correct": 0, "total": 0},
            "step2_multi": {"correct": 0, "total": 0},
            "step3_single_z5": {"correct": 0, "total": 0},
            "step3_single_z6": {"correct": 0, "total": 0},
            "step3_multi": {"correct": 0, "total": 0},
            "operation_single": {"correct": 0, "total": 0},
            "operation_multi": {"correct": 0, "total": 0},
            "comparison": {"correct": 0, "total": 0},
        },
        
        # === HOLDOUT ANALYSIS (Memorization Test) ===
        "holdout": {
            # By holdout type
            "by_type": {
                "tuple": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op": {"c": 0, "t": 0}},
                "value": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op": {"c": 0, "t": 0}},
                "seen": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op": {"c": 0, "t": 0}},
            },
            # By specific value novelty
            "novel_step1": {"correct": 0, "total": 0},
            "seen_step1": {"correct": 0, "total": 0},
            "novel_step3": {"correct": 0, "total": 0},
            "seen_step3": {"correct": 0, "total": 0},
        },
        
        # === CODI CORRECTNESS ===
        "codi_analysis": {
            "codi_correct": 0,
            "codi_total": 0,
            "ao_step3_when_codi_correct": {"correct": 0, "total": 0},
            "ao_step3_when_codi_wrong": {"correct": 0, "total": 0},
            "ao_matches_codi_wrong": {"matches": 0, "total": 0},
            "ao_step1_when_codi_correct": {"correct": 0, "total": 0},
            "ao_step1_when_codi_wrong": {"correct": 0, "total": 0},
        },
        
        # === RARITY ANALYSIS ===
        "rarity": {
            "step1_rare": {"correct": 0, "total": 0},
            "step1_medium": {"correct": 0, "total": 0},
            "step1_common": {"correct": 0, "total": 0},
            "step3_rare": {"correct": 0, "total": 0},
            "step3_medium": {"correct": 0, "total": 0},
            "step3_common": {"correct": 0, "total": 0},
        },
        
        # === OPERATION BREAKDOWN ===
        "by_operation": {
            "add": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op_detect": {"c": 0, "t": 0}},
            "sub": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op_detect": {"c": 0, "t": 0}},
            "mul": {"step1": {"c": 0, "t": 0}, "step3": {"c": 0, "t": 0}, "op_detect": {"c": 0, "t": 0}},
        },
        
        # === EXAMPLES ===
        "examples": {
            "novel_correct": [],
            "novel_wrong": [],
            "codi_wrong_ao_correct": [],
            "codi_wrong_ao_wrong": [],
        },
    }
    
    # Questions
    step1_q = "What was calculated in the first step?"
    step2_q = "What was calculated in the second step?"
    step3_q = "What is the final answer?"
    op_q = "What mathematical operation was used in the first step?"
    compare_q = "Which is larger: the result of step 1 or step 2?"
    
    # Collect latents
    print("\nCollecting latents...")
    all_latents = []
    codi_outputs = []
    codi_correct_list = []
    
    for problem in tqdm(test_problems, desc="Collecting"):
        ground_truth = str(problem["step3"])
        result = codi.collect_latents(
            problem["prompt"],
            ground_truth_answer=ground_truth,
            return_hidden_states=False
        )
        
        if len(result.latent_vectors) >= 6:
            all_latents.append(result.latent_vectors[:6])
            codi_outputs.append(result.predicted_answer)
            codi_correct_list.append(result.is_correct)
        else:
            all_latents.append(None)
            codi_outputs.append(None)
            codi_correct_list.append(None)
    
    # Shuffle if requested
    if args.shuffle:
        import random
        random.seed(42)
        valid_indices = [i for i, l in enumerate(all_latents) if l is not None]
        shuffled = valid_indices.copy()
        random.shuffle(shuffled)
        shuffle_map = dict(zip(valid_indices, shuffled))
    
    # Evaluate
    print("\nEvaluating...")
    
    for i, problem in enumerate(tqdm(test_problems, desc="Evaluating")):
        latents = all_latents[i]
        if latents is None:
            continue
        
        if args.shuffle and i in shuffle_map:
            latents = all_latents[shuffle_map[i]]
        
        # Problem info
        step1 = problem["step1"]
        step2 = problem["step2"]
        step3 = problem["step3"]
        op = problem["operation"]
        holdout_type = problem.get("holdout_type", "unknown")
        novel_s1 = problem.get("novel_step1", False)
        novel_s3 = problem.get("novel_step3", False)
        
        # CODI info
        codi_out = codi_outputs[i]
        codi_correct = codi_correct_list[i]
        codi_pred_num = extract_number(codi_out)
        
        z2 = latents[1]
        z4 = latents[3]
        z5 = latents[4]
        z6 = latents[5]
        
        # === STEP 1 ===
        resp_s1_single = ao_generate(ao, [z2], step1_q)
        pred_s1_single = extract_number(resp_s1_single)
        correct_s1_single = (pred_s1_single == step1)
        
        resp_s1_multi = ao_generate(ao, latents, step1_q)
        pred_s1_multi = extract_number(resp_s1_multi)
        correct_s1_multi = (pred_s1_multi == step1)
        
        results["overall"]["step1_single"]["total"] += 1
        results["overall"]["step1_multi"]["total"] += 1
        if correct_s1_single:
            results["overall"]["step1_single"]["correct"] += 1
        if correct_s1_multi:
            results["overall"]["step1_multi"]["correct"] += 1
        
        # Holdout tracking for step1
        results["holdout"]["by_type"][holdout_type]["step1"]["t"] += 1
        if correct_s1_single:
            results["holdout"]["by_type"][holdout_type]["step1"]["c"] += 1
        
        if novel_s1:
            results["holdout"]["novel_step1"]["total"] += 1
            if correct_s1_single:
                results["holdout"]["novel_step1"]["correct"] += 1
        else:
            results["holdout"]["seen_step1"]["total"] += 1
            if correct_s1_single:
                results["holdout"]["seen_step1"]["correct"] += 1
        
        # Rarity for step1
        s1_rarity = categorize_rarity(step1, train_freqs.get("step1", {}))
        results["rarity"][f"step1_{s1_rarity}"]["total"] += 1
        if correct_s1_single:
            results["rarity"][f"step1_{s1_rarity}"]["correct"] += 1
        
        # By operation
        results["by_operation"][op]["step1"]["t"] += 1
        if correct_s1_single:
            results["by_operation"][op]["step1"]["c"] += 1
        
        # === STEP 2 ===
        resp_s2_single = ao_generate(ao, [z4], step2_q)
        pred_s2_single = extract_number(resp_s2_single)
        correct_s2_single = (pred_s2_single == step2)
        
        resp_s2_multi = ao_generate(ao, latents, step2_q)
        pred_s2_multi = extract_number(resp_s2_multi)
        correct_s2_multi = (pred_s2_multi == step2)
        
        results["overall"]["step2_single"]["total"] += 1
        results["overall"]["step2_multi"]["total"] += 1
        if correct_s2_single:
            results["overall"]["step2_single"]["correct"] += 1
        if correct_s2_multi:
            results["overall"]["step2_multi"]["correct"] += 1
        
        # === STEP 3 ===
        resp_s3_z5 = ao_generate(ao, [z5], step3_q)
        pred_s3_z5 = extract_number(resp_s3_z5)
        correct_s3_z5 = (pred_s3_z5 == step3)
        
        resp_s3_z6 = ao_generate(ao, [z6], step3_q)
        pred_s3_z6 = extract_number(resp_s3_z6)
        correct_s3_z6 = (pred_s3_z6 == step3)
        
        resp_s3_multi = ao_generate(ao, latents, step3_q)
        pred_s3_multi = extract_number(resp_s3_multi)
        correct_s3_multi = (pred_s3_multi == step3)
        
        results["overall"]["step3_single_z5"]["total"] += 1
        results["overall"]["step3_single_z6"]["total"] += 1
        results["overall"]["step3_multi"]["total"] += 1
        if correct_s3_z5:
            results["overall"]["step3_single_z5"]["correct"] += 1
        if correct_s3_z6:
            results["overall"]["step3_single_z6"]["correct"] += 1
        if correct_s3_multi:
            results["overall"]["step3_multi"]["correct"] += 1
        
        # Holdout tracking for step3
        results["holdout"]["by_type"][holdout_type]["step3"]["t"] += 1
        if correct_s3_multi:
            results["holdout"]["by_type"][holdout_type]["step3"]["c"] += 1
        
        if novel_s3:
            results["holdout"]["novel_step3"]["total"] += 1
            if correct_s3_multi:
                results["holdout"]["novel_step3"]["correct"] += 1
        else:
            results["holdout"]["seen_step3"]["total"] += 1
            if correct_s3_multi:
                results["holdout"]["seen_step3"]["correct"] += 1
        
        # Rarity for step3
        s3_rarity = categorize_rarity(step3, train_freqs.get("step3", {}))
        results["rarity"][f"step3_{s3_rarity}"]["total"] += 1
        if correct_s3_multi:
            results["rarity"][f"step3_{s3_rarity}"]["correct"] += 1
        
        # By operation
        results["by_operation"][op]["step3"]["t"] += 1
        if correct_s3_multi:
            results["by_operation"][op]["step3"]["c"] += 1
        
        # === CODI CORRECTNESS ANALYSIS ===
        if codi_correct is not None:
            results["codi_analysis"]["codi_total"] += 1
            if codi_correct:
                results["codi_analysis"]["codi_correct"] += 1
                results["codi_analysis"]["ao_step3_when_codi_correct"]["total"] += 1
                results["codi_analysis"]["ao_step1_when_codi_correct"]["total"] += 1
                if correct_s3_multi:
                    results["codi_analysis"]["ao_step3_when_codi_correct"]["correct"] += 1
                if correct_s1_single:
                    results["codi_analysis"]["ao_step1_when_codi_correct"]["correct"] += 1
            else:
                results["codi_analysis"]["ao_step3_when_codi_wrong"]["total"] += 1
                results["codi_analysis"]["ao_step1_when_codi_wrong"]["total"] += 1
                results["codi_analysis"]["ao_matches_codi_wrong"]["total"] += 1
                
                if correct_s3_multi:
                    results["codi_analysis"]["ao_step3_when_codi_wrong"]["correct"] += 1
                if correct_s1_single:
                    results["codi_analysis"]["ao_step1_when_codi_wrong"]["correct"] += 1
                if pred_s3_multi == codi_pred_num:
                    results["codi_analysis"]["ao_matches_codi_wrong"]["matches"] += 1
                
                # Store example
                if len(results["examples"]["codi_wrong_ao_correct"]) < 5 and correct_s3_multi:
                    results["examples"]["codi_wrong_ao_correct"].append({
                        "problem": problem["prompt"][:100],
                        "true_step3": step3,
                        "codi_output": codi_out,
                        "ao_output": resp_s3_multi,
                    })
                elif len(results["examples"]["codi_wrong_ao_wrong"]) < 5 and not correct_s3_multi:
                    results["examples"]["codi_wrong_ao_wrong"].append({
                        "problem": problem["prompt"][:100],
                        "true_step3": step3,
                        "codi_output": codi_out,
                        "ao_output": resp_s3_multi,
                    })
        
        # === OPERATION DETECTION ===
        resp_op_single = ao_generate(ao, [z2], op_q)
        pred_op_single = extract_operation(resp_op_single)
        correct_op_single = (pred_op_single == op)
        
        resp_op_multi = ao_generate(ao, latents, op_q)
        pred_op_multi = extract_operation(resp_op_multi)
        correct_op_multi = (pred_op_multi == op)
        
        results["overall"]["operation_single"]["total"] += 1
        results["overall"]["operation_multi"]["total"] += 1
        if correct_op_single:
            results["overall"]["operation_single"]["correct"] += 1
        if correct_op_multi:
            results["overall"]["operation_multi"]["correct"] += 1
        
        results["holdout"]["by_type"][holdout_type]["op"]["t"] += 1
        if correct_op_single:
            results["holdout"]["by_type"][holdout_type]["op"]["c"] += 1
        
        results["by_operation"][op]["op_detect"]["t"] += 1
        if correct_op_single:
            results["by_operation"][op]["op_detect"]["c"] += 1
        
        # === COMPARISON ===
        resp_cmp = ao_generate(ao, latents, compare_q)
        expected_cmp = "step 2" if step2 > step1 else "step 1"
        correct_cmp = expected_cmp in resp_cmp.lower()
        
        results["overall"]["comparison"]["total"] += 1
        if correct_cmp:
            results["overall"]["comparison"]["correct"] += 1
        
        # === EXAMPLES for novel values ===
        if novel_s1 or novel_s3:
            if correct_s3_multi and len(results["examples"]["novel_correct"]) < 5:
                results["examples"]["novel_correct"].append({
                    "step1": step1, "step3": step3, "novel_s1": novel_s1, "novel_s3": novel_s3,
                    "ao_step3": resp_s3_multi, "correct": True,
                })
            elif not correct_s3_multi and len(results["examples"]["novel_wrong"]) < 5:
                results["examples"]["novel_wrong"].append({
                    "step1": step1, "step3": step3, "novel_s1": novel_s1, "novel_s3": novel_s3,
                    "ao_step3": resp_s3_multi, "correct": False,
                })
    
    # === PRINT RESULTS ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    def pct(d):
        if d["total"] == 0:
            return "N/A"
        return f'{100 * d["correct"] / d["total"]:.1f}%'
    
    def pct2(d, k1="c", k2="t"):
        if d[k2] == 0:
            return "N/A"
        return f'{100 * d[k1] / d[k2]:.1f}%'
    
    print("\n--- Overall Accuracy ---")
    print(f"Step 1 (z2 only):  {pct(results['overall']['step1_single'])}")
    print(f"Step 1 (all 6):    {pct(results['overall']['step1_multi'])}")
    print(f"Step 2 (z4 only):  {pct(results['overall']['step2_single'])}")
    print(f"Step 2 (all 6):    {pct(results['overall']['step2_multi'])}")
    print(f"Step 3 (z5 only):  {pct(results['overall']['step3_single_z5'])}")
    print(f"Step 3 (z6 only):  {pct(results['overall']['step3_single_z6'])}")
    print(f"Step 3 (all 6):    {pct(results['overall']['step3_multi'])}")
    print(f"Operation (z2):    {pct(results['overall']['operation_single'])}")
    print(f"Operation (all):   {pct(results['overall']['operation_multi'])}")
    print(f"Comparison:        {pct(results['overall']['comparison'])}")
    
    print("\n--- Holdout Analysis (Memorization Test) ---")
    for htype in ["seen", "value", "tuple"]:
        h = results["holdout"]["by_type"][htype]
        print(f"{htype.upper():6s}: step1={pct2(h['step1'])} step3={pct2(h['step3'])} op={pct2(h['op'])}")
    
    print("\n  By specific value novelty:")
    print(f"    Novel step1: {pct(results['holdout']['novel_step1'])} (n={results['holdout']['novel_step1']['total']})")
    print(f"    Seen step1:  {pct(results['holdout']['seen_step1'])} (n={results['holdout']['seen_step1']['total']})")
    print(f"    Novel step3: {pct(results['holdout']['novel_step3'])} (n={results['holdout']['novel_step3']['total']})")
    print(f"    Seen step3:  {pct(results['holdout']['seen_step3'])} (n={results['holdout']['seen_step3']['total']})")
    
    print("\n--- CODI Correctness Analysis ---")
    ca = results["codi_analysis"]
    if ca["codi_total"] > 0:
        print(f"CODI accuracy: {ca['codi_correct']}/{ca['codi_total']} ({100*ca['codi_correct']/ca['codi_total']:.1f}%)")
        print(f"AO step3 when CODI correct: {pct(ca['ao_step3_when_codi_correct'])}")
        print(f"AO step3 when CODI wrong:   {pct(ca['ao_step3_when_codi_wrong'])}")
        print(f"AO matches CODI's wrong answer: {ca['ao_matches_codi_wrong']['matches']}/{ca['ao_matches_codi_wrong']['total']}")
    
    print("\n--- Rarity Analysis ---")
    for step in ["step1", "step3"]:
        print(f"  {step}:")
        for rarity in ["rare", "medium", "common"]:
            r = results["rarity"][f"{step}_{rarity}"]
            print(f"    {rarity:7s}: {pct(r)} (n={r['total']})")
    
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
