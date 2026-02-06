#!/usr/bin/env python3
"""
Test that operation detection isn't just memorizing step1 values.

Key test: If step1=12 appeared in training as ADDITION, 
and we test on step1=12 from MULTIPLICATION, can AO still get the operation right?

This proves the AO reads operation info from latents, not just memorizing value→operation mappings.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codi_wrapper import CODIWrapper
from src.activation_oracle import ActivationOracle, AOConfig


def compute_step1(X, Y, op):
    if op == "add":
        return X + Y
    elif op == "sub":
        return X - Y if X > Y else None
    else:  # mul
        return X * Y


def load_codi():
    return CODIWrapper.from_pretrained()


def load_ao(checkpoint_path, injection_layer=1):
    config = AOConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        injection_layer=injection_layer,
    )
    ao = ActivationOracle.from_pretrained(config=config, lora_path=checkpoint_path)
    ao.eval_mode()
    return ao


def ao_generate(ao, latent_vectors, question, max_new_tokens=32):
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


def extract_operation(text):
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


# Templates
TEMPLATES = {
    "add": [
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team?",
    ],
    "sub": [
        "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team?",
    ],
    "mul": [
        "A team starts with {X} groups of {Y} members each. Then each current member recruits {Z} additional people. How many people are there now on the team?",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/operation_memorization_test.json")
    parser.add_argument("--injection_layer", type=int, default=1, help="Layer to inject activations (must match training)")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 70)
    print("Operation Memorization Test")
    print("=" * 70)
    print("\nGoal: Test if AO distinguishes operations for the SAME step1 value")
    print("If it can, operation detection is NOT just memorizing value→op mappings\n")
    
    # Find step1 values achievable by multiple operations
    # step1 values and which operations can produce them
    step1_to_ops = defaultdict(set)
    step1_to_examples = defaultdict(list)
    
    for X in range(2, 11):
        for Y in range(2, 11):
            for op in ["add", "sub", "mul"]:
                if op == "mul" and (X > 6 or Y > 6):
                    continue
                s1 = compute_step1(X, Y, op)
                if s1 is not None and s1 > 0:
                    step1_to_ops[s1].add(op)
                    step1_to_examples[s1].append((X, Y, op))
    
    # Find values with multiple operations
    multi_op_values = {v: ops for v, ops in step1_to_ops.items() if len(ops) >= 2}
    
    print(f"Step1 values achievable by 2+ operations: {len(multi_op_values)}")
    print(f"Examples:")
    for val in sorted(multi_op_values.keys())[:10]:
        ops = multi_op_values[val]
        examples = step1_to_examples[val][:3]
        print(f"  {val}: {ops} - e.g., {examples}")
    
    # Load models
    print("\nLoading CODI model...")
    codi = load_codi()
    
    print(f"Loading AO from {args.checkpoint}...")
    print(f"Injection layer: {args.injection_layer}")
    ao = load_ao(args.checkpoint, injection_layer=args.injection_layer)
    
    # Generate test cases: pairs with same step1, different operations
    test_pairs = []
    
    for step1_val, ops in multi_op_values.items():
        ops_list = list(ops)
        if len(ops_list) < 2:
            continue
        
        # Get examples for each operation
        examples_by_op = defaultdict(list)
        for X, Y, op in step1_to_examples[step1_val]:
            examples_by_op[op].append((X, Y))
        
        # Create pairs
        for i, op1 in enumerate(ops_list):
            for op2 in ops_list[i+1:]:
                if examples_by_op[op1] and examples_by_op[op2]:
                    X1, Y1 = random.choice(examples_by_op[op1])
                    X2, Y2 = random.choice(examples_by_op[op2])
                    test_pairs.append({
                        "step1": step1_val,
                        "case_a": {"X": X1, "Y": Y1, "op": op1},
                        "case_b": {"X": X2, "Y": Y2, "op": op2},
                    })
    
    random.shuffle(test_pairs)
    test_pairs = test_pairs[:args.n_test]
    
    print(f"\nGenerated {len(test_pairs)} test pairs (same step1, different operations)")
    
    # Evaluate
    op_q = "What mathematical operation was used in the first step?"
    
    results = {
        "total_pairs": 0,
        "both_correct": 0,  # AO gets BOTH operations right (strongest evidence)
        "case_a_correct": 0,
        "case_b_correct": 0,
        "by_op_pair": defaultdict(lambda: {"total": 0, "both_correct": 0}),
        "examples": [],
    }
    
    print("\nEvaluating...")
    
    for pair in tqdm(test_pairs, desc="Testing"):
        step1_val = pair["step1"]
        
        # Generate prompts
        Z = random.randint(2, 5)
        
        # Case A
        ca = pair["case_a"]
        prompt_a = TEMPLATES[ca["op"]][0].format(X=ca["X"], Y=ca["Y"], Z=Z)
        
        # Case B  
        cb = pair["case_b"]
        prompt_b = TEMPLATES[cb["op"]][0].format(X=cb["X"], Y=cb["Y"], Z=Z)
        
        # Collect latents
        result_a = codi.collect_latents(prompt_a, return_hidden_states=False)
        result_b = codi.collect_latents(prompt_b, return_hidden_states=False)
        
        if len(result_a.latent_vectors) < 6 or len(result_b.latent_vectors) < 6:
            continue
        
        z2_a = result_a.latent_vectors[1]
        z2_b = result_b.latent_vectors[1]
        
        # Ask AO about operation
        resp_a = ao_generate(ao, [z2_a], op_q)
        resp_b = ao_generate(ao, [z2_b], op_q)
        
        pred_a = extract_operation(resp_a)
        pred_b = extract_operation(resp_b)
        
        correct_a = (pred_a == ca["op"])
        correct_b = (pred_b == cb["op"])
        both_correct = correct_a and correct_b
        
        results["total_pairs"] += 1
        if correct_a:
            results["case_a_correct"] += 1
        if correct_b:
            results["case_b_correct"] += 1
        if both_correct:
            results["both_correct"] += 1
        
        # Track by operation pair
        op_pair = tuple(sorted([ca["op"], cb["op"]]))
        results["by_op_pair"][op_pair]["total"] += 1
        if both_correct:
            results["by_op_pair"][op_pair]["both_correct"] += 1
        
        # Store example
        if len(results["examples"]) < 10:
            results["examples"].append({
                "step1": step1_val,
                "case_a": {"op": ca["op"], "X": ca["X"], "Y": ca["Y"], 
                          "ao_response": resp_a, "ao_pred": pred_a, "correct": correct_a},
                "case_b": {"op": cb["op"], "X": cb["X"], "Y": cb["Y"],
                          "ao_response": resp_b, "ao_pred": pred_b, "correct": correct_b},
                "both_correct": both_correct,
            })
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    total = results["total_pairs"]
    if total > 0:
        print(f"\nTotal test pairs (same step1, different ops): {total}")
        print(f"\nBoth operations correct: {results['both_correct']}/{total} ({100*results['both_correct']/total:.1f}%)")
        print(f"  (This is the key metric - proves AO reads operation, not memorizing)")
        print(f"\nIndividual accuracy:")
        print(f"  Case A correct: {results['case_a_correct']}/{total} ({100*results['case_a_correct']/total:.1f}%)")
        print(f"  Case B correct: {results['case_b_correct']}/{total} ({100*results['case_b_correct']/total:.1f}%)")
        
        print(f"\nBy operation pair:")
        for op_pair, stats in sorted(results["by_op_pair"].items()):
            pct = 100 * stats["both_correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {op_pair[0]} vs {op_pair[1]}: {stats['both_correct']}/{stats['total']} ({pct:.1f}%)")
        
        print(f"\n--- Examples ---")
        for ex in results["examples"][:5]:
            print(f"\nstep1={ex['step1']}:")
            ca = ex["case_a"]
            cb = ex["case_b"]
            print(f"  {ca['X']} {ca['op']} {ca['Y']} → AO says '{ca['ao_pred']}' {'✓' if ca['correct'] else '✗'}")
            print(f"  {cb['X']} {cb['op']} {cb['Y']} → AO says '{cb['ao_pred']}' {'✓' if cb['correct'] else '✗'}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if total > 0:
        both_pct = 100 * results["both_correct"] / total
        if both_pct > 70:
            print(f"\n✓ STRONG EVIDENCE: AO correctly distinguishes operations {both_pct:.1f}% of the time")
            print("  even when the step1 VALUE is identical.")
            print("  This proves operation detection is NOT value memorization.")
        elif both_pct > 50:
            print(f"\n~ MODERATE EVIDENCE: AO distinguishes operations {both_pct:.1f}% of the time.")
            print("  Better than chance, suggests some genuine operation reading.")
        else:
            print(f"\n✗ WEAK: Only {both_pct:.1f}% correct on same-value different-operation pairs.")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert defaultdict for JSON
    results["by_op_pair"] = dict(results["by_op_pair"])
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
