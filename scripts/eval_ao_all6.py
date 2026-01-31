#!/usr/bin/env python3
"""
Combined evaluation for AO trained with all 6 latents.

Tests all tasks:
- Extraction (step 1 and step 2)
- Operation detection (direct and binary)
- Operand extraction (first, second, full calculation)
- Magnitude
- Comparison
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# QUESTION TEMPLATES
# =============================================================================

EXTRACTION_STEP1_Q = "What was calculated in the first step?"
EXTRACTION_STEP2_Q = "What was calculated in the second step?"
OPERATION_DIRECT_Q = "What operation was performed in step 1?"
OPERATION_BINARY_Q = {
    "addition": "Is the first operation addition?",
    "subtraction": "Is the first operation subtraction?",
    "multiplication": "Is the first operation multiplication?",
}
FIRST_OPERAND_Q = "What was the first number in the calculation?"
SECOND_OPERAND_Q = "What was the second number in the calculation?"
FULL_CALCULATION_Q = "What calculation was performed in step 1?"
MAGNITUDE_STEP1_Q = {
    10: "Is the step 1 result greater than 10?",
    50: "Is the step 1 result greater than 50?",
    100: "Is the step 1 result greater than 100?",
}
COMPARISON_Q = "Which calculation step produced the larger result?"


def load_codi_model(config_path="configs/default.yaml"):
    from src.codi_wrapper import CODIWrapper
    import yaml
    
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


def load_ao_model(checkpoint_dir: str):
    from src.activation_oracle import ActivationOracle, AOConfig
    
    config = AOConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_r=64,
        lora_alpha=128,
    )
    
    ao = ActivationOracle.from_pretrained(config=config, lora_path=checkpoint_dir)
    ao.eval_mode()
    return ao


def ao_generate(ao, question: str, latent_vectors: list, max_new_tokens: int = 20) -> str:
    """Generate AO response with all latent vectors."""
    vectors = []
    for v in latent_vectors:
        if isinstance(v, torch.Tensor):
            vectors.append(v)
        else:
            vectors.append(torch.tensor(v))
    
    prompt = ao.create_prompt(question=question, activation_vectors=vectors)
    return ao.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0)


def extract_number(response: str) -> int | None:
    """Extract a number from the response."""
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return int(numbers[0])
    return None


def extract_calculation(response: str) -> tuple[int | None, str | None, int | None]:
    """Extract X, op, Y from a calculation response."""
    response_clean = response.strip().lower()
    
    patterns_with_op = [
        (r'(\d+)\s*\+\s*(\d+)', 'add'),
        (r'(\d+)\s*[-−]\s*(\d+)', 'sub'),
        (r'(\d+)\s*[*×]\s*(\d+)', 'mul'),
        (r'(\d+)\s*x\s*(\d+)', 'mul'),
        (r'(\d+)\s+plus\s+(\d+)', 'add'),
        (r'(\d+)\s+minus\s+(\d+)', 'sub'),
        (r'(\d+)\s+times\s+(\d+)', 'mul'),
    ]
    
    for pattern, op in patterns_with_op:
        match = re.search(pattern, response_clean)
        if match:
            return int(match.group(1)), op, int(match.group(2))
    
    return None, None, None


def op_to_name(op: str) -> str:
    return {"add": "addition", "sub": "subtraction", "mul": "multiplication"}[op]


def main():
    parser = argparse.ArgumentParser(description="Combined AO evaluation (all 6 latents)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ao_all6")
    parser.add_argument("--problems", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/ao_all6_evaluation.json")
    parser.add_argument("--shuffle", action="store_true", 
                        help="Shuffle latent-to-problem mapping (sanity check - should crash accuracy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    
    print("=" * 60)
    if args.shuffle:
        print("Combined AO Evaluation (All 6 Latents) - SHUFFLED SANITY CHECK")
    else:
        print("Combined AO Evaluation (All 6 Latents)")
    print("=" * 60)
    
    if args.shuffle:
        print("\n*** SHUFFLE MODE: Latents will be mismatched with problems ***")
        print("*** Expected: accuracy should crash to near-random ***\n")
        random.seed(args.seed)
    
    # Load problems
    with open(args.problems) as f:
        data = json.load(f)
    all_problems = data["problems"]
    test_problems = all_problems[-args.n_test:]
    print(f"Testing on {len(test_problems)} held-out problems")
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    print("Loading Activation Oracle...")
    ao = load_ao_model(args.checkpoint)
    
    # First pass: collect all latents and track CODI accuracy
    print("\nCollecting latents for all test problems...")
    all_latents_list = []
    codi_correct_list = []
    codi_predictions = []
    valid_problems = []
    for problem in tqdm(test_problems, desc="Collecting latents"):
        ground_truth = str(problem["step2"])  # Final answer
        result = codi.collect_latents(
            problem["prompt"], 
            ground_truth_answer=ground_truth,
            return_hidden_states=False
        )
        if len(result.latent_vectors) >= 6:
            all_latents_list.append(result.latent_vectors[:6])
            codi_correct_list.append(result.is_correct)
            codi_predictions.append(result.predicted_answer)
            valid_problems.append(problem)
    
    codi_accuracy = sum(1 for c in codi_correct_list if c) / len(codi_correct_list) * 100
    print(f"Collected latents for {len(valid_problems)} problems")
    print(f"CODI accuracy on test set: {codi_accuracy:.1f}% ({sum(1 for c in codi_correct_list if c)}/{len(codi_correct_list)})")
    
    # Shuffle if requested
    if args.shuffle:
        print("Shuffling latent-to-problem mapping...")
        indices = list(range(len(all_latents_list)))
        random.shuffle(indices)
        all_latents_list = [all_latents_list[i] for i in indices]
        codi_correct_list = [codi_correct_list[i] for i in indices]
        codi_predictions = [codi_predictions[i] for i in indices]
        print(f"Shuffled with seed {args.seed}")
    
    # Results storage
    results = {
        "config": {
            "shuffle": args.shuffle, 
            "seed": args.seed if args.shuffle else None,
            "codi_accuracy": codi_accuracy,
        },
        "extraction_step1": {"correct": 0, "total": 0, "predictions": []},
        "extraction_step2": {"correct": 0, "total": 0, "predictions": []},
        "operation_direct": {"correct": 0, "total": 0, "predictions": [], "per_op": {}},
        "operation_binary": {"correct": 0, "total": 0, "predictions": []},
        "operand_first": {"correct": 0, "total": 0, "predictions": []},
        "operand_second": {"correct": 0, "total": 0, "predictions": []},
        "full_calculation_strict": {"correct": 0, "total": 0, "predictions": []},
        "full_calculation_semantic": {"correct": 0, "total": 0},
        "magnitude_step1": {"correct": 0, "total": 0, "predictions": []},
        "comparison": {"correct": 0, "total": 0, "predictions": []},
    }
    
    print("\nRunning evaluations...")
    print("-" * 60)
    
    for i, (problem, all_latents, codi_correct) in enumerate(tqdm(
            zip(valid_problems, all_latents_list, codi_correct_list), 
            desc="Evaluating", total=len(valid_problems))):
        # Ground truth (from problem, may not match latents if shuffled!)
        X = problem["X"]
        Y = problem["Y"]
        op = problem["operation"]
        op_name = op_to_name(op)
        step1 = problem["step1"]
        step2 = problem["step2"]
        op_symbol = {"add": "+", "sub": "-", "mul": "*"}[op]
        
        # --- Extraction Step 1 ---
        response = ao_generate(ao, EXTRACTION_STEP1_Q, all_latents)
        pred = extract_number(response)
        correct = (pred == step1)
        results["extraction_step1"]["total"] += 1
        if correct:
            results["extraction_step1"]["correct"] += 1
        results["extraction_step1"]["predictions"].append({
            "true": step1, "pred": pred, "response": response, "correct": correct,
            "codi_correct": codi_correct
        })
        
        # --- Extraction Step 2 ---
        response = ao_generate(ao, EXTRACTION_STEP2_Q, all_latents)
        pred = extract_number(response)
        correct = (pred == step2)
        results["extraction_step2"]["total"] += 1
        if correct:
            results["extraction_step2"]["correct"] += 1
        results["extraction_step2"]["predictions"].append({
            "true": step2, "pred": pred, "response": response, "correct": correct,
            "codi_correct": codi_correct
        })
        
        # --- Operation Direct ---
        response = ao_generate(ao, OPERATION_DIRECT_Q, all_latents)
        response_lower = response.lower()
        pred_op = None
        for check_op in ["addition", "subtraction", "multiplication"]:
            if check_op in response_lower:
                pred_op = check_op
                break
        correct = (pred_op == op_name)
        results["operation_direct"]["total"] += 1
        if correct:
            results["operation_direct"]["correct"] += 1
        # Per-op tracking
        if op_name not in results["operation_direct"]["per_op"]:
            results["operation_direct"]["per_op"][op_name] = {"correct": 0, "total": 0}
        results["operation_direct"]["per_op"][op_name]["total"] += 1
        if correct:
            results["operation_direct"]["per_op"][op_name]["correct"] += 1
        results["operation_direct"]["predictions"].append({
            "true": op_name, "pred": pred_op, "response": response, "correct": correct,
            "codi_correct": codi_correct
        })
        
        # --- Operation Binary ---
        for check_op, question in OPERATION_BINARY_Q.items():
            response = ao_generate(ao, question, all_latents, max_new_tokens=5)
            response_lower = response.lower()
            pred_yes = "yes" in response_lower and "no" not in response_lower[:response_lower.find("yes")] if "yes" in response_lower else False
            true_yes = (op_name == check_op)
            correct = (pred_yes == true_yes)
            results["operation_binary"]["total"] += 1
            if correct:
                results["operation_binary"]["correct"] += 1
        
        # --- Operand First (X) ---
        response = ao_generate(ao, FIRST_OPERAND_Q, all_latents)
        pred = extract_number(response)
        correct = (pred == X)
        results["operand_first"]["total"] += 1
        if correct:
            results["operand_first"]["correct"] += 1
        results["operand_first"]["predictions"].append({
            "true": X, "pred": pred, "response": response, "correct": correct,
            "codi_correct": codi_correct
        })
        
        # --- Operand Second (Y) ---
        response = ao_generate(ao, SECOND_OPERAND_Q, all_latents)
        pred = extract_number(response)
        correct = (pred == Y)
        results["operand_second"]["total"] += 1
        if correct:
            results["operand_second"]["correct"] += 1
        results["operand_second"]["predictions"].append({
            "true": Y, "pred": pred, "response": response, "correct": correct,
            "codi_correct": codi_correct
        })
        
        # --- Full Calculation ---
        response = ao_generate(ao, FULL_CALCULATION_Q, all_latents)
        pred_x, pred_op, pred_y = extract_calculation(response)
        
        # Strict: exact match
        correct_strict = (pred_x == X and pred_y == Y and pred_op == op)
        results["full_calculation_strict"]["total"] += 1
        if correct_strict:
            results["full_calculation_strict"]["correct"] += 1
        
        # Semantic: correct operation and result
        correct_semantic = False
        pred_result = None
        if pred_x is not None and pred_y is not None and pred_op is not None:
            if pred_op == op:
                if pred_op == "add":
                    pred_result = pred_x + pred_y
                elif pred_op == "sub":
                    pred_result = pred_x - pred_y
                elif pred_op == "mul":
                    pred_result = pred_x * pred_y
                correct_semantic = (pred_result == step1)
        results["full_calculation_semantic"]["total"] += 1
        if correct_semantic:
            results["full_calculation_semantic"]["correct"] += 1
        
        results["full_calculation_strict"]["predictions"].append({
            "true": f"{X} {op_symbol} {Y} = {step1}",
            "response": response,
            "pred_x": pred_x, "pred_op": pred_op, "pred_y": pred_y,
            "pred_result": pred_result,
            "correct_strict": correct_strict,
            "correct_semantic": correct_semantic,
            "codi_correct": codi_correct,
        })
        
        # --- Magnitude Step 1 ---
        for threshold, question in MAGNITUDE_STEP1_Q.items():
            response = ao_generate(ao, question, all_latents, max_new_tokens=5)
            response_lower = response.lower()
            pred_yes = "yes" in response_lower
            true_yes = step1 > threshold
            correct = (pred_yes == true_yes)
            results["magnitude_step1"]["total"] += 1
            if correct:
                results["magnitude_step1"]["correct"] += 1
        
        # --- Comparison ---
        response = ao_generate(ao, COMPARISON_Q, all_latents, max_new_tokens=10)
        response_lower = response.lower()
        pred_larger = "step 2" if "2" in response_lower or "second" in response_lower else "step 1"
        true_larger = "step 2" if step2 > step1 else "step 1"
        correct = (pred_larger == true_larger)
        results["comparison"]["total"] += 1
        if correct:
            results["comparison"]["correct"] += 1
        results["comparison"]["predictions"].append({
            "step1": step1, "step2": step2, "response": response,
            "pred": pred_larger, "true": true_larger, "correct": correct
        })
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for task in ["extraction_step1", "extraction_step2", "operation_direct", 
                 "operation_binary", "operand_first", "operand_second",
                 "full_calculation_strict", "full_calculation_semantic",
                 "magnitude_step1", "comparison"]:
        data = results[task]
        acc = 100 * data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"{task}: {acc:.1f}% ({data['correct']}/{data['total']})")
    
    # Per-operation breakdown for operation_direct
    print("\nOperation Direct per-class:")
    for op_name, data in results["operation_direct"]["per_op"].items():
        acc = 100 * data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"  {op_name}: {acc:.1f}% ({data['correct']}/{data['total']})")
    
    # Sample full calculation predictions
    print("\n" + "-" * 60)
    print("Sample full_calculation predictions:")
    for pred in results["full_calculation_strict"]["predictions"][:5]:
        print(f"  True: {pred['true']}")
        print(f"  Response: {pred['response']!r}")
        if pred['pred_x'] is not None:
            op_sym = {"add": "+", "sub": "-", "mul": "*"}.get(pred['pred_op'], "?")
            print(f"  Parsed: {pred['pred_x']} {op_sym} {pred['pred_y']} = {pred['pred_result']}")
        print(f"  Strict: {pred['correct_strict']}, Semantic: {pred['correct_semantic']}")
        print()
    
    # Analysis by CODI correctness
    print("\n" + "=" * 60)
    print("AO ACCURACY BY CODI CORRECTNESS")
    print("=" * 60)
    
    # Key tasks to analyze
    tasks_to_analyze = ["extraction_step1", "extraction_step2", "operation_direct", 
                        "operand_first", "operand_second", "full_calculation_strict"]
    
    for task in tasks_to_analyze:
        preds = results[task]["predictions"]
        if not preds or "codi_correct" not in preds[0]:
            continue
        
        # Split by CODI correctness
        codi_right = [p for p in preds if p.get("codi_correct")]
        codi_wrong = [p for p in preds if not p.get("codi_correct")]
        
        # Handle full_calculation_strict differently (uses correct_strict)
        if task == "full_calculation_strict":
            acc_right = 100 * sum(1 for p in codi_right if p["correct_strict"]) / len(codi_right) if codi_right else 0
            acc_wrong = 100 * sum(1 for p in codi_wrong if p["correct_strict"]) / len(codi_wrong) if codi_wrong else 0
        else:
            acc_right = 100 * sum(1 for p in codi_right if p["correct"]) / len(codi_right) if codi_right else 0
            acc_wrong = 100 * sum(1 for p in codi_wrong if p["correct"]) / len(codi_wrong) if codi_wrong else 0
        
        print(f"{task}:")
        print(f"  CODI correct (n={len(codi_right)}): {acc_right:.1f}%")
        print(f"  CODI wrong   (n={len(codi_wrong)}): {acc_wrong:.1f}%")
        diff = acc_right - acc_wrong
        print(f"  Difference: {diff:+.1f}%")
        print()
    
    # Store CODI analysis in results
    results["codi_analysis"] = {
        "codi_accuracy": codi_accuracy,
        "codi_correct_count": sum(1 for c in codi_correct_list if c),
        "codi_total": len(codi_correct_list),
    }
    
    # Summary
    results["summary"] = {
        task: {
            "accuracy": 100 * data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"]
        }
        for task, data in results.items() 
        if task not in ["summary", "config"] and isinstance(data, dict) and "total" in data
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
