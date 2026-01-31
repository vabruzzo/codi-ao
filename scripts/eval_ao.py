#!/usr/bin/env python3
"""
Evaluate Activation Oracle across all question types.

Reports metrics for:
- Extraction (numeric values)
- Operation detection (direct + binary)
- Magnitude classification
- Multi-latent comparison
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


# Question templates (same as training)
EXTRACTION_QUESTIONS = [
    "What is the intermediate calculation result?",
    "What value was computed?",
]

OPERATION_DIRECT = [
    "What operation was performed?",
    "What arithmetic operation was used?",
]

OPERATION_BINARY = {
    "addition": ["Is this an addition operation?"],
    "subtraction": ["Is this a subtraction operation?"],
    "multiplication": ["Is this a multiplication operation?"],
}

MAGNITUDE_QUESTIONS = {
    10: ["Is the result greater than 10?"],
    50: ["Is the result greater than 50?"],
    100: ["Is the result greater than 100?"],
}

COMPARISON_QUESTIONS = [
    "Which calculation step produced the larger result?",
]


def format_oracle_prompt(question: str, num_latents: int = 1) -> str:
    placeholders = " ?" * num_latents
    return f"Layer 50%:{placeholders} {question}"


def op_to_name(op: str) -> str:
    return {"add": "addition", "sub": "subtraction", "mul": "multiplication"}[op]


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
    
    ao = ActivationOracle.from_pretrained(config=config)
    ao.load_lora(checkpoint_dir)
    ao.eval_mode()
    
    return ao


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    # Remove common prefixes
    for prefix in ["the answer is ", "answer: ", "result: "]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    return answer.strip()


def extract_number(text: str) -> int | None:
    """Extract first number from text."""
    import re
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    return None


def evaluate_extraction(ao, problems, latents, position: int, step: int):
    """Evaluate numeric extraction accuracy."""
    results = {"correct": 0, "total": 0, "predictions": []}
    
    question = random.choice(EXTRACTION_QUESTIONS)
    prompt = format_oracle_prompt(question, 1)
    
    for i, (problem, latent_set) in enumerate(zip(problems, latents)):
        latent = latent_set[position]
        true_value = problem[f"step{step}"]
        
        response = ao.generate(
            prompt=prompt,
            activation_vectors=[latent],
            max_new_tokens=10,
        )
        
        pred_value = extract_number(response)
        is_correct = pred_value == true_value
        
        results["total"] += 1
        if is_correct:
            results["correct"] += 1
        
        results["predictions"].append({
            "true": true_value,
            "pred": pred_value,
            "response": response,
            "correct": is_correct,
        })
    
    results["accuracy"] = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    return results


def evaluate_operation_direct(ao, problems, latents, position: int):
    """Evaluate direct operation classification."""
    results = {"correct": 0, "total": 0, "per_op": defaultdict(lambda: {"correct": 0, "total": 0}), "predictions": []}
    
    question = random.choice(OPERATION_DIRECT)
    prompt = format_oracle_prompt(question, 1)
    
    for problem, latent_set in zip(problems, latents):
        latent = latent_set[position]
        true_op = op_to_name(problem["operation"])
        
        response = ao.generate(
            prompt=prompt,
            activation_vectors=[latent],
            max_new_tokens=10,
        )
        
        response_lower = normalize_answer(response)
        
        # Check which operation is in the response
        pred_op = None
        for op in ["addition", "subtraction", "multiplication"]:
            if op in response_lower or op[:3] in response_lower:
                pred_op = op
                break
        
        is_correct = pred_op == true_op
        
        results["total"] += 1
        results["per_op"][true_op]["total"] += 1
        if is_correct:
            results["correct"] += 1
            results["per_op"][true_op]["correct"] += 1
        
        results["predictions"].append({
            "true": true_op,
            "pred": pred_op,
            "response": response,
            "correct": is_correct,
        })
    
    results["accuracy"] = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    for op in results["per_op"]:
        op_data = results["per_op"][op]
        op_data["accuracy"] = op_data["correct"] / op_data["total"] * 100 if op_data["total"] > 0 else 0
    
    return results


def evaluate_operation_binary(ao, problems, latents, position: int):
    """Evaluate binary operation questions (is this X?)."""
    results = {"correct": 0, "total": 0, "per_op": defaultdict(lambda: {"correct": 0, "total": 0}), "predictions": []}
    
    for problem, latent_set in zip(problems, latents):
        latent = latent_set[position]
        true_op = op_to_name(problem["operation"])
        
        # Ask about each operation
        for check_op, questions in OPERATION_BINARY.items():
            question = random.choice(questions)
            prompt = format_oracle_prompt(question, 1)
            
            response = ao.generate(
                prompt=prompt,
                activation_vectors=[latent],
                max_new_tokens=5,
            )
            
            response_lower = normalize_answer(response)
            pred_yes = "yes" in response_lower
            true_yes = true_op == check_op
            
            is_correct = pred_yes == true_yes
            
            results["total"] += 1
            results["per_op"][check_op]["total"] += 1
            if is_correct:
                results["correct"] += 1
                results["per_op"][check_op]["correct"] += 1
            
            results["predictions"].append({
                "true_op": true_op,
                "check_op": check_op,
                "true_answer": "yes" if true_yes else "no",
                "pred_answer": "yes" if pred_yes else "no",
                "response": response,
                "correct": is_correct,
            })
    
    results["accuracy"] = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    for op in results["per_op"]:
        op_data = results["per_op"][op]
        op_data["accuracy"] = op_data["correct"] / op_data["total"] * 100 if op_data["total"] > 0 else 0
    
    return results


def evaluate_magnitude(ao, problems, latents, position: int, step: int):
    """Evaluate magnitude classification."""
    results = {"correct": 0, "total": 0, "per_threshold": {}, "predictions": []}
    
    for threshold, questions in MAGNITUDE_QUESTIONS.items():
        results["per_threshold"][threshold] = {"correct": 0, "total": 0}
        
        question = random.choice(questions)
        prompt = format_oracle_prompt(question, 1)
        
        for problem, latent_set in zip(problems, latents):
            latent = latent_set[position]
            true_value = problem[f"step{step}"]
            true_yes = true_value > threshold
            
            response = ao.generate(
                prompt=prompt,
                activation_vectors=[latent],
                max_new_tokens=5,
            )
            
            response_lower = normalize_answer(response)
            pred_yes = "yes" in response_lower
            
            is_correct = pred_yes == true_yes
            
            results["total"] += 1
            results["per_threshold"][threshold]["total"] += 1
            if is_correct:
                results["correct"] += 1
                results["per_threshold"][threshold]["correct"] += 1
            
            results["predictions"].append({
                "value": true_value,
                "threshold": threshold,
                "true": "yes" if true_yes else "no",
                "pred": "yes" if pred_yes else "no",
                "correct": is_correct,
            })
    
    results["accuracy"] = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    for t in results["per_threshold"]:
        t_data = results["per_threshold"][t]
        t_data["accuracy"] = t_data["correct"] / t_data["total"] * 100 if t_data["total"] > 0 else 0
    
    return results


def evaluate_comparison(ao, problems, latents):
    """Evaluate multi-latent comparison."""
    results = {"correct": 0, "total": 0, "predictions": []}
    
    question = random.choice(COMPARISON_QUESTIONS)
    prompt = format_oracle_prompt(question, 2)
    
    for problem, latent_set in zip(problems, latents):
        latent_z2 = latent_set[1]  # Step 1
        latent_z4 = latent_set[3]  # Step 2
        
        step1 = problem["step1"]
        step2 = problem["step2"]
        true_answer = "step 2" if step2 > step1 else "step 1"
        
        response = ao.generate(
            prompt=prompt,
            activation_vectors=[latent_z2, latent_z4],
            max_new_tokens=10,
        )
        
        response_lower = normalize_answer(response)
        
        if "step 2" in response_lower or "second" in response_lower:
            pred_answer = "step 2"
        elif "step 1" in response_lower or "first" in response_lower:
            pred_answer = "step 1"
        else:
            pred_answer = None
        
        is_correct = pred_answer == true_answer
        
        results["total"] += 1
        if is_correct:
            results["correct"] += 1
        
        results["predictions"].append({
            "step1": step1,
            "step2": step2,
            "true": true_answer,
            "pred": pred_answer,
            "response": response,
            "correct": is_correct,
        })
    
    results["accuracy"] = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ao_study")
    parser.add_argument("--problems", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--output", type=str, default="results/ao_evaluation.json")
    parser.add_argument("--n_test", type=int, default=200, help="Number of test problems")
    parser.add_argument("--seed", type=int, default=999)  # Different seed for test set
    args = parser.parse_args()
    
    print("=" * 60)
    print("Activation Oracle Evaluation")
    print("=" * 60)
    
    random.seed(args.seed)
    
    # Load problems
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    
    # Use last N problems as test set (not used in training)
    problems = data["problems"][-args.n_test:]
    print(f"Using {len(problems)} test problems")
    
    # Count operations
    op_counts = defaultdict(int)
    for p in problems:
        op_counts[p["operation"]] += 1
    print(f"Operations: {dict(op_counts)}")
    
    # Load CODI
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    # Collect latents
    print("Collecting latents...")
    all_latents = []
    for problem in tqdm(problems, desc="Collecting"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        all_latents.append(result.latent_vectors)
    
    # Load AO
    print(f"\nLoading Activation Oracle from {args.checkpoint}...")
    ao = load_ao_model(args.checkpoint)
    
    # Run evaluations
    results = {"config": {"n_test": len(problems), "seed": args.seed}}
    
    print("\n" + "=" * 60)
    print("Running evaluations...")
    print("=" * 60)
    
    # 1. Extraction - Step 1 (z2)
    print("\n--- Extraction (Step 1, z2) ---")
    results["extraction_step1"] = evaluate_extraction(ao, problems, all_latents, position=1, step=1)
    print(f"  Accuracy: {results['extraction_step1']['accuracy']:.1f}%")
    
    # 2. Extraction - Step 2 (z4)
    print("\n--- Extraction (Step 2, z4) ---")
    results["extraction_step2"] = evaluate_extraction(ao, problems, all_latents, position=3, step=2)
    print(f"  Accuracy: {results['extraction_step2']['accuracy']:.1f}%")
    
    # 3. Operation - Direct (z2)
    print("\n--- Operation Direct (z2) ---")
    results["operation_direct"] = evaluate_operation_direct(ao, problems, all_latents, position=1)
    print(f"  Accuracy: {results['operation_direct']['accuracy']:.1f}%")
    for op, data in results["operation_direct"]["per_op"].items():
        print(f"    {op}: {data['accuracy']:.1f}%")
    
    # 4. Operation - Binary (z2)
    print("\n--- Operation Binary (z2) ---")
    results["operation_binary"] = evaluate_operation_binary(ao, problems, all_latents, position=1)
    print(f"  Accuracy: {results['operation_binary']['accuracy']:.1f}%")
    
    # 5. Magnitude - Step 1 (z2)
    print("\n--- Magnitude (Step 1, z2) ---")
    results["magnitude_step1"] = evaluate_magnitude(ao, problems, all_latents, position=1, step=1)
    print(f"  Accuracy: {results['magnitude_step1']['accuracy']:.1f}%")
    for t, data in results["magnitude_step1"]["per_threshold"].items():
        print(f"    >{t}: {data['accuracy']:.1f}%")
    
    # 6. Comparison (multi-latent)
    print("\n--- Comparison (z2 + z4) ---")
    results["comparison"] = evaluate_comparison(ao, problems, all_latents)
    print(f"  Accuracy: {results['comparison']['accuracy']:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<25} {'Accuracy':>10}")
    print("-" * 35)
    print(f"{'Extraction Step 1':<25} {results['extraction_step1']['accuracy']:>9.1f}%")
    print(f"{'Extraction Step 2':<25} {results['extraction_step2']['accuracy']:>9.1f}%")
    print(f"{'Operation Direct':<25} {results['operation_direct']['accuracy']:>9.1f}%")
    print(f"{'Operation Binary':<25} {results['operation_binary']['accuracy']:>9.1f}%")
    print(f"{'Magnitude':<25} {results['magnitude_step1']['accuracy']:>9.1f}%")
    print(f"{'Comparison (multi)':<25} {results['comparison']['accuracy']:>9.1f}%")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert defaultdicts for JSON serialization
    def convert_defaultdict(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert_defaultdict))
    
    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
