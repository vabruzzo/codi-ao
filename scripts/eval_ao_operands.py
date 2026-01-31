#!/usr/bin/env python3
"""
Zero-shot evaluation: Can the AO extract operands from z2?

Tests whether the AO can generalize to questions it wasn't trained on:
1. First operand extraction
2. Second operand extraction  
3. Full calculation reconstruction
"""

import argparse
import json
import re
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activation_oracle import ActivationOracle, AOConfig, AOPrompt
from src.codi_wrapper import CODIWrapper


def ao_generate(ao: ActivationOracle, question: str, latent_vectors: list, max_new_tokens: int = 20) -> str:
    """Helper to generate AO response with proper prompt construction."""
    prompt = ao.create_prompt(
        question=question,
        activation_vectors=latent_vectors
    )
    return ao.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0)


def extract_number(response: str) -> int | None:
    """Extract a number from the response."""
    # Look for integers in the response
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return int(numbers[0])
    return None


def extract_calculation(response: str) -> tuple[int | None, str | None, int | None]:
    """
    Try to extract X, op, Y from a calculation response.
    Returns (first_num, op_symbol, second_num) or (None, None, None)
    """
    # Patterns to try:
    # "8 - 4", "8-4", "8 minus 4", "8 + 4", "8 plus 4", "8 * 4", "8 times 4", "8 x 4"
    
    patterns = [
        r'(\d+)\s*[\-−]\s*(\d+)',  # 8 - 4 or 8-4
        r'(\d+)\s*\+\s*(\d+)',      # 8 + 4
        r'(\d+)\s*[\*×x]\s*(\d+)',  # 8 * 4, 8 × 4, 8 x 4
        r'(\d+)\s+minus\s+(\d+)',   # 8 minus 4
        r'(\d+)\s+plus\s+(\d+)',    # 8 plus 4
        r'(\d+)\s+times\s+(\d+)',   # 8 times 4
        r'(\d+)\s+multiplied\s+by\s+(\d+)',  # 8 multiplied by 4
    ]
    
    op_map = {
        '-': 'sub', '−': 'sub', 'minus': 'sub',
        '+': 'add', 'plus': 'add',
        '*': 'mul', '×': 'mul', 'x': 'mul', 'times': 'mul', 'multiplied': 'mul'
    }
    
    response_lower = response.lower()
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            # Determine operation from the pattern
            full_match = match.group(0)
            op = None
            for key, val in op_map.items():
                if key in full_match:
                    op = val
                    break
            return first, op, second
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Zero-shot operand extraction evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ao_study")
    parser.add_argument("--problems", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/ao_operands.json")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Zero-Shot Operand Extraction Evaluation")
    print("=" * 60)
    
    # Load problems
    with open(args.problems) as f:
        all_problems = json.load(f)
    
    # Use last n_test as held-out (same as main eval)
    test_problems = all_problems[-args.n_test:]
    print(f"Testing on {len(test_problems)} held-out problems")
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\nLoading CODI model...")
    codi = CODIWrapper(device=device)
    
    print("Loading Activation Oracle...")
    config = AOConfig()
    ao = ActivationOracle(config)
    ao.load_checkpoint(args.checkpoint)
    ao.model.eval()
    
    # Questions to test
    questions = {
        "first_operand": "What was the first number in the calculation?",
        "second_operand": "What was the second number in the calculation?",
        "full_calculation": "What calculation was performed?",
    }
    
    results = {
        "first_operand": {"correct": 0, "total": 0, "predictions": []},
        "second_operand": {"correct": 0, "total": 0, "predictions": []},
        "full_calculation": {"correct": 0, "total": 0, "predictions": []},
    }
    
    print("\nRunning evaluations...")
    print("-" * 60)
    
    for i, problem in enumerate(test_problems):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(test_problems)}...")
        
        # Get ground truth
        X = problem["X"]
        Y = problem["Y"]
        op = problem["operation"]
        step1 = problem["step1"]
        
        # Collect latents
        latent_result = codi.collect_latents(problem["prompt"])
        z2 = latent_result["latent_vectors"][1]  # z2 = index 1
        
        # Test first operand
        response = ao_generate(ao, questions["first_operand"], [z2])
        pred = extract_number(response)
        correct = (pred == X)
        results["first_operand"]["total"] += 1
        if correct:
            results["first_operand"]["correct"] += 1
        results["first_operand"]["predictions"].append({
            "idx": i,
            "X": X, "Y": Y, "op": op, "step1": step1,
            "response": response,
            "pred": pred,
            "true": X,
            "correct": correct
        })
        
        # Test second operand
        response = ao_generate(ao, questions["second_operand"], [z2])
        pred = extract_number(response)
        correct = (pred == Y)
        results["second_operand"]["total"] += 1
        if correct:
            results["second_operand"]["correct"] += 1
        results["second_operand"]["predictions"].append({
            "idx": i,
            "X": X, "Y": Y, "op": op, "step1": step1,
            "response": response,
            "pred": pred,
            "true": Y,
            "correct": correct
        })
        
        # Test full calculation
        response = ao_generate(ao, questions["full_calculation"], [z2])
        pred_x, pred_op, pred_y = extract_calculation(response)
        # Correct if both operands and operation match
        correct = (pred_x == X and pred_y == Y and pred_op == op)
        results["full_calculation"]["total"] += 1
        if correct:
            results["full_calculation"]["correct"] += 1
        results["full_calculation"]["predictions"].append({
            "idx": i,
            "X": X, "Y": Y, "op": op, "step1": step1,
            "response": response,
            "pred_x": pred_x,
            "pred_op": pred_op,
            "pred_y": pred_y,
            "correct": correct
        })
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (Zero-Shot)")
    print("=" * 60)
    
    for task, data in results.items():
        acc = 100 * data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"{task}: {acc:.1f}% ({data['correct']}/{data['total']})")
    
    # Add summary
    results["summary"] = {
        task: {
            "accuracy": 100 * data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"]
        }
        for task, data in results.items() if task != "summary"
    }
    
    # Show some example responses
    print("\n" + "-" * 60)
    print("Sample responses:")
    print("-" * 60)
    for task in ["first_operand", "second_operand", "full_calculation"]:
        print(f"\n{task}:")
        for pred in results[task]["predictions"][:3]:
            print(f"  True: X={pred['X']}, Y={pred['Y']}, op={pred['op']}")
            print(f"  Response: {pred['response']!r}")
            print(f"  Correct: {pred['correct']}")
            print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
