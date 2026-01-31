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
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activation_oracle import ActivationOracle, AOConfig
from src.codi_wrapper import CODIWrapper


def load_codi_model(config_path="configs/default.yaml"):
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
    """Helper to generate AO response with proper prompt construction."""
    # Convert latent vectors to tensors if needed
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
    # Look for integers in the response
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return int(numbers[0])
    return None


def extract_calculation(response: str) -> tuple[int | None, str | None, int | None]:
    """
    Try to extract X, op, Y from a calculation response.
    Returns (first_num, op, second_num) where op is 'add'/'sub'/'mul' or None.
    """
    response_clean = response.strip()
    
    # Each pattern explicitly captures the operator for clarity
    # Format: (pattern, op_name)
    patterns_with_op = [
        # Symbol-based patterns
        (r'(\d+)\s*\+\s*(\d+)', 'add'),           # 8 + 4
        (r'(\d+)\s*[-−]\s*(\d+)', 'sub'),         # 8 - 4, 8 − 4 (ascii and unicode minus)
        (r'(\d+)\s*[*×]\s*(\d+)', 'mul'),         # 8 * 4, 8 × 4
        (r'(\d+)\s*x\s*(\d+)', 'mul'),            # 8 x 4 (letter x for multiplication)
        # Word-based patterns  
        (r'(\d+)\s+plus\s+(\d+)', 'add'),         # 8 plus 4
        (r'(\d+)\s+minus\s+(\d+)', 'sub'),        # 8 minus 4
        (r'(\d+)\s+times\s+(\d+)', 'mul'),        # 8 times 4
        (r'(\d+)\s+multiplied\s+by\s+(\d+)', 'mul'),  # 8 multiplied by 4
    ]
    
    response_lower = response_clean.lower()
    
    for pattern, op in patterns_with_op:
        match = re.search(pattern, response_lower)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            return first, op, second
    
    return None, None, None


def test_parsing():
    """Quick sanity check for parsing."""
    test_cases = [
        ("8 - 4", 8, "sub", 4),
        ("8-4", 8, "sub", 4),
        ("8 + 4", 8, "add", 4),
        ("4 * 6", 4, "mul", 6),
        ("4 × 6", 4, "mul", 6),
        ("4 x 6", 4, "mul", 6),
        ("10 minus 6", 10, "sub", 6),
        ("1 + 10", 1, "add", 10),
        ("The answer is 8 - 4", 8, "sub", 4),
        ("4", None, None, None),  # Just a number, no operation
        ("subtraction", None, None, None),  # No numbers
    ]
    
    print("Testing parsing...")
    all_passed = True
    for response, exp_x, exp_op, exp_y in test_cases:
        x, op, y = extract_calculation(response)
        passed = (x == exp_x and op == exp_op and y == exp_y)
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
            print(f"  {status} '{response}' -> got ({x}, {op}, {y}), expected ({exp_x}, {exp_op}, {exp_y})")
        else:
            print(f"  {status} '{response}' -> ({x}, {op}, {y})")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Zero-shot operand extraction evaluation")
    parser.add_argument("--test-parsing", action="store_true", help="Run parsing tests and exit")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ao_study")
    parser.add_argument("--problems", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/ao_operands.json")
    args = parser.parse_args()
    
    # Run parsing tests if requested
    if args.test_parsing:
        passed = test_parsing()
        sys.exit(0 if passed else 1)
    
    print("=" * 60)
    print("Operand Extraction Evaluation")
    print("=" * 60)
    
    # Load problems
    with open(args.problems) as f:
        data = json.load(f)
    all_problems = data["problems"]
    
    # Use last n_test as held-out (same as main eval)
    test_problems = all_problems[-args.n_test:]
    print(f"Testing on {len(test_problems)} held-out problems")
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    print("Loading Activation Oracle...")
    ao = load_ao_model(args.checkpoint)
    
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
        latent_result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        z2 = latent_result.latent_vectors[1]  # z2 = index 1
        
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
        # Strict: both operands and operation match exactly
        correct_strict = (pred_x == X and pred_y == Y and pred_op == op)
        # Semantic: operation matches and result is correct
        correct_semantic = False
        pred_result = None
        if pred_x is not None and pred_y is not None and pred_op is not None:
            if pred_op == op:  # Operation must match
                if pred_op == "add":
                    pred_result = pred_x + pred_y
                elif pred_op == "sub":
                    pred_result = pred_x - pred_y
                elif pred_op == "mul":
                    pred_result = pred_x * pred_y
                correct_semantic = (pred_result == step1)
        
        results["full_calculation"]["total"] += 1
        if correct_strict:
            results["full_calculation"]["correct"] += 1
        results["full_calculation"]["predictions"].append({
            "idx": i,
            "X": X, "Y": Y, "op": op, "step1": step1,
            "response": response,
            "pred_x": pred_x,
            "pred_op": pred_op,
            "pred_y": pred_y,
            "pred_result": pred_result,
            "correct_strict": correct_strict,
            "correct_semantic": correct_semantic,
        })
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for task, data in results.items():
        acc = 100 * data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"{task}: {acc:.1f}% ({data['correct']}/{data['total']})")
    
    # Calculate semantic accuracy for full_calculation
    semantic_correct = sum(1 for p in results["full_calculation"]["predictions"] if p["correct_semantic"])
    semantic_total = results["full_calculation"]["total"]
    semantic_acc = 100 * semantic_correct / semantic_total if semantic_total > 0 else 0
    print(f"\nfull_calculation (semantic): {semantic_acc:.1f}% ({semantic_correct}/{semantic_total})")
    print("  (semantic = correct operation AND pred_x op pred_y = actual result)")
    
    # Add summary
    results["summary"] = {
        task: {
            "accuracy": 100 * data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"]
        }
        for task, data in results.items() if task != "summary"
    }
    results["summary"]["full_calculation_semantic"] = {
        "accuracy": semantic_acc,
        "correct": semantic_correct,
        "total": semantic_total
    }
    
    # Show some example responses
    print("\n" + "-" * 60)
    print("Sample responses:")
    print("-" * 60)
    for task in ["first_operand", "second_operand"]:
        print(f"\n{task}:")
        for pred in results[task]["predictions"][:3]:
            print(f"  True: X={pred['X']}, Y={pred['Y']}, op={pred['op']}, result={pred['step1']}")
            print(f"  Response: {pred['response']!r}")
            print(f"  Correct: {pred['correct']}")
            print()
    
    # Special handling for full_calculation to show both metrics
    print(f"\nfull_calculation:")
    for pred in results["full_calculation"]["predictions"][:5]:
        op_symbol = {"add": "+", "sub": "-", "mul": "*"}.get(pred["op"], "?")
        print(f"  True: {pred['X']} {op_symbol} {pred['Y']} = {pred['step1']}")
        print(f"  Response: {pred['response']!r}")
        if pred["pred_x"] is not None:
            pred_op_sym = {"add": "+", "sub": "-", "mul": "*"}.get(pred["pred_op"], "?")
            print(f"  Parsed: {pred['pred_x']} {pred_op_sym} {pred['pred_y']} = {pred['pred_result']}")
        print(f"  Strict: {pred['correct_strict']}, Semantic: {pred['correct_semantic']}")
        print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
