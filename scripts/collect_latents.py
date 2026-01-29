#!/usr/bin/env python3
"""
Script to collect latent vectors from CODI and validate the MVP.

This script:
1. Loads CODI model
2. Runs on test prompts to collect latent vectors
3. Verifies that z3/z5 store intermediate results (per LessWrong findings)
4. Reports logit lens accuracy as the MVP baseline

Usage:
    python scripts/collect_latents.py --n_samples 100 --verbose
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm


def extract_number(s: Optional[str]) -> Optional[str]:
    """Extract first number from string for comparison."""
    if s is None:
        return None
    match = re.search(r'-?\d+\.?\d*', s.strip())
    return match.group() if match else None


def create_test_prompts(n: int, seed: int = 42) -> list[dict]:
    """
    Create test prompts for 3-step math problems with known intermediate results.
    """
    random.seed(seed)
    
    templates = [
        {
            "template": "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
            "step1": lambda x, y, z: x + y,
            "step2": lambda x, y, z: (x + y) * z,
            "final": lambda x, y, z: (x + y) + (x + y) * z,
        },
        {
            "template": "A store has {X} items. {Y} items are sold. Then {Z} items are added. How many items are in the store now? Give the answer only and nothing else.",
            "step1": lambda x, y, z: x - y,
            "step2": lambda x, y, z: (x - y) + z,
            "final": lambda x, y, z: (x - y) + z,
        },
        {
            "template": "There are {X} students. {Y} students join a club, and then each club member recruits {Z} more students. How many students are in the club now? Give the answer only and nothing else.",
            "step1": lambda x, y, z: y,
            "step2": lambda x, y, z: y * z,
            "final": lambda x, y, z: y + y * z,
        },
    ]
    
    prompts = []
    for i in range(n):
        template = random.choice(templates)
        x = random.randint(2, 10)
        y = random.randint(2, 8)
        z = random.randint(2, 5)
        
        prompt = template["template"].format(X=x, Y=y, Z=z)
        step1 = template["step1"](x, y, z)
        step2 = template["step2"](x, y, z)
        final = template["final"](x, y, z)
        
        prompts.append({
            "prompt": prompt,
            "x": x, "y": y, "z": z,
            "step1_result": step1,
            "step2_result": step2,
            "final_answer": final,
        })
    
    return prompts


def run_mvp_validation(wrapper, test_prompts, verbose=True):
    """
    Run MVP validation: check logit lens accuracy on z3 and z5.
    """
    z3_correct = 0
    z5_correct = 0
    z3_total = 0
    z5_total = 0
    
    details = []
    
    iterator = tqdm(test_prompts, desc="Validating MVP") if verbose else test_prompts
    
    for item in iterator:
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=str(item["final_answer"]),
        )
        
        if len(result.latent_vectors) < 5:
            continue
        
        # Check z3 (index 2) for step 1 result
        z3_lens = wrapper.logit_lens(result.latent_vectors[2])
        z3_top1, z3_prob = z3_lens.get_top1_at_final_layer()
        
        # Check z5 (index 4) for step 2 result  
        z5_lens = wrapper.logit_lens(result.latent_vectors[4])
        z5_top1, z5_prob = z5_lens.get_top1_at_final_layer()
        
        step1_gt = str(item["step1_result"])
        step2_gt = str(item["step2_result"])
        
        # Extract number from prediction and compare exactly
        z3_num = extract_number(z3_top1)
        z5_num = extract_number(z5_top1)
        
        z3_match = z3_num is not None and z3_num == step1_gt
        z5_match = z5_num is not None and z5_num == step2_gt
        
        z3_total += 1
        z5_total += 1
        
        if z3_match:
            z3_correct += 1
        if z5_match:
            z5_correct += 1
        
        details.append({
            "prompt": item["prompt"][:50] + "...",
            "step1_gt": step1_gt,
            "step2_gt": step2_gt,
            "z3_pred": z3_top1,
            "z3_prob": z3_prob,
            "z3_match": z3_match,
            "z5_pred": z5_top1,
            "z5_prob": z5_prob,
            "z5_match": z5_match,
            "model_answer": result.predicted_answer,
            "correct": result.is_correct,
        })
    
    z3_acc = z3_correct / z3_total if z3_total > 0 else 0
    z5_acc = z5_correct / z5_total if z5_total > 0 else 0
    
    return {
        "z3_accuracy": z3_acc,
        "z3_correct": z3_correct,
        "z3_total": z3_total,
        "z5_accuracy": z5_acc,
        "z5_correct": z5_correct,
        "z5_total": z5_total,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect latents and validate MVP")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--output", type=str, default="data/mvp_validation.json", help="Output file")
    parser.add_argument("--checkpoint", type=str, default="bcywinski/codi_llama1b-answer_only")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("CODI Activation Oracle - MVP Validation")
    print("=" * 60)
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create test prompts
    print(f"\nCreating {args.n_samples} test prompts...")
    test_prompts = create_test_prompts(args.n_samples, seed=args.seed)
    
    # Save test prompts
    prompts_path = Path("data/test_prompts.json")
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_path, "w") as f:
        json.dump(test_prompts, f, indent=2)
    print(f"Saved test prompts to {prompts_path}")
    
    # Load CODI
    print(f"\nLoading CODI model...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Base model: {args.base_model}")
    
    from src.codi_wrapper import CODIWrapper
    
    wrapper = CODIWrapper.from_pretrained(
        checkpoint_path=args.checkpoint,
        model_name_or_path=args.base_model,
        device=args.device,
    )
    
    print(f"  Hidden size: {wrapper.hidden_size}")
    print(f"  Num layers: {wrapper.num_layers}")
    
    # Run MVP validation
    print(f"\nRunning MVP validation...")
    results = run_mvp_validation(wrapper, test_prompts, verbose=args.verbose)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MVP VALIDATION RESULTS")
    print("=" * 60)
    print(f"z3 (Step 1) Accuracy: {results['z3_accuracy']:.2%} ({results['z3_correct']}/{results['z3_total']})")
    print(f"z5 (Step 2) Accuracy: {results['z5_accuracy']:.2%} ({results['z5_correct']}/{results['z5_total']})")
    
    # Check exit criteria (matches config and README: 90%)
    MVP_THRESHOLD = 0.90
    z3_pass = results['z3_accuracy'] >= MVP_THRESHOLD
    z5_pass = results['z5_accuracy'] >= MVP_THRESHOLD
    
    print("\n" + "-" * 60)
    print(f"MVP Exit Criteria (threshold: {MVP_THRESHOLD:.0%}):")
    print(f"  z3: {'PASS ✓' if z3_pass else 'FAIL ✗'}")
    print(f"  z5: {'PASS ✓' if z5_pass else 'FAIL ✗'}")
    
    if z3_pass and z5_pass:
        print("\n✓ MVP VALIDATION PASSED - Ready to proceed to Phase 2")
    else:
        print("\n✗ MVP VALIDATION FAILED - Investigate before proceeding")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")
    
    # Print some examples
    if args.verbose:
        print("\n" + "-" * 60)
        print("Example results:")
        for i, d in enumerate(results["details"][:5]):
            print(f"\n  [{i+1}] {d['prompt']}")
            print(f"      Step1 GT: {d['step1_gt']}, z3 pred: {d['z3_pred']} (prob={d['z3_prob']:.3f}) {'✓' if d['z3_match'] else '✗'}")
            print(f"      Step2 GT: {d['step2_gt']}, z5 pred: {d['z5_pred']} (prob={d['z5_prob']:.3f}) {'✓' if d['z5_match'] else '✗'}")


if __name__ == "__main__":
    main()
