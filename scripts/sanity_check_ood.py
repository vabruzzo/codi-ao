#!/usr/bin/env python3
"""
Quick sanity check: Test AO on out-of-distribution numbers (1-100 instead of 1-10).

This tests whether the AO learned actual latent interpretation or just memorized
patterns from the narrow training distribution.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.codi_wrapper import CODIWrapper
from src.activation_oracle import ActivationOracle, format_oracle_prompt


def create_ood_prompts(n: int, min_val: int = 1, max_val: int = 100, seed: int = 999) -> list[dict]:
    """Create test prompts with larger number ranges."""
    random.seed(seed)
    
    ADDITION_TEMPLATES = [
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. They hire {Y} more employees. Then each current employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. They enroll {Y} new students. Then each current student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
    ]
    
    SUBTRACTION_TEMPLATES = [
        "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. {Y} employees resign. Then each remaining employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. {Y} students transfer out. Then each remaining student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
    ]
    
    prompts = []
    for i in range(n):
        is_addition = random.random() < 0.5
        
        if is_addition:
            template = random.choice(ADDITION_TEMPLATES)
            x = random.randint(min_val, max_val)
            y = random.randint(min_val, max_val)
            z = random.randint(min_val, max_val)
            step1 = x + y
            step2 = (x + y) * z
            final = (x + y) + (x + y) * z
            op_type = "addition"
        else:
            template = random.choice(SUBTRACTION_TEMPLATES)
            x = random.randint(min_val + 1, max_val)
            y = random.randint(min_val, x - 1)
            z = random.randint(min_val, max_val)
            step1 = x - y
            step2 = (x - y) * z
            final = (x - y) + (x - y) * z
            op_type = "subtraction"
        
        prompt = template.format(X=x, Y=y, Z=z)
        
        prompts.append({
            "prompt": prompt,
            "type": op_type,
            "x": x, "y": y, "z": z,
            "step1_result": str(step1),
            "step2_result": str(step2),
            "final_answer": str(final),
        })
    
    return prompts


def run_sanity_check(n_samples: int = 100, number_range: tuple = (1, 100)):
    """Run the sanity check evaluation."""
    
    min_val, max_val = number_range
    print(f"\n{'='*60}")
    print(f"SANITY CHECK: Numbers {min_val}-{max_val}")
    print(f"{'='*60}\n")
    
    # Generate OOD test prompts
    print(f"Generating {n_samples} test prompts with numbers {min_val}-{max_val}...")
    prompts = create_ood_prompts(n_samples, min_val=min_val, max_val=max_val)
    
    # Load models
    print("Loading CODI model...")
    wrapper = CODIWrapper()
    
    print("Loading Activation Oracle...")
    ao = ActivationOracle()
    
    # Run evaluation
    z2_ao_correct = 0
    z4_ao_correct = 0
    z2_ll_correct = 0
    z4_ll_correct = 0
    total = 0
    
    errors = []
    
    print(f"\nEvaluating...")
    for item in tqdm(prompts):
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=item["final_answer"],
        )
        
        # Test z2 (step 1)
        z2_latent = result.latent_vectors[1]  # Index 1 = z2
        
        # Logit lens for z2
        z2_ll_pred = wrapper.logit_lens(z2_latent)
        z2_ll_ok = z2_ll_pred.strip() == item["step1_result"]
        if z2_ll_ok:
            z2_ll_correct += 1
        
        # AO for z2
        z2_ao_prompt = format_oracle_prompt(
            question="What is the intermediate calculation result?",
            num_activations=1,
        )
        z2_ao_output = ao.query(z2_ao_prompt, [z2_latent])
        z2_ao_ok = z2_ao_output.strip() == item["step1_result"]
        if z2_ao_ok:
            z2_ao_correct += 1
        
        # Test z4 (step 2)
        z4_latent = result.latent_vectors[3]  # Index 3 = z4
        
        # Logit lens for z4
        z4_ll_pred = wrapper.logit_lens(z4_latent)
        z4_ll_ok = z4_ll_pred.strip() == item["step2_result"]
        if z4_ll_ok:
            z4_ll_correct += 1
        
        # AO for z4
        z4_ao_prompt = format_oracle_prompt(
            question="What is the intermediate calculation result?",
            num_activations=1,
        )
        z4_ao_output = ao.query(z4_ao_prompt, [z4_latent])
        z4_ao_ok = z4_ao_output.strip() == item["step2_result"]
        if z4_ao_ok:
            z4_ao_correct += 1
        
        total += 1
        
        # Track some errors for analysis
        if not z2_ao_ok or not z4_ao_ok:
            errors.append({
                "prompt": item["prompt"][:80] + "...",
                "x": item["x"], "y": item["y"], "z": item["z"],
                "step1_expected": item["step1_result"],
                "step1_ao": z2_ao_output.strip(),
                "step1_ll": z2_ll_pred.strip(),
                "step2_expected": item["step2_result"],
                "step2_ao": z4_ao_output.strip(),
                "step2_ll": z4_ll_pred.strip(),
                "z2_ao_ok": z2_ao_ok,
                "z4_ao_ok": z4_ao_ok,
            })
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: Numbers {min_val}-{max_val} (n={total*2} latent tests)")
    print(f"{'='*60}\n")
    
    print("Step 1 (z2):")
    print(f"  Logit Lens: {z2_ll_correct}/{total} ({100*z2_ll_correct/total:.1f}%)")
    print(f"  AO:         {z2_ao_correct}/{total} ({100*z2_ao_correct/total:.1f}%)")
    
    print(f"\nStep 2 (z4):")
    print(f"  Logit Lens: {z4_ll_correct}/{total} ({100*z4_ll_correct/total:.1f}%)")
    print(f"  AO:         {z4_ao_correct}/{total} ({100*z4_ao_correct/total:.1f}%)")
    
    combined_ao = z2_ao_correct + z4_ao_correct
    combined_ll = z2_ll_correct + z4_ll_correct
    combined_total = total * 2
    
    print(f"\nCombined:")
    print(f"  Logit Lens: {combined_ll}/{combined_total} ({100*combined_ll/combined_total:.1f}%)")
    print(f"  AO:         {combined_ao}/{combined_total} ({100*combined_ao/combined_total:.1f}%)")
    
    # Show some error examples
    if errors:
        print(f"\n{'='*60}")
        print(f"SAMPLE ERRORS (showing up to 10)")
        print(f"{'='*60}\n")
        
        for i, err in enumerate(errors[:10]):
            print(f"[{i+1}] x={err['x']}, y={err['y']}, z={err['z']}")
            print(f"    Step1: expected={err['step1_expected']}, AO={err['step1_ao']}, LL={err['step1_ll']} {'✓' if err['z2_ao_ok'] else '✗'}")
            print(f"    Step2: expected={err['step2_expected']}, AO={err['step2_ao']}, LL={err['step2_ll']} {'✓' if err['z4_ao_ok'] else '✗'}")
            print()
    
    return {
        "z2_ao": z2_ao_correct / total,
        "z4_ao": z4_ao_correct / total,
        "z2_ll": z2_ll_correct / total,
        "z4_ll": z4_ll_correct / total,
        "combined_ao": combined_ao / combined_total,
        "combined_ll": combined_ll / combined_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100, help="Number of test prompts")
    parser.add_argument("--min_val", type=int, default=1, help="Minimum number")
    parser.add_argument("--max_val", type=int, default=100, help="Maximum number")
    args = parser.parse_args()
    
    # Run the sanity check
    results = run_sanity_check(
        n_samples=args.n_samples,
        number_range=(args.min_val, args.max_val),
    )
    
    # Also run on original range for comparison
    print(f"\n\n{'#'*60}")
    print("COMPARISON: Original range (1-10)")
    print(f"{'#'*60}")
    
    results_orig = run_sanity_check(
        n_samples=args.n_samples,
        number_range=(1, 10),
    )
    
    # Summary comparison
    print(f"\n\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Range':<15} {'AO':<15} {'Logit Lens':<15}")
    print(f"{'-'*45}")
    print(f"{'1-10':<15} {100*results_orig['combined_ao']:.1f}%{'':<8} {100*results_orig['combined_ll']:.1f}%")
    print(f"{f'{args.min_val}-{args.max_val}':<15} {100*results['combined_ao']:.1f}%{'':<8} {100*results['combined_ll']:.1f}%")
    
    drop = results_orig['combined_ao'] - results['combined_ao']
    print(f"\nAO accuracy drop: {100*drop:.1f}%")
