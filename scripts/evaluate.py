#!/usr/bin/env python3
"""
Evaluation script for CODI Activation Oracle.

Usage:
    # Evaluate AO against baselines
    python scripts/evaluate.py --ao_path checkpoints/ao --n_samples 100
    
    # Baseline only (no AO)
    python scripts/evaluate.py --baseline_only
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def main():
    parser = argparse.ArgumentParser(description="Evaluate CODI Activation Oracle")
    parser.add_argument("--ao_path", type=str, default=None, help="Path to trained AO")
    parser.add_argument("--baseline_only", action="store_true", help="Only run baselines")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="reports/evaluation.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CODI Activation Oracle - Evaluation")
    print("=" * 60)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load test prompts
    test_path = Path("data/test_prompts.json")
    if not test_path.exists():
        print(f"No test prompts found at {test_path}")
        print("Run scripts/collect_latents.py first to generate test data")
        
        # Generate on the fly
        print("\nGenerating test prompts...")
        from scripts.collect_latents import create_test_prompts
        test_prompts = create_test_prompts(args.n_samples, seed=args.seed)
    else:
        with open(test_path) as f:
            test_prompts = json.load(f)
        
        if len(test_prompts) > args.n_samples:
            test_prompts = random.sample(test_prompts, args.n_samples)
    
    print(f"Loaded {len(test_prompts)} test prompts")
    
    # Load CODI wrapper
    print("\nLoading CODI model...")
    from src.codi_wrapper import CODIWrapper
    
    wrapper = CODIWrapper.from_pretrained(device=args.device)
    
    # Load AO if provided
    ao = None
    if args.ao_path and not args.baseline_only:
        print(f"\nLoading Activation Oracle from {args.ao_path}...")
        from src.activation_oracle import ActivationOracle, AOConfig
        
        config = AOConfig(device=args.device)
        ao = ActivationOracle.from_pretrained(
            config=config,
            lora_path=args.ao_path,
        )
        ao.eval_mode()
    
    # Run evaluation
    print("\nRunning evaluation...")
    from src.evaluation.evaluator import CODIAOEvaluator
    
    evaluator = CODIAOEvaluator(
        codi_wrapper=wrapper,
        activation_oracle=ao,
    )
    
    summary = evaluator.evaluate_intermediate_results(
        test_prompts=test_prompts,
        positions=[2, 4],
        verbose=args.verbose,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.save(str(output_path))
    print(f"\nSaved results to {output_path}")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Method':<20} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 42)
    print(f"{'Logit Lens':<20} {summary.logit_lens_accuracy:>10.2%} {summary.logit_lens_correct:>10}")
    if ao:
        print(f"{'Activation Oracle':<20} {summary.ao_accuracy:>10.2%} {summary.ao_correct:>10}")
    
    # AO vs baseline comparison
    if ao and summary.logit_lens_accuracy > 0:
        improvement = (summary.ao_accuracy - summary.logit_lens_accuracy) / summary.logit_lens_accuracy * 100
        print(f"\nAO vs Logit Lens: {improvement:+.1f}%")
        
        if summary.ao_accuracy >= summary.logit_lens_accuracy:
            print("✓ AO meets or exceeds logit lens baseline")
        else:
            print("✗ AO below logit lens baseline")


if __name__ == "__main__":
    main()
