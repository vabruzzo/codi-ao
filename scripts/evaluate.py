#!/usr/bin/env python3
"""
Evaluation script for CODI Activation Oracle.

Usage:
    # Evaluate AO on single-latent QA (default, backwards compatible)
    python scripts/evaluate.py --ao_path checkpoints/ao --n_samples 100
    
    # Evaluate all task types (QA + Classification + Multi-latent)
    python scripts/evaluate.py --ao_path checkpoints/ao --eval_all
    
    # Evaluate specific task types
    python scripts/evaluate.py --ao_path checkpoints/ao --eval_classification --eval_multi_latent
    
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


def generate_classification_test_examples(
    codi_wrapper,
    n_prompts: int = 50,
    seed: int = 123,
) -> list[dict]:
    """Generate classification test examples from fresh prompts."""
    from scripts.collect_latents import create_synthetic_prompts
    from src.datasets.classification import CLASSIFICATION_TASKS
    from src.activation_oracle import format_oracle_prompt, DEFAULT_PLACEHOLDER_TOKEN
    
    random.seed(seed)
    prompts = create_synthetic_prompts(n_prompts, seed=seed)
    
    examples = []
    for item in prompts:
        result = codi_wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=str(item["final_answer"]),
        )
        
        if len(result.latent_vectors) < 6:
            continue
        
        # Generate classification examples for z2 and z4
        for lat_pos, step_idx in [(1, 0), (3, 1)]:
            latent_vec = result.latent_vectors[lat_pos]
            step_result = str(item[f"step{step_idx + 1}_result"])
            cot_step = f"{step_result}"  # Simplified
            
            # Pick 2 random classification tasks
            task_names = random.sample(list(CLASSIFICATION_TASKS.keys()), min(2, len(CLASSIFICATION_TASKS)))
            
            for task_name in task_names:
                task = CLASSIFICATION_TASKS[task_name]
                
                try:
                    is_true = task["condition"](cot_step, step_result, lat_pos)
                except Exception:
                    continue
                
                question = random.choice(task["questions"])
                oracle_prompt = format_oracle_prompt(
                    question=question,
                    num_activations=1,
                    placeholder_token=DEFAULT_PLACEHOLDER_TOKEN,
                )
                
                examples.append({
                    "prompt": oracle_prompt,
                    "latent_vectors": [latent_vec.tolist()],
                    "latent_positions": [lat_pos],
                    "question": question,
                    "answer": "Yes" if is_true else "No",
                    "question_type": task["type"],
                })
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate CODI Activation Oracle")
    parser.add_argument("--ao_path", type=str, default=None, help="Path to trained AO")
    parser.add_argument("--baseline_only", action="store_true", help="Only run baselines")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of test prompts")
    parser.add_argument("--output", type=str, default="reports/evaluation.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)  # Different from training seed (42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fresh_test", action="store_true", help="Generate fresh test prompts (ignore cached)")
    
    # Task-specific evaluation flags
    parser.add_argument("--eval_qa", action="store_true", help="Evaluate single-latent QA (default if no flags)")
    parser.add_argument("--eval_classification", action="store_true", help="Evaluate classification tasks")
    parser.add_argument("--eval_multi_latent", action="store_true", help="Evaluate multi-latent QA")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate all task types")
    
    args = parser.parse_args()
    
    # Default to QA only if no flags specified
    if not (args.eval_qa or args.eval_classification or args.eval_multi_latent or args.eval_all):
        args.eval_qa = True
    
    # --eval_all enables everything
    if args.eval_all:
        args.eval_qa = True
        args.eval_classification = True
        args.eval_multi_latent = True
    
    print("=" * 60)
    print("CODI Activation Oracle - Evaluation")
    print("=" * 60)
    print(f"Tasks: QA={args.eval_qa}, Classification={args.eval_classification}, Multi-latent={args.eval_multi_latent}")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load test prompts
    # IMPORTANT: Use different seed than training (42) to ensure held-out test set
    test_path = Path("data/test_prompts.json")
    
    if args.fresh_test or not test_path.exists():
        print("\nGenerating fresh test prompts (held-out from training)...")
        from scripts.collect_latents import create_synthetic_prompts
        test_prompts = create_synthetic_prompts(args.n_samples, seed=args.seed)
        print(f"  Generated {len(test_prompts)} prompts with seed={args.seed}")
    else:
        # WARNING: Cached prompts may overlap with training data
        print(f"\nLoading cached test prompts from {test_path}")
        print("  (Use --fresh_test to generate held-out data)")
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
    
    # Initialize evaluator
    from src.evaluation.evaluator import CODIAOEvaluator
    
    evaluator = CODIAOEvaluator(
        codi_wrapper=wrapper,
        activation_oracle=ao,
    )
    
    # Collect all results
    all_results = {}
    
    # 1. Single-latent QA evaluation (original behavior)
    if args.eval_qa:
        print("\n" + "=" * 60)
        print("SINGLE-LATENT QA EVALUATION")
        print("=" * 60)
        
        qa_summary = evaluator.evaluate_intermediate_results(
            test_prompts=test_prompts,
            positions=[1, 3],  # z2 and z4
            verbose=args.verbose,
        )
        
        # Print all examples (qualitative output)
        evaluator._print_all_examples(qa_summary)
        
        # Get examples as dicts for JSON
        qa_examples = evaluator.get_examples_as_dicts(qa_summary)
        
        all_results["qa"] = {
            "total": qa_summary.total_examples,
            "ao_accuracy": qa_summary.ao_accuracy,
            "ao_correct": qa_summary.ao_correct,
            "logit_lens_accuracy": qa_summary.logit_lens_accuracy,
            "logit_lens_correct": qa_summary.logit_lens_correct,
            "z2_ao_accuracy": qa_summary.z2_ao_accuracy,
            "z4_ao_accuracy": qa_summary.z4_ao_accuracy,
            "examples": qa_examples,  # Include all individual results
        }
    
    # 2. Classification evaluation
    if args.eval_classification and ao is not None:
        print("\n" + "=" * 60)
        print("CLASSIFICATION EVALUATION")
        print("=" * 60)
        
        print("Generating classification test examples...")
        class_examples = generate_classification_test_examples(
            wrapper,
            n_prompts=min(args.n_samples, 50),  # Limit for speed
            seed=args.seed + 1000,  # Different seed for classification
        )
        print(f"  Generated {len(class_examples)} classification examples")
        
        class_results = evaluator.evaluate_classification(
            test_examples=class_examples,
            verbose=True,
        )
        
        all_results["classification"] = class_results
    
    # 3. Multi-latent QA evaluation
    if args.eval_multi_latent and ao is not None:
        print("\n" + "=" * 60)
        print("MULTI-LATENT QA EVALUATION")
        print("=" * 60)
        
        multi_results = evaluator.evaluate_multi_latent_qa(
            test_prompts=test_prompts[:min(50, len(test_prompts))],  # Limit for speed
            verbose=True,
        )
        
        all_results["multi_latent"] = multi_results
    
    # Save all results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    if "qa" in all_results:
        qa = all_results["qa"]
        print(f"\nSingle-Latent QA:")
        print(f"  Logit Lens: {qa['logit_lens_accuracy']:.2%} ({qa['logit_lens_correct']}/{qa['total']})")
        if ao:
            print(f"  AO:         {qa['ao_accuracy']:.2%} ({qa['ao_correct']}/{qa['total']})")
    
    if "classification" in all_results:
        cls = all_results["classification"]
        print(f"\nClassification:")
        print(f"  Overall:    {cls['accuracy']:.2%} ({cls['correct']}/{cls['total']})")
        if "by_type" in cls:
            for ctype, r in sorted(cls["by_type"].items()):
                print(f"  {ctype}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")
    
    if "multi_latent" in all_results:
        ml = all_results["multi_latent"]
        print(f"\nMulti-Latent QA:")
        print(f"  Overall:    {ml['overall']['accuracy']:.2%} ({ml['overall']['correct']}/{ml['overall']['total']})")
        print(f"  Step 1:     {ml['step1']['accuracy']:.2%}")
        print(f"  Step 2:     {ml['step2']['accuracy']:.2%}")
        print(f"  Final:      {ml['final']['accuracy']:.2%}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
