#!/usr/bin/env python3
"""
Script to collect latent vectors from CODI and validate the MVP.

This script:
1. Loads CODI model
2. Runs on test prompts to collect latent vectors
3. Verifies that z2/z4 store intermediate results via logit lens
4. Reports logit lens accuracy as the MVP baseline

Index Mapping Note:
    The LessWrong blog refers to "z3 and z5" (the third and fifth latent vectors).
    However, their counting INCLUDES an initial position (prompt/bocot output):
    - Their indexing: [0:initial, 1:iter0, 2:iter1, 3:iter2, 4:iter3, 5:iter4, 6:iter5]
    - "Third" = their index 2 = iteration 1 output
    - "Fifth" = their index 4 = iteration 3 output
    
    Our code stores only the 6 iteration outputs (no initial):
    - Our indexing: [0:iter0, 1:iter1, 2:iter2, 3:iter3, 4:iter4, 5:iter5]
    - Step 1 result at our index 1 (z2) = their index 2 = "third"
    - Step 2 result at our index 3 (z4) = their index 4 = "fifth"

Usage:
    python scripts/collect_latents.py --n_samples 100 --verbose --synthetic
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


def parse_gsm8k_answer(answer: str) -> tuple[list[str], str]:
    """
    Parse GSM8k answer to extract intermediate steps and final answer.
    
    GSM8k format: "Step text <<expr=result>>... #### final_answer"
    
    Returns:
        (list of intermediate results, final answer)
    """
    # Extract intermediate calculations: <<expr=result>>
    calc_pattern = r"<<[^>]*=\s*([^>]+)>>"
    intermediate_results = re.findall(calc_pattern, answer)
    
    # Clean up results (remove commas, whitespace)
    intermediate_results = [r.strip().replace(",", "") for r in intermediate_results]
    
    # Extract final answer after ####
    final_match = re.search(r"####\s*(.+)", answer)
    final_answer = final_match.group(1).strip().replace(",", "") if final_match else ""
    
    return intermediate_results, final_answer


def load_gsm8k_prompts(n: int, split: str = "test", seed: int = 42) -> list[dict]:
    """
    Load prompts from GSM8k dataset.
    
    Args:
        n: Number of prompts to load
        split: "train" or "test" (use "test" for evaluation)
        seed: Random seed for sampling
    
    Returns:
        List of prompt dicts with intermediate results
    """
    from datasets import load_dataset
    
    print(f"Loading GSM8k {split} set...")
    dataset = load_dataset("gsm8k", "main", split=split)
    
    # Sample n examples
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    prompts = []
    skipped = 0
    
    for idx in indices:
        item = dataset[idx]
        question = item["question"]
        answer = item["answer"]
        
        intermediate_results, final_answer = parse_gsm8k_answer(answer)
        
        # Skip problems with fewer than 2 intermediate steps
        # (we need at least step1 for z2 and step2 for z4)
        if len(intermediate_results) < 2:
            skipped += 1
            continue
        
        # Add instruction suffix (same format CODI was trained on)
        prompt = f"{question} Give the answer only and nothing else."
        
        prompts.append({
            "prompt": prompt,
            "question": question,
            "full_answer": answer,
            "intermediate_results": intermediate_results,
            "step1_result": intermediate_results[0] if intermediate_results else None,
            "step2_result": intermediate_results[1] if len(intermediate_results) > 1 else None,
            "final_answer": final_answer,
            "num_steps": len(intermediate_results),
        })
    
    if skipped > 0:
        print(f"  Skipped {skipped} problems with < 2 intermediate steps")
    
    return prompts


def create_synthetic_prompts(n: int, seed: int = 42) -> list[dict]:
    """
    Create test prompts using the EXACT templates from the CODI paper.
    
    All templates follow a 3-step structure:
    - Addition: (X+Y)*Z + (X+Y)
    - Subtraction: (X-Y)*Z + (X-Y)
    
    Step 1 result -> z2 (index 1), Step 2 result -> z4 (index 3)
    """
    random.seed(seed)
    
    # Templates from references/codi/src/templates.py
    ADDITION_TEMPLATES = [
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. They hire {Y} more employees. Then each current employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. They enroll {Y} new students. Then each current student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
        "A club starts with {X} members. They add {Y} new members. Then each current member invites {Z} additional people. How many people are there now in the club? Give the answer only and nothing else.",
        "A restaurant starts with {X} customers. They welcome {Y} more customers. Then each current customer brings {Z} additional customers. How many customers are there now in the restaurant? Give the answer only and nothing else.",
        "A gym starts with {X} members. They sign up {Y} new members. Then each current member refers {Z} additional people. How many people are there now in the gym? Give the answer only and nothing else.",
        "A band starts with {X} musicians. They add {Y} more musicians. Then each current musician brings {Z} additional musicians. How many musicians are there now in the band? Give the answer only and nothing else.",
        "A community starts with {X} residents. They welcome {Y} new residents. Then each current resident invites {Z} additional people. How many people are there now in the community? Give the answer only and nothing else.",
        "A group starts with {X} participants. They add {Y} new participants. Then each current participant brings {Z} additional people. How many people are there now in the group? Give the answer only and nothing else.",
        "A workshop starts with {X} attendees. They register {Y} more attendees. Then each current attendee brings {Z} additional people. How many people are there now in the workshop? Give the answer only and nothing else.",
    ]
    
    SUBTRACTION_TEMPLATES = [
        "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. {Y} employees resign. Then each remaining employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. {Y} students transfer out. Then each remaining student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
        "A club starts with {X} members. {Y} members quit. Then each remaining member invites {Z} additional people. How many people are there now in the club? Give the answer only and nothing else.",
        "A restaurant starts with {X} customers. {Y} customers leave. Then each remaining customer brings {Z} additional customers. How many customers are there now in the restaurant? Give the answer only and nothing else.",
        "A gym starts with {X} members. {Y} members cancel. Then each remaining member refers {Z} additional people. How many people are there now in the gym? Give the answer only and nothing else.",
        "A band starts with {X} musicians. {Y} musicians depart. Then each remaining musician brings {Z} additional musicians. How many musicians are there now in the band? Give the answer only and nothing else.",
        "A community starts with {X} residents. {Y} residents move away. Then each remaining resident invites {Z} additional people. How many people are there now in the community? Give the answer only and nothing else.",
        "A group starts with {X} participants. {Y} participants drop out. Then each remaining participant brings {Z} additional people. How many people are there now in the group? Give the answer only and nothing else.",
        "A workshop starts with {X} attendees. {Y} attendees withdraw. Then each remaining attendee brings {Z} additional people. How many people are there now in the workshop? Give the answer only and nothing else.",
    ]
    
    templates = []
    
    # Addition templates: Step1 = X+Y, Step2 = (X+Y)*Z, Final = (X+Y) + (X+Y)*Z
    for t in ADDITION_TEMPLATES:
        templates.append({
            "template": t,
            "type": "addition",
            "step1": lambda x, y, z: x + y,
            "step2": lambda x, y, z: (x + y) * z,
            "final": lambda x, y, z: (x + y) + (x + y) * z,
        })
    
    # Subtraction templates: Step1 = X-Y, Step2 = (X-Y)*Z, Final = (X-Y) + (X-Y)*Z
    for t in SUBTRACTION_TEMPLATES:
        templates.append({
            "template": t,
            "type": "subtraction",
            "step1": lambda x, y, z: x - y,
            "step2": lambda x, y, z: (x - y) * z,
            "final": lambda x, y, z: (x - y) + (x - y) * z,
        })
    
    prompts = []
    for i in range(n):
        template = random.choice(templates)
        
        # Use numbers 1-10 as in the paper
        if template["type"] == "addition":
            x = random.randint(1, 10)
            y = random.randint(1, 10)
            z = random.randint(1, 10)
        else:
            # For subtraction, ensure X > Y to avoid negative numbers
            x = random.randint(2, 10)
            y = random.randint(1, x - 1)
            z = random.randint(1, 10)
        
        prompt = template["template"].format(X=x, Y=y, Z=z)
        step1 = template["step1"](x, y, z)
        step2 = template["step2"](x, y, z)
        final = template["final"](x, y, z)
        
        prompts.append({
            "prompt": prompt,
            "type": template["type"],
            "x": x, "y": y, "z": z,
            "step1_result": str(step1),
            "step2_result": str(step2),
            "final_answer": str(final),
        })
    
    return prompts


def run_mvp_validation(wrapper, test_prompts, verbose=True, diagnose=False):
    """
    Run MVP validation: check logit lens accuracy on z2 and z4.
    
    Args:
        diagnose: If True, check all latent positions to find where steps are encoded
    """
    z2_correct = 0
    z4_correct = 0
    z2_total = 0
    z4_total = 0
    
    # Track which latent position best matches each step (for diagnosis)
    step1_hits_by_position = {i: 0 for i in range(6)}
    step2_hits_by_position = {i: 0 for i in range(6)}
    final_hits_by_position = {i: 0 for i in range(6)}
    
    details = []
    
    iterator = tqdm(test_prompts, desc="Validating MVP") if verbose else test_prompts
    
    for item in iterator:
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=str(item["final_answer"]),
        )
        
        if len(result.latent_vectors) < 6:
            continue
        
        step1_gt = str(item["step1_result"])
        step2_gt = str(item["step2_result"])
        final_gt = str(item["final_answer"])
        
        # Get predictions from all latent positions
        all_preds = []
        for i in range(6):
            lens_result = wrapper.logit_lens(result.latent_vectors[i])
            top1, prob = lens_result.get_top1_at_final_layer()
            num = extract_number(top1)
            all_preds.append({"token": top1, "prob": prob, "num": num})
            
            # Track hits for diagnosis
            if num == step1_gt:
                step1_hits_by_position[i] += 1
            if num == step2_gt:
                step2_hits_by_position[i] += 1
            if num == final_gt:
                final_hits_by_position[i] += 1
        
        # Use z2 (index 1) for step 1, z4 (index 3) for step 2
        # Note: Blog says "z3/z5" counting WITH prompt position at 0
        # Our latent_vectors excludes prompt, so we use indices 1 and 3
        z2_num = all_preds[1]["num"]
        z4_num = all_preds[3]["num"]
        
        z2_match = z2_num is not None and z2_num == step1_gt
        z4_match = z4_num is not None and z4_num == step2_gt
        
        z2_total += 1
        z4_total += 1
        
        if z2_match:
            z2_correct += 1
        if z4_match:
            z4_correct += 1
        
        detail = {
            "prompt": item["prompt"][:50] + "...",
            "step1_gt": step1_gt,
            "step2_gt": step2_gt,
            "final_gt": final_gt,
            "z2_pred": all_preds[1]["token"],
            "z2_prob": all_preds[1]["prob"],
            "z2_match": z2_match,
            "z4_pred": all_preds[3]["token"],
            "z4_prob": all_preds[3]["prob"],
            "z4_match": z4_match,
            "model_answer": result.predicted_answer,
            "correct": result.is_correct,
        }
        
        # Add all latent predictions for diagnosis
        if diagnose:
            detail["all_latent_preds"] = [
                {"z": i+1, "pred": p["token"], "num": p["num"], "prob": p["prob"]}
                for i, p in enumerate(all_preds)
            ]
        
        details.append(detail)
    
    z2_acc = z2_correct / z2_total if z2_total > 0 else 0
    z4_acc = z4_correct / z4_total if z4_total > 0 else 0
    
    result = {
        "z2_accuracy": z2_acc,
        "z2_correct": z2_correct,
        "z2_total": z2_total,
        "z4_accuracy": z4_acc,
        "z4_correct": z4_correct,
        "z4_total": z4_total,
        "details": details,
    }
    
    # Add diagnostic info
    if diagnose:
        result["diagnosis"] = {
            "step1_hits_by_position": {f"z{k+1}": v for k, v in step1_hits_by_position.items()},
            "step2_hits_by_position": {f"z{k+1}": v for k, v in step2_hits_by_position.items()},
            "final_hits_by_position": {f"z{k+1}": v for k, v in final_hits_by_position.items()},
            "total_samples": z2_total,
        }
    
    return result


def generate_training_data(
    wrapper,
    prompts: list[dict],
    output_path: str,
    placeholder_token: str,
    verbose: bool = True,
    multi_latent_ratio: float = 0.33,
) -> dict:
    """
    Generate training data by collecting latents from CODI.
    
    Creates a mix of:
    - Single-latent examples: z2 or z4 paired with their intermediate results
    - Multi-latent examples: all 6 vectors with full reasoning questions
    
    Args:
        wrapper: CODIWrapper instance
        prompts: List of prompts with step1_result, step2_result, final_answer
        output_path: Path to save JSONL output
        placeholder_token: Placeholder token for oracle prompts
        verbose: Show progress bar
        multi_latent_ratio: Fraction of examples that use all 6 vectors (default 0.33)
        
    Returns:
        Dict with generation statistics
    """
    from src.activation_oracle import format_oracle_prompt
    from src.datasets.latent_qa import INTERMEDIATE_RESULT_TEMPLATES, MULTI_LATENT_TEMPLATES
    
    examples = []
    skipped = 0
    single_count = 0
    multi_count = 0
    
    iterator = tqdm(prompts, desc="Generating training data") if verbose else prompts
    
    for item in iterator:
        # Run CODI to get latents
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=str(item["final_answer"]),
        )
        
        if len(result.latent_vectors) < 6:
            skipped += 1
            continue
        
        # Get ground truth values
        step1_gt = str(item["step1_result"])
        step2_gt = str(item["step2_result"])
        final_gt = str(item["final_answer"])
        
        # Helper to convert vector to list
        def vec_to_list(v):
            return v.cpu().tolist() if hasattr(v, 'cpu') else v
        
        # Decide if this prompt generates single or multi-latent examples
        use_multi = random.random() < multi_latent_ratio
        
        if use_multi:
            # Multi-latent example: all 6 vectors
            all_vectors = [vec_to_list(result.latent_vectors[i]) for i in range(6)]
            question = random.choice(MULTI_LATENT_TEMPLATES)
            
            # Generate appropriate answer based on question type
            if "first step" in question.lower() or "step 1" in question.lower():
                answer = step1_gt
            elif "second" in question.lower() or "step 2" in question.lower():
                answer = step2_gt
            elif "final" in question.lower() or "answer" in question.lower():
                answer = final_gt
            elif "error" in question.lower() or "correct" in question.lower():
                # For verification questions, we assume correct reasoning
                answer = "Yes, the reasoning is correct."
            else:
                # Full reasoning summary
                answer = f"Step 1: {step1_gt}, Step 2: {step2_gt}, Final: {final_gt}"
            
            oracle_prompt = format_oracle_prompt(
                question=question,
                num_activations=6,
                placeholder_token=placeholder_token,
                multi_latent=True,
            )
            
            examples.append({
                "prompt": oracle_prompt,
                "latent_vectors": all_vectors,
                "latent_positions": [0, 1, 2, 3, 4, 5],
                "question": question,
                "answer": answer,
                "source_prompt": item["prompt"],
                "cot_step": f"Full: {step1_gt} -> {step2_gt} -> {final_gt}",
                "question_type": "full_reasoning",
                "is_multi_latent": True,
            })
            multi_count += 1
            
        else:
            # Single-latent examples: z2 and z4
            
            # Example for z2 (Step 1 result)
            z2_vector = result.latent_vectors[1]
            question = random.choice(INTERMEDIATE_RESULT_TEMPLATES)
            oracle_prompt = format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=placeholder_token,
            )
            
            examples.append({
                "prompt": oracle_prompt,
                "latent_vectors": [vec_to_list(z2_vector)],
                "latent_positions": [1],
                "question": question,
                "answer": step1_gt,
                "source_prompt": item["prompt"],
                "cot_step": f"Step 1 -> {step1_gt}",
                "question_type": "intermediate_result",
                "is_multi_latent": False,
            })
            single_count += 1
            
            # Example for z4 (Step 2 result)
            z4_vector = result.latent_vectors[3]
            question = random.choice(INTERMEDIATE_RESULT_TEMPLATES)
            oracle_prompt = format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=placeholder_token,
            )
            
            examples.append({
                "prompt": oracle_prompt,
                "latent_vectors": [vec_to_list(z4_vector)],
                "latent_positions": [3],
                "question": question,
                "answer": step2_gt,
                "source_prompt": item["prompt"],
                "cot_step": f"Step 2 -> {step2_gt}",
                "question_type": "intermediate_result",
                "is_multi_latent": False,
            })
            single_count += 1
    
    # Shuffle examples
    random.shuffle(examples)
    
    # Save to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    stats = {
        "total_examples": len(examples),
        "single_latent_examples": single_count,
        "multi_latent_examples": multi_count,
        "skipped_prompts": skipped,
        "output_path": str(output_file),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect latents and validate MVP")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--output", type=str, default="data/mvp_validation.json", help="Output file")
    parser.add_argument("--checkpoint", type=str, default="bcywinski/codi_llama1b-answer_only")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic prompts instead of GSM8k")
    parser.add_argument("--diagnose", action="store_true", help="Check all latent positions to find best mapping")
    parser.add_argument("--generate_data", action="store_true", help="Generate training data instead of just validating")
    parser.add_argument("--train_output", type=str, default="data/latent_qa_train.jsonl", help="Output path for training data")
    parser.add_argument("--gsm8k_split", type=str, default="test", choices=["train", "test"], help="GSM8k split to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CODI Activation Oracle - MVP Validation")
    print("=" * 60)
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load prompts
    if args.synthetic:
        print(f"\nCreating {args.n_samples} synthetic prompts...")
        test_prompts = create_synthetic_prompts(args.n_samples, seed=args.seed)
    else:
        split = args.gsm8k_split if args.generate_data else "test"
        print(f"\nLoading {args.n_samples} prompts from GSM8k {split} set...")
        try:
            test_prompts = load_gsm8k_prompts(args.n_samples, split=split, seed=args.seed)
            print(f"  Loaded {len(test_prompts)} prompts with ≥2 intermediate steps")
        except Exception as e:
            print(f"  Failed to load GSM8k: {e}")
            print("  Falling back to synthetic prompts...")
            test_prompts = create_synthetic_prompts(args.n_samples, seed=args.seed)
    
    # Save test prompts
    prompts_path = Path("data/test_prompts.json")
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_path, "w") as f:
        json.dump(test_prompts, f, indent=2, default=str)
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
    
    # Generate training data if requested
    if args.generate_data:
        print(f"\nGenerating training data...")
        
        # Get placeholder token from AO config for consistency
        from src.activation_oracle import DEFAULT_PLACEHOLDER_TOKEN
        
        stats = generate_training_data(
            wrapper=wrapper,
            prompts=test_prompts,
            output_path=args.train_output,
            placeholder_token=DEFAULT_PLACEHOLDER_TOKEN,
            verbose=args.verbose,
        )
        
        print("\n" + "=" * 60)
        print("TRAINING DATA GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Single-latent examples: {stats['single_latent_examples']}")
        print(f"  Multi-latent examples: {stats['multi_latent_examples']}")
        print(f"  Skipped prompts: {stats['skipped_prompts']}")
        print(f"  Saved to: {stats['output_path']}")
        
        # Still run validation to show baselines
        print("\nAlso running validation to establish baselines...")
    
    # Run MVP validation
    print(f"\nRunning MVP validation...")
    results = run_mvp_validation(wrapper, test_prompts, verbose=args.verbose, diagnose=args.diagnose)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MVP VALIDATION RESULTS")
    print("=" * 60)
    print(f"z2 (Step 1) Accuracy: {results['z2_accuracy']:.2%} ({results['z2_correct']}/{results['z2_total']})")
    print(f"z4 (Step 2) Accuracy: {results['z4_accuracy']:.2%} ({results['z4_correct']}/{results['z4_total']})")
    
    # Print diagnostic info if available
    if args.diagnose and "diagnosis" in results:
        diag = results["diagnosis"]
        print("\n" + "-" * 60)
        print("DIAGNOSTIC: Which latent position matches each step?")
        print(f"  (out of {diag['total_samples']} samples)")
        print("\n  Step 1 intermediate result found in:")
        for pos, hits in diag["step1_hits_by_position"].items():
            pct = hits / diag["total_samples"] * 100 if diag["total_samples"] > 0 else 0
            bar = "█" * int(pct / 5) 
            print(f"    {pos}: {hits:3d} ({pct:5.1f}%) {bar}")
        print("\n  Step 2 intermediate result found in:")
        for pos, hits in diag["step2_hits_by_position"].items():
            pct = hits / diag["total_samples"] * 100 if diag["total_samples"] > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    {pos}: {hits:3d} ({pct:5.1f}%) {bar}")
        print("\n  Final answer found in:")
        for pos, hits in diag["final_hits_by_position"].items():
            pct = hits / diag["total_samples"] * 100 if diag["total_samples"] > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    {pos}: {hits:3d} ({pct:5.1f}%) {bar}")
    
    # Check exit criteria
    # 85% threshold: z2 should be ~100%, z4 should be ≥85%
    # This matches the original CODI blog post findings
    MVP_THRESHOLD = 0.85
    z2_pass = results['z2_accuracy'] >= MVP_THRESHOLD
    z4_pass = results['z4_accuracy'] >= MVP_THRESHOLD
    
    print("\n" + "-" * 60)
    print(f"MVP Exit Criteria (threshold: {MVP_THRESHOLD:.0%}):")
    print(f"  z2 (Step 1): {'PASS ✓' if z2_pass else 'FAIL ✗'}")
    print(f"  z4 (Step 2): {'PASS ✓' if z4_pass else 'FAIL ✗'}")
    
    if z2_pass and z4_pass:
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
            print(f"      Step1 GT: {d['step1_gt']}, z2 pred: {d['z2_pred']} (prob={d['z2_prob']:.3f}) {'✓' if d['z2_match'] else '✗'}")
            print(f"      Step2 GT: {d['step2_gt']}, z4 pred: {d['z4_pred']} (prob={d['z4_prob']:.3f}) {'✓' if d['z4_match'] else '✗'}")


if __name__ == "__main__":
    main()
