#!/usr/bin/env python3
"""
Generate AO training data from holdout problems.
Runs CODI on training problems and creates QA pairs for AO fine-tuning.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codi_wrapper import CODIWrapper


# Question templates for different extraction tasks
QUESTION_TEMPLATES = {
    "step1": [
        "What was calculated in the first step?",
        "What is the result of the first calculation?",
        "What number was computed first?",
    ],
    "step2": [
        "What was calculated in the second step?",
        "What is the result of the second calculation?",
        "What number was computed in step two?",
    ],
    "step3": [
        "What is the final answer?",
        "What was calculated in the third step?",
        "What is the result of the final calculation?",
    ],
    "operation": [
        "What mathematical operation was used in the first step?",
        "Was the first step addition, subtraction, or multiplication?",
        "What operation combined the first two numbers?",
    ],
    "comparison": [
        "Which is larger: the result of step 1 or step 2?",
        "Is step 2 greater than step 1?",
    ],
}

OPERATION_NAMES = {
    "add": "addition",
    "sub": "subtraction",
    "mul": "multiplication",
}


def load_codi():
    """Load CODI model."""
    return CODIWrapper.from_pretrained()


def create_training_examples(problem, latents, codi_output, codi_correct):
    """Create multiple training examples from a single problem."""
    examples = []
    
    step1 = problem["step1"]
    step2 = problem["step2"]
    step3 = problem["step3"]
    operation = problem["operation"]
    
    # We have 6 latents: z1-z6 (indices 0-5)
    # Based on prior analysis: z2 (idx 1) has step1, z4 (idx 3) has step2
    
    z2 = latents[1]  # Step 1 latent
    z4 = latents[3]  # Step 2 latent
    all_latents = latents[:6]
    
    import random
    
    # === Step 1 extraction ===
    # Single latent (z2)
    q = random.choice(QUESTION_TEMPLATES["step1"])
    examples.append({
        "question": q,
        "answer": str(step1),
        "latent_indices": [1],
        "task": "step1_single",
        "ground_truth_step1": step1,
    })
    
    # Multi-latent (all 6)
    q = random.choice(QUESTION_TEMPLATES["step1"])
    examples.append({
        "question": q,
        "answer": str(step1),
        "latent_indices": [0, 1, 2, 3, 4, 5],
        "task": "step1_multi",
        "ground_truth_step1": step1,
    })
    
    # === Step 2 extraction ===
    # Single latent (z4)
    q = random.choice(QUESTION_TEMPLATES["step2"])
    examples.append({
        "question": q,
        "answer": str(step2),
        "latent_indices": [3],
        "task": "step2_single",
        "ground_truth_step2": step2,
    })
    
    # Multi-latent
    q = random.choice(QUESTION_TEMPLATES["step2"])
    examples.append({
        "question": q,
        "answer": str(step2),
        "latent_indices": [0, 1, 2, 3, 4, 5],
        "task": "step2_multi",
        "ground_truth_step2": step2,
    })
    
    # === Step 3 / Final answer extraction ===
    # Multi-latent only (single latent doesn't work well for final answer)
    q = random.choice(QUESTION_TEMPLATES["step3"])
    examples.append({
        "question": q,
        "answer": str(step3),
        "latent_indices": [0, 1, 2, 3, 4, 5],
        "task": "step3_multi",
        "ground_truth_step3": step3,
        "codi_output": codi_output,
        "codi_correct": codi_correct,
    })
    
    # === Operation detection ===
    op_name = OPERATION_NAMES[operation]
    
    # Single latent (z2)
    q = random.choice(QUESTION_TEMPLATES["operation"])
    examples.append({
        "question": q,
        "answer": op_name,
        "latent_indices": [1],
        "task": "operation_single",
        "ground_truth_operation": operation,
    })
    
    # Multi-latent
    q = random.choice(QUESTION_TEMPLATES["operation"])
    examples.append({
        "question": q,
        "answer": op_name,
        "latent_indices": [0, 1, 2, 3, 4, 5],
        "task": "operation_multi",
        "ground_truth_operation": operation,
    })
    
    # === Comparison ===
    comparison_answer = "step 2" if step2 > step1 else "step 1"
    q = random.choice(QUESTION_TEMPLATES["comparison"])
    examples.append({
        "question": q,
        "answer": comparison_answer,
        "latent_indices": [0, 1, 2, 3, 4, 5],
        "task": "comparison",
    })
    
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/problems_holdout.json",
                        help="Path to holdout problems JSON")
    parser.add_argument("--output", type=str, default="data/ao_training_holdout.jsonl",
                        help="Output JSONL file for AO training")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Max training problems to process")
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    
    train_problems = data["train_problems"]
    if args.max_problems:
        train_problems = train_problems[:args.max_problems]
    
    print(f"Processing {len(train_problems)} training problems")
    
    # Load CODI
    print("\nLoading CODI model...")
    codi = load_codi()
    
    # Process problems
    all_examples = []
    codi_correct_count = 0
    
    for problem in tqdm(train_problems, desc="Collecting latents"):
        # Get ground truth
        ground_truth = str(problem["step3"])
        
        # Collect latents and CODI's prediction
        result = codi.collect_latents(
            problem["prompt"],
            ground_truth_answer=ground_truth,
            return_hidden_states=False
        )
        
        if len(result.latent_vectors) < 6:
            continue
        
        latents = result.latent_vectors[:6]
        codi_output = result.predicted_answer
        codi_correct = result.is_correct
        
        if codi_correct:
            codi_correct_count += 1
        
        # Create training examples
        examples = create_training_examples(problem, latents, codi_output, codi_correct)
        
        # Add latent vectors to each example
        for ex in examples:
            indices = ex["latent_indices"]
            ex["latent_vectors"] = [latents[i].tolist() if hasattr(latents[i], 'tolist') 
                                     else latents[i] for i in indices]
            # Add problem metadata
            ex["problem_X"] = problem["X"]
            ex["problem_Y"] = problem["Y"]
            ex["problem_Z"] = problem["Z"]
            ex["problem_operation"] = problem["operation"]
        
        all_examples.extend(examples)
    
    print(f"\nCODI accuracy on training set: {codi_correct_count}/{len(train_problems)} "
          f"({100*codi_correct_count/len(train_problems):.1f}%)")
    print(f"Generated {len(all_examples)} training examples")
    
    # Count by task
    from collections import Counter
    task_counts = Counter(ex["task"] for ex in all_examples)
    print("\nExamples by task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
