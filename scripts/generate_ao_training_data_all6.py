#!/usr/bin/env python3
"""
Generate Activation Oracle training data using ALL 6 latents for every question.

This is a variant that always passes all 6 latent vectors (z1-z6) to the AO,
to test whether having full context improves operand extraction.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.latent_qa import LatentQAExample


# =============================================================================
# QUESTION TEMPLATES (same as original)
# =============================================================================

EXTRACTION_STEP1 = [
    "What was calculated in the first step?",
    "What is the result of step 1?",
    "First calculation result?",
    "What is the intermediate calculation result?",
    "What value was computed in step 1?",
]

EXTRACTION_STEP2 = [
    "What was calculated in the second step?",
    "What is the result of step 2?",
    "Second calculation result?",
    "What is the final calculation result?",
    "What value was computed in step 2?",
]

OPERATION_DIRECT = [
    "What operation was performed in step 1?",
    "What arithmetic operation was used in the first step?",
    "Name the operation in step 1.",
    "What calculation type was used first?",
]

OPERATION_BINARY = {
    "addition": [
        "Is the first operation addition?",
        "Was addition used in step 1?",
        "Did step 1 add numbers?",
    ],
    "subtraction": [
        "Is the first operation subtraction?",
        "Was subtraction used in step 1?",
        "Did step 1 subtract?",
    ],
    "multiplication": [
        "Is the first operation multiplication?",
        "Was multiplication used in step 1?",
        "Did step 1 multiply?",
    ],
}

FIRST_OPERAND_QUESTIONS = [
    "What was the first number in the calculation?",
    "What was the first operand in step 1?",
    "What number came first in the first operation?",
]

SECOND_OPERAND_QUESTIONS = [
    "What was the second number in the calculation?",
    "What was the second operand in step 1?",
    "What number came second in the first operation?",
]

FULL_CALCULATION_QUESTIONS = [
    "What calculation was performed in step 1?",
    "Describe the full arithmetic operation for step 1.",
    "What was the complete first calculation?",
]

MAGNITUDE_QUESTIONS_STEP1 = {
    10: ["Is the step 1 result greater than 10?", "Is step 1 more than 10?"],
    50: ["Is the step 1 result greater than 50?", "Is step 1 above 50?"],
    100: ["Is the step 1 result greater than 100?", "Does step 1 exceed 100?"],
}

MAGNITUDE_QUESTIONS_STEP2 = {
    10: ["Is the step 2 result greater than 10?", "Is step 2 more than 10?"],
    50: ["Is the step 2 result greater than 50?", "Is step 2 above 50?"],
    100: ["Is the step 2 result greater than 100?", "Does step 2 exceed 100?"],
}

COMPARISON_QUESTIONS = [
    "Which calculation step produced the larger result?",
    "Is the second step result greater than the first?",
    "Which step has the bigger value?",
    "Compare the two steps: which is larger?",
]


def format_oracle_prompt(question: str, num_latents: int = 6) -> str:
    """Format prompt for the Activation Oracle with all 6 latents."""
    placeholders = " ?" * num_latents
    return f"Layer 50%:{placeholders} {question}"


def op_to_name(op: str) -> str:
    """Convert short op name to full name."""
    return {"add": "addition", "sub": "subtraction", "mul": "multiplication"}[op]


def generate_qa_from_problem(
    problem: dict,
    all_latents: list,  # All 6 latent vectors
) -> list[LatentQAExample]:
    """Generate QA examples using all 6 latents for every question."""
    examples = []
    
    step1 = problem["step1"]
    step2 = problem["step2"]
    op = problem["operation"]
    op_name = op_to_name(op)
    X = problem["X"]
    Y = problem["Y"]
    op_symbol = {"add": "+", "sub": "-", "mul": "*"}[op]
    
    # Convert all latents to lists
    latent_list = [lat.cpu().tolist() if isinstance(lat, torch.Tensor) else lat for lat in all_latents]
    positions = [0, 1, 2, 3, 4, 5]  # All 6 positions
    
    # -------------------------------------------------------------------------
    # 1. EXTRACTION - Step 1
    # -------------------------------------------------------------------------
    for q in EXTRACTION_STEP1:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(step1),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="extraction_step1",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 2. EXTRACTION - Step 2
    # -------------------------------------------------------------------------
    for q in EXTRACTION_STEP2:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(step2),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="extraction_step2",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 3. OPERATION - Direct
    # -------------------------------------------------------------------------
    for q in OPERATION_DIRECT:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=op_name,
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operation_direct",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 4. OPERATION - Binary
    # -------------------------------------------------------------------------
    for check_op, questions in OPERATION_BINARY.items():
        is_this_op = op_name == check_op
        q = random.choice(questions)
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer="yes" if is_this_op else "no",
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operation_binary",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 5. OPERAND - First (X)
    # -------------------------------------------------------------------------
    for q in FIRST_OPERAND_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(X),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operand_first",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 6. OPERAND - Second (Y)
    # -------------------------------------------------------------------------
    for q in SECOND_OPERAND_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(Y),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operand_second",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 7. FULL CALCULATION
    # -------------------------------------------------------------------------
    full_calc = f"{X} {op_symbol} {Y}"
    for q in FULL_CALCULATION_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=full_calc,
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="full_calculation",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 8. MAGNITUDE - Step 1
    # -------------------------------------------------------------------------
    for threshold, questions in MAGNITUDE_QUESTIONS_STEP1.items():
        q = random.choice(questions)
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer="yes" if step1 > threshold else "no",
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="magnitude_step1",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 9. MAGNITUDE - Step 2
    # -------------------------------------------------------------------------
    for threshold, questions in MAGNITUDE_QUESTIONS_STEP2.items():
        q = random.choice(questions)
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer="yes" if step2 > threshold else "no",
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="magnitude_step2",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 10. COMPARISON
    # -------------------------------------------------------------------------
    for q in COMPARISON_QUESTIONS:
        if "greater" in q.lower() and "second" in q.lower():
            answer = "yes" if step2 > step1 else "no"
        elif "larger" in q.lower() or "bigger" in q.lower():
            answer = "step 2" if step2 > step1 else "step 1"
        else:
            answer = "step 2" if step2 > step1 else "step 1"
        
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=answer,
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="comparison",
            source_prompt="synthetic",
            is_multi_latent=True,
        ))
    
    return examples


def load_codi_model(config_path="configs/default.yaml"):
    """Load the CODI model."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--output", type=str, default="data/ao_training_data_all6.jsonl")
    parser.add_argument("--n_problems", type=int, default=None, help="Limit problems")
    parser.add_argument("--holdout", type=int, default=200, help="Hold out last N problems for testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Activation Oracle Training Data Generation (ALL 6 LATENTS)")
    print("=" * 60)
    
    random.seed(args.seed)
    
    # Load problems
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    problems = data["problems"]
    
    # Hold out last N problems for testing
    if args.holdout > 0:
        problems = problems[:-args.holdout]
        print(f"Holding out last {args.holdout} problems for testing")
    
    if args.n_problems:
        problems = problems[:args.n_problems]
    
    print(f"Using {len(problems)} problems for training")
    
    # Count operations
    op_counts = {}
    for p in problems:
        op_counts[p["operation"]] = op_counts.get(p["operation"], 0) + 1
    print(f"Operations: {op_counts}")
    
    # Load CODI
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    # Collect latents and generate training data
    print("\nCollecting latents and generating training data...")
    all_examples = []
    
    for problem in tqdm(problems, desc="Processing problems"):
        # Collect latents from CODI
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        
        if len(result.latent_vectors) < 6:
            continue
        
        # Get all 6 latents
        all_latents = result.latent_vectors[:6]
        
        # Generate QA examples with all 6 latents
        examples = generate_qa_from_problem(problem, all_latents)
        all_examples.extend(examples)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Stats
    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    
    from collections import Counter
    by_type = Counter(ex.question_type for ex in all_examples)
    print("\nBy question type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count} ({100*count/len(all_examples):.1f}%)")
    
    print(f"\nAll examples use 6 latents (multi-latent)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    
    print(f"\nSaved to {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT: Train the Activation Oracle")
    print("=" * 60)
    print(f"python scripts/train.py --data {args.output} --output_dir checkpoints/ao_all6")


if __name__ == "__main__":
    main()
