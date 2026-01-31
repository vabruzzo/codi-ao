#!/usr/bin/env python3
"""
Generate Activation Oracle training data for 3-step problems.

Uses ALL 6 latents and includes questions about:
- Step 1 (z2): First intermediate result (X op Y)
- Step 2 (z4): Second intermediate result (step1 * Z)
- Step 3 (z6): Final answer (step1 + step2)
- Operations at each step
- Operands
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
# QUESTION TEMPLATES
# =============================================================================

# Step 1 extraction (z2)
STEP1_QUESTIONS = [
    "What was calculated in the first step?",
    "What is the result of step 1?",
    "What is the first intermediate value?",
    "What was the first calculation result?",
]

# Step 2 extraction (z4)
STEP2_QUESTIONS = [
    "What was calculated in the second step?",
    "What is the result of step 2?",
    "What is the second intermediate value?",
    "What was the second calculation result?",
]

# Step 3 / Final answer extraction (z6)
STEP3_QUESTIONS = [
    "What is the final answer?",
    "What was calculated in the third step?",
    "What is the result of step 3?",
    "What is the total?",
    "What is the final result?",
]

# Operation questions (for step 1)
OPERATION_STEP1_QUESTIONS = [
    "What operation was performed in step 1?",
    "What arithmetic operation was used in the first step?",
    "What calculation type was used first?",
]

# Operand questions (for step 1)
FIRST_OPERAND_QUESTIONS = [
    "What was the first number in step 1?",
    "What was the first operand in the first calculation?",
]

SECOND_OPERAND_QUESTIONS = [
    "What was the second number in step 1?",
    "What was the second operand in the first calculation?",
]

# Comparison questions
COMPARISON_QUESTIONS = [
    "Which step produced a larger result: step 1 or step 2?",
    "Is step 2's result greater than step 1's result?",
    "Compare the first and second intermediate values.",
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
    """Generate QA examples for a 3-step problem using all 6 latents."""
    examples = []
    
    step1 = problem["step1"]
    step2 = problem["step2"]
    step3 = problem["step3"]  # = final_answer
    op1 = problem["operation"]  # Step 1 operation
    op1_name = op_to_name(op1)
    X = problem["X"]
    Y = problem["Y"]
    Z = problem["Z"]
    op1_symbol = {"add": "+", "sub": "-", "mul": "*"}[op1]
    
    # Convert all latents to lists
    latent_list = [lat.cpu().tolist() if isinstance(lat, torch.Tensor) else lat for lat in all_latents]
    positions = [0, 1, 2, 3, 4, 5]  # All 6 positions
    
    # -------------------------------------------------------------------------
    # 1. STEP 1 EXTRACTION (target: step1)
    # -------------------------------------------------------------------------
    for q in STEP1_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(step1),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="extraction_step1",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 2. STEP 2 EXTRACTION (target: step2)
    # -------------------------------------------------------------------------
    for q in STEP2_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(step2),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="extraction_step2",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 3. STEP 3 / FINAL ANSWER EXTRACTION (target: step3 = final_answer)
    # -------------------------------------------------------------------------
    for q in STEP3_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(step3),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="extraction_step3",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 4. OPERATION DETECTION (Step 1)
    # -------------------------------------------------------------------------
    for q in OPERATION_STEP1_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=op1_name,
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operation_step1",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 5. OPERAND EXTRACTION (Step 1)
    # -------------------------------------------------------------------------
    for q in FIRST_OPERAND_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(X),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operand_first",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    for q in SECOND_OPERAND_QUESTIONS:
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=str(Y),
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="operand_second",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 6. FULL CALCULATION (Step 1)
    # -------------------------------------------------------------------------
    full_calc = f"{X} {op1_symbol} {Y}"
    examples.append(LatentQAExample(
        prompt=format_oracle_prompt("What calculation was performed in step 1?", 6),
        question="What calculation was performed in step 1?",
        answer=full_calc,
        latent_vectors=latent_list,
        latent_positions=positions,
        question_type="full_calculation_step1",
        source_prompt="synthetic_3step",
        is_multi_latent=True,
    ))
    
    # -------------------------------------------------------------------------
    # 7. COMPARISON
    # -------------------------------------------------------------------------
    for q in COMPARISON_QUESTIONS:
        if "greater" in q.lower():
            answer = "yes" if step2 > step1 else "no"
        else:
            answer = "step 2" if step2 > step1 else "step 1"
        
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer=answer,
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="comparison",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 8. MAGNITUDE QUESTIONS
    # -------------------------------------------------------------------------
    for threshold in [10, 50, 100]:
        q = f"Is the step 1 result greater than {threshold}?"
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer="yes" if step1 > threshold else "no",
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="magnitude_step1",
            source_prompt="synthetic_3step",
            is_multi_latent=True,
        ))
        
        q = f"Is the final answer greater than {threshold}?"
        examples.append(LatentQAExample(
            prompt=format_oracle_prompt(q, 6),
            question=q,
            answer="yes" if step3 > threshold else "no",
            latent_vectors=latent_list,
            latent_positions=positions,
            question_type="magnitude_step3",
            source_prompt="synthetic_3step",
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
    parser.add_argument("--problems", type=str, default="data/synthetic_problems_3step.json")
    parser.add_argument("--output", type=str, default="data/ao_training_data_3step.jsonl")
    parser.add_argument("--n_problems", type=int, default=None, help="Limit problems")
    parser.add_argument("--holdout", type=int, default=200, help="Hold out last N problems for testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("3-Step AO Training Data Generation")
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
    print(f"Step 1 operations: {op_counts}")
    
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
    print(f"uv run python scripts/train.py --data {args.output} --output_dir checkpoints/ao_3step --epochs 2 --batch_size 16")


if __name__ == "__main__":
    main()
