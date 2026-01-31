#!/usr/bin/env python3
"""
Generate Activation Oracle training data from synthetic problems.

Creates diverse QA pairs:
1. Extraction - numeric values from latents
2. Operation classification - add/sub/mul detection
3. Magnitude classification - threshold comparisons
4. Multi-latent comparison - which step is larger

Uses seeded synthetic problems, collects CODI latents, generates training data.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingExample:
    """A training example for the AO."""
    prompt: str
    question: str
    answer: str
    latent_vectors: list
    latent_positions: list
    question_type: str
    source: str
    is_multi_latent: bool


# =============================================================================
# QUESTION TEMPLATES
# =============================================================================

EXTRACTION_QUESTIONS = [
    "What is the intermediate calculation result?",
    "What value was computed?",
    "What number does this represent?",
    "What was the result of this step?",
    "Extract the numeric value.",
    "What numeric value is encoded here?",
    "What is the computed result?",
]

EXTRACTION_STEP1 = [
    "What was calculated in the first step?",
    "What is the result of step 1?",
    "First calculation result?",
]

EXTRACTION_STEP2 = [
    "What was calculated in the second step?",
    "What is the result of step 2?",
    "Second calculation result?",
]

# Direct operation questions (what we want to test)
OPERATION_DIRECT = [
    "What operation was performed?",
    "What arithmetic operation was used?",
    "Name the operation in this step.",
    "What calculation type is this?",
]

OPERATION_BINARY = {
    "addition": [
        "Is this an addition operation?",
        "Was addition used in this step?",
        "Did this step add numbers?",
    ],
    "subtraction": [
        "Is this a subtraction operation?",
        "Was subtraction used?",
        "Did this step subtract?",
    ],
    "multiplication": [
        "Is this a multiplication operation?",
        "Was multiplication used?",
        "Did this step multiply?",
    ],
}

MAGNITUDE_QUESTIONS = {
    10: ["Is the result greater than 10?", "Is this value more than 10?"],
    20: ["Is the result greater than 20?", "Does this exceed 20?"],
    50: ["Is the result greater than 50?", "Is this value above 50?"],
    100: ["Is the result greater than 100?", "Does this exceed 100?"],
}

COMPARISON_QUESTIONS = [
    "Which calculation step produced the larger result?",
    "Is the second step result greater than the first?",
    "Which step has the bigger value?",
    "Compare the two steps: which is larger?",
]


def format_oracle_prompt(question: str, num_latents: int = 1) -> str:
    """Format prompt for the Activation Oracle."""
    placeholders = " ?" * num_latents
    return f"Layer 50%:{placeholders} {question}"


def op_to_name(op: str) -> str:
    """Convert short op name to full name."""
    return {"add": "addition", "sub": "subtraction", "mul": "multiplication"}[op]


def generate_qa_from_problem(
    problem: dict,
    latent_z2: list,  # Step 1 result (position 1, 0-indexed)
    latent_z4: list,  # Step 2 result (position 3, 0-indexed)
) -> list[TrainingExample]:
    """Generate diverse QA examples from a problem with collected latents."""
    examples = []
    
    step1 = problem["step1"]
    step2 = problem["step2"]
    op = problem["operation"]
    op_name = op_to_name(op)
    
    # -------------------------------------------------------------------------
    # 1. EXTRACTION - Step 1 (from z2)
    # -------------------------------------------------------------------------
    for q in EXTRACTION_QUESTIONS + EXTRACTION_STEP1:
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer=str(step1),
            latent_vectors=[latent_z2],
            latent_positions=[1],  # z2 is index 1
            question_type="extraction_step1",
            source="synthetic",
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 2. EXTRACTION - Step 2 (from z4)
    # -------------------------------------------------------------------------
    for q in EXTRACTION_QUESTIONS + EXTRACTION_STEP2:
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer=str(step2),
            latent_vectors=[latent_z4],
            latent_positions=[3],  # z4 is index 3
            question_type="extraction_step2",
            source="synthetic",
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 3. OPERATION - Direct question (Step 1 only, since step 2 is always mul)
    # -------------------------------------------------------------------------
    for q in OPERATION_DIRECT:
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer=op_name,
            latent_vectors=[latent_z2],
            latent_positions=[1],
            question_type="operation_direct",
            source="synthetic",
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 4. OPERATION - Binary questions (is this X?)
    # -------------------------------------------------------------------------
    for check_op, questions in OPERATION_BINARY.items():
        is_this_op = op_name == check_op
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if is_this_op else "no",
            latent_vectors=[latent_z2],
            latent_positions=[1],
            question_type="operation_binary",
            source="synthetic",
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 5. MAGNITUDE - Step 1
    # -------------------------------------------------------------------------
    for threshold, questions in MAGNITUDE_QUESTIONS.items():
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if step1 > threshold else "no",
            latent_vectors=[latent_z2],
            latent_positions=[1],
            question_type="magnitude_step1",
            source="synthetic",
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 6. MAGNITUDE - Step 2
    # -------------------------------------------------------------------------
    for threshold, questions in MAGNITUDE_QUESTIONS.items():
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if step2 > threshold else "no",
            latent_vectors=[latent_z4],
            latent_positions=[3],
            question_type="magnitude_step2",
            source="synthetic",
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 7. MULTI-LATENT - Comparison
    # -------------------------------------------------------------------------
    for q in COMPARISON_QUESTIONS:
        if "greater" in q.lower() and "second" in q.lower():
            answer = "yes" if step2 > step1 else "no"
        elif "larger" in q.lower() or "bigger" in q.lower():
            answer = "step 2" if step2 > step1 else "step 1"
        else:
            answer = "step 2" if step2 > step1 else "step 1"
        
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 2),
            question=q,
            answer=answer,
            latent_vectors=[latent_z2, latent_z4],
            latent_positions=[1, 3],
            question_type="comparison",
            source="synthetic",
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 8. MULTI-LATENT - Step extraction with both latents
    # -------------------------------------------------------------------------
    q = "Given both steps, what was calculated first?"
    examples.append(TrainingExample(
        prompt=format_oracle_prompt(q, 2),
        question=q,
        answer=str(step1),
        latent_vectors=[latent_z2, latent_z4],
        latent_positions=[1, 3],
        question_type="multi_extraction_step1",
        source="synthetic",
        is_multi_latent=True,
    ))
    
    q = "Given both steps, what was calculated second?"
    examples.append(TrainingExample(
        prompt=format_oracle_prompt(q, 2),
        question=q,
        answer=str(step2),
        latent_vectors=[latent_z2, latent_z4],
        latent_positions=[1, 3],
        question_type="multi_extraction_step2",
        source="synthetic",
        is_multi_latent=True,
    ))
    
    return examples


def load_codi_model(config_path="configs/default.yaml"):
    """Load the CODI model."""
    from src.codi_wrapper import CODIWrapper
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = "cuda"
    
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
    parser.add_argument("--output", type=str, default="data/ao_training_data.jsonl")
    parser.add_argument("--n_problems", type=int, default=None, help="Limit problems")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Activation Oracle Training Data Generation")
    print("=" * 60)
    
    random.seed(args.seed)
    
    # Load problems
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    problems = data["problems"]
    
    if args.n_problems:
        problems = problems[:args.n_problems]
    
    print(f"Using {len(problems)} problems")
    
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
        
        # Get z2 (index 1) and z4 (index 3)
        latent_z2 = result.latent_vectors[1].cpu().tolist()
        latent_z4 = result.latent_vectors[3].cpu().tolist()
        
        # Generate QA examples
        examples = generate_qa_from_problem(problem, latent_z2, latent_z4)
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
    
    multi = sum(1 for ex in all_examples if ex.is_multi_latent)
    print(f"\nSingle-latent: {len(all_examples) - multi}")
    print(f"Multi-latent: {multi}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    
    print(f"\nSaved to {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT: Train the Activation Oracle")
    print("=" * 60)
    print(f"python scripts/train.py --data {args.output} --output checkpoints/ao_study")


if __name__ == "__main__":
    main()
