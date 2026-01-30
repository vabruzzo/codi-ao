#!/usr/bin/env python3
"""
Phase 4 Training Data Generation.

Combines:
1. GSM8k 2-step problems (with real CODI latents)
2. Expanded synthetic problems (1-10, 1-100, novel entities, edge cases)

Collects CODI latents at generation time and creates diverse QA pairs.

Usage:
    python scripts/generate_phase4_data.py --target 50000 --output data/phase4_train.jsonl
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingExample:
    """A training example for the AO."""
    prompt: str                    # Oracle prompt (question + placeholder)
    question: str                  # The QA question
    answer: str                    # Expected answer
    latent_vectors: list           # List of latent vectors (each is list of floats)
    latent_positions: list[int]    # Which latent positions were used
    question_type: str             # extraction, classification, comparison
    source: str                    # gsm8k_2step, synthetic_expanded
    num_steps: int                 # Number of reasoning steps in original problem
    is_multi_latent: bool          # Whether this uses multiple latents


# =============================================================================
# QUESTION TEMPLATES
# =============================================================================

EXTRACTION_QUESTIONS = [
    "What is the intermediate calculation result?",
    "What value was computed?",
    "What number does this represent?",
    "Tell me the calculated value.",
    "What was the result of this step?",
    "Extract the numeric value.",
    "What is stored in this reasoning step?",
    "Decode this activation to a number.",
    "What numeric value is encoded here?",
    "What is the computed result?",
]

EXTRACTION_STEP1 = [
    "What was calculated in the first step?",
    "What is the result of step 1?",
    "First calculation result?",
    "What was computed first?",
]

EXTRACTION_STEP2 = [
    "What was calculated in the second step?",
    "What is the result of step 2?",
    "Second calculation result?",
    "What was computed in the later step?",
]

COMPARISON_QUESTIONS = [
    "Which calculation step produced the larger result?",
    "Is the second step result greater than the first?",
    "Which step has the bigger value?",
    "Compare the two steps: which is larger?",
]

OPERATION_QUESTIONS = {
    "addition": ["Is this an addition operation?", "Was addition used in this step?"],
    "subtraction": ["Is this a subtraction operation?", "Was subtraction used?"],
    "multiplication": ["Is this a multiplication operation?", "Was multiplication used?"],
    "division": ["Is this a division operation?", "Was division used?"],
}

MAGNITUDE_QUESTIONS = {
    10: ["Is the result greater than 10?", "Is this value more than 10?"],
    50: ["Is the result greater than 50?", "Is this value above 50?"],
    100: ["Is the result greater than 100?", "Does this exceed 100?"],
    1000: ["Is the result greater than 1000?", "Is this above 1000?"],
}

POSITION_QUESTIONS = [
    "Is this the first calculation step?",
    "Is this step 1?",
    "Is this one of the first calculation steps?",
]

STRUCTURE_QUESTIONS = [
    "Is this a computational step?",
    "Does this step perform a calculation?",
    "Is this step performing a calculation?",
]


def format_oracle_prompt(question: str, num_latents: int = 1) -> str:
    """Format prompt for the Activation Oracle.
    
    Uses the default placeholder token " ?" (space + question mark).
    Format: "Layer 50%: ? ? ... ? {question}"
    """
    # The placeholder token is " ?" and they are concatenated after ":"
    # So for 2 latents: "Layer 50%: ? ? {question}"
    placeholders = " ?" * num_latents
    return f"Layer 50%:{placeholders} {question}"


def generate_qa_from_latents(
    problem_prompt: str,
    step1_result: int,
    step2_result: int,
    step1_op: str,
    step2_op: str,
    latent_z1: list,  # Latent vector for step 1 (index 1)
    latent_z3: list,  # Latent vector for step 2 (index 3)
    source: str,
    num_steps: int = 2,
) -> list[TrainingExample]:
    """Generate diverse QA examples from collected latents."""
    examples = []
    
    # -------------------------------------------------------------------------
    # 1. SINGLE-LATENT EXTRACTION - Step 1
    # -------------------------------------------------------------------------
    for q in EXTRACTION_QUESTIONS + EXTRACTION_STEP1:
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer=str(step1_result),
            latent_vectors=[latent_z1],
            latent_positions=[1],
            question_type="extraction",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 2. SINGLE-LATENT EXTRACTION - Step 2
    # -------------------------------------------------------------------------
    for q in EXTRACTION_QUESTIONS + EXTRACTION_STEP2:
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer=str(step2_result),
            latent_vectors=[latent_z3],
            latent_positions=[3],
            question_type="extraction",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 3. CLASSIFICATION - Operation (Step 1)
    # -------------------------------------------------------------------------
    for op, questions in OPERATION_QUESTIONS.items():
        is_this_op = step1_op == op
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if is_this_op else "no",
            latent_vectors=[latent_z1],
            latent_positions=[1],
            question_type="classification_operation",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 4. CLASSIFICATION - Operation (Step 2)
    # -------------------------------------------------------------------------
    for op, questions in OPERATION_QUESTIONS.items():
        is_this_op = step2_op == op
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if is_this_op else "no",
            latent_vectors=[latent_z3],
            latent_positions=[3],
            question_type="classification_operation",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 5. CLASSIFICATION - Magnitude
    # -------------------------------------------------------------------------
    for threshold, questions in MAGNITUDE_QUESTIONS.items():
        # Step 1
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if step1_result > threshold else "no",
            latent_vectors=[latent_z1],
            latent_positions=[1],
            question_type="classification_magnitude",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
        # Step 2
        q = random.choice(questions)
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes" if step2_result > threshold else "no",
            latent_vectors=[latent_z3],
            latent_positions=[3],
            question_type="classification_magnitude",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 6. CLASSIFICATION - Position
    # -------------------------------------------------------------------------
    for q in POSITION_QUESTIONS:
        # Step 1 - is first
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes",
            latent_vectors=[latent_z1],
            latent_positions=[1],
            question_type="classification_position",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
        # Step 2 - not first
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="no",
            latent_vectors=[latent_z3],
            latent_positions=[3],
            question_type="classification_position",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 7. CLASSIFICATION - Structure
    # -------------------------------------------------------------------------
    for q in STRUCTURE_QUESTIONS:
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 1),
            question=q,
            answer="yes",  # All our latents are computational steps
            latent_vectors=[latent_z1],
            latent_positions=[1],
            question_type="classification_structure",
            source=source,
            num_steps=num_steps,
            is_multi_latent=False,
        ))
    
    # -------------------------------------------------------------------------
    # 8. MULTI-LATENT - Comparison
    # -------------------------------------------------------------------------
    for q in COMPARISON_QUESTIONS:
        if "greater" in q.lower() or "larger" in q.lower():
            if "second" in q.lower():
                answer = "yes" if step2_result > step1_result else "no"
            else:
                answer = "step 2" if step2_result > step1_result else "step 1"
        else:
            answer = "step 2" if step2_result > step1_result else "step 1"
        
        examples.append(TrainingExample(
            prompt=format_oracle_prompt(q, 2),
            question=q,
            answer=answer,
            latent_vectors=[latent_z1, latent_z3],
            latent_positions=[1, 3],
            question_type="comparison",
            source=source,
            num_steps=num_steps,
            is_multi_latent=True,
        ))
    
    # -------------------------------------------------------------------------
    # 9. MULTI-LATENT - Step extraction
    # -------------------------------------------------------------------------
    q = "What was calculated in the first step?"
    examples.append(TrainingExample(
        prompt=format_oracle_prompt(q, 2),
        question=q,
        answer=str(step1_result),
        latent_vectors=[latent_z1, latent_z3],
        latent_positions=[1, 3],
        question_type="multi_extraction_step1",
        source=source,
        num_steps=num_steps,
        is_multi_latent=True,
    ))
    
    q = "What was the result of the second calculation?"
    examples.append(TrainingExample(
        prompt=format_oracle_prompt(q, 2),
        question=q,
        answer=str(step2_result),
        latent_vectors=[latent_z1, latent_z3],
        latent_positions=[1, 3],
        question_type="multi_extraction_step2",
        source=source,
        num_steps=num_steps,
        is_multi_latent=True,
    ))
    
    return examples


def load_gsm8k_2step(data_dir: str = "data/gsm8k_2step") -> list[dict]:
    """Load GSM8k 2-step problems."""
    train_path = Path(data_dir) / "gsm8k_train.json"
    if not train_path.exists():
        print(f"  Warning: {train_path} not found")
        return []
    
    with open(train_path) as f:
        data = json.load(f)
    
    # Convert to our format
    problems = []
    for item in data:
        if item["num_steps"] != 2 or len(item["steps"]) < 2:
            continue
        
        problems.append({
            "prompt": item["question"] + " Give the answer only and nothing else.",
            "step1_result": int(float(item["steps"][0]["result"])),
            "step2_result": int(float(item["steps"][1]["result"])),
            "step1_op": item["steps"][0]["operation"],
            "step2_op": item["steps"][1]["operation"],
            "source": "gsm8k_2step",
            "final_answer": item["final_answer"],
        })
    
    return problems


def load_synthetic_expanded(data_dir: str = "data/synthetic_expanded") -> list[dict]:
    """Load expanded synthetic problems."""
    problems_path = Path(data_dir) / "problems.json"
    if not problems_path.exists():
        print(f"  Warning: {problems_path} not found")
        return []
    
    with open(problems_path) as f:
        data = json.load(f)
    
    # Convert to our format
    problems = []
    for item in data:
        problems.append({
            "prompt": item["prompt"],
            "step1_result": item["step1_result"],
            "step2_result": item["step2_result"],
            "step1_op": item["step1_operation"],
            "step2_op": item["step2_operation"],
            "source": f"synthetic_{item['number_range']}",
            "final_answer": item["final_answer"],
        })
    
    return problems


def collect_latents_and_generate(
    wrapper,
    problems: list[dict],
    target_count: int,
    seed: int = 42,
) -> list[TrainingExample]:
    """Collect latents from CODI and generate training examples."""
    random.seed(seed)
    
    all_examples = []
    problems_used = 0
    
    # Shuffle problems
    random.shuffle(problems)
    
    pbar = tqdm(problems, desc="Collecting latents")
    for problem in pbar:
        if len(all_examples) >= target_count:
            break
        
        # Collect latents from CODI
        try:
            result = wrapper.collect_latents(
                prompt=problem["prompt"],
                ground_truth_answer=str(problem["final_answer"]),
            )
        except Exception as e:
            continue
        
        if len(result.latent_vectors) < 6:
            continue
        
        # Get z1 (index 1) and z3 (index 3) latents
        latent_z1 = result.latent_vectors[1].cpu().tolist()
        latent_z3 = result.latent_vectors[3].cpu().tolist()
        
        # Generate QA examples
        examples = generate_qa_from_latents(
            problem_prompt=problem["prompt"],
            step1_result=problem["step1_result"],
            step2_result=problem["step2_result"],
            step1_op=problem["step1_op"],
            step2_op=problem["step2_op"],
            latent_z1=latent_z1,
            latent_z3=latent_z3,
            source=problem["source"],
            num_steps=2,
        )
        
        all_examples.extend(examples)
        problems_used += 1
        
        pbar.set_postfix({
            "problems": problems_used,
            "examples": len(all_examples),
        })
    
    return all_examples[:target_count]


def save_training_data(examples: list[TrainingExample], output_path: str):
    """Save as JSONL for training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    
    print(f"\nSaved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 4 training data")
    parser.add_argument("--gsm8k_dir", type=str, default="data/gsm8k_2step")
    parser.add_argument("--synthetic_dir", type=str, default="data/synthetic_expanded")
    parser.add_argument("--output", type=str, default="data/phase4_train.jsonl")
    parser.add_argument("--target", type=int, default=50000, help="Target number of examples")
    parser.add_argument("--gsm8k_ratio", type=float, default=0.3, help="Fraction from GSM8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 4 Training Data Generation")
    print("=" * 60)
    
    # Load problems
    print("\nLoading problems...")
    gsm8k_problems = load_gsm8k_2step(args.gsm8k_dir)
    print(f"  GSM8k 2-step: {len(gsm8k_problems)} problems")
    
    synthetic_problems = load_synthetic_expanded(args.synthetic_dir)
    print(f"  Synthetic expanded: {len(synthetic_problems)} problems")
    
    if not gsm8k_problems and not synthetic_problems:
        print("ERROR: No problems found! Run parse_gsm8k.py and generate_expanded_synthetic.py first.")
        return
    
    # Calculate targets
    n_gsm8k = int(args.target * args.gsm8k_ratio)
    n_synthetic = args.target - n_gsm8k
    
    # Adjust if we don't have enough problems
    # Each problem generates ~80 QA examples
    examples_per_problem = 80
    needed_gsm8k = n_gsm8k // examples_per_problem + 1
    needed_synthetic = n_synthetic // examples_per_problem + 1
    
    print(f"\nTarget: {args.target} examples ({n_gsm8k} GSM8k + {n_synthetic} synthetic)")
    print(f"Need ~{needed_gsm8k} GSM8k problems, have {len(gsm8k_problems)}")
    print(f"Need ~{needed_synthetic} synthetic problems, have {len(synthetic_problems)}")
    
    # Load CODI model
    print("\nLoading CODI model...")
    from src.codi_wrapper import CODIWrapper
    wrapper = CODIWrapper.from_pretrained(device=args.device)
    
    # Collect and generate
    print("\nGenerating GSM8k examples...")
    random.seed(args.seed)
    gsm8k_examples = collect_latents_and_generate(
        wrapper,
        gsm8k_problems[:needed_gsm8k * 2],  # Sample more in case some fail
        target_count=n_gsm8k,
        seed=args.seed,
    )
    
    print(f"\nGenerated {len(gsm8k_examples)} GSM8k examples")
    
    print("\nGenerating synthetic examples...")
    synthetic_examples = collect_latents_and_generate(
        wrapper,
        synthetic_problems[:needed_synthetic * 2],
        target_count=n_synthetic,
        seed=args.seed + 1,
    )
    
    print(f"\nGenerated {len(synthetic_examples)} synthetic examples")
    
    # Combine and shuffle
    all_examples = gsm8k_examples + synthetic_examples
    random.shuffle(all_examples)
    
    # Stats
    print("\n" + "=" * 60)
    print("FINAL DATASET STATS")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    
    from collections import Counter
    
    by_source = Counter(ex.source for ex in all_examples)
    print("\nBy source:")
    for s, count in sorted(by_source.items()):
        print(f"  {s}: {count} ({100*count/len(all_examples):.1f}%)")
    
    by_type = Counter(ex.question_type for ex in all_examples)
    print("\nBy question type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count} ({100*count/len(all_examples):.1f}%)")
    
    multi = sum(1 for ex in all_examples if ex.is_multi_latent)
    print(f"\nSingle-latent: {len(all_examples) - multi} ({100*(len(all_examples)-multi)/len(all_examples):.1f}%)")
    print(f"Multi-latent: {multi} ({100*multi/len(all_examples):.1f}%)")
    
    # Save
    save_training_data(all_examples, args.output)
    
    print("\n" + "=" * 60)
    print("READY FOR TRAINING")
    print("=" * 60)
    print(f"\nRun:")
    print(f"  python scripts/train.py --data {args.output} --output checkpoints/ao_phase4 --epochs 2")


if __name__ == "__main__":
    main()
