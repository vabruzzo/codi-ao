#!/usr/bin/env python3
"""
Generate expanded synthetic math dataset for AO training.

This expands on the original synthetic templates with:
1. Multiple number ranges (1-10, 1-100)
2. More entity types (zoo, library, hospital, etc.)
3. More operations and templates
4. Edge cases built into training data
5. Diverse question phrasings

The goal is to make the AO more robust to OOD inputs.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# =============================================================================
# TEMPLATES - Expanded from original 20 to 50+
# =============================================================================

# Original entities
ORIGINAL_ENTITIES = [
    ("team", "members", "recruit"),
    ("club", "members", "invite"),
    ("school", "students", "bring"),
    ("company", "employees", "hire"),
    ("group", "participants", "add"),
    ("band", "musicians", "add"),
    ("restaurant", "customers", "welcome"),
    ("gym", "members", "refer"),
    ("community", "residents", "invite"),
]

# Novel entities for OOD robustness
NOVEL_ENTITIES = [
    ("zoo", "animals", "attract"),
    ("library", "books", "acquire"),
    ("hospital", "patients", "admit"),
    ("farm", "crops", "plant"),
    ("orchestra", "musicians", "recruit"),
    ("museum", "visitors", "attract"),
    ("store", "products", "stock"),
    ("park", "visitors", "attract"),
    ("theater", "seats", "add"),
    ("fleet", "vehicles", "acquire"),
]

# All entities combined
ALL_ENTITIES = ORIGINAL_ENTITIES + NOVEL_ENTITIES

# Addition templates
ADDITION_TEMPLATES = [
    "A {entity} starts with {x} {unit}. They {action} {y} more {unit}. Then each current {unit_singular} brings {z} additional {unit}. How many {unit} are there now in the {entity}?",
    "A {entity} has {x} {unit}. {y} new {unit} join. Each {unit_singular} then invites {z} others. What is the total number of {unit}?",
    "There are {x} {unit} in a {entity}. After adding {y} more {unit}, each one brings in {z} additional {unit}. How many {unit} total?",
    "Starting with {x} {unit}, a {entity} gains {y} more. Then every {unit_singular} recruits {z} more. Final count?",
]

# Subtraction templates
SUBTRACTION_TEMPLATES = [
    "A {entity} starts with {x} {unit}. {y} {unit} leave. Then each remaining {unit_singular} brings {z} additional {unit}. How many {unit} are there now?",
    "A {entity} has {x} {unit}. After {y} {unit} depart, each remaining {unit_singular} invites {z} others. What is the total?",
    "There are {x} {unit} in a {entity}. {y} leave, and each remaining one brings {z} more. Total {unit}?",
    "Starting with {x} {unit}, {y} are removed from a {entity}. Each remaining {unit_singular} then recruits {z}. Final count?",
]


@dataclass
class SyntheticProblem:
    """A synthetic math problem with ground truth."""
    id: str
    prompt: str
    step1_result: int
    step2_result: int
    final_answer: int
    step1_operation: str
    step2_operation: str
    num_steps: int
    x: int
    y: int
    z: int
    entity_type: str
    number_range: str  # "1-10", "1-100", etc.
    template_type: str  # "addition", "subtraction"
    is_edge_case: bool


def generate_problem(
    x: int,
    y: int,
    z: int,
    entity_info: tuple,
    template: str,
    template_type: str,
    number_range: str,
    problem_id: str,
    is_edge_case: bool = False,
) -> Optional[SyntheticProblem]:
    """Generate a single synthetic problem."""
    entity, unit, action = entity_info
    unit_singular = unit.rstrip("s") if unit.endswith("s") else unit
    
    # Format the prompt
    prompt = template.format(
        entity=entity,
        unit=unit,
        unit_singular=unit_singular,
        action=action,
        x=x,
        y=y,
        z=z,
    )
    prompt += " Give the answer only and nothing else."
    
    # Calculate ground truth
    if template_type == "addition":
        step1 = x + y
        step1_op = "addition"
    else:  # subtraction
        step1 = x - y
        step1_op = "subtraction"
        if step1 <= 0:
            return None  # Skip invalid problems
    
    step2 = step1 * z
    step2_op = "multiplication"
    
    final = step1 + step2
    
    return SyntheticProblem(
        id=problem_id,
        prompt=prompt,
        step1_result=step1,
        step2_result=step2,
        final_answer=final,
        step1_operation=step1_op,
        step2_operation=step2_op,
        num_steps=2,
        x=x,
        y=y,
        z=z,
        entity_type=entity,
        number_range=number_range,
        template_type=template_type,
        is_edge_case=is_edge_case,
    )


def generate_standard_problems(
    n_per_range: int,
    seed: int = 42,
) -> list[SyntheticProblem]:
    """Generate standard problems across multiple number ranges."""
    random.seed(seed)
    problems = []
    problem_id = 0
    
    # Number ranges to include
    ranges = [
        ("1-10", 1, 10),
        ("1-100", 1, 100),
    ]
    
    for range_name, min_val, max_val in ranges:
        for _ in range(n_per_range):
            # Random entity
            entity_info = random.choice(ALL_ENTITIES)
            
            # Random template and type
            if random.random() < 0.6:  # 60% addition
                template = random.choice(ADDITION_TEMPLATES)
                template_type = "addition"
                x = random.randint(min_val, max_val)
                y = random.randint(min_val, max_val)
            else:  # 40% subtraction
                template = random.choice(SUBTRACTION_TEMPLATES)
                template_type = "subtraction"
                x = random.randint(min_val + 2, max_val)  # Ensure x > y possible
                y = random.randint(min_val, x - 1)
            
            z = random.randint(min_val, max_val)
            
            problem = generate_problem(
                x=x, y=y, z=z,
                entity_info=entity_info,
                template=template,
                template_type=template_type,
                number_range=range_name,
                problem_id=f"synthetic_{problem_id}",
            )
            
            if problem:
                problems.append(problem)
                problem_id += 1
    
    return problems


def generate_edge_cases(seed: int = 42) -> list[SyntheticProblem]:
    """Generate edge case problems for robustness."""
    random.seed(seed)
    problems = []
    
    edge_cases = [
        # (x, y, z, description)
        (1, 0, 5, "step1_is_1"),
        (5, 2, 1, "step2_equals_step1"),
        (10, 10, 10, "large_result"),
        (5, 5, 5, "all_same"),
        (1, 0, 1, "both_steps_1"),
        (2, 0, 1, "minimal"),
        (1, 1, 1, "tiny_numbers"),
        (10, 9, 10, "step1_is_1_subtract"),
        (100, 50, 2, "round_numbers"),
        (7, 7, 7, "sevens"),
    ]
    
    for i, (x, y, z, desc) in enumerate(edge_cases):
        entity_info = random.choice(ORIGINAL_ENTITIES)
        
        # Use addition for edge cases with y=0, subtraction otherwise
        if y == 0:
            template = random.choice(ADDITION_TEMPLATES)
            template_type = "addition"
        else:
            template = random.choice(ADDITION_TEMPLATES if random.random() < 0.5 else SUBTRACTION_TEMPLATES)
            template_type = "addition" if "more" in template else "subtraction"
        
        problem = generate_problem(
            x=x, y=y, z=z,
            entity_info=entity_info,
            template=template,
            template_type=template_type,
            number_range="edge_case",
            problem_id=f"edge_{desc}",
            is_edge_case=True,
        )
        
        if problem:
            problems.append(problem)
    
    return problems


def generate_qa_examples(problems: list[SyntheticProblem]) -> list[dict]:
    """Generate QA training examples from synthetic problems."""
    examples = []
    
    # Extraction question templates (diverse phrasings)
    extraction_generic = [
        "What is the intermediate calculation result?",
        "What value was computed?",
        "What number does this represent?",
        "Tell me the calculated value.",
        "What was the result of this step?",
        "Extract the number from this computation.",
        "What is stored in this reasoning step?",
        "Decode this activation to a number.",
        "What numeric value is encoded here?",
        "What is the computed result?",
    ]
    
    extraction_step1 = [
        "What was calculated in the first step?",
        "What is the result of step 1?",
        "First calculation result?",
        "What was computed first?",
        "Initial calculation value?",
    ]
    
    extraction_step2 = [
        "What was calculated in the second step?",
        "What is the result of step 2?",
        "Second calculation result?",
        "What was computed second?",
        "Later calculation value?",
    ]
    
    for p in problems:
        # Step 1 extraction
        for template in extraction_generic + extraction_step1:
            examples.append({
                "problem_id": p.id,
                "prompt": p.prompt,
                "num_steps": p.num_steps,
                "step_number": 1,
                "latent_position": "z2",
                "qa_question": template,
                "qa_answer": str(p.step1_result),
                "qa_type": "extraction",
                "operation": p.step1_operation,
                "number_range": p.number_range,
                "is_edge_case": p.is_edge_case,
            })
        
        # Step 2 extraction
        for template in extraction_generic + extraction_step2:
            examples.append({
                "problem_id": p.id,
                "prompt": p.prompt,
                "num_steps": p.num_steps,
                "step_number": 2,
                "latent_position": "z4",
                "qa_question": template,
                "qa_answer": str(p.step2_result),
                "qa_type": "extraction",
                "operation": p.step2_operation,
                "number_range": p.number_range,
                "is_edge_case": p.is_edge_case,
            })
        
        # Operation classification for step 1
        for op in ["addition", "subtraction", "multiplication", "division"]:
            is_this_op = p.step1_operation == op
            examples.append({
                "problem_id": p.id,
                "prompt": p.prompt,
                "num_steps": p.num_steps,
                "step_number": 1,
                "latent_position": "z2",
                "qa_question": f"Is this step performing {op}?",
                "qa_answer": "yes" if is_this_op else "no",
                "qa_type": "classification_operation",
                "operation": p.step1_operation,
                "number_range": p.number_range,
                "is_edge_case": p.is_edge_case,
            })
        
        # Operation classification for step 2
        for op in ["addition", "subtraction", "multiplication", "division"]:
            is_this_op = p.step2_operation == op
            examples.append({
                "problem_id": p.id,
                "prompt": p.prompt,
                "num_steps": p.num_steps,
                "step_number": 2,
                "latent_position": "z4",
                "qa_question": f"Is this step performing {op}?",
                "qa_answer": "yes" if is_this_op else "no",
                "qa_type": "classification_operation",
                "operation": p.step2_operation,
                "number_range": p.number_range,
                "is_edge_case": p.is_edge_case,
            })
        
        # Magnitude classification
        for step_num, result, latent in [(1, p.step1_result, "z2"), (2, p.step2_result, "z4")]:
            for threshold in [10, 50, 100, 1000]:
                examples.append({
                    "problem_id": p.id,
                    "prompt": p.prompt,
                    "num_steps": p.num_steps,
                    "step_number": step_num,
                    "latent_position": latent,
                    "qa_question": f"Is the result greater than {threshold}?",
                    "qa_answer": "yes" if result > threshold else "no",
                    "qa_type": "classification_magnitude",
                    "operation": p.step1_operation if step_num == 1 else p.step2_operation,
                    "number_range": p.number_range,
                    "is_edge_case": p.is_edge_case,
                })
        
        # Position classification
        for step_num, latent in [(1, "z2"), (2, "z4")]:
            examples.append({
                "problem_id": p.id,
                "prompt": p.prompt,
                "num_steps": p.num_steps,
                "step_number": step_num,
                "latent_position": latent,
                "qa_question": "Is this the first calculation step?",
                "qa_answer": "yes" if step_num == 1 else "no",
                "qa_type": "classification_position",
                "operation": p.step1_operation if step_num == 1 else p.step2_operation,
                "number_range": p.number_range,
                "is_edge_case": p.is_edge_case,
            })
    
    return examples


def print_stats(problems: list[SyntheticProblem]):
    """Print dataset statistics."""
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}\n")
    
    print(f"Total problems: {len(problems)}")
    
    # By number range
    by_range = {}
    for p in problems:
        by_range[p.number_range] = by_range.get(p.number_range, 0) + 1
    print("\nBy number range:")
    for r, count in sorted(by_range.items()):
        print(f"  {r}: {count}")
    
    # By template type
    by_type = {}
    for p in problems:
        by_type[p.template_type] = by_type.get(p.template_type, 0) + 1
    print("\nBy template type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")
    
    # By entity
    by_entity = {}
    for p in problems:
        by_entity[p.entity_type] = by_entity.get(p.entity_type, 0) + 1
    print("\nBy entity type:")
    for e, count in sorted(by_entity.items()):
        print(f"  {e}: {count}")
    
    # Edge cases
    n_edge = sum(1 for p in problems if p.is_edge_case)
    print(f"\nEdge cases: {n_edge}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate expanded synthetic math data")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_expanded",
                        help="Output directory")
    parser.add_argument("--n_per_range", type=int, default=10000,
                        help="Number of problems per number range")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--generate_qa", action="store_true",
                        help="Also generate QA training examples")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate problems
    print("Generating standard problems...")
    standard_problems = generate_standard_problems(args.n_per_range, args.seed)
    
    print("Generating edge cases...")
    edge_cases = generate_edge_cases(args.seed)
    
    all_problems = standard_problems + edge_cases
    
    # Print stats
    print_stats(all_problems)
    
    # Print examples
    print(f"\n{'='*70}")
    print("EXAMPLE PROBLEMS")
    print(f"{'='*70}\n")
    for p in all_problems[:5]:
        print(f"ID: {p.id}")
        print(f"Prompt: {p.prompt[:100]}...")
        print(f"Step 1: {p.step1_result} ({p.step1_operation})")
        print(f"Step 2: {p.step2_result} ({p.step2_operation})")
        print(f"Final: {p.final_answer}")
        print(f"Range: {p.number_range}, Entity: {p.entity_type}")
        print("-" * 50)
    
    # Save problems
    problems_path = output_dir / "problems.json"
    with open(problems_path, "w") as f:
        json.dump([asdict(p) for p in all_problems], f, indent=2)
    print(f"\nSaved {len(all_problems)} problems to {problems_path}")
    
    # Generate QA examples if requested
    if args.generate_qa:
        print("\nGenerating QA training examples...")
        qa_examples = generate_qa_examples(all_problems)
        
        qa_path = output_dir / "qa_examples.json"
        with open(qa_path, "w") as f:
            json.dump(qa_examples, f, indent=2)
        print(f"Saved {len(qa_examples)} QA examples to {qa_path}")
    
    print("\nDone!")
