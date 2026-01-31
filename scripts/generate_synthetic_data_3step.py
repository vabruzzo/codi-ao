#!/usr/bin/env python3
"""
Generate 3-step synthetic math problems matching the LessWrong paper's setup.

Problem structure:
- Step 1: X op Y = step1 (first intermediate)
- Step 2: step1 * Z = step2 (second intermediate)
- Step 3: step1 + step2 = final_answer

Example (from LessWrong):
"A team starts with 3 members. They recruit 5 new members. 
Then each current member recruits 2 additional people. 
How many people are there now on the team?"

Step 1: 3 + 5 = 8
Step 2: 8 * 2 = 16
Step 3: 8 + 16 = 24
"""

import argparse
import json
import random
from pathlib import Path


# Entity types with natural language variations
ENTITIES = [
    {"entity": "team", "things": "members", "thing": "member"},
    {"entity": "company", "things": "employees", "thing": "employee"},
    {"entity": "school", "things": "students", "thing": "student"},
    {"entity": "farm", "things": "animals", "thing": "animal"},
    {"entity": "store", "things": "items", "thing": "item"},
    {"entity": "library", "things": "books", "thing": "book"},
    {"entity": "garden", "things": "plants", "thing": "plant"},
    {"entity": "zoo", "things": "animals", "thing": "animal"},
    {"entity": "club", "things": "members", "thing": "member"},
    {"entity": "group", "things": "people", "thing": "person"},
]

# Templates for ADDITION (Step 1: X + Y)
# Structure: start with X, add Y, then each brings Z more, ask for total
TEMPLATES_ADD = [
    "A {entity} starts with {X} {things}. They recruit {Y} new {things}. Then each current {thing} recruits {Z} additional people. How many {things} are there now in total?",
    "A {entity} has {X} {things}. They add {Y} more. Then each {thing} brings in {Z} new ones. What is the total count?",
    "There are {X} {things} in a {entity}. After {Y} more join, each {thing} invites {Z} others. How many {things} are there in the end?",
    "A {entity} begins with {X} {things}. {Y} additional {things} arrive. Then every {thing} recruits {Z} more. What is the final total?",
    "Starting with {X} {things}, a {entity} gains {Y} more. Each {thing} then brings {Z} friends. How many {things} total?",
]

# Templates for SUBTRACTION (Step 1: X - Y)
# Structure: start with X, lose Y, then remaining multiply by Z, ask for total
TEMPLATES_SUB = [
    "A {entity} starts with {X} {things}. They lose {Y} of them. Then each remaining {thing} recruits {Z} new ones. How many {things} are there in total?",
    "A {entity} has {X} {things}. After {Y} leave, each remaining {thing} brings {Z} more. What is the total count?",
    "There are {X} {things} in a {entity}. {Y} depart, then each remaining {thing} invites {Z} others. How many {things} are there in the end?",
    "A {entity} begins with {X} {things}. {Y} are removed. Then every remaining {thing} recruits {Z} more. What is the final total?",
    "Starting with {X} {things}, a {entity} loses {Y}. Each remaining {thing} then brings {Z} friends. How many {things} total?",
]

# Templates for MULTIPLICATION (Step 1: X * Y)
# Structure: X groups of Y each, then each brings Z more, ask for total
TEMPLATES_MUL = [
    "A {entity} has {X} groups with {Y} {things} each. Then each {thing} recruits {Z} new ones. How many {things} are there in total?",
    "There are {X} teams of {Y} {things} in a {entity}. Each {thing} then brings {Z} more. What is the total count?",
    "A {entity} starts with {X} sets of {Y} {things}. Every {thing} invites {Z} others. How many {things} are there in the end?",
    "In a {entity}, there are {X} clusters with {Y} {things} each. Then each {thing} recruits {Z} more. What is the final total?",
    "A {entity} begins with {X} batches of {Y} {things}. Each {thing} then brings {Z} friends. How many {things} total?",
]


def generate_problem(seed: int, operation: str = None):
    """
    Generate a single 3-step problem with known ground truth.
    
    Structure:
    - Step 1: X op Y = step1
    - Step 2: step1 * Z = step2
    - Step 3: step1 + step2 = final_answer
    """
    random.seed(seed)
    
    entity_info = random.choice(ENTITIES)
    entity = entity_info["entity"]
    things = entity_info["things"]
    thing = entity_info["thing"]
    
    X = random.randint(2, 10)
    Y = random.randint(2, 10)
    Z = random.randint(2, 6)  # Keep Z smaller to avoid huge numbers
    
    # Choose operation if not specified
    if operation is None:
        operation = random.choice(["add", "sub", "mul"])
    
    if operation == "add":
        template = random.choice(TEMPLATES_ADD)
        step1 = X + Y
    elif operation == "sub":
        # Ensure X > Y for valid subtraction with reasonable result
        if X <= Y:
            X, Y = max(X, Y) + random.randint(1, 3), min(X, Y)
        template = random.choice(TEMPLATES_SUB)
        step1 = X - Y
    else:  # mul
        # Keep X and Y smaller for multiplication to avoid huge step1
        X = random.randint(2, 6)
        Y = random.randint(2, 6)
        template = random.choice(TEMPLATES_MUL)
        step1 = X * Y
    
    # Step 2: step1 * Z (always multiplication)
    step2 = step1 * Z
    
    # Step 3: step1 + step2 = final answer (always addition)
    final_answer = step1 + step2
    
    prompt = template.format(
        entity=entity,
        things=things,
        thing=thing,
        X=X, Y=Y, Z=Z
    )
    
    return {
        "seed": seed,
        "prompt": prompt,
        "X": X,
        "Y": Y, 
        "Z": Z,
        "step1": step1,           # X op Y
        "step2": step2,           # step1 * Z
        "step3": final_answer,    # step1 + step2
        "final_answer": final_answer,
        "operation": operation,   # Operation for step1 (add/sub/mul)
        "op_step2": "mul",        # Always multiplication
        "op_step3": "add",        # Always addition
    }


def generate_dataset(n_samples: int, base_seed: int = 42, balanced: bool = True):
    """Generate a dataset of 3-step synthetic problems."""
    problems = []
    
    if balanced:
        # Equal split across operations (for step 1)
        ops = ["add", "sub", "mul"]
        per_op = n_samples // 3
        remainder = n_samples % 3
        
        idx = 0
        for i, op in enumerate(ops):
            count = per_op + (1 if i < remainder else 0)
            for j in range(count):
                seed = base_seed + idx
                problems.append(generate_problem(seed, operation=op))
                idx += 1
    else:
        # Random operations
        for i in range(n_samples):
            seed = base_seed + i
            problems.append(generate_problem(seed))
    
    # Shuffle with fixed seed
    random.seed(base_seed)
    random.shuffle(problems)
    
    return problems


def main():
    parser = argparse.ArgumentParser(description="Generate 3-step synthetic math problems")
    parser.add_argument("--n_samples", type=int, default=1200, help="Number of problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/synthetic_problems_3step.json")
    parser.add_argument("--balanced", action="store_true", default=True, help="Balance operations")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating 3-Step Synthetic Math Problems")
    print("=" * 60)
    print(f"\nStructure:")
    print("  Step 1: X op Y = step1")
    print("  Step 2: step1 * Z = step2")
    print("  Step 3: step1 + step2 = final_answer")
    
    print(f"\nGenerating {args.n_samples} problems (seed={args.seed})...")
    
    problems = generate_dataset(args.n_samples, base_seed=args.seed, balanced=args.balanced)
    
    # Count operations
    op_counts = {}
    for p in problems:
        op_counts[p["operation"]] = op_counts.get(p["operation"], 0) + 1
    
    print(f"Step 1 operation distribution: {op_counts}")
    
    # Show example
    print("\n--- Example Problem ---")
    ex = problems[0]
    print(f"Prompt: {ex['prompt']}")
    print(f"X={ex['X']}, Y={ex['Y']}, Z={ex['Z']}")
    print(f"Step 1: {ex['X']} {ex['operation']} {ex['Y']} = {ex['step1']}")
    print(f"Step 2: {ex['step1']} * {ex['Z']} = {ex['step2']}")
    print(f"Step 3: {ex['step1']} + {ex['step2']} = {ex['final_answer']}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "n_samples": args.n_samples,
                "seed": args.seed,
                "balanced": args.balanced,
                "structure": "3-step",
                "step1_ops": ["add", "sub", "mul"],
                "step2_op": "mul",
                "step3_op": "add",
            },
            "problems": problems,
        }, f, indent=2)
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
