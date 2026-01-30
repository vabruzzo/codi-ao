#!/usr/bin/env python3
"""
Generate seeded synthetic math problems for the study.

This creates the dataset used by both logit lens and linear probe experiments.
Uses fixed seed for reproducibility.
"""

import argparse
import json
import random
from pathlib import Path


ENTITIES = [
    ("team", "members", "recruits", "loses"),
    ("company", "employees", "hires", "fires"),
    ("school", "students", "enrolls", "expels"),
    ("farm", "animals", "buys", "sells"),
    ("store", "items", "stocks", "sells"),
    ("library", "books", "acquires", "donates"),
    ("garden", "plants", "grows", "removes"),
    ("zoo", "animals", "adopts", "transfers"),
    ("museum", "artifacts", "acquires", "loans"),
    ("hospital", "patients", "admits", "discharges"),
]

TEMPLATES_ADD = [
    "A {entity} has {X} {things}. They {action1} {Y} more. Then they multiply the total by {Z}. How many {things} are there now?",
    "A {entity} starts with {X} {things}. They {action1} {Y} additional {things}. Then each {thing} brings {Z} more. How many {things} total?",
    "There are {X} {things} in a {entity}. After {action1}ing {Y} more, the count is multiplied by {Z}. What is the final count?",
]

TEMPLATES_SUB = [
    "A {entity} has {X} {things}. They {action2} {Y} of them. Then each remaining {thing} produces {Z} offspring. How many {things} total?",
    "A {entity} starts with {X} {things}. After {action2}ing {Y}, each remaining {thing} is multiplied by {Z}. How many {things}?",
    "There are {X} {things} in a {entity}. After {action2}ing {Y}, the remaining count is multiplied by {Z}. What is the result?",
]

TEMPLATES_MUL = [
    "A {entity} has {X} {things}. Each {thing} produces {Y} offspring. Then {Z} more {things} join. How many {things} total?",
    "A {entity} starts with {X} {things}. After multiplying by {Y}, they add {Z} more. How many {things}?",
    "There are {X} {things} in a {entity}. The count is multiplied by {Y}, then {Z} are added. What is the final count?",
]


def generate_problem(seed: int, operation: str = None):
    """Generate a single synthetic problem with known ground truth."""
    random.seed(seed)
    
    entity, things, action1, action2 = random.choice(ENTITIES)
    thing = things[:-1] if things.endswith('s') else things
    
    X = random.randint(1, 10)
    Y = random.randint(1, 10)
    Z = random.randint(2, 10)  # At least 2 for meaningful multiplication
    
    # Choose operation if not specified
    if operation is None:
        operation = random.choice(["add", "sub", "mul"])
    
    if operation == "add":
        template = random.choice(TEMPLATES_ADD)
        step1 = X + Y
        step2 = step1 * Z
    elif operation == "sub":
        # Ensure X > Y for valid subtraction
        if X <= Y:
            X, Y = max(X, Y) + 1, min(X, Y)
        template = random.choice(TEMPLATES_SUB)
        step1 = X - Y
        step2 = step1 * Z
    else:  # mul
        template = random.choice(TEMPLATES_MUL)
        step1 = X * Y
        step2 = step1 + Z
    
    prompt = template.format(
        entity=entity,
        things=things,
        thing=thing,
        action1=action1,
        action2=action2,
        X=X, Y=Y, Z=Z
    )
    
    return {
        "seed": seed,
        "prompt": prompt,
        "X": X,
        "Y": Y, 
        "Z": Z,
        "step1": step1,
        "step2": step2,
        "operation": operation,
    }


def generate_dataset(n_samples: int, base_seed: int = 42, balanced: bool = True):
    """Generate a dataset of synthetic problems."""
    problems = []
    
    if balanced:
        # Equal split across operations
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
    parser = argparse.ArgumentParser(description="Generate synthetic math problems")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--balanced", action="store_true", default=True, help="Balance operations")
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} synthetic problems (seed={args.seed})...")
    
    problems = generate_dataset(args.n_samples, base_seed=args.seed, balanced=args.balanced)
    
    # Count operations
    op_counts = {}
    for p in problems:
        op_counts[p["operation"]] = op_counts.get(p["operation"], 0) + 1
    
    print(f"Operation distribution: {op_counts}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "n_samples": args.n_samples,
                "seed": args.seed,
                "balanced": args.balanced,
            },
            "problems": problems,
        }, f, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
