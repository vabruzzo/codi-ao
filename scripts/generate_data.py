#!/usr/bin/env python3
"""
Generate 3-step math problems with rigorous holdout controls.

Holdout strategies:
1. Step value holdout: Random step1/step2/step3 values never seen in training
2. Tuple holdout: Specific (X, Y, Z) combinations never seen in training

This allows direct measurement of memorization vs generalization.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# Templates from LessWrong CODI repo
ADDITION_TEMPLATES = [
    "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team?",
    "A company starts with {X} employees. They hire {Y} more employees. Then each current employee brings in {Z} additional people. How many people are there now in the company?",
    "A school starts with {X} students. They enroll {Y} new students. Then each current student brings {Z} additional students. How many students are there now in the school?",
    "A gym starts with {X} members. They sign up {Y} new members. Then each current member refers {Z} additional people. How many people are there now in the gym?",
    "A club starts with {X} members. They add {Y} new members. Then each current member invites {Z} additional people. How many people are there now in the club?",
]

SUBTRACTION_TEMPLATES = [
    "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team?",
    "A company starts with {X} employees. {Y} employees resign. Then each remaining employee brings in {Z} additional people. How many people are there now in the company?",
    "A school starts with {X} students. {Y} students transfer out. Then each remaining student brings {Z} additional students. How many students are there now in the school?",
    "A gym starts with {X} members. {Y} members cancel. Then each remaining member refers {Z} additional people. How many people are there now in the gym?",
    "A club starts with {X} members. {Y} members quit. Then each remaining member invites {Z} additional people. How many people are there now in the club?",
]

MULTIPLICATION_TEMPLATES = [
    "A team starts with {X} groups of {Y} members each. Then each current member recruits {Z} additional people. How many people are there now on the team?",
    "A company starts with {X} departments of {Y} employees each. Then each current employee brings in {Z} additional people. How many people are there now in the company?",
    "A school starts with {X} classes of {Y} students each. Then each current student brings {Z} additional students. How many students are there now in the school?",
    "A gym starts with {X} sessions of {Y} members each. Then each current member refers {Z} additional people. How many people are there now in the gym?",
    "A club starts with {X} teams of {Y} members each. Then each current member invites {Z} additional people. How many people are there now in the club?",
]


def compute_steps(X, Y, Z, operation):
    """Compute step1, step2, step3 for a problem."""
    if operation == "add":
        step1 = X + Y
    elif operation == "sub":
        step1 = X - Y
    else:  # mul
        step1 = X * Y
    
    step2 = step1 * Z
    step3 = step1 + step2
    return step1, step2, step3


def generate_problem(X, Y, Z, operation, template_idx=0):
    """Generate a single 3-step problem."""
    step1, step2, step3 = compute_steps(X, Y, Z, operation)
    
    if operation == "add":
        templates = ADDITION_TEMPLATES
    elif operation == "sub":
        templates = SUBTRACTION_TEMPLATES
    else:
        templates = MULTIPLICATION_TEMPLATES
    
    template = templates[template_idx % len(templates)]
    prompt = template.format(X=X, Y=Y, Z=Z)
    
    return {
        "prompt": prompt,
        "X": X,
        "Y": Y,
        "Z": Z,
        "operation": operation,
        "step1": step1,
        "step2": step2,
        "step3": step3,
    }


def get_all_valid_combinations(x_range, y_range, z_range, operations):
    """Generate all valid (X, Y, Z, op) combinations."""
    combos = []
    for X in range(x_range[0], x_range[1] + 1):
        for Y in range(y_range[0], y_range[1] + 1):
            for Z in range(z_range[0], z_range[1] + 1):
                for op in operations:
                    # Validate
                    if op == "sub" and X <= Y:
                        continue
                    if op == "mul" and (X > 6 or Y > 6):
                        continue  # Keep multiplication small
                    combos.append((X, Y, Z, op))
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--value_holdout_ratio", type=float, default=0.15,
                        help="Fraction of unique step values to hold out")
    parser.add_argument("--tuple_holdout_ratio", type=float, default=0.10,
                        help="Fraction of (X,Y,Z,op) tuples to hold out entirely")
    parser.add_argument("--operand_swap_ratio", type=float, default=0.15,
                        help="Fraction of test using same (X,Y) as train but different operation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/problems.json")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Value ranges
    X_RANGE = (2, 10)
    Y_RANGE = (2, 10)
    Z_RANGE = (2, 6)
    OPERATIONS = ["add", "sub", "mul"]
    
    # Get all valid combinations
    all_combos = get_all_valid_combinations(X_RANGE, Y_RANGE, Z_RANGE, OPERATIONS)
    print(f"Total valid (X,Y,Z,op) combinations: {len(all_combos)}")
    
    # Compute all possible step values
    all_step1 = set()
    all_step2 = set()
    all_step3 = set()
    combo_to_steps = {}
    
    for combo in all_combos:
        X, Y, Z, op = combo
        s1, s2, s3 = compute_steps(X, Y, Z, op)
        all_step1.add(s1)
        all_step2.add(s2)
        all_step3.add(s3)
        combo_to_steps[combo] = (s1, s2, s3)
    
    print(f"\nPossible step1 values: {len(all_step1)} unique ({min(all_step1)}-{max(all_step1)})")
    print(f"Possible step2 values: {len(all_step2)} unique ({min(all_step2)}-{max(all_step2)})")
    print(f"Possible step3 values: {len(all_step3)} unique ({min(all_step3)}-{max(all_step3)})")
    
    # === TUPLE HOLDOUT ===
    # Randomly select (X,Y,Z,op) tuples to hold out entirely
    n_tuple_holdout = max(1, int(len(all_combos) * args.tuple_holdout_ratio))
    holdout_tuples = set(random.sample(all_combos, n_tuple_holdout))
    train_combos = [c for c in all_combos if c not in holdout_tuples]
    
    print(f"\n=== Tuple Holdout ===")
    print(f"Held-out tuples: {len(holdout_tuples)}")
    print(f"Training tuples: {len(train_combos)}")
    
    # === VALUE HOLDOUT ===
    # From the training combos, compute which step values are possible
    train_step1 = set()
    train_step2 = set()
    train_step3 = set()
    for combo in train_combos:
        s1, s2, s3 = combo_to_steps[combo]
        train_step1.add(s1)
        train_step2.add(s2)
        train_step3.add(s3)
    
    # Hold out some values from training
    n_holdout_s1 = max(1, int(len(train_step1) * args.value_holdout_ratio))
    n_holdout_s2 = max(1, int(len(train_step2) * args.value_holdout_ratio))
    n_holdout_s3 = max(1, int(len(train_step3) * args.value_holdout_ratio))
    
    holdout_step1 = set(random.sample(sorted(train_step1), n_holdout_s1))
    holdout_step2 = set(random.sample(sorted(train_step2), n_holdout_s2))
    holdout_step3 = set(random.sample(sorted(train_step3), n_holdout_s3))
    
    print(f"\n=== Value Holdout ===")
    print(f"Held-out step1 ({len(holdout_step1)}): {sorted(holdout_step1)}")
    print(f"Held-out step2 ({len(holdout_step2)}): {sorted(holdout_step2)[:15]}...")
    print(f"Held-out step3 ({len(holdout_step3)}): {sorted(holdout_step3)[:15]}...")
    
    # === GENERATE TRAINING PROBLEMS ===
    # Exclude: held-out tuples AND problems with held-out values
    train_problems = []
    train_value_counts = {"step1": Counter(), "step2": Counter(), "step3": Counter()}
    
    # Shuffle training combos
    random.shuffle(train_combos)
    
    for combo in train_combos:
        if len(train_problems) >= args.n_train:
            break
            
        X, Y, Z, op = combo
        s1, s2, s3 = combo_to_steps[combo]
        
        # Skip if any value is in holdout
        if s1 in holdout_step1 or s2 in holdout_step2 or s3 in holdout_step3:
            continue
        
        template_idx = random.randint(0, 4)
        problem = generate_problem(X, Y, Z, op, template_idx)
        train_problems.append(problem)
        
        train_value_counts["step1"][s1] += 1
        train_value_counts["step2"][s2] += 1
        train_value_counts["step3"][s3] += 1
    
    # If we need more, generate with repetition (different templates)
    while len(train_problems) < args.n_train:
        combo = random.choice(train_combos)
        X, Y, Z, op = combo
        s1, s2, s3 = combo_to_steps[combo]
        
        if s1 in holdout_step1 or s2 in holdout_step2 or s3 in holdout_step3:
            continue
        
        template_idx = random.randint(0, 4)
        problem = generate_problem(X, Y, Z, op, template_idx)
        train_problems.append(problem)
        
        train_value_counts["step1"][s1] += 1
        train_value_counts["step2"][s2] += 1
        train_value_counts["step3"][s3] += 1
    
    print(f"\nGenerated {len(train_problems)} training problems")
    print(f"Unique step1 in train: {len(train_value_counts['step1'])}")
    print(f"Unique step3 in train: {len(train_value_counts['step3'])}")
    
    # === TRACK (X, Y) -> operations used in training ===
    # For the "same operands, different operation" test
    train_xy_to_ops = defaultdict(set)
    for p in train_problems:
        train_xy_to_ops[(p["X"], p["Y"])].add(p["operation"])
    
    # Find (X, Y) pairs that could have a DIFFERENT operation in test
    # (i.e., train used add, test could use sub or mul)
    swappable_xy = []
    for (X, Y), ops_used in train_xy_to_ops.items():
        possible_ops = set()
        if X + Y <= 20:  # reasonable add result
            possible_ops.add("add")
        if X > Y:  # valid subtraction
            possible_ops.add("sub")
        if X <= 6 and Y <= 6:  # reasonable multiplication
            possible_ops.add("mul")
        
        # Operations NOT used in training for this (X, Y)
        unused_ops = possible_ops - ops_used
        if unused_ops:
            swappable_xy.append((X, Y, list(unused_ops), list(ops_used)))
    
    print(f"\n=== Operand Swap Candidates ===")
    print(f"(X,Y) pairs with unused operations: {len(swappable_xy)}")
    
    # === GENERATE TEST PROBLEMS ===
    # Categories:
    # 1. novel_tuple: From held-out (X,Y,Z,op) tuples
    # 2. novel_value: Has at least one held-out step value
    # 3. seen: All values seen in training
    # 4. operand_swap: Same (X,Y) as training, but DIFFERENT operation
    
    test_novel_tuple = []
    test_novel_value = []
    test_seen = []
    test_operand_swap = []
    
    target_per_category = args.n_test // 4  # Now 4 categories
    
    # Generate from held-out tuples
    holdout_tuple_list = list(holdout_tuples)
    random.shuffle(holdout_tuple_list)
    
    for combo in holdout_tuple_list:
        if len(test_novel_tuple) >= target_per_category:
            break
        X, Y, Z, op = combo
        s1, s2, s3 = combo_to_steps[combo]
        
        template_idx = random.randint(0, 4)
        problem = generate_problem(X, Y, Z, op, template_idx)
        
        # Tag the problem
        problem["holdout_type"] = "tuple"
        problem["novel_tuple"] = True
        problem["novel_step1"] = s1 in holdout_step1
        problem["novel_step2"] = s2 in holdout_step2
        problem["novel_step3"] = s3 in holdout_step3
        problem["novel_any_value"] = problem["novel_step1"] or problem["novel_step2"] or problem["novel_step3"]
        problem["operand_swap"] = False
        
        test_novel_tuple.append(problem)
    
    # Generate with held-out VALUES (but from training tuples)
    attempts = 0
    while len(test_novel_value) < target_per_category and attempts < 10000:
        attempts += 1
        combo = random.choice(train_combos)
        X, Y, Z, op = combo
        s1, s2, s3 = combo_to_steps[combo]
        
        has_novel = (s1 in holdout_step1) or (s2 in holdout_step2) or (s3 in holdout_step3)
        if not has_novel:
            continue
        
        template_idx = random.randint(0, 4)
        problem = generate_problem(X, Y, Z, op, template_idx)
        
        problem["holdout_type"] = "value"
        problem["novel_tuple"] = False
        problem["novel_step1"] = s1 in holdout_step1
        problem["novel_step2"] = s2 in holdout_step2
        problem["novel_step3"] = s3 in holdout_step3
        problem["novel_any_value"] = True
        problem["operand_swap"] = False
        
        test_novel_value.append(problem)
    
    # Generate SEEN problems (all values in training)
    attempts = 0
    while len(test_seen) < target_per_category and attempts < 10000:
        attempts += 1
        combo = random.choice(train_combos)
        X, Y, Z, op = combo
        s1, s2, s3 = combo_to_steps[combo]
        
        # Must have NO novel values
        if s1 in holdout_step1 or s2 in holdout_step2 or s3 in holdout_step3:
            continue
        
        template_idx = random.randint(0, 4)
        problem = generate_problem(X, Y, Z, op, template_idx)
        
        problem["holdout_type"] = "seen"
        problem["novel_tuple"] = False
        problem["novel_step1"] = False
        problem["novel_step2"] = False
        problem["novel_step3"] = False
        problem["novel_any_value"] = False
        problem["operand_swap"] = False
        
        test_seen.append(problem)
    
    # Generate OPERAND SWAP problems: same (X,Y) as training, DIFFERENT operation
    # This tests if AO reads operation info vs memorizing (X,Y) -> output
    random.shuffle(swappable_xy)
    
    for X, Y, unused_ops, train_ops_used in swappable_xy:
        if len(test_operand_swap) >= target_per_category:
            break
        
        # Pick an unused operation
        new_op = random.choice(unused_ops)
        Z = random.randint(2, 6)
        
        s1, s2, s3 = compute_steps(X, Y, Z, new_op)
        
        template_idx = random.randint(0, 4)
        problem = generate_problem(X, Y, Z, new_op, template_idx)
        
        problem["holdout_type"] = "operand_swap"
        problem["novel_tuple"] = False  # (X,Y) seen, but different op
        problem["novel_step1"] = s1 in holdout_step1
        problem["novel_step2"] = s2 in holdout_step2
        problem["novel_step3"] = s3 in holdout_step3
        problem["novel_any_value"] = problem["novel_step1"] or problem["novel_step2"] or problem["novel_step3"]
        problem["operand_swap"] = True
        problem["train_ops_for_xy"] = train_ops_used  # What operations train used for this (X,Y)
        
        test_operand_swap.append(problem)
    
    # Combine and shuffle
    test_problems = test_novel_tuple + test_novel_value + test_seen + test_operand_swap
    random.shuffle(test_problems)
    
    print(f"\n=== Test Set ===")
    print(f"Total: {len(test_problems)}")
    print(f"  Novel tuple (held-out X,Y,Z,op): {len(test_novel_tuple)}")
    print(f"  Novel value (held-out step values): {len(test_novel_value)}")
    print(f"  Seen (all values in training): {len(test_seen)}")
    print(f"  Operand swap (same X,Y, different op): {len(test_operand_swap)}")
    
    # Detailed breakdown
    n_novel_s1 = sum(1 for p in test_problems if p.get("novel_step1"))
    n_novel_s2 = sum(1 for p in test_problems if p.get("novel_step2"))
    n_novel_s3 = sum(1 for p in test_problems if p.get("novel_step3"))
    print(f"\n  With novel step1: {n_novel_s1}")
    print(f"  With novel step2: {n_novel_s2}")
    print(f"  With novel step3: {n_novel_s3}")
    
    # Operation balance
    train_ops = Counter(p["operation"] for p in train_problems)
    test_ops = Counter(p["operation"] for p in test_problems)
    print(f"\n=== Operation Balance ===")
    print(f"Train: {dict(train_ops)}")
    print(f"Test:  {dict(test_ops)}")
    
    # === SAVE ===
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "config": {
            "n_train": len(train_problems),
            "n_test": len(test_problems),
            "value_holdout_ratio": args.value_holdout_ratio,
            "tuple_holdout_ratio": args.tuple_holdout_ratio,
            "seed": args.seed,
            "x_range": X_RANGE,
            "y_range": Y_RANGE,
            "z_range": Z_RANGE,
            "operations": OPERATIONS,
        },
        "holdout": {
            "tuples": [list(t) for t in sorted(holdout_tuples)],
            "step1_values": sorted(holdout_step1),
            "step2_values": sorted(holdout_step2),
            "step3_values": sorted(holdout_step3),
        },
        "train_value_frequencies": {
            "step1": dict(train_value_counts["step1"]),
            "step2": dict(train_value_counts["step2"]),
            "step3": dict(train_value_counts["step3"]),
        },
        "train_problems": train_problems,
        "test_problems": test_problems,
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
