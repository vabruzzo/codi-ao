#!/usr/bin/env python3
"""
Analyze AO performance on rare vs common values from training.
This helps determine if AO is memorizing or truly reading latents.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def generate_problems_for_analysis(n, seed):
    """Regenerate problems to get value frequencies."""
    random.seed(seed)
    problems = []
    for i in range(n):
        X = random.randint(2, 10)
        Y = random.randint(2, 10)
        Z = random.randint(2, 6)
        op = random.choice(['add', 'sub', 'mul'])
        
        if op == 'add':
            step1 = X + Y
        elif op == 'sub':
            if X <= Y:
                X, Y = max(X, Y) + random.randint(1, 3), min(X, Y)
                X = min(X, 10)
            step1 = X - Y
        else:
            X = random.randint(2, 6)
            Y = random.randint(2, 6)
            step1 = X * Y
        
        step2 = step1 * Z
        step3 = step1 + step2
        
        problems.append({
            'step1': step1, 'step2': step2, 'step3': step3,
            'X': X, 'Y': Y, 'Z': Z, 'op': op
        })
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True, help="Path to problems JSON")
    parser.add_argument("--n_train", type=int, default=1000, help="Number of training problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for generation")
    args = parser.parse_args()
    
    # Load actual problems
    with open(args.problems) as f:
        all_problems = json.load(f)
    
    train = all_problems[:args.n_train]
    test = all_problems[args.n_train:]
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    # Count frequencies in training
    train_step1_counts = Counter(p['step1'] for p in train)
    train_step3_counts = Counter(p['step3'] for p in train)
    
    # Define rare thresholds
    RARE_STEP1_THRESHOLD = 15
    RARE_STEP3_THRESHOLD = 10
    
    rare_step1_values = {v for v, c in train_step1_counts.items() if c <= RARE_STEP1_THRESHOLD}
    rare_step3_values = {v for v, c in train_step3_counts.items() if c <= RARE_STEP3_THRESHOLD}
    common_step1_values = {v for v, c in train_step1_counts.items() if c >= 50}
    common_step3_values = {v for v, c in train_step3_counts.items() if c >= 20}
    
    print(f"\nRare step1 values (≤{RARE_STEP1_THRESHOLD} in train): {sorted(rare_step1_values)}")
    print(f"Common step1 values (≥50 in train): {sorted(common_step1_values)}")
    print(f"\nRare step3 values (≤{RARE_STEP3_THRESHOLD} in train): {len(rare_step3_values)} values")
    print(f"Common step3 values (≥20 in train): {sorted(common_step3_values)}")
    
    # Categorize test problems
    test_rare_step1 = [(i, p) for i, p in enumerate(test) if p['step1'] in rare_step1_values]
    test_common_step1 = [(i, p) for i, p in enumerate(test) if p['step1'] in common_step1_values]
    test_rare_step3 = [(i, p) for i, p in enumerate(test) if p['step3'] in rare_step3_values]
    test_common_step3 = [(i, p) for i, p in enumerate(test) if p['step3'] in common_step3_values]
    
    print(f"\n=== Test Set Breakdown ===")
    print(f"Problems with rare step1: {len(test_rare_step1)}")
    print(f"Problems with common step1: {len(test_common_step1)}")
    print(f"Problems with rare step3: {len(test_rare_step3)}")
    print(f"Problems with common step3: {len(test_common_step3)}")
    
    # Output indices for use in evaluation
    print(f"\n=== Test Indices for Rare Values ===")
    print(f"Rare step1 test indices: {[i for i, p in test_rare_step1]}")
    print(f"Rare step3 test indices: {[i for i, p in test_rare_step3][:20]}...")  # First 20
    
    # Show some examples
    print(f"\n=== Example Problems with Rare Step3 ===")
    for i, p in test_rare_step3[:5]:
        train_count = train_step3_counts.get(p['step3'], 0)
        print(f"  Test #{i}: step1={p['step1']}, step2={p['step2']}, step3={p['step3']} (seen {train_count}x in train)")


if __name__ == "__main__":
    main()
