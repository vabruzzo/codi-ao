#!/usr/bin/env python3
"""
Explore what CODI actually stores in latents for GSM8k problems.

Before training AO on GSM8k, we need to understand:
1. What values does CODI store in z2/z4 for real problems?
2. Do they match our parsed intermediate steps?
3. Are z2/z4 positions even meaningful for variable-step problems?

This script runs logit lens on GSM8k and compares with parsed ground truth.
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_gsm8k_steps(answer_text: str) -> list[str]:
    """Extract intermediate values from GSM8k answer."""
    pattern = r"<<[^=]+=([^>]+)>>"
    matches = re.findall(pattern, answer_text)
    return [m.strip().replace(",", "") for m in matches]


def parse_final_answer(answer_text: str) -> str:
    """Extract final answer after ####."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def explore_gsm8k_latents(n_samples: int = 20, seed: int = 42):
    """Explore what CODI stores in latents for GSM8k."""
    from datasets import load_dataset
    from src.codi_wrapper import CODIWrapper
    
    print("Loading CODI model...")
    wrapper = CODIWrapper.from_pretrained()
    
    print("Loading GSM8k dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    print(f"\n{'='*80}")
    print("EXPLORING GSM8K LATENTS")
    print(f"{'='*80}\n")
    
    matches = {"z2": 0, "z4": 0, "any": 0}
    total = 0
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        question = item["question"]
        answer_text = item["answer"]
        
        # Parse ground truth
        parsed_steps = parse_gsm8k_steps(answer_text)
        final_answer = parse_final_answer(answer_text)
        
        if not parsed_steps or not final_answer:
            continue
        
        # Format prompt like CODI expects
        prompt = f"{question} Give the answer only and nothing else."
        
        # Collect latents
        result = wrapper.collect_latents(prompt, ground_truth_answer=final_answer)
        
        if len(result.latent_vectors) < 6:
            continue
        
        # Run logit lens on all 6 positions
        ll_results = {}
        for pos in range(6):
            ll_result = wrapper.logit_lens(result.latent_vectors[pos])
            top_token, prob = ll_result.get_top1_at_final_layer()
            ll_results[pos] = (top_token.strip(), prob)
        
        # Check which parsed steps appear in which latent positions
        found_in = {step: [] for step in parsed_steps}
        for pos, (token, prob) in ll_results.items():
            for step in parsed_steps:
                if token == step:
                    found_in[step].append(f"z{pos}")
        
        # Print detailed info
        print(f"\n[{i+1}] Problem: {question[:80]}...")
        print(f"    Parsed steps: {parsed_steps}")
        print(f"    Final answer: {final_answer}")
        print(f"    CODI output: {result.predicted_answer} ({'✓' if result.is_correct else '✗'})")
        print(f"    Logit lens at each position:")
        for pos, (token, prob) in ll_results.items():
            marker = ""
            if token in parsed_steps:
                marker = f" ← matches step {parsed_steps.index(token)+1}!"
            elif token == final_answer:
                marker = " ← matches FINAL ANSWER!"
            print(f"      z{pos}: '{token}' (prob={prob:.3f}){marker}")
        
        # Track matches
        total += 1
        z2_token = ll_results[1][0]  # z2 is index 1
        z4_token = ll_results[3][0]  # z4 is index 3
        
        if len(parsed_steps) >= 1 and z2_token == parsed_steps[0]:
            matches["z2"] += 1
        if len(parsed_steps) >= 2 and z4_token == parsed_steps[1]:
            matches["z4"] += 1
        
        # Check if ANY latent contains ANY parsed step
        any_match = any(len(positions) > 0 for positions in found_in.values())
        if any_match:
            matches["any"] += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total problems analyzed: {total}")
    print(f"z2 matches step 1: {matches['z2']}/{total} ({100*matches['z2']/total:.1f}%)")
    print(f"z4 matches step 2: {matches['z4']}/{total} ({100*matches['z4']/total:.1f}%)")
    print(f"Any latent matches any step: {matches['any']}/{total} ({100*matches['any']/total:.1f}%)")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
If z2/z4 don't consistently match parsed steps, it means:
1. CODI uses latents differently for GSM8k than our synthetic templates
2. We can't just use parsed step values as ground truth
3. We need a different strategy for GSM8k training

Options:
A) Use logit lens output as pseudo-labels (self-supervised)
B) Only train on problems where z2/z4 clearly match steps
C) Skip GSM8k and focus on expanded synthetic data
D) Use GSM8k only for evaluation, not training
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore GSM8k latents")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    explore_gsm8k_latents(args.n_samples, args.seed)
