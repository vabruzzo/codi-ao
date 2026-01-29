#!/usr/bin/env python3
"""
Phase 3 Training Data Generation Script.

Generates 100K+ diverse training examples using real CODI latent vectors.

Question types:
1. Extraction (generic and position-aware)
2. Classification (magnitude, operation, position, structure)
3. Comparison (multi-latent)
4. Operation identification

Usage:
    # Generate 100K examples from synthetic prompts
    python scripts/generate_phase3_data.py --n_prompts 10000 --target 100000
    
    # Generate from GSM8k
    python scripts/generate_phase3_data.py --use_gsm8k --n_prompts 5000 --target 100000
    
    # Quick test
    python scripts/generate_phase3_data.py --n_prompts 100 --target 1000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 training data")
    parser.add_argument("--n_prompts", type=int, default=10000, 
                        help="Number of unique prompts to generate")
    parser.add_argument("--target", type=int, default=100000,
                        help="Target number of training examples")
    parser.add_argument("--output", type=str, default="data/phase3_train.jsonl",
                        help="Output file path")
    parser.add_argument("--use_gsm8k", action="store_true",
                        help="Use GSM8k dataset instead of synthetic")
    parser.add_argument("--gsm8k_split", type=str, default="train",
                        help="GSM8k split to use (train/test)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    import random
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Phase 3 Training Data Generation")
    print("=" * 60)
    print(f"Target examples: {args.target:,}")
    print(f"Prompts to use: {args.n_prompts:,}")
    print(f"Output: {args.output}")
    print()
    
    # Load CODI model
    print("Loading CODI model...")
    from src.codi_wrapper import CODIWrapper
    wrapper = CODIWrapper.from_pretrained(device=args.device)
    
    # Generate or load prompts
    if args.use_gsm8k:
        print(f"\nLoading GSM8k {args.gsm8k_split} set...")
        from scripts.collect_latents import load_gsm8k_prompts
        prompts = load_gsm8k_prompts(args.n_prompts, split=args.gsm8k_split, seed=args.seed)
    else:
        print(f"\nGenerating {args.n_prompts} synthetic prompts...")
        from scripts.collect_latents import create_synthetic_prompts
        prompts = create_synthetic_prompts(args.n_prompts, seed=args.seed)
    
    print(f"  Loaded {len(prompts)} prompts")
    
    # Initialize Phase 3 generator
    print("\nInitializing Phase 3 data generator...")
    from src.datasets.latent_qa import Phase3DataGenerator, save_dataset
    
    generator = Phase3DataGenerator(codi_wrapper=wrapper)
    
    # Generate diverse examples
    print(f"\nGenerating {args.target:,} diverse training examples...")
    examples = generator.generate_diverse_examples(
        prompts=prompts,
        target_count=args.target,
        verbose=True,
    )
    
    print(f"\nGenerated {len(examples):,} examples")
    
    # Analyze distribution
    print("\nQuestion type distribution:")
    type_counts = {}
    for ex in examples:
        q_type = ex.question_type
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    for q_type, count in sorted(type_counts.items()):
        pct = 100 * count / len(examples)
        print(f"  {q_type}: {count:,} ({pct:.1f}%)")
    
    # Count single vs multi-latent
    multi_count = sum(1 for ex in examples if ex.is_multi_latent)
    single_count = len(examples) - multi_count
    print(f"\nSingle-latent: {single_count:,} ({100*single_count/len(examples):.1f}%)")
    print(f"Multi-latent:  {multi_count:,} ({100*multi_count/len(examples):.1f}%)")
    
    # Save dataset
    print(f"\nSaving to {args.output}...")
    save_dataset(examples, args.output)
    
    # Show sample examples
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)
    
    import random
    samples = random.sample(examples, min(10, len(examples)))
    for i, ex in enumerate(samples):
        print(f"\n[{i+1}] Type: {ex.question_type}")
        print(f"    Q: {ex.question}")
        print(f"    A: {ex.answer}")
        print(f"    Latents: {len(ex.latent_vectors)} vector(s)")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nNext step: Train with")
    print(f"  python scripts/train.py --data {args.output}")


if __name__ == "__main__":
    main()
