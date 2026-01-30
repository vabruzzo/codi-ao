#!/usr/bin/env python3
"""
Prepare Phase 4 training data by combining:
1. GSM8k 2-step problems (with verified latent mapping)
2. Expanded synthetic math (1-10, 1-100 ranges, novel entities, edge cases)

This script:
1. Loads both QA datasets
2. Balances them appropriately
3. Creates the final training JSONL for the AO
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: Path) -> list:
    """Load JSON file."""
    print(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} examples")
    return data


def analyze_dataset(examples: list, name: str):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    
    # By QA type
    qa_types = Counter(e.get("qa_type", "unknown") for e in examples)
    print("\nBy QA type:")
    for t, count in sorted(qa_types.items()):
        print(f"  {t}: {count} ({100*count/len(examples):.1f}%)")
    
    # By source (if available)
    sources = Counter(e.get("source", e.get("number_range", "unknown")) for e in examples)
    print("\nBy source/range:")
    for s, count in sorted(sources.items()):
        print(f"  {s}: {count} ({100*count/len(examples):.1f}%)")
    
    # By step number
    steps = Counter(e.get("step_number", 0) for e in examples)
    print("\nBy step number:")
    for s, count in sorted(steps.items()):
        print(f"  step {s}: {count} ({100*count/len(examples):.1f}%)")


def convert_to_training_format(examples: list, source_name: str) -> list:
    """
    Convert QA examples to the format expected by the AO trainer.
    
    The trainer expects:
    {
        "prompt": str,           # The math problem prompt
        "question": str,         # The QA question
        "answer": str,           # The expected answer
        "latent_position": int,  # Which latent index to use (1 for z1, 3 for z3)
        "step_number": int,      # Which step this is (1 or 2)
        "source": str,           # Where this came from
    }
    """
    converted = []
    
    for e in examples:
        # Get latent position
        lat_pos = e.get("latent_position")
        if lat_pos is None:
            # For synthetic data, use standard mapping
            step = e.get("step_number", 1)
            if step == 1:
                lat_pos = 1  # z1
            else:
                lat_pos = 3  # z3
        
        # Get the prompt (might be "question" or "prompt" in source data)
        prompt = e.get("prompt") or e.get("question", "")
        
        converted.append({
            "prompt": prompt,
            "question": e.get("qa_question", ""),
            "answer": str(e.get("qa_answer", e.get("answer", ""))),
            "latent_position": lat_pos,
            "step_number": e.get("step_number", 1),
            "qa_type": e.get("qa_type", "extraction"),
            "source": source_name,
        })
    
    return converted


def balance_and_combine(
    gsm8k_examples: list,
    synthetic_examples: list,
    gsm8k_ratio: float = 0.3,
    max_total: int = 200000,
    seed: int = 42,
) -> list:
    """
    Combine datasets with appropriate balancing.
    
    Args:
        gsm8k_ratio: Fraction of final dataset that should be GSM8k
        max_total: Maximum total examples (to keep training manageable)
    """
    random.seed(seed)
    
    # Calculate target sizes
    n_gsm8k = min(len(gsm8k_examples), int(max_total * gsm8k_ratio))
    n_synthetic = min(len(synthetic_examples), max_total - n_gsm8k)
    
    print(f"\nBalancing: {n_gsm8k} GSM8k + {n_synthetic} synthetic = {n_gsm8k + n_synthetic} total")
    
    # Sample
    if len(gsm8k_examples) > n_gsm8k:
        gsm8k_sampled = random.sample(gsm8k_examples, n_gsm8k)
    else:
        gsm8k_sampled = gsm8k_examples
    
    if len(synthetic_examples) > n_synthetic:
        synthetic_sampled = random.sample(synthetic_examples, n_synthetic)
    else:
        synthetic_sampled = synthetic_examples
    
    # Combine and shuffle
    combined = gsm8k_sampled + synthetic_sampled
    random.shuffle(combined)
    
    return combined


def save_training_data(examples: list, output_path: Path):
    """Save as JSONL for training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    
    print(f"\nSaved {len(examples)} examples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Phase 4 training data")
    parser.add_argument("--gsm8k_dir", type=str, default="data/gsm8k_2step",
                        help="GSM8k data directory")
    parser.add_argument("--synthetic_dir", type=str, default="data/synthetic_expanded",
                        help="Synthetic data directory")
    parser.add_argument("--output", type=str, default="data/phase4_train.jsonl",
                        help="Output training file")
    parser.add_argument("--gsm8k_ratio", type=float, default=0.3,
                        help="Fraction of dataset that should be GSM8k")
    parser.add_argument("--max_total", type=int, default=200000,
                        help="Maximum total training examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Load datasets
    gsm8k_path = Path(args.gsm8k_dir) / "gsm8k_train_qa.json"
    synthetic_path = Path(args.synthetic_dir) / "qa_examples.json"
    
    gsm8k_raw = load_json(gsm8k_path)
    synthetic_raw = load_json(synthetic_path)
    
    # Analyze raw data
    analyze_dataset(gsm8k_raw, "GSM8k 2-step")
    analyze_dataset(synthetic_raw, "Synthetic Expanded")
    
    # Convert to training format
    print("\nConverting to training format...")
    gsm8k_converted = convert_to_training_format(gsm8k_raw, "gsm8k_2step")
    synthetic_converted = convert_to_training_format(synthetic_raw, "synthetic_expanded")
    
    # Balance and combine
    combined = balance_and_combine(
        gsm8k_converted,
        synthetic_converted,
        gsm8k_ratio=args.gsm8k_ratio,
        max_total=args.max_total,
        seed=args.seed,
    )
    
    # Final stats
    analyze_dataset(combined, "Combined (Final)")
    
    # Save
    save_training_data(combined, Path(args.output))
    
    print("\n" + "="*60)
    print("READY FOR TRAINING")
    print("="*60)
    print(f"\nRun:")
    print(f"  python scripts/train.py --data {args.output} --output checkpoints/ao_phase4 --epochs 2")


if __name__ == "__main__":
    main()
