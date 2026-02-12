"""Build QA training dataset from extracted activations.

Usage:
    python scripts/03_generate_qa.py --config configs/thin.yaml
    python scripts/03_generate_qa.py --activations-dir data/activations --output-dir data/qa_datasets
"""

import argparse
import random

import yaml
from pathlib import Path
from transformers import AutoTokenizer

from src.config import ExtractionConfig, QAConfig, AOTrainingConfig
from src.activation_extractor import load_activation_records
from src.cot_parser import parse_problem
from src.thought_alignment import align_thoughts_to_steps, alignment_summary
from src.qa_generator import generate_all_qa_pairs, balance_binary_pairs
from src.ao_dataset import create_training_datapoint, save_dataset
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate QA training dataset")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--activations-dir", type=str, default="data/activations")
    parser.add_argument("--output-dir", type=str, default="data/qa_datasets")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--train-split", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        qa_config = QAConfig(**cfg.get("qa", {}))
        model_name = cfg.get("ao_training", {}).get("model_name", args.model_name)
        activations_dir = cfg.get("extraction", {}).get("output_dir", args.activations_dir)
    else:
        qa_config = QAConfig(output_dir=args.output_dir, seed=args.seed)
        model_name = args.model_name
        activations_dir = args.activations_dir

    set_seed(qa_config.seed)

    # Load activation records
    records_path = Path(activations_dir) / "activation_records.pt"
    print(f"Loading activation records from {records_path}")
    records = load_activation_records(str(records_path))
    print(f"Loaded {len(records)} activation records")

    # Load tokenizer for creating TrainingDataPoints
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate QA pairs for each problem
    all_qa_pairs = []
    alignment_stats = []

    for record in records:
        # Parse CoT
        parsed = parse_problem(
            problem_id=record.problem_id,
            question=record.question,
            cot_raw=record.cot_raw,
            answer=record.answer_gt,
        )

        # Align thoughts to steps
        alignments = align_thoughts_to_steps(record.thoughts, parsed.steps)
        stats = alignment_summary(alignments)
        alignment_stats.append(stats)

        # Generate QA pairs
        qa_pairs = generate_all_qa_pairs(
            record=record,
            parsed=parsed,
            alignments=alignments,
            layers=list(ExtractionConfig().layers),
        )
        all_qa_pairs.extend(qa_pairs)

    print(f"Generated {len(all_qa_pairs)} QA pairs")

    # Balance binary pairs
    all_qa_pairs = balance_binary_pairs(all_qa_pairs)
    print(f"After balancing: {len(all_qa_pairs)} QA pairs")

    # Print alignment statistics
    avg_alignment_rate = sum(s["alignment_rate"] for s in alignment_stats) / max(len(alignment_stats), 1)
    print(f"Average alignment rate: {avg_alignment_rate:.3f}")

    # Print category distribution
    from collections import Counter
    cat_counts = Counter(f"cat{p.category}" for p in all_qa_pairs)
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Convert to TrainingDataPoints
    print("Converting to TrainingDataPoints...")
    # Build a lookup from problem_id to record
    record_lookup = {r.problem_id: r for r in records}

    training_data = []
    skipped = 0
    errors = {}
    for qa_pair in all_qa_pairs:
        record = record_lookup.get(qa_pair.problem_id)
        if record is None:
            skipped += 1
            continue
        try:
            dp = create_training_datapoint(qa_pair, record, tokenizer)
            training_data.append(dp)
        except (AssertionError, ValueError, Exception) as e:
            err_key = str(e)[:100]
            errors[err_key] = errors.get(err_key, 0) + 1
            skipped += 1
            continue

    if errors:
        print(f"Errors during conversion ({skipped} total):")
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  [{count}x] {err}")

    print(f"Created {len(training_data)} TrainingDataPoints (skipped {skipped})")

    # Train/val split
    random.shuffle(training_data)
    split_idx = int(len(training_data) * args.train_split)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Save
    output_dir = Path(qa_config.output_dir)
    save_dataset(train_data, str(output_dir / "train.pt"))
    save_dataset(eval_data, str(output_dir / "eval.pt"))

    print("Done!")


if __name__ == "__main__":
    main()
