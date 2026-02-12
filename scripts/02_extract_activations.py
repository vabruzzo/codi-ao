"""Run CODI inference and extract per-thought activations.

Usage:
    python scripts/02_extract_activations.py --config configs/thin.yaml
    python scripts/02_extract_activations.py --num-problems 100 --ckpt-dir checkpoints/codi-llama
"""

import argparse
import random

import yaml
from datasets import load_dataset, load_from_disk
from pathlib import Path

from src.config import CODIConfig, ExtractionConfig
from src.codi_loader import load_codi_model
from src.activation_extractor import extract_activations_batch, save_activation_records
from src.utils import set_seed


def load_problems(config: ExtractionConfig, max_problems: int | None = None):
    """Load problems from the dataset."""
    # Try loading from disk first (pre-downloaded)
    local_path = Path("data/gsm8k_aug")
    if local_path.exists():
        print(f"Loading dataset from {local_path}")
        dataset = load_from_disk(str(local_path))
    else:
        print(f"Downloading dataset: {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    # Sample if needed
    num = max_problems or config.num_problems
    if num < len(dataset):
        random.seed(config.seed)
        indices = random.sample(range(len(dataset)), num)
        dataset = dataset.select(indices)

    # Convert to list of dicts
    problems = []
    for item in dataset:
        problems.append({
            "question": item["question"],
            "cot": item.get("cot", ""),
            "answer": item.get("answer", ""),
        })

    print(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Extract CODI activations")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/codi-llama")
    parser.add_argument("--output-dir", type=str, default="data/activations")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config from YAML or defaults
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        codi_config = CODIConfig(**cfg.get("codi", {}))
        extraction_config = ExtractionConfig(**cfg.get("extraction", {}))
    else:
        codi_config = CODIConfig(
            model_name=args.model_name,
            ckpt_path=args.ckpt_dir,
        )
        extraction_config = ExtractionConfig(
            num_problems=args.num_problems,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    set_seed(extraction_config.seed)

    # Load CODI model
    print("Loading CODI model...")
    model, tokenizer = load_codi_model(
        model_name=codi_config.model_name,
        ckpt_dir=codi_config.ckpt_path,
        num_latent=codi_config.num_latent,
        inf_latent_iterations=codi_config.inf_latent_iterations,
        use_prj=codi_config.use_prj,
        prj_dim=codi_config.prj_dim,
        remove_eos=codi_config.remove_eos,
        use_lora=codi_config.use_lora,
        lora_r=codi_config.lora_r,
        lora_alpha=codi_config.lora_alpha,
    )

    # Load problems
    problems = load_problems(extraction_config)

    # Extract activations
    print(f"Extracting activations for {len(problems)} problems...")
    records = extract_activations_batch(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        codi_config=codi_config,
        extraction_config=extraction_config,
    )

    # Report statistics
    correct = sum(1 for r in records if r.prediction_correct)
    total = sum(1 for r in records if r.prediction_correct is not None)
    print(f"CODI accuracy: {correct}/{total} ({100*correct/max(total,1):.1f}%)")

    # Save
    save_activation_records(records, extraction_config.output_dir)


if __name__ == "__main__":
    main()
