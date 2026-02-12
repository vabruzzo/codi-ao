"""Train the Activation Oracle on CODI thought activations.

Usage:
    python scripts/04_train.py --config configs/thin.yaml
    python scripts/04_train.py --train-data data/qa_datasets/train.pt --eval-data data/qa_datasets/eval.pt
"""

import argparse

import yaml

from src.config import AOTrainingConfig
from src.ao_trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the Activation Oracle")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--train-data", type=str, default="data/qa_datasets/train.pt")
    parser.add_argument("--eval-data", type=str, default="data/qa_datasets/eval.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ao")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        training_config = AOTrainingConfig(**cfg.get("ao_training", {}))
    else:
        training_config = AOTrainingConfig(
            model_name=args.model_name,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            lr=args.lr,
            train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            seed=args.seed,
        )

    train(training_config)


if __name__ == "__main__":
    main()
