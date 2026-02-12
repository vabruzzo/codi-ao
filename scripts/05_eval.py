"""Run evaluation suite on the trained Activation Oracle.

Usage:
    python scripts/05_eval.py --config configs/thin.yaml
    python scripts/05_eval.py --checkpoint checkpoints/ao/final --eval-data data/qa_datasets/eval.pt
"""

import argparse

import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import EvalConfig, AOTrainingConfig
from src.ao_dataset import load_dataset_from_file
from src.ao_eval import evaluate, print_results, save_results
from src.steering import get_injection_submodule


def load_trained_ao(checkpoint_path: str, base_model_name: str, hook_layer: int = 1):
    """Load the trained AO model from a checkpoint."""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    injection_submodule = get_injection_submodule(model, hook_layer)

    return model, tokenizer, injection_submodule


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Activation Oracle")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ao/final")
    parser.add_argument("--eval-data", type=str, default="data/qa_datasets/eval.pt")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        eval_config = EvalConfig(**cfg.get("eval", {}))
        ao_config = AOTrainingConfig(**cfg.get("ao_training", {}))
        checkpoint = eval_config.ao_checkpoint_path or args.checkpoint
        eval_data_path = eval_config.eval_data_path or args.eval_data
        model_name = ao_config.model_name
        hook_layer = ao_config.hook_onto_layer
    else:
        eval_config = EvalConfig(
            eval_batch_size=args.batch_size,
            output_dir=str(args.output).rsplit("/", 1)[0] if "/" in args.output else "results",
        )
        checkpoint = args.checkpoint
        eval_data_path = args.eval_data
        model_name = args.model_name
        hook_layer = 1

    # Load model
    print(f"Loading trained AO from {checkpoint}")
    model, tokenizer, injection_submodule = load_trained_ao(
        checkpoint, model_name, hook_layer
    )

    # Load eval data
    print(f"Loading eval data from {eval_data_path}")
    eval_data = load_dataset_from_file(eval_data_path)
    print(f"Loaded {len(eval_data)} eval examples")

    # Run evaluation
    metrics = evaluate(
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        injection_submodule=injection_submodule,
        config=eval_config,
    )

    # Display and save results
    print_results(metrics)
    save_results(metrics, args.output)


if __name__ == "__main__":
    main()
