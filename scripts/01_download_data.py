"""Download GSM8k-Aug dataset and CODI checkpoints from HuggingFace."""

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def download_gsm8k_aug(output_dir: str = "data/gsm8k_aug"):
    """Download GSM8k-Aug training set."""
    print("Downloading GSM8k-Aug...")
    dataset = load_dataset("zen-E/GSM8k-Aug", split="train")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(dataset)} examples to {output_path}")
    return dataset


def download_codi_checkpoint(
    model_id: str = "zen-E/CODI-llama3.2-1b-Instruct",
    output_dir: str = "checkpoints/codi-llama",
):
    """Download CODI pretrained checkpoint."""
    print(f"Downloading CODI checkpoint: {model_id}")
    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
    )
    print(f"Saved checkpoint to {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download data and checkpoints")
    parser.add_argument("--dataset-dir", default="data/gsm8k_aug")
    parser.add_argument("--checkpoint-dir", default="checkpoints/codi-llama")
    parser.add_argument(
        "--model-id",
        default="zen-E/CODI-llama3.2-1b-Instruct",
        help="HuggingFace model ID for CODI checkpoint",
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()

    if not args.skip_dataset:
        download_gsm8k_aug(args.dataset_dir)

    if not args.skip_checkpoint:
        download_codi_checkpoint(args.model_id, args.checkpoint_dir)

    print("Done!")


if __name__ == "__main__":
    main()
