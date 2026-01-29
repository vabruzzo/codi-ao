#!/usr/bin/env python3
"""
Training script for CODI Activation Oracle.

Usage:
    # MVP training (small scale)
    python scripts/train.py --mode mvp --n_samples 10000
    
    # Full training
    python scripts/train.py --mode full
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class LatentQADataset(Dataset):
    """Dataset for training Activation Oracle."""
    
    def __init__(
        self,
        examples: list,
        tokenizer,
        max_length: int = 256,
        placeholder_token: str | None = None,
    ):
        """
        Args:
            examples: List of training examples
            tokenizer: Tokenizer to use
            max_length: Max sequence length
            placeholder_token: Placeholder token to search for. If None, uses default.
        """
        from src.activation_oracle import (
            DEFAULT_PLACEHOLDER_TOKEN,
            validate_placeholder_token,
            validate_repeated_placeholder,
        )
        
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.placeholder_token = placeholder_token or DEFAULT_PLACEHOLDER_TOKEN
        
        # Validate single token
        self.special_token_id = validate_placeholder_token(tokenizer, self.placeholder_token)
        
        # Validate repeated tokens don't merge (check with max expected placeholders)
        # Most examples have 1 placeholder, but validate up to a reasonable max
        max_placeholders = self._find_max_placeholders()
        if max_placeholders > 1:
            validate_repeated_placeholder(tokenizer, self.placeholder_token, max_placeholders)
    
    def _find_max_placeholders(self) -> int:
        """Find the maximum number of placeholders in any example.
        
        Note: This uses string counting, which works correctly for the default
        placeholder token " ?" (space + question mark) since the leading space
        prevents matching trailing "?" in natural text. If using a custom token
        that appears in natural text, this count may be inflated.
        """
        max_count = 1
        for ex in self.examples[:100]:  # Sample first 100 to avoid scanning all
            count = ex["prompt"].count(self.placeholder_token)
            max_count = max(max_count, count)
        return max_count
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Format: prompt + answer
        full_text = ex["prompt"] + " " + ex["answer"]
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create labels (mask prompt tokens)
        prompt_encoding = self.tokenizer(
            ex["prompt"],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_len] = -100  # Mask prompt
        
        # Get latent vectors - handle both old (latent_vector) and new (latent_vectors) format
        if "latent_vectors" in ex:
            # New format: list of vectors
            latent_vectors = [torch.tensor(v, dtype=torch.float32) for v in ex["latent_vectors"]]
            # Stack into (num_vectors, hidden_dim)
            latent_tensor = torch.stack(latent_vectors)
        else:
            # Old format: single vector
            latent_tensor = torch.tensor(ex["latent_vector"], dtype=torch.float32).unsqueeze(0)
        
        # Find placeholder positions using the special token
        positions = (encoding["input_ids"][0] == self.special_token_id).nonzero(as_tuple=True)[0]
        
        if len(positions) == 0:
            raise ValueError(
                f"No placeholder tokens found in example {idx}. "
                f"Prompt: {ex['prompt'][:100]}..."
            )
        
        # Validate vector count matches position count
        num_vectors = latent_tensor.shape[0]
        num_positions = len(positions)
        if num_vectors != num_positions:
            raise ValueError(
                f"Example {idx}: vector count ({num_vectors}) != placeholder count ({num_positions}). "
                f"Prompt: {ex['prompt'][:100]}..."
            )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "latent_vectors": latent_tensor,  # Shape: (num_vectors, hidden_dim)
            "positions": positions.tolist(),
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    
    # Handle variable-length positions and vectors
    # Each item has latent_vectors of shape (num_vectors, hidden_dim)
    latent_vectors = [b["latent_vectors"] for b in batch]
    positions = [b["positions"] for b in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "latent_vectors": latent_vectors,  # List of tensors, each (K_b, hidden_dim)
        "positions": positions,  # List of position lists
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    verbose: bool = True,
):
    """Train for one epoch."""
    model.train_mode()
    
    total_loss = 0
    num_batches = 0
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if verbose else dataloader
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(iterator):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Get per-sample vectors and positions
        # latent_vectors is list of tensors, positions is list of lists
        latent_vectors = batch["latent_vectors"]
        positions = batch["positions"]
        
        batch_size = input_ids.shape[0]
        
        # Prepare per-sample activation vectors: each should be (K_b, hidden_dim)
        # where K_b is the number of positions for that sample
        activation_vectors = []
        for i in range(batch_size):
            vec = latent_vectors[i].to(device)
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)  # (hidden_dim,) -> (1, hidden_dim)
            activation_vectors.append(vec)
        
        # Forward pass with per-sample injection
        outputs = model.forward_with_injection(
            input_ids=input_ids,
            attention_mask=attention_mask,
            activation_vectors=activation_vectors,
            positions=positions,  # Already list of lists
            labels=labels,
        )
        
        loss = outputs["loss"]
        if loss is None:
            continue
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if verbose:
            iterator.set_postfix({"loss": total_loss / num_batches})
    
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Train CODI Activation Oracle")
    parser.add_argument("--mode", type=str, choices=["mvp", "full", "phase2"], default="mvp")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data (default: auto-detect)")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit number of samples (default: use all)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="checkpoints/ao")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CODI Activation Oracle - Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    if args.n_samples:
        print(f"Max samples: {args.n_samples}")
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize AO config first (needed for placeholder token consistency)
    print("\nInitializing Activation Oracle config...")
    from src.activation_oracle import ActivationOracle, AOConfig
    
    config = AOConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device=args.device,
        lora_r=64,
        lora_alpha=128,
    )
    
    # Determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    elif args.mode == "phase2":
        data_path = Path("data/phase2/train.jsonl")
    else:
        data_path = Path("data/latent_qa_train.jsonl")
    
    if not data_path.exists():
        print(f"\nNo training data found at {data_path}")
        if args.mode == "phase2":
            print("Run scripts/generate_phase2_data.py first to generate training data")
            sys.exit(1)
        
        print("Generating synthetic data for testing...")
        from src.datasets.latent_qa import create_synthetic_examples
        n_samples = args.n_samples or 10000
        examples = create_synthetic_examples(
            n_samples,
            placeholder_token=config.placeholder_token,
        )
        
        examples_dict = [e.to_dict() for e in examples]
        
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w") as f:
            for ex in examples_dict:
                f.write(json.dumps(ex) + "\n")
        
        print(f"Saved {len(examples)} synthetic examples to {data_path}")
    
    # Load training data
    print(f"\nLoading training data from {data_path}...")
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    
    if args.n_samples and len(examples) > args.n_samples:
        examples = random.sample(examples, args.n_samples)
    
    print(f"Loaded {len(examples)} examples")
    
    # Show task breakdown if available
    task_counts = {}
    for ex in examples[:1000]:  # Sample first 1000
        task = ex.get("task", ex.get("question_type", "unknown"))
        task_counts[task] = task_counts.get(task, 0) + 1
    if len(task_counts) > 1:
        print("Task breakdown (sampled):")
        for task, count in task_counts.items():
            print(f"  {task}: {count}")
    
    # Initialize AO model
    print("\nInitializing Activation Oracle model...")
    
    ao = ActivationOracle.from_pretrained(config=config)
    print(f"Trainable parameters: {ao.num_trainable_parameters:,}")
    
    # Create dataset and dataloader
    # Pass the placeholder token from AO config to ensure consistency
    dataset = LatentQADataset(
        examples,
        ao.tokenizer,
        placeholder_token=config.placeholder_token,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(ao.model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model=ao,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation,
            verbose=args.verbose,
        )
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ao.save_lora(str(output_dir))
    print(f"\nSaved model to {output_dir}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
