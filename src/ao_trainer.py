"""AO training loop adapted from activation_oracles/nl_probes/sft.py.

Simplified for CODI:
- No DDP (single GPU; DDP can be added for full runs)
- No on-the-fly materialization (all activations pre-computed)
- No SAE-specific logic

Core loop:
1. Load LLaMA-3.2-1B-Instruct (fresh, NOT CODI's weights)
2. Apply LoRA
3. For each batch: apply steering hook → forward pass → CE loss → backward
4. AdamW with linear warmup + decay
"""

import math
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import AOTrainingConfig
from src.ao_dataset import (
    TrainingDataPoint,
    BatchData,
    construct_batch,
    get_prompt_only,
    load_dataset_from_file,
    save_dataset,
)
from src.steering import get_steering_hook, add_hook, get_injection_submodule
from src.utils import set_seed


def load_ao_model(config: AOTrainingConfig):
    """Load a fresh LLaMA model as the AO base and apply LoRA.

    Returns:
        Tuple of (model, tokenizer, injection_submodule).
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model = model.to("cuda")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get the injection submodule (transformer layer for hook)
    injection_submodule = get_injection_submodule(model, config.hook_onto_layer)

    return model, tokenizer, injection_submodule


def train_step(
    batch: BatchData,
    model,
    injection_submodule,
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run a single training step with activation injection.

    Returns:
        The loss tensor.
    """
    hook_fn = get_steering_hook(
        vectors=batch.steering_vectors,
        positions=batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    with add_hook(injection_submodule, hook_fn):
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )

    return outputs.loss


@torch.no_grad()
def eval_step(
    eval_data: list[TrainingDataPoint],
    model,
    tokenizer,
    injection_submodule,
    config: AOTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> dict:
    """Run evaluation and return metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    indices = list(range(len(eval_data)))
    for start in range(0, min(len(eval_data), max_batches * config.eval_batch_size), config.eval_batch_size):
        batch_points = eval_data[start:start + config.eval_batch_size]
        if not batch_points:
            break

        batch = construct_batch(batch_points, tokenizer, device)
        loss = train_step(batch, model, injection_submodule, config.steering_coefficient, device, dtype)
        total_loss += loss.item()
        num_batches += 1

    model.train()

    return {
        "eval_loss": total_loss / max(num_batches, 1),
        "num_eval_batches": num_batches,
    }


def train(
    config: AOTrainingConfig,
    train_data: list[TrainingDataPoint] | None = None,
    eval_data: list[TrainingDataPoint] | None = None,
):
    """Main training loop.

    Args:
        config: Training configuration.
        train_data: Training data (loaded from config.train_data_path if None).
        eval_data: Evaluation data (loaded from config.eval_data_path if None).
    """
    set_seed(config.seed)

    # Load data if not provided
    if train_data is None:
        print(f"Loading training data from {config.train_data_path}")
        train_data = load_dataset_from_file(config.train_data_path)
    if eval_data is None and config.eval_data_path:
        print(f"Loading eval data from {config.eval_data_path}")
        eval_data = load_dataset_from_file(config.eval_data_path)

    print(f"Training examples: {len(train_data)}")
    if eval_data:
        print(f"Evaluation examples: {len(eval_data)}")

    # Load model
    model, tokenizer, injection_submodule = load_ao_model(config)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if config.bf16 else torch.float32

    # Set up optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    total_steps = (
        math.ceil(len(train_data) / config.train_batch_size)
        * config.num_epochs
        // config.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Try to set up wandb
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config={
                "model_name": config.model_name,
                "lr": config.lr,
                "epochs": config.num_epochs,
                "batch_size": config.train_batch_size,
                "lora_r": config.lora_r,
                "steering_coefficient": config.steering_coefficient,
                "train_examples": len(train_data),
            },
        )
    except ImportError:
        print("WandB not available, skipping logging")

    # Training loop
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(config.num_epochs):
        # Shuffle training data
        indices = list(range(len(train_data)))
        random.shuffle(indices)

        epoch_loss = 0.0
        epoch_steps = 0

        for start in range(0, len(train_data), config.train_batch_size):
            batch_indices = indices[start:start + config.train_batch_size]
            batch_points = [train_data[i] for i in batch_indices]

            if not batch_points:
                break

            batch = construct_batch(batch_points, tokenizer, device)
            loss = train_step(
                batch, model, injection_submodule,
                config.steering_coefficient, device, dtype,
            )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            epoch_steps += 1

            if epoch_steps % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1}/{config.num_epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )
                    if wandb_run:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + start / len(train_data),
                        }, step=global_step)

                # Evaluation
                if eval_data and global_step % config.eval_steps == 0:
                    eval_metrics = eval_step(
                        eval_data, model, tokenizer, injection_submodule,
                        config, device, dtype,
                    )
                    print(f"  Eval loss: {eval_metrics['eval_loss']:.4f}")
                    if wandb_run:
                        wandb.log({
                            "eval/loss": eval_metrics["eval_loss"],
                        }, step=global_step)

                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        save_path = output_dir / "best"
                        model.save_pretrained(str(save_path))
                        tokenizer.save_pretrained(str(save_path))
                        print(f"  Saved best model (loss={best_eval_loss:.4f})")

                # Checkpoint
                if global_step % config.save_steps == 0:
                    save_path = output_dir / f"step_{global_step}"
                    model.save_pretrained(str(save_path))
                    tokenizer.save_pretrained(str(save_path))
                    print(f"  Saved checkpoint at step {global_step}")

    # Final save
    save_path = output_dir / "final"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Training complete. Final model saved to {save_path}")

    # Final eval
    if eval_data:
        eval_metrics = eval_step(
            eval_data, model, tokenizer, injection_submodule,
            config, device, dtype,
        )
        print(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")

    if wandb_run:
        wandb.finish()

    return model, tokenizer
