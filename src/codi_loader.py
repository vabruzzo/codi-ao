"""Load CODI model from checkpoint for activation extraction.

Adapts the loading pattern from codi/test.py.
"""

import sys
import os
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

# Import CODI's src/model.py directly by file path to avoid
# conflicting with our own src/ package.
import importlib.util

CODI_REPO = Path(__file__).parent.parent / "codi"
_codi_model_path = CODI_REPO / "src" / "model.py"

_spec = importlib.util.spec_from_file_location("codi_model", str(_codi_model_path))
_codi_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_codi_module)

CODI_cls = _codi_module.CODI
ModelArguments = _codi_module.ModelArguments
TrainingArguments = _codi_module.TrainingArguments


def _get_lora_config(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    """Build LoRA config matching the CODI training setup."""
    name_lower = model_name.lower()
    if any(n in name_lower for n in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "gpt2" in name_lower:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif "phi" in name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def load_codi_model(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    ckpt_dir: str = "",
    num_latent: int = 6,
    inf_latent_iterations: int = 6,
    use_prj: bool = True,
    prj_dim: int = 2048,
    prj_dropout: float = 0.0,
    prj_no_ln: bool = False,
    remove_eos: bool = True,
    use_lora: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    full_precision: bool = True,
    device: str = "cuda",
) -> tuple:
    """Load a CODI model from checkpoint.

    Returns:
        Tuple of (CODI model, tokenizer).
    """
    # Build argument dataclasses matching CODI's expected format
    model_args = ModelArguments(
        model_name_or_path=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        full_precision=full_precision,
        train=False,  # Inference mode
        lora_init=True,
        ckpt_dir=ckpt_dir,
    )

    # Create a minimal TrainingArguments — only the fields CODI's __init__ reads
    training_args = TrainingArguments(
        output_dir="/tmp/codi_eval",
        num_latent=num_latent,
        inf_latent_iterations=inf_latent_iterations,
        use_prj=use_prj,
        prj_dim=prj_dim,
        prj_dropout=prj_dropout,
        prj_no_ln=prj_no_ln,
        remove_eos=remove_eos,
        use_lora=use_lora,
        bf16=True,
        print_loss=False,
    )

    lora_config = _get_lora_config(model_name, lora_r, lora_alpha, lora_dropout)

    # Instantiate model
    model = CODI_cls(model_args, training_args, lora_config)

    # Load checkpoint weights
    if ckpt_dir:
        ckpt_path = Path(ckpt_dir)
        safetensors_path = ckpt_path / "model.safetensors"
        bin_path = ckpt_path / "pytorch_model.bin"

        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_dir}. "
                f"Expected model.safetensors or pytorch_model.bin"
            )

        model.load_state_dict(state_dict, strict=False)
        model.codi.tie_weights()

    # Set up tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id

    # Move to device and set eval mode
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.eval()

    return model, tokenizer


def load_codi_from_config(config) -> tuple:
    """Load CODI model from a CODIConfig dataclass."""
    from src.config import CODIConfig

    return load_codi_model(
        model_name=config.model_name,
        ckpt_dir=config.ckpt_path,
        num_latent=config.num_latent,
        inf_latent_iterations=config.inf_latent_iterations,
        use_prj=config.use_prj,
        prj_dim=config.prj_dim,
        prj_dropout=config.prj_dropout,
        prj_no_ln=config.prj_no_ln,
        remove_eos=config.remove_eos,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        full_precision=config.full_precision,
        device=config.device,
    )
