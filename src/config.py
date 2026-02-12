"""Centralized configuration dataclasses for the CODI-AO pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CODIConfig:
    """Configuration for loading and running the CODI model."""

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ckpt_path: str = ""  # Path to CODI checkpoint directory
    num_latent: int = 6
    inf_latent_iterations: int = 6
    use_prj: bool = True
    prj_dim: int = 2048
    prj_dropout: float = 0.0
    prj_no_ln: bool = False
    remove_eos: bool = True
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    full_precision: bool = True
    greedy: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    device: str = "cuda"


@dataclass
class ExtractionConfig:
    """Configuration for extracting activations from CODI."""

    num_problems: int = 100  # Number of problems to process
    dataset_name: str = "zen-E/GSM8k-Aug"
    dataset_split: str = "train"
    layers: list[int] = field(default_factory=lambda: [4, 8, 12])
    extract_post_projection: bool = True
    decode_top_k: int = 5
    batch_size: int = 1  # Process one problem at a time (CODI uses iterative loop)
    output_dir: str = "data/activations"
    seed: int = 42


@dataclass
class QAConfig:
    """Configuration for QA dataset generation."""

    examples_per_category: dict[str, int] = field(default_factory=lambda: {
        "cat1_intermediate_result": 25000,
        "cat2_operation_classification": 30000,
        "cat3_full_reasoning": 15000,
        "cat4_problem_properties": 15000,
        "cat5_context_prediction": 25000,
        "cat6_thought_informativeness": 10000,
    })
    num_paraphrases: int = 12  # Paraphrases per template
    train_val_split: float = 0.88  # 22k/25k train, 3k/25k val
    output_dir: str = "data/qa_datasets"
    seed: int = 42


@dataclass
class AOTrainingConfig:
    """Configuration for training the Activation Oracle."""

    # Model
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    hook_onto_layer: int = 1  # Which AO layer to inject activations into

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Training
    num_epochs: int = 1
    lr: float = 1e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = False

    # Steering
    steering_coefficient: float = 1.0

    # Logging / Saving
    eval_steps: int = 2000
    save_steps: int = 5000
    logging_steps: int = 50
    output_dir: str = "checkpoints/ao"
    wandb_project: str = "codi-ao"
    wandb_run_name: str = ""

    # Data
    train_data_path: str = ""
    eval_data_path: str = ""

    seed: int = 42


@dataclass
class EvalConfig:
    """Configuration for evaluating the trained AO."""

    ao_checkpoint_path: str = ""
    eval_data_path: str = ""
    activations_dir: str = "data/activations"
    output_dir: str = "results"

    eval_batch_size: int = 32
    max_new_tokens: int = 64
    do_sample: bool = False  # Greedy for evaluation

    # Which evaluations to run
    run_intermediate_result: bool = True
    run_operation_classification: bool = True
    run_full_reasoning: bool = True
    run_problem_properties: bool = True
    run_context_prediction: bool = True
    run_thought_informativeness: bool = True
    run_codi_baseline_comparison: bool = True
    run_error_localization: bool = True

    seed: int = 42


# Thin pipeline overrides
THIN_EXTRACTION_CONFIG = ExtractionConfig(
    num_problems=100,
    batch_size=1,
    output_dir="data/activations_thin",
)

THIN_QA_CONFIG = QAConfig(
    examples_per_category={
        "cat1_intermediate_result": 100,
        "cat2_operation_classification": 150,
        "cat3_full_reasoning": 50,
        "cat4_problem_properties": 100,
        "cat5_context_prediction": 100,
        "cat6_thought_informativeness": 70,
    },
    output_dir="data/qa_datasets_thin",
)

THIN_TRAINING_CONFIG = AOTrainingConfig(
    num_epochs=3,
    train_batch_size=4,
    gradient_accumulation_steps=4,
    eval_steps=50,
    save_steps=100,
    output_dir="checkpoints/ao_thin",
)
