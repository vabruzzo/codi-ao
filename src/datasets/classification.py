"""
Classification Dataset Generator for CODI Activation Oracle.

Generates binary yes/no classification questions about latent vectors:
- Operation type (is this addition? multiplication?)
- Result properties (is the result > 100? negative?)
- Step position (is this an early step? the final calculation?)
- Correctness (is this reasoning correct?)
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm


@dataclass
class ClassificationExample:
    """A single binary classification training example."""

    # Input
    prompt: str  # The oracle prompt (with placeholders)
    latent_vectors: list[list[float]]  # List of latent vectors to inject
    latent_positions: list[int]  # Which latent positions (e.g., [1] or [0,1,2,3,4,5])

    # Target
    question: str  # The yes/no question
    answer: str  # "Yes" or "No"

    # Metadata
    source_prompt: str = ""  # Original math problem
    classification_type: str = ""  # Type of classification
    ground_truth_value: str = ""  # The actual value being classified
    is_multi_latent: bool = False  # True if using multiple vectors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClassificationExample":
        # Handle backward compatibility with old format
        if "latent_vector" in d and "latent_vectors" not in d:
            d["latent_vectors"] = [d.pop("latent_vector")]
        if "latent_position" in d and "latent_positions" not in d:
            d["latent_positions"] = [d.pop("latent_position")]
        if "layer_percent" in d:
            d.pop("layer_percent")
        if "is_multi_latent" not in d:
            d["is_multi_latent"] = len(d.get("latent_vectors", [])) > 1
        return cls(**d)


# Classification task definitions
# Each task has: question templates, condition function, task type
CLASSIFICATION_TASKS = {
    # Operation type detection
    "is_addition": {
        "questions": [
            "Is this step performing addition?",
            "Was addition used in this calculation?",
            "Is this an addition operation?",
            "Does this step involve adding numbers?",
        ],
        "condition": lambda step, result, pos: "+" in step,
        "type": "operation",
    },
    "is_subtraction": {
        "questions": [
            "Is this step performing subtraction?",
            "Was subtraction used in this calculation?",
            "Is this a subtraction operation?",
            "Does this step involve subtracting numbers?",
        ],
        "condition": lambda step, result, pos: "-" in step and "=" in step,
        "type": "operation",
    },
    "is_multiplication": {
        "questions": [
            "Is this step performing multiplication?",
            "Was multiplication used in this calculation?",
            "Is this a multiplication operation?",
            "Does this step involve multiplying numbers?",
        ],
        "condition": lambda step, result, pos: "*" in step or "×" in step,
        "type": "operation",
    },
    "is_division": {
        "questions": [
            "Is this step performing division?",
            "Was division used in this calculation?",
            "Is this a division operation?",
            "Does this step involve dividing numbers?",
        ],
        "condition": lambda step, result, pos: "/" in step or "÷" in step,
        "type": "operation",
    },
    # Result properties
    "result_greater_than_100": {
        "questions": [
            "Is the result greater than 100?",
            "Is the computed value more than 100?",
            "Does this step produce a result above 100?",
            "Is the intermediate result larger than 100?",
        ],
        "condition": lambda step, result, pos: _safe_float(result) > 100,
        "type": "result_property",
    },
    "result_greater_than_50": {
        "questions": [
            "Is the result greater than 50?",
            "Is the computed value more than 50?",
            "Does this step produce a result above 50?",
        ],
        "condition": lambda step, result, pos: _safe_float(result) > 50,
        "type": "result_property",
    },
    "result_less_than_10": {
        "questions": [
            "Is the result less than 10?",
            "Is the computed value smaller than 10?",
            "Does this step produce a result below 10?",
        ],
        "condition": lambda step, result, pos: _safe_float(result) < 10,
        "type": "result_property",
    },
    "result_is_negative": {
        "questions": [
            "Is the result negative?",
            "Is the computed value less than zero?",
            "Does this step produce a negative number?",
        ],
        "condition": lambda step, result, pos: _safe_float(result) < 0,
        "type": "result_property",
    },
    "result_is_whole_number": {
        "questions": [
            "Is the result a whole number?",
            "Is the computed value an integer?",
            "Does this step produce a whole number result?",
        ],
        "condition": lambda step, result, pos: _is_whole_number(result),
        "type": "result_property",
    },
    # Step position
    "is_early_step": {
        "questions": [
            "Is this an early reasoning step?",
            "Is this one of the first calculation steps?",
            "Is this near the beginning of the reasoning?",
        ],
        "condition": lambda step, result, pos: pos < 3,
        "type": "position",
    },
    "is_late_step": {
        "questions": [
            "Is this a late reasoning step?",
            "Is this near the end of the reasoning?",
            "Is this one of the final calculation steps?",
        ],
        "condition": lambda step, result, pos: pos >= 4,
        "type": "position",
    },
    "is_calculation_step": {
        "questions": [
            "Is this step performing a calculation?",
            "Does this step compute a value?",
            "Is this a computational step?",
        ],
        "condition": lambda step, result, pos: pos in [1, 3],  # z2 and z4
        "type": "structure",
    },
}


def _safe_float(s: str) -> float:
    """Safely convert string to float, returning 0 on failure."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _is_whole_number(s: str) -> bool:
    """Check if string represents a whole number."""
    try:
        f = float(s.replace(",", ""))
        return f == int(f)
    except (ValueError, AttributeError):
        return False


class ClassificationDatasetGenerator:
    """
    Generator for binary classification training data.

    Usage:
        from src.codi_wrapper import CODIWrapper

        wrapper = CODIWrapper.from_pretrained()
        generator = ClassificationDatasetGenerator(wrapper)

        examples = generator.generate_from_prompts(
            prompts=math_problems,
            tasks=["is_addition", "result_greater_than_100"],
        )
    """

    def __init__(
        self,
        codi_wrapper: Optional["CODIWrapper"] = None,
        placeholder_token: Optional[str] = None,
    ):
        """
        Args:
            codi_wrapper: CODI model wrapper for collecting latents
            placeholder_token: Placeholder token to use in prompts. If None, uses default.
        """
        from ..activation_oracle import DEFAULT_PLACEHOLDER_TOKEN
        
        self.codi_wrapper = codi_wrapper
        self.placeholder_token = placeholder_token or DEFAULT_PLACEHOLDER_TOKEN

        # Position to step mapping
        self.position_to_step = {
            1: 0,  # z2 → first intermediate result
            3: 1,  # z4 → second intermediate result
        }

    def generate_from_prompts(
        self,
        prompts: list[dict],
        tasks: Optional[list[str]] = None,
        n_per_prompt: int = 2,
        balance_answers: bool = True,
        verbose: bool = True,
    ) -> list[ClassificationExample]:
        """
        Generate classification examples from math problem prompts.

        Args:
            prompts: List of dicts with 'prompt', 'cot_steps', 'results' keys
            tasks: List of task names from CLASSIFICATION_TASKS (None = all)
            n_per_prompt: Number of examples per prompt per task
            balance_answers: Try to balance Yes/No answers
            verbose: Show progress

        Returns:
            List of ClassificationExample objects
        """
        if self.codi_wrapper is None:
            raise ValueError("CODI wrapper not provided")

        if tasks is None:
            tasks = list(CLASSIFICATION_TASKS.keys())

        examples = []
        yes_count = 0
        no_count = 0

        iterator = tqdm(prompts, desc="Generating classification data") if verbose else prompts

        for item in iterator:
            prompt = item["prompt"]
            cot_steps = item.get("cot_steps", [])
            results = item.get("results", [])

            # Collect latent vectors
            result = self.codi_wrapper.collect_latents(prompt)

            if len(result.latent_vectors) < 5:
                continue

            # Generate examples for key positions
            for lat_pos, step_idx in self.position_to_step.items():
                if lat_pos >= len(result.latent_vectors):
                    continue

                latent_vec = result.latent_vectors[lat_pos]
                cot_step = cot_steps[step_idx] if step_idx < len(cot_steps) else ""
                step_result = results[step_idx] if step_idx < len(results) else ""

                for task_name in tasks:
                    task = CLASSIFICATION_TASKS[task_name]
                    condition = task["condition"]

                    # Evaluate condition
                    try:
                        is_true = condition(cot_step, step_result, lat_pos)
                    except Exception:
                        continue

                    answer = "Yes" if is_true else "No"

                    # Balance answers if requested
                    if balance_answers:
                        if answer == "Yes" and yes_count > no_count + 100:
                            continue
                        if answer == "No" and no_count > yes_count + 100:
                            continue

                    # Generate examples
                    for _ in range(n_per_prompt):
                        question = str(random.choice(task["questions"]))
                        task_type = str(task["type"])
                        # Format oracle prompt using the configured placeholder token
                        from ..activation_oracle import format_oracle_prompt

                        oracle_prompt = format_oracle_prompt(
                            question=question,
                            num_activations=1,
                            placeholder_token=self.placeholder_token,
                        )

                        example = ClassificationExample(
                            prompt=oracle_prompt,
                            latent_vectors=[latent_vec.tolist()],
                            latent_positions=[lat_pos],
                            question=question,
                            answer=answer,
                            source_prompt=prompt,
                            classification_type=task_type,
                            ground_truth_value=step_result,
                            is_multi_latent=False,
                        )
                        examples.append(example)

                        if answer == "Yes":
                            yes_count += 1
                        else:
                            no_count += 1

        if verbose:
            print(f"Generated {len(examples)} examples (Yes: {yes_count}, No: {no_count})")

        return examples


def save_dataset(examples: list[ClassificationExample], path: str):
    """Save dataset to JSONL file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    print(f"Saved {len(examples)} examples to {filepath}")


def load_dataset(path: str) -> list[ClassificationExample]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(ClassificationExample.from_dict(json.loads(line)))
    return examples


def create_synthetic_examples(
    n: int = 100,
    placeholder_token: Optional[str] = None,
) -> list[ClassificationExample]:
    """
    Create synthetic examples for testing without running CODI.
    
    Args:
        n: Number of examples to create
        placeholder_token: Placeholder token to use. If None, uses default.
    """
    from ..activation_oracle import DEFAULT_PLACEHOLDER_TOKEN, format_oracle_prompt

    placeholder_token = placeholder_token or DEFAULT_PLACEHOLDER_TOKEN
    examples = []

    for i in range(n):
        task_name = random.choice(list(CLASSIFICATION_TASKS.keys()))
        task = CLASSIFICATION_TASKS[task_name]
        question = str(random.choice(task["questions"]))
        task_type = str(task["type"])
        answer = random.choice(["Yes", "No"])

        oracle_prompt = format_oracle_prompt(
            question=question,
            num_activations=1,
            placeholder_token=placeholder_token,
        )

        lat_pos = random.choice([1, 3])  # z2 or z4
        example = ClassificationExample(
            prompt=oracle_prompt,
            latent_vectors=[torch.randn(2048).tolist()],
            latent_positions=[lat_pos],
            question=question,
            answer=answer,
            source_prompt=f"Synthetic problem {i}",
            classification_type=task_type,
            ground_truth_value=str(random.randint(1, 200)),
            is_multi_latent=False,
        )
        examples.append(example)

    return examples


if __name__ == "__main__":
    # Test synthetic generation
    print("Creating synthetic classification examples...")
    examples = create_synthetic_examples(20)

    print(f"\nGenerated {len(examples)} examples")

    # Count by type
    type_counts: dict[str, int] = {}
    answer_counts: dict[str, int] = {"Yes": 0, "No": 0}
    for ex in examples:
        type_counts[ex.classification_type] = type_counts.get(ex.classification_type, 0) + 1
        answer_counts[ex.answer] += 1

    print("\nBy type:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    print("\nBy answer:")
    for a, c in answer_counts.items():
        print(f"  {a}: {c}")

    print("\nExample:")
    ex = examples[0]
    print(f"  Prompt: {ex.prompt}")
    print(f"  Question: {ex.question}")
    print(f"  Answer: {ex.answer}")
    print(f"  Type: {ex.classification_type}")
