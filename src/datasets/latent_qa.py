"""
Latent-to-CoT QA Dataset Generator.

Generates question-answer pairs by aligning CODI's latent vectors
with the intermediate steps from the teacher's explicit CoT.

Based on LessWrong findings:
- z3 (index 2) stores Step 1 intermediate result
- z5 (index 4) stores Step 2 intermediate result
"""

import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm


@dataclass
class LatentQAExample:
    """A single training example for Latent QA."""
    
    # Input
    prompt: str  # The oracle prompt (with placeholders)
    latent_vector: list[float]  # The latent vector to inject
    latent_position: int  # Which latent position (0-5)
    layer_percent: int  # Layer the latent was collected from
    
    # Target
    question: str  # The question being asked
    answer: str  # The expected answer
    
    # Metadata
    source_prompt: str = ""  # Original math problem
    cot_step: str = ""  # The CoT step this latent aligns to
    question_type: str = ""  # "intermediate_result", "operation", "structure"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "LatentQAExample":
        return cls(**d)


# Question templates for different query types
INTERMEDIATE_RESULT_TEMPLATES = [
    "What is the intermediate calculation result?",
    "What number is computed at this step?",
    "What value is stored in this reasoning position?",
    "What is the result of this calculation?",
    "What intermediate value does this represent?",
    "What is the numeric output of this step?",
    "What calculation result is stored here?",
    "What number does this step produce?",
    "What is the computed value at this position?",
    "What intermediate result is encoded here?",
    "What is the value after this computation?",
    "What number results from this reasoning step?",
    "What is stored as the intermediate result?",
    "What numerical value is represented here?",
    "What is the output of this calculation step?",
    "What intermediate answer is stored?",
    "What is the result value at this step?",
    "What number is the result of this computation?",
    "What value was calculated here?",
    "What is the numeric result stored in this position?",
]

OPERATION_TYPE_TEMPLATES = [
    "What mathematical operation was performed?",
    "What type of calculation was done?",
    "Is this an addition, subtraction, multiplication, or division?",
    "What operation produced this result?",
    "What arithmetic operation was used?",
    "What kind of math operation is this?",
    "What calculation type was performed?",
    "Is this step adding, subtracting, multiplying, or dividing?",
    "What mathematical process was applied?",
    "What operation was computed here?",
]

STRUCTURE_TEMPLATES = [
    "Is this a transitional step or a calculation step?",
    "Does this step perform a meaningful calculation?",
    "Is this an intermediate computation or a placeholder?",
    "Is this step storing a calculation result?",
    "Does this represent an actual computation?",
]


def extract_operation(step: str) -> str:
    """Extract the operation type from a CoT step string."""
    if '+' in step:
        return "addition"
    elif '-' in step:
        return "subtraction"
    elif '*' in step or '×' in step:
        return "multiplication"
    elif '/' in step or '÷' in step:
        return "division"
    else:
        return "unknown"


def extract_result(step: str) -> str:
    """Extract the numeric result from a CoT step string."""
    # Try to find result after '='
    if '=' in step:
        result = step.split('=')[-1].strip()
        # Clean up any trailing characters
        result = re.sub(r'[^0-9.\-]', '', result)
        return result
    return ""


class LatentQADatasetGenerator:
    """
    Generator for Latent-to-CoT QA training data.
    
    Usage:
        from src.codi_wrapper import CODIWrapper
        
        wrapper = CODIWrapper.from_pretrained()
        generator = LatentQADatasetGenerator(wrapper)
        
        examples = generator.generate_from_prompts(
            prompts=math_problems,
            n_per_prompt=5,
        )
    """
    
    def __init__(
        self,
        codi_wrapper: Optional["CODIWrapper"] = None,
        layer_percent: int = 50,
        placeholder_token: Optional[str] = None,
    ):
        """
        Args:
            codi_wrapper: CODI model wrapper for collecting latents
            layer_percent: Layer percentage for prompt format
            placeholder_token: Placeholder token to use in prompts. If None, uses default.
        """
        from ..activation_oracle import DEFAULT_PLACEHOLDER_TOKEN
        
        self.codi_wrapper = codi_wrapper
        self.layer_percent = layer_percent
        self.placeholder_token = placeholder_token or DEFAULT_PLACEHOLDER_TOKEN
        
        # Latent position to CoT step mapping (per LessWrong findings)
        # For 3-step problems: z3 → step1, z5 → step2
        self.position_to_step = {
            2: 0,  # z3 → first intermediate result
            4: 1,  # z5 → second intermediate result
        }
    
    def generate_from_prompts(
        self,
        prompts: list[dict],
        n_per_prompt: int = 5,
        question_types: list[str] = ["intermediate_result", "operation"],
        verbose: bool = True,
    ) -> list[LatentQAExample]:
        """
        Generate QA examples from a list of math problem prompts.
        
        Args:
            prompts: List of dicts with 'prompt' key (and optionally 'cot_steps', 'results')
            n_per_prompt: Number of QA pairs to generate per prompt
            question_types: Types of questions to generate
            verbose: Whether to show progress
        
        Returns:
            List of LatentQAExample objects
        """
        if self.codi_wrapper is None:
            raise ValueError("CODI wrapper not provided")
        
        examples = []
        iterator = tqdm(prompts, desc="Generating QA pairs") if verbose else prompts
        
        for item in iterator:
            prompt = item["prompt"]
            
            # Collect latent vectors from CODI
            result = self.codi_wrapper.collect_latents(prompt)
            
            if len(result.latent_vectors) < 5:
                continue
            
            # Get CoT steps if available
            if "cot_steps" in item:
                cot_steps = item["cot_steps"]
                results = item.get("results", [extract_result(s) for s in cot_steps])
            else:
                # Try to run teacher task
                try:
                    _, cot_steps, results = self.codi_wrapper.run_teacher_task(prompt)
                except Exception:
                    cot_steps = []
                    results = []
            
            # Generate examples for key latent positions
            for lat_pos, step_idx in self.position_to_step.items():
                if lat_pos >= len(result.latent_vectors):
                    continue
                
                latent_vec = result.latent_vectors[lat_pos]
                
                # Get corresponding CoT step info
                cot_step = cot_steps[step_idx] if step_idx < len(cot_steps) else ""
                step_result = results[step_idx] if step_idx < len(results) else ""
                operation = extract_operation(cot_step)
                
                # Generate questions
                for _ in range(n_per_prompt):
                    q_type = random.choice(question_types)
                    
                    if q_type == "intermediate_result" and step_result:
                        question = random.choice(INTERMEDIATE_RESULT_TEMPLATES)
                        answer = step_result
                    elif q_type == "operation" and operation != "unknown":
                        question = random.choice(OPERATION_TYPE_TEMPLATES)
                        answer = operation
                    elif q_type == "structure":
                        question = random.choice(STRUCTURE_TEMPLATES)
                        # z3 and z5 should be calculation steps
                        answer = "calculation step" if lat_pos in [2, 4] else "transitional"
                    else:
                        continue
                    
                    # Format oracle prompt using the configured placeholder token
                    from ..activation_oracle import format_oracle_prompt
                    oracle_prompt = format_oracle_prompt(
                        question=question,
                        num_activations=1,
                        layer_percent=self.layer_percent,
                        placeholder_token=self.placeholder_token,
                    )
                    
                    example = LatentQAExample(
                        prompt=oracle_prompt,
                        latent_vector=latent_vec.tolist(),
                        latent_position=lat_pos,
                        layer_percent=self.layer_percent,
                        question=question,
                        answer=answer,
                        source_prompt=prompt,
                        cot_step=cot_step,
                        question_type=q_type,
                    )
                    examples.append(example)
        
        return examples
    
    def generate_from_gsm8k(
        self,
        data_path: str,
        n_samples: Optional[int] = None,
        n_per_prompt: int = 5,
    ) -> list[LatentQAExample]:
        """
        Generate QA examples from GSM8k dataset.
        
        Args:
            data_path: Path to GSM8k JSONL file
            n_samples: Max number of samples to process (None = all)
            n_per_prompt: Number of QA pairs per prompt
        
        Returns:
            List of LatentQAExample objects
        """
        prompts = []
        
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if n_samples and i >= n_samples:
                    break
                
                item = json.loads(line)
                
                # GSM8k format
                question = item.get("question", item.get("input", ""))
                cot = item.get("cot", item.get("rationale", ""))
                answer = item.get("answer", item.get("output", ""))
                
                # Parse CoT steps
                cot_steps = []
                results = []
                
                # Try GSM8k-Aug format: <<expr=result>>
                matches = re.findall(r'<<([^>]+)>>', cot)
                if matches:
                    for match in matches:
                        cot_steps.append(match)
                        if '=' in match:
                            results.append(match.split('=')[-1].strip())
                
                prompts.append({
                    "prompt": question + " Give the answer only and nothing else.",
                    "cot_steps": cot_steps,
                    "results": results,
                    "final_answer": answer,
                })
        
        return self.generate_from_prompts(
            prompts=prompts,
            n_per_prompt=n_per_prompt,
        )


def save_dataset(examples: list[LatentQAExample], path: str):
    """Save dataset to JSONL file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    
    print(f"Saved {len(examples)} examples to {filepath}")


def load_dataset(path: str) -> list[LatentQAExample]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(LatentQAExample.from_dict(json.loads(line)))
    return examples


def create_synthetic_examples(
    n: int = 100,
    placeholder_token: Optional[str] = None,
) -> list[LatentQAExample]:
    """
    Create synthetic examples for testing without running CODI.
    
    Uses random vectors (not meaningful, just for testing pipeline).
    
    Args:
        n: Number of examples to create
        placeholder_token: Placeholder token to use. If None, uses default.
    """
    from ..activation_oracle import DEFAULT_PLACEHOLDER_TOKEN, format_oracle_prompt
    
    placeholder_token = placeholder_token or DEFAULT_PLACEHOLDER_TOKEN
    examples = []
    
    for i in range(n):
        # Random "intermediate result"
        result = str(random.randint(1, 100))
        operation = random.choice(["addition", "subtraction", "multiplication", "division"])
        
        question = random.choice(INTERMEDIATE_RESULT_TEMPLATES)
        oracle_prompt = format_oracle_prompt(
            question=question,
            num_activations=1,
            layer_percent=50,
            placeholder_token=placeholder_token,
        )
        
        example = LatentQAExample(
            prompt=oracle_prompt,
            latent_vector=torch.randn(2048).tolist(),  # Random vector
            latent_position=random.choice([2, 4]),
            layer_percent=50,
            question=question,
            answer=result,
            source_prompt=f"Synthetic problem {i}",
            cot_step=f"{random.randint(1,10)}+{random.randint(1,10)}={result}",
            question_type="intermediate_result",
        )
        examples.append(example)
    
    return examples


if __name__ == "__main__":
    # Test synthetic generation
    print("Creating synthetic examples...")
    examples = create_synthetic_examples(10)
    
    print(f"\nGenerated {len(examples)} examples")
    print("\nExample:")
    ex = examples[0]
    print(f"  Prompt: {ex.prompt}")
    print(f"  Question: {ex.question}")
    print(f"  Answer: {ex.answer}")
    print(f"  Latent position: {ex.latent_position}")
    print(f"  Vector shape: {len(ex.latent_vector)}")
