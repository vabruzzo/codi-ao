"""
Latent-to-CoT QA Dataset Generator.

Generates question-answer pairs by aligning CODI's latent vectors
with the intermediate steps from the teacher's explicit CoT.

Latent position mapping (for 3-step math problems):
- z2 (index 1) stores Step 1 intermediate result
- z4 (index 3) stores Step 2 intermediate result

Note: LessWrong says "z3/z5" but their indexing includes initial position.
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
    """A single training example for Latent QA.
    
    Supports both single-latent (one vector) and multi-latent (all 6 vectors) examples.
    """
    
    # Input
    prompt: str  # The oracle prompt (with placeholders)
    latent_vectors: list[list[float]]  # List of latent vectors (1 for single, 6 for multi)
    latent_positions: list[int]  # Which latent positions (e.g., [1] or [0,1,2,3,4,5])
    
    # Target
    question: str  # The question being asked
    answer: str  # The expected answer
    
    # Metadata
    source_prompt: str = ""  # Original math problem
    cot_step: str = ""  # The CoT step this latent aligns to
    question_type: str = ""  # "intermediate_result", "operation", "structure", "full_reasoning"
    is_multi_latent: bool = False  # True if using all 6 vectors
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "LatentQAExample":
        # Handle backward compatibility with old format
        if "latent_vector" in d and "latent_vectors" not in d:
            d["latent_vectors"] = [d.pop("latent_vector")]
        if "latent_position" in d and "latent_positions" not in d:
            d["latent_positions"] = [d.pop("latent_position")]
        if "layer_percent" in d:
            d.pop("layer_percent")  # Remove deprecated field
        if "is_multi_latent" not in d:
            d["is_multi_latent"] = len(d.get("latent_vectors", [])) > 1
        return cls(**d)


# =============================================================================
# QUESTION TEMPLATES - Organized by Type for Phase 3 Diverse Training
# =============================================================================

# -----------------------------------------------------------------------------
# 1. EXTRACTION QUESTIONS (Single-latent) - What value is stored?
# -----------------------------------------------------------------------------

# Generic extraction (no position info - original style)
EXTRACTION_GENERIC = [
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
]

# Position-aware extraction for Step 1 (z2)
EXTRACTION_STEP1 = [
    "What is the first intermediate result?",
    "What was calculated in step 1?",
    "What is the step 1 result?",
    "What number was computed first?",
    "What is the first calculation's output?",
    "What value did the first step produce?",
    "What is stored from the first computation?",
    "What result came from step one?",
]

# Position-aware extraction for Step 2 (z4)
EXTRACTION_STEP2 = [
    "What is the second intermediate result?",
    "What was calculated in step 2?",
    "What is the step 2 result?",
    "What number was computed second?",
    "What is the second calculation's output?",
    "What value did the second step produce?",
    "What is stored from the second computation?",
    "What result came from step two?",
]

# Combine all extraction templates
INTERMEDIATE_RESULT_TEMPLATES = EXTRACTION_GENERIC  # Backward compatibility

# -----------------------------------------------------------------------------
# 2. CLASSIFICATION QUESTIONS (Yes/No) - Properties of the latent
# -----------------------------------------------------------------------------

# Magnitude classification
CLASSIFICATION_MAGNITUDE = {
    "greater_than_10": [
        "Is the result greater than 10?",
        "Is this value more than 10?",
        "Does this computation produce a result above 10?",
    ],
    "greater_than_50": [
        "Is the result greater than 50?",
        "Is this value more than 50?",
        "Does this step produce a number above 50?",
    ],
    "greater_than_100": [
        "Is the result greater than 100?",
        "Is this value more than 100?",
        "Is the computed value above 100?",
    ],
    "less_than_10": [
        "Is the result less than 10?",
        "Is this value below 10?",
        "Does this produce a single digit result?",
    ],
    "is_positive": [
        "Is the result positive?",
        "Is this a positive number?",
        "Is the computed value greater than zero?",
    ],
    "is_negative": [
        "Is the result negative?",
        "Is this a negative number?",
        "Is the value below zero?",
    ],
}

# Operation classification
CLASSIFICATION_OPERATION = {
    "is_addition": [
        "Was addition used in this step?",
        "Is this an addition operation?",
        "Did this step involve adding?",
    ],
    "is_subtraction": [
        "Was subtraction used in this step?",
        "Is this a subtraction operation?",
        "Did this step involve subtracting?",
    ],
    "is_multiplication": [
        "Was multiplication used in this step?",
        "Is this a multiplication operation?",
        "Did this step involve multiplying?",
    ],
}

# Position classification
CLASSIFICATION_POSITION = {
    "is_first_step": [
        "Is this the first calculation step?",
        "Is this step 1?",
        "Is this one of the first calculation steps?",
    ],
    "is_second_step": [
        "Is this the second calculation step?",
        "Is this step 2?",
        "Is this the second computation?",
    ],
}

# Structure classification
CLASSIFICATION_STRUCTURE = [
    "Is this a computational step?",
    "Does this step perform a calculation?",
    "Is this an actual computation?",
    "Is meaningful work done in this step?",
]

# -----------------------------------------------------------------------------
# 3. OPERATION TYPE QUESTIONS - What operation was performed?
# -----------------------------------------------------------------------------

OPERATION_TYPE_TEMPLATES = [
    "What mathematical operation was performed?",
    "What type of calculation was done?",
    "What operation produced this result?",
    "What arithmetic operation was used?",
    "What kind of math operation is this?",
    "Was this addition, subtraction, or multiplication?",
]

# -----------------------------------------------------------------------------
# 4. MULTI-LATENT QUESTIONS - Require all 6 latent vectors
# -----------------------------------------------------------------------------

# Step extraction with full context
MULTI_LATENT_EXTRACTION = [
    "What was calculated in the first step?",
    "What was the result of the second calculation?",
    "What intermediate results were computed?",
    "What is stored in the first reasoning position?",
    "What is stored in the second reasoning position?",
]

# Comparison questions
MULTI_LATENT_COMPARISON = [
    "Which calculation step produced the larger result?",
    "Is the second step result greater than the first?",
    "Which step has the smaller value?",
    "Is step 1's result larger than step 2's?",
    "Compare the two intermediate results: which is bigger?",
]

# Relationship questions
MULTI_LATENT_RELATIONSHIP = [
    "What is the difference between step 1 and step 2?",
    "How do the intermediate results relate?",
    "Is step 2 a multiple of step 1?",
]

# Legacy multi-latent templates (backward compatibility)
MULTI_LATENT_TEMPLATES = MULTI_LATENT_EXTRACTION + MULTI_LATENT_COMPARISON

STRUCTURE_TEMPLATES = CLASSIFICATION_STRUCTURE  # Backward compatibility


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
        
        # Latent position to CoT step mapping
        # For 3-step problems: z2 → step1, z4 → step2
        self.position_to_step = {
            1: 0,  # z2 → first intermediate result
            3: 1,  # z4 → second intermediate result
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
                        # z2 and z4 should be calculation steps
                        answer = "calculation step" if lat_pos in [1, 3] else "transitional"
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
            latent_position=random.choice([1, 3]),  # z2 or z4
            layer_percent=50,
            question=question,
            answer=result,
            source_prompt=f"Synthetic problem {i}",
            cot_step=f"{random.randint(1,10)}+{random.randint(1,10)}={result}",
            question_type="intermediate_result",
        )
        examples.append(example)
    
    return examples


# =============================================================================
# PHASE 3: Diverse Training Data Generator
# =============================================================================

class Phase3DataGenerator:
    """
    Comprehensive data generator for Phase 3 training.
    
    Generates diverse question types:
    1. Extraction (generic and position-aware)
    2. Classification (magnitude, operation, position, structure)
    3. Comparison (multi-latent)
    4. Operation identification
    
    Target: 100K+ diverse examples from real CODI latents.
    """
    
    def __init__(
        self,
        codi_wrapper: Optional["CODIWrapper"] = None,
        placeholder_token: Optional[str] = None,
    ):
        from ..activation_oracle import DEFAULT_PLACEHOLDER_TOKEN, format_oracle_prompt
        
        self.codi_wrapper = codi_wrapper
        self.placeholder_token = placeholder_token or DEFAULT_PLACEHOLDER_TOKEN
        self.format_oracle_prompt = format_oracle_prompt
        
        # Question type weights (controls distribution)
        self.question_weights = {
            "extraction_generic": 0.15,
            "extraction_position": 0.15,
            "classification_magnitude": 0.20,
            "classification_operation": 0.10,
            "classification_position": 0.10,
            "classification_structure": 0.05,
            "operation_type": 0.10,
            "multi_latent_extraction": 0.05,
            "multi_latent_comparison": 0.10,
        }
    
    def generate_diverse_examples(
        self,
        prompts: list[dict],
        target_count: int = 100000,
        verbose: bool = True,
    ) -> list[LatentQAExample]:
        """
        Generate diverse training examples from prompts.
        
        Args:
            prompts: List of dicts with 'prompt', 'step1_result', 'step2_result', etc.
            target_count: Target number of examples to generate
            verbose: Show progress
        
        Returns:
            List of diverse LatentQAExample objects
        """
        if self.codi_wrapper is None:
            raise ValueError("CODI wrapper required for real latent collection")
        
        examples = []
        examples_per_prompt = max(1, target_count // len(prompts))
        
        iterator = tqdm(prompts, desc="Generating Phase 3 data") if verbose else prompts
        
        for item in iterator:
            prompt = item["prompt"]
            
            # Collect latent vectors
            try:
                result = self.codi_wrapper.collect_latents(prompt)
            except Exception:
                continue
            
            if len(result.latent_vectors) < 6:
                continue
            
            # Extract ground truth values
            step1_result = str(item.get("step1_result", ""))
            step2_result = str(item.get("step2_result", ""))
            final_answer = str(item.get("final_answer", ""))
            op_type = item.get("type", "addition")  # "addition" or "subtraction"
            
            # Get latent vectors
            z2_vec = result.latent_vectors[1]  # Step 1
            z4_vec = result.latent_vectors[3]  # Step 2
            all_vecs = [v for v in result.latent_vectors[:6]]
            
            # Generate diverse examples for this prompt
            prompt_examples = []
            
            # 1. EXTRACTION - Generic (single latent)
            if step1_result:
                prompt_examples.extend(self._generate_extraction_examples(
                    z2_vec, step1_result, 1, prompt, "generic", examples_per_prompt // 10
                ))
            if step2_result:
                prompt_examples.extend(self._generate_extraction_examples(
                    z4_vec, step2_result, 3, prompt, "generic", examples_per_prompt // 10
                ))
            
            # 2. EXTRACTION - Position-aware (single latent)
            if step1_result:
                prompt_examples.extend(self._generate_extraction_examples(
                    z2_vec, step1_result, 1, prompt, "step1", examples_per_prompt // 10
                ))
            if step2_result:
                prompt_examples.extend(self._generate_extraction_examples(
                    z4_vec, step2_result, 3, prompt, "step2", examples_per_prompt // 10
                ))
            
            # 3. CLASSIFICATION - Magnitude
            if step1_result:
                prompt_examples.extend(self._generate_magnitude_examples(
                    z2_vec, step1_result, 1, prompt, examples_per_prompt // 10
                ))
            if step2_result:
                prompt_examples.extend(self._generate_magnitude_examples(
                    z4_vec, step2_result, 3, prompt, examples_per_prompt // 10
                ))
            
            # 4. CLASSIFICATION - Operation
            prompt_examples.extend(self._generate_operation_class_examples(
                z2_vec, op_type, 1, prompt, examples_per_prompt // 10
            ))
            prompt_examples.extend(self._generate_operation_class_examples(
                z4_vec, "multiplication", 3, prompt, examples_per_prompt // 10  # Step 2 is always multiplication
            ))
            
            # 5. CLASSIFICATION - Position
            prompt_examples.extend(self._generate_position_examples(
                z2_vec, 1, prompt, examples_per_prompt // 10
            ))
            prompt_examples.extend(self._generate_position_examples(
                z4_vec, 3, prompt, examples_per_prompt // 10
            ))
            
            # 6. CLASSIFICATION - Structure
            prompt_examples.extend(self._generate_structure_examples(
                z2_vec, 1, prompt, examples_per_prompt // 20
            ))
            prompt_examples.extend(self._generate_structure_examples(
                z4_vec, 3, prompt, examples_per_prompt // 20
            ))
            
            # 7. OPERATION TYPE
            prompt_examples.extend(self._generate_operation_type_examples(
                z2_vec, op_type, 1, prompt, examples_per_prompt // 10
            ))
            
            # 8. MULTI-LATENT - Extraction
            if step1_result and step2_result:
                prompt_examples.extend(self._generate_multi_extraction_examples(
                    all_vecs, step1_result, step2_result, prompt, examples_per_prompt // 10
                ))
            
            # 9. MULTI-LATENT - Comparison
            if step1_result and step2_result:
                prompt_examples.extend(self._generate_comparison_examples(
                    all_vecs, step1_result, step2_result, prompt, examples_per_prompt // 10
                ))
            
            examples.extend(prompt_examples)
            
            # Check if we've reached target
            if len(examples) >= target_count:
                break
        
        # Shuffle and trim to target
        random.shuffle(examples)
        return examples[:target_count]
    
    def _generate_extraction_examples(
        self, latent_vec, result, position, source_prompt, extraction_type, count
    ) -> list[LatentQAExample]:
        """Generate extraction question examples."""
        examples = []
        
        if extraction_type == "generic":
            templates = EXTRACTION_GENERIC
        elif extraction_type == "step1":
            templates = EXTRACTION_STEP1
        elif extraction_type == "step2":
            templates = EXTRACTION_STEP2
        else:
            templates = EXTRACTION_GENERIC
        
        for _ in range(count):
            question = random.choice(templates)
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[latent_vec.tolist()],
                latent_positions=[position],
                question=question,
                answer=result,
                source_prompt=source_prompt,
                question_type=f"extraction_{extraction_type}",
            ))
        
        return examples
    
    def _generate_magnitude_examples(
        self, latent_vec, result, position, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate magnitude classification examples."""
        examples = []
        
        try:
            value = float(result)
        except ValueError:
            return examples
        
        magnitude_checks = [
            ("greater_than_10", value > 10),
            ("greater_than_50", value > 50),
            ("greater_than_100", value > 100),
            ("less_than_10", value < 10),
            ("is_positive", value > 0),
        ]
        
        for _ in range(count):
            check_type, is_true = random.choice(magnitude_checks)
            question = random.choice(CLASSIFICATION_MAGNITUDE[check_type])
            answer = "Yes" if is_true else "No"
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[latent_vec.tolist()],
                latent_positions=[position],
                question=question,
                answer=answer,
                source_prompt=source_prompt,
                question_type="classification_magnitude",
            ))
        
        return examples
    
    def _generate_operation_class_examples(
        self, latent_vec, actual_op, position, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate operation classification examples."""
        examples = []
        
        op_checks = [
            ("is_addition", actual_op == "addition"),
            ("is_subtraction", actual_op == "subtraction"),
            ("is_multiplication", actual_op == "multiplication"),
        ]
        
        for _ in range(count):
            check_type, is_true = random.choice(op_checks)
            question = random.choice(CLASSIFICATION_OPERATION[check_type])
            answer = "Yes" if is_true else "No"
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[latent_vec.tolist()],
                latent_positions=[position],
                question=question,
                answer=answer,
                source_prompt=source_prompt,
                question_type="classification_operation",
            ))
        
        return examples
    
    def _generate_position_examples(
        self, latent_vec, position, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate position classification examples."""
        examples = []
        
        is_step1 = position == 1
        is_step2 = position == 3
        
        pos_checks = [
            ("is_first_step", is_step1),
            ("is_second_step", is_step2),
        ]
        
        for _ in range(count):
            check_type, is_true = random.choice(pos_checks)
            question = random.choice(CLASSIFICATION_POSITION[check_type])
            answer = "Yes" if is_true else "No"
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[latent_vec.tolist()],
                latent_positions=[position],
                question=question,
                answer=answer,
                source_prompt=source_prompt,
                question_type="classification_position",
            ))
        
        return examples
    
    def _generate_structure_examples(
        self, latent_vec, position, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate structure classification examples."""
        examples = []
        
        # z2 and z4 are computation steps
        is_computation = position in [1, 3]
        
        for _ in range(count):
            question = random.choice(CLASSIFICATION_STRUCTURE)
            answer = "Yes" if is_computation else "No"
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[latent_vec.tolist()],
                latent_positions=[position],
                question=question,
                answer=answer,
                source_prompt=source_prompt,
                question_type="classification_structure",
            ))
        
        return examples
    
    def _generate_operation_type_examples(
        self, latent_vec, operation, position, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate operation type identification examples."""
        examples = []
        
        for _ in range(count):
            question = random.choice(OPERATION_TYPE_TEMPLATES)
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=1,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[latent_vec.tolist()],
                latent_positions=[position],
                question=question,
                answer=operation,
                source_prompt=source_prompt,
                question_type="operation_type",
            ))
        
        return examples
    
    def _generate_multi_extraction_examples(
        self, all_vecs, step1_result, step2_result, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate multi-latent extraction examples."""
        examples = []
        
        step_answers = [
            ("What was calculated in the first step?", step1_result),
            ("What was the result of the second calculation?", step2_result),
            ("What is stored in the first reasoning position?", step1_result),
            ("What is stored in the second reasoning position?", step2_result),
        ]
        
        for _ in range(count):
            question, answer = random.choice(step_answers)
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=6,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[v.tolist() for v in all_vecs],
                latent_positions=[0, 1, 2, 3, 4, 5],
                question=question,
                answer=answer,
                source_prompt=source_prompt,
                question_type="multi_latent_extraction",
                is_multi_latent=True,
            ))
        
        return examples
    
    def _generate_comparison_examples(
        self, all_vecs, step1_result, step2_result, source_prompt, count
    ) -> list[LatentQAExample]:
        """Generate multi-latent comparison examples."""
        examples = []
        
        try:
            v1 = float(step1_result)
            v2 = float(step2_result)
        except ValueError:
            return examples
        
        # Determine comparison answers
        if v2 > v1:
            larger_step = "step 2"
            smaller_step = "step 1"
            step2_greater = "Yes"
            step1_greater = "No"
        elif v1 > v2:
            larger_step = "step 1"
            smaller_step = "step 2"
            step2_greater = "No"
            step1_greater = "Yes"
        else:
            larger_step = "equal"
            smaller_step = "equal"
            step2_greater = "No"
            step1_greater = "No"
        
        comparison_qa = [
            ("Which calculation step produced the larger result?", larger_step),
            ("Which step has the smaller value?", smaller_step),
            ("Is the second step result greater than the first?", step2_greater),
            ("Is step 1's result larger than step 2's?", step1_greater),
        ]
        
        for _ in range(count):
            question, answer = random.choice(comparison_qa)
            
            oracle_prompt = self.format_oracle_prompt(
                question=question,
                num_activations=6,
                placeholder_token=self.placeholder_token,
            )
            
            examples.append(LatentQAExample(
                prompt=oracle_prompt,
                latent_vectors=[v.tolist() for v in all_vecs],
                latent_positions=[0, 1, 2, 3, 4, 5],
                question=question,
                answer=answer,
                source_prompt=source_prompt,
                question_type="multi_latent_comparison",
                is_multi_latent=True,
            ))
        
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
    print(f"  Latent positions: {ex.latent_positions}")
    print(f"  Vector count: {len(ex.latent_vectors)}")
