"""
CODI Wrapper for Activation Oracle training.

This module provides utilities to:
1. Load a pre-trained CODI model
2. Run both teacher (explicit CoT) and student (latent CoT) tasks
3. Collect latent vectors at each position with metadata
4. Perform logit lens analysis on latent vectors
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import our local CODI model implementation
from .codi_model import CODI


@dataclass
class LatentCollectionResult:
    """Result of running CODI and collecting latent vectors."""

    # Input
    prompt: str
    ground_truth_answer: Optional[str] = None

    # Latent vectors - pre-projection (for logit lens / interpretation)
    latent_vectors: list[torch.Tensor] = field(
        default_factory=list
    )  # List of (hidden_dim,) tensors - these are interpretable via logit lens
    latent_vectors_post_prj: list[torch.Tensor] = field(default_factory=list)  # After projection (input to next iteration)

    # Teacher task outputs (explicit CoT)
    teacher_cot: Optional[str] = None
    cot_steps: list[str] = field(default_factory=list)  # Parsed individual steps
    intermediate_results: list[str] = field(default_factory=list)  # Extracted numbers

    # Student task outputs
    predicted_answer: Optional[str] = None
    is_correct: Optional[bool] = None

    # Hidden states for all positions (prompt + latent + generated)
    # Shape: (seq_len, num_layers, hidden_dim)
    hidden_states: Optional[torch.Tensor] = None

    # Metadata
    num_latent_iterations: int = 6


@dataclass
class LogitLensResult:
    """Result of logit lens analysis on a single latent vector."""

    latent_index: int
    layer_results: list[dict] = field(default_factory=list)  # Per-layer top-k tokens and probs

    def get_top1_token(self, layer: int) -> tuple[str, float]:
        """Get top-1 token and probability at a specific layer."""
        for lr in self.layer_results:
            if lr["layer"] == layer:
                return lr["top_tokens"][0], lr["top_probs"][0]
        return None, 0.0

    def get_top1_at_final_layer(self) -> tuple[str, float]:
        """Get top-1 token at the final layer."""
        if self.layer_results:
            final = self.layer_results[-1]
            return final["top_tokens"][0], final["top_probs"][0]
        return None, 0.0


class CODIWrapper:
    """
    Wrapper for CODI model that provides utilities for AO training.

    Usage:
        wrapper = CODIWrapper.from_pretrained(
            checkpoint_path="bcywinski/codi_llama1b-answer_only",
            model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
        )

        result = wrapper.collect_latents(
            prompt="A team starts with 3 members...",
            ground_truth="24"
        )

        # Access latent vectors
        z3 = result.latent_vectors[2]  # Step 1 result (per LessWrong)
        z5 = result.latent_vectors[4]  # Step 2 result
    """

    def __init__(
        self,
        model: CODI,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_latent_iterations: int = 6,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.num_latent_iterations = num_latent_iterations

        # Ensure special tokens are set up
        self._setup_special_tokens()

        # Cache model components for logit lens
        self._lm_head = self._get_lm_head()
        self._layer_norm = self._get_layer_norm()
        self._embed_tokens = self._get_embed_tokens()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str = "bcywinski/codi_llama1b-answer_only",
        model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
        lora_r: int = 128,
        lora_alpha: int = 32,
        num_latent: int = 6,
        use_prj: bool = True,
        device: str = "cuda",
        dtype: str = "bfloat16",
        checkpoint_save_path: Optional[str] = None,
        **kwargs,
    ) -> "CODIWrapper":
        """Load a pre-trained CODI model."""

        if checkpoint_save_path is None:
            checkpoint_save_path = f"./checkpoints/{checkpoint_path.replace('/', '_')}"

        model = CODI.from_pretrained(
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_name_or_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_latent=num_latent,
            use_prj=use_prj,
            device=device,
            dtype=dtype,
            strict=False,
            checkpoint_save_path=checkpoint_save_path,
            remove_eos=False,
            full_precision=True,
            **kwargs,
        )

        tokenizer = model.tokenizer
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=torch_dtype,
            num_latent_iterations=num_latent,
        )

    def _setup_special_tokens(self):
        """Ensure tokenizer has required special tokens."""
        # Special tokens should already be set up by CODI.from_pretrained()
        # Just get the token IDs
        self.bocot_id = self.tokenizer.convert_tokens_to_ids("<|bocot|>")
        self.eocot_id = self.tokenizer.convert_tokens_to_ids("<|eocot|>")

    def _get_lm_head(self):
        """Get the language model head (unembedding matrix)."""
        codi = self.model.codi
        if hasattr(codi, "get_base_model"):
            return codi.get_base_model().lm_head
        return codi.lm_head

    def _get_layer_norm(self):
        """Get the final layer norm before the lm_head."""
        codi = self.model.codi
        if hasattr(codi, "get_base_model"):
            base = codi.get_base_model()
        else:
            base = codi

        if hasattr(base, "model") and hasattr(base.model, "norm"):
            return base.model.norm
        if hasattr(base, "transformer") and hasattr(base.transformer, "ln_f"):
            return base.transformer.ln_f
        return None

    def _get_embed_tokens(self):
        """Get the token embedding matrix."""
        codi = self.model.codi
        if hasattr(codi, "get_base_model"):
            base = codi.get_base_model()
        else:
            base = codi

        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens
        if hasattr(base, "transformer") and hasattr(base.transformer, "wte"):
            return base.transformer.wte
        return None

    @property
    def num_layers(self) -> int:
        """Get the number of layers in the model."""
        return self.model.codi.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.model.codi.config.hidden_size

    def collect_latents(
        self,
        prompt: str,
        ground_truth_answer: Optional[str] = None,
        max_new_tokens: int = 32,
        return_hidden_states: bool = True,
        return_pre_projection: bool = True,
    ) -> LatentCollectionResult:
        """
        Run CODI on a prompt and collect all latent vectors.

        Args:
            prompt: The input question/prompt
            ground_truth_answer: Optional ground truth for correctness checking
            max_new_tokens: Maximum tokens to generate for the answer
            return_hidden_states: Whether to return full hidden states
            return_pre_projection: Whether to return pre-projection latent vectors

        Returns:
            LatentCollectionResult with all collected data
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run student task (latent CoT)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                num_latent_iterations=self.num_latent_iterations,
                temperature=0.1,
                top_k=40,
                top_p=0.95,
                greedy=True,
                return_latent_vectors=True,
                remove_eos=False,
                output_hidden_states=return_hidden_states,
                sot_token=self.bocot_id,
                eot_token=self.eocot_id,
            )

        # Extract latent vectors
        # latent_vectors: pre-projection (for logit lens - interpretable)
        # latent_vectors_post_prj: post-projection (input to next iteration)
        latent_vectors = []
        latent_vectors_post_prj = []

        if "latent_vectors" in output:
            for lv in output["latent_vectors"]:
                # Squeeze batch and sequence dimensions
                vec = lv.squeeze(0).squeeze(0).cpu()
                latent_vectors.append(vec)

        if "latent_vectors_post_prj" in output:
            for lv in output["latent_vectors_post_prj"]:
                vec = lv.squeeze(0).squeeze(0).cpu()
                latent_vectors_post_prj.append(vec)

        # Decode predicted answer
        predicted_answer = self.tokenizer.decode(
            output["sequences"][0], skip_special_tokens=True
        ).strip()

        # Check correctness
        is_correct = None
        if ground_truth_answer is not None:
            is_correct = self._check_answer(predicted_answer, ground_truth_answer)

        # Get hidden states
        hidden_states = None
        if return_hidden_states and "hidden_states" in output:
            hidden_states = output["hidden_states"].cpu()

        return LatentCollectionResult(
            prompt=prompt,
            ground_truth_answer=ground_truth_answer,
            latent_vectors=latent_vectors,  # Pre-projection (for logit lens)
            latent_vectors_post_prj=latent_vectors_post_prj,  # Post-projection
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            hidden_states=hidden_states,
            num_latent_iterations=self.num_latent_iterations,
        )

    def run_teacher_task(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> tuple[str, list[str], list[str]]:
        """
        Run the teacher task (explicit CoT) and parse the output.

        Returns:
            tuple of (full_cot_string, list_of_steps, list_of_intermediate_results)
        """
        # Tokenize with CoT generation
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                verbalize_cot=True,  # Use teacher mode
                sot_token=self.bocot_id,
                eot_token=self.eocot_id,
            )

        cot_text = self.tokenizer.decode(output["sequences"][0], skip_special_tokens=True)

        # Parse CoT steps
        steps, results = self._parse_cot_steps(cot_text)

        return cot_text, steps, results

    def _parse_cot_steps(self, cot_text: str) -> tuple[list[str], list[str]]:
        """
        Parse CoT text into individual steps and intermediate results.

        GSM8k-Aug format: <<600*30/100=180>> <<600*10/100=60>> ...
        GSM8k-NL format: Natural language with calculations
        """
        steps = []
        results = []

        # Try GSM8k-Aug format first: <<expr=result>>
        aug_pattern = r"<<([^>]+)>>"
        matches = re.findall(aug_pattern, cot_text)

        if matches:
            for match in matches:
                steps.append(match)
                # Extract result after '='
                if "=" in match:
                    result = match.split("=")[-1].strip()
                    results.append(result)
        else:
            # Try to extract numbers from natural language
            # Look for "= X" patterns
            eq_pattern = r"=\s*([-]?\d+\.?\d*)"
            matches = re.findall(eq_pattern, cot_text)
            results = matches

            # Split by sentences or common delimiters
            sentences = re.split(r"[.;]", cot_text)
            steps = [s.strip() for s in sentences if s.strip()]

        return steps, results

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        # Extract numbers from both
        pred_nums = re.findall(r"[-]?\d+\.?\d*", predicted)
        gt_nums = re.findall(r"[-]?\d+\.?\d*", ground_truth)

        if pred_nums and gt_nums:
            try:
                return float(pred_nums[-1]) == float(gt_nums[-1])
            except ValueError:
                pass

        return predicted.strip().lower() == ground_truth.strip().lower()

    def logit_lens(
        self,
        latent_vector: torch.Tensor,
        top_k: int = 10,
        apply_layer_norm: bool = True,
    ) -> LogitLensResult:
        """
        Apply logit lens to a single latent vector.

        This projects the latent vector through the unembedding matrix
        to see what tokens it most strongly represents.

        Args:
            latent_vector: A single latent vector of shape (hidden_dim,)
            top_k: Number of top tokens to return
            apply_layer_norm: Whether to apply layer norm before projection

        Returns:
            LogitLensResult with top-k tokens and probabilities
        """
        vec = latent_vector.to(self.device).unsqueeze(0)  # (1, hidden_dim)

        with torch.no_grad():
            if apply_layer_norm and self._layer_norm is not None:
                vec = self._layer_norm(vec)

            logits = self._lm_head(vec)  # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

            top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices[0]]
            top_probs_list = top_probs[0].cpu().tolist()

        return LogitLensResult(
            latent_index=-1,  # Set by caller
            layer_results=[
                {
                    "layer": self.num_layers - 1,  # Final layer
                    "top_tokens": top_tokens,
                    "top_probs": top_probs_list,
                    "top_indices": top_indices[0].cpu().tolist(),
                }
            ],
        )

    def logit_lens_all_layers(
        self,
        hidden_states: torch.Tensor,
        position: int,
        top_k: int = 10,
        apply_layer_norm: bool = True,
    ) -> LogitLensResult:
        """
        Apply logit lens at all layers for a specific position.

        Args:
            hidden_states: Full hidden states, shape (seq_len, num_layers, hidden_dim)
            position: The sequence position to analyze
            top_k: Number of top tokens to return
            apply_layer_norm: Whether to apply layer norm before projection

        Returns:
            LogitLensResult with top-k tokens at each layer
        """
        layer_results = []

        with torch.no_grad():
            for layer_idx in range(hidden_states.shape[1]):
                vec = hidden_states[position, layer_idx, :].to(self.device).unsqueeze(0)

                if apply_layer_norm and self._layer_norm is not None:
                    vec = self._layer_norm(vec)

                logits = self._lm_head(vec)
                probs = F.softmax(logits, dim=-1)

                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

                top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices[0]]

                layer_results.append(
                    {
                        "layer": layer_idx,
                        "top_tokens": top_tokens,
                        "top_probs": top_probs[0].cpu().tolist(),
                        "top_indices": top_indices[0].cpu().tolist(),
                    }
                )

        return LogitLensResult(
            latent_index=position,
            layer_results=layer_results,
        )

    def verify_latent_positions(
        self,
        prompts: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        Verify that z3 and z5 store intermediate results as per LessWrong findings.

        Args:
            prompts: List of dicts with 'prompt', 'step1_result', 'step2_result' keys
            verbose: Whether to print progress

        Returns:
            Dict with accuracy metrics for z3 and z5
        """
        z3_correct = 0
        z5_correct = 0
        total = 0

        for item in prompts:
            result = self.collect_latents(item["prompt"])

            if len(result.latent_vectors) < 5:
                continue

            # Check z3 (index 2) for step 1 result
            z3_lens = self.logit_lens(result.latent_vectors[2])
            z3_top1, z3_prob = z3_lens.get_top1_at_final_layer()

            # Check z5 (index 4) for step 2 result
            z5_lens = self.logit_lens(result.latent_vectors[4])
            z5_top1, z5_prob = z5_lens.get_top1_at_final_layer()

            step1_result = str(item.get("step1_result", ""))
            step2_result = str(item.get("step2_result", ""))

            if z3_top1 and step1_result in z3_top1:
                z3_correct += 1

            if z5_top1 and step2_result in z5_top1:
                z5_correct += 1

            total += 1

            if verbose and total % 10 == 0:
                print(
                    f"Processed {total}/{len(prompts)}, "
                    f"z3 acc: {z3_correct / total:.2%}, "
                    f"z5 acc: {z5_correct / total:.2%}"
                )

        return {
            "z3_accuracy": z3_correct / total if total > 0 else 0,
            "z5_accuracy": z5_correct / total if total > 0 else 0,
            "total": total,
        }


def create_test_prompts(n: int = 10) -> list[dict]:
    """
    Create test prompts for validating latent positions.

    These are 3-step math problems where we know the intermediate results.
    """
    import random

    templates = [
        {
            "template": "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
            "step1": lambda x, y, z: x + y,
            "step2": lambda x, y, z: (x + y) * z,
            "final": lambda x, y, z: (x + y) + (x + y) * z,
        },
        {
            "template": "A store has {X} items. They sell {Y}% of them. Then they receive {Z} new items. How many items are in the store now? Give the answer only and nothing else.",
            "step1": lambda x, y, z: int(x * y / 100),
            "step2": lambda x, y, z: x - int(x * y / 100),
            "final": lambda x, y, z: x - int(x * y / 100) + z,
        },
    ]

    prompts = []
    for _ in range(n):
        template = random.choice(templates)
        x = random.randint(2, 10)
        y = random.randint(2, 10)
        z = random.randint(2, 5)

        prompt = template["template"].format(X=x, Y=y, Z=z)
        step1 = template["step1"](x, y, z)
        step2 = template["step2"](x, y, z)
        final = template["final"](x, y, z)

        prompts.append(
            {
                "prompt": prompt,
                "x": x,
                "y": y,
                "z": z,
                "step1_result": step1,
                "step2_result": step2,
                "final_answer": final,
            }
        )

    return prompts


if __name__ == "__main__":
    # Quick test
    print("Loading CODI model...")
    wrapper = CODIWrapper.from_pretrained()

    print("\nTesting latent collection...")
    test_prompt = "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team? Give the answer only and nothing else."

    result = wrapper.collect_latents(prompt=test_prompt, ground_truth_answer="24")

    print(f"Prompt: {test_prompt[:50]}...")
    print(f"Predicted answer: {result.predicted_answer}")
    print(f"Is correct: {result.is_correct}")
    print(f"Number of latent vectors: {len(result.latent_vectors)}")

    # Test logit lens
    print("\nLogit lens on each latent vector:")
    for i, lv in enumerate(result.latent_vectors):
        lens = wrapper.logit_lens(lv, top_k=5)
        top1, prob = lens.get_top1_at_final_layer()
        print(f"  z{i}: top1 = '{top1}' (prob={prob:.3f})")
