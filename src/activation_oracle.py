"""
Activation Oracle for interpreting CODI's latent reasoning vectors.

This module implements the AO model following the methodology from:
"Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers"

Key features:
- Norm-matched additive injection of activation vectors
- Placeholder token format for flexible queries
- LoRA fine-tuning for efficient training
"""

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


# Default special token for activation injection (space + question mark tokenizes as single token)
# This matches the original AO paper's approach for LLaMA models
DEFAULT_PLACEHOLDER_TOKEN = " ?"


@dataclass
class AOConfig:
    """Configuration for Activation Oracle."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    injection_layer: int = 1  # Layer to inject activations (after this layer)
    placeholder_token: str = (
        DEFAULT_PLACEHOLDER_TOKEN  # Token used as placeholder for activation injection
    )

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all"  # "all" for all linear layers

    # Generation settings
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_k: int = 40
    top_p: float = 0.95

    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"


def get_introspection_prefix(
    layer: int | str,
    num_positions: int,
    placeholder_token: str = DEFAULT_PLACEHOLDER_TOKEN,
) -> str:
    """
    Create the introspection prefix with placeholder tokens.

    Format matches original AO paper:
        Layer: {layer}\n
         ? ? ?...\n

    Args:
        layer: Layer identifier (int or string like "50%")
        num_positions: Number of placeholder tokens to include
        placeholder_token: Token to use as placeholder (default: " ?")

    Returns:
        Formatted prefix string
    """
    prefix = f"Layer: {layer}\n"
    prefix += placeholder_token * num_positions
    prefix += "\n"
    return prefix


def validate_repeated_placeholder(
    tokenizer,
    placeholder_token: str,
    num_positions: int,
) -> None:
    """
    Validate that repeated placeholder tokens don't merge during tokenization.

    Some tokenizers may merge repeated patterns (e.g., "??" -> single token).
    This validates that N repetitions produce exactly N tokens.

    Args:
        tokenizer: The tokenizer to validate with
        placeholder_token: The placeholder token string
        num_positions: Number of repetitions to validate

    Raises:
        ValueError: If tokens merge unexpectedly
    """
    if num_positions <= 1:
        return  # Single token already validated by validate_placeholder_token

    repeated = placeholder_token * num_positions
    token_ids = tokenizer.encode(repeated, add_special_tokens=False)

    if len(token_ids) != num_positions:
        raise ValueError(
            f"Repeated placeholder tokens merge during tokenization: "
            f"{num_positions} repetitions of {placeholder_token!r} produced {len(token_ids)} tokens "
            f"(expected {num_positions}). Token IDs: {token_ids}. "
            f"Try a placeholder token with a leading space (e.g., ' ?') to prevent merging."
        )


@dataclass
class AOPrompt:
    """A prompt for the Activation Oracle with activation vectors to inject."""

    text: str  # The prompt text (with placeholder tokens)
    activation_vectors: list[torch.Tensor] = field(default_factory=list)  # Vectors to inject
    placeholder_positions: list[int] = field(
        default_factory=list
    )  # Token positions to inject at (set after tokenization)
    layer: int = -1  # Layer the activations were collected from (-1 = final)
    num_placeholders: int = 0  # Number of placeholder tokens expected

    @classmethod
    def from_question(
        cls,
        question: str,
        activation_vectors: list[torch.Tensor],
        layer: int = -1,
        layer_percent: int = 50,
        placeholder_token: str = DEFAULT_PLACEHOLDER_TOKEN,
    ) -> "AOPrompt":
        """
        Create an AO prompt from a question and activation vectors.

        **IMPORTANT**: If using with an ActivationOracle instance, prefer
        `ao.create_prompt()` instead to ensure the placeholder token matches
        the oracle's configuration. Using this method directly with the default
        placeholder_token may cause mismatches if AOConfig.placeholder_token
        was customized.

        Format matches original AO paper:
            Layer: {layer}\n
             ? ? ?...\n
            {question}

        Args:
            question: The question to ask about the activations
            activation_vectors: List of activation tensors to inject
            layer: Specific layer number (-1 to use layer_percent)
            layer_percent: Layer as percentage of model depth
            placeholder_token: Token to use as placeholder (must match AO config!)
        """
        num_acts = len(activation_vectors)

        if layer == -1:
            layer_str = f"{layer_percent}%"
        else:
            layer_str = str(layer)

        prefix = get_introspection_prefix(layer_str, num_acts, placeholder_token)
        text = prefix + question

        return cls(
            text=text,
            activation_vectors=activation_vectors,
            layer=layer,
            num_placeholders=num_acts,
        )


def get_norm_matched_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    device: torch.device,
    dtype: torch.dtype,
    coefficient: float = 1.0,
) -> Callable:
    """
    Create a forward hook that performs norm-matched additive injection.

    Matches the original AO paper's approach:
    For each batch item b and slot k, replace the residual at token index positions[b][k]
    with normalize(vectors[b][k]) * ||resid[b, positions[b][k], :]|| * coefficient.

    Args:
        vectors: List of activation tensors, one per batch element. Each tensor is (K_b, hidden_dim)
                 where K_b is the number of positions for that sample.
        positions: List of position lists, one per batch element. Each inner list has K_b positions.
        device: Device to perform computation on
        dtype: Data type for computation
        coefficient: Scaling coefficient (default 1.0)

    Returns:
        A forward hook function
    """
    if len(vectors) != len(positions):
        raise ValueError(
            f"vectors and positions must have same batch length: "
            f"got {len(vectors)} vectors, {len(positions)} positions"
        )
    B = len(vectors)

    if B == 0:
        return lambda m, i, o: o

    # Pre-normalize vectors and validate K matches positions
    normed_list = []
    for b, v_b in enumerate(vectors):
        if v_b.dim() == 1:
            v_b = v_b.unsqueeze(0)  # (hidden_dim,) -> (1, hidden_dim)

        K_b = v_b.shape[0]
        num_pos_b = len(positions[b])
        if K_b != num_pos_b:
            raise ValueError(
                f"Batch element {b}: vector count ({K_b}) != position count ({num_pos_b}). "
                f"Each sample must have matching vectors and positions."
            )

        normed_list.append(torch.nn.functional.normalize(v_b.to(device).to(dtype), dim=-1).detach())

    def hook_fn(module, _input, output):
        # Handle tuple vs tensor output
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            is_tuple = True
        else:
            resid_BLD = output
            is_tuple = False

        B_actual, L, d_model = resid_BLD.shape

        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        # Skip if we're in generation mode (seq_len == 1)
        if L <= 1:
            return (resid_BLD, *rest) if is_tuple else resid_BLD

        # Per-batch element injection
        for b in range(B):
            pos_b = positions[b]
            if len(pos_b) == 0:
                continue

            pos_tensor = torch.tensor(pos_b, dtype=torch.long, device=device)

            # Validate positions
            if pos_tensor.min() < 0 or pos_tensor.max() >= L:
                raise IndexError(
                    f"Position out of range for batch {b}: positions={pos_b}, seq_len={L}"
                )

            # Gather original activations at requested slots
            orig_KD = resid_BLD[b, pos_tensor, :]  # (K_b, d)
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # (K_b, 1)

            # Build steered vectors: normalize(v) * ||h|| * coefficient
            steered_KD = (normed_list[b] * norms_K1 * coefficient).to(dtype)  # (K_b, d)

            # Additive injection: h' = h + steered
            resid_BLD[b, pos_tensor, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if is_tuple else resid_BLD

    return hook_fn


@contextlib.contextmanager
def add_hook(module: nn.Module, hook: Callable):
    """Context manager to temporarily add a forward hook to a module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def validate_placeholder_token(tokenizer, placeholder_token: str) -> int:
    """
    Validate that a placeholder token encodes to exactly one token ID.

    Args:
        tokenizer: The tokenizer to use
        placeholder_token: The placeholder token string

    Returns:
        The single token ID

    Raises:
        ValueError: If the token doesn't encode to exactly one ID
    """
    token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Placeholder token {placeholder_token!r} encodes to {len(token_ids)} tokens "
            f"(IDs: {token_ids}), but must encode to exactly 1 token. "
            f"Try a different placeholder token for this tokenizer."
        )
    return token_ids[0]


def find_placeholder_positions(
    token_ids: list[int],
    tokenizer,
    num_positions: int,
    placeholder_token: str = DEFAULT_PLACEHOLDER_TOKEN,
    validate_no_merging: bool = True,
) -> list[int]:
    """
    Find placeholder token positions in tokenized input.

    Matches the original AO paper's approach:
    - Finds exactly num_positions occurrences of the special token
    - Validates they are consecutive

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer to encode the special token
        num_positions: Expected number of placeholder positions
        placeholder_token: The placeholder token string
        validate_no_merging: If True, validate that repeated tokens don't merge

    Returns:
        List of position indices

    Raises:
        ValueError: If wrong number of positions found or they're not consecutive
    """
    # Validate single token
    special_token_id = validate_placeholder_token(tokenizer, placeholder_token)

    # Validate repeated tokens don't merge
    if validate_no_merging and num_positions > 1:
        validate_repeated_placeholder(tokenizer, placeholder_token, num_positions)

    # Find all occurrences
    positions = []
    for i, tid in enumerate(token_ids):
        if tid == special_token_id:
            positions.append(i)
        if len(positions) == num_positions:
            break

    if len(positions) != num_positions:
        raise ValueError(
            f"Expected {num_positions} placeholder positions, found {len(positions)}. "
            f"Placeholder token: {placeholder_token!r}, token ID: {special_token_id}"
        )

    # Validate consecutive positions
    if len(positions) > 1:
        if positions[-1] - positions[0] != num_positions - 1:
            raise ValueError(
                f"Placeholder positions are not consecutive: {positions}. "
                f"This may indicate token merging - try a different placeholder token."
            )

    return positions


class ActivationOracle:
    """
    Activation Oracle for interpreting CODI latent vectors.

    Usage:
        ao = ActivationOracle.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        # Create a prompt with activation vectors
        prompt = AOPrompt.from_question(
            question="What is the intermediate calculation result?",
            activation_vectors=[z3_vector],  # From CODI
            layer_percent=50,
        )

        # Get the oracle's answer
        answer = ao.generate(prompt)
        print(answer)  # e.g., "8"
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        config: AOConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

        # Get the injection layer module
        self._injection_submodule = self._get_submodule(config.injection_layer)

        # Ensure tokenizer is set up
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Validate placeholder token at init time (fail fast)
        self._placeholder_token_id = validate_placeholder_token(tokenizer, config.placeholder_token)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        config: Optional[AOConfig] = None,
        lora_path: Optional[str] = None,
        **model_kwargs,
    ) -> "ActivationOracle":
        """
        Load an Activation Oracle from a pretrained model.

        Args:
            model_name: HuggingFace model name
            config: AOConfig (uses defaults if not provided)
            lora_path: Path to trained LoRA adapter (if any)
            **model_kwargs: Additional arguments for model loading
        """
        if config is None:
            config = AOConfig(model_name=model_name)

        # Determine dtype
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=config.device,
            **model_kwargs,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Apply LoRA if configured and no pre-trained adapter
        if config.use_lora and lora_path is None:
            target_modules = (
                None if config.lora_target_modules == "all" else config.lora_target_modules
            )
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif lora_path is not None:
            # Load pre-trained LoRA adapter
            model = PeftModel.from_pretrained(model, lora_path)

        model.eval()

        return cls(model=model, tokenizer=tokenizer, config=config)

    def _get_submodule(self, layer: int) -> nn.Module:
        """Get the transformer layer module for hook injection."""
        model_name = self.config.model_name.lower()

        # Unwrap PEFT model if needed
        base_model = self.model
        if hasattr(self.model, "get_base_model"):
            base_model = self.model.get_base_model()
        elif hasattr(self.model, "base_model"):
            base_model = self.model.base_model.model

        # Get layers based on architecture
        if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
            if hasattr(base_model, "model"):
                return base_model.model.layers[layer]
            return base_model.layers[layer]
        elif "gpt2" in model_name:
            return base_model.transformer.h[layer]
        elif "gemma" in model_name:
            if hasattr(base_model, "model"):
                return base_model.model.layers[layer]
            return base_model.layers[layer]
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

    def _find_placeholder_positions(
        self,
        input_ids: torch.Tensor,
        num_positions: int,
    ) -> list[int]:
        """
        Find positions of placeholder tokens in input_ids.

        Uses the config's placeholder token and validates consecutive positions.
        """
        token_list = input_ids[0].tolist()
        return find_placeholder_positions(
            token_list,
            self.tokenizer,
            num_positions,
            placeholder_token=self.config.placeholder_token,
        )

    def generate(
        self,
        prompt: AOPrompt,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_full_text: bool = False,
    ) -> str:
        """
        Generate an answer from the Activation Oracle.

        Args:
            prompt: AOPrompt with text and activation vectors
            max_new_tokens: Override config max_new_tokens
            temperature: Override config temperature
            return_full_text: If True, return full generated text including prompt

        Returns:
            Generated answer string
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt.text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Find placeholder positions using robust method
        num_expected = prompt.num_placeholders or len(prompt.activation_vectors)
        positions = self._find_placeholder_positions(inputs["input_ids"], num_expected)

        # Validate we have the right number of vectors
        if len(prompt.activation_vectors) != len(positions):
            raise ValueError(
                f"Number of activation vectors ({len(prompt.activation_vectors)}) "
                f"doesn't match number of placeholder positions ({len(positions)})"
            )

        # Stack vectors for this single-sample batch: (K, hidden_dim)
        vecs_stacked = torch.stack([v.to(self.device) for v in prompt.activation_vectors])

        # Create injection hook (batch size 1)
        hook = get_norm_matched_steering_hook(
            vectors=[vecs_stacked],  # List of 1 tensor
            positions=[positions],  # List of 1 position list
            device=self.device,
            dtype=self.dtype,
        )

        # Generate with hook
        with torch.no_grad(), add_hook(self._injection_submodule, hook):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode output
        if return_full_text:
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Remove the input prompt from output
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

        return generated.strip()

    def create_prompt(
        self,
        question: str,
        activation_vectors: list[torch.Tensor],
        layer: int = -1,
        layer_percent: int = 50,
    ) -> AOPrompt:
        """
        Create an AOPrompt using this oracle's configured placeholder token.

        This ensures the prompt is compatible with the oracle's configuration.
        Use this instead of AOPrompt.from_question() to avoid placeholder mismatches.

        Args:
            question: The question to ask about the activations
            activation_vectors: List of activation tensors to inject
            layer: Specific layer number (-1 to use layer_percent)
            layer_percent: Layer as percentage of model depth

        Returns:
            AOPrompt configured for this oracle
        """
        return AOPrompt.from_question(
            question=question,
            activation_vectors=activation_vectors,
            layer=layer,
            layer_percent=layer_percent,
            placeholder_token=self.config.placeholder_token,
        )

    def forward_with_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        activation_vectors: list[torch.Tensor],
        positions: list[list[int]],
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with activation injection (for training).

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            activation_vectors: List of tensors, one per batch element. Each is (K_b, hidden_dim).
            positions: List of position lists, one per batch element.
            labels: Optional labels for loss computation

        Returns:
            Dict with 'loss' and 'logits'
        """
        batch_size = input_ids.shape[0]

        # Validate batch alignment
        if len(activation_vectors) != batch_size:
            raise ValueError(
                f"Number of activation vector sets ({len(activation_vectors)}) "
                f"doesn't match batch size ({batch_size})"
            )
        if len(positions) != batch_size:
            raise ValueError(
                f"Number of position sets ({len(positions)}) "
                f"doesn't match batch size ({batch_size})"
            )

        hook = get_norm_matched_steering_hook(
            vectors=activation_vectors,
            positions=positions,
            device=self.device,
            dtype=self.dtype,
        )

        with add_hook(self._injection_submodule, hook):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits,
        }

    def train_mode(self):
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def save_lora(self, path: str):
        """Save LoRA adapter weights."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        else:
            raise ValueError("Model doesn't support save_pretrained (is LoRA applied?)")

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def format_oracle_prompt(
    question: str,
    num_activations: int = 1,
    layer_percent: int = 50,
    placeholder_token: str = DEFAULT_PLACEHOLDER_TOKEN,
) -> str:
    """
    Format a prompt for the Activation Oracle.

    Uses the standard format from the AO paper:
        Layer: {layer}\n
         ? ? ?...\n
        {question}

    Args:
        question: The question to ask
        num_activations: Number of activation vectors to inject
        layer_percent: Layer percentage the activations came from
        placeholder_token: The placeholder token to use

    Returns:
        Formatted prompt string
    """
    prefix = get_introspection_prefix(
        f"{layer_percent}%",
        num_activations,
        placeholder_token=placeholder_token,
    )
    return prefix + question


if __name__ == "__main__":
    # Quick test of AO components
    print("Testing Activation Oracle components...")

    # Test prompt formatting
    prompt_text = format_oracle_prompt(
        question="What is the intermediate calculation result?",
        num_activations=1,
        layer_percent=50,
    )
    print(f"Formatted prompt: {prompt_text}")

    # Test AOPrompt creation
    dummy_vector = torch.randn(2048)  # LLaMA-1B hidden size
    prompt = AOPrompt.from_question(
        question="What number is stored here?",
        activation_vectors=[dummy_vector],
        layer_percent=50,
    )
    print(f"AOPrompt text: {prompt.text}")
    print(f"Number of vectors: {len(prompt.activation_vectors)}")

    print("\nTo test full generation, run with a GPU and model loaded.")
