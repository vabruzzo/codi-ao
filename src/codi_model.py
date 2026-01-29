"""
CODI Model Implementation.

Replicated from the original CODI codebase for use with Activation Oracle training.
Based on: https://arxiv.org/abs/2502.21074

CODI compresses Chain-of-Thought reasoning into continuous latent vectors via
self-distillation between a teacher (explicit CoT) and student (latent CoT) task.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class CODI(nn.Module):
    """
    CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation.

    This model wraps a base LLM with:
    1. LoRA adapters for efficient fine-tuning
    2. A projection layer for latent reasoning vectors
    3. Support for both teacher (explicit CoT) and student (latent CoT) modes
    """

    def __init__(
        self,
        codi: nn.Module,
        tokenizer: AutoTokenizer,
        num_latent: int = 6,
        use_prj: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.codi = codi
        self.tokenizer = tokenizer
        self.num_latent = num_latent
        self.use_prj = use_prj
        self.pad_token_id = pad_token_id

        # Projection layer for latent vectors
        if use_prj:
            hidden_size = codi.config.hidden_size
            self.prj = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.prj = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        model_name_or_path: str,
        lora_r: int = 128,
        lora_alpha: int = 32,
        num_latent: int = 6,
        use_prj: bool = True,
        device: str = "cuda",
        dtype: str = "bfloat16",
        strict: bool = False,
        checkpoint_save_path: Optional[str] = None,
        remove_eos: bool = False,
        full_precision: bool = True,
        **kwargs,
    ) -> "CODI":
        """
        Load a pretrained CODI model.

        Args:
            checkpoint_path: HuggingFace checkpoint ID (e.g., "bcywinski/codi_llama1b-answer_only")
            model_name_or_path: Base model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            num_latent: Number of latent reasoning positions
            use_prj: Whether to use the projection layer
            device: Device to load model on
            dtype: Data type ("bfloat16" or "float16")
            strict: Whether to require strict weight loading
            checkpoint_save_path: Local path to save/load checkpoint
            remove_eos: Whether to remove EOS token handling
            full_precision: Whether to keep projection in full precision

        Returns:
            Loaded CODI model
        """
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

        # Create CODI wrapper
        codi_model = cls(
            codi=model,
            tokenizer=tokenizer,
            num_latent=num_latent,
            use_prj=use_prj,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Download and load checkpoint
        if checkpoint_save_path is None:
            checkpoint_save_path = f"./checkpoints/{checkpoint_path.replace('/', '_')}"

        os.makedirs(checkpoint_save_path, exist_ok=True)

        # Try different possible checkpoint filenames
        possible_files = [
            "model.safetensors",
            "adapter_model.safetensors",
            "pytorch_model.bin",
            "adapter_model.bin",
        ]

        checkpoint_file = None
        for filename in possible_files:
            local_path = Path(checkpoint_save_path) / filename
            if local_path.exists():
                checkpoint_file = local_path
                break
            try:
                print(f"Trying to download {filename} from {checkpoint_path}...")
                hf_hub_download(
                    repo_id=checkpoint_path,
                    filename=filename,
                    local_dir=checkpoint_save_path,
                )
                checkpoint_file = local_path
                print(f"Successfully downloaded {filename}")
                break
            except Exception:
                print(f"  {filename} not found, trying next...")
                continue

        if checkpoint_file is None or not checkpoint_file.exists():
            # List available files in the repo
            from huggingface_hub import list_repo_files
            available = list_repo_files(checkpoint_path)
            raise FileNotFoundError(
                f"Could not find checkpoint in {checkpoint_path}. "
                f"Available files: {available}"
            )

        # Load weights based on file type
        print(f"Loading weights from {checkpoint_file}...")
        if str(checkpoint_file).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(str(checkpoint_file))
        else:
            state_dict = torch.load(str(checkpoint_file), map_location="cpu")

        # Load with appropriate strictness
        missing, unexpected = codi_model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"Missing keys: {len(missing)} (this may be normal for LoRA)")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        codi_model.to(device)
        codi_model.eval()

        return codi_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass through the base model."""
        return self.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_new_tokens: int = 256,
        num_latent_iterations: int = 6,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.95,
        greedy: bool = True,
        return_latent_vectors: bool = False,
        remove_eos: bool = False,
        output_hidden_states: bool = False,
        skip_thinking: bool = False,
        verbalize_cot: bool = False,
        sot_token: Optional[int] = None,
        eot_token: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Generate with latent reasoning.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            tokenizer: Tokenizer (uses self.tokenizer if None)
            max_new_tokens: Maximum new tokens to generate
            num_latent_iterations: Number of latent reasoning steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            greedy: Whether to use greedy decoding
            return_latent_vectors: Whether to return latent vectors
            remove_eos: Whether to remove EOS tokens
            output_hidden_states: Whether to return hidden states
            skip_thinking: Skip latent reasoning (baseline mode)
            verbalize_cot: Use explicit CoT (teacher mode)
            sot_token: Start of thought token ID
            eot_token: End of thought token ID

        Returns:
            Dict with 'sequences', 'latent_vectors' (if requested), 'hidden_states' (if requested)
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Add start-of-thought token
        if sot_token is not None:
            sot_tensor = torch.tensor([[tokenizer.eos_token_id, sot_token]], device=device)
            sot_tensor = sot_tensor.expand(batch_size, -1)
            input_ids = torch.cat([input_ids, sot_tensor], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 2, device=device)], dim=1
                )

        latent_vectors = []
        all_hidden_states = []

        with torch.no_grad():
            # Initial forward pass
            outputs = self.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            if output_hidden_states:
                # Stack hidden states: (num_layers, batch, seq, hidden)
                hs = torch.stack([h for h in outputs.hidden_states], dim=0)
                all_hidden_states.append(hs)

            # Get initial latent embedding from last position
            latent_embd = outputs.hidden_states[-1][:, -1:, :]  # (batch, 1, hidden)

            if not skip_thinking:
                # Latent reasoning iterations
                for i in range(num_latent_iterations):
                    # Project latent embedding
                    if self.use_prj:
                        latent_embd = self.prj(latent_embd)
                        latent_embd = latent_embd.to(dtype=self.codi.dtype)

                    if return_latent_vectors:
                        latent_vectors.append(latent_embd.clone())

                    # Forward with latent embedding
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_key_values = outputs.past_key_values

                    if output_hidden_states:
                        hs = torch.stack([h for h in outputs.hidden_states], dim=0)
                        all_hidden_states.append(hs)

                    # Get next latent embedding
                    latent_embd = outputs.hidden_states[-1][:, -1:, :]

            # Add end-of-thought token
            if eot_token is not None:
                eot_tensor = torch.tensor([[eot_token]], device=device)
                eot_tensor = eot_tensor.expand(batch_size, -1)

                outputs = self.codi(
                    input_ids=eot_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values

            # Generate answer tokens
            generated = []
            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]  # (batch, vocab)

                if greedy:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    # Apply temperature and sampling
                    logits = logits / temperature
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float("-inf")
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float("-inf")
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated.append(next_token)

                # Check for EOS
                if (next_token == tokenizer.eos_token_id).all():
                    break

                # Continue generation
                outputs = self.codi(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=output_hidden_states,
                )
                past_key_values = outputs.past_key_values

            # Concatenate generated tokens
            if generated:
                generated_ids = torch.cat(generated, dim=1)
                sequences = torch.cat([input_ids, generated_ids], dim=1)
            else:
                sequences = input_ids

        result = {"sequences": sequences}

        if return_latent_vectors:
            result["latent_vectors"] = latent_vectors

        if output_hidden_states and all_hidden_states:
            # Concatenate along sequence dimension
            result["hidden_states"] = torch.cat(all_hidden_states, dim=2)

        return result
