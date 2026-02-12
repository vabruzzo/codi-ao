"""Norm-matched activation steering hook for the AO.

Adapted from activation_oracles/nl_probes/utils/steering_hooks.py.
Injects CODI thought activations into the AO's residual stream at a
configurable transformer layer.

The steering mechanism:
  steered = normalize(injected_vector) * ||original_activation|| * coefficient + original_activation

This preserves the magnitude distribution of the residual stream while
adding a directional signal from the CODI activations.
"""

import contextlib
from typing import Callable

import torch
import torch.nn.functional as F


def get_steering_hook(
    vectors: list[torch.Tensor],  # len B, each [K_b, d_model]
    positions: list[list[int]],  # len B, each [K_b]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """Create a forward hook that injects activation vectors into the residual stream.

    Supports variable numbers of injection positions per batch element.

    For each batch item b and position k:
      new_residual = norm(vector[b][k]) * ||residual[b, pos[b][k], :]|| * coeff + residual

    Args:
        vectors: Pre-computed activation vectors, one tensor per batch element.
                 Each tensor has shape [K_b, d_model] where K_b varies.
        positions: Token positions where activations are injected.
                   Each list has length K_b matching the corresponding vectors tensor.
        steering_coefficient: Scaling factor (typically 1.0).
        device: Target device.
        dtype: Target dtype.

    Returns:
        A hook function to register on a transformer layer.
    """
    assert len(vectors) == len(positions), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    # Pre-normalize vectors (we never backprop through these)
    normed_list = [F.normalize(v_b.to(device, dtype), dim=-1).detach() for v_b in vectors]

    def hook_fn(module, _input, output):
        # Handle different output formats across model families
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: hook got B={B_actual}, expected B={B}")

        # Only modify during the full forward pass (not during token-by-token generation)
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        # Apply steering per batch element
        for b in range(B):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            assert pos_b.min() >= 0 and pos_b.max() < L, (
                f"Position out of range: {positions[b]} with L={L}"
            )

            # Gather original activations at injection positions
            orig_KD = resid_BLD[b, pos_b, :]  # [K_b, d_model]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # [K_b, 1]

            # Build steered vectors: normalized_direction * original_norm * coefficient
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)

            # Additive steering: original + steered
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Context manager to temporarily add a forward hook to a module.

    Example:
        with add_hook(model.model.layers[1], hook_fn):
            output = model(input_ids)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_injection_submodule(model, layer_idx: int):
    """Get the transformer layer submodule for hook injection.

    Works with LLaMA, Mistral, and similar HuggingFace architectures.
    The hook is placed on the full transformer layer (not just attention or MLP).
    """
    # Try common architecture patterns
    base_model = model

    # Unwrap PeftModel if present
    if hasattr(model, "base_model"):
        base_model = model.base_model
    if hasattr(base_model, "model"):
        base_model = base_model.model

    # LLaMA / Mistral / Qwen pattern
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model.layers[layer_idx]

    # GPT2 pattern
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return base_model.transformer.h[layer_idx]

    # Direct layers access
    if hasattr(base_model, "layers"):
        return base_model.layers[layer_idx]

    raise ValueError(
        f"Cannot find transformer layers in model architecture. "
        f"Model type: {type(model)}"
    )
