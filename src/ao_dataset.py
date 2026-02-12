"""Build AO TrainingDataPoint objects from QA pairs + activations.

Converts QAPairs into the format expected by the AO training loop:
- Prompt format: "Layer: {layer}\n ? ? ? ... \n{question}"
- Pre-computed steering vectors: [num_positions, hidden_dim]
- Labels: -100 for prompt tokens, real IDs for response tokens
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from src.qa_generator import QAPair
from src.activation_extractor import ActivationRecord

# Special token for activation injection positions (matches AO convention)
SPECIAL_TOKEN = " ?"


@dataclass
class TrainingDataPoint:
    """Training data point for the AO, compatible with the AO framework."""

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layer: int
    steering_vectors: torch.Tensor | None  # [num_positions, hidden_dim]
    positions: list[int]  # Token positions where activations are injected
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None = None
    context_positions: list[int] | None = None
    ds_label: str | None = None
    meta_info: dict = field(default_factory=dict)


@dataclass
class BatchData:
    """Batch of training data with tensors."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Generate the prefix that marks where activations will be injected."""
    prefix = f"Layer: {layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


def _ensure_int_list(token_ids) -> list[int]:
    """Ensure token IDs are a flat list of Python ints."""
    if hasattr(token_ids, 'ids'):
        # tokenizers.Encoding object
        return list(token_ids.ids)
    if isinstance(token_ids, (list, tuple)):
        if len(token_ids) > 0 and hasattr(token_ids[0], 'ids'):
            # List of Encoding objects
            result = []
            for enc in token_ids:
                result.extend(enc.ids)
            return result
        return [int(x) for x in token_ids]
    # torch tensor or numpy array
    if hasattr(token_ids, 'tolist'):
        return token_ids.tolist()
    return list(token_ids)


def _apply_chat_template(tokenizer, messages, add_generation_prompt=False) -> list[int]:
    """Apply chat template and return a flat list of int token IDs."""
    # Step 1: Get the formatted chat string (always reliable)
    chat_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    # Step 2: Tokenize the string ourselves (no ambiguity about return type)
    token_ids = tokenizer.encode(chat_string, add_special_tokens=False)
    return token_ids


def find_special_token_positions(
    token_ids: list[int],
    num_positions: int,
    tokenizer: AutoTokenizer,
) -> list[int]:
    """Find positions of '?' tokens by decoding each token and checking for '?'."""
    positions = []
    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if '?' in decoded and len(positions) < num_positions:
            positions.append(i)

    if len(positions) != num_positions:
        # Debug: show what each token decodes to
        debug_tokens = [(i, tokenizer.decode([tid])) for i, tid in enumerate(token_ids[:40])]
        raise ValueError(
            f"Expected {num_positions} '?' tokens, found {len(positions)}. "
            f"First 40 decoded tokens: {debug_tokens}"
        )
    return positions


def create_training_datapoint(
    qa_pair: QAPair,
    record: ActivationRecord,
    tokenizer: AutoTokenizer,
    hook_layer: int = 1,
) -> TrainingDataPoint:
    """Convert a QAPair + activations into a TrainingDataPoint.

    Args:
        qa_pair: The QA pair with question, answer, thought indices, layer source.
        record: The activation record containing thought vectors.
        tokenizer: Tokenizer for the AO model.
        hook_layer: Which AO layer to inject into (for the prompt prefix).

    Returns:
        TrainingDataPoint ready for AO training.
    """
    num_positions = len(qa_pair.thought_indices)

    # Build prompt with introspection prefix
    prefix = get_introspection_prefix(qa_pair.layer_source, num_positions)
    prompt = prefix + qa_pair.question

    # Format as chat messages
    input_messages = [{"role": "user", "content": prompt}]
    full_messages = input_messages + [{"role": "assistant", "content": qa_pair.answer}]

    # Tokenize prompt-only (to find where response starts)
    input_prompt_ids = _apply_chat_template(tokenizer, input_messages, add_generation_prompt=True)

    # Tokenize full sequence (prompt + response)
    full_prompt_ids = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

    # Create labels: -100 for prompt tokens, real IDs for response tokens
    assistant_start_idx = len(input_prompt_ids)
    labels = list(full_prompt_ids)
    for i in range(assistant_start_idx):
        labels[i] = -100

    # Find special token positions
    positions = find_special_token_positions(full_prompt_ids, num_positions, tokenizer)

    # Gather activation vectors for the specified thoughts and layer
    vectors = []
    for thought_idx in qa_pair.thought_indices:
        thought = record.thoughts[thought_idx]

        if qa_pair.activation_type == "post_projection" and thought.post_projection is not None:
            vec = thought.post_projection.squeeze()
        else:
            # Use pre_projection at the specified layer
            layer = qa_pair.layer_source
            if layer in thought.pre_projection:
                vec = thought.pre_projection[layer].squeeze()
            else:
                # Fallback: use any available layer
                available_layers = sorted(thought.pre_projection.keys())
                if available_layers:
                    vec = thought.pre_projection[available_layers[-1]].squeeze()
                else:
                    raise ValueError(
                        f"No activation available for thought {thought_idx} at layer {layer}"
                    )
        vectors.append(vec)

    steering_vectors = torch.stack(vectors)  # [num_positions, hidden_dim]
    assert steering_vectors.shape[0] == num_positions
    assert len(positions) == num_positions

    return TrainingDataPoint(
        datapoint_type=f"codi_{qa_pair.category_name}",
        input_ids=full_prompt_ids,
        labels=labels,
        layer=qa_pair.layer_source,
        steering_vectors=steering_vectors.cpu().clone().detach(),
        positions=positions,
        feature_idx=-1,
        target_output=qa_pair.answer,
        context_input_ids=None,
        context_positions=None,
        ds_label=f"cat{qa_pair.category}_{qa_pair.subtype}" if qa_pair.subtype else f"cat{qa_pair.category}",
        meta_info={
            "problem_id": qa_pair.problem_id,
            "category": qa_pair.category,
            "subtype": qa_pair.subtype,
        },
    )


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    """Construct a padded batch from training data points.

    Left-pads sequences and adjusts positions accordingly.
    """
    max_length = max(len(dp.input_ids) for dp in training_data)

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_positions = []
    batch_steering_vectors = []

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for dp in training_data:
        padding_length = max_length - len(dp.input_ids)

        padded_input_ids = [pad_id] * padding_length + dp.input_ids
        padded_labels = [-100] * padding_length + dp.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        # Adjust positions for left-padding
        padded_positions = [p + padding_length for p in dp.positions]
        batch_positions.append(padded_positions)

        if dp.steering_vectors is not None:
            batch_steering_vectors.append(dp.steering_vectors.to(device))
        else:
            raise ValueError("All steering vectors must be pre-computed for CODI")

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
    )


def get_prompt_only(dp: TrainingDataPoint) -> TrainingDataPoint:
    """Strip the response tokens, keeping only the prompt for evaluation."""
    prompt_ids = []
    prompt_labels = []

    for i, label in enumerate(dp.labels):
        if label != -100:
            break
        prompt_ids.append(dp.input_ids[i])
        prompt_labels.append(label)

    return TrainingDataPoint(
        datapoint_type=dp.datapoint_type,
        input_ids=prompt_ids,
        labels=prompt_labels,
        layer=dp.layer,
        steering_vectors=dp.steering_vectors,
        positions=dp.positions,
        feature_idx=dp.feature_idx,
        target_output=dp.target_output,
        ds_label=dp.ds_label,
        meta_info=dp.meta_info,
    )


def save_dataset(data: list[TrainingDataPoint], path: str) -> None:
    """Save training data to a torch .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for dp in data:
        serializable.append({
            "datapoint_type": dp.datapoint_type,
            "input_ids": dp.input_ids,
            "labels": dp.labels,
            "layer": dp.layer,
            "steering_vectors": dp.steering_vectors,
            "positions": dp.positions,
            "feature_idx": dp.feature_idx,
            "target_output": dp.target_output,
            "ds_label": dp.ds_label,
            "meta_info": dp.meta_info,
        })

    torch.save({"data": serializable, "num_examples": len(data)}, path)
    print(f"Saved {len(data)} training examples to {path}")


def load_dataset_from_file(path: str) -> list[TrainingDataPoint]:
    """Load training data from a torch .pt file."""
    raw = torch.load(path, map_location="cpu", weights_only=False)

    data = []
    for item in raw["data"]:
        data.append(TrainingDataPoint(
            datapoint_type=item["datapoint_type"],
            input_ids=item["input_ids"],
            labels=item["labels"],
            layer=item["layer"],
            steering_vectors=item["steering_vectors"],
            positions=item["positions"],
            feature_idx=item["feature_idx"],
            target_output=item["target_output"],
            ds_label=item.get("ds_label"),
            meta_info=item.get("meta_info", {}),
        ))

    return data
