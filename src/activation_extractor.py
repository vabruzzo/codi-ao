"""Extract per-thought activations from a CODI model.

For each problem, runs the CODI inference loop and captures:
- Pre-projection hidden states at layers 4, 8, 12 (25/50/75% depth)
- Post-projection vectors
- Decoded top-k tokens via lm_head
- The model's predicted answer and correctness

Mirrors the inference pattern in codi/probe_latent_token.py and codi/test.py.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from src.config import CODIConfig, ExtractionConfig
from src.utils import extract_number, save_activations, format_number


@dataclass
class ThoughtRecord:
    """Activation record for a single CODI thought."""

    thought_idx: int  # 0 = initial encoding, 1-6 = latent iterations
    pre_projection: dict[int, torch.Tensor]  # {layer_idx: [hidden_dim]}
    post_projection: torch.Tensor | None  # [hidden_dim], None if use_prj=False
    decoded_top_k: list[tuple[str, float]]  # [(token_str, prob), ...]


@dataclass
class ActivationRecord:
    """Full activation record for one problem."""

    problem_id: str
    question: str
    cot_raw: str
    answer_gt: str
    thoughts: list[ThoughtRecord] = field(default_factory=list)
    predicted_answer: str | None = None
    prediction_correct: bool | None = None


def _extract_single_thought(
    outputs,
    model,
    tokenizer,
    layers: list[int],
    decode_top_k: int,
    use_prj: bool,
) -> tuple[ThoughtRecord, torch.Tensor]:
    """Extract a ThoughtRecord from a forward pass output.

    Returns:
        Tuple of (ThoughtRecord, latent_embd after projection for next step).
    """
    # Get last-layer hidden state at last position (the thought vector)
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]

    # Extract pre-projection hidden states at specified layers
    # hidden_states is a tuple: (embedding_output, layer_0_output, ..., layer_N_output)
    # So layer L is at index L+1
    pre_projection = {}
    for layer_idx in layers:
        hs_idx = layer_idx + 1  # +1 because index 0 is embedding output
        if hs_idx < len(outputs.hidden_states):
            pre_projection[layer_idx] = outputs.hidden_states[hs_idx][:, -1, :].detach().cpu()

    # Decode via lm_head BEFORE projection (this is what the thought "wants to say")
    with torch.no_grad():
        logits = model.codi.lm_head(last_hidden.unsqueeze(1))  # [batch, 1, vocab]
        probs = F.softmax(logits[:, 0, :], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=decode_top_k, dim=-1)

    decoded_top_k = []
    for k_idx in range(decode_top_k):
        token_id = top_k_indices[0, k_idx].item()
        prob = top_k_probs[0, k_idx].item()
        token_str = tokenizer.decode([token_id])
        decoded_top_k.append((token_str, prob))

    # Apply projection
    post_projection = None
    if use_prj:
        projected = model.prj(last_hidden.unsqueeze(1))  # [batch, 1, hidden_dim]
        post_projection = projected[:, 0, :].detach().cpu()
        latent_embd = projected  # For next iteration
    else:
        latent_embd = last_hidden.unsqueeze(1)

    thought = ThoughtRecord(
        thought_idx=-1,  # Will be set by caller
        pre_projection=pre_projection,
        post_projection=post_projection,
        decoded_top_k=decoded_top_k,
    )

    return thought, latent_embd


def _generate_answer(
    model,
    tokenizer,
    past_key_values,
    config: CODIConfig,
) -> str:
    """Generate answer tokens after the latent thought loop."""
    device = next(model.parameters()).device

    # Prepare EOT embedding
    if config.remove_eos:
        eot_ids = torch.tensor([model.eot_id], dtype=torch.long, device=device)
    else:
        eot_ids = torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)

    eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).unsqueeze(0)
    output = eot_emb

    pred_tokens = []
    for _ in range(config.max_new_tokens):
        out = model.codi(
            inputs_embeds=output,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            output_attentions=False,
            past_key_values=past_key_values,
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

        if config.greedy:
            next_token_id = torch.argmax(logits, dim=-1).item()
        else:
            logits = logits / config.temperature
            if config.top_k > 1:
                top_k_values, _ = torch.topk(logits, config.top_k, dim=-1)
                min_val = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_val] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

        if next_token_id == tokenizer.eos_token_id:
            break

        pred_tokens.append(next_token_id)
        output = model.get_embd(model.codi, model.model_name)(
            torch.tensor([next_token_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

    return tokenizer.decode(pred_tokens, skip_special_tokens=True)


def extract_activations_single(
    model,
    tokenizer,
    question: str,
    codi_config: CODIConfig,
    extraction_config: ExtractionConfig,
) -> tuple[list[ThoughtRecord], str]:
    """Extract all thought activations for a single problem.

    Returns:
        Tuple of (list of ThoughtRecords, predicted answer string).
    """
    device = next(model.parameters()).device
    layers = extraction_config.layers
    decode_top_k = extraction_config.decode_top_k

    # Tokenize question and append bot_id
    tokens = tokenizer(question, return_tensors="pt").to(device)
    if codi_config.remove_eos:
        bot_tensor = torch.tensor([[model.bot_id]], dtype=torch.long, device=device)
    else:
        bot_tensor = torch.tensor(
            [[tokenizer.eos_token_id, model.bot_id]], dtype=torch.long, device=device
        )
    input_ids = torch.cat([tokens["input_ids"], bot_tensor], dim=1)
    attention_mask = torch.cat(
        [tokens["attention_mask"], torch.ones_like(bot_tensor)], dim=1
    )

    thoughts = []

    with torch.no_grad():
        # Initial encoding: Thought 0
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        thought_0, latent_embd = _extract_single_thought(
            outputs, model, tokenizer, layers, decode_top_k, codi_config.use_prj
        )
        thought_0.thought_idx = 0
        thoughts.append(thought_0)

        # Latent iterations: Thoughts 1-6
        for i in range(codi_config.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            thought_i, latent_embd = _extract_single_thought(
                outputs, model, tokenizer, layers, decode_top_k, codi_config.use_prj
            )
            thought_i.thought_idx = i + 1
            thoughts.append(thought_i)

        # Generate answer
        predicted_answer = _generate_answer(model, tokenizer, past_key_values, codi_config)

    return thoughts, predicted_answer


def extract_activations_batch(
    model,
    tokenizer,
    problems: list[dict],
    codi_config: CODIConfig,
    extraction_config: ExtractionConfig,
    show_progress: bool = True,
) -> list[ActivationRecord]:
    """Extract activations for a batch of problems.

    Args:
        model: Loaded CODI model.
        tokenizer: Tokenizer.
        problems: List of dicts with 'question', 'cot', 'answer' keys.
        codi_config: CODI configuration.
        extraction_config: Extraction configuration.
        show_progress: Whether to show a progress bar.

    Returns:
        List of ActivationRecord objects.
    """
    records = []
    iterator = tqdm(problems, desc="Extracting activations") if show_progress else problems

    for idx, problem in enumerate(iterator):
        question = problem["question"].strip().replace("  ", " ")
        cot_raw = problem.get("cot", "")
        answer_gt = str(problem.get("answer", ""))

        # Extract answer from GSM8k format if needed
        if "####" in answer_gt:
            answer_gt = answer_gt.split("####")[-1].strip()

        thoughts, predicted_answer = extract_activations_single(
            model, tokenizer, question, codi_config, extraction_config
        )

        # Check correctness
        pred_num = extract_number(predicted_answer)
        gt_num = extract_number(answer_gt)
        prediction_correct = None
        if pred_num is not None and gt_num is not None:
            prediction_correct = abs(pred_num - gt_num) < 1e-3

        record = ActivationRecord(
            problem_id=str(idx),
            question=question,
            cot_raw=cot_raw,
            answer_gt=answer_gt,
            thoughts=thoughts,
            predicted_answer=predicted_answer,
            prediction_correct=prediction_correct,
        )
        records.append(record)

    return records


def save_activation_records(records: list[ActivationRecord], output_dir: str) -> None:
    """Save activation records to disk as a torch .pt file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = []
    for record in records:
        thoughts_data = []
        for thought in record.thoughts:
            thoughts_data.append({
                "thought_idx": thought.thought_idx,
                "pre_projection": thought.pre_projection,  # dict of tensors
                "post_projection": thought.post_projection,
                "decoded_top_k": thought.decoded_top_k,
            })

        serializable.append({
            "problem_id": record.problem_id,
            "question": record.question,
            "cot_raw": record.cot_raw,
            "answer_gt": record.answer_gt,
            "thoughts": thoughts_data,
            "predicted_answer": record.predicted_answer,
            "prediction_correct": record.prediction_correct,
        })

    save_path = output_path / "activation_records.pt"
    torch.save(
        {"records": serializable, "num_problems": len(records)},
        save_path,
    )
    print(f"Saved {len(records)} activation records to {save_path}")


def load_activation_records(path: str) -> list[ActivationRecord]:
    """Load activation records from a torch .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    records = []

    for item in data["records"]:
        thoughts = []
        for t in item["thoughts"]:
            thoughts.append(ThoughtRecord(
                thought_idx=t["thought_idx"],
                pre_projection=t["pre_projection"],
                post_projection=t["post_projection"],
                decoded_top_k=t["decoded_top_k"],
            ))

        records.append(ActivationRecord(
            problem_id=item["problem_id"],
            question=item["question"],
            cot_raw=item["cot_raw"],
            answer_gt=item["answer_gt"],
            thoughts=thoughts,
            predicted_answer=item["predicted_answer"],
            prediction_correct=item["prediction_correct"],
        ))

    return records
