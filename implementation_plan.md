# CODI Activation Oracle — Implementation Plan

## Context

CODI compresses chain-of-thought reasoning into 6 continuous latent vectors. These vectors carry structured reasoning information but can only be inspected today by projecting them back into vocabulary space — a lossy, inflexible method. We're building an Activation Oracle (AO) that allows arbitrary natural language queries about the latent reasoning process, including error localization (debugging *why* CODI gets a problem wrong).

Both submodules (codi/, activation_oracles/) are mature codebases. The work is primarily **integration**: extracting CODI activations, generating QA training data, and adapting the AO training pipeline. Strategy: **thin end-to-end first** (~100 problems), then scale to 25k.

---

## File Structure

```
src/
  __init__.py
  config.py                  # All config dataclasses
  codi_loader.py             # Load CODI model + checkpoint
  activation_extractor.py    # Run CODI inference, extract per-thought activations
  cot_parser.py              # Parse GSM8k-Aug CoT into structured steps
  thought_alignment.py       # Align 6-7 thoughts → CoT steps via lm_head decoding
  qa_templates.py            # Template strings + paraphrases for all 6 categories
  qa_generator.py            # Generate QA pairs from activations + parsed CoT
  ao_dataset.py              # Build TrainingDataPoint objects for the AO
  steering.py                # Norm-matched injection hook (adapted from AO repo)
  ao_trainer.py              # AO training loop (adapted from AO sft.py)
  ao_eval.py                 # Evaluation: exact match, BLEU, error localization
  utils.py                   # Shared helpers (number extraction, I/O)
scripts/
  01_download_data.py        # Download GSM8k-Aug + CODI checkpoints
  02_extract_activations.py  # Run extraction over N problems
  03_generate_qa.py          # Build QA dataset from activations
  04_train.py                # Launch AO training
  05_eval.py                 # Run evaluation suite
  run_thin_pipeline.sh       # One-command E2E (100 problems)
configs/
  thin.yaml                  # 100-problem validation config
  full.yaml                  # 25k-problem production config
```

---

## Implementation Order

### Step 1: Config + Utils + CoT Parser (no GPU needed)

**Files**: `src/config.py`, `src/utils.py`, `src/cot_parser.py`, `src/qa_templates.py`

`config.py` — All dataclasses:
- `CODIConfig`: model_name, ckpt_path, num_latent=6, use_prj=True, prj_dim=2048, remove_eos=True, lora_r=128, lora_alpha=32
- `ExtractionConfig`: num_problems, layers=[4,8,12], batch_size, output_path, extract_post_projection
- `QAConfig`: examples_per_category dict, num_paraphrases, train_val_split_ratio
- `AOTrainingConfig`: model_name, hook_onto_layer=1, lr, epochs, batch_size, lora_r=64, lora_alpha=128, steering_coefficient=1.0
- `EvalConfig`: eval_batch_size, generation_kwargs, metrics

`cot_parser.py` — Parse `"<<600*30/100=180>> <<600*10/100=60>>"` into structured `CotStep` records with operands, operations, result, and `uses_previous_result` flag. Pure string processing, testable offline.

`qa_templates.py` — Hand-written template strings (10-15 paraphrases per question type across all 6 categories). Pure data file.

### Step 2: CODI Loader (needs GPU + checkpoint)

**File**: `src/codi_loader.py`

Adapts the loading pattern from `codi/test.py` lines 47-100:
1. Create CODI `ModelArguments` + `TrainingArguments` from our `CODIConfig`
2. Instantiate `CODI(model_args, training_args, lora_config)`
3. Load `model.safetensors` from checkpoint
4. Tie weights, move to device, set eval mode
5. Return `(model, tokenizer)`

Key params from `codi/scripts/test_llama1b.sh`: num_latent=6, use_prj=True, prj_dim=2048, remove_eos=True, greedy=True.

### Step 3: Activation Extractor (needs GPU)

**File**: `src/activation_extractor.py`

For each problem, runs the CODI inference loop and captures:

```
Per thought (7 total: initial encoding + 6 iterations):
  pre_projection:  {layer_4: [2048], layer_8: [2048], layer_12: [2048]}
  post_projection: [2048]  (after model.prj())
  decoded_top_5:   [(token_str, prob), ...]

Per problem:
  predicted_answer, prediction_correct, question, cot_raw, answer_gt
```

The extraction flow mirrors `codi/probe_latent_token.py` lines 178-211:
1. Tokenize question, append `bot_id`
2. Forward pass with `output_hidden_states=True` → capture `hidden_states[5][:,-1,:]` (layer 4), `[9]` (layer 8), `[13]` (layer 12), `[-1]` (last layer)
3. Decode via `model.codi.lm_head(hidden_state)` → top-5 tokens
4. Apply `model.prj()` → capture post-projection vector
5. Feed projected vector as `inputs_embeds` → repeat 6 times
6. After all thoughts: append `eot_id`, generate answer tokens

**Storage**: torch `.pt` files. ~114 KB/problem (7 thoughts × 4 vectors × 2048 × bf16).
- 100 problems (thin): ~11 MB
- 25k problems (full): ~2.8 GB

**Note on 7 vs 6 thoughts**: The initial encoding output (thought 0) is the representation of `bot_id` — the model's "first impression" before any latent iteration. We extract all 7 but mark thought 0 specially. The plan.md's "6 continuous thoughts" refers to the 6 iteration outputs.

### Step 4: Thought Alignment

**File**: `src/thought_alignment.py`

Maps each of the 7 thoughts to CoT steps:
1. For each thought, check if decoded top-1 token is a number
2. If it matches an intermediate result from parsed CoT → align to that step
3. If it decodes to punctuation/filler → mark as "transitional"

Output: `ThoughtAlignment(thought_idx, matched_step_idx, is_transitional, confidence)`

### Step 5: QA Generator

**File**: `src/qa_generator.py`

Generates QA pairs across 6 categories. For each problem with activations + parsed CoT + alignment:

| Category | Input | # positions | Target | Thin count | Full count |
|----------|-------|-------------|--------|------------|------------|
| 1. Intermediate Result | Single aligned thought | 1 | Number string ("180") | ~100 | ~25k |
| 2. Operation Classification | Single aligned thought | 1 | "Yes"/"No" | ~150 | ~30k |
| 3. Full Reasoning | All 6-7 thoughts | 6-7 | NL description of steps | ~50 | ~15k |
| 4. Problem Properties | All 6-7 thoughts | 6-7 | Number or "Yes"/"No" | ~100 | ~15k |
| 5. Context Prediction | 1-6 thoughts (variable) | 1-6 | Missing results/answer | ~100 | ~25k |
| 6. Thought Informativeness | Single thought | 1 | "computational"/"transitional" | ~70 | ~10k |

Each example randomly draws its source layer from {4, 8, 12} and its question template from the paraphrase pool. Binary categories are balanced (equal Yes/No).

### Step 6: AO Dataset Builder

**File**: `src/ao_dataset.py`

Converts QA pairs → `TrainingDataPoint` objects compatible with the AO framework.

Key interface: uses `create_training_datapoint()` pattern from `activation_oracles/nl_probes/utils/dataset_utils.py`:
- Prompt format: `"Layer: {layer}\n ? ? ? ... \n{question_text}"`
- `?` tokens mark injection positions
- `steering_vectors`: always pre-computed tensor of shape `[num_positions, 2048]`
- Labels: -100 for prompt tokens, real token IDs for response
- Chat template applied via `tokenizer.apply_chat_template`

**Critical**: CODI activations require the iterative latent loop and CANNOT be computed on-the-fly. All `steering_vectors` are pre-computed, `context_input_ids` is always None.

### Step 7: Steering Hook

**File**: `src/steering.py`

Adapted from `activation_oracles/nl_probes/utils/steering_hooks.py`. The norm-matched additive steering:

```
steered = F.normalize(injected_vector) × ||original_activation|| × coefficient + original_activation
```

Applied as a forward hook on transformer layer 1 of the AO model. Handles variable numbers of injection positions per batch element (ragged tensors stored as Python lists).

### Step 8: AO Trainer

**File**: `src/ao_trainer.py`

Adapted from `activation_oracles/nl_probes/sft.py`. Simplified vs. the AO repo:
- No DDP (single GPU for thin pipeline, add DDP for full runs)
- No on-the-fly materialization (all activations pre-computed)
- No SAE-specific logic

Core loop:
1. Load `meta-llama/Llama-3.2-1B-Instruct` (fresh checkpoint, NOT CODI's)
2. Apply LoRA (rank 64, alpha 128, all-linear)
3. For each batch: apply steering hook → forward pass → cross-entropy loss → backward
4. AdamW with linear warmup (10%) + linear decay
5. Periodic eval + checkpoint saving
6. WandB logging

### Step 9: Evaluation

**File**: `src/ao_eval.py`

Metrics per category:
- **Cat 1** (intermediate result): Exact numeric match, stratified by CoT step count (1-step, 2-step, 3-step)
- **Cat 2** (operation classification): Binary accuracy + format correctness
- **Cat 3** (full reasoning): BLEU-4, ROUGE-L, optional LLM judge faithfulness
- **Cat 4** (properties): Mixed exact match + binary accuracy
- **Cat 5** (context prediction): Exact match for numbers
- **Cat 6** (informativeness): Binary accuracy

**Comparison to CODI baseline**: For the same held-out problems, compare AO intermediate result recovery vs. CODI's `lm_head` top-1 decoding.

**Error localization**: For problems CODI gets wrong, ask AO "What intermediate result does this thought compute?" for each thought, compare to ground truth, identify which step diverged.

### Step 10: Pipeline Scripts + Thin Runner

**Files**: `scripts/01_download_data.py` through `scripts/05_eval.py`, `scripts/run_thin_pipeline.sh`

Each script is a standalone entry point that reads config from YAML. The thin runner chains them all with 100 problems.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Activation type | Pre-projection hidden states (Option B) | Live in standard LLaMA space; AO has "privileged access" |
| Extraction layers | 4, 8, 12 (25/50/75% of 16 layers) | Follows AO paper convention |
| Injection layer | Layer 1 of AO | Follows AO paper default |
| AO base model | Fresh LLaMA-3.2-1B-Instruct | Same architecture as CODI; fresh weights for intact language modeling |
| AO repo integration | Import core abstractions, adapt training loop | Reuse `TrainingDataPoint`, steering hooks; simplify training for our case |
| Thought count | Extract all 7 (initial + 6 iterations) | Mark thought 0 specially; more data is better |

---

## Thin Pipeline Success Criteria

The thin pipeline (100 problems, ~570 QA examples) validates the E2E flow, NOT accuracy:

1. No crashes or shape mismatches
2. Training loss decreases
3. AO produces text (not garbage) when given injected activations
4. Format correctness >30% on binary questions

Estimated runtime: ~35 minutes on A100.

---

## Full Pipeline Success Criteria

From plan.md:
- **Minimum**: AO recovers intermediate results ≥75% on 3-step problems (matching CODI's vocab projection)
- **Good**: Faithful NL descriptions + OOD generalization to SVAMP/MultiArith
- **Excellent**: Error localization works — AO can identify which step went wrong on incorrect problems

---

## Verification Plan

1. **Unit tests** for `cot_parser.py` — parse known CoT strings, verify structured output
2. **Shape assertions** in `activation_extractor.py` — verify all tensors are `[2048]`
3. **Sanity check** after extraction — decoded top-1 tokens should match CODI paper's interpretability analysis
4. **Training curve** — loss should decrease; if flat, check injection hook is actually firing
5. **Baseline comparison** — AO must beat random (50% on binary, 0% on exact match) to confirm activations carry signal
6. **Error localization** — qualitative inspection on 10-20 wrong-answer problems before computing aggregate metrics
