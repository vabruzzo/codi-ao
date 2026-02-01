# CODI Activation Oracle

Interpreting latent reasoning in CODI models using Activation Oracles.

## Overview

This project investigates whether **Activation Oracles** can extract information from CODI's latent reasoning vectors that traditional interpretability methods cannot.

We use **3-step math problems** matching the LessWrong paper's setup:
- Step 1: X op Y = step1 (first intermediate)
- Step 2: step1 * Z = step2 (second intermediate)  
- Step 3: step1 + step2 = final_answer

**Main Result**: The AO extracts operation type with **82.5% accuracy** where Logit Lens achieves **0%** - demonstrating that CODI's latents encode information in nonlinearly decodable forms that traditional interpretability methods miss.

## Results Summary

### Logit Lens Results (Baseline)

| Task | Top-10 Acc | Top-1 Acc | Avg Prob |
|------|-----------|-----------|----------|
| **Step 1 (z2)** | **100%** | **98%** | 0.956 |
| **Step 2 (z4)** | **100%** | **87%** | 0.795 |
| Step 3 / Final (z5) | 1% | 0% | ~0 |
| Step 3 / Final (z6) | 0% | 0% | 0 |
| Operation (z2) | **0%** | - | ~0 |

**Key Findings**:
1. **Intermediate steps are linearly encoded**: z2 and z4 strongly encode step 1 and step 2 results (98% and 87% top-1 accuracy)
2. **Final answer is NOT linearly encoded**: Neither z5 nor z6 contain the final answer in a form Logit Lens can read (0% accuracy)
3. **Operation type is NOT linearly encoded**: 0% accuracy, operation tokens appear in top-10 only 0.5% of the time (1/200 cases)

**What Logit Lens actually sees in z2:**
```
Top tokens: "4" (98%), "11" (99%), "12" (99%), "16" (100%), ...
```
z2 encodes the **numeric result** so strongly that operation keywords ("add", "subtract", etc.) have essentially **zero probability**. The information is there (the AO can extract it), but not in a linearly decodable form.

### Activation Oracle Results

| Task | AO (single latent) | AO (all 6) | Logit Lens |
|------|-------------------|------------|------------|
| **Step 1 (z2)** | **93.0%** | **97.5%** | 98% (top-1) |
| **Step 2 (z4)** | **61.0%** | **77.0%** | 87% (top-1) |
| Step 3 (z5) | 21.5% | - | 0% |
| Step 3 (z6) | 7.0% | - | 0% |
| **Step 3 (multi)** | - | **61.5%** | 0% |
| **Operation (z2)** | **82.5%** | **85.5%** | **0%** |
| First operand | 26.5% | 18.5% | - |
| Second operand | 26.0% | 31.5% | - |
| Full calc (strict) | 12.0% | 9.0% | - |
| Full calc (semantic) | 33.5% | 53.0% | - |
| **Comparison** | - | **100%** | - |

### Key Findings

1. **AO extracts what Logit Lens cannot**: Operation detection jumps from **0% → 82.5%**. The operation type IS encoded in z2, but requires nonlinear decoding.

2. **Position-specific encoding confirmed**: 
   - z2 is excellent for step1 (93%), poor for step3 (when used as z5/z6: 21.5%/7%)
   - This rules out "all latents contain everything" - there's specialized encoding per position

3. **Multi-latent context helps significantly**:
   - Step 3 extraction: 21.5% (z5 alone) → **61.5%** (all 6 latents)
   - The final answer appears distributed across latents or requires earlier step context

4. **Operands are poorly encoded (~26%)**: Latents focus on intermediate computations, not original input values (X, Y)

5. **Comparison is trivial (100%)**: When given all latents, the AO perfectly determines which step result is larger

---

## Background

### CODI Model

CODI (Chain of Discrete Ideas) performs "latent reasoning" by producing continuous vectors instead of text tokens for chain-of-thought. The model we use ([bcywinski/codi_llama1b-answer_only](https://huggingface.co/bcywinski/codi_llama1b-answer_only)) is based on LLaMA 3.2 1B and produces 6 latent vectors (z1-z6) for reasoning.

Based on prior work ([LessWrong blog](https://www.lesswrong.com/posts/bDpKHD5haQ3pqjEfq/can-we-interpret-latent-reasoning-using-current-mechanistic)):
- **z2** (index 1): Step 1 intermediate result
- **z4** (index 3): Step 2 intermediate result
- **z6** (index 5): Step 3 / final answer (?)

### Activation Oracles

From the [Activation Oracles paper](https://arxiv.org/abs/2311.07328): A separate LLM fine-tuned to decode information from activation vectors via natural language Q&A. Uses:
- **Norm-matched additive injection** to insert activation vectors
- **LoRA fine-tuning** for efficient training
- **Placeholder tokens** (`" ?"`) to mark injection points

---

## Methodology

### Problem Structure (3-Step)

**Example:**
> "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team?"

**Computation:**
- Step 1: 3 + 5 = **8** (X + Y)
- Step 2: 8 * 2 = **16** (step1 * Z)
- Step 3: 8 + 16 = **24** (step1 + step2 = final answer)

### Data Generation

- **1,200 synthetic math problems**: Seeded generation (seed=42) with balanced operations (add/sub/mul)
- **Train/test split**: 1,000 training / 200 held-out test problems
- **3-step structure**:
  - Step 1: X op Y (where op = add/sub/mul)
  - Step 2: step1 * Z (always multiplication)
  - Step 3: step1 + step2 (always addition = final answer)

### Training Data Structure

Training includes BOTH single-latent and multi-latent examples:

| Example Type | Latents | Questions |
|-------------|---------|-----------|
| Single z2 | position 1 only | step1 extraction, operation, operands |
| Single z4 | position 3 only | step2 extraction |
| Single z5 | position 4 only | step3/final answer |
| Single z6 | position 5 only | step3/final answer |
| Multi (all 6) | positions 0-5 | all questions |

This enables evaluation of:
1. **What does each latent encode?** (single-latent tests)
2. **Does multi-latent context help?** (compare single vs multi)

### Logit Lens Baseline

- **Layer norm applied**: Final layer norm before projecting to vocabulary space
- **Top-K analysis**: Check if target number/token appears in top-10 predictions
- **Top-1 accuracy**: Whether the highest probability token is correct
- **Operation detection**: Sum probability mass over operation-related tokens ("add", "addition", "plus", "+", "subtract", etc.)
- **No-match tracking**: If no operation tokens appear in top-K, marked as "no_match" (not counted as correct for any operation)

### Activation Oracle Evaluation

- **Held-out test set**: 200 problems not seen during training
- **Single vs multi**: Compare single-latent (z2, z4, z5, z6) vs all-6-latent performance
- **No problem context**: The AO sees only the question and latent vector(s), not the original prompt
- **Shuffle sanity check**: Randomize latent-problem mapping to verify AO uses latent information

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codi-ao.git
cd codi-ao

# Install dependencies (requires uv)
uv sync
```

## Usage

### 1. Generate 3-Step Synthetic Problems

```bash
uv run python scripts/generate_synthetic_data_3step.py \
    --n_samples 1200 \
    --seed 42 \
    --output data/synthetic_problems_3step.json
```

### 2. Generate AO Training Data

Collects CODI latents and creates Q&A pairs (holds out last 200 for testing):

```bash
uv run python scripts/generate_ao_training_data_3step.py \
    --problems data/synthetic_problems_3step.json \
    --output data/ao_training_data_3step.jsonl \
    --holdout 200
```

### 3. Run Logit Lens Baseline

```bash
uv run python scripts/eval_logit_lens_3step.py \
    --problems data/synthetic_problems_3step.json \
    --n_test 200 \
    --output new_results/logit_lens_3step.json
```

### 4. Train Activation Oracle

```bash
uv run python scripts/train.py \
    --data data/ao_training_data_3step.jsonl \
    --output_dir checkpoints/ao_3step \
    --epochs 2 \
    --batch_size 64
```

### 5. Evaluate Activation Oracle

```bash
# Normal evaluation
uv run python scripts/eval_ao_3step.py \
    --checkpoint checkpoints/ao_3step \
    --problems data/synthetic_problems_3step.json \
    --n_test 200 \
    --output new_results/ao_3step_evaluation.json

# Shuffle sanity check
uv run python scripts/eval_ao_3step.py \
    --checkpoint checkpoints/ao_3step \
    --problems data/synthetic_problems_3step.json \
    --n_test 200 \
    --shuffle \
    --output new_results/ao_3step_shuffle.json
```

---

## Project Structure

```
codi-ao/
├── configs/
│   └── default.yaml                    # Model configuration
├── scripts/
│   ├── generate_synthetic_data_3step.py    # Generate 3-step math problems
│   ├── generate_ao_training_data_3step.py  # Collect latents + create QA pairs
│   ├── train.py                            # Train Activation Oracle
│   ├── eval_logit_lens_3step.py            # Logit Lens baseline
│   └── eval_ao_3step.py                    # Evaluate trained AO
├── src/
│   ├── activation_oracle.py    # AO model implementation
│   ├── codi_wrapper.py         # CODI model interface
│   ├── codi_model.py           # CODI model implementation
│   └── datasets/
│       └── latent_qa.py        # Training data utilities
├── new_results/                # Evaluation outputs
│   ├── logit_lens_3step.json
│   └── ao_3step_evaluation.json
└── checkpoints/                # Trained models
    └── ao_3step/
```

---

## Discussion

### Methodological Validity

A key concern: are we just teaching the AO to output what we *want* to find, rather than what's actually encoded?

**Evidence against this concern:**

1. **Held-out test data**: The AO is evaluated on 200 problems never seen during training. If it memorized mappings, it would fail on new problems.

2. **Position-specific accuracy patterns**: z2 achieves 93% for step1 but z5/z6 achieve only 21.5%/7% for step3. If all latents contained the same information, accuracy would be uniform. The fact that different positions excel at different tasks confirms specialized encoding.

3. **AO can't re-derive answers**: The AO never sees the original problem text - only the latent vector and a question. It cannot re-solve the problem.

### Why Logit Lens Fails for Operations

Logit Lens projects latents directly to vocabulary space. For step values, this works because numbers dominate the representation. But operation type appears encoded in a way that doesn't linearly map to tokens like "add" or "multiply" - it requires the nonlinear decoder capacity of the AO to extract.

This is consistent with the hypothesis that some information is encoded in "directions" or "features" that don't correspond to single vocabulary tokens.

---

## References

- [CODI Paper](https://arxiv.org/abs/2310.06632) - Original latent reasoning framework
- [LessWrong Blog Post](https://www.lesswrong.com/posts/bDpKHD5haQ3pqjEfq/can-we-interpret-latent-reasoning-using-current-mechanistic) - CODI interpretability study
- [Activation Oracles Paper](https://arxiv.org/abs/2311.07328) - Natural language probing method

## License

MIT
