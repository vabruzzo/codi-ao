# CODI Activation Oracle

Interpreting CODI's latent reasoning through Activation Oracles.

## Overview

This project trains an **Activation Oracle (AO)** to decode information from CODI's continuous chain-of-thought latent vectors. CODI compresses explicit reasoning into 6 latent vectors (z1-z6) that drive computation without visible intermediate steps.

**Key Questions:**
1. What information is encoded in each latent position?
2. Can we extract intermediate computation steps?
3. Can we detect which mathematical operation was used?
4. Is the AO genuinely reading latents or memorizing patterns?

## Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Complete Experiment Pipeline

### Step 1: Generate Problems with Holdout Controls

This creates train/test data with rigorous controls to detect memorization:
- **Value holdout**: Random step values never seen in training
- **Tuple holdout**: Specific (X,Y,Z,op) combinations held out entirely  
- **Operand swap**: Same (X,Y) operands but different operations

```bash
python scripts/generate_data_holdout.py \
    --n_train 1000 \
    --n_test 200 \
    --value_holdout_ratio 0.15 \
    --tuple_holdout_ratio 0.10 \
    --seed 42 \
    --output data/problems_holdout.json
```

### Step 2: Generate AO Training Data (requires GPU with CODI)

Runs CODI on training problems to collect latent vectors:

```bash
uv run python scripts/generate_ao_training_holdout.py \
    --input data/problems_holdout.json \
    --output data/ao_training_holdout.jsonl
```

### Step 3: Train Activation Oracle

Fine-tunes a LLaMA model with LoRA to decode latent vectors:

```bash
uv run python scripts/train.py \
    --data data/ao_training_holdout.jsonl \
    --output_dir checkpoints/ao_holdout \
    --epochs 2 \
    --batch_size 4
```

### Step 4: Comprehensive AO Evaluation

Tests extraction accuracy with detailed breakdowns:

```bash
uv run python scripts/eval_ao_holdout.py \
    --checkpoint checkpoints/ao_holdout \
    --problems data/problems_holdout.json \
    --output results/ao_holdout_eval.json
```

**Outputs:**
- Overall accuracy for step1, step2, step3, operation detection
- Single vs multi-latent comparison
- Holdout analysis (seen vs novel values/tuples)
- CODI correctness correlation
- Per-operation breakdown

### Step 5: Logit Lens Baseline

Compares AO to simple linear projection (Logit Lens):

```bash
uv run python scripts/eval_logit_lens_holdout.py \
    --problems data/problems_holdout.json \
    --output results/logit_lens_holdout_eval.json
```

### Step 6: Operation Memorization Test

Critical test: Can AO distinguish operations when step1 value is identical?

```bash
uv run python scripts/test_operation_not_memorized.py \
    --checkpoint checkpoints/ao_holdout \
    --n_test 100 \
    --output results/operation_memorization_test.json
```

**Example:** If training saw `5 + 7 = 12` (addition), can AO correctly identify `3 × 4 = 12` (multiplication) as multiplication? This proves the AO reads operation info, not just memorizing value→operation mappings.

## Problem Structure

3-step math problems following the LessWrong CODI interpretability setup:

```
Step 1: X op Y = step1    (op ∈ {add, sub, mul})
Step 2: step1 × Z = step2
Step 3: step1 + step2 = final_answer
```

Example: "A team starts with 5 members. They recruit 3 new members. Then each current member recruits 2 additional people. How many people are there now?"
- Step 1: 5 + 3 = 8
- Step 2: 8 × 2 = 16  
- Step 3: 8 + 16 = 24

## Holdout Controls

The experiment uses multiple controls to distinguish genuine latent reading from memorization:

| Control | Description | What It Tests |
|---------|-------------|---------------|
| **Value holdout** | Random step values never in training | Can AO extract values it never saw? |
| **Tuple holdout** | (X,Y,Z,op) combos never in training | Can AO generalize to new problems? |
| **Operand swap** | Same (X,Y), different operation | Is AO reading operation info? |
| **Rarity analysis** | Accuracy by training frequency | Does AO rely on common patterns? |

## Expected Results

Based on prior experiments:

| Task | Logit Lens | AO (single) | AO (multi) |
|------|------------|-------------|------------|
| Step 1 extraction | ~85% | ~93% | ~97% |
| Step 2 extraction | ~60% | ~61% | ~77% |
| Step 3 (final) | ~0% | ~20% | ~60% |
| Operation detection | ~0% | ~82% | ~85% |

**Key findings:**
1. **Logit Lens fails for operations** - operation info is non-linearly encoded
2. **AO succeeds at operations** - proves it reads more than numeric values
3. **Multi-latent helps** - information is distributed across latents
4. **Operand swap test** - if AO distinguishes same-value different-operations, confirms genuine reading

## Project Structure

```
codi-ao/
├── scripts/
│   ├── generate_data_holdout.py      # Generate problems with controls
│   ├── generate_ao_training_holdout.py  # Create AO training data
│   ├── train.py                      # Train Activation Oracle
│   ├── eval_ao_holdout.py            # Comprehensive AO evaluation
│   ├── eval_logit_lens_holdout.py    # Logit Lens baseline
│   └── test_operation_not_memorized.py  # Operation memorization test
├── src/
│   ├── activation_oracle.py          # AO model implementation
│   ├── codi_wrapper.py               # CODI interface
│   ├── codi_model.py                 # CODI model
│   └── datasets/                     # Dataset utilities
├── configs/
│   └── default.yaml
└── pyproject.toml
```

## References

- [CODI: Compressing Chain-of-Thought into Continuous Space](https://arxiv.org/abs/2502.21074)
- [LessWrong: Interpreting CODI's Continuous Chain-of-Thought](https://www.lesswrong.com/)
- [Activation Oracles (Anthropic)](https://www.anthropic.com/)
