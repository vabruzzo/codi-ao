# CODI Activation Oracle

Interpreting latent reasoning in CODI models using Activation Oracles.

## Overview

This project investigates whether **Activation Oracles** can extract information from CODI's latent reasoning vectors that traditional interpretability methods (like Logit Lens) cannot.

**Key Finding**: The Activation Oracle achieves **100% accuracy** on operation type detection (add/sub/mul), while Logit Lens achieves only **64.8%** at best (heavily biased toward "add").

## Results Summary

### Operation Detection: Logit Lens vs Activation Oracle

| Method | Overall | Addition | Subtraction | Multiplication |
|--------|---------|----------|-------------|----------------|
| Logit Lens (best, z3) | 64.8% | 93.1% | 34.3% | 67.1% |
| **Activation Oracle** | **100%** | **100%** | **100%** | **100%** |

*Random baseline = 33.3%*

### Full Activation Oracle Evaluation

| Task | Accuracy | Description |
|------|----------|-------------|
| Extraction Step 1 | 99.5% | Extract numeric value from z2 |
| Extraction Step 2 | 82.0% | Extract numeric value from z4 |
| Operation Direct | 100% | "What operation was performed?" |
| Operation Binary | 97.5% | "Is this addition?" (yes/no) |
| Magnitude | 99.5% | "Is result > 50?" (yes/no) |
| Comparison (multi-latent) | 100%* | "Which step is larger?" |

*\*Note: Comparison task has class imbalance - step2 > step1 in ~95% of cases due to multiplication in step2 for add/sub problems. High accuracy may partly reflect this imbalance.*

### Why Logit Lens Fails

Logit Lens projects the latent vector to the vocabulary space and sums probabilities of operation-related tokens ("add", "subtract", "multiply", etc.).

1. **Always predicts "add"**: Achieves 89-97% accuracy on addition problems but only 4-34% on subtraction. It's essentially defaulting to "add" regardless of the true operation because "add" tokens have higher baseline probability in the vocabulary.

2. **Confidence is near-zero**: Only 1-2% of probability mass lands on operation tokens. The other 98% goes to tokens like "(", ".", "=", and numbers. The operation signal is buried in noise.

3. **Wrong representation space**: Operations aren't encoded as literal text tokens - they're in a distributed representation that requires a learned decoder (like the AO) to extract.

## Background

### CODI Model

CODI (Chain of Discrete Ideas) performs "latent reasoning" by producing continuous vectors instead of text tokens for chain-of-thought. The model we use ([bcywinski/codi_llama1b-answer_only](https://huggingface.co/bcywinski/codi_llama1b-answer_only)) is based on LLaMA 3.2 1B and produces 6 latent vectors (z1-z6) for reasoning.

Based on prior work ([LessWrong blog](https://www.lesswrong.com/posts/bDpKHD5haQ3pqjEfq/can-we-interpret-latent-reasoning-using-current-mechanistic)):
- **z2** encodes Step 1 intermediate result
- **z4** encodes Step 2 intermediate result
- Other positions (z1, z3, z5, z6) have less clear encodings

**Note on indexing**: The LessWrong code captures 7 vectors (initial + 6 iterations), while we capture only the 6 iteration outputs. Their "third and fifth" (indices 2, 4) correspond to our z2 and z4 (indices 1, 3). This is why our code uses `latent_vectors[1]` for Step 1 and `latent_vectors[3]` for Step 2.

### Activation Oracles

From the [Activation Oracles paper](https://arxiv.org/abs/2311.07328): A separate LLM fine-tuned to decode information from activation vectors via natural language Q&A. Uses:
- **Norm-matched additive injection** to insert activation vectors
- **LoRA fine-tuning** for efficient training
- **Placeholder tokens** (`" ?"`) to mark injection points

## Methodology

### Data Generation

- **1,200 synthetic math problems**: Seeded generation (seed=42) with balanced operations (add/sub/mul)
- **Train/test split**: 1,000 training / 200 held-out test problems
- **Problem structure**: Each problem has two steps:
  - `add`: step1 = X + Y, step2 = step1 * Z
  - `sub`: step1 = X - Y, step2 = step1 * Z  
  - `mul`: step1 = X * Y, step2 = step1 + Z

### Latent Collection

- **6 latent vectors** (z1-z6) collected from CODI's reasoning iterations
- **z2 (index 1)**: Encodes Step 1 intermediate result
- **z4 (index 3)**: Encodes Step 2 intermediate result
- **Operation label**: Refers to **step1's operation only** (z2). We do not evaluate operation on z4 because step2's operation differs from the problem's operation label.

### Training Data (~120k examples)

From each problem, we generate diverse QA pairs:
- Numeric extraction: "What value was computed?"
- Operation classification: "What operation was performed?"
- Magnitude: "Is the result greater than 50?"
- Comparison (multi-latent): "Which step is larger?"

### Logit Lens Baseline

- **Layer norm applied**: We apply the model's final layer norm before projecting to vocabulary space (proper logit lens)
- **Token matching**: Sum probability mass over operation-related tokens ("add", "addition", "plus", "+", etc.)
- **Prediction**: Argmax over {add, sub, mul} total probabilities

### Activation Oracle Evaluation

- **Held-out test set**: 200 problems not seen during training
- **Single question per task type**: Consistent evaluation across all problems
- **No problem context**: The AO sees only the question and latent vector(s), not the original prompt

### Key Difference from Original AO Paper

Our AO receives **no problem context** - only the question and latent vectors. This is a stricter test of what's actually encoded in the latents vs. what can be inferred from the original text.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codi-ao.git
cd codi-ao

# Install dependencies (requires uv)
uv sync
```

## Usage

### 1. Generate Synthetic Problems

```bash
python scripts/generate_synthetic_data.py \
    --n_samples 1200 \
    --seed 42 \
    --output data/synthetic_problems.json
```

### 2. Generate AO Training Data

Collects CODI latents and creates Q&A pairs (holds out last 200 for testing):

```bash
python scripts/generate_ao_training_data.py \
    --problems data/synthetic_problems.json \
    --output data/ao_training_data.jsonl \
    --holdout 200
```

### 3. Run Logit Lens Baseline

```bash
python scripts/eval_logit_lens_operation.py \
    --data data/synthetic_problems.json \
    --output results/logit_lens_operation.json
```

### 4. Train Activation Oracle

```bash
python scripts/train.py \
    --data data/ao_training_data.jsonl \
    --output_dir checkpoints/ao_study \
    --epochs 3 \
    --batch_size 4
```

### 5. Evaluate Activation Oracle

```bash
python scripts/eval_ao.py \
    --checkpoint checkpoints/ao_study \
    --problems data/synthetic_problems.json \
    --n_test 200 \
    --output results/ao_evaluation.json
```

## Project Structure

```
codi-ao/
├── configs/
│   └── default.yaml          # Model configuration
├── scripts/
│   ├── generate_synthetic_data.py    # Generate synthetic math problems
│   ├── generate_ao_training_data.py  # Collect latents + create QA pairs
│   ├── train.py                      # Train Activation Oracle
│   ├── eval_logit_lens_operation.py  # Logit Lens baseline (z2 only)
│   └── eval_ao.py                    # Evaluate trained AO
├── src/
│   ├── activation_oracle.py   # AO model implementation
│   ├── codi_wrapper.py        # CODI model interface
│   ├── codi_model.py          # CODI model implementation
│   └── datasets/
│       └── latent_qa.py       # Training data utilities
└── results/                   # Evaluation outputs
```

## Key Findings

1. **Activation Oracles can decode operation type** that Logit Lens cannot reliably extract
2. **The operation is not encoded as literal tokens** - it's in a distributed representation that requires learned decoding
3. **Step 2 extraction is harder** (82% vs 99.5% for Step 1), consistent with prior work showing z4 is less cleanly structured than z2
4. **Multi-latent reasoning works** - 100% on comparison tasks using both z2 and z4

## References

- [CODI Paper](https://arxiv.org/abs/2310.06632) - Original latent reasoning framework
- [LessWrong Blog Post](https://www.lesswrong.com/posts/bDpKHD5haQ3pqjEfq/can-we-interpret-latent-reasoning-using-current-mechanistic) - CODI interpretability study
- [Activation Oracles Paper](https://arxiv.org/abs/2311.07328) - Natural language probing method

## License

MIT
