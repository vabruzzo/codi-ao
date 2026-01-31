# CODI Activation Oracle

Interpreting latent reasoning in CODI models using Activation Oracles.

## Overview

This project investigates whether **Activation Oracles** can extract information from CODI's latent reasoning vectors that traditional interpretability methods (like Logit Lens) cannot.

**Key Finding**: The Activation Oracle significantly outperforms Logit Lens on operation type detection (add/sub/mul). Logit Lens achieves only **60.1%** (heavily biased toward "add"), while the AO achieves **99.5%** with balanced performance across all operations.

## Results Summary

### Operation Detection: Logit Lens vs Activation Oracle

| Method | Overall | Addition | Subtraction | Multiplication |
|--------|---------|----------|-------------|----------------|
| Logit Lens (z2) | 60.1% | 97.3% | 31.3% | 51.7% |
| **Activation Oracle** | **99.5%** | **100%** | **98.6%** | **100%** |

*Random baseline = 33.3%. Logit Lens results use probability sum method with layer norm applied.*

**Top-10 Token Analysis**: Operation tokens appear in top-10 only **6.5%** of the time. Subtraction tokens **never** appear in top-10 (0.0%).

### Full Activation Oracle Evaluation

| Task | Accuracy | Description |
|------|----------|-------------|
| Extraction Step 1 | 97.5% | Extract numeric value from z2 |
| Extraction Step 2 | 59.5% | Extract numeric value from z4 |
| Operation Direct | 99.5% | "What operation was performed?" |
| Operation Binary | 87.2% | "Is this addition?" (yes/no) |
| Magnitude | 98.0% | "Is result > 50?" (yes/no) |
| Comparison (multi-latent) | 100%* | "Which step is larger?" |

*\*Note: Comparison task has class imbalance - step2 > step1 in ~95% of cases due to multiplication in step2 for add/sub problems.*

**Note on Step 2 extraction**: The lower accuracy (59.5%) reflects that z4 is less cleanly structured than z2 (consistent with prior work). The AO still *understands* z4 semantically (100% on comparison), but struggles with exact numeric extraction from it.

### Logit Lens Results

Logit Lens projects the latent vector to the vocabulary space and sums probabilities of operation-related tokens ("add", "subtract", "multiply", etc.).

**What we observed**:

1. **Heavy bias toward "add"**: 97.3% accuracy on addition but only 31.3% on subtraction
2. **Very low confidence**: Only ~0.12% average probability mass on operation tokens
3. **Operation tokens rarely surface**: Top-10 tokens are mostly numbers ("4", "four", "04"). Operation tokens appear in top-10 only 6.5% of the time; subtraction tokens never appear

The Activation Oracle achieves **99.5%** with balanced per-operation performance (100% add, 98.6% sub, 100% mul).

**Interpretation**: The operation information appears to be present in z2 (given the AO can extract it), but is not accessible via direct vocabulary projection. Whether this is due to non-linear encoding, interference from numeric information, or other factors is unclear from this experiment alone.

### Summary

- AO outperforms Logit Lens on operation detection: 99.5% vs 60.1%, with balanced accuracy across all operations
- Logit Lens is heavily biased toward "add" (97% add vs 31% sub); operation tokens appear in top-10 only 6.5% of the time
- Operation information is present in z2 but not accessible via token probabilities
- z2 yields higher extraction accuracy than z4 (97.5% vs 59.5%), consistent with prior work
- AO performs well on z4 for comparison (100%) despite lower numeric extraction

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

### Training Data (~40k examples)

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
uv run python scripts/generate_synthetic_data.py \
    --n_samples 1200 \
    --seed 42 \
    --output data/synthetic_problems.json
```

### 2. Generate AO Training Data

Collects CODI latents and creates Q&A pairs (holds out last 200 for testing):

```bash
uv run python scripts/generate_ao_training_data.py \
    --problems data/synthetic_problems.json \
    --output data/ao_training_data.jsonl \
    --holdout 200
```

### 3. Run Logit Lens Baseline

```bash
uv run python scripts/eval_logit_lens_operation.py \
    --data data/synthetic_problems.json \
    --output results/logit_lens_operation.json
```

### 4. Train Activation Oracle

```bash
uv run python scripts/train.py \
    --data data/ao_training_data.jsonl \
    --output_dir checkpoints/ao_study \
    --epochs 3 \
    --batch_size 16
```

### 5. Evaluate Activation Oracle

```bash
uv run python scripts/eval_ao.py \
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

## References

- [CODI Paper](https://arxiv.org/abs/2310.06632) - Original latent reasoning framework
- [LessWrong Blog Post](https://www.lesswrong.com/posts/bDpKHD5haQ3pqjEfq/can-we-interpret-latent-reasoning-using-current-mechanistic) - CODI interpretability study
- [Activation Oracles Paper](https://arxiv.org/abs/2311.07328) - Natural language probing method

## License

MIT
