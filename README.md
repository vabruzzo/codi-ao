# CODI Activation Oracle

Training and evaluating Activation Oracles to interpret CODI's latent reasoning vectors.

## Overview

[CODI](https://arxiv.org/abs/2502.21074) compresses Chain-of-Thought reasoning into continuous latent vectors. This project trains an [Activation Oracle](https://arxiv.org/abs/2512.15674) to interpret what computations are encoded in those latent vectors.

**Key insight**: CODI's teacher task provides explicit CoT that aligns with the student's latent reasoning, giving us natural supervision for training the oracle.

## Quick Start

```bash
# Install dependencies
uv sync

# Run MVP validation (verifies z2/z4 store intermediate results)
uv run python scripts/collect_latents.py --n_samples 100 --verbose --synthetic

# Train the Activation Oracle
uv run python scripts/train.py --mode mvp --n_samples 10000

# Evaluate against baselines
uv run python scripts/evaluate.py --ao_path checkpoints/ao
```

## Project Structure

```
codi-ao/
├── src/
│   ├── codi_wrapper.py          # Load CODI, collect latent vectors
│   ├── activation_oracle.py     # AO model with injection mechanism
│   ├── datasets/                # Dataset generators
│   │   ├── latent_qa.py         # CoT-aligned QA pairs
│   │   └── classification.py    # Binary classification tasks
│   └── evaluation/              # Evaluation suite
│       ├── baselines.py         # Logit lens, linear probe
│       └── evaluator.py         # Main evaluation harness
├── scripts/
│   ├── collect_latents.py       # MVP validation
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── configs/
│   └── default.yaml             # Configuration
└── PLAN.md                      # Detailed implementation plan
```

## Phased Implementation

### Phase 1: MVP Validation
- Verify that z2 and z4 store intermediate results (per LessWrong findings)
  - Note: Paper says "z3/z5" but counts differently (includes initial position)
  - Our indices: z2=Step1 (100%), z4=Step2 (85%)
- Establish logit lens baseline (target: 85%+ accuracy)
- **Exit criteria**: Logit lens meets threshold before proceeding

### Phase 2: Full Data Generation
- Latent-to-CoT QA dataset (~64k samples)
- Classification dataset (~336k samples)
- Context prediction dataset (~600k samples)

### Phase 3: Training
- LoRA fine-tuning (rank 64, alpha 128)
- Group-by-length batching for efficiency
- Periodic evaluation checkpoints

### Phase 4: Evaluation
- Compare AO vs. logit lens, linear probe
- Out-of-distribution testing (SVAMP, MultiArith)
- Ablations (projected vs. raw latents, injection layer, etc.)

## Key Technical Details

### Latent Position Mapping
For 3-step math problems (using exact templates from CODI paper):
- **z2 (index 1)**: Stores Step 1 intermediate result (100% accuracy)
- **z4 (index 3)**: Stores Step 2 intermediate result (85% accuracy)

Note: The LessWrong blog says "z3/z5" but their indexing includes an initial position.
Our code excludes that, so their z3/z5 = our z2/z4.

### Injection Mechanism
Norm-matched additive injection:
```
h'_i = h_i + (||h_i|| / ||v_i||) * v_i
```

### Configuration
See `configs/default.yaml` for all settings.

## References

- [CODI Paper](https://arxiv.org/abs/2502.21074): Compressing Chain-of-Thought into Continuous Space
- [Activation Oracles Paper](https://arxiv.org/abs/2512.15674): Training LLMs as General-Purpose Activation Explainers
- [LessWrong Post](https://www.lesswrong.com/posts/...): Can we interpret latent reasoning using current MI tools?

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management
- PyTorch 2.0+
- Transformers 4.40+
- CUDA-capable GPU (8GB+ VRAM recommended)

## License

MIT
