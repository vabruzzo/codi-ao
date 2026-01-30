# CODI Activation Oracle

Activation Oracle for interpreting CODI's latent reasoning vectors.

## Overview

This project trains an Activation Oracle (AO) to decode CODI's compressed chain-of-thought latent vectors into human-readable intermediate results.

**CODI** compresses explicit chain-of-thought reasoning into 6 continuous latent vectors. This project investigates what information is stored in these vectors and whether we can decode it.

## Setup

```bash
uv sync
```

## Quick Start

```bash
# Phase 0: Explore what's in each latent position
python scripts/explore_latents.py --n_samples 200 --verbose

# Train the Activation Oracle
python scripts/train.py --data data/train.jsonl --epochs 2

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/ao
```

## Project Structure

```
codi-ao/
├── src/
│   ├── activation_oracle.py   # AO model with injection mechanism
│   ├── codi_wrapper.py        # CODI model interface
│   └── codi_model.py          # CODI model implementation
├── scripts/
│   ├── explore_latents.py     # Phase 0: Logit lens exploration
│   ├── train.py               # Train the AO
│   └── evaluate.py            # Evaluation suite
├── configs/
│   └── default.yaml           # Configuration
└── STUDY.md                   # Detailed study plan
```

## Key Findings

See `STUDY.md` for the full research plan and results.

## References

- [CODI Paper](https://arxiv.org/abs/2502.21074): Compressing Chain-of-Thought into Continuous Space
- [LessWrong Blog](https://www.lesswrong.com/posts/YGAimivLxycZcqRFR): Interpretability analysis of CODI
- [Activation Oracle Paper](https://arxiv.org/abs/2410.04075): Training LLMs as activation explainers
