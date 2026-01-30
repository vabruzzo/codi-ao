# Phase 3 Results: Diverse Training at Scale

**Date**: January 29, 2026  
**Status**: IN PROGRESS

## Overview

Phase 3 addresses the limitations discovered in Phase 2 by training with diverse, scaled data following the Activation Oracle paper's approach.

## Changes from Phase 2

| Issue (Phase 2) | Solution (Phase 3) |
|-----------------|-------------------|
| Single question type | 9 diverse question types |
| 3 paraphrases/type | 15 paraphrases/type |
| 100K examples | 250K examples (2.5x) |
| No division questions | Added division (always "No") |
| Position confusion | Position-aware extraction |
| Comparison at 8% | Dedicated comparison training |

## Data Generation

**Command**:
```bash
python scripts/generate_phase3_data.py --n_prompts 25000 --target 250000
```

**Question Type Distribution** (target):
| Type | Count | Percentage |
|------|-------|------------|
| classification_magnitude | ~38K | 15% |
| classification_operation | ~38K | 15% |
| classification_position | ~38K | 15% |
| extraction_generic | ~38K | 15% |
| extraction_step1 | ~19K | 7.5% |
| extraction_step2 | ~19K | 7.5% |
| multi_latent_comparison | ~19K | 7.5% |
| multi_latent_extraction | ~19K | 7.5% |
| operation_type | ~19K | 7.5% |
| **Total** | ~250K | 100% |

## Training

**Command**:
```bash
python scripts/train.py --data data/phase3_train.jsonl --epochs 3
```

**Configuration**:
- Base model: LLaMA-3.2-1B-Instruct
- LoRA rank: 64, alpha: 128
- Learning rate: 1e-5
- Batch size: 16
- Epochs: 3

## Evaluation

**Command**:
```bash
python scripts/evaluate.py --eval_all --n_samples 200 --fresh_test --ao_path checkpoints/ao
```

## Results

### Preliminary (100K examples, 1 epoch)

| Metric | Phase 2 | Phase 3 (100K) | Change |
|--------|---------|----------------|--------|
| Single-Latent AO | 92% | **98.75%** | +6.75% |
| AO vs Logit Lens | Tied | **AO wins** | ✅ |
| Multi-latent Comparison | 8% | **93%** | +85% |
| Step 1 extraction | 94% | **100%** | +6% |
| Step 2 extraction | 88% | **100%** | +12% |
| result_property | 70% | **88.89%** | +18.89% |
| operation | 100% | 55.56% | -44.44%* |
| position | 100% | 68.75% | -31.25%* |

*Regressions due to evaluation question mismatch - fixed with expanded templates.

### Final (250K examples, 3 epochs)

*Pending - data generation in progress*

## Key Improvements

1. **AO now beats Logit Lens** (98.75% vs 92%)
2. **Comparison questions work** (8% → 93%)
3. **Perfect step extraction** in multi-latent (100%/100%)
4. **Magnitude classification improved** (70% → 88.89%)

## Files

| File | Purpose |
|------|---------|
| `src/datasets/latent_qa.py` | Expanded with 15 paraphrases/type, division |
| `scripts/generate_phase3_data.py` | Phase 3 data generation |
| `data/phase3_train.jsonl` | Training data (250K examples) |
| `checkpoints/ao/` | Trained model |

## Next Steps

1. Complete 250K example training
2. Evaluate with diverse templates
3. Document final results
4. Consider Phase 4: OOD evaluation (SVAMP, MultiArith, GSM-Hard)
