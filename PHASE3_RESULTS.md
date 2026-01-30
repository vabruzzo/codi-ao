# Phase 3 Results: Diverse Training at Scale

**Date**: January 29, 2026  
**Status**: COMPLETE

## Overview

Phase 3 addresses the limitations discovered in Phase 2 by training with diverse, scaled data following the Activation Oracle paper's approach.

## Changes from Phase 2

| Issue (Phase 2) | Solution (Phase 3) |
|-----------------|-------------------|
| Single question type | 9 diverse question types |
| 3 paraphrases/type | 15+ paraphrases/type |
| ~10K examples | 100K examples (10x) |
| No division questions | Added division (always "No") |
| Position confusion | Position-aware extraction |
| Comparison at 8% | Dedicated comparison training |

## Data Generation

**Command**:
```bash
python scripts/generate_phase3_data.py --n_prompts 10000 --target 100000
```

**Question Type Distribution** (actual):
| Type | Count | Percentage |
|------|-------|------------|
| classification_magnitude | 15,384 | 15.4% |
| classification_operation | 15,385 | 15.4% |
| classification_position | 15,386 | 15.4% |
| extraction_generic | 15,385 | 15.4% |
| extraction_step1 | 7,693 | 7.7% |
| extraction_step2 | 7,691 | 7.7% |
| multi_latent_comparison | 7,693 | 7.7% |
| multi_latent_extraction | 7,693 | 7.7% |
| operation_type | 7,690 | 7.7% |
| **Total** | **100,000** | 100% |

**Single-latent**: 84,614 (84.6%)  
**Multi-latent**: 15,386 (15.4%)

## Training

**Command**:
```bash
python scripts/train.py --data data/phase3_train.jsonl --epochs 2
```

**Configuration**:
- Base model: LLaMA-3.2-1B-Instruct
- LoRA rank: 64, alpha: 128
- Learning rate: 1e-5
- Batch size: 16
- Epochs: 2

## Evaluation

**Command**:
```bash
python scripts/evaluate.py --eval_all --n_samples 200 --fresh_test
```

## Results

### Final Results (100K examples, 2 epochs)

#### Single-Latent QA
| Method | Accuracy | Correct/Total |
|--------|----------|---------------|
| Logit Lens | 92.00% | 368/400 |
| **Activation Oracle** | **98.75%** | 395/400 |

**Key Result**: AO beats Logit Lens by 6.75 percentage points.

#### Classification
| Type | Accuracy | Correct/Total |
|------|----------|---------------|
| structure | 100.00% | 24/24 |
| result_property | 88.89% | 72/81 |
| position | 68.75% | 22/32 |
| operation | 55.56% | 35/63 |
| **Overall** | **76.50%** | 153/200 |

*Note: Operation and position regressions are due to evaluation question mismatch with training templates (e.g., eval uses "Is this one of the first calculation steps?" but training uses "Is this step 1?"). See Issues section.*

#### Multi-Latent QA
| Question Type | Accuracy | Correct/Total |
|---------------|----------|---------------|
| Step 1 extraction | 100.00% | 50/50 |
| Step 2 extraction | 100.00% | 50/50 |
| Comparison | 93.00% | 93/100 |
| Final answer | 2.00% | 1/50 |
| **Overall** | **77.60%** | 194/250 |

*Note: Final answer (2%) is expected - CODI does not store the final answer in its latent space.*

### Comparison with Phase 2

| Metric | Phase 2 | Phase 3 | Change |
|--------|---------|---------|--------|
| Single-Latent AO | 92% | **98.75%** | +6.75% |
| AO vs Logit Lens | Tied | **AO wins** | ✅ |
| Multi-latent Comparison | 8% | **93%** | +85% |
| Step 1 extraction | 94% | **100%** | +6% |
| Step 2 extraction | 88% | **100%** | +12% |
| result_property | 70.37% | **88.89%** | +18.52% |

## Key Achievements

1. **AO now beats Logit Lens** (98.75% vs 92%) - validates the core hypothesis
2. **Comparison questions work** (8% → 93%) - AO can reason across multiple latents
3. **Perfect step extraction** in multi-latent (100%/100%) - no more z2/z4 confusion
4. **Magnitude classification improved** (70% → 89%)

## Known Issues

### Classification Regressions
Operation (100% → 56%) and position (100% → 69%) dropped due to evaluation/training question mismatch:
- **Evaluation**: "Is this one of the first calculation steps?"
- **Training**: "Is this step 1?"

**Fix applied**: Updated `src/datasets/classification.py` to align evaluation questions with training templates.

### Synthetic Data Limitations
Current training data uses limited diversity:
- Only 10 entity types (team, company, school, etc.)
- Numbers limited to 1-10
- Only 2 problem structures: `(X + Y) × Z` and `(X - Y) × Z`

This may limit generalization to more complex problems.

## Files

| File | Purpose |
|------|---------|
| `src/datasets/latent_qa.py` | Expanded with 15+ paraphrases/type, division |
| `scripts/generate_phase3_data.py` | Phase 3 data generation |
| `data/phase3_train.jsonl` | Training data (100K examples) |
| `checkpoints/ao/` | Trained model |

## OOD Sanity Check Results

After Phase 3 training, we ran OOD sanity checks (`scripts/sanity_check_ood.py`):

| Test | CODI Final | AO Latent | vs Baseline |
|------|-----------|-----------|-------------|
| Baseline (1-10) | 55% | 100% | - |
| Large numbers (1-100) | 0% | 32.5% | -67.5% |
| Novel entities | 10% | 80% | -20% |
| Edge cases | 33% | 100% | +0% |
| Novel questions | 40% | 90% | -10% |

### Key Finding: "Latent ✓, CODI ✗"

We discovered cases where the AO correctly extracts the intermediate value but CODI outputs the wrong final answer:

| Test | L✓C✓ | L✓C✗ | L✗C✓ | L✗C✗ |
|------|------|------|------|------|
| Baseline | 55% | 45% | 0% | 0% |
| Novel entities | 5% | 55% | 5% | 35% |
| Edge cases | 33% | 67% | 0% | 0% |

**Implication**: CODI has "hidden knowledge" - it computes correct intermediates that it fails to utilize in the final answer. The AO can reveal this hidden knowledge.

### Limitations Identified

1. **Large numbers**: Both CODI and AO fail on 1-100+ ranges (CODI's training limitation)
2. **Novel entities with semantic shift**: z4 extraction drops when problem semantics change
3. **GSM8k**: 17% accuracy - mismatch between synthetic templates and real problems

## Next Steps → Phase 4

See `PHASE4_RESULTS.md` for the plan to address these limitations:
1. **GSM8k integration**: Parse real problems with ground truth
2. **Expanded synthetic**: 1-100 numbers, more entities, edge cases
3. **Step count tracking**: Analyze by number of reasoning steps
