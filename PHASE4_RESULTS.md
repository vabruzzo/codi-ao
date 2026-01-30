# Phase 4 Results: Robust Multi-Domain Training

**Date**: January 30, 2026  
**Status**: IN PROGRESS

## Overview

Phase 4 addresses the generalization limitations found in Phase 3's OOD testing by:
1. Integrating GSM8k with parsed ground truth
2. Expanding synthetic math with more variety
3. Tracking step counts everywhere
4. Building a robust evaluation suite

## Progress

### 4.1 GSM8k Integration

**Script**: `scripts/parse_gsm8k.py`

| Task | Status | Notes |
|------|--------|-------|
| Parse GSM8k solutions | ⬜ Pending | Extract `<<expr=result>>` patterns |
| Track step counts | ⬜ Pending | Stratify by 2-step, 3-step, etc. |
| Train/test split | ⬜ Pending | 80/20 split, stratified |
| Generate QA examples | ⬜ Pending | Multiple phrasings per step |

**Command to run**:
```bash
python scripts/parse_gsm8k.py --generate_qa --output_dir data/gsm8k
```

### 4.2 Expanded Synthetic Math

**Script**: `scripts/generate_expanded_synthetic.py`

| Expansion | Status | Notes |
|-----------|--------|-------|
| Number ranges 1-10, 1-100 | ⬜ Pending | 10K per range |
| Novel entity types | ⬜ Pending | Zoo, library, hospital, etc. |
| Edge cases | ⬜ Pending | step1=1, step1=step2, etc. |
| Diverse question phrasings | ⬜ Pending | 10+ templates |

**Command to run**:
```bash
python scripts/generate_expanded_synthetic.py --generate_qa --n_per_range 10000
```

### 4.3 OOD Sanity Check Suite

**Script**: `scripts/sanity_check_ood.py`

Already implemented tests:
- ✅ Baseline (1-10)
- ✅ Large numbers (1-100, 1-1000)
- ✅ Novel entities
- ✅ Edge cases
- ✅ Novel question phrasings
- ✅ GSM8k (optional)

**Command to run**:
```bash
python scripts/sanity_check_ood.py --n_samples 50 --gsm8k
```

## Results

### GSM8k Parsing Stats

*(To be filled after running)*

| Metric | Value |
|--------|-------|
| Total parsed | - |
| 2-step problems | - |
| 3-step problems | - |
| 4+ step problems | - |
| Train set size | - |
| Test set size | - |

### Expanded Synthetic Stats

*(To be filled after running)*

| Number Range | Problems | QA Examples |
|--------------|----------|-------------|
| 1-10 | - | - |
| 1-100 | - | - |
| Edge cases | - | - |
| **Total** | - | - |

## Key Metrics to Track

For all evaluations:

```python
{
    "num_steps": int,           # Number of reasoning steps in problem
    "step_being_tested": int,   # Which step (1, 2, 3, ...)
    "latent_position": str,     # z2, z4, etc.
    "ao_correct": bool,
    "ll_correct": bool,
    "codi_correct": bool,
}
```

## Next Steps

1. Run GSM8k parser to generate data
2. Run expanded synthetic generator
3. Combine with existing Phase 3 data
4. Train AO on combined dataset
5. Evaluate on all OOD tests
6. Analyze performance by step count

## Commands Summary

```bash
# Step 1: Parse GSM8k
python scripts/parse_gsm8k.py --generate_qa --output_dir data/gsm8k

# Step 2: Generate expanded synthetic
python scripts/generate_expanded_synthetic.py --generate_qa --n_per_range 10000

# Step 3: Run OOD sanity checks (before training)
python scripts/sanity_check_ood.py --n_samples 100 --gsm8k

# Step 4: Train (after data generation)
# TBD - need to create combined training script

# Step 5: Evaluate (after training)
python scripts/evaluate.py --ao_path checkpoints/ao_phase4 --eval_all
```
