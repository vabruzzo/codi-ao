# Phase 2 Results: Evaluation Harness & Analysis

**Date**: January 29, 2026  
**Status**: ✓ COMPLETE

## Overview

Phase 2 pivoted from the original plan (full data generation) to focus on building a comprehensive evaluation harness and analyzing the trained AO's capabilities. This informed what diverse training was needed for Phase 3.

**Note**: The original PLAN.md described Phase 2 as "Full Data Generation" but we found it more valuable to first understand the AO's limitations through rigorous evaluation before scaling up data generation.

## What We Built

### 1. Evaluation Harness (`scripts/evaluate.py`)
- **Single-latent QA**: Tests extraction of intermediate results from z2/z4
- **Classification**: Tests yes/no questions about latent properties
- **Multi-latent QA**: Tests with all 6 latents, including comparison questions
- **Full qualitative output**: See every AO input prompt and output

### 2. Bug Fixes
- **Logit lens correctness bug**: Empty predictions (`""`) were being marked as correct because `"" in "42"` returns `True` in Python. Fixed to return `False` for empty predictions.

### 3. Key Discoveries
- **Final answer not stored in latents**: CODI architecture only stores intermediate steps (z2=step1, z4=step2). The final answer is computed at output generation time, not during latent reasoning.
- **z5/z6 unused**: Patching these with random vectors doesn't affect accuracy.

## Evaluation Results

### Single-Latent QA
| Method | Accuracy | Notes |
|--------|----------|-------|
| Logit Lens | 93% | Baseline (projects latent → vocab) |
| AO | 92% | Matches baseline |
| AO on z2 | 100% | Perfect on step 1 |
| AO on z4 | 84% | Sometimes returns step 1 value instead |

### Classification (88% overall)
| Type | Accuracy | Notes |
|------|----------|-------|
| operation | 100% | "Is this multiplication?" |
| position | 100% | "Is this the first step?" |
| structure | 100% | "Is this a computation?" |
| result_property | 70% | "Is result > 50?" - struggles with magnitude |

### Multi-Latent QA
| Type | Accuracy | Notes |
|------|----------|-------|
| Step 1 extraction | 94% | Works well |
| Step 2 extraction | 88% | Works well |
| Final answer | 16% | Expected - not stored in latents |
| Comparison | 8% | Not trained for this task |

## Key Findings

### 1. AO Can Read But Not Reason
The AO successfully extracts numeric values from individual latents (92%) but cannot:
- Compare values across latents (8%)
- Judge magnitude thresholds accurately (70%)
- Find information not stored in latents (16%)

### 2. Training Data Was Too Narrow
Our training only included one task type: "What is the intermediate result?" → number

The AO paper trains on diverse tasks:
- System Prompt QA
- Binary classification (7 datasets)
- Context prediction (self-supervised)
- ~1 million examples total

### 3. z2/z4 Confusion Pattern
When AO fails on z4, it often returns the z2 (step 1) value:
| Ground Truth (z4) | AO Output | Actual z2 |
|-------------------|-----------|-----------|
| 180 | 18 | 18 ✓ |
| 170 | 17 | 17 ✓ |
| 162 | 18 | 18 ✓ |

The AO hasn't learned to distinguish which latent position it's reading.

### 4. CODI Architecture Limitation
From the CODI paper:
> "The final answer is not represented in the latent reasoning vectors, as the training traces do not include the last step in which the model produces the final answer."

This is fundamental to how CODI was trained - not something we can fix without retraining CODI.

## Recommendations for Phase 3

1. **Diversify training data**: Add classification, comparison, and description questions
2. **Add position awareness**: Include step number in questions to help AO distinguish z2 vs z4
3. **Scale up**: Target 100K+ examples (paper uses 1M)
4. **Use real latents**: Ensure training uses actual CODI latent vectors, not random

## Files Changed
- `src/evaluation/evaluator.py` - Added qualitative output, comparison questions
- `src/evaluation/baselines.py` - Fixed empty prediction bug
- `scripts/evaluate.py` - Added multi-latent eval, comparison metrics
