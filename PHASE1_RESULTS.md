# Phase 1: MVP Validation Results

**Date**: January 29, 2026  
**Status**: ✓ PASSED

## Objective

Validate that an Activation Oracle (AO) can decode CODI's latent vectors into human-readable intermediate calculation results, matching or exceeding a logit lens baseline.

---

## Background

### CODI Model

CODI (Chain-of-Thought via Dual-Inference) compresses chain-of-thought reasoning into 6 latent vectors instead of generating explicit reasoning tokens. This enables faster inference while maintaining accuracy.

- **Architecture**: LLaMA-1B with latent reasoning heads
- **Training data**: GSM8k-Aug (math word problems)
- **Key finding** (from LessWrong blog): Latent positions z2 and z4 encode intermediate calculation results

### Activation Oracle

The Activation Oracle is a model trained to decode activations from another model into natural language. We adapt this approach to decode CODI's latent vectors.

- **Base model**: LLaMA-3.2-1B-Instruct
- **Adaptation**: LoRA (rank 64, alpha 128)
- **Injection**: Norm-matched additive at layer 1

---

## Approach

### Data Generation

1. Generate synthetic math problems using CODI's training templates
   - Format: `(X+Y)*Z + (X+Y)` structure
   - Known intermediate results: step1 = X+Y, step2 = (X+Y)*Z

2. Run CODI on each problem to collect latent vectors
   - z2 (index 1) → step 1 intermediate result
   - z4 (index 3) → step 2 intermediate result

3. Create training examples pairing latents with ground truth results
   - Prompt format: `"Layer 50%: ? What is the intermediate calculation result?"`
   - Answer: The numeric result (e.g., "15")

### Training

| Setting | Value |
|---------|-------|
| Training examples | 10,000 |
| Prompts | 5,000 (2 examples each: z2 and z4) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Trainable params | 6.8M (0.55% of model) |
| Training time | ~3 minutes |

### Evaluation

- **Test set**: 200 fresh prompts (seed 123, different from training seed 42)
- **Metric**: Exact match accuracy on intermediate results
- **Baseline**: Logit lens (project latent → vocab → argmax)

---

## Results

### Logit Lens Baseline (established during data generation)

| Position | Accuracy | Description |
|----------|----------|-------------|
| z2 (Step 1) | 100.00% | First intermediate result |
| z4 (Step 2) | 85.36% | Second intermediate result |

### Final Evaluation (held-out test set)

| Method | z2 Accuracy | z4 Accuracy | Overall |
|--------|-------------|-------------|---------|
| **Logit Lens** | 100.00% | 93.50% | 96.75% |
| **Activation Oracle** | 99.50% | **96.00%** | **97.75%** |

### Key Findings

1. **AO beats logit lens overall**: 97.75% vs 96.75% (+1.0%)
2. **AO excels on z4**: 96.0% vs 93.5% (+2.5%) - the harder position
3. **Generalization confirmed**: Trained on seed 42, tested on seed 123
4. **Efficient training**: Only 3 epochs needed to exceed baseline

---

## Technical Details

### Prompt Format

```
Layer 50%: ? What is the intermediate calculation result?
```

The `?` is a placeholder token where the latent vector is injected via norm-matched addition:

```
h' = h + (||h|| / ||v||) * v
```

### Question Templates (20 variations)

- "What is the intermediate calculation result?"
- "What number is computed at this step?"
- "What value is stored in this reasoning position?"
- ... (17 more paraphrases)

### Index Mapping

The LessWrong blog refers to "z3 and z5" but their indexing includes an initial position. Our implementation excludes that:

| Blog Reference | Our Index | Contains |
|----------------|-----------|----------|
| z3 | 1 (z2) | Step 1 result |
| z5 | 3 (z4) | Step 2 result |

---

## Files

| File | Purpose |
|------|---------|
| `scripts/collect_latents.py` | Data generation and baseline validation |
| `scripts/train.py` | AO training loop |
| `scripts/evaluate.py` | Evaluation against baselines |
| `src/activation_oracle.py` | AO model with injection mechanism |
| `src/codi_wrapper.py` | CODI model interface |
| `data/latent_qa_train.jsonl` | Training data (10k examples) |
| `checkpoints/ao/` | Trained LoRA adapter |

---

## Limitations

1. **Template-only data**: Trained and tested on synthetic templates with known structure
2. **Fixed positions**: Assumes z2/z4 always encode step1/step2 (may not hold for diverse problems)
3. **Single task**: Only tested intermediate result extraction, not classification or error detection

---

## Next Steps (Phase 2)

1. **Expand templates**: More diverse problem structures
2. **Add classification tasks**: "What operation was performed?" (add/sub/mul/div)
3. **Test on GSM8k**: Evaluate transfer to natural language word problems
4. **Error detection**: Can AO identify when CODI made a reasoning mistake?
5. **Scale training**: Move toward 1M example dataset with full task mix

---

## Commands Reference

```bash
# Generate training data
uv run python scripts/collect_latents.py \
  --n_samples 5000 \
  --generate_data \
  --synthetic \
  --verbose

# Train AO
uv run python scripts/train.py \
  --mode mvp \
  --epochs 3 \
  --batch_size 16 \
  --lr 1e-5 \
  --verbose

# Evaluate
uv run python scripts/evaluate.py \
  --ao_path checkpoints/ao \
  --n_samples 200 \
  --fresh_test \
  --verbose
```

---

## Conclusion

Phase 1 MVP successfully validated that an Activation Oracle can decode CODI's latent vectors. The trained AO exceeds the logit lens baseline, demonstrating that the injection mechanism works and the model learns to extract meaningful information from compressed reasoning representations.
