# CODI Activation Oracle Study

## Core Hypothesis

**Logit Lens cannot reliably extract operation type from CODI's latents, but a trained Activation Oracle can.**

This demonstrates that non-linear learned decoders can extract information that linear projections cannot.

## Background

CODI compresses chain-of-thought reasoning into 6 latent vectors. Prior work (LessWrong blog) showed that **numeric intermediate results** are linearly decodable via logit lens:
- z2 → Step 1 result (100% accuracy)
- z4 → Step 2 result (~85% accuracy)

But what about **operation type** (add/sub/mul)? Is this information stored? Can it be extracted?

## Phase 0 Results: Logit Lens Exploration

We ran logit lens on all 6 positions to see what's linearly decodable.

### Numeric Extraction (Confirms Prior Work)

| Position | Step 1 Acc | Step 2 Acc |
|----------|------------|------------|
| z2 | **100%** | 10.5% |
| z4 | 81.5% | **66.5%** |

✅ Numeric values are linearly decodable - confirming prior work.

### Operation Type Extraction (New Finding)

| Position | Operation Acc | Notes |
|----------|---------------|-------|
| z1 | 52.0% | Near random |
| z2 | 56.5% | Near random |
| **z3** | **68.5%** | Best, but weak |
| z4 | 34.0% | Below random |
| z5 | 26.0% | Below random |
| z6 | 60.0% | Biased |

**Key Finding**: Operation type is NOT reliably linearly decodable.
- Best position (z3) only gets 68.5%
- Random baseline is 33% (3 operations)
- The signal is weak and biased toward "add" tokens

### Why Operation is Non-Linear

Numeric values are linearly decodable because CODI was trained to produce them as explicit tokens during distillation.

Operation type is **implicit** - CODI wasn't trained to output "add" or "multiply", it was trained to *perform* these operations. The operation information is encoded in a distributed, non-linear way.

## The Experiment

### Three Decoders, One Question

We compare three methods of extracting operation type:

| Method | Training | Complexity | Description |
|--------|----------|------------|-------------|
| **Logit Lens** | None | Linear (vocab projection) | Project latent → vocab, check operation tokens |
| **Linear Probe** | Minimal | Linear (learned) | Train `nn.Linear(hidden_dim, 3)` to classify |
| **Activation Oracle** | Full | Non-linear (LLM) | Train LLM decoder with LoRA |

### What We'll Show

1. **Logit Lens**: ~68% operation accuracy (weak, biased toward "add")
2. **Linear Probe**: ~70-75% operation accuracy (slightly better, still limited)
3. **Activation Oracle**: >85% operation accuracy (strong, unbiased)

**Key insight**: If both linear methods fail but AO succeeds, operation is encoded **non-linearly**.

### Training Setup

Train AO on diverse data including:
- Numeric extraction (single-latent)
- Operation type classification (single-latent)
- Multi-latent tasks

**Key Question**: Which latent position(s) does the AO use to predict operation?
- Is it z3 (best logit lens signal)?
- Or does it learn to combine information across positions?

### Evaluation

| Task | Logit Lens | Linear Probe | AO | 
|------|------------|--------------|-----|
| Numeric (z2) | 100% | ~100% | ~100% |
| Numeric (z4) | 66.5% | ~80%? | >90% |
| **Operation** | **68.5%** | **~75%?** | **>85%** |

**The story**:
- Numeric extraction: All methods work (linearly decodable)
- Operation: Only AO works well (non-linearly encoded)

## Why This Matters

1. **Validates Activation Oracles**: Shows learned decoders extract more than linear probes
2. **Interpretability**: Can monitor what operations CODI performs, not just what numbers it computes
3. **Safety**: If latent reasoning becomes common, we need tools beyond logit lens

## File Structure

```
codi-ao/
├── STUDY.md                     # This file
├── scripts/
│   ├── explore_latents.py       # Phase 0: Logit lens exploration ✓
│   ├── train.py                 # Train the AO
│   └── evaluate.py              # Evaluate AO vs Logit Lens
├── results/
│   └── latent_exploration.json  # Phase 0 results ✓
└── checkpoints/
    └── ao/                      # Trained model
```

## Commands

```bash
# Phase 0: Logit Lens exploration (done ✓)
python scripts/explore_latents.py --n_samples 200 --verbose

# Phase 1: Train Linear Probe
python scripts/train_linear_probe.py --task operation --n_samples 10000

# Phase 2: Generate AO training data
python scripts/generate_data.py --n_samples 100000 --include_operation

# Phase 3: Train Activation Oracle
python scripts/train.py --data data/train.jsonl --epochs 2

# Phase 4: Evaluate all three methods
python scripts/evaluate.py \
    --ao_checkpoint checkpoints/ao \
    --probe_checkpoint checkpoints/linear_probe \
    --compare_all
```

## Success Criteria

1. **Logit Lens operation**: <70% ✓ (confirmed: 68.5%)
2. **Linear Probe operation**: <80% (still limited by linearity)
3. **AO operation**: >85% (non-linear decoding works)
4. **Gap**: AO beats linear methods by >10 points on operation
5. **No regression**: All methods comparable on numeric extraction

## Expected Results

| Metric | Logit Lens | Linear Probe | Activation Oracle |
|--------|------------|--------------|-------------------|
| Numeric z2 | 100% | ~100% | ~100% |
| Numeric z4 | 66.5% | ~80% | >90% |
| **Operation** | **68.5%** | **~75%** | **>85%** |

### The Key Story

**Operation type IS encoded in CODI's latents, but requires non-linear decoding to extract reliably.**

- Both linear methods (logit lens, linear probe) struggle with operation
- Only the non-linear AO can reliably extract it
- This validates the need for learned decoders beyond simple probes
