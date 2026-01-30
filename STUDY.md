# CODI Activation Oracle Study

## Overview

This study trains an Activation Oracle (AO) to decode CODI's latent reasoning vectors and compares against Logit Lens baselines wherever possible.

**Key Question**: Can a learned decoder (AO) extract more information from CODI's latents than a simple linear projection (Logit Lens)?

## CODI Latent Structure

CODI produces 6 latent vectors (z1-z6). The LessWrong blog found that:
- **z2** (index 1) stores Step 1 numeric result
- **z4** (index 3) stores Step 2 numeric result

But what about the other positions? They might encode:
- Operation type (add/sub/mul)?
- Problem structure?
- Final answer?
- Something else entirely?

### Phase 0: Exploratory Logit Lens (All 6 Positions)

Before training, we run Logit Lens on **all 6 positions** to discover what's linearly decodable:

```bash
python scripts/explore_latents.py \
    --n_samples 200 \
    --output results/latent_exploration.json
```

For each position z1-z6, we test:
1. **Numeric extraction**: Does argmax give a meaningful number?
2. **Operation tokens**: Are add/sub/mul tokens high probability?
3. **Position tokens**: Are "first"/"second" tokens high probability?
4. **Top-k tokens**: What are the highest probability tokens overall?

**Expected output**:
```
z1: top tokens = [?, ?, ?], numeric = ?, operation = ?
z2: top tokens = [15, 16, ...], numeric = step1_value, operation = ?
z3: top tokens = [?, ?, ?], numeric = ?, operation = ?
z4: top tokens = [45, 48, ...], numeric = step2_value, operation = ?
z5: top tokens = [?, ?, ?], numeric = ?, operation = ?
z6: top tokens = [?, ?, ?], numeric = ?, operation = ?
```

This tells us:
- Which positions encode numeric values (we know z2/z4, but confirm)
- Which positions (if any) encode operation type
- Whether z1/z3/z5/z6 contain any interpretable signal
- What information the AO could potentially extract beyond what LL finds

## Logit Lens Baselines

We extend Logit Lens beyond numeric extraction to test what information is linearly decodable from CODI's latent space. All Logit Lens tests are on **z2 and z4** (the meaningful positions).

### Logit Lens for Numeric Extraction (Standard)
```python
def logit_lens_numeric(latent, model, tokenizer):
    """Project latent (z2 or z4) → vocab, return top numeric token."""
    logits = latent @ model.lm_head.weight.T
    top_id = logits.argmax()
    return tokenizer.decode(top_id)
```

### Logit Lens for Operation Type (New)
```python
def logit_lens_operation(latent, model, tokenizer):
    """Check which operation tokens have highest probability."""
    logits = latent @ model.lm_head.weight.T
    probs = softmax(logits)
    
    # Define operation token sets
    add_tokens = tokenizer.encode("add addition plus +", add_special_tokens=False)
    sub_tokens = tokenizer.encode("subtract subtraction minus -", add_special_tokens=False)
    mul_tokens = tokenizer.encode("multiply multiplication times *", add_special_tokens=False)
    
    # Sum probabilities for each operation
    add_prob = probs[add_tokens].sum()
    sub_prob = probs[sub_tokens].sum()
    mul_prob = probs[mul_tokens].sum()
    
    return max([("add", add_prob), ("sub", sub_prob), ("mul", mul_prob)], key=lambda x: x[1])
```

### Logit Lens for Magnitude (New)
```python
def logit_lens_magnitude(latent, model, tokenizer, threshold=50):
    """Check if projected value > threshold."""
    logits = latent @ model.lm_head.weight.T
    
    # Get probabilities for numbers > threshold vs <= threshold
    high_tokens = [tokenizer.encode(str(n))[0] for n in range(threshold+1, 200)]
    low_tokens = [tokenizer.encode(str(n))[0] for n in range(0, threshold+1)]
    
    high_prob = softmax(logits)[high_tokens].sum()
    low_prob = softmax(logits)[low_tokens].sum()
    
    return "Yes" if high_prob > low_prob else "No"
```

## Training Setup

### Data Generation

We follow the Activation Oracle paper's diverse training approach:

**Total**: ~100,000 examples

| Question Type | Count | Single/Multi | Description |
|---------------|-------|--------------|-------------|
| extraction_generic | 15,000 | Single | "What is the intermediate result?" |
| extraction_step1 | 8,000 | Single | "What is the step 1 result?" |
| extraction_step2 | 8,000 | Single | "What is the step 2 result?" |
| operation_type | 15,000 | Single | "What operation was performed?" → add/sub/mul |
| classification_magnitude | 15,000 | Single | "Is result > 50?" → Yes/No |
| classification_position | 8,000 | Single | "Is this step 1?" → Yes/No |
| multi_extraction | 8,000 | Multi (6 latents) | "What is step 1/step 2 result?" |
| multi_comparison | 8,000 | Multi (6 latents) | "Which step is larger?" |
| multi_operation | 8,000 | Multi (6 latents) | "What operations were performed?" |
| multi_sequence | 7,000 | Multi (6 latents) | "What calculations were done in order?" |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | LLaMA-3.2-1B-Instruct |
| Adapter | LoRA (rank 64, alpha 128) |
| Trainable Params | ~6.8M (0.55%) |
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Epochs | 2 |
| Injection Layer | 1 |

### Prompt Formats

**Single-Latent** (1 placeholder):
```
Layer 50%: ? What is the intermediate result?
Layer 50%: ? What operation was performed?
Layer 50%: ? Is the result greater than 50?
```

**Multi-Latent** (6 placeholders for all CODI latents):
```
Layer 50%: ? ? ? ? ? ? What is the step 1 result?
Layer 50%: ? ? ? ? ? ? Which step has the larger value?
```

## Evaluation: AO vs Logit Lens

We compare AO and Logit Lens on every task where Logit Lens is applicable:

### Tasks with Logit Lens Comparison

| Task | AO Applicable | LL Applicable | How LL Works |
|------|---------------|---------------|--------------|
| Numeric Extraction | Yes | **Yes** | argmax over number tokens |
| Operation Type | Yes | **Yes** | Compare add/sub/mul token probs |
| Magnitude (>50?) | Yes | **Yes** | Compare high vs low number probs |
| Position (step 1?) | Yes | **Maybe** | Check for ordinal tokens? |

### Tasks without Logit Lens Comparison

| Task | Why No LL |
|------|-----------|
| Multi-Latent Comparison | Requires reasoning across latents |
| Multi-Latent Sequence | Requires ordering/aggregation |
| Free-form Description | Requires generation |

## Metrics

For each task:
```python
{
    "task": str,
    "ao_accuracy": float,
    "logit_lens_accuracy": float,      # None if not applicable
    "ao_vs_ll_delta": float,           # AO - LL (positive = AO wins)
    "random_baseline": float,          # Expected accuracy from random guessing
    "n_samples": int,
}
```

**Summary Metrics**:
- `ao_avg`: Average AO accuracy across all tasks
- `ll_avg`: Average LL accuracy (where applicable)
- `ao_wins`: Number of tasks where AO > LL
- `ll_wins`: Number of tasks where LL > AO

## Expected Results

### Single-Latent Tasks (z2 and z4 only)

| Task | Position | Random | Logit Lens | AO Target | AO vs LL |
|------|----------|--------|------------|-----------|----------|
| Numeric Extraction | z2 | ~1% | ~100% | 100% | 0% |
| Numeric Extraction | z4 | ~1% | ~85% | >95% | **+10%** |
| Operation Type | z2 | 33% | ? | >85% | ? |
| Operation Type | z4 | 33% | ? | >85% | ? |
| Magnitude (>50) | z2/z4 | 50% | ? | >85% | ? |
| Position (step 1?) | z2 | 50% | ? | >90% | ? |
| Position (step 2?) | z4 | 50% | ? | >90% | ? |

### Multi-Latent Tasks

| Task | Random | Logit Lens | AO Target |
|------|--------|------------|-----------|
| Step 1 Extraction | ~1% | N/A | 100% |
| Step 2 Extraction | ~1% | N/A | 100% |
| Comparison | 50% | N/A | >90% |
| Sequence | ~1% | N/A | >80% |

## Research Questions

1. **What's in each latent position?**
   - z2/z4 have numeric results (known) - but what else?
   - Do z1/z3/z5/z6 encode anything meaningful?
   - Is operation type stored somewhere? Which position?

2. **What can Logit Lens extract?**
   - Numeric values: Yes from z2/z4 (known)
   - Operation type: Unknown - which position? Test all 6!
   - Magnitude: Unknown - test this!
   - Position: Unknown - test this!

3. **Where does AO beat Logit Lens?**
   - On harder extraction (z4)?
   - On classification tasks?
   - On information that's not linearly decodable?

4. **What requires a learned decoder?**
   - Multi-latent reasoning?
   - Information spread across positions?
   - Non-linear combinations?

5. **Hidden Knowledge**
   - Can AO reveal correct intermediates when CODI outputs wrong answers?
   - Is this information accessible via LL too?

## File Structure

```
codi-ao/
├── STUDY.md                     # This file
├── scripts/
│   ├── explore_latents.py       # Explore all 6 positions with Logit Lens
│   ├── generate_data.py         # Generate training data
│   ├── train.py                 # Train the AO
│   ├── evaluate.py              # Run full evaluation
│   └── logit_lens.py            # Logit Lens baselines
├── data/
│   └── train.jsonl              # Training data
├── checkpoints/
│   └── ao/                      # Trained model
└── results/
    ├── latent_exploration.json  # What's in each latent position
    ├── logit_lens_baseline.json # LL results
    ├── evaluation.json          # Full AO results
    └── figures/                 # Plots
```

## Commands

```bash
# 0. FIRST: Explore all 6 latent positions with Logit Lens
python scripts/explore_latents.py \
    --n_samples 200 \
    --output results/latent_exploration.json

# 1. Generate training data (informed by exploration results)
python scripts/generate_data.py \
    --n_samples 100000 \
    --output data/train.jsonl

# 2. Evaluate Logit Lens baselines on meaningful positions
python scripts/logit_lens.py \
    --positions all \
    --tasks all \
    --n_samples 500 \
    --output results/logit_lens_baseline.json

# 3. Train the AO
python scripts/train.py \
    --data data/train.jsonl \
    --epochs 2 \
    --output checkpoints/ao

# 4. Evaluate AO and compare
python scripts/evaluate.py \
    --checkpoint checkpoints/ao \
    --ll_baseline results/logit_lens_baseline.json \
    --output results/evaluation.json
```

## Success Criteria

1. **Establish LL baselines**: Know what Logit Lens can/cannot extract
2. **AO beats LL on extraction**: >5% improvement on numeric extraction
3. **AO enables new tasks**: Multi-latent comparison >85% (LL cannot do this)
4. **Clear conclusions**: Identify what requires learned vs linear decoding

## Key Insight We're Testing

The Activation Oracle paper shows that learned decoders can extract more information than linear probes. We test whether this holds for CODI's compressed reasoning:

- **If LL can extract operation type**: CODI's latents have rich linear structure
- **If only AO can extract operation type**: Non-linear decoding is needed
- **If neither can extract operation type**: This information may not be stored in latents
