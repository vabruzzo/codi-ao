# CODI Activation Oracle - 3-Step Experiment

Replicating the LessWrong paper's setup with 3-step math problems.

## Plan

### Phase 1: Data Generation
- [x] Create `generate_synthetic_data_3step.py` for 3-step problems
- [ ] Generate 1200 problems (1000 train, 200 test)

### Phase 2: Training Data
- [x] Create `generate_ao_training_data_3step.py` for 3 steps
- [x] Include BOTH single-latent and multi-latent examples
- [ ] Generate training data with questions for step1, step2, step3

### Phase 3: Training
- [ ] Train AO on new data

### Phase 4: Evaluation
- [x] Create `eval_ao_3step.py` (single + multi latent tests)
- [ ] Create `eval_logit_lens_3step.py` for step3
- [ ] Run all evaluations

### Phase 5: Analysis
- [ ] Compare to LessWrong findings
- [ ] Write up results

---

## Problem Structure (matching LessWrong)

**Example:**
> "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team?"

**Computation:**
- Step 1: 3 + 5 = **8** (X + Y)
- Step 2: 8 * 2 = **16** (step1 * Z)
- Step 3: 8 + 16 = **24** (step1 + step2 = final answer)

**Latent encoding (expected):**
- z2 (index 1): step1 result
- z4 (index 3): step2 result
- z6 (index 5): step3 / final answer (?)

---

## Training Data Structure

Training includes BOTH single-latent and multi-latent examples:

| Example Type | Latents | Questions |
|-------------|---------|-----------|
| Single z2 | position 1 only | step1 extraction, operation, operands |
| Single z4 | position 3 only | step2 extraction |
| Single z6 | position 5 only | step3/final answer extraction |
| Multi (all 6) | positions 0-5 | all questions |

This enables evaluation of:
1. **What does each latent encode?** (single-latent tests)
2. **Does multi-latent context help?** (compare single vs multi)

---

## Results

*To be filled after experiments*

### Extraction Results

| Task | Logit Lens | AO (single latent) | AO (all 6) |
|------|------------|-------------------|------------|
| Step 1 (z2) | TBD | TBD | TBD |
| Step 2 (z4) | TBD | TBD | TBD |
| Step 3 / Final (z6) | TBD | TBD | TBD |

### Other Metrics

| Task | AO (single) | AO (multi) |
|------|-------------|------------|
| Operation (step 1) | TBD | TBD |
| First operand | TBD | TBD |
| Second operand | TBD | TBD |
| Full calculation | TBD | TBD |
| Comparison | N/A | TBD |

---

## Commands

### Phase 1: Generate 3-step problems
```bash
uv run python scripts/generate_synthetic_data_3step.py \
    --n_samples 1200 \
    --seed 42 \
    --output data/synthetic_problems_3step.json
```

### Phase 2: Generate training data
```bash
uv run python scripts/generate_ao_training_data_3step.py \
    --problems data/synthetic_problems_3step.json \
    --output data/ao_training_data_3step.jsonl \
    --holdout 200
```

### Phase 3: Train AO
```bash
uv run python scripts/train.py \
    --data data/ao_training_data_3step.jsonl \
    --output_dir checkpoints/ao_3step \
    --epochs 2 \
    --batch_size 16
```

### Phase 4: Evaluate
```bash
# AO evaluation - normal
uv run python scripts/eval_ao_3step.py \
    --checkpoint checkpoints/ao_3step \
    --problems data/synthetic_problems_3step.json \
    --n_test 200 \
    --output results/ao_3step_evaluation.json

# AO evaluation - shuffle sanity check
uv run python scripts/eval_ao_3step.py \
    --checkpoint checkpoints/ao_3step \
    --problems data/synthetic_problems_3step.json \
    --n_test 200 \
    --shuffle \
    --output results/ao_3step_shuffle.json

# Logit Lens evaluation (to be created)
uv run python scripts/eval_logit_lens_3step.py \
    --data data/synthetic_problems_3step.json \
    --output results/logit_lens_3step.json
```

---

## Progress Log

### Session Start
- Created `generate_synthetic_data_3step.py` for 3-step problems
- Created `generate_ao_training_data_3step.py` for training data
  - Includes BOTH single-latent and multi-latent examples
  - Single z2 → step1, operation, operands
  - Single z4 → step2
  - Single z6 → step3/final answer
  - Multi (all 6) → all questions
- Created `eval_ao_3step.py` for comprehensive evaluation
  - Tests single vs multi latent performance
  - Includes shuffle sanity check
- Next: Generate problems and training data
