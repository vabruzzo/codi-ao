# CODI Activation Oracle - 3-Step Experiment

Replicating the LessWrong paper's setup with 3-step math problems.

## Plan

### Phase 1: Data Generation
- [x] Create `generate_synthetic_data_3step.py` for 3-step problems
- [ ] Generate 1200 problems (1000 train, 200 test)

### Phase 2: Training Data
- [x] Create `generate_ao_training_data_3step.py` for 3 steps
- [ ] Generate training data with questions for step1, step2, step3

### Phase 3: Training
- [ ] Train AO on new data

### Phase 4: Evaluation
- [ ] Update `eval_ao_all6.py` for step3 extraction
- [ ] Update `eval_logit_lens_results.py` for step3
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

## Results

*To be filled after experiments*

| Task | Logit Lens | AO (single latent) | AO (all 6) |
|------|------------|-------------------|------------|
| Step 1 extraction | TBD | TBD | TBD |
| Step 2 extraction | TBD | TBD | TBD |
| Step 3 / Final answer | TBD | TBD | TBD |
| Operation detection | TBD | TBD | TBD |

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
# AO evaluation (to be created)
uv run python scripts/eval_ao_3step.py \
    --checkpoint checkpoints/ao_3step \
    --problems data/synthetic_problems_3step.json \
    --n_test 200 \
    --output results/ao_3step_evaluation.json

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
- Next: Generate problems and training data
