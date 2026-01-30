# CODI Activation Oracle - Implementation Plan

## Scope & Assumptions

- **Target model**: LLaMA-1B CODI checkpoint
- **Phased delivery**: MVP validation first, then scale to full 1M-example mix
- **Reference code**: Do not modify `codi/` and `activation_oracles/` directories; replicate needed logic
- **Design docs**: `codi.txt`, `activation_oracles.txt`, `codi_blog.txt`

---

## Architecture Overview

### Project Structure

```
codi-ao/
├── src/
│   ├── codi_wrapper.py          # Load CODI, run teacher/student, collect latents
│   ├── activation_oracle.py     # AO model with injection mechanism
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── latent_qa.py         # CoT-aligned QA pairs
│   │   ├── classification.py    # Binary classification tasks
│   │   └── context_pred.py      # Self-supervised context prediction
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Main training loop
│   │   └── config.py            # Training configuration
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py         # Evaluation harness
│       └── baselines.py         # Logit lens, linear probe
├── scripts/
│   ├── collect_latents.py       # Data generation
│   ├── train.py                 # Training entry point
│   └── evaluate.py              # Evaluation entry point
├── configs/
│   └── default.yaml             # Default hyperparameters
└── data/                        # Generated datasets (gitignored)
```

### Data Flow

```
[Math Problem] → [CODI Model] → [Latents + Teacher CoT]
                                        ↓
                              [Dataset Generators]
                                        ↓
                              [Training Mixture]
                                        ↓
                              [Activation Oracle]
                                        ↓
                              [Evaluation Suite] → [Metrics]
```

---

## Phase 1: MVP (Validation Loop)

**Goal**: Validate that the core hypothesis works before scaling up.

### 1.1 CODI Wrapper

Implement `src/codi_wrapper.py` with:
- Load CODI model from checkpoint
- Run teacher task → explicit CoT output
- Run student task → collect 6 latent vectors + bot token
- Optional: collect pre-projection vectors (for ablation)
- Layer selection for activation collection (25%, 50%, 75% depth)

**Validation**: Verify latent positions store intermediate results:
- z2 (index 1) → Step 1 intermediate result (100% accuracy achieved)
- z4 (index 3) → Step 2 intermediate result (85% accuracy achieved)

Note: LessWrong says "z3/z5" but their indexing includes initial position.
Our code excludes that, so their z3/z5 = our z2/z4.

### 1.2 Minimal Dataset Builders

Build GSM8k-style dataset generator (10k-50k samples):
- Parse teacher CoT into steps and results
- Align steps to latent indices
- Generate QA pairs for intermediate results

Build small classification set using same samples:
- Operation type (add/sub/mul/div)
- Numeric properties (>100, negative, whole number)

### 1.3 Activation Oracle Core

Implement `src/activation_oracle.py`:
- Same base model as CODI (LLaMA-1B)
- LoRA adapters (rank 64, alpha 128)
- Norm-matched additive injection at layer 1
- Placeholder token: `"?"`
- Prompt format: `Layer {L}: <ACT> ... <ACT> {question}`

**Injection formula**:
```
h'_i = h_i + (||h_i|| / ||v_i||) * v_i
```

### 1.4 Baselines + MVP Evaluation

Implement baselines:
- **Logit lens**: Project latent → vocab, argmax
- **Linear probe**: Logistic regression on latent → step result

Evaluate on held-out GSM8k-style subset:
- Intermediate result extraction accuracy
- Compare AO vs. baselines

### 1.5 MVP Exit Criteria

**Do not proceed to Phase 2 until**:
- [x] Logit lens replicates findings: z2=100% (Step 1), z4=85% (Step 2) ✓
- [x] Pipeline runs end-to-end without errors ✓
- [x] AO meets or exceeds logit lens on intermediate result extraction for z2/z4 ✓

**Results (Jan 29, 2026)**:
- AO: 97.75% overall (z2: 99.5%, z4: 96.0%)
- Logit Lens: 96.75% overall (z2: 100%, z4: 93.5%)
- See `PHASE1_RESULTS.md` for full details

---

## Phase 2: Evaluation Harness & Analysis (REVISED)

**Note**: Original plan called for full data generation, but we pivoted to evaluation first to understand AO limitations before scaling.

### 2.1 What We Built

- **Evaluation harness** (`scripts/evaluate.py`) with:
  - Single-latent QA evaluation
  - Classification evaluation (operation, position, magnitude, structure)
  - Multi-latent QA with comparison questions
  - Full qualitative output (every prompt/response)
  
- **Bug fixes**:
  - Empty prediction correctness bug in logit lens baseline

### 2.2 Key Discoveries

| Finding | Impact |
|---------|--------|
| Final answer not in latents | Can't ask AO for final answer |
| z2/z4 confusion pattern | Need position-aware training |
| Comparison at 8% | Need multi-latent comparison training |
| result_property at 70% | Need magnitude classification training |

### 2.3 Results (Pre-Phase 3)

- Single-Latent: AO 92%, Logit Lens 93%
- Classification: 88% overall (operation 100%, result_property 70%)
- Multi-Latent Comparison: 8%

See `PHASE2_RESULTS.md` for full details.

---

## Phase 3: Diverse Data Generation & Training (CURRENT)

### 3.0 Goals

Based on Phase 2 findings, Phase 3 addresses:
1. **Question diversity**: 15 paraphrases per type (matching AO paper)
2. **Task diversity**: Extraction, classification, comparison
3. **Scale**: 250K-1M examples (25-100% of AO paper)
4. **Position awareness**: Step-specific questions to reduce z2/z4 confusion

### 3.1 Data Generation

**Script**: `scripts/generate_phase3_data.py`

| Question Type | Templates | Description |
|---------------|-----------|-------------|
| extraction_generic | 15 | "What is the result?" |
| extraction_step1 | 15 | "What was calculated in step 1?" |
| extraction_step2 | 15 | "What was calculated in step 2?" |
| classification_magnitude | 15 each × 6 thresholds | "Is result > 50?" |
| classification_operation | 15 each × 4 ops | "Was division used?" |
| classification_position | 15 each × 2 positions | "Is this step 1?" |
| classification_structure | 15 | "Is this a computation?" |
| multi_latent_comparison | 15 | "Which step is larger?" |
| operation_type | 15 | "What operation was performed?" |

**Target**: 25K prompts → 250K examples (25% of AO paper scale)

### 3.2 Training Configuration

### 3.1 Configuration

```yaml
# LoRA
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: all_linear

# Optimization
learning_rate: 1e-5
batch_size: 16
warmup_ratio: 0.1
scheduler: linear_decay
max_grad_norm: 1.0
num_epochs: 1

# Activation settings
collection_layers: [0.25, 0.50, 0.75]  # As fractions of depth
injection_layer: 1

# Input format
single_latent_ratio: 0.67  # 2/3 single, 1/3 multi
```

### 3.2 Training Details

- Group-by-length batching for efficiency (~30% speedup)
- Periodic eval checkpoints on MVP metrics
- WandB logging for loss curves and eval metrics
- Save checkpoints at 10k, 50k, 100k, 500k, 1M steps

### 3.3 Text Inversion Mitigation

To prevent AO from just recovering input text:
- Limit context in training prompts
- Vary activation positions randomly
- Use single-token inputs for 2/3 of classification data

---

## Phase 4: Evaluation & Ablations

### 4.1 In-Distribution Evaluation

- **Intermediate result accuracy**: Exact match on z2/z4 results
- **Operation classification**: Binary accuracy
- **Reasoning description**: Qualitative assessment

### 4.2 Out-of-Distribution Evaluation

| Dataset | Description |
|---------|-------------|
| SVAMP | Grade-school arithmetic variations |
| MultiArith | Multi-step word problems |
| GSM-Hard | Larger magnitude numbers |

### 4.3 Error Detection

- Given latent vectors from incorrect CODI predictions
- Ask AO: "Does this reasoning contain errors?"
- Measure precision/recall for error detection

### 4.4 Ablations

| Ablation | Options |
|----------|---------|
| Latent type | Projected vs. pre-projection |
| Injection layer | Layer 0 vs. layer 1 |
| Input format | Single vs. multi-latent |
| Dataset mix | MVP-only vs. full mix |
| Collection layer | 25% vs. 50% vs. 75% depth |

### 4.5 Baselines for Comparison

| Method | Description |
|--------|-------------|
| Logit Lens | Project latent to vocab, argmax (from LessWrong) |
| Linear Probe | Train linear classifier: latent → result |
| PatchScopes | Untrained AO (zero-shot injection) |
| SPQA-only | AO trained only on QA task |

---

## Phase 5: Reporting

### 5.1 Metrics Summary

- Table of all eval metrics across conditions
- Learning curves (loss, eval accuracy over training)
- Ablation results with confidence intervals

### 5.2 Qualitative Analysis

- Examples of successful decoding
- Examples of failures with analysis
- Comparison of AO responses vs. logit lens output

### 5.3 Artifacts

Save to `reports/`:
- `metrics.json`: All numeric results
- `examples.json`: Curated success/failure cases
- `ablations.csv`: Ablation study results
- `figures/`: Plots and visualizations

---

## Key Technical Decisions

### Projected vs. Pre-Projection Latents

| Option | Pros | Cons |
|--------|------|------|
| **Projected** (default) | What CODI uses; reasoning-relevant | Different space than residual stream |
| **Pre-projection** | More residual-stream-like | May miss projection-added info |

**Decision**: Start with projected (what model actually uses). Treat pre-projection as ablation.

### Which Layer to Inject At

Per AO paper: Layer 1 outperforms Layer 0 for LoRA fine-tuning (11% improvement on some tasks).

### Single vs. Multi-Latent Inputs

- **Single**: Better for position-specific questions
- **Multi**: Better for holistic reasoning questions
- **Mix**: 2/3 single, 1/3 multi (per AO paper)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Latent alignment noise | Validate z2/z4 mapping on MVP before scaling (done: 100%/85%) |
| Projection space mismatch | Compare projected vs. pre-proj latents in ablation |
| Text inversion leakage | Limit context, vary positions, use single-token inputs |
| Small model capacity | Start with LLaMA-1B; scale if needed |

---

## Success Criteria

1. **AO ≥ logit lens** on intermediate result extraction (baseline: 97% single-step)
2. **AO answers questions logit lens can't** (operation type, reasoning structure)
3. **AO generalizes to OOD datasets** without retraining
4. **AO detects reasoning errors** with reasonable precision/recall

---

## Files to Create

### Phase 1 (MVP)
- [ ] `src/codi_wrapper.py`
- [ ] `src/activation_oracle.py`
- [ ] `src/datasets/latent_qa.py`
- [ ] `src/datasets/classification.py`
- [ ] `src/evaluation/baselines.py`
- [ ] `src/evaluation/evaluator.py`
- [ ] `scripts/collect_latents.py`
- [ ] `scripts/train.py`
- [ ] `scripts/evaluate.py`
- [ ] `configs/default.yaml`

### Phase 2 (Full Data)
- [ ] `src/datasets/context_pred.py`
- [ ] Extended dataset generation scripts

### Phase 3-5
- [ ] `src/training/trainer.py`
- [ ] `src/training/config.py`
- [ ] Evaluation and reporting scripts
