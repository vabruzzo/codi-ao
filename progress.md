# CODI-AO Progress Log

## Project Overview
Train an Activation Oracle (AO) to interpret CODI's continuous thought representations using natural language question-answering. CODI compresses chain-of-thought reasoning into 6 continuous latent vectors. The AO allows arbitrary natural language queries about the latent reasoning process.

## Repository Structure
- **codi/** — CODI submodule (github.com/zhenyi4/codi): model, training, evaluation, interpretability
- **activation_oracles/** — AO submodule (github.com/adamkarvonen/activation_oracles): full AO training framework (11k+ lines)
- **src/** — Main integration code (12 modules, fully implemented)
- **scripts/** — Pipeline scripts (5 numbered scripts + thin runner)
- **configs/** — YAML configuration (thin.yaml + full.yaml)
- **data/** — Generated datasets (gitignored)
- **checkpoints/** — Model checkpoints (gitignored)
- **results/** — Evaluation results (gitignored)

## What Exists
- CODI: complete implementation with pretrained checkpoints (zen-E/CODI-llama3.2-1b-Instruct, zen-E/CODI-gpt2)
- Activation Oracles: complete training framework with LatentQA, classification, past-lens datasets, evaluation infrastructure, distributed training, WandB integration
- Virtual environment with dependencies pinned (uv.lock)
- Detailed research plan (plan.md)
- Implementation plan (implementation_plan.md)

---

## Progress

### 2026-02-12 — Session 1: Project Review & Planning

#### Completed
- [x] Read full plan.md (284 lines, 5 phases)
- [x] Explored complete project structure
- [x] Inventoried CODI submodule: model.py (411 lines), train.py, test.py, probe_latent_token.py
- [x] Inventoried AO submodule: nl_probes/ (11,128 lines), 38+ experiment scripts, dataset loaders, evaluation framework
- [x] Confirmed v2 branch is a clean rewrite — old src/ and scripts/ files deleted
- [x] Identified key technical decisions (activation type, injection mechanism, base model)
- [x] Created progress.md file

#### Key Findings
- Both submodules are mature, well-tested codebases
- AO framework already supports: LoRA training, activation injection via residual stream hooks, multi-dataset training, evaluation
- CODI model exposes: hidden states at arbitrary layers, post-MLP projection vectors, decoded tokens per thought position
- The integration work is primarily data preparation and pipeline scripting — the core ML infrastructure exists

#### Decisions Made
- **Hardware**: A100/H100 cloud-hosted (can't run locally)
- **Strategy**: Thin end-to-end pipeline first (100 problems), then scale to 25k
- **GSM8k-Aug**: Download from HuggingFace (zen-E/GSM8k-Aug)
- **AO repo**: Use as reference, build custom pipeline adapted for CODI
- **Scope**: All milestones targeted (intermediate recovery → full reasoning → error localization)

---

### 2026-02-12 — Session 1: Implementation (continued)

#### Implementation Plan Created
- [x] Wrote detailed implementation plan (implementation_plan.md)
- [x] Defined 10-step implementation order with data format specifications
- [x] Specified exact interfaces between all modules

#### Step 1: Foundation Modules (no GPU needed)
- [x] `src/config.py` — All config dataclasses (CODIConfig, ExtractionConfig, QAConfig, AOTrainingConfig, EvalConfig) + thin pipeline presets
- [x] `src/utils.py` — Shared helpers: seed setting, number extraction, numeric/binary matching, file I/O
- [x] `src/cot_parser.py` — GSM8k-Aug CoT parser: splits `<<expr=result>>` blocks into CotStep records with operands, operations, result, uses_previous_result flag. Also includes `describe_full_reasoning()` for natural language generation
- [x] `src/qa_templates.py` — 190+ hand-written question template paraphrases across all 6 categories and 16+ sub-types

#### Step 2: CODI Model Loader
- [x] `src/codi_loader.py` — Adapts loading pattern from codi/test.py: creates ModelArguments + TrainingArguments, builds CODI, loads safetensors checkpoint, ties weights. Supports both safetensors and pytorch_model.bin formats. Returns (model, tokenizer) tuple.

#### Step 3: Activation Extractor
- [x] `src/activation_extractor.py` — Runs CODI inference loop per problem, captures 7 thoughts (initial + 6 iterations). For each thought: pre-projection hidden states at layers 4/8/12, post-projection vector, decoded top-5 tokens via lm_head. Also generates predicted answer and checks correctness. Includes save/load serialization to .pt files.

#### Step 4: Thought Alignment
- [x] `src/thought_alignment.py` — Maps thoughts to CoT steps by matching decoded top-1 token against intermediate results. Handles transitional (non-matching) thoughts. Includes summary statistics for alignment quality monitoring.

#### Step 5: QA Generator
- [x] `src/qa_generator.py` — Master generator producing QA pairs across all 6 categories:
  - Cat 1: Intermediate result recovery (single-thought + multi-thought variants)
  - Cat 2: Operation classification (8 sub-types: multiplication, addition, subtraction, division, result_gt_100, uses_previous, first_step, multi_operand)
  - Cat 3: Full reasoning description (programmatic NL generation)
  - Cat 4: Problem properties (6 sub-types: num_steps, final_answer, has_subtraction, negative_intermediate, answer_gt_1000, has_percentage)
  - Cat 5: Context prediction (predict_later, predict_earlier, predict_answer)
  - Cat 6: Thought informativeness (meaningful + computational_vs_transitional)
  - Includes `balance_binary_pairs()` for Yes/No distribution balancing

#### Step 6: AO Dataset Builder
- [x] `src/ao_dataset.py` — TrainingDataPoint + BatchData dataclasses. `create_training_datapoint()` converts QAPair → tokenized training example with chat template, -100 label masking, and special `?` token position finding. `construct_batch()` handles left-padding + position adjustment. Includes save/load for .pt files.

#### Step 7: Steering Hook
- [x] `src/steering.py` — Norm-matched additive steering adapted from AO repo. `get_steering_hook()` creates a forward hook that injects activations: `steered = F.normalize(v) * ||orig|| * coeff + orig`. `add_hook()` context manager. `get_injection_submodule()` resolves transformer layer for LLaMA/GPT2 architectures.

#### Step 8: AO Trainer
- [x] `src/ao_trainer.py` — Full training loop: loads fresh LLaMA-3.2-1B-Instruct, applies LoRA (r=64, α=128), trains with steering hook injection. AdamW + linear warmup/decay. Periodic eval + checkpoint saving. WandB logging (optional). Single GPU (no DDP for thin pipeline).

#### Step 9: Evaluation Suite
- [x] `src/ao_eval.py` — `generate_responses()` with steering hook during generation. Category-specific scoring: `score_numeric_exact_match()`, `score_binary()`, `score_text_similarity()`. `error_localization()` for identifying which thought diverged from ground truth on wrong-answer problems. JSON result saving + pretty-printing.

#### Step 10: Pipeline Scripts & Configs
- [x] `scripts/01_download_data.py` — Downloads GSM8k-Aug + CODI checkpoint from HuggingFace
- [x] `scripts/02_extract_activations.py` — YAML-configurable activation extraction
- [x] `scripts/03_generate_qa.py` — YAML-configurable QA dataset generation with balancing + stats
- [x] `scripts/04_train.py` — YAML-configurable AO training launcher
- [x] `scripts/05_eval.py` — YAML-configurable evaluation with LoRA loading
- [x] `scripts/run_thin_pipeline.sh` — One-command E2E runner (Steps 1-5, ~35 min on A100)
- [x] `configs/thin.yaml` — 100-problem thin pipeline config
- [x] `configs/full.yaml` — 25k-problem production config

---

## File Inventory

### Source Modules (src/)
| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~130 | All configuration dataclasses |
| `utils.py` | ~85 | Shared helpers |
| `cot_parser.py` | ~165 | GSM8k-Aug CoT parsing |
| `qa_templates.py` | ~310 | 190+ question template paraphrases |
| `codi_loader.py` | ~120 | CODI model loading |
| `activation_extractor.py` | ~255 | Thought activation extraction |
| `thought_alignment.py` | ~115 | Thought-to-step alignment |
| `qa_generator.py` | ~340 | QA pair generation (6 categories) |
| `ao_dataset.py` | ~260 | TrainingDataPoint creation + batching |
| `steering.py` | ~130 | Norm-matched injection hook |
| `ao_trainer.py` | ~250 | AO training loop |
| `ao_eval.py` | ~235 | Evaluation suite |

### Scripts (scripts/)
| File | Purpose |
|------|---------|
| `01_download_data.py` | Download datasets + checkpoints |
| `02_extract_activations.py` | Run CODI, save activations |
| `03_generate_qa.py` | Build QA dataset |
| `04_train.py` | Train the AO |
| `05_eval.py` | Run evaluation |
| `run_thin_pipeline.sh` | One-command E2E |

### Configs (configs/)
| File | Purpose |
|------|---------|
| `thin.yaml` | 100 problems, ~570 QA examples, 3 epochs |
| `full.yaml` | 25k problems, ~120k QA examples, 1 epoch |

---

## Architecture Notes

### CODI Model Key Details
- Base: LLaMA-3.2-1B-Instruct (hidden_dim=2048, 16 layers)
- 6 continuous thought tokens per problem (configurable via `num_latent`)
- Special tokens: `bot_id` (begin of thought), `eot_id` (end of thought)
- Post-MLP projection: Dropout → Linear(2048→2048) → GELU → Linear(2048→2048) → LayerNorm
- Distillation loss: smooth_l1 between student/teacher hidden states across ALL layers

### Activation Extraction Flow
```
Question tokens → encode (output_hidden_states=True) → Thought 0
  ↓ (prj → inputs_embeds)
Forward pass → Thought 1
  ↓ (prj → inputs_embeds)
... (repeat 5 more times)
Forward pass → Thought 6
  ↓ (eot_id → generate answer tokens)
```

Each thought: extract hidden_states at layers 4, 8, 12 + post-projection + lm_head top-5 decode.

### Injection Mechanism
```
AO Prompt: "Layer: 4\n ? ? ? ? ? ? ? \nDescribe the reasoning steps..."
                      ↑ ↑ ↑ ↑ ↑ ↑ ↑
                      t0 t1 t2 t3 t4 t5 t6  (CODI thought vectors)

At AO layer 1: steered = normalize(thought_vec) × ||orig|| × 1.0 + orig
```

### Key Technical Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Activation type | Pre-projection hidden states | Standard LLaMA space; AO has "privileged access" |
| Extraction layers | 4, 8, 12 (25/50/75%) | Follows AO paper convention |
| Injection layer | Layer 1 of AO | Follows AO paper default |
| AO base model | Fresh LLaMA-3.2-1B-Instruct | Same architecture; fresh weights for language modeling |
| Thought count | 7 (initial + 6 iterations) | Thought 0 marked specially |

---

## Next Steps (requires GPU)

### Immediate: Run Thin Pipeline
1. Set up cloud instance (A100/H100)
2. Clone repo, install dependencies
3. Run `bash scripts/run_thin_pipeline.sh`
4. Verify: no crashes, loss decreases, AO produces text

### After Thin Pipeline Validates:
1. Scale to full pipeline (25k problems, `configs/full.yaml`)
2. Run ablations (Cat 1+2 only, Cat 1+2+3+4, full)
3. Learning rate sweep [1e-6, 3e-6, 1e-5, 3e-5]
4. OOD evaluation on SVAMP, GSM-Hard, MultiArith
5. Error localization analysis on wrong-answer problems

---

## Phase Checklist

### Phase 1: Setup & Verification
- [ ] Download CODI-llama3.2-1b-Instruct checkpoint
- [ ] Reproduce GSM8k accuracy (expect 55.6%)
- [ ] Run probe_latent_token.py, verify interpretability patterns

### Phase 2: Activation Collection
- [x] Build activation extraction pipeline (src/activation_extractor.py)
- [x] Build CoT parser (src/cot_parser.py)
- [x] Build thought alignment (src/thought_alignment.py)
- [ ] Run extraction on 100 problems (thin)
- [ ] Run extraction on 25k problems (full)

### Phase 3: Build AO Training Dataset
- [x] Build QA generator with all 6 categories (src/qa_generator.py)
- [x] Build template paraphrases (src/qa_templates.py)
- [x] Build dataset builder (src/ao_dataset.py)
- [ ] Generate thin dataset (~570 examples)
- [ ] Generate full dataset (~120k examples)

### Phase 4: Train the AO
- [x] Build training loop (src/ao_trainer.py)
- [x] Build steering hook (src/steering.py)
- [ ] Run thin training (3 epochs, ~15 min)
- [ ] Run full training (1 epoch, ~2-3 hours)
- [ ] Ablation runs
- [ ] Learning rate sweep

### Phase 5: Evaluation
- [x] Build evaluation suite (src/ao_eval.py)
- [ ] Run thin evaluation
- [ ] Run full evaluation
- [ ] Comparison to CODI vocab projection
- [ ] OOD evaluation
- [ ] Error localization
