# CODI-AO: Activation Oracle for Continuous Chain-of-Thought

**Train a language model to interpret CODI's latent reasoning via natural language Q&A.**

CODI ([Zhen et al., 2025](https://arxiv.org/abs/2502.21074)) compresses explicit chain-of-thought reasoning into 6 continuous latent vectors through self-distillation. These vectors carry structured reasoning information — intermediate results, operation types, step dependencies — but can only be inspected today by projecting them back into vocabulary space, a lossy and inflexible method.

CODI-AO trains an [Activation Oracle](https://arxiv.org/abs/2512.15674) on these continuous thought representations, enabling arbitrary natural language queries about CODI's latent reasoning process: *"What intermediate result does this thought compute?"*, *"Describe the full reasoning chain"*, *"Is this a computational or transitional step?"*

---

## Why This Works

CODI's continuous thoughts are unusually well-suited for Activation Oracle training:

1. **The information is guaranteed to be there.** CODI's distillation loss explicitly trains the continuous thoughts to encode the same reasoning as the explicit CoT. Unlike arbitrary model activations, these representations were *optimized* to carry this information.

2. **Ground truth is free.** Every training problem has paired explicit CoT from GSM8k-Aug. The CoT annotations *are* the labels — no expensive human annotation or LLM-generated labels needed.

3. **The AO paper's hardest setting maps directly here.** The AO paper's strongest results came from tasks where information exists in activations but is absent from the input text. That's exactly CODI's student path: the input is just the question, the continuous thoughts contain the reasoning, and the CoT tokens never appear in the text stream.

---

## Architecture

```
                          CODI (frozen)                         Activation Oracle
                    ┌─────────────────────┐              ┌──────────────────────────┐
                    │                     │              │                          │
  Question ──────► │  Iterative Latent   │──thoughts──► │  Layer: 4                │
  "Out of 600      │  Loop (6 rounds)    │   [2048]     │  ? ? ? ? ? ? ?           │
   employees..."   │                     │              │  What intermediate       │──► "180"
                    │  Extract at layers  │              │  result does thought     │
                    │  4, 8, 12          │              │  3 compute?              │
                    └─────────────────────┘              └──────────────────────────┘
                     LLaMA-3.2-1B + LoRA                  LLaMA-3.2-1B + LoRA
                     (CODI checkpoint)                    (fresh checkpoint)

  Injection: At AO layer 1, norm-matched additive steering
  steered = normalize(thought_vec) × ‖original‖ × coefficient + original
```

The AO is a **fresh** LLaMA-3.2-1B-Instruct (not CODI's checkpoint) — it needs intact language modeling capabilities to produce natural language answers. CODI thought activations are injected into its residual stream at special `?` token positions via a forward hook on transformer layer 1.

---

## QA Categories

The AO is trained on 6 categories of questions about CODI's latent reasoning:

| # | Category | Input | Example Question | Example Answer |
|---|----------|-------|-----------------|----------------|
| 1 | **Intermediate Result** | Single thought | "What numerical result does this computation produce?" | "180" |
| 2 | **Operation Classification** | Single thought | "Does this step involve multiplication?" | "Yes" |
| 3 | **Full Reasoning** | All 7 thoughts | "Describe the reasoning steps used to solve this problem." | "First, multiply 600 by 30 and divide by 100 to get 180. Then..." |
| 4 | **Problem Properties** | All 7 thoughts | "How many arithmetic steps are involved?" | "4" |
| 5 | **Context Prediction** | Subset of thoughts | "Given these earlier steps, what is the final answer?" | "360" |
| 6 | **Thought Informativeness** | Single thought | "Is this a computational step or transitional?" | "computational" |

Each category uses 10-15 paraphrased question templates (190+ total) to prevent the AO from overfitting to specific phrasings. Binary questions are balanced by construction.

---

## Repository Structure

```
codi-ao/
├── src/                              # Core implementation (12 modules)
│   ├── config.py                     # All configuration dataclasses
│   ├── codi_loader.py                # Load CODI model + checkpoint
│   ├── activation_extractor.py       # Run CODI inference, extract per-thought activations
│   ├── cot_parser.py                 # Parse GSM8k-Aug CoT into structured steps
│   ├── thought_alignment.py          # Align 7 thoughts → CoT steps via lm_head decoding
│   ├── qa_templates.py               # 190+ question paraphrases across 6 categories
│   ├── qa_generator.py               # Generate QA pairs from activations + parsed CoT
│   ├── ao_dataset.py                 # Build tokenized training examples with injection positions
│   ├── steering.py                   # Norm-matched activation injection hook
│   ├── ao_trainer.py                 # AO training loop (LoRA + AdamW + WandB)
│   ├── ao_eval.py                    # Evaluation: exact match, BLEU, error localization
│   └── utils.py                      # Shared helpers
├── scripts/                          # Pipeline entry points
│   ├── 01_download_data.py           # Download GSM8k-Aug + CODI checkpoints
│   ├── 02_extract_activations.py     # Extract activations from CODI
│   ├── 03_generate_qa.py             # Build QA dataset
│   ├── 04_train.py                   # Train the Activation Oracle
│   ├── 05_eval.py                    # Run evaluation suite
│   └── run_thin_pipeline.sh          # One-command end-to-end runner
├── configs/
│   ├── thin.yaml                     # 100 problems, ~570 QA examples (validation)
│   └── full.yaml                     # 25k problems, ~120k QA examples (production)
├── codi/                             # CODI submodule (github.com/zhenyi4/codi)
├── activation_oracles/               # AO submodule (github.com/adamkarvonen/activation_oracles)
├── plan.md                           # Original research plan
├── implementation_plan.md            # Detailed implementation plan
└── progress.md                       # Development log
```

---

## Setup

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (A100/H100 recommended)
- ~10 GB disk for checkpoints and data

### Installation

```bash
git clone --recurse-submodules https://github.com/<your-org>/codi-ao.git
cd codi-ao

# Install dependencies
uv sync
source .venv/bin/activate

# Login to HuggingFace (needed for model downloads)
huggingface-cli login --token <your_token>
```

---

## Usage

### Thin Pipeline (Recommended First Run)

The thin pipeline validates the entire end-to-end flow with 100 problems (~570 QA examples). Expected runtime: ~35 minutes on A100.

```bash
bash scripts/run_thin_pipeline.sh
```

This runs all 5 stages:
1. **Download** — GSM8k-Aug dataset + CODI-llama checkpoint from HuggingFace
2. **Extract** — Run CODI on 100 problems, capture thought activations at layers 4/8/12
3. **Generate QA** — Build ~570 training examples across all 6 categories
4. **Train** — Train AO with LoRA for 3 epochs (batch_size=4, grad_accum=4)
5. **Evaluate** — Score per-category accuracy, save results to JSON

Output: `checkpoints/ao_thin/final` (model) + `results/thin/eval_results.json` (metrics)

### Full Pipeline

After the thin pipeline validates, scale to 25k problems:

```bash
# Edit scripts to use full config, or run each step manually:
python scripts/01_download_data.py
python scripts/02_extract_activations.py --config configs/full.yaml
python scripts/03_generate_qa.py --config configs/full.yaml
python scripts/04_train.py --config configs/full.yaml
python scripts/05_eval.py --config configs/full.yaml
```

### Individual Steps

Each script accepts a `--config` flag for YAML configuration or direct CLI arguments:

```bash
# Extract activations with custom settings
python scripts/02_extract_activations.py --config configs/thin.yaml

# Train with CLI overrides
python scripts/04_train.py --model-name meta-llama/Llama-3.2-1B-Instruct \
    --train-data data/qa_datasets/train.pt \
    --eval-data data/qa_datasets/eval.pt \
    --epochs 3 --lr 1e-5

# Evaluate a specific checkpoint
python scripts/05_eval.py --checkpoint checkpoints/ao/final \
    --eval-data data/qa_datasets/eval.pt
```

---

## Technical Details

### Activation Extraction

For each problem, CODI's inference loop produces 7 thoughts (initial encoding + 6 latent iterations). At each thought we extract:

- **Pre-projection hidden states** at layers 4, 8, 12 — these live in standard LLaMA space and are what the AO receives
- **Post-projection vectors** — CODI's internal representation (stored but not used for AO training by default)
- **Decoded top-5 tokens** — via `lm_head`, used for thought-to-step alignment

Storage: ~114 KB/problem (bf16). 100 problems = ~11 MB, 25k problems = ~2.8 GB.

### Thought Alignment

CODI uses 6 latent iterations regardless of problem complexity. Alignment maps each thought to a CoT step by matching the decoded top-1 token against known intermediate results. Non-matching thoughts are marked "transitional." CODI reports 97.1% alignment accuracy for 1-step problems, 75% for 3-step.

### Training

- **Base model**: Fresh LLaMA-3.2-1B-Instruct (same architecture as CODI, fresh weights)
- **LoRA**: rank=64, alpha=128, dropout=0.05, all linear layers
- **Optimizer**: AdamW with linear warmup (10%) + linear decay
- **Injection**: Layer 1 of AO, norm-matched additive steering (coefficient=1.0)
- **Precision**: bfloat16

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Activation type | Pre-projection hidden states | Standard LLaMA space; AO has "privileged access" to own architecture |
| Extraction layers | 4, 8, 12 (25/50/75% depth) | Follows AO paper convention |
| Injection layer | Layer 1 of AO | Follows AO paper default |
| AO base model | Fresh LLaMA-3.2-1B | Same architecture as CODI; needs intact language modeling |
| Thought count | 7 (initial + 6 iterations) | Thought 0 = model's "first impression"; more data is better |
| Pre-computed activations | All vectors stored in .pt files | CODI's iterative latent loop can't be run on-the-fly during AO training |

---

## Success Criteria

### Thin Pipeline (Validation)
- No crashes or shape mismatches through the full pipeline
- Training loss decreases over epochs
- AO produces coherent text (not garbage) when given injected activations
- Format correctness >30% on binary questions

### Full Pipeline
- **Minimum**: AO recovers intermediate results ≥75% on 3-step problems (matching CODI's vocab projection baseline)
- **Good**: Faithful NL descriptions of full reasoning chains + OOD generalization to SVAMP/MultiArith
- **Excellent**: Error localization works — AO identifies which thought diverged from ground truth on problems CODI gets wrong

---

## Planned Experiments

1. **Ablation runs** — Train on category subsets (Cat 1+2 only, Cat 1-4, full) to measure diversity contribution
2. **Learning rate sweep** — [1e-6, 3e-6, 1e-5, 3e-5]
3. **Layer comparison** — Compare AO accuracy when given layer 4 vs 8 vs 12 activations
4. **OOD evaluation** — Test on SVAMP, GSM-Hard, MultiArith without additional training
5. **Error localization** — On wrong-answer problems, query each thought position and identify the divergence point

---

## References

- **CODI**: Zhen et al., [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074) (EMNLP 2025)
- **Activation Oracles**: Karvonen et al., [Activation Oracles](https://arxiv.org/abs/2512.15674)
- **GSM8k-Aug**: Augmented GSM8k dataset on HuggingFace ([zen-E/GSM8k-Aug](https://huggingface.co/datasets/zen-E/GSM8k-Aug))
- **CODI Checkpoints**: [zen-E/CODI-llama3.2-1b-Instruct](https://huggingface.co/zen-E/CODI-llama3.2-1b-Instruct), [zen-E/CODI-gpt2](https://huggingface.co/zen-E/CODI-gpt2)
