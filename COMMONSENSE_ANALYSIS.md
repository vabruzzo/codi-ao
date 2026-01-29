# CommonsenseQA Latent Interpretability Analysis

## Overview

This document details our systematic analysis of CODI's latent vector interpretability on CommonsenseQA, extending the math-focused analysis from the LessWrong blog post to non-mathematical reasoning tasks.

**Key Question**: Can we interpret what CODI "thinks" during commonsense reasoning the same way we can for math?

**TL;DR**: Logit Lens can predict CODI's output behavior (99% in top-5, 70% at z3 top-1), but the latent representations are noisier and less structured than math. The model encodes answer choices alongside reasoning fragments, not clean intermediate results.

---

## Background

### The LessWrong Math Analysis

The original LessWrong analysis of CODI found remarkably clean interpretability for math:

| Position | Math Encoding | Accuracy |
|----------|---------------|----------|
| z2 | Step 1 intermediate result (e.g., "6") | 100% |
| z4 | Step 2 result / final answer (e.g., "24") | 85% |
| z5-z6 | `<\|eocot\|>` (end of chain-of-thought) | - |

This worked because:
1. Math problems have discrete numeric intermediate values
2. Small numbers tokenize as single tokens
3. The template problems were specifically designed for this analysis

### Our Question

CODI was also trained on CommonsenseQA. Does the same interpretability hold for non-math reasoning? CommonsenseQA is fundamentally different:
- Multiple choice (A, B, C, D, E) not numeric
- No clear "intermediate calculation" - just reasoning to a choice
- Concepts may be multi-token

---

## Methodology

### Dataset

We used **zen-E/CommonsenseQA-GPT4omini** - the exact dataset CODI was trained on - to ensure fair evaluation. We analyzed 200 validation examples.

### Prompt Format

Following CODI's test.py:
```
{question} Output only the answer and nothing else.
```

### Metrics Collected

1. **CODI Accuracy**: Does the model output the correct letter?
2. **Latent Step Usage**: Where does `<|eocot|>` first appear?
3. **Correct Answer in Latent**: Is the ground-truth letter in z2/z3's top-5?
4. **Predicted Answer in Latent**: Is the model's actual output in z2/z3's top-5?
5. **z2/z3 Top-1 Match**: Does the highest probability token match the output?
6. **Cross-tabulation**: Correlation between latent prediction and correctness
7. **Token Composition**: What types of tokens appear in top-5?

### Robust Answer Extraction

We implemented careful extraction to handle tokenization differences:
- Direct prompt substring removal
- Marker-based extraction ("nothing else.")
- Whitespace normalization
- Only counted exact letter matches (A-E), not substrings

---

## Results

### 1. Basic Performance

| Metric | Value |
|--------|-------|
| CODI Accuracy | 63.0% (126/200) |
| Latent steps used | 3 (eocot at z4 in 99.5% of cases) |

**Comparison to math**: Math uses 4-5 latent steps; CommonsenseQA uses only 3. The model "thinks less" on commonsense tasks.

### 2. Can Logit Lens Predict Model Behavior?

**This is the key interpretability question: Can we read what the model will output?**

| Metric | Value |
|--------|-------|
| Predicted letter in z2/z3 top-5 | **99.0%** |
| z2 top-1 matches output | 45.5% |
| z3 top-1 matches output | **70.0%** |

**Interpretation**: 
- **99%** of the time, we can find the model's answer in the latents
- z3 is the "decision point" - 70% of outputs match z3's top token
- z2 is still deliberating (only 45.5% match)

### 3. Does Correct Answer Appear in Latents?

| Metric | Value |
|--------|-------|
| Correct letter in z2/z3 top-5 | 90.5% |
| First appears in z2 | 51.5% |
| First appears in z3 | 38.0% |

The model "considers" the correct answer 90.5% of the time but only commits to it 63% of the time.

### 4. Correlation with Correctness

**Cross-tabulation of z2 top-1 prediction vs correctness:**

|  | z2 matches output | z2 doesn't match |
|---|---|---|
| **Correct** | 57 (28.5%) | 69 (34.5%) |
| **Incorrect** | 34 (17.0%) | 40 (20.0%) |

**Conditional probabilities:**
- P(correct | z2 matches output) = **62.6%**
- P(correct | z2 doesn't match) = **63.3%**
- Baseline accuracy = 63.0%

**These are essentially identical!** Whether z2 "commits" to an answer has **zero predictive value** for correctness. We can read what the model will say, but not whether it's right.

### 5. What's in the Top-5? (Token Composition)

When we look at z2's top-5 tokens (excluding the predicted letter):

| Token Type | Count | Percentage |
|------------|-------|------------|
| Non-letter tokens | 612 | **82%** |
| Competing letters (A-E) | 137 | 18% |

**Most common non-letter tokens:**
| Token | Count | Type |
|-------|-------|------|
| `"` | 79 | Punctuation |
| `.` | 49 | Punctuation |
| `typically` | 46 | Reasoning |
| `when` | 43 | Reasoning |
| `to` | 34 | Common word |
| `specifically` | 12 | Reasoning |
| `However` | 9 | Reasoning |

**Key insight**: The top-5 is NOT just "all five letters A-E". When the predicted letter appears, it stands out against reasoning tokens and punctuation. This means Logit Lens is genuinely reading the model's choice, not just seeing a uniform distribution.

---

## Comparison: Math vs CommonsenseQA

| Aspect | Math (Templates) | CommonsenseQA |
|--------|------------------|---------------|
| z2 encodes | Intermediate numeric result | Letter + reasoning fragments |
| z2 accuracy | 100% | 45.5% (top-1 match) |
| Latent steps | 4-5 (eocot at z5/z6) | 3 (eocot at z4) |
| Token type | Single numbers | Mix of letters + words |
| Interpretability | Clean, deterministic | Noisy, probabilistic |
| Predicts output | Yes | Yes (99% in top-5) |
| Predicts correctness | N/A (nearly 100% correct) | No (no correlation) |

---

## Key Findings

### 1. Logit Lens Works for Behavior Prediction
We can reliably predict what CODI will output (99% in top-5, 70% z3 top-1 match). The latents encode the model's decision.

### 2. No Correctness Signal
Whether the latent "commits" to an answer has no bearing on whether that answer is correct. The latents tell us what the model thinks, not whether it's right.

### 3. Fewer Reasoning Steps
CommonsenseQA uses only 3 latent steps vs 4-5 for math. The model "thinks less" - possibly because commonsense doesn't require multi-step calculation.

### 4. Mixed Token Encoding
Unlike math's clean numeric encoding, CommonsenseQA latents contain:
- Answer choice letters (A-E)
- Reasoning words ("typically", "when", "However")
- Punctuation

This suggests CODI is encoding reasoning fragments, not just choices.

### 5. z3 is the Decision Point
z3 top-1 matches output 70% of the time vs z2's 45.5%. The model finalizes its choice between z2 and z3.

---

## Limitations

1. **Dataset**: We tested on CODI's training data. Out-of-distribution performance may differ.

2. **Single-token bias**: Logit Lens only captures single tokens. Multi-token reasoning may be invisible.

3. **Correlation ≠ Causation**: We observe what tokens appear, not why.

4. **Model-specific**: These findings are for bcywinski/codi_llama1b-answer_only. Other CODI variants may differ.

---

## Implications for Activation Oracles

This analysis motivates Activation Oracle training:

1. **Logit Lens limitations**: The "reasoning fragments" in z2 (typically, when, However) suggest semantic content that Logit Lens can't fully decode. An AO might extract more meaningful interpretations.

2. **Multi-latent context**: Giving an AO all 6 latent vectors might reveal patterns invisible to single-position Logit Lens analysis.

3. **Beyond single tokens**: AOs generate full text responses, potentially capturing multi-token concepts that Logit Lens misses.

4. **Training signal exists**: The 99% "predicted in top-5" rate shows there IS interpretable signal in the latents - the question is whether AOs can decode it better than Logit Lens.

---

## Reproducibility

### Commands
```bash
# Run the full analysis
uv run python scripts/analyze_commonsense_latents.py --n_examples 200 --use_real_data

# With verbose per-example output
uv run python scripts/analyze_commonsense_latents.py --n_examples 200 --use_real_data --verbose
```

### Output Files
- `reports/commonsense_latent_analysis.json` - Full metrics in JSON format

### Code
- `scripts/analyze_commonsense_latents.py` - Main analysis script
- `scripts/test_commonsense.py` - Quick manual inspection tool

---

## Conclusion

CODI's latent representations on CommonsenseQA are **interpretable but noisy**. Unlike math's clean numeric encoding, commonsense latents contain a mix of answer choices and reasoning fragments. We can predict the model's output (99% accuracy) but not its correctness (no correlation).

This suggests that while CODI does "reason" in its latent space for commonsense tasks, this reasoning is less structured than mathematical calculation. Activation Oracles may provide a path to deeper interpretation by decoding the semantic content that Logit Lens can only partially reveal.
