Activation Oracle for CODI Latent Reasoning
Project Summary
Train an Activation Oracle (AO) to interpret CODI's continuous thought representations using natural language question-answering. CODI compresses chain-of-thought reasoning into six continuous latent vectors. These vectors carry structured reasoning information (verified by CODI's own interpretability analysis) but can only be inspected today by projecting them back into vocabulary space — a lossy, inflexible method. An AO trained on these representations would allow arbitrary natural language queries about the latent reasoning process.
Why This Works
CODI's continuous thoughts are unusually well-suited for AO training compared to arbitrary model activations:

The information is guaranteed to be there. The distillation loss explicitly trains the continuous thoughts to encode the same reasoning information as the explicit CoT. Unlike PersonaQA (where fine-tuned knowledge was brittle and barely above chance on reformulated questions), CODI's latent representations were optimized to carry this information.
Ground truth is free. Every training problem already has paired explicit CoT. You don't need to generate labels — the CoT annotations from GSM8k-Aug are the labels.
The AO paper's hardest setting maps directly to this. The AO paper's strongest results came from tasks where information exists in activations but is absent from the input text. That's exactly the CODI student path: the input is just the question, the continuous thoughts contain the reasoning, and the answer follows. The CoT tokens never appear in the student's text stream.


Phase 1: Setup and Verification
Duration: 1-2 days
1.1 Download Pretrained Models
Grab the CODI checkpoints from HuggingFace:

zen-E/CODI-llama3.2-1b-Instruct (primary — larger model, better activation richness)
zen-E/CODI-gpt2 (secondary — useful for fast iteration and debugging)

Clone the CODI repo from github.com/zhenyi4/codi.
1.2 Reproduce Reported Results
Run their evaluation scripts to confirm:

LLaMA-1B on GSM8k: expect 55.6%
LLaMA-1B on SVAMP: expect 61.1%
LLaMA-1B on MultiArith: expect 96.1%

If numbers match, the checkpoint is good. If they don't, debug before proceeding.
1.3 Reproduce Interpretability Analysis
Run probe_latent_token.py on a sample of problems. Verify that:

Decoded continuous thoughts show recognizable intermediate results
The pattern of "meaningful token, filler token, meaningful token" described in the paper is visible
The attention analysis shows thoughts attending to relevant operands

This gives you a qualitative baseline for what "interpretability" looks like without an AO.
1.4 Understand the Activation Geometry
Before collecting data, characterize the continuous thought representations:

Dimensionality (LLaMA-1B hidden dim is 2048)
Norm distribution across the six thought positions
Norm distribution across layers
Whether the projected (post-MLP) or pre-projection representations are more informative
Variance across problems of different complexity

This informs decisions about what to feed the AO and whether any normalization is needed.

Phase 2: Activation Collection
Duration: Half a day
2.1 What to Collect
For each problem, run the CODI student path and save:
Per continuous thought token (6 per problem):

Hidden states from layers at 25%, 50%, 75% depth (following AO paper convention)
The post-MLP-projection vector (CODI-specific representation)
The decoded top-5 vocabulary tokens (for ground truth alignment)
Attention weights over the input (for ground truth alignment)

Per problem:

Hidden state at the answer-generating token (the colon in "The answer is:") from all three layer depths
The model's predicted answer
Whether the prediction was correct

Metadata:

The original question text
The ground truth CoT string
The parsed CoT steps (operands, operators, intermediate results)
The ground truth final answer
Number of reasoning steps required

2.2 Data Sources
Primary: GSM8k-Aug training set — Sample 25k problems from the full 385k. This is more than enough for a 1B model with LoRA training. Ensure the sample is stratified by number of CoT steps (1-step through 5+ step problems in proportion) to avoid biasing the AO toward easy problems.
Parse each CoT into structured step records:
problem_id: 12345
question: "Out of 600 employees..."
cot_raw: "<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>"
cot_parsed:
  - step: 1, operands: [600, 30, 100], operation: "multiply_divide", result: 180
  - step: 2, operands: [600, 10, 100], operation: "multiply_divide", result: 60
  - step: 3, operands: [180, 60], operation: "add", result: 240
  - step: 4, operands: [600, 240], operation: "subtract", result: 360
answer: 360
num_steps: 4
Keep an additional 5k problems as a scaling reserve — if validation loss is still dropping at the end of training, generate more QA pairs from these and continue.
Secondary: GSM8k-Aug-NL — Same problems with natural language CoT. Useful for a later extension testing whether the AO can describe reasoning in natural language, not just recover numbers.
Held out entirely: GSM8k test set (1319 problems), SVAMP, GSM-Hard, MultiArith — these are for final evaluation only. Collect activations from these but never use them for AO training.
2.3 Alignment Between Thoughts and Steps
A critical preprocessing step. CODI uses 6 continuous thoughts for all problems regardless of complexity. A 2-step problem and a 5-step problem both produce 6 thoughts. You need to figure out which thoughts correspond to which steps.
Use CODI's own decoded tokens as the alignment signal:

For each thought, check if the top-1 decoded token is a number
If it matches an intermediate result from the parsed CoT, mark that alignment
Thoughts that decode to punctuation/filler are marked as "transitional"

This doesn't need to be perfect — it's used for generating training labels, not as the evaluation metric. The CODI paper reports 97.1% alignment accuracy for single-step problems, dropping to 75% for 3-step problems. That's good enough for training data.
2.4 Storage Estimates
Per problem: ~6 thoughts × 3 layers × 2048 floats × 4 bytes = ~147 KB
For 25k problems: ~3.6 GB
This fits comfortably in memory. Store as numpy arrays or a single HDF5 file.

Phase 3: Build the AO Training Dataset
Duration: 1-2 days (scripting and iteration)
Generate approximately 120k question-answer pairs across six categories from the 25k collected problems. For every category, generate 10-15 paraphrases of each question template to avoid the AO overfitting to specific phrasings. This is roughly 5 QA pairs per problem on average, with some problems generating more (multi-category) and some less.
Category 1: Intermediate Result Recovery (~25k examples)
Input: Single continuous thought activation (from thoughts aligned to a known step)
Question templates:

"What numerical result does this computation produce?"
"What is the intermediate value at this reasoning step?"
"What number does this calculation yield?"
Target: The intermediate result as a string (e.g., "180")

Also generate multi-thought variants:
Input: All six thought activations
Question: "What are the intermediate results in order?"
Target: "180, 60, 240, 360"
Category 2: Operation Classification (~30k examples)
Input: Single continuous thought activation
Binary question templates:

"Does this step involve multiplication?" → Yes/No
"Does this step involve addition?" → Yes/No
"Is the result of this step greater than 100?" → Yes/No
"Does this step use a result from a previous step?" → Yes/No
"Is this the first computation in the reasoning chain?" → Yes/No
"Does this step involve more than two operands?" → Yes/No

Ensure balanced yes/no distribution by construction. For each question type, sample equal numbers of positive and negative examples.
Category 3: Full Reasoning Description (~15k examples)
Input: All six thought activations
Question templates:

"Describe the reasoning steps used to solve this problem."
"What sequence of calculations leads to the answer?"
"Explain the mathematical reasoning encoded here."
"Walk through the computation step by step."
Target: The full CoT string, or a natural language summary generated from the parsed steps.

For the natural language variant, programmatically generate descriptions like: "First, multiply 600 by 30 and divide by 100 to get 180. Then multiply 600 by 10 and divide by 100 to get 60. Add these to get 240. Subtract from 600 to get 360."
Category 4: Problem-Level Property Questions (~15k examples)
Input: All six thought activations
Question templates:

"How many arithmetic steps are involved?" → "4"
"What is the final answer?" → "360"
"Does this problem require subtraction?" → Yes/No
"Are any intermediate results negative?" → Yes/No
"Is the final answer larger than 1000?" → Yes/No
"Does this problem involve computing a percentage?" → Yes/No

These are derivable from the parsed CoT steps with simple programmatic rules.
Category 5: Context Prediction (~25k examples)
Adapted from the AO paper's self-supervised task. Given a subset of thought activations, predict what's missing:

Given thoughts 1-3, predict what the later intermediate results are
Given thoughts 4-6, predict what the earlier intermediate results were
Given all thoughts, predict what the original question was about (topic keywords)
Given all thoughts, predict the final answer

Vary the number of input thoughts (1 to 6) and which positions are included.
Category 6: Thought Informativeness (~10k examples)
Input: Single continuous thought activation
Question templates:

"Is this thought encoding a meaningful intermediate result?" → Yes/No
"Is this a computational step or a transitional state?" → "computational" / "transitional"
"Does this thought carry significant reasoning information?" → Yes/No

Ground truth from the decoded token analysis: if the top-1 decoded token is a number matching a CoT step, it's meaningful. If it decodes to punctuation or filler, it's transitional.
3.1 Layer Variation
Across all categories, draw activations from layers at 25%, 50%, and 75% depth in roughly equal proportion, following the AO paper. This teaches the AO to work with activations from different layers.
3.2 Input Format Variation
Mix single-thought and multi-thought inputs:

Categories 1, 2, 6: primarily single-thought (with some multi-thought variants)
Categories 3, 4, 5: primarily multi-thought (all six)
Include some examples with 2-4 thoughts to teach the AO about partial input

3.3 Train/Validation Split

22k problems for AO training → ~105k QA pairs
3k problems for AO validation → ~15k QA pairs (hyperparameter selection, early stopping)
5k reserve problems for scaling up if needed
GSM8k test set, SVAMP, GSM-Hard, MultiArith held out entirely

If the initial 120k examples prove insufficient (validation loss still dropping, accuracy clearly improving with more data), generate additional QA pairs from the reserve and scale up to ~200k. The fast training cycle makes this cheap to test.

Phase 4: Train the Activation Oracle
Duration: 1-2 days (including ablation runs)
4.1 Architecture
Use LLaMA-3.2-1B-Instruct as the AO base model — the same architecture as CODI's base. This follows the AO paper's finding that models have privileged access to their own internals.
Important distinction: The AO is a fresh LLaMA-3.2-1B-Instruct, NOT the CODI checkpoint. The AO needs its standard language modeling capabilities intact. CODI's weights have been modified by the distillation training and may not produce coherent natural language for the QA task.
Train with LoRA:

Rank: 64
Alpha: 128
Dropout: 0.05
Target modules: all linear layers
Learning rate: sweep [1e-6, 3e-6, 1e-5, 3e-5]

4.2 Injection Mechanism
Follow the AO paper's approach:

Use placeholder tokens (e.g., " ?") for each injected activation
After transformer layer 1, add the activation vector to the residual stream at placeholder positions
Norm-match: scale the injected vector to match the norm of the existing activation at that position

Oracle prompt format:
Layer {layer_num}: <ACT> <ACT> <ACT> <ACT> <ACT> <ACT> What are the intermediate results?
For single-thought questions:
Layer {layer_num}: <ACT> Does this step involve multiplication?
4.3 Key Implementation Decision: Which Activations to Inject
This needs experimentation. Three options:
Option A: Post-MLP projection vectors. These are the actual continuous thought representations that CODI uses. They're CODI-specific — the MLP projection transforms them into a space optimized for CODI's reasoning. The AO would need to learn this space from scratch.
Option B: Pre-projection hidden states. The residual stream activations before CODI's MLP projection. These live in the standard LLaMA activation space, which the AO (being based on LLaMA) might find easier to interpret. But they may contain less reasoning-specific information.
Option C: Both. Train on a mix, using a flag in the prompt (e.g., "Projected:" vs "Hidden:") to distinguish.
Start with Option B (pre-projection hidden states) since it's closest to what the AO paper does. Test Option A if results are underwhelming.
4.4 Training

Batch size: 16 (or 64 if GPU memory allows)
One epoch over the 120k examples (scale to 2-3 epochs if validation loss is still dropping)
AdamW optimizer with linear warmup (10% of steps) and linear decay
Mixed precision (bfloat16)
Generate activations on-the-fly during training to save storage, OR load from pre-computed files

Estimated compute: ~2-3 hours per run on a single A100. This means you can do 5-6 experimental runs in a single day, which is valuable for iterating on dataset composition and hyperparameters.
If validation loss plateaus but accuracy on held-out problems is still low, that's a signal to generate more data from the 5k reserve problems rather than training longer on the same data.
4.5 Ablation Training Runs
Following the AO paper's methodology, train several variants to measure the contribution of each dataset category. At ~2-3 hours per run, you can fit all three ablations plus the main run in a single day:

Categories 1+2 only (structured single-thought questions, ~55k examples)
Categories 1+2+3+4 (add multi-thought and open-ended, ~85k examples)
Full dataset (all six categories, ~120k examples)

This tells you whether diversity helps, matching the AO paper's finding that it consistently does.

Phase 5: Evaluation
Duration: 1-2 days
5.1 In-Distribution Evaluation
On held-out GSM8k-Aug problems:
Intermediate result recovery: Exact match accuracy. For each correctly-solved problem, inject individual thought activations and ask for the intermediate result. Compare to CODI's vocabulary projection baseline (their Table 4 reports 97.1% for 1-step, 83.9% for 2-step, 75.0% for 3-step).
Operation classification: Accuracy on binary questions from held-out problems.
Full reasoning description: BLEU/ROUGE against ground truth CoT. Also use an LLM judge to assess whether the AO's description is faithful to the actual reasoning.
Answer prediction: Given all six thought activations, can the AO predict the final answer? Exact match accuracy.
5.2 Comparison to CODI's Built-In Interpretability
For a sample of 200-500 problems, collect:

CODI's decoded top-5 tokens per thought position (the existing method)
AO's response to "Describe this reasoning step" per thought position

Have an LLM judge rate which provides more faithful, complete, and useful information about the reasoning. The AO should win on completeness (it can give natural language descriptions) while CODI's projection might win on precision for simple cases (it directly shows the intermediate number).
5.3 Out-of-Distribution Evaluation
Run CODI on SVAMP, GSM-Hard, and MultiArith. Collect activations. Ask the AO the same questions without any additional training. This tests whether the AO generalizes to problems with different structure, difficulty, and number ranges.
Key metric: Does AO accuracy degrade gracefully on OOD problems, or does it collapse? The AO paper found that diversely-trained AOs generalize well to OOD settings.
5.4 Error Localization (the high-value test)
This is the most practically interesting evaluation. For problems where CODI gets the wrong answer:

Collect the six thought activations
Ask the AO "What intermediate result does this thought compute?" for each position
Compare the AO's claimed intermediate results to the ground truth CoT steps
Can the AO identify which step went wrong?

If this works, you've built a debugger for latent reasoning — something that doesn't exist today. Even partial success here would be a meaningful contribution.
5.5 GSM8k-Aug-NL Extension
If the main results look good, repeat phases 2-5 using CODI trained on GSM8k-Aug-NL (the natural language CoT variant). This tests whether the AO can handle latent representations of more complex, verbose reasoning chains. The compression ratio is much higher (7.8× vs 3.1×) so there's more information packed into the same six thoughts.
5.6 Commonsense Extension
The CODI repo includes commonsense reasoning training scripts. If time permits, train a CODI model on commonsense data (this requires actual CODI training since no pretrained checkpoint is available for this) and test whether the AO trained on math activations transfers to commonsense reasoning activations. This is a strong test of generalization — the AO paper showed that diverse training enables cross-task transfer.

Compute Budget
PhaseTaskEstimated TimeHardware1Verification2-4 hours1x A1002Activation collection (25k problems)1-2 hours1x A1003Dataset constructionCPU onlyAny machine4AO training (main)2-3 hours1x A1004AO training (ablations, ~3 runs)6-9 hours1x A1005Evaluation2-4 hours1x A100Total~15-22 GPU hours1x A100
Easily fits in a single day of A100 time for the core pipeline, with ablations spread over a second day. The fast iteration cycle means you can run an experiment in the morning, analyze results over lunch, and adjust for an afternoon run.

Risk Assessment
Low risk: Intermediate result recovery from individual thoughts. CODI's own interpretability analysis already shows this information is decodable. An AO should do at least as well as vocabulary projection.
Medium risk: Full reasoning description from all six thoughts. Requires the AO to synthesize across positions and produce coherent multi-step descriptions. May need prompt engineering or more training data in Category 3.
High risk: Error localization on wrong answers. The continuous thought representations for incorrectly-solved problems may be qualitatively different from correct ones (the reasoning went off-track, so the activations encode different information than any training example). Still worth attempting — even negative results here are informative.
Unknown risk: Whether LLaMA-1B scale is sufficient. The AO paper's smallest model was 9B parameters. At 1B, the hidden dimension is 2048 — there's meaningfully less capacity for the AO to work with. If results are weak across the board, the bottleneck might be model scale rather than the approach itself. The mitigation is to try the GPT-2 CODI checkpoint first (fast iteration) and only invest in the LLaMA-1B pipeline if GPT-2 shows promise.

Success Criteria
Minimum viable result: AO recovers intermediate results from individual thought activations with accuracy comparable to or better than CODI's vocabulary projection (75%+ on 3-step problems).
Good result: AO provides faithful natural language descriptions of the full reasoning chain and generalizes to OOD math benchmarks.
Excellent result: AO can localize errors in CODI's latent reasoning on problems it gets wrong, functioning as a debugger for continuous chain-of-thought.