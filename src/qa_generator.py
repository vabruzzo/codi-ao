"""Generate QA pairs across all 6 categories from activation records.

Takes activation records + parsed CoT + alignment data and produces
QA pairs that will be converted to AO TrainingDataPoints.
"""

import random
from dataclasses import dataclass, field

from src.activation_extractor import ActivationRecord, ThoughtRecord
from src.cot_parser import (
    ParsedProblem,
    CotStep,
    describe_full_reasoning,
    has_operation,
    has_any_negative_result,
    involves_percentage,
)
from src.thought_alignment import ThoughtAlignment, get_aligned_thoughts, get_transitional_thoughts
from src.utils import format_number
from src import qa_templates


@dataclass
class QAPair:
    """A single QA pair for AO training."""

    problem_id: str
    category: int  # 1-6
    category_name: str
    question: str  # Natural language question
    answer: str  # Target response
    thought_indices: list[int]  # Which thoughts to inject (0-6)
    layer_source: int  # Which layer the activations come from
    activation_type: str = "pre_projection"  # or "post_projection"
    subtype: str = ""  # For categories with sub-types (e.g., "multiplication")


def _pick_template(templates: list[str]) -> str:
    """Randomly select a question template."""
    return random.choice(templates)


def _pick_layer(layers: list[int]) -> int:
    """Randomly select a source layer."""
    return random.choice(layers)


# =============================================================================
# Category 1: Intermediate Result Recovery
# =============================================================================

def generate_cat1(
    record: ActivationRecord,
    parsed: ParsedProblem,
    alignments: list[ThoughtAlignment],
    layers: list[int],
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate intermediate result recovery QA pairs."""
    pairs = []
    aligned = get_aligned_thoughts(alignments)

    # Single-thought: ask for individual intermediate results
    for alignment in aligned:
        if alignment.matched_step_idx is None:
            continue
        step = parsed.steps[alignment.matched_step_idx - 1]  # steps are 1-indexed
        qa = QAPair(
            problem_id=record.problem_id,
            category=1,
            category_name="intermediate_result",
            question=_pick_template(qa_templates.CAT1_SINGLE_THOUGHT),
            answer=format_number(step.result),
            thought_indices=[alignment.thought_idx],
            layer_source=_pick_layer(layers),
        )
        pairs.append(qa)

    # Multi-thought variant: all results in order
    if len(parsed.steps) >= 2:
        all_results = ", ".join(format_number(s.result) for s in parsed.steps)
        all_thought_indices = list(range(len(record.thoughts)))
        qa = QAPair(
            problem_id=record.problem_id,
            category=1,
            category_name="intermediate_result_all",
            question=_pick_template(qa_templates.CAT1_ALL_THOUGHTS),
            answer=all_results,
            thought_indices=all_thought_indices,
            layer_source=_pick_layer(layers),
        )
        pairs.append(qa)

    if max_examples and len(pairs) > max_examples:
        pairs = random.sample(pairs, max_examples)
    return pairs


# =============================================================================
# Category 2: Operation Classification
# =============================================================================

def _generate_cat2_subtype(
    record: ActivationRecord,
    parsed: ParsedProblem,
    alignments: list[ThoughtAlignment],
    layers: list[int],
    subtype: str,
    templates: list[str],
    ground_truth_fn,
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate binary classification pairs for a Cat2 sub-type."""
    pairs = []
    aligned = get_aligned_thoughts(alignments)

    for alignment in aligned:
        if alignment.matched_step_idx is None:
            continue
        step = parsed.steps[alignment.matched_step_idx - 1]
        answer = "Yes" if ground_truth_fn(step, alignment) else "No"
        qa = QAPair(
            problem_id=record.problem_id,
            category=2,
            category_name="operation_classification",
            question=_pick_template(templates),
            answer=answer,
            thought_indices=[alignment.thought_idx],
            layer_source=_pick_layer(layers),
            subtype=subtype,
        )
        pairs.append(qa)

    if max_examples and len(pairs) > max_examples:
        pairs = random.sample(pairs, max_examples)
    return pairs


def generate_cat2(
    record: ActivationRecord,
    parsed: ParsedProblem,
    alignments: list[ThoughtAlignment],
    layers: list[int],
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate operation classification QA pairs across all sub-types."""
    all_pairs = []

    sub_configs = [
        ("multiplication", qa_templates.CAT2_MULTIPLICATION,
         lambda s, a: has_operation(s, "multiply")),
        ("addition", qa_templates.CAT2_ADDITION,
         lambda s, a: has_operation(s, "add")),
        ("subtraction", qa_templates.CAT2_SUBTRACTION,
         lambda s, a: has_operation(s, "subtract")),
        ("division", qa_templates.CAT2_DIVISION,
         lambda s, a: has_operation(s, "divide")),
        ("result_gt_100", qa_templates.CAT2_RESULT_GT_100,
         lambda s, a: s.result > 100),
        ("uses_previous", qa_templates.CAT2_USES_PREVIOUS,
         lambda s, a: s.uses_previous_result),
        ("first_step", qa_templates.CAT2_FIRST_STEP,
         lambda s, a: s.step_idx == 1),
        ("multi_operand", qa_templates.CAT2_MULTI_OPERAND,
         lambda s, a: len(s.operands) > 2),
    ]

    for subtype, templates, gt_fn in sub_configs:
        pairs = _generate_cat2_subtype(
            record, parsed, alignments, layers, subtype, templates, gt_fn
        )
        all_pairs.extend(pairs)

    if max_examples and len(all_pairs) > max_examples:
        all_pairs = random.sample(all_pairs, max_examples)
    return all_pairs


# =============================================================================
# Category 3: Full Reasoning Description
# =============================================================================

def generate_cat3(
    record: ActivationRecord,
    parsed: ParsedProblem,
    layers: list[int],
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate full reasoning description QA pairs."""
    if not parsed.steps:
        return []

    description = describe_full_reasoning(parsed.steps)
    all_thought_indices = list(range(len(record.thoughts)))

    qa = QAPair(
        problem_id=record.problem_id,
        category=3,
        category_name="full_reasoning",
        question=_pick_template(qa_templates.CAT3_FULL_REASONING),
        answer=description,
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
    )
    return [qa]


# =============================================================================
# Category 4: Problem-Level Properties
# =============================================================================

def generate_cat4(
    record: ActivationRecord,
    parsed: ParsedProblem,
    layers: list[int],
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate problem-level property QA pairs."""
    if not parsed.steps:
        return []

    all_thought_indices = list(range(len(record.thoughts)))
    pairs = []

    # Num steps (open-ended)
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=4,
        category_name="problem_properties",
        question=_pick_template(qa_templates.CAT4_NUM_STEPS),
        answer=str(parsed.num_steps),
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="num_steps",
    ))

    # Final answer (open-ended)
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=4,
        category_name="problem_properties",
        question=_pick_template(qa_templates.CAT4_FINAL_ANSWER),
        answer=format_number(parsed.final_answer),
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="final_answer",
    ))

    # Binary: has subtraction
    has_sub = any(has_operation(s, "subtract") for s in parsed.steps)
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=4,
        category_name="problem_properties",
        question=_pick_template(qa_templates.CAT4_HAS_SUBTRACTION),
        answer="Yes" if has_sub else "No",
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="has_subtraction",
    ))

    # Binary: negative intermediate
    has_neg = has_any_negative_result(parsed.steps)
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=4,
        category_name="problem_properties",
        question=_pick_template(qa_templates.CAT4_NEGATIVE_INTERMEDIATE),
        answer="Yes" if has_neg else "No",
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="negative_intermediate",
    ))

    # Binary: answer > 1000
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=4,
        category_name="problem_properties",
        question=_pick_template(qa_templates.CAT4_ANSWER_GT_1000),
        answer="Yes" if parsed.final_answer > 1000 else "No",
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="answer_gt_1000",
    ))

    # Binary: has percentage
    has_pct = involves_percentage(parsed.steps)
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=4,
        category_name="problem_properties",
        question=_pick_template(qa_templates.CAT4_HAS_PERCENTAGE),
        answer="Yes" if has_pct else "No",
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="has_percentage",
    ))

    if max_examples and len(pairs) > max_examples:
        pairs = random.sample(pairs, max_examples)
    return pairs


# =============================================================================
# Category 5: Context Prediction
# =============================================================================

def generate_cat5(
    record: ActivationRecord,
    parsed: ParsedProblem,
    layers: list[int],
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate context prediction QA pairs."""
    if len(parsed.steps) < 2:
        return []

    all_thought_indices = list(range(len(record.thoughts)))
    num_thoughts = len(record.thoughts)
    pairs = []

    # Given first half of thoughts, predict later results
    if num_thoughts >= 4:
        mid = num_thoughts // 2
        early_thoughts = list(range(mid))
        later_results = ", ".join(
            format_number(s.result) for s in parsed.steps[len(parsed.steps) // 2:]
        )
        if later_results:
            pairs.append(QAPair(
                problem_id=record.problem_id,
                category=5,
                category_name="context_prediction",
                question=_pick_template(qa_templates.CAT5_PREDICT_LATER),
                answer=later_results,
                thought_indices=early_thoughts,
                layer_source=_pick_layer(layers),
                subtype="predict_later",
            ))

    # Given second half, predict earlier results
    if num_thoughts >= 4:
        mid = num_thoughts // 2
        late_thoughts = list(range(mid, num_thoughts))
        earlier_results = ", ".join(
            format_number(s.result) for s in parsed.steps[:len(parsed.steps) // 2]
        )
        if earlier_results:
            pairs.append(QAPair(
                problem_id=record.problem_id,
                category=5,
                category_name="context_prediction",
                question=_pick_template(qa_templates.CAT5_PREDICT_EARLIER),
                answer=earlier_results,
                thought_indices=late_thoughts,
                layer_source=_pick_layer(layers),
                subtype="predict_earlier",
            ))

    # Given all thoughts, predict final answer
    pairs.append(QAPair(
        problem_id=record.problem_id,
        category=5,
        category_name="context_prediction",
        question=_pick_template(qa_templates.CAT5_PREDICT_ANSWER),
        answer=format_number(parsed.final_answer),
        thought_indices=all_thought_indices,
        layer_source=_pick_layer(layers),
        subtype="predict_answer",
    ))

    if max_examples and len(pairs) > max_examples:
        pairs = random.sample(pairs, max_examples)
    return pairs


# =============================================================================
# Category 6: Thought Informativeness
# =============================================================================

def generate_cat6(
    record: ActivationRecord,
    alignments: list[ThoughtAlignment],
    layers: list[int],
    max_examples: int | None = None,
) -> list[QAPair]:
    """Generate thought informativeness QA pairs."""
    pairs = []

    for alignment in alignments:
        # Binary: meaningful or not
        is_meaningful = not alignment.is_transitional
        pairs.append(QAPair(
            problem_id=record.problem_id,
            category=6,
            category_name="thought_informativeness",
            question=_pick_template(qa_templates.CAT6_MEANINGFUL),
            answer="Yes" if is_meaningful else "No",
            thought_indices=[alignment.thought_idx],
            layer_source=_pick_layer(layers),
            subtype="meaningful",
        ))

        # Categorical: computational vs transitional
        pairs.append(QAPair(
            problem_id=record.problem_id,
            category=6,
            category_name="thought_informativeness",
            question=_pick_template(qa_templates.CAT6_COMPUTATIONAL_VS_TRANSITIONAL),
            answer="computational" if is_meaningful else "transitional",
            thought_indices=[alignment.thought_idx],
            layer_source=_pick_layer(layers),
            subtype="comp_vs_trans",
        ))

    if max_examples and len(pairs) > max_examples:
        pairs = random.sample(pairs, max_examples)
    return pairs


# =============================================================================
# Master Generator
# =============================================================================

def generate_all_qa_pairs(
    record: ActivationRecord,
    parsed: ParsedProblem,
    alignments: list[ThoughtAlignment],
    layers: list[int] = (4, 8, 12),
) -> list[QAPair]:
    """Generate all QA pairs for a single problem across all 6 categories.

    Args:
        record: Activation record for the problem.
        parsed: Parsed CoT for the problem.
        alignments: Thought-to-step alignments.
        layers: Available source layers for activation extraction.

    Returns:
        List of all QA pairs across all categories.
    """
    all_pairs = []
    all_pairs.extend(generate_cat1(record, parsed, alignments, layers))
    all_pairs.extend(generate_cat2(record, parsed, alignments, layers))
    all_pairs.extend(generate_cat3(record, parsed, layers))
    all_pairs.extend(generate_cat4(record, parsed, layers))
    all_pairs.extend(generate_cat5(record, parsed, layers))
    all_pairs.extend(generate_cat6(record, alignments, layers))
    return all_pairs


def balance_binary_pairs(pairs: list[QAPair]) -> list[QAPair]:
    """Balance Yes/No distribution for binary classification pairs.

    For each (category, subtype) with Yes/No answers, downsample the
    majority class to match the minority class count.
    """
    from collections import defaultdict

    # Group by (category, subtype)
    groups: dict[tuple, dict[str, list[QAPair]]] = defaultdict(lambda: defaultdict(list))
    non_binary = []

    for pair in pairs:
        if pair.answer in ("Yes", "No"):
            key = (pair.category, pair.subtype)
            groups[key][pair.answer].append(pair)
        else:
            non_binary.append(pair)

    balanced = list(non_binary)
    for key, answer_groups in groups.items():
        yes_pairs = answer_groups.get("Yes", [])
        no_pairs = answer_groups.get("No", [])
        min_count = min(len(yes_pairs), len(no_pairs))
        if min_count > 0:
            balanced.extend(random.sample(yes_pairs, min_count))
            balanced.extend(random.sample(no_pairs, min_count))
        else:
            # If one side is empty, keep the other (unbalanced but informative)
            balanced.extend(yes_pairs)
            balanced.extend(no_pairs)

    return balanced
