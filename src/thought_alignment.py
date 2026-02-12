"""Align CODI's continuous thoughts to CoT steps using lm_head decoding.

Each CODI problem produces 7 thought vectors (initial + 6 iterations).
Each GSM8k-Aug problem has N CoT steps (1-5+).
This module maps thoughts to steps by checking if the decoded top-1 token
matches an intermediate result from the parsed CoT.
"""

from dataclasses import dataclass

from src.activation_extractor import ThoughtRecord
from src.cot_parser import CotStep
from src.utils import extract_number


@dataclass
class ThoughtAlignment:
    """Alignment between a single thought and a CoT step."""

    thought_idx: int
    matched_step_idx: int | None  # None if no match (transitional thought)
    confidence: float  # Probability of the top-1 decoded token
    is_transitional: bool  # True if thought doesn't map to a step
    decoded_token: str  # The top-1 decoded token string
    decoded_value: float | None  # Parsed numeric value, if applicable


def align_thoughts_to_steps(
    thoughts: list[ThoughtRecord],
    steps: list[CotStep],
    tolerance: float = 1.0,
) -> list[ThoughtAlignment]:
    """Align thoughts to CoT steps using decoded token matching.

    For each thought:
    1. Check if the decoded top-1 token is a number
    2. If it matches an intermediate result from parsed CoT, align to that step
    3. If no match, mark as transitional

    Args:
        thoughts: List of ThoughtRecords from activation extraction.
        steps: List of parsed CoT steps.
        tolerance: Numeric tolerance for matching (default 1.0 to handle rounding).

    Returns:
        List of ThoughtAlignment objects, one per thought.
    """
    # Build a mapping from result values to step indices
    result_to_step: list[tuple[float, int]] = [
        (step.result, step.step_idx) for step in steps
    ]

    alignments = []
    used_steps: set[int] = set()  # Track which steps have been matched

    for thought in thoughts:
        if not thought.decoded_top_k:
            alignments.append(ThoughtAlignment(
                thought_idx=thought.thought_idx,
                matched_step_idx=None,
                confidence=0.0,
                is_transitional=True,
                decoded_token="",
                decoded_value=None,
            ))
            continue

        top_token, top_prob = thought.decoded_top_k[0]
        decoded_value = extract_number(top_token)

        matched_step_idx = None
        if decoded_value is not None:
            # Try to match against step results (prefer unused steps)
            best_match = None
            best_distance = float("inf")

            for result_val, step_idx in result_to_step:
                distance = abs(decoded_value - result_val)
                if distance <= tolerance and distance < best_distance:
                    # Prefer unmatched steps
                    if step_idx not in used_steps or best_match is None:
                        best_match = step_idx
                        best_distance = distance

            if best_match is not None:
                matched_step_idx = best_match
                used_steps.add(best_match)

        alignments.append(ThoughtAlignment(
            thought_idx=thought.thought_idx,
            matched_step_idx=matched_step_idx,
            confidence=top_prob,
            is_transitional=(matched_step_idx is None),
            decoded_token=top_token,
            decoded_value=decoded_value,
        ))

    return alignments


def get_aligned_thoughts(
    alignments: list[ThoughtAlignment],
) -> list[ThoughtAlignment]:
    """Return only the thoughts that are aligned to a step (non-transitional)."""
    return [a for a in alignments if not a.is_transitional]


def get_transitional_thoughts(
    alignments: list[ThoughtAlignment],
) -> list[ThoughtAlignment]:
    """Return only transitional thoughts (not aligned to any step)."""
    return [a for a in alignments if a.is_transitional]


def alignment_summary(alignments: list[ThoughtAlignment]) -> dict:
    """Generate a summary of the alignment quality."""
    total = len(alignments)
    aligned = len(get_aligned_thoughts(alignments))
    transitional = len(get_transitional_thoughts(alignments))
    avg_confidence = (
        sum(a.confidence for a in alignments if not a.is_transitional) / max(aligned, 1)
    )

    return {
        "total_thoughts": total,
        "aligned": aligned,
        "transitional": transitional,
        "alignment_rate": aligned / max(total, 1),
        "avg_aligned_confidence": avg_confidence,
    }
