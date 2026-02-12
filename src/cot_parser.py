"""Parse GSM8k-Aug chain-of-thought strings into structured step records.

GSM8k-Aug uses the format: "<<expr1=result1>> <<expr2=result2>> ..."
Example: "<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>"
"""

import re
from dataclasses import dataclass, field


@dataclass
class CotStep:
    """A single computation step extracted from a CoT string."""

    step_idx: int  # 1-indexed
    expression: str  # e.g., "600*30/100=180"
    operands: list[float] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)  # e.g., ["multiply", "divide"]
    result: float = 0.0
    uses_previous_result: bool = False  # True if any operand matches a prior step's result


@dataclass
class ParsedProblem:
    """Fully parsed problem with structured CoT steps."""

    problem_id: str
    question: str
    cot_raw: str
    steps: list[CotStep] = field(default_factory=list)
    final_answer: float = 0.0
    num_steps: int = 0


def _classify_operation(char: str) -> str:
    """Map an operator character to a readable operation name."""
    mapping = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "/": "divide",
    }
    return mapping.get(char, "unknown")


def _extract_operands_and_operations(expr_lhs: str) -> tuple[list[float], list[str]]:
    """Extract operands and operations from the left-hand side of an expression.

    Examples:
        "600*30/100" -> ([600.0, 30.0, 100.0], ["multiply", "divide"])
        "180+60"     -> ([180.0, 60.0], ["add"])
        "600-240"    -> ([600.0, 240.0], ["subtract"])
    """
    # Split by operators while keeping them
    tokens = re.split(r"([+\-*/])", expr_lhs)
    operands = []
    operations = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in "+-*/":
            operations.append(_classify_operation(token))
        else:
            try:
                operands.append(float(token))
            except ValueError:
                continue

    return operands, operations


def parse_cot_string(cot_raw: str) -> list[CotStep]:
    """Parse a GSM8k-Aug CoT string into structured steps.

    Args:
        cot_raw: CoT string in format "<<expr1=result1>> <<expr2=result2>> ..."

    Returns:
        List of CotStep objects.
    """
    # Find all <<...>> blocks
    pattern = r"<<(.+?)>>"
    matches = re.findall(pattern, cot_raw)

    if not matches:
        return []

    steps = []
    previous_results: set[float] = set()

    for idx, expr in enumerate(matches, start=1):
        # Split on "=" to get LHS (expression) and RHS (result)
        parts = expr.rsplit("=", 1)
        if len(parts) != 2:
            continue

        lhs, rhs = parts[0].strip(), parts[1].strip()

        try:
            result = float(rhs)
        except ValueError:
            continue

        operands, operations = _extract_operands_and_operations(lhs)

        # Check if any operand matches a previous step's result
        uses_prev = any(
            abs(op - prev) < 1e-6
            for op in operands
            for prev in previous_results
        )

        step = CotStep(
            step_idx=idx,
            expression=expr,
            operands=operands,
            operations=operations,
            result=result,
            uses_previous_result=uses_prev,
        )
        steps.append(step)
        previous_results.add(result)

    return steps


def parse_problem(
    problem_id: str,
    question: str,
    cot_raw: str,
    answer: str | float,
) -> ParsedProblem:
    """Parse a full problem into structured form.

    Args:
        problem_id: Unique identifier for the problem.
        question: The math problem text.
        cot_raw: The raw CoT string from GSM8k-Aug.
        answer: The ground truth final answer.

    Returns:
        A ParsedProblem with structured step records.
    """
    steps = parse_cot_string(cot_raw)

    if isinstance(answer, str):
        answer = answer.replace(",", "")
        try:
            final_answer = float(answer)
        except ValueError:
            final_answer = 0.0
    else:
        final_answer = float(answer)

    return ParsedProblem(
        problem_id=str(problem_id),
        question=question,
        cot_raw=cot_raw,
        steps=steps,
        final_answer=final_answer,
        num_steps=len(steps),
    )


def has_operation(step: CotStep, operation: str) -> bool:
    """Check if a step involves a specific operation."""
    return operation in step.operations


def has_any_negative_result(steps: list[CotStep]) -> bool:
    """Check if any step produces a negative result."""
    return any(s.result < 0 for s in steps)


def involves_percentage(steps: list[CotStep]) -> bool:
    """Heuristic: check if the problem involves percentage computation.

    Looks for patterns like dividing by 100 or multiplying by a fraction of 100.
    """
    for step in steps:
        if 100.0 in step.operands and "divide" in step.operations:
            return True
        # Pattern: X * Y / 100
        if len(step.operands) >= 3 and 100.0 in step.operands:
            return True
    return False


def describe_step(step: CotStep) -> str:
    """Generate a natural language description of a single computation step."""
    if not step.operands or not step.operations:
        return f"compute {step.result}"

    from src.utils import format_number

    parts = [format_number(step.operands[0])]
    for op, operand in zip(step.operations, step.operands[1:]):
        verb = {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply by",
            "divide": "divide by",
        }.get(op, op)
        parts.append(f"{verb} {format_number(operand)}")

    return ", ".join(parts)


def describe_full_reasoning(steps: list[CotStep]) -> str:
    """Generate a natural language description of the full reasoning chain."""
    from src.utils import format_number

    if not steps:
        return "No computation steps found."

    ordinals = ["First", "Then", "Next", "After that", "Finally"]
    parts = []

    for i, step in enumerate(steps):
        prefix = ordinals[min(i, len(ordinals) - 1)]
        step_desc = describe_step(step)
        result_str = format_number(step.result)
        parts.append(f"{prefix}, {step_desc} to get {result_str}.")

    return " ".join(parts)
