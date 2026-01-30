#!/usr/bin/env python3
"""
Parse GSM8k dataset to extract intermediate calculation steps.

This script:
1. Loads GSM8k from HuggingFace
2. Parses solutions to extract intermediate values
3. Tracks number of steps per problem
4. Creates train/test split
5. Saves parsed data for AO training

GSM8k answer format:
  "Natalia sold 48/2 = <<48/2=24>>24 clips in May.
   Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
   #### 72"
  
We extract: [(expression, result), ...] from <<expr=result>> patterns
"""

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ParsedStep:
    """A single calculation step extracted from GSM8k solution."""
    expression: str      # e.g., "48/2"
    result: str          # e.g., "24"
    operation: str       # e.g., "division"
    step_number: int     # 1-indexed


@dataclass
class ParsedProblem:
    """A fully parsed GSM8k problem."""
    id: str
    question: str
    answer_text: str
    final_answer: str
    steps: list[dict]    # List of ParsedStep as dicts
    num_steps: int
    split: str           # "train" or "test"


def detect_operation(expression: str) -> str:
    """Detect the primary operation in an expression."""
    if "+" in expression:
        return "addition"
    elif "-" in expression:
        return "subtraction"
    elif "*" in expression or "×" in expression:
        return "multiplication"
    elif "/" in expression or "÷" in expression:
        return "division"
    else:
        return "unknown"


def parse_steps(answer_text: str) -> list[ParsedStep]:
    """
    Parse all calculation steps from GSM8k answer text.
    
    Looks for patterns like: <<48/2=24>>
    Returns list of ParsedStep objects.
    """
    # Pattern: <<expression=result>>
    pattern = r"<<([^=]+)=([^>]+)>>"
    matches = re.findall(pattern, answer_text)
    
    steps = []
    for i, (expr, result) in enumerate(matches, 1):
        # Clean up the result (remove commas, whitespace)
        result = result.strip().replace(",", "")
        expr = expr.strip()
        
        # Try to parse result as number
        try:
            # Handle decimals
            if "." in result:
                float(result)
            else:
                int(result)
        except ValueError:
            # Skip non-numeric results
            continue
        
        steps.append(ParsedStep(
            expression=expr,
            result=result,
            operation=detect_operation(expr),
            step_number=i,
        ))
    
    return steps


def parse_final_answer(answer_text: str) -> Optional[str]:
    """Extract final answer after #### marker."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        # Clean up: remove commas, convert to number
        answer = match.group(1).strip().replace(",", "")
        return answer
    return None


def load_and_parse_gsm8k(
    test_ratio: float = 0.2,
    seed: int = 42,
    min_steps: int = 1,
    max_steps: int = 10,
    only_two_step: bool = False,
) -> tuple[list[ParsedProblem], list[ParsedProblem]]:
    """
    Load GSM8k and parse all problems.
    
    Args:
        test_ratio: Fraction of data to hold out for testing
        seed: Random seed for reproducibility
        min_steps: Minimum number of steps to include
        max_steps: Maximum number of steps to include
        only_two_step: If True, only include problems with exactly 2 steps
    
    Returns:
        (train_problems, test_problems)
    """
    from datasets import load_dataset
    
    print("Loading GSM8k dataset from HuggingFace...")
    dataset = load_dataset("gsm8k", "main")
    
    # Combine train and test from HF for our own split
    all_items = []
    for split_name in ["train", "test"]:
        for i, item in enumerate(dataset[split_name]):
            all_items.append({
                "id": f"{split_name}_{i}",
                "question": item["question"],
                "answer": item["answer"],
            })
    
    print(f"Total problems: {len(all_items)}")
    
    # Parse all problems
    parsed = []
    step_counts = {}
    
    for item in all_items:
        steps = parse_steps(item["answer"])
        final_answer = parse_final_answer(item["answer"])
        
        if not steps or not final_answer:
            continue
        
        num_steps = len(steps)
        
        # Track ALL step counts for reporting
        step_counts[num_steps] = step_counts.get(num_steps, 0) + 1
        
        # Filter by step count
        if only_two_step:
            if num_steps != 2:
                continue
        else:
            if num_steps < min_steps or num_steps > max_steps:
                continue
        
        parsed.append(ParsedProblem(
            id=item["id"],
            question=item["question"],
            answer_text=item["answer"],
            final_answer=final_answer,
            steps=[asdict(s) for s in steps],
            num_steps=num_steps,
            split="",  # Will be set below
        ))
    
    print("\nStep count distribution (all parsed):")
    for n_steps in sorted(step_counts.keys()):
        marker = " ← selected" if (only_two_step and n_steps == 2) else ""
        print(f"  {n_steps} steps: {step_counts[n_steps]} problems{marker}")
    
    if only_two_step:
        print(f"\nFiltered to 2-step problems: {len(parsed)}")
    else:
        print(f"\nParsed {len(parsed)} problems with {min_steps}-{max_steps} steps")
    
    # Stratified split by step count
    random.seed(seed)
    
    # Group by step count
    by_steps = {}
    for p in parsed:
        if p.num_steps not in by_steps:
            by_steps[p.num_steps] = []
        by_steps[p.num_steps].append(p)
    
    train_problems = []
    test_problems = []
    
    for num_steps, problems in by_steps.items():
        random.shuffle(problems)
        n_test = max(1, int(len(problems) * test_ratio))
        
        for i, p in enumerate(problems):
            if i < n_test:
                p.split = "test"
                test_problems.append(p)
            else:
                p.split = "train"
                train_problems.append(p)
    
    print(f"\nSplit: {len(train_problems)} train, {len(test_problems)} test")
    
    return train_problems, test_problems


def save_parsed_data(
    train: list[ParsedProblem],
    test: list[ParsedProblem],
    output_dir: Path,
):
    """Save parsed data to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "gsm8k_train.json"
    test_path = output_dir / "gsm8k_test.json"
    stats_path = output_dir / "gsm8k_stats.json"
    
    # Convert to dicts
    train_data = [asdict(p) for p in train]
    test_data = [asdict(p) for p in test]
    
    # Compute stats
    def compute_stats(problems):
        step_dist = {}
        op_dist = {}
        for p in problems:
            step_dist[p.num_steps] = step_dist.get(p.num_steps, 0) + 1
            for s in p.steps:
                op = s["operation"]
                op_dist[op] = op_dist.get(op, 0) + 1
        return {
            "total": len(problems),
            "step_distribution": dict(sorted(step_dist.items())),
            "operation_distribution": op_dist,
        }
    
    stats = {
        "train": compute_stats(train),
        "test": compute_stats(test),
    }
    
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved train data to {train_path}")
    
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved test data to {test_path}")
    
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


def print_examples(problems: list[ParsedProblem], n: int = 3):
    """Print example parsed problems."""
    print(f"\n{'='*70}")
    print("EXAMPLE PARSED PROBLEMS")
    print(f"{'='*70}\n")
    
    for p in problems[:n]:
        print(f"ID: {p.id}")
        print(f"Question: {p.question[:100]}...")
        print(f"Final Answer: {p.final_answer}")
        print(f"Num Steps: {p.num_steps}")
        print("Steps:")
        for s in p.steps:
            print(f"  Step {s['step_number']}: {s['expression']} = {s['result']} ({s['operation']})")
        print("-" * 50)


def generate_qa_examples(problems: list[ParsedProblem]) -> list[dict]:
    """
    Generate QA training examples from parsed problems.
    
    For 2-step problems, we use:
    - z1 (latent index 1) for step 1
    - z3 (latent index 3) for step 2
    
    This matches what we discovered CODI actually stores in GSM8k latents.
    
    For each step, generate:
    - Extraction question: "What is the result of step N?"
    - Classification questions about operation, magnitude, etc.
    """
    examples = []
    
    extraction_templates = [
        "What is the intermediate calculation result?",
        "What value was computed?",
        "What is the result of this calculation step?",
        "Tell me the number from this reasoning step.",
        "What was calculated here?",
        "Extract the numeric value.",
        "What number is stored here?",
        "Decode this activation.",
    ]
    
    # Map step number to latent position (for 2-step problems)
    # Based on exploration: z1 (index 1) has step 1, z3 (index 3) has step 2
    STEP_TO_LATENT = {
        1: 1,  # Step 1 -> latent index 1 (z1)
        2: 3,  # Step 2 -> latent index 3 (z3)
    }
    
    for p in problems:
        # Only generate examples for 2-step problems with clear mapping
        if p.num_steps != 2:
            continue
        
        for step in p.steps:
            step_num = step["step_number"]
            if step_num not in STEP_TO_LATENT:
                continue
            
            latent_pos = STEP_TO_LATENT[step_num]
            
            # Extraction questions
            for template in extraction_templates:
                examples.append({
                    "problem_id": p.id,
                    "question": p.question,
                    "num_steps": p.num_steps,
                    "step_number": step_num,
                    "latent_position": latent_pos,
                    "latent_name": f"z{latent_pos}",
                    "qa_question": template,
                    "qa_answer": step["result"],
                    "qa_type": "extraction",
                    "operation": step["operation"],
                    "expression": step["expression"],
                    "source": "gsm8k_2step",
                })
            
            # Operation classification
            for op in ["addition", "subtraction", "multiplication", "division"]:
                is_this_op = step["operation"] == op
                examples.append({
                    "problem_id": p.id,
                    "question": p.question,
                    "num_steps": p.num_steps,
                    "step_number": step_num,
                    "latent_position": latent_pos,
                    "latent_name": f"z{latent_pos}",
                    "qa_question": f"Is this step performing {op}?",
                    "qa_answer": "yes" if is_this_op else "no",
                    "qa_type": "classification_operation",
                    "operation": step["operation"],
                    "expression": step["expression"],
                    "source": "gsm8k_2step",
                })
            
            # Result magnitude classification
            try:
                result_val = float(step["result"])
                for threshold in [10, 50, 100, 500, 1000]:
                    examples.append({
                        "problem_id": p.id,
                        "question": p.question,
                        "num_steps": p.num_steps,
                        "step_number": step_num,
                        "latent_position": latent_pos,
                        "latent_name": f"z{latent_pos}",
                        "qa_question": f"Is the result greater than {threshold}?",
                        "qa_answer": "yes" if result_val > threshold else "no",
                        "qa_type": "classification_magnitude",
                        "operation": step["operation"],
                        "expression": step["expression"],
                        "source": "gsm8k_2step",
                    })
            except ValueError:
                pass
    
    return examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse GSM8k for AO training")
    parser.add_argument("--output_dir", type=str, default="data/gsm8k",
                        help="Output directory for parsed data")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction to hold out for testing")
    parser.add_argument("--min_steps", type=int, default=1,
                        help="Minimum number of reasoning steps")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Maximum number of reasoning steps")
    parser.add_argument("--only_two_step", action="store_true",
                        help="Only include problems with exactly 2 reasoning steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--generate_qa", action="store_true",
                        help="Also generate QA training examples")
    args = parser.parse_args()
    
    # Parse GSM8k
    train, test = load_and_parse_gsm8k(
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        only_two_step=args.only_two_step,
    )
    
    # Print examples
    print_examples(train, n=5)
    
    # Save parsed data
    output_dir = Path(args.output_dir)
    save_parsed_data(train, test, output_dir)
    
    # Optionally generate QA examples
    if args.generate_qa:
        print("\nGenerating QA training examples...")
        train_qa = generate_qa_examples(train)
        test_qa = generate_qa_examples(test)
        
        with open(output_dir / "gsm8k_train_qa.json", "w") as f:
            json.dump(train_qa, f, indent=2)
        print(f"Saved {len(train_qa)} train QA examples")
        
        with open(output_dir / "gsm8k_test_qa.json", "w") as f:
            json.dump(test_qa, f, indent=2)
        print(f"Saved {len(test_qa)} test QA examples")
    
    print("\nDone!")
