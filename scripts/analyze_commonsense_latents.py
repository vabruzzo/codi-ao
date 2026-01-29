#!/usr/bin/env python3
"""
Systematic analysis of CODI latent vectors on CommonsenseQA.

This script analyzes patterns in how CODI encodes commonsense reasoning:
1. When does <|eocot|> appear? (how many latent steps are used)
2. Does z2 encode task-relevant concepts?
3. Does any latent encode the answer choice (A, B, C, D, E)?
4. How do patterns differ between correct vs incorrect predictions?

Usage:
    python scripts/analyze_commonsense_latents.py --n_examples 50
    python scripts/analyze_commonsense_latents.py --n_examples 100 --use_real_data
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def load_commonsenseqa(n_examples: int = 100, split: str = "validation") -> list[dict]:
    """Load CommonsenseQA dataset - tries CODI's training data first, then fallback."""
    try:
        from datasets import load_dataset
        
        # Try to load the EXACT dataset CODI was trained on
        print(f"Loading zen-E/CommonsenseQA-GPT4omini (CODI's training data)...")
        try:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
            # This dataset has train/validation splits
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["validation"] if "validation" in dataset else dataset["train"]
            
            examples = []
            for item in data:
                # The zen-E dataset should have 'question' and 'answer' fields
                # Question already includes choices in the correct format
                question = item["question"].strip()
                answer = item["answer"].strip()
                
                examples.append({
                    "question": question,
                    "answer": answer,
                })
                
                if len(examples) >= n_examples:
                    break
            
            print(f"Loaded {len(examples)} examples from zen-E/CommonsenseQA-GPT4omini")
            return examples
            
        except Exception as e:
            print(f"Could not load zen-E dataset: {e}")
            print("Falling back to standard commonsense_qa...")
        
        # Fallback to standard CommonsenseQA
        print(f"Loading commonsense_qa {split} split from HuggingFace...")
        dataset = load_dataset("commonsense_qa", split=split)
        
        examples = []
        for item in dataset:
            # Format: question with choices
            question = item["question"]
            choices = item["choices"]
            
            # Build choice string - match CODI's expected format
            choice_labels = choices["label"]  # ['A', 'B', 'C', 'D', 'E']
            choice_texts = choices["text"]
            
            choice_str = " ".join([f"{l}: {t}" for l, t in zip(choice_labels, choice_texts)])
            full_question = f"{question} Choices: {choice_str}"
            
            examples.append({
                "question": full_question,
                "answer": item["answerKey"],
            })
            
            if len(examples) >= n_examples:
                break
        
        print(f"Loaded {len(examples)} examples from commonsense_qa (fallback)")
        return examples
        
    except ImportError:
        print("Warning: 'datasets' library not installed. Run: pip install datasets")
        print("Falling back to built-in examples...")
        return None
    except Exception as e:
        print(f"Warning: Could not load CommonsenseQA: {e}")
        print("Falling back to built-in examples...")
        return None


# Fallback examples (subset from actual CommonsenseQA or similar)
FALLBACK_EXAMPLES = [
    {"question": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Choices: A: ignore B: enforce C: authoritarian D: yell at E: avoid", "answer": "A"},
    {"question": "Where would you put a plate after cleaning it? Choices: A: refrigerator B: table C: cupboard D: dishwasher E: restaurant", "answer": "C"},
    {"question": "What do people aim to do at work? Choices: A: complete job B: learn from each other C: kill animals D: have fun E: make money", "answer": "E"},
    {"question": "Where is a business card likely to be filed? Choices: A: wallet B: pocket C: briefcase D: desk drawer E: rolodex", "answer": "E"},
    {"question": "What is the result of having to wait a long time for something? Choices: A: frustration B: irritation C: annoyance D: bothered E: anger", "answer": "A"},
    {"question": "If you are prone to being deceptive, you would be thought of as what? Choices: A: untrustworthy B: unethical C: sneaky D: devious E: dishonest", "answer": "A"},
    {"question": "The artist was sitting quietly pondering, then suddenly they got inspired and began to what? Choices: A: paint B: think C: create art D: write E: meditate", "answer": "C"},
    {"question": "What could happen to a token that you have a sentimental attachment to? Choices: A: value B: keep C: save D: treasure E: possess", "answer": "D"},
    {"question": "Where might a computer be bought? Choices: A: office B: house C: store D: school E: library", "answer": "C"},
    {"question": "What do you do when you want to express gratitude? Choices: A: thank B: smile C: say goodbye D: wave E: help others", "answer": "A"},
    {"question": "Where would you find a ticket booth at an amusement park? Choices: A: movie theater B: train station C: carnival D: entrance E: mall", "answer": "D"},
    {"question": "What is a treat that a dog likes? Choices: A: bone B: ball C: toy D: leash E: collar", "answer": "A"},
    {"question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? Choices: A: bank B: move theater C: department store D: mall E: building", "answer": "A"},
    {"question": "Where do you put your groceries after buying them? Choices: A: car B: shelf C: refrigerator D: bag E: kitchen", "answer": "A"},
    {"question": "What happens to your body when you exercise? Choices: A: sweat B: sleep C: eat D: drink E: rest", "answer": "A"},
    {"question": "Where would you find a jellyfish? Choices: A: pool B: desert C: ocean D: river E: lake", "answer": "C"},
    {"question": "What might a person do when they feel scared? Choices: A: laugh B: run C: sing D: dance E: sleep", "answer": "B"},
    {"question": "What do you call a baby dog? Choices: A: kitten B: puppy C: calf D: cub E: chick", "answer": "B"},
    {"question": "Where would you store cold food? Choices: A: oven B: pantry C: refrigerator D: cabinet E: counter", "answer": "C"},
    {"question": "What happens when you mix red and blue? Choices: A: green B: yellow C: purple D: orange E: brown", "answer": "C"},
]


def format_prompt(question: str) -> str:
    """Format prompt exactly as CODI expects (from test.py line 147)."""
    return f"{question} Output only the answer and nothing else."


def extract_generation(full_output: str, prompt: str) -> str:
    """
    Extract only the model's generated text, removing the prompt.
    
    Handles tokenization differences by using multiple strategies.
    """
    # Strategy 1: Direct substring removal (most common case)
    if prompt in full_output:
        return full_output[len(prompt):].strip()
    
    # Strategy 2: Find the end marker and take everything after
    markers = ["nothing else.", "nothing else", "Output only the answer"]
    for marker in markers:
        if marker in full_output:
            idx = full_output.rfind(marker) + len(marker)
            return full_output[idx:].strip()
    
    # Strategy 3: Normalize whitespace and try again
    normalized_prompt = " ".join(prompt.split())
    normalized_output = " ".join(full_output.split())
    if normalized_prompt in normalized_output:
        idx = normalized_output.find(normalized_prompt) + len(normalized_prompt)
        return normalized_output[idx:].strip()
    
    # Strategy 4: Last resort - assume last part is generation
    # Take everything after the last "?" which likely ends the question
    if "?" in full_output:
        idx = full_output.rfind("?") + 1
        remainder = full_output[idx:].strip()
        # Skip past "Output only..." if present
        for marker in markers:
            if marker in remainder:
                idx = remainder.find(marker) + len(marker)
                remainder = remainder[idx:].strip()
        return remainder
    
    return full_output.strip()


def extract_answer_from_generation(full_output: str, prompt: str) -> str:
    """
    Extract answer letter from model's GENERATED tokens only (not prompt).
    
    Args:
        full_output: The full decoded sequence (prompt + generation)
        prompt: The original prompt that was sent to the model
    
    Returns:
        Single letter A-E if found in generation, else ""
    """
    import re
    
    # First extract just the generation
    generation = extract_generation(full_output, prompt)
    
    # Empty generation
    if not generation:
        return ""
    
    # Check if generation is just a single letter
    if generation in "ABCDE":
        return generation
    
    # Check first non-whitespace character
    if generation[0] in "ABCDE":
        return generation[0]
    
    # Look for pattern like "A)" or "A:" or "A." at start
    match = re.match(r'^([A-E])[\s\.\:\)\,]', generation)
    if match:
        return match.group(1)
    
    # Look for standalone letter (word boundary)
    match = re.search(r'\b([A-E])\b', generation)
    if match:
        return match.group(1)
    
    # NO fallback to any A-E letter - too permissive (picks up "A" from "Also", etc.)
    # If we can't find a clear answer letter, return empty
    return ""


def main():
    parser = argparse.ArgumentParser(description="Analyze CODI latents on CommonsenseQA")
    parser.add_argument("--n_examples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_real_data", action="store_true", help="Load actual CommonsenseQA from HuggingFace")
    args = parser.parse_args()

    print("=" * 70)
    print("CODI CommonsenseQA Latent Analysis")
    print("=" * 70)

    # Load examples
    if args.use_real_data:
        examples = load_commonsenseqa(n_examples=args.n_examples)
        if examples is None:
            examples = FALLBACK_EXAMPLES[:args.n_examples]
            print(f"Using {len(examples)} fallback examples")
    else:
        examples = FALLBACK_EXAMPLES[:args.n_examples]
        print(f"Using {len(examples)} built-in examples (use --use_real_data for actual CommonsenseQA)")

    # Load CODI
    print("\nLoading CODI model...")
    from src.codi_wrapper import CODIWrapper
    wrapper = CODIWrapper.from_pretrained(device=args.device)

    # Get special token for eocot detection
    eocot_token = "<|eocot|>"
    # Also check for common eocot-like patterns (model may output similar tokens)
    eocot_patterns = ["<|eocot|>", "<|eot|>", "eocot", "eot"]
    
    # Analysis containers
    results = []
    z_position_tokens = defaultdict(Counter)  # z1, z2, etc. -> token -> count
    eocot_first_position = Counter()  # Which z position first shows eocot
    correct_vs_incorrect = {"correct": [], "incorrect": []}
    answer_in_latent = Counter()  # Does any z contain the answer?

    print(f"\nAnalyzing {len(examples)} examples...")
    print("-" * 70)

    for i, example in enumerate(examples):
        prompt = format_prompt(example["question"])
        expected = example["answer"].strip().upper()  # Normalize expected answer

        # Collect latents
        result = wrapper.collect_latents(prompt, ground_truth_answer=expected)
        
        # Get raw generation for debugging (using robust extraction) - MUST be before verbose print
        raw_generation = extract_generation(result.predicted_answer, prompt)
        
        # Extract answer from GENERATION ONLY (not prompt)
        predicted = extract_answer_from_generation(result.predicted_answer, prompt)
        is_correct = predicted == expected

        if args.verbose:
            # Show what the model actually generated (without prompt)
            gen_preview = raw_generation[:50] if raw_generation else "(empty)"
            print(f"\n[{i+1}] Expected: {expected}, Generated: '{gen_preview}...', Extracted: {predicted}, Correct: {is_correct}")
        
        # Analyze each latent position
        example_data = {
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "raw_generation": raw_generation[:100],  # First 100 chars
            "latents": [],
        }

        first_eocot_pos = None
        answer_found_in_z = None

        for j, lv in enumerate(result.latent_vectors[:6]):
            lens = wrapper.logit_lens(lv, top_k=10)
            top_tokens = lens.layer_results[0]["top_tokens"]
            top_probs = lens.layer_results[0]["top_probs"]

            z_name = f"z{j + 1}"
            top1_token = top_tokens[0]
            top1_prob = top_probs[0]

            # Track token distribution
            z_position_tokens[z_name][top1_token] += 1

            # Track first eocot position (check multiple patterns)
            if first_eocot_pos is None:
                for pattern in eocot_patterns:
                    if pattern.lower() in top1_token.lower():
                        first_eocot_pos = j + 1
                        break
                # Also check if probability is very high (>0.9) for any eocot-like token
                if first_eocot_pos is None and top1_prob > 0.9:
                    if any(p.lower() in top1_token.lower() for p in eocot_patterns):
                        first_eocot_pos = j + 1

            # Check if answer letter appears as EXACT match in top tokens
            # (not just as substring - "A" in "Apple" shouldn't count)
            for k, tok in enumerate(top_tokens[:5]):
                tok_clean = tok.strip()
                # Only count if token IS the letter or letter with punctuation
                if tok_clean == expected or tok_clean in [f"{expected}.", f"{expected}:", f"{expected})"]:
                    if answer_found_in_z is None:  # Only count first occurrence
                        answer_found_in_z = z_name
                        answer_in_latent[z_name] += 1
                    break

            example_data["latents"].append({
                "position": z_name,
                "top1": top1_token,
                "top1_prob": top1_prob,
                "top5": list(zip(top_tokens[:5], top_probs[:5])),
            })

            if args.verbose:
                print(f"  {z_name}: '{top1_token}' ({top1_prob:.2f})")

        # Track eocot position
        if first_eocot_pos:
            eocot_first_position[first_eocot_pos] += 1
        else:
            eocot_first_position["never"] += 1

        results.append(example_data)
        if is_correct:
            correct_vs_incorrect["correct"].append(example_data)
        else:
            correct_vs_incorrect["incorrect"].append(example_data)

    # Print analysis
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # 1. Overall accuracy
    n_correct = len(correct_vs_incorrect["correct"])
    n_total = len(results)
    print(f"\n1. CODI Accuracy: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
    
    # Show some example generations for sanity check
    print("\n   Sample generations (first 5):")
    for i, ex in enumerate(results[:5]):
        raw = ex.get('raw_generation', '')[:40]
        print(f"   [{i+1}] Expected: {ex['expected']}, Generated: '{raw}...', Extracted: {ex['predicted']}, Correct: {ex['correct']}")

    # 2. When does eocot first appear?
    print("\n2. First <|eocot|> Position (how many latent steps used):")
    for pos, count in sorted(eocot_first_position.items()):
        pct = count / n_total * 100
        print(f"   z{pos}: {count} examples ({pct:.1f}%)")

    # 3. Top tokens at each position
    print("\n3. Most Common Top-1 Tokens by Position:")
    for z_name in ["z1", "z2", "z3", "z4", "z5", "z6"]:
        print(f"\n   {z_name}:")
        for token, count in z_position_tokens[z_name].most_common(5):
            pct = count / n_total * 100
            print(f"      '{token}': {count} ({pct:.1f}%)")

    # 4. Answer found in latents?
    print("\n4. Answer Letter Found in Top-5 Tokens:")
    for z_name in ["z1", "z2", "z3", "z4", "z5", "z6"]:
        count = answer_in_latent[z_name]
        pct = count / n_total * 100
        print(f"   {z_name}: {count} times ({pct:.1f}%)")

    # 5. Correct vs Incorrect patterns
    print("\n5. Correct vs Incorrect Predictions:")
    print(f"   Correct: {len(correct_vs_incorrect['correct'])}")
    print(f"   Incorrect: {len(correct_vs_incorrect['incorrect'])}")

    # Compare z2 tokens for correct vs incorrect
    print("\n   z2 top tokens for CORRECT predictions:")
    correct_z2 = Counter()
    for ex in correct_vs_incorrect["correct"]:
        if len(ex["latents"]) > 1:
            correct_z2[ex["latents"][1]["top1"]] += 1
    for tok, cnt in correct_z2.most_common(5):
        print(f"      '{tok}': {cnt}")

    print("\n   z2 top tokens for INCORRECT predictions:")
    incorrect_z2 = Counter()
    for ex in correct_vs_incorrect["incorrect"]:
        if len(ex["latents"]) > 1:
            incorrect_z2[ex["latents"][1]["top1"]] += 1
    for tok, cnt in incorrect_z2.most_common(5):
        print(f"      '{tok}': {cnt}")

    # Save detailed results
    output_path = Path("reports/commonsense_latent_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Counter objects for JSON serialization
    summary = {
        "n_examples": n_total,
        "accuracy": n_correct / n_total,
        "eocot_first_position": dict(eocot_first_position),
        "z_position_top_tokens": {k: dict(v.most_common(10)) for k, v in z_position_tokens.items()},
        "answer_in_latent": dict(answer_in_latent),
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_path}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
Compare these findings to math (from LessWrong):
- Math: z2 encodes Step 1 result (100%), z4 encodes Step 2 result (85%)
- Math: <|eocot|> typically appears in z5/z6 (uses 4-5 latent steps)

Questions to answer:
1. Does commonsense use fewer latent steps? (eocot appearing earlier)
2. Does z2 encode task-relevant concepts for commonsense?
3. Is there a consistent pattern across examples?
""")


if __name__ == "__main__":
    main()
