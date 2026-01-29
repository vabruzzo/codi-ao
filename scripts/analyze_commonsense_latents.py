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
    """Load actual CommonsenseQA dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        
        print(f"Loading CommonsenseQA {split} split from HuggingFace...")
        dataset = load_dataset("commonsense_qa", split=split)
        
        examples = []
        for item in dataset:
            # Format: question with choices
            question = item["question"]
            choices = item["choices"]
            
            # Build choice string
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
        
        print(f"Loaded {len(examples)} examples from CommonsenseQA")
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
    return f"{question} Give the answer only and nothing else."


def extract_answer(prediction: str) -> str:
    """Extract answer letter from prediction."""
    # Look for single letter answer at the end or standalone
    prediction = prediction.strip()
    
    # Check last character
    if prediction and prediction[-1] in "ABCDE":
        return prediction[-1]
    
    # Check for letter followed by punctuation at end
    for i in range(len(prediction) - 1, -1, -1):
        if prediction[i] in "ABCDE":
            return prediction[i]
    
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

    # Get special token ID for eocot
    eocot_token = "<|eocot|>"
    
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
        expected = example["answer"]

        # Collect latents
        result = wrapper.collect_latents(prompt, ground_truth_answer=expected)
        predicted = extract_answer(result.predicted_answer)
        is_correct = predicted == expected

        if args.verbose:
            print(f"\n[{i+1}] Expected: {expected}, Predicted: {predicted}, Correct: {is_correct}")

        # Analyze each latent position
        example_data = {
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
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

            # Track first eocot position
            if eocot_token in top1_token and first_eocot_pos is None:
                first_eocot_pos = j + 1

            # Check if answer letter appears in top tokens
            for k, tok in enumerate(top_tokens[:5]):
                if expected in tok:
                    answer_found_in_z = z_name
                    answer_in_latent[z_name] += 1

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
