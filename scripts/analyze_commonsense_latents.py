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


def parse_choices(question: str) -> dict[str, str]:
    """
    Parse choice texts from a CommonsenseQA question.
    
    Returns dict mapping letter -> choice text, e.g.:
    {"A": "ignore", "B": "enforce", "C": "authoritarian", ...}
    """
    import re
    
    choices = {}
    
    # Pattern: "A: text" or "A) text" or just "A text" followed by next choice or end
    # Look for patterns like "A: ignore B: enforce" or "A: ignore, B: enforce"
    pattern = r'([A-E])[\s:\)]+([^A-E:]+?)(?=\s+[A-E][\s:\)]|$)'
    
    matches = re.findall(pattern, question)
    for letter, text in matches:
        # Clean up the text
        text = text.strip().rstrip(',').strip()
        if text:
            choices[letter] = text.lower()
    
    # If regex didn't work well, try simpler approach
    if len(choices) < 3:
        # Look for "Choices:" section
        if "Choices:" in question:
            choices_part = question.split("Choices:")[-1]
        elif "choices:" in question.lower():
            idx = question.lower().find("choices:")
            choices_part = question[idx + 8:]
        else:
            choices_part = question
        
        # Split by letter patterns
        parts = re.split(r'\s+([A-E])[\s:\)]+', choices_part)
        # parts will be like ['', 'A', 'ignore', 'B', 'enforce', ...]
        for i in range(1, len(parts) - 1, 2):
            letter = parts[i]
            text = parts[i + 1].strip().rstrip(',').strip() if i + 1 < len(parts) else ""
            # Take only first word/phrase before next letter
            text = re.split(r'\s+[A-E][\s:\)]', text)[0].strip()
            if text and letter not in choices:
                choices[letter] = text.lower()
    
    return choices


def token_matches_concept(token: str, concept: str) -> bool:
    """
    Check if a token matches a concept (answer choice text).
    
    Handles:
    - Exact match (case-insensitive)
    - Token is start of concept (e.g., "Amaz" matches "amazon")
    - Concept is start of token (e.g., "amazon" matches "Amazon.com")
    - Token contains the concept as a word
    """
    token_clean = token.strip().lower()
    concept_clean = concept.strip().lower()
    
    # Skip very short tokens (likely punctuation)
    if len(token_clean) < 2:
        return False
    
    # Skip common filler tokens
    filler = {'the', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'is', 'it', 'therefore', 'thus', 'so'}
    if token_clean in filler:
        return False
    
    # Exact match
    if token_clean == concept_clean:
        return True
    
    # Token starts with concept (concept is prefix)
    if len(concept_clean) >= 3 and token_clean.startswith(concept_clean):
        return True
    
    # Concept starts with token (token is prefix) - for truncated tokens
    if len(token_clean) >= 3 and concept_clean.startswith(token_clean):
        return True
    
    # For multi-word concepts, check if token matches any word
    concept_words = concept_clean.split()
    for word in concept_words:
        if len(word) >= 3:
            if token_clean == word or token_clean.startswith(word) or word.startswith(token_clean):
                return True
    
    return False


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


def load_held_out_dataset(n_examples: int = 100) -> list[dict]:
    """Load a held-out commonsense dataset (not used in CODI training)."""
    try:
        from datasets import load_dataset
        
        # Try PIQA first (Physical Intuition QA)
        print("Loading PIQA (held-out dataset)...")
        try:
            dataset = load_dataset("piqa", split="validation")
            
            examples = []
            for item in dataset:
                # PIQA format: goal, sol1, sol2, label (0 or 1)
                goal = item["goal"]
                choices = [item["sol1"], item["sol2"]]
                label = item["label"]
                answer_letter = ["A", "B"][label]
                
                # Format like CommonsenseQA
                choice_str = " ".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
                full_question = f"{goal} Choices: {choice_str}"
                
                examples.append({
                    "question": full_question,
                    "answer": answer_letter,
                    "choices": {chr(65+i): c for i, c in enumerate(choices)},
                })
                
                if len(examples) >= n_examples:
                    break
            
            print(f"Loaded {len(examples)} examples from PIQA")
            return examples
        except Exception as e:
            print(f"Could not load PIQA: {e}")
        
        # Try ARC-Easy as fallback
        print("Trying ARC-Easy...")
        try:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
            
            examples = []
            for item in dataset:
                question = item["question"]
                choices_data = item["choices"]
                labels = choices_data["label"]
                texts = choices_data["text"]
                answer_key = item["answerKey"]
                
                # Format choices
                choice_str = " ".join([f"{l}: {t}" for l, t in zip(labels, texts)])
                full_question = f"{question} Choices: {choice_str}"
                
                examples.append({
                    "question": full_question,
                    "answer": answer_key,
                    "choices": {l: t for l, t in zip(labels, texts)},
                })
                
                if len(examples) >= n_examples:
                    break
            
            print(f"Loaded {len(examples)} examples from ARC-Easy")
            return examples
        except Exception as e:
            print(f"Could not load ARC-Easy: {e}")
        
        return None
    except Exception as e:
        print(f"Could not load held-out dataset: {e}")
        return None


def shuffle_choices(question: str, answer: str, choices: dict = None) -> tuple[str, str, dict]:
    """
    Shuffle the choice order in a question and return new question, new answer letter, and mapping.
    
    This tests if the latent tracks the CONTENT or just the POSITION (A/B/C/D/E).
    """
    import re
    
    # If choices dict not provided, try to parse from question
    if choices is None:
        choices = parse_choices(question)
    
    if len(choices) < 2:
        return question, answer, {}
    
    # Get original answer content
    original_answer_text = choices.get(answer, "")
    
    # Shuffle the letters
    letters = list(choices.keys())
    texts = [choices[l] for l in letters]
    
    # Create shuffled mapping
    random.shuffle(letters)
    new_choices = {letters[i]: texts[i] for i in range(len(texts))}
    
    # Find new answer letter (which letter now has the original answer text)
    new_answer = answer  # default
    for letter, text in new_choices.items():
        if text.lower() == original_answer_text.lower():
            new_answer = letter
            break
    
    # Rebuild the question with shuffled choices
    # Find and replace the choices section
    choice_str = " ".join([f"{l}: {new_choices[l]}" for l in sorted(new_choices.keys())])
    
    # Try to find the choices section and replace it
    if "Choices:" in question:
        base = question.split("Choices:")[0]
        new_question = f"{base}Choices: {choice_str}"
    else:
        # Just append
        new_question = f"{question} Choices: {choice_str}"
    
    return new_question, new_answer, {"original": answer, "shuffled": new_answer, "mapping": new_choices}


def calculate_entropy(probs: list[float]) -> float:
    """Calculate entropy of a probability distribution."""
    import math
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_margin(probs: list[float]) -> float:
    """Calculate margin between top-1 and top-2 probabilities."""
    if len(probs) < 2:
        return probs[0] if probs else 0.0
    sorted_probs = sorted(probs, reverse=True)
    return sorted_probs[0] - sorted_probs[1]


def main():
    parser = argparse.ArgumentParser(description="Analyze CODI latents on CommonsenseQA")
    parser.add_argument("--n_examples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_real_data", action="store_true", help="Load actual CommonsenseQA from HuggingFace")
    
    # New ablation flags
    parser.add_argument("--held_out", action="store_true", help="Use SocialIQA as held-out test set")
    parser.add_argument("--shuffle_choices", action="store_true", help="Shuffle choice order to test position vs content")
    parser.add_argument("--report_entropy", action="store_true", help="Report entropy and margin at z2/z3")
    parser.add_argument("--no_choices_prompt", action="store_true", help="Test prompt without explicit Choices: format")
    args = parser.parse_args()

    print("=" * 70)
    print("CODI CommonsenseQA Latent Analysis")
    if args.held_out:
        print("(Using HELD-OUT dataset: SocialIQA)")
    if args.shuffle_choices:
        print("(SHUFFLE ABLATION: randomizing choice order)")
    if args.report_entropy:
        print("(Reporting entropy/margin metrics)")
    if args.no_choices_prompt:
        print("(NO-CHOICES prompt ablation)")
    print("=" * 70)

    # Load examples
    if args.held_out:
        examples = load_held_out_dataset(n_examples=args.n_examples)
        if examples is None:
            print("Failed to load held-out dataset, falling back to CommonsenseQA")
            examples = load_commonsenseqa(n_examples=args.n_examples)
    elif args.use_real_data:
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
    
    # Three types of "answer in latent" tracking (for CORRECT answer)
    answer_letter_in_latent = Counter()   # Exact letter match (A, B, C, D, E)
    answer_concept_in_latent = Counter()  # Concept match (the choice text)
    answer_combined_in_latent = Counter() # Either letter OR concept
    
    # NEW: Track if model's PREDICTED output appears in latent (regardless of correctness)
    # This tells us: can we read what the model will commit to?
    predicted_in_latent = Counter()       # Position where predicted letter first appears
    z2_top1_matches_output = 0            # z2's top-1 token matches model's output
    z3_top1_matches_output = 0            # z3's top-1 token matches model's output
    
    # Cross-tabulation: latent prediction vs correctness
    z2_matches_and_correct = 0
    z2_matches_and_incorrect = 0
    z2_no_match_and_correct = 0
    z2_no_match_and_incorrect = 0
    
    # Track what OTHER tokens appear in top-5 alongside the predicted letter
    other_tokens_in_top5 = Counter()  # tokens that aren't A-E
    letter_tokens_in_top5 = Counter() # tokens that ARE A-E (competing choices)
    
    # Entropy and margin tracking
    z2_entropies = []
    z3_entropies = []
    z2_margins = []
    z3_margins = []
    
    # Shuffle ablation tracking
    shuffle_latent_tracks_original = 0  # Latent shows ORIGINAL answer letter (content tracking)
    shuffle_latent_tracks_shuffled = 0  # Latent shows SHUFFLED answer letter (position tracking)
    shuffle_total = 0

    print(f"\nAnalyzing {len(examples)} examples...")
    print("-" * 70)

    for i, example in enumerate(examples):
        question = example["question"]
        expected = example["answer"].strip().upper()
        original_expected = expected  # Keep track for shuffle ablation
        
        # Get choices dict if available (for shuffle ablation)
        choices_dict = example.get("choices", None)
        
        # Apply shuffle ablation if requested
        shuffle_info = None
        if args.shuffle_choices:
            question, expected, shuffle_info = shuffle_choices(question, expected, choices_dict)
            shuffle_total += 1
        
        # Apply no-choices prompt ablation if requested
        if args.no_choices_prompt:
            # Remove "Choices:" section from prompt
            if "Choices:" in question:
                question = question.split("Choices:")[0].strip()
        
        prompt = format_prompt(question)

        # Collect latents
        result = wrapper.collect_latents(prompt, ground_truth_answer=expected)
        
        # Get raw generation for debugging (using robust extraction) - MUST be before verbose print
        raw_generation = extract_generation(result.predicted_answer, prompt)
        
        # Extract answer from GENERATION ONLY (not prompt)
        predicted = extract_answer_from_generation(result.predicted_answer, prompt)
        is_correct = predicted == expected

        # Parse choices to get answer concept for this example
        choices = parse_choices(question)
        answer_concept = choices.get(expected, "")  # e.g., "ignore" for answer "A"
        
        if args.verbose:
            # Show what the model actually generated (without prompt)
            gen_preview = raw_generation[:50] if raw_generation else "(empty)"
            print(f"\n[{i+1}] Expected: {expected} ('{answer_concept}'), Generated: '{gen_preview}...', Extracted: {predicted}, Correct: {is_correct}")
        
        # Analyze each latent position
        example_data = {
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "raw_generation": raw_generation[:100],  # First 100 chars
            "latents": [],
        }

        first_eocot_pos = None
        letter_found_in_z = None
        concept_found_in_z = None
        combined_found_in_z = None
        
        # answer_concept was already parsed above for verbose output

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

            # Check for answer in latent - THREE types of matches
            found_letter = False
            found_concept = False
            
            for k, tok in enumerate(top_tokens[:5]):
                tok_clean = tok.strip()
                
                # 1. Letter match: exact letter or letter with punctuation
                if not found_letter:
                    if tok_clean == expected or tok_clean in [f"{expected}.", f"{expected}:", f"{expected})"]:
                        found_letter = True
                
                # 2. Concept match: token matches the answer's choice text
                if not found_concept and answer_concept:
                    if token_matches_concept(tok, answer_concept):
                        found_concept = True
            
            # Track first occurrence of each match type
            if found_letter and letter_found_in_z is None:
                letter_found_in_z = z_name
                answer_letter_in_latent[z_name] += 1
            
            if found_concept and concept_found_in_z is None:
                concept_found_in_z = z_name
                answer_concept_in_latent[z_name] += 1
            
            if (found_letter or found_concept) and combined_found_in_z is None:
                combined_found_in_z = z_name
                answer_combined_in_latent[z_name] += 1

            example_data["latents"].append({
                "position": z_name,
                "top1": top1_token,
                "top1_prob": top1_prob,
                "top5": list(zip(top_tokens[:5], top_probs[:5])),
            })
            
            if args.verbose:
                print(f"  {z_name}: '{top1_token}' ({top1_prob:.2f})")
        
        # Store match info in example data
        example_data["answer_concept"] = answer_concept
        example_data["letter_found_in"] = letter_found_in_z
        example_data["concept_found_in"] = concept_found_in_z
        example_data["combined_found_in"] = combined_found_in_z
        
        # NEW: Check if model's PREDICTED output appears in latents
        # This measures: can we read what the model will commit to?
        if predicted and len(example_data["latents"]) >= 3:
            z2_data = example_data["latents"][1]  # z2
            z3_data = example_data["latents"][2]  # z3
            
            # Check z2 top-1
            z2_top1 = z2_data["top1"].strip()
            z2_matches = z2_top1 == predicted or z2_top1 in [f"{predicted}.", f"{predicted}:", f"{predicted})"]
            if z2_matches:
                z2_top1_matches_output += 1
                if is_correct:
                    z2_matches_and_correct += 1
                else:
                    z2_matches_and_incorrect += 1
            else:
                if is_correct:
                    z2_no_match_and_correct += 1
                else:
                    z2_no_match_and_incorrect += 1
            
            # Check z3 top-1
            z3_top1 = z3_data["top1"].strip()
            if z3_top1 == predicted or z3_top1 in [f"{predicted}.", f"{predicted}:", f"{predicted})"]:
                z3_top1_matches_output += 1
            
            # Check if predicted letter appears anywhere in z2/z3 top-5
            predicted_found = None
            for z_idx, z_data in [(1, z2_data), (2, z3_data)]:
                for tok, prob in z_data["top5"]:
                    tok_clean = tok.strip()
                    if tok_clean == predicted or tok_clean in [f"{predicted}.", f"{predicted}:", f"{predicted})"]:
                        if predicted_found is None:
                            predicted_found = f"z{z_idx + 1}"
                            predicted_in_latent[predicted_found] += 1
                        break
                if predicted_found:
                    break
            
            example_data["predicted_found_in"] = predicted_found
            
            # Track what OTHER tokens appear in z2 top-5 (not the predicted letter)
            for tok, prob in z2_data["top5"]:
                tok_clean = tok.strip()
                # Skip the predicted letter itself
                is_predicted = tok_clean == predicted or tok_clean in [f"{predicted}.", f"{predicted}:", f"{predicted})"]
                if is_predicted:
                    continue
                # Classify as letter or non-letter
                is_letter = tok_clean in list("ABCDE") or any(tok_clean == f"{L}{p}" for L in "ABCDE" for p in [".", ":", ")"])
                if is_letter:
                    letter_tokens_in_top5[tok_clean] += 1
                else:
                    other_tokens_in_top5[tok_clean] += 1
            
            # Track entropy and margin if requested
            if args.report_entropy:
                z2_probs = [p for _, p in z2_data["top5"]]
                z3_probs = [p for _, p in z3_data["top5"]]
                z2_entropies.append(calculate_entropy(z2_probs))
                z3_entropies.append(calculate_entropy(z3_probs))
                z2_margins.append(calculate_margin(z2_probs))
                z3_margins.append(calculate_margin(z3_probs))
            
            # Track shuffle ablation results
            if args.shuffle_choices and shuffle_info:
                original_letter = shuffle_info.get("original", "")
                shuffled_letter = shuffle_info.get("shuffled", "")
                
                # Check if z2/z3 top-5 contains original or shuffled letter
                all_top5_tokens = [t.strip() for t, _ in z2_data["top5"]] + [t.strip() for t, _ in z3_data["top5"]]
                
                has_original = any(t == original_letter or t in [f"{original_letter}.", f"{original_letter}:"] for t in all_top5_tokens)
                has_shuffled = any(t == shuffled_letter or t in [f"{shuffled_letter}.", f"{shuffled_letter}:"] for t in all_top5_tokens)
                
                if has_original and original_letter != shuffled_letter:
                    shuffle_latent_tracks_original += 1  # Content tracking
                if has_shuffled:
                    shuffle_latent_tracks_shuffled += 1  # Position tracking

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

    # 4. Answer found in latents? - THREE metrics
    print("\n4. Answer Found in Top-5 Tokens (by position of first match):")
    
    # Calculate totals
    total_letter = sum(answer_letter_in_latent.values())
    total_concept = sum(answer_concept_in_latent.values())
    total_combined = sum(answer_combined_in_latent.values())
    
    print(f"\n   TOTALS:")
    print(f"      Letter matches (exact A-E):     {total_letter}/{n_total} ({total_letter/n_total*100:.1f}%)")
    print(f"      Concept matches (choice text):  {total_concept}/{n_total} ({total_concept/n_total*100:.1f}%)")
    print(f"      Combined (letter OR concept):   {total_combined}/{n_total} ({total_combined/n_total*100:.1f}%)")
    
    print("\n   By position - LETTER matches:")
    for z_name in ["z1", "z2", "z3", "z4", "z5", "z6"]:
        count = answer_letter_in_latent[z_name]
        pct = count / n_total * 100
        print(f"      {z_name}: {count} ({pct:.1f}%)")
    
    print("\n   By position - CONCEPT matches:")
    for z_name in ["z1", "z2", "z3", "z4", "z5", "z6"]:
        count = answer_concept_in_latent[z_name]
        pct = count / n_total * 100
        print(f"      {z_name}: {count} ({pct:.1f}%)")
    
    print("\n   By position - COMBINED (letter OR concept):")
    for z_name in ["z1", "z2", "z3", "z4", "z5", "z6"]:
        count = answer_combined_in_latent[z_name]
        pct = count / n_total * 100
        print(f"      {z_name}: {count} ({pct:.1f}%)")
    
    # NEW SECTION: Can Logit Lens predict what the model will OUTPUT?
    print("\n" + "=" * 70)
    print("KEY METRIC: Can Logit Lens predict model's output?")
    print("=" * 70)
    print("(This measures interpretability: can we read the model's decision from latents?)")
    
    total_predicted = sum(predicted_in_latent.values())
    print(f"\n   Model's PREDICTED letter found in z2/z3 top-5: {total_predicted}/{n_total} ({total_predicted/n_total*100:.1f}%)")
    print(f"   z2 top-1 matches model output: {z2_top1_matches_output}/{n_total} ({z2_top1_matches_output/n_total*100:.1f}%)")
    print(f"   z3 top-1 matches model output: {z3_top1_matches_output}/{n_total} ({z3_top1_matches_output/n_total*100:.1f}%)")
    
    print("\n   By position (where predicted letter first appears):")
    for z_name in ["z2", "z3"]:
        count = predicted_in_latent[z_name]
        pct = count / n_total * 100
        print(f"      {z_name}: {count} ({pct:.1f}%)")
    
    print(f"\n   Cross-tabulation (z2 top-1 prediction vs correctness):")
    print(f"      z2 matches output AND correct:   {z2_matches_and_correct:3d} ({z2_matches_and_correct/n_total*100:5.1f}%)")
    print(f"      z2 matches output AND incorrect: {z2_matches_and_incorrect:3d} ({z2_matches_and_incorrect/n_total*100:5.1f}%)")
    print(f"      z2 no match AND correct:         {z2_no_match_and_correct:3d} ({z2_no_match_and_correct/n_total*100:5.1f}%)")
    print(f"      z2 no match AND incorrect:       {z2_no_match_and_incorrect:3d} ({z2_no_match_and_incorrect/n_total*100:5.1f}%)")
    
    # Calculate conditional probabilities
    if z2_top1_matches_output > 0:
        p_correct_given_match = z2_matches_and_correct / z2_top1_matches_output * 100
    else:
        p_correct_given_match = 0
    no_match_total = z2_no_match_and_correct + z2_no_match_and_incorrect
    if no_match_total > 0:
        p_correct_given_no_match = z2_no_match_and_correct / no_match_total * 100
    else:
        p_correct_given_no_match = 0
    
    print(f"\n   Conditional probabilities:")
    print(f"      P(correct | z2 matches output):     {p_correct_given_match:.1f}%")
    print(f"      P(correct | z2 doesn't match):      {p_correct_given_no_match:.1f}%")
    print(f"      (baseline accuracy: {n_correct/n_total*100:.1f}%)")
    
    # What OTHER tokens appear in z2 top-5?
    total_other = sum(other_tokens_in_top5.values())
    total_letters = sum(letter_tokens_in_top5.values())
    print(f"\n   What else is in z2 top-5 (besides predicted letter)?")
    print(f"      Other choice letters (A-E): {total_letters} tokens")
    print(f"      Non-letter tokens:          {total_other} tokens")
    print(f"\n      Most common NON-LETTER tokens in z2 top-5:")
    for tok, count in other_tokens_in_top5.most_common(10):
        print(f"         '{tok}': {count}")
    print(f"\n      Competing LETTER tokens in z2 top-5:")
    for tok, count in letter_tokens_in_top5.most_common(10):
        print(f"         '{tok}': {count}")
    
    # Report entropy and margin if requested
    if args.report_entropy and z2_entropies:
        print("\n" + "=" * 70)
        print("ENTROPY & MARGIN ANALYSIS (commitment strength)")
        print("=" * 70)
        avg_z2_entropy = sum(z2_entropies) / len(z2_entropies)
        avg_z3_entropy = sum(z3_entropies) / len(z3_entropies)
        avg_z2_margin = sum(z2_margins) / len(z2_margins)
        avg_z3_margin = sum(z3_margins) / len(z3_margins)
        
        print(f"\n   Average entropy (lower = more confident):")
        print(f"      z2: {avg_z2_entropy:.3f} bits")
        print(f"      z3: {avg_z3_entropy:.3f} bits")
        print(f"\n   Average margin (top1 - top2, higher = more confident):")
        print(f"      z2: {avg_z2_margin:.3f}")
        print(f"      z3: {avg_z3_margin:.3f}")
        
        # Max entropy for 5 items is log2(5) = 2.32 bits
        print(f"\n   Interpretation:")
        print(f"      Max entropy for 5 items: 2.32 bits")
        print(f"      z2 is {'less' if avg_z2_entropy > avg_z3_entropy else 'more'} confident than z3")
    
    # Report shuffle ablation if requested
    if args.shuffle_choices and shuffle_total > 0:
        print("\n" + "=" * 70)
        print("SHUFFLE ABLATION (content vs position tracking)")
        print("=" * 70)
        print(f"\n   Total shuffled examples: {shuffle_total}")
        print(f"   Latent contains ORIGINAL answer letter: {shuffle_latent_tracks_original} ({shuffle_latent_tracks_original/shuffle_total*100:.1f}%)")
        print(f"   Latent contains SHUFFLED answer letter: {shuffle_latent_tracks_shuffled} ({shuffle_latent_tracks_shuffled/shuffle_total*100:.1f}%)")
        
        print(f"\n   Interpretation:")
        print(f"      If 'original' >> 'shuffled': latent tracks CONTENT (the actual answer)")
        print(f"      If 'shuffled' >> 'original': latent tracks POSITION (just the letter)")
        print(f"      If similar: latent tracks BOTH or neither clearly")

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
        "answer_in_latent": {
            "letter_matches": {
                "total": total_letter,
                "total_pct": total_letter / n_total,
                "by_position": dict(answer_letter_in_latent),
            },
            "concept_matches": {
                "total": total_concept,
                "total_pct": total_concept / n_total,
                "by_position": dict(answer_concept_in_latent),
            },
            "combined_matches": {
                "total": total_combined,
                "total_pct": total_combined / n_total,
                "by_position": dict(answer_combined_in_latent),
            },
        },
        "predicted_in_latent": {
            "total": total_predicted,
            "total_pct": total_predicted / n_total,
            "z2_top1_matches": z2_top1_matches_output,
            "z3_top1_matches": z3_top1_matches_output,
            "z2_top1_pct": z2_top1_matches_output / n_total,
            "z3_top1_pct": z3_top1_matches_output / n_total,
            "by_position": dict(predicted_in_latent),
            "cross_tab": {
                "z2_matches_and_correct": z2_matches_and_correct,
                "z2_matches_and_incorrect": z2_matches_and_incorrect,
                "z2_no_match_and_correct": z2_no_match_and_correct,
                "z2_no_match_and_incorrect": z2_no_match_and_incorrect,
            },
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_path}")

    # 6. Show some concept match examples
    print("\n6. Sample Concept Matches (where concept found but not letter):")
    concept_only_examples = [ex for ex in results if ex.get("concept_found_in") and not ex.get("letter_found_in")]
    for ex in concept_only_examples[:5]:
        print(f"   Expected: {ex['expected']} ('{ex.get('answer_concept', '')}'), "
              f"Concept in: {ex['concept_found_in']}, "
              f"z2 top: '{ex['latents'][1]['top1'] if len(ex['latents']) > 1 else 'N/A'}'")
    
    if not concept_only_examples:
        print("   (No examples where concept matched but letter didn't)")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"""
Compare these findings to math (from LessWrong):
- Math: z2 encodes Step 1 result (100%), z4 encodes Step 2 result (85%)
- Math: <|eocot|> typically appears in z5/z6 (uses 4-5 latent steps)

INTERPRETABILITY (can we read what model will output?):
- z2 top-1 predicts output: {z2_top1_matches_output/n_total*100:.1f}%
- z3 top-1 predicts output: {z3_top1_matches_output/n_total*100:.1f}%
- Predicted in z2/z3 top-5: {total_predicted/n_total*100:.1f}%

CORRECT answer encoding:
- Correct letter in latent: {total_letter}/{n_total} ({total_letter/n_total*100:.1f}%)
- Correct concept in latent: {total_concept}/{n_total} ({total_concept/n_total*100:.1f}%)
""")


if __name__ == "__main__":
    main()
