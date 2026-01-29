#!/usr/bin/env python3
"""
Test CODI model on CommonsenseQA examples.

This tests whether latent vectors encode meaningful information
for non-math reasoning tasks.

Usage:
    python scripts/test_commonsense.py
    python scripts/test_commonsense.py --ao_path checkpoints/ao
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


# CommonsenseQA examples (from CODI paper and similar format)
COMMONSENSE_EXAMPLES = [
    {
        "question": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Choices: A: ignore B: enforce C: authoritarian D: yell at E: avoid",
        "answer": "A",
        "reasoning": "Sanctions undermine/dismiss efforts, so 'ignore' fits best",
    },
    {
        "question": "Where would you put a plate after cleaning it? Choices: A: refrigerator B: table C: cupboard D: dishwasher E: restaurant",
        "answer": "C",
        "reasoning": "Clean plates go in cupboard for storage",
    },
    {
        "question": "What do people aim to do at work? Choices: A: complete job B: learn from each other C: kill animals D: have fun E: make money",
        "answer": "E",
        "reasoning": "Primary aim of work is to earn money",
    },
    {
        "question": "Where is a business card likely to be filed? Choices: A: wallet B: pocket C: briefcase D: desk drawer E: rolodex",
        "answer": "E",
        "reasoning": "Business cards are typically filed in a rolodex",
    },
    {
        "question": "What is the result of having to wait a long time for something and not being able to get it immediately? Choices: A: frustration B: irritation C: annoyance D: bothered E: anger",
        "answer": "A",
        "reasoning": "Waiting and not getting something leads to frustration",
    },
    {
        "question": "If you are prone to being deceptive, you would be thought of as what? Choices: A: untrustworthy B: unethical C: sneaky D: devious E: dishonest",
        "answer": "A",
        "reasoning": "Deceptive people are considered untrustworthy",
    },
    {
        "question": "What might a person see at the scene of a car accident? Choices: A: injuries B: traffic C: damage D: glass E: fire",
        "answer": "C",
        "reasoning": "Damage is most common at accident scenes",
    },
    {
        "question": "The artist was sitting quietly pondering, then suddenly they got inspired and began to what? Choices: A: paint B: think C: create art D: write E: meditate",
        "answer": "C",
        "reasoning": "Artists who get inspired begin to create art",
    },
]


def format_prompt(example: dict) -> str:
    """Format a CommonsenseQA example as a prompt."""
    return f"{example['question']} Give the answer only and nothing else."


def main():
    parser = argparse.ArgumentParser(description="Test CODI on CommonsenseQA")
    parser.add_argument("--ao_path", type=str, default=None, help="Path to trained AO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_examples", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("CODI CommonsenseQA Test")
    print("=" * 60)

    # Load CODI
    print("\nLoading CODI model...")
    from src.codi_wrapper import CODIWrapper

    wrapper = CODIWrapper.from_pretrained(device=args.device)

    # Load AO if provided
    ao = None
    if args.ao_path:
        print(f"\nLoading Activation Oracle from {args.ao_path}...")
        from src.activation_oracle import ActivationOracle, AOConfig

        config = AOConfig(device=args.device)
        ao = ActivationOracle.from_pretrained(
            config=config,
            lora_path=args.ao_path,
        )
        ao.eval_mode()

    # Test examples
    print(f"\nTesting {min(args.n_examples, len(COMMONSENSE_EXAMPLES))} CommonsenseQA examples...")
    print("=" * 60)

    for i, example in enumerate(COMMONSENSE_EXAMPLES[: args.n_examples]):
        prompt = format_prompt(example)
        expected = example["answer"]

        print(f"\n[Example {i + 1}]")
        print(f"Q: {example['question'][:80]}...")
        print(f"Expected: {expected}")
        print(f"Reasoning: {example['reasoning']}")

        # Collect latents
        result = wrapper.collect_latents(prompt, ground_truth_answer=expected)

        print(f"CODI predicted: {result.predicted_answer}")
        print(f"Correct: {result.is_correct}")

        # Logit lens on each latent vector
        print("\nLogit Lens (top-3 tokens per latent):")
        for j, lv in enumerate(result.latent_vectors):
            lens = wrapper.logit_lens(lv, top_k=5)
            top_tokens = lens.layer_results[0]["top_tokens"][:3]
            top_probs = lens.layer_results[0]["top_probs"][:3]
            tokens_str = ", ".join(
                [f"'{t}'({p:.2f})" for t, p in zip(top_tokens, top_probs)]
            )
            print(f"  z{j + 1}: {tokens_str}")

        # AO test if available
        if ao is not None:
            print("\nActivation Oracle queries:")

            # Test on z2 and z4 (where math results are typically stored)
            for pos_idx, pos_name in [(1, "z2"), (3, "z4")]:
                if pos_idx < len(result.latent_vectors):
                    latent_vec = result.latent_vectors[pos_idx]

                    # Ask what's stored
                    ao_prompt = ao.create_prompt(
                        question="What information is stored here?",
                        activation_vectors=[latent_vec],
                    )
                    ao_response = ao.generate(ao_prompt, max_new_tokens=50)
                    print(f"  {pos_name} - 'What is stored?': {ao_response[:60]}...")

            # Multi-latent test
            if len(result.latent_vectors) >= 6:
                all_latents = [v for v in result.latent_vectors[:6]]
                ao_prompt = ao.create_prompt(
                    question="What is the reasoning process?",
                    activation_vectors=all_latents,
                )
                ao_response = ao.generate(ao_prompt, max_new_tokens=100)
                print(f"  All latents - 'Reasoning?': {ao_response[:80]}...")

        print("-" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key observations to look for:
1. Do latent vectors encode the answer choice (A, B, C, etc.)?
2. Do they encode key reasoning concepts?
3. Is the pattern consistent across examples?
4. How does this compare to math (where z2/z4 encode numbers)?

Note: The CODI model was trained on CommonsenseQA, so it should
produce meaningful latent representations. However, the LessWrong
analysis only examined math - this is exploratory territory!
""")


if __name__ == "__main__":
    main()
