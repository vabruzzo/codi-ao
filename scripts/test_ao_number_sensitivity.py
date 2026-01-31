#!/usr/bin/env python3
"""
Test: Does the AO output change when we modify numbers in the prompt?

This is a controlled test:
1. Take a problem template
2. Generate variants with different numbers
3. Collect latents from CODI for each variant
4. Ask the AO the same questions
5. Verify AO outputs track the number changes

If the AO correctly tracks number changes → it's reading from latents
If the AO gives same/random outputs → something is wrong
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_codi_model(config_path="configs/default.yaml"):
    from src.codi_wrapper import CODIWrapper
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    codi = CODIWrapper.from_pretrained(
        checkpoint_path=config["model"]["codi_checkpoint"],
        model_name_or_path=config["model"]["codi_base_model"],
        lora_r=config["model"]["codi_lora_r"],
        lora_alpha=config["model"]["codi_lora_alpha"],
        num_latent=config["model"]["codi_num_latent"],
        use_prj=config["model"]["codi_use_prj"],
        device=device,
    )
    return codi


def load_ao_model(checkpoint_dir: str):
    from src.activation_oracle import ActivationOracle, AOConfig
    
    config = AOConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_r=64,
        lora_alpha=128,
    )
    
    ao = ActivationOracle.from_pretrained(config=config, lora_path=checkpoint_dir)
    ao.eval_mode()
    return ao


def ao_generate(ao, question: str, latent_vectors: list, max_new_tokens: int = 20) -> str:
    vectors = []
    for v in latent_vectors:
        if isinstance(v, torch.Tensor):
            vectors.append(v)
        else:
            vectors.append(torch.tensor(v))
    
    prompt = ao.create_prompt(question=question, activation_vectors=vectors)
    return ao.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0)


def extract_number(response: str) -> int | None:
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return int(numbers[0])
    return None


# Problem templates with placeholders
TEMPLATES = [
    {
        "template": "A store has {X} items. They sell {Y} of them. Then they restock by multiplying the remaining by {Z}. How many items are there now?",
        "operation": "sub",
        "calc_step1": lambda X, Y, Z: X - Y,
        "calc_step2": lambda X, Y, Z: (X - Y) * Z,
    },
    {
        "template": "There are {X} birds. {Y} more arrive. Then the count is multiplied by {Z}. What is the total?",
        "operation": "add",
        "calc_step1": lambda X, Y, Z: X + Y,
        "calc_step2": lambda X, Y, Z: (X + Y) * Z,
    },
    {
        "template": "A farmer has {X} animals. The count is multiplied by {Y}, then {Z} are added. What is the result?",
        "operation": "mul",
        "calc_step1": lambda X, Y, Z: X * Y,
        "calc_step2": lambda X, Y, Z: X * Y + Z,
    },
]

# Number variants to test (X, Y, Z)
VARIANTS = [
    (5, 2, 3),
    (10, 4, 2),
    (8, 3, 5),
    (15, 7, 4),
    (6, 1, 8),
]


def main():
    parser = argparse.ArgumentParser(description="Test AO sensitivity to number changes")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ao_all6")
    args = parser.parse_args()
    
    print("=" * 70)
    print("AO Number Sensitivity Test")
    print("=" * 70)
    print("\nThis test verifies the AO outputs change when we modify numbers in the prompt.")
    print("Same prompt structure, different numbers → should get different AO outputs.\n")
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading CODI model...")
    codi = load_codi_model()
    
    print("Loading Activation Oracle...")
    ao = load_ao_model(args.checkpoint)
    
    # Questions to ask
    questions = {
        "step1_result": "What was calculated in the first step?",
        "operation": "What operation was performed in step 1?",
        "first_operand": "What was the first number in the calculation?",
    }
    
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    
    total_tests = 0
    correct_tracking = 0
    
    for template_info in TEMPLATES:
        template = template_info["template"]
        op = template_info["operation"]
        calc_step1 = template_info["calc_step1"]
        
        print(f"\n--- Template: {op} ---")
        print(f"Template: {template[:60]}...")
        print()
        
        for X, Y, Z in VARIANTS:
            # Generate prompt
            prompt = template.format(X=X, Y=Y, Z=Z)
            step1 = calc_step1(X, Y, Z)
            
            # Collect latents
            result = codi.collect_latents(prompt, return_hidden_states=False)
            if len(result.latent_vectors) < 6:
                print(f"  Skipping X={X}, Y={Y}, Z={Z} - not enough latents")
                continue
            
            all_latents = result.latent_vectors[:6]
            
            # Test step1 result extraction
            response = ao_generate(ao, questions["step1_result"], all_latents)
            pred_step1 = extract_number(response)
            step1_correct = (pred_step1 == step1)
            
            # Test first operand extraction
            response_op = ao_generate(ao, questions["first_operand"], all_latents)
            pred_X = extract_number(response_op)
            X_correct = (pred_X == X)
            
            total_tests += 2
            if step1_correct:
                correct_tracking += 1
            if X_correct:
                correct_tracking += 1
            
            status_step1 = "✓" if step1_correct else "✗"
            status_X = "✓" if X_correct else "✗"
            
            print(f"  X={X}, Y={Y}, Z={Z} → step1={step1}")
            print(f"    Step1 result: pred={pred_step1} {status_step1}")
            print(f"    First operand: pred={pred_X} {status_X}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tracking_rate = 100 * correct_tracking / total_tests if total_tests > 0 else 0
    print(f"Total tests: {total_tests}")
    print(f"Correct tracking: {correct_tracking} ({tracking_rate:.1f}%)")
    
    if tracking_rate > 80:
        print("\n✓ AO correctly tracks number changes - it's reading from the latents!")
    elif tracking_rate > 50:
        print("\n~ AO partially tracks number changes - some information is encoded")
    else:
        print("\n✗ AO does not track number changes well - potential issue")
    
    # Detailed breakdown
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("If tracking rate is high → AO is genuinely extracting from latents")
    print("If tracking rate is low → AO may be confabulating or memorizing patterns")


if __name__ == "__main__":
    main()
