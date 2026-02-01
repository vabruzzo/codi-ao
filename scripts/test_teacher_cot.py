#!/usr/bin/env python3
"""
Test CODI's teacher mode to see what chain-of-thought output looks like.
This helps us understand how to parse it for AO training ground truth.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codi_wrapper import CODIWrapper


def main():
    print("Loading CODI model...")
    codi = CODIWrapper.from_pretrained()
    
    # Test prompts - our 3-step math problems
    test_prompts = [
        "A baker starts with 4 cups of flour. She adds 5 more cups. Then she triples the total amount. Finally, she combines the tripled amount with the original total. How many cups does she have?",
        "A store has 6 items. They receive 3 more items. Then each item is packaged into groups of 2. Finally, they combine all packaged groups with the unpacked items. What is the total count?",
        "Tom has 8 apples. He gives away 3 apples. Then he multiplies his remaining apples by 4. Finally, he adds the multiplied amount to his original remaining count. How many apples does he have now?",
    ]
    
    print("\n" + "=" * 70)
    print("TESTING TEACHER MODE (verbalize_cot=True)")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Problem {i+1} ---")
        print(f"Prompt: {prompt[:100]}...")
        
        try:
            # Run teacher task
            full_cot, steps, results = codi.run_teacher_task(prompt, max_new_tokens=512)
            
            print(f"\nFull CoT output:")
            print("-" * 40)
            print(full_cot)
            print("-" * 40)
            
            print(f"\nParsed steps: {steps}")
            print(f"Parsed results: {results}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Also test with a simple arithmetic problem
    print("\n" + "=" * 70)
    print("TESTING WITH SIMPLE ARITHMETIC")
    print("=" * 70)
    
    simple_prompts = [
        "What is 7 + 5?",
        "Calculate: 12 * 3 = ?",
        "If you have 20 apples and give away 8, then double what remains, how many do you have?",
    ]
    
    for i, prompt in enumerate(simple_prompts):
        print(f"\n--- Simple {i+1} ---")
        print(f"Prompt: {prompt}")
        
        try:
            full_cot, steps, results = codi.run_teacher_task(prompt, max_new_tokens=256)
            
            print(f"Full CoT: {full_cot}")
            print(f"Parsed steps: {steps}")
            print(f"Parsed results: {results}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Compare teacher vs student mode
    print("\n" + "=" * 70)
    print("COMPARING TEACHER vs STUDENT MODE")
    print("=" * 70)
    
    test_prompt = "A farmer has 5 chickens. He buys 3 more. Then each chicken lays 2 eggs. How many eggs are there in total?"
    
    print(f"\nPrompt: {test_prompt}")
    
    # Teacher mode
    print("\n[TEACHER MODE - explicit CoT]")
    try:
        full_cot, steps, results = codi.run_teacher_task(test_prompt, max_new_tokens=256)
        print(f"Output: {full_cot}")
        print(f"Steps: {steps}")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Student mode (latent CoT)
    print("\n[STUDENT MODE - latent CoT]")
    try:
        latent_result = codi.collect_latents(test_prompt, return_hidden_states=False)
        print(f"Output: {latent_result.predicted_answer}")
        print(f"Num latents: {len(latent_result.latent_vectors)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
