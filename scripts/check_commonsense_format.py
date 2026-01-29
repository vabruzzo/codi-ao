#!/usr/bin/env python3
"""Quick script to check CommonsenseQA dataset formats."""

from datasets import load_dataset

print("=" * 60)
print("Checking zen-E/CommonsenseQA-GPT4omini (CODI's training data)")
print("=" * 60)

try:
    ds = load_dataset("zen-E/CommonsenseQA-GPT4omini")
    print(f"\nSplits available: {list(ds.keys())}")
    
    # Get first example from validation or train
    split = "validation" if "validation" in ds else "train"
    example = ds[split][0]
    
    print(f"\nFields: {list(example.keys())}")
    print(f"\nExample from {split}:")
    for key, value in example.items():
        print(f"\n{key}:")
        print(f"  {str(value)[:500]}...")
        
except Exception as e:
    print(f"Error loading zen-E dataset: {e}")

print("\n" + "=" * 60)
print("Checking standard commonsense_qa")
print("=" * 60)

try:
    ds = load_dataset("commonsense_qa", split="validation")
    example = ds[0]
    
    print(f"\nFields: {list(example.keys())}")
    print(f"\nExample:")
    for key, value in example.items():
        print(f"\n{key}:")
        print(f"  {value}")
        
except Exception as e:
    print(f"Error loading commonsense_qa: {e}")
