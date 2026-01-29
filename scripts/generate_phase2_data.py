#!/usr/bin/env python3
"""
Phase 2 Data Generation Script.

Generates the full training dataset with:
- QA examples (~64k, 16%)
- Classification examples (~336k, 84%)

Total: ~400k examples

Usage:
    python scripts/generate_phase2_data.py --n_prompts 50000 --verbose
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 training data")
    parser.add_argument("--n_prompts", type=int, default=50000, help="Number of prompts to generate")
    parser.add_argument("--output_dir", type=str, default="data/phase2", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="bcywinski/codi_llama1b-answer_only")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--qa_ratio", type=float, default=0.50, help="Ratio of prompts used for QA (higher because classification generates more per prompt)")
    parser.add_argument("--multi_latent_ratio", type=float, default=0.33, help="Ratio of multi-latent examples in QA")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 Data Generation")
    print("=" * 60)
    print(f"Target prompts: {args.n_prompts}")
    print(f"QA ratio: {args.qa_ratio:.0%}")
    print(f"Classification ratio: {1 - args.qa_ratio:.0%}")
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic prompts
    print(f"\nGenerating {args.n_prompts} synthetic prompts...")
    from scripts.collect_latents import create_synthetic_prompts
    prompts = create_synthetic_prompts(args.n_prompts, seed=args.seed)
    print(f"  Generated {len(prompts)} prompts")
    
    # Load CODI model
    print(f"\nLoading CODI model...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Base model: {args.base_model}")
    
    from src.codi_wrapper import CODIWrapper
    wrapper = CODIWrapper.from_pretrained(
        checkpoint_path=args.checkpoint,
        model_name_or_path=args.base_model,
        device=args.device,
    )
    print(f"  Hidden size: {wrapper.hidden_size}")
    
    # Get placeholder token
    from src.activation_oracle import DEFAULT_PLACEHOLDER_TOKEN
    placeholder_token = DEFAULT_PLACEHOLDER_TOKEN
    
    # Split prompts for QA vs Classification
    n_qa = int(len(prompts) * args.qa_ratio)
    n_class = len(prompts) - n_qa
    
    qa_prompts = prompts[:n_qa]
    class_prompts = prompts[n_qa:]
    
    print(f"\nData split:")
    print(f"  QA prompts: {len(qa_prompts)}")
    print(f"  Classification prompts: {len(class_prompts)}")
    
    # Generate QA examples
    print(f"\n{'=' * 60}")
    print("Generating QA examples...")
    print("=" * 60)
    
    from src.activation_oracle import format_oracle_prompt
    from src.datasets.latent_qa import INTERMEDIATE_RESULT_TEMPLATES, MULTI_LATENT_TEMPLATES
    
    qa_examples = []
    qa_single = 0
    qa_multi = 0
    
    iterator = tqdm(qa_prompts, desc="QA data") if args.verbose else qa_prompts
    
    for item in iterator:
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=str(item["final_answer"]),
        )
        
        if len(result.latent_vectors) < 6:
            continue
        
        step1_gt = str(item["step1_result"])
        step2_gt = str(item["step2_result"])
        final_gt = str(item["final_answer"])
        
        def vec_to_list(v):
            return v.cpu().tolist() if hasattr(v, 'cpu') else v
        
        use_multi = random.random() < args.multi_latent_ratio
        
        if use_multi:
            # Multi-latent example
            all_vectors = [vec_to_list(result.latent_vectors[i]) for i in range(6)]
            question = random.choice(MULTI_LATENT_TEMPLATES)
            
            if "first step" in question.lower():
                answer = step1_gt
            elif "second" in question.lower():
                answer = step2_gt
            elif "final" in question.lower() or "answer" in question.lower():
                answer = final_gt
            elif "error" in question.lower() or "correct" in question.lower():
                answer = "Yes, the reasoning is correct."
            else:
                answer = f"Step 1: {step1_gt}, Step 2: {step2_gt}, Final: {final_gt}"
            
            oracle_prompt = format_oracle_prompt(
                question=question,
                num_activations=6,
                placeholder_token=placeholder_token,
                multi_latent=True,
            )
            
            qa_examples.append({
                "prompt": oracle_prompt,
                "latent_vectors": all_vectors,
                "latent_positions": [0, 1, 2, 3, 4, 5],
                "question": question,
                "answer": answer,
                "source_prompt": item["prompt"],
                "question_type": "full_reasoning",
                "is_multi_latent": True,
                "task": "qa",
            })
            qa_multi += 1
        else:
            # Single-latent examples: z2 and z4
            for lat_pos, step_gt in [(1, step1_gt), (3, step2_gt)]:
                vec = result.latent_vectors[lat_pos]
                question = random.choice(INTERMEDIATE_RESULT_TEMPLATES)
                oracle_prompt = format_oracle_prompt(
                    question=question,
                    num_activations=1,
                    placeholder_token=placeholder_token,
                )
                
                qa_examples.append({
                    "prompt": oracle_prompt,
                    "latent_vectors": [vec_to_list(vec)],
                    "latent_positions": [lat_pos],
                    "question": question,
                    "answer": step_gt,
                    "source_prompt": item["prompt"],
                    "question_type": "intermediate_result",
                    "is_multi_latent": False,
                    "task": "qa",
                })
                qa_single += 1
    
    print(f"\nQA examples: {len(qa_examples)}")
    print(f"  Single-latent: {qa_single}")
    print(f"  Multi-latent: {qa_multi}")
    
    # Generate Classification examples
    print(f"\n{'=' * 60}")
    print("Generating Classification examples...")
    print("=" * 60)
    
    from src.datasets.classification import CLASSIFICATION_TASKS
    
    class_examples = []
    yes_count = 0
    no_count = 0
    
    iterator = tqdm(class_prompts, desc="Classification data") if args.verbose else class_prompts
    
    for item in iterator:
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=str(item["final_answer"]),
        )
        
        if len(result.latent_vectors) < 6:
            continue
        
        cot_steps = [f"{item['step1_result']}", f"{item['step2_result']}"]
        results = [str(item["step1_result"]), str(item["step2_result"])]
        
        def vec_to_list(v):
            return v.cpu().tolist() if hasattr(v, 'cpu') else v
        
        # Generate examples for z2 and z4
        for lat_pos, step_idx in [(1, 0), (3, 1)]:
            latent_vec = result.latent_vectors[lat_pos]
            cot_step = cot_steps[step_idx] if step_idx < len(cot_steps) else ""
            step_result = results[step_idx] if step_idx < len(results) else ""
            
            # Try each classification task
            for task_name, task in CLASSIFICATION_TASKS.items():
                condition = task["condition"]
                
                try:
                    is_true = condition(cot_step, step_result, lat_pos)
                except Exception:
                    continue
                
                answer = "Yes" if is_true else "No"
                
                # Balance answers
                if answer == "Yes" and yes_count > no_count + 1000:
                    continue
                if answer == "No" and no_count > yes_count + 1000:
                    continue
                
                question = random.choice(task["questions"])
                oracle_prompt = format_oracle_prompt(
                    question=question,
                    num_activations=1,
                    placeholder_token=placeholder_token,
                )
                
                class_examples.append({
                    "prompt": oracle_prompt,
                    "latent_vectors": [vec_to_list(latent_vec)],
                    "latent_positions": [lat_pos],
                    "question": question,
                    "answer": answer,
                    "source_prompt": item["prompt"],
                    "question_type": task["type"],
                    "is_multi_latent": False,
                    "task": "classification",
                })
                
                if answer == "Yes":
                    yes_count += 1
                else:
                    no_count += 1
    
    print(f"\nClassification examples: {len(class_examples)}")
    print(f"  Yes: {yes_count}")
    print(f"  No: {no_count}")
    
    # Combine and shuffle
    print(f"\n{'=' * 60}")
    print("Combining datasets...")
    print("=" * 60)
    
    all_examples = qa_examples + class_examples
    random.shuffle(all_examples)
    
    print(f"Total examples: {len(all_examples)}")
    print(f"  QA: {len(qa_examples)} ({len(qa_examples)/len(all_examples)*100:.1f}%)")
    print(f"  Classification: {len(class_examples)} ({len(class_examples)/len(all_examples)*100:.1f}%)")
    
    # Save to JSONL
    output_path = output_dir / "train.jsonl"
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    
    # Save metadata
    metadata = {
        "n_prompts": args.n_prompts,
        "total_examples": len(all_examples),
        "qa_examples": len(qa_examples),
        "qa_single": qa_single,
        "qa_multi": qa_multi,
        "classification_examples": len(class_examples),
        "classification_yes": yes_count,
        "classification_no": no_count,
        "seed": args.seed,
        "qa_ratio": args.qa_ratio,
        "multi_latent_ratio": args.multi_latent_ratio,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    
    print(f"\n{'=' * 60}")
    print("PHASE 2 DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total examples: {len(all_examples)}")


if __name__ == "__main__":
    main()
