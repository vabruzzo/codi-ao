#!/usr/bin/env python3
"""
Evaluate Logit Lens for intermediate result extraction.

Tests whether the step1 result (from z2) and step2 result (from z4)
appear in the top-k tokens when projecting latents to vocabulary space.

This complements the operation detection eval and directly compares
to AO's result extraction capability.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_codi_model(config_path="configs/default.yaml"):
    """Load the CODI model."""
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


def get_lm_head(model):
    """Get the language model head from various model structures."""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    elif hasattr(model, 'codi') and hasattr(model.codi, 'lm_head'):
        return model.codi.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        return model.model.lm_head
    else:
        raise ValueError(f"Cannot find lm_head in model type {type(model)}")


def get_layer_norm(model):
    """Get the final layer norm from various model structures."""
    if hasattr(model, 'codi'):
        codi = model.codi
        if hasattr(codi, 'get_base_model'):
            base = codi.get_base_model()
        else:
            base = codi
        if hasattr(base, 'model') and hasattr(base.model, 'norm'):
            return base.model.norm
    
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    
    return None


def get_number_token_ids(tokenizer, number: int) -> list[int]:
    """
    Get all token IDs that could represent a number.
    Includes: "4", " 4", "04", "four", " four", etc.
    """
    token_ids = set()
    
    # String representations
    str_variants = [
        str(number),           # "4"
        f" {number}",          # " 4"
        f"{number} ",          # "4 " (with trailing space)
        f"0{number}" if number < 10 else None,  # "04"
        f" 0{number}" if number < 10 else None, # " 04"
    ]
    
    # Word representations for small numbers
    number_words = {
        0: ["zero", "Zero"],
        1: ["one", "One"],
        2: ["two", "Two"],
        3: ["three", "Three"],
        4: ["four", "Four"],
        5: ["five", "Five"],
        6: ["six", "Six"],
        7: ["seven", "Seven"],
        8: ["eight", "Eight"],
        9: ["nine", "Nine"],
        10: ["ten", "Ten"],
        11: ["eleven", "Eleven"],
        12: ["twelve", "Twelve"],
    }
    
    if number in number_words:
        for word in number_words[number]:
            str_variants.extend([word, f" {word}"])
    
    # Get token IDs for each variant
    for variant in str_variants:
        if variant is None:
            continue
        try:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            # Only include single-token representations
            if len(ids) == 1:
                token_ids.add(ids[0])
        except:
            pass
    
    return list(token_ids)


def analyze_latent_for_number(latent, lm_head, layer_norm, tokenizer, target_number, device, top_k=10):
    """
    Analyze whether the target number appears in top-k tokens.
    
    Returns:
    - Whether target appears in top-k
    - The rank if found (1-indexed), 0 if not found
    - Top-k tokens and probs
    - Probability mass on target number tokens
    """
    with torch.no_grad():
        latent = latent.to(device)
        
        # Apply layer norm before projection
        if layer_norm is not None:
            latent = layer_norm(latent.unsqueeze(0)).squeeze(0)
        
        # Project to logits
        if hasattr(lm_head, 'weight'):
            logits = latent @ lm_head.weight.T
        else:
            logits = lm_head(latent.unsqueeze(0)).squeeze(0)
        
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k tokens
        topk_probs, topk_ids = probs.topk(top_k)
        topk_tokens = [tokenizer.decode([tid]) for tid in topk_ids.tolist()]
        
        # Get target number token IDs
        target_token_ids = get_number_token_ids(tokenizer, target_number)
        target_token_ids_set = set(target_token_ids)
        
        # Check if target is in top-k
        found_in_topk = False
        rank = 0
        for r, tid in enumerate(topk_ids.tolist(), 1):
            if tid in target_token_ids_set:
                found_in_topk = True
                rank = r
                break
        
        # Get probability mass on target tokens
        if target_token_ids:
            target_prob = probs[target_token_ids].sum().item()
        else:
            target_prob = 0.0
        
        # Get top-1 prediction as a number (if possible)
        top1_token = topk_tokens[0].strip()
        try:
            top1_number = int(top1_token)
        except ValueError:
            top1_number = None
        
    return {
        "found_in_topk": found_in_topk,
        "rank": rank,
        "target_prob": target_prob,
        "top1_token": topk_tokens[0],
        "top1_number": top1_number,
        "top1_correct": top1_number == target_number,
        "topk_tokens": topk_tokens,
        "topk_probs": topk_probs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Logit Lens for result extraction")
    parser.add_argument("--data", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--output", type=str, default="results/logit_lens_results.json")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Logit Lens Result Extraction Evaluation")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    
    problems = data["problems"]
    if args.n_samples:
        problems = problems[:args.n_samples]
    
    print(f"Loaded {len(problems)} problems")
    
    # Load model
    print("\nLoading CODI model...")
    codi = load_codi_model()
    model = codi.model
    tokenizer = codi.tokenizer
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    device = next(lm_head.parameters()).device
    
    print(f"Layer norm: {'Found' if layer_norm else 'Not found'}")
    
    # Collect latents
    print(f"\nCollecting latents for {len(problems)} problems...")
    all_latents = []
    for problem in tqdm(problems, desc="Collecting latents"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        all_latents.append(result.latent_vectors)
    
    # Results storage
    results = {
        "config": {
            "n_samples": len(problems),
            "top_k": args.top_k,
            "layer_norm_applied": layer_norm is not None,
        },
        "step1_z2": {
            "total": 0,
            "in_topk": 0,
            "top1_correct": 0,
            "predictions": [],
        },
        "step2_z4": {
            "total": 0,
            "in_topk": 0,
            "top1_correct": 0,
            "predictions": [],
        },
    }
    
    print(f"\nAnalyzing result extraction (top-{args.top_k})...")
    
    for i, (latent_set, problem) in enumerate(tqdm(zip(all_latents, problems), 
                                                     total=len(problems), desc="Analyzing")):
        step1 = problem["step1"]
        step2 = problem["step2"]
        
        # Analyze z2 for step1
        z2 = latent_set[1]
        analysis_z2 = analyze_latent_for_number(z2, lm_head, layer_norm, tokenizer, 
                                                 step1, device, args.top_k)
        
        results["step1_z2"]["total"] += 1
        if analysis_z2["found_in_topk"]:
            results["step1_z2"]["in_topk"] += 1
        if analysis_z2["top1_correct"]:
            results["step1_z2"]["top1_correct"] += 1
        
        results["step1_z2"]["predictions"].append({
            "idx": i,
            "true": step1,
            "found_in_topk": analysis_z2["found_in_topk"],
            "rank": analysis_z2["rank"],
            "target_prob": analysis_z2["target_prob"],
            "top1_token": analysis_z2["top1_token"],
            "top1_number": analysis_z2["top1_number"],
            "top1_correct": analysis_z2["top1_correct"],
            "topk_tokens": analysis_z2["topk_tokens"][:5],  # Just top 5 for brevity
        })
        
        # Analyze z4 for step2
        z4 = latent_set[3]
        analysis_z4 = analyze_latent_for_number(z4, lm_head, layer_norm, tokenizer,
                                                 step2, device, args.top_k)
        
        results["step2_z4"]["total"] += 1
        if analysis_z4["found_in_topk"]:
            results["step2_z4"]["in_topk"] += 1
        if analysis_z4["top1_correct"]:
            results["step2_z4"]["top1_correct"] += 1
        
        results["step2_z4"]["predictions"].append({
            "idx": i,
            "true": step2,
            "found_in_topk": analysis_z4["found_in_topk"],
            "rank": analysis_z4["rank"],
            "target_prob": analysis_z4["target_prob"],
            "top1_token": analysis_z4["top1_token"],
            "top1_number": analysis_z4["top1_number"],
            "top1_correct": analysis_z4["top1_correct"],
            "topk_tokens": analysis_z4["topk_tokens"][:5],
        })
    
    # Compute summary stats
    for key in ["step1_z2", "step2_z4"]:
        d = results[key]
        d["in_topk_pct"] = 100 * d["in_topk"] / d["total"] if d["total"] > 0 else 0
        d["top1_accuracy"] = 100 * d["top1_correct"] / d["total"] if d["total"] > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n--- Step 1 Result (z2) ---")
    print(f"In top-{args.top_k}: {results['step1_z2']['in_topk_pct']:.1f}% ({results['step1_z2']['in_topk']}/{results['step1_z2']['total']})")
    print(f"Top-1 exact match: {results['step1_z2']['top1_accuracy']:.1f}% ({results['step1_z2']['top1_correct']}/{results['step1_z2']['total']})")
    
    print(f"\n--- Step 2 Result (z4) ---")
    print(f"In top-{args.top_k}: {results['step2_z4']['in_topk_pct']:.1f}% ({results['step2_z4']['in_topk']}/{results['step2_z4']['total']})")
    print(f"Top-1 exact match: {results['step2_z4']['top1_accuracy']:.1f}% ({results['step2_z4']['top1_correct']}/{results['step2_z4']['total']})")
    
    # Show some examples
    print("\n" + "-" * 60)
    print("Sample predictions (Step 1 / z2):")
    for pred in results["step1_z2"]["predictions"][:5]:
        status = "✓" if pred["found_in_topk"] else "✗"
        print(f"  True: {pred['true']}, Top tokens: {pred['topk_tokens']}, Found: {status} (rank={pred['rank']})")
    
    print("\nSample predictions (Step 2 / z4):")
    for pred in results["step2_z4"]["predictions"][:5]:
        status = "✓" if pred["found_in_topk"] else "✗"
        print(f"  True: {pred['true']}, Top tokens: {pred['topk_tokens']}, Found: {status} (rank={pred['rank']})")
    
    # Comparison to AO
    print("\n" + "=" * 60)
    print("COMPARISON TO ACTIVATION ORACLE")
    print("=" * 60)
    print(f"                    | Logit Lens (top-{args.top_k}) | AO (z2 only) | AO (all 6)")
    print(f"Step 1 (z2)         | {results['step1_z2']['in_topk_pct']:>5.1f}%              | 97.5%        | 99.5%")
    print(f"Step 2 (z4)         | {results['step2_z4']['in_topk_pct']:>5.1f}%              | 59.5%        | 69.5%")
    print("\n(AO numbers from prior experiments, using exact match)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
