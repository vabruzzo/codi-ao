#!/usr/bin/env python3
"""
Evaluate Logit Lens for operation type detection.

Outputs precise metrics for charting:
- Per-position accuracy
- Per-operation accuracy  
- Top-5 token analysis
- Confidence scores
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


# Operation token sets for logit lens
OPERATION_TOKENS = {
    "add": [
        "add", "Add", "ADD", "addition", "Addition", "plus", "Plus", "+",
        " add", " Add", " addition", " plus", " Plus", "added", "adding",
        "sum", "Sum", " sum", "total", " total",
    ],
    "sub": [
        "sub", "Sub", "subtract", "Subtract", "subtraction", "minus", "Minus", "-",
        " sub", " subtract", " minus", " subtraction", "removed", "removing",
        "difference", " difference", "less", " less",
    ],
    "mul": [
        "mul", "Mul", "multiply", "Multiply", "multiplication", "times", "Times", 
        "*", "×", " mul", " multiply", " times", " multiplication",
        "product", " product", "multiplied", "multiplying",
    ],
}


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
    # Try CODI structure first
    if hasattr(model, 'codi'):
        codi = model.codi
        if hasattr(codi, 'get_base_model'):
            base = codi.get_base_model()
        else:
            base = codi
        if hasattr(base, 'model') and hasattr(base.model, 'norm'):
            return base.model.norm
    
    # Try direct model structure
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    
    # Try transformer structure (GPT-2 style)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f
    
    return None


def get_operation_token_ids(tokenizer):
    """Get token IDs for operation-related tokens."""
    op_token_ids = {}
    for op, tokens in OPERATION_TOKENS.items():
        ids = []
        for tok in tokens:
            try:
                tok_ids = tokenizer.encode(tok, add_special_tokens=False)
                if len(tok_ids) == 1:
                    ids.append(tok_ids[0])
            except:
                pass
        op_token_ids[op] = list(set(ids))  # Dedupe
    return op_token_ids


def analyze_latent_for_operation(latent, lm_head, layer_norm, tokenizer, op_token_ids, device, top_k=10):
    """
    Analyze a single latent vector for operation type.
    
    Returns detailed metrics including:
    - Probability sum method: sum probs of operation tokens, predict argmax
    - Top-k method: check if any operation token appears in top-k
    """
    with torch.no_grad():
        latent = latent.to(device)
        
        # Apply layer norm before projection (proper logit lens)
        if layer_norm is not None:
            latent = layer_norm(latent.unsqueeze(0)).squeeze(0)
        
        # Project to logits
        if hasattr(lm_head, 'weight'):
            logits = latent @ lm_head.weight.T
        else:
            logits = lm_head(latent.unsqueeze(0)).squeeze(0)
        
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k tokens overall
        topk_probs, topk_ids = probs.topk(top_k)
        topk_ids_set = set(topk_ids.tolist())
        topk_tokens = [tokenizer.decode([tid]) for tid in topk_ids.tolist()]
        
        # Get operation probabilities (sum method)
        op_probs = {}
        op_max_token = {}
        op_max_prob = {}
        op_in_topk = {}  # New: which ops have tokens in top-k
        op_topk_rank = {}  # New: best rank for each op in top-k
        
        for op, token_ids in op_token_ids.items():
            if token_ids:
                op_probs_tensor = probs[token_ids]
                total_prob = op_probs_tensor.sum().item()
                max_idx = op_probs_tensor.argmax().item()
                max_prob = op_probs_tensor[max_idx].item()
                max_token = tokenizer.decode([token_ids[max_idx]])
                
                # Check if any token is in top-k
                tokens_in_topk = topk_ids_set.intersection(set(token_ids))
                op_in_topk[op] = len(tokens_in_topk) > 0
                
                # Find best rank in top-k (1-indexed, 0 means not in top-k)
                best_rank = 0
                for rank, tid in enumerate(topk_ids.tolist(), 1):
                    if tid in token_ids:
                        best_rank = rank
                        break
                op_topk_rank[op] = best_rank
            else:
                total_prob = 0.0
                max_prob = 0.0
                max_token = ""
                op_in_topk[op] = False
                op_topk_rank[op] = 0
            
            op_probs[op] = total_prob
            op_max_prob[op] = max_prob
            op_max_token[op] = max_token
        
        # Method 1: Probability sum (predict op with highest total prob)
        predicted_op_prob_sum = max(op_probs.keys(), key=lambda k: op_probs[k])
        confidence = op_probs[predicted_op_prob_sum]
        
        # Method 2: Top-k (predict op with best rank, or None if no op tokens in top-k)
        ops_with_rank = [(op, rank) for op, rank in op_topk_rank.items() if rank > 0]
        if ops_with_rank:
            predicted_op_topk = min(ops_with_rank, key=lambda x: x[1])[0]
        else:
            predicted_op_topk = None  # No operation token in top-k
        
        # Margin: difference between top and second (prob sum method)
        sorted_probs = sorted(op_probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
    return {
        # Probability sum method
        "predicted_op": predicted_op_prob_sum,
        "confidence": confidence,
        "margin": margin,
        "op_probs": op_probs,
        "op_max_token": op_max_token,
        "op_max_prob": op_max_prob,
        # Top-k method
        "predicted_op_topk": predicted_op_topk,
        "op_in_topk": op_in_topk,
        "op_topk_rank": op_topk_rank,
        "any_op_in_topk": any(op_in_topk.values()),
        # General
        "topk_tokens": topk_tokens,
        "topk_probs": topk_probs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Logit Lens for operation detection")
    parser.add_argument("--data", type=str, default="data/synthetic_problems.json", help="Input data")
    parser.add_argument("--output", type=str, default="results/logit_lens_operation.json")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit samples (default: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Logit Lens Operation Detection Evaluation")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    
    problems = data["problems"]
    if args.n_samples:
        problems = problems[:args.n_samples]
    
    print(f"Loaded {len(problems)} problems")
    
    # Count operations
    op_counts = defaultdict(int)
    for p in problems:
        op_counts[p["operation"]] += 1
    print(f"Operation distribution: {dict(op_counts)}")
    
    # Load model
    print("\nLoading CODI model...")
    codi = load_codi_model()
    model = codi.model
    tokenizer = codi.tokenizer
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    device = next(lm_head.parameters()).device
    
    if layer_norm is not None:
        print("Layer norm found - will apply before projection (proper logit lens)")
    else:
        print("WARNING: Layer norm not found - logit lens may be less accurate")
    
    # Get operation token IDs
    op_token_ids = get_operation_token_ids(tokenizer)
    print(f"Operation token counts: add={len(op_token_ids['add'])}, sub={len(op_token_ids['sub'])}, mul={len(op_token_ids['mul'])}")
    
    # Collect latents
    print(f"\nCollecting latents for {len(problems)} problems...")
    all_latents = []
    for problem in tqdm(problems, desc="Collecting latents"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        all_latents.append(result.latent_vectors)
    
    # Analyze z2 only (position index 1)
    # Note: We only evaluate z2 because the operation label refers to step1's operation,
    # and z2 encodes step1's result. Other positions (especially z4) encode step2's result
    # which uses a different operation, making the ground truth labels invalid for them.
    print("\nAnalyzing operation detection on z2 (step1's latent)...")
    
    pos = 1  # z2
    pos_name = "z2"
    top_k = 10  # For top-k method
    
    results = {
        "config": {
            "n_samples": len(problems),
            "data_seed": data["config"]["seed"],
            "operation_distribution": dict(op_counts),
            "position": pos_name,
            "layer_norm_applied": layer_norm is not None,
            "top_k": top_k,
        },
        "total": len(problems),
        # Method 1: Probability sum
        "prob_sum": {
            "correct": 0,
            "per_operation": {op: {"total": 0, "correct": 0} for op in ["add", "sub", "mul"]},
        },
        # Method 2: Top-k presence
        "topk": {
            "correct": 0,
            "any_op_in_topk": 0,  # How often ANY op token is in top-k
            "per_operation": {op: {"total": 0, "correct": 0, "in_topk": 0} for op in ["add", "sub", "mul"]},
        },
        "predictions": [],
        "avg_confidence": 0.0,
        "avg_margin": 0.0,
    }
    
    confidences = []
    margins = []
    
    print(f"\n--- Position {pos_name} ---")
    
    for i, (latent_set, problem) in enumerate(zip(all_latents, problems)):
        latent = latent_set[pos]
        true_op = problem["operation"]
        
        analysis = analyze_latent_for_operation(latent, lm_head, layer_norm, tokenizer, op_token_ids, device, top_k=top_k)
        
        # Method 1: Probability sum
        pred_op_prob = analysis["predicted_op"]
        is_correct_prob = pred_op_prob == true_op
        
        results["prob_sum"]["per_operation"][true_op]["total"] += 1
        if is_correct_prob:
            results["prob_sum"]["correct"] += 1
            results["prob_sum"]["per_operation"][true_op]["correct"] += 1
        
        # Method 2: Top-k
        pred_op_topk = analysis["predicted_op_topk"]
        is_correct_topk = pred_op_topk == true_op
        
        results["topk"]["per_operation"][true_op]["total"] += 1
        if is_correct_topk:
            results["topk"]["correct"] += 1
            results["topk"]["per_operation"][true_op]["correct"] += 1
        
        if analysis["any_op_in_topk"]:
            results["topk"]["any_op_in_topk"] += 1
        if analysis["op_in_topk"].get(true_op, False):
            results["topk"]["per_operation"][true_op]["in_topk"] += 1
        
        confidences.append(analysis["confidence"])
        margins.append(analysis["margin"])
        
        # Store prediction details
        prediction = {
            "idx": i,
            "true_op": true_op,
            "pred_op_prob_sum": pred_op_prob,
            "correct_prob_sum": is_correct_prob,
            "pred_op_topk": pred_op_topk,
            "correct_topk": is_correct_topk,
            "confidence": analysis["confidence"],
            "margin": analysis["margin"],
            "op_probs": analysis["op_probs"],
            "op_in_topk": analysis["op_in_topk"],
            "op_topk_rank": analysis["op_topk_rank"],
            "topk_tokens": analysis["topk_tokens"],
            "topk_probs": analysis["topk_probs"],
        }
        results["predictions"].append(prediction)
    
    # Compute aggregates
    total = results["total"]
    
    # Prob sum method
    results["prob_sum"]["accuracy"] = results["prob_sum"]["correct"] / total * 100
    for op in ["add", "sub", "mul"]:
        op_data = results["prob_sum"]["per_operation"][op]
        if op_data["total"] > 0:
            op_data["accuracy"] = op_data["correct"] / op_data["total"] * 100
        else:
            op_data["accuracy"] = 0.0
    
    # Top-k method
    results["topk"]["accuracy"] = results["topk"]["correct"] / total * 100
    results["topk"]["any_op_in_topk_pct"] = results["topk"]["any_op_in_topk"] / total * 100
    for op in ["add", "sub", "mul"]:
        op_data = results["topk"]["per_operation"][op]
        if op_data["total"] > 0:
            op_data["accuracy"] = op_data["correct"] / op_data["total"] * 100
            op_data["in_topk_pct"] = op_data["in_topk"] / op_data["total"] * 100
        else:
            op_data["accuracy"] = 0.0
            op_data["in_topk_pct"] = 0.0
    
    results["avg_confidence"] = sum(confidences) / len(confidences)
    results["avg_margin"] = sum(margins) / len(margins)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (z2 only)")
    print("=" * 60)
    
    print("\n--- Method 1: Probability Sum ---")
    print(f"Overall accuracy: {results['prob_sum']['accuracy']:.1f}%")
    print(f"Avg confidence: {results['avg_confidence']:.4f} (total prob on winning op tokens)")
    print(f"Avg margin: {results['avg_margin']:.4f}")
    for op in ["add", "sub", "mul"]:
        op_data = results["prob_sum"]["per_operation"][op]
        print(f"  {op}: {op_data['accuracy']:.1f}% ({op_data['correct']}/{op_data['total']})")
    
    print(f"\n--- Method 2: Top-{top_k} Token Check ---")
    print(f"Overall accuracy: {results['topk']['accuracy']:.1f}% (when op token in top-{top_k})")
    print(f"Any op token in top-{top_k}: {results['topk']['any_op_in_topk_pct']:.1f}%")
    for op in ["add", "sub", "mul"]:
        op_data = results["topk"]["per_operation"][op]
        print(f"  {op}: {op_data['accuracy']:.1f}% acc, {op_data['in_topk_pct']:.1f}% in top-{top_k}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
