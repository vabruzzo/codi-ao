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


def analyze_latent_for_operation(latent, lm_head, tokenizer, op_token_ids, device):
    """
    Analyze a single latent vector for operation type.
    
    Returns detailed metrics for charting.
    """
    with torch.no_grad():
        latent = latent.to(device)
        
        # Project to logits
        if hasattr(lm_head, 'weight'):
            logits = latent @ lm_head.weight.T
        else:
            logits = lm_head(latent.unsqueeze(0)).squeeze(0)
        
        probs = F.softmax(logits, dim=-1)
        
        # Get top-5 tokens overall
        top5_probs, top5_ids = probs.topk(5)
        top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]
        
        # Get operation probabilities
        op_probs = {}
        op_max_token = {}
        op_max_prob = {}
        
        for op, token_ids in op_token_ids.items():
            if token_ids:
                op_probs_tensor = probs[token_ids]
                total_prob = op_probs_tensor.sum().item()
                max_idx = op_probs_tensor.argmax().item()
                max_prob = op_probs_tensor[max_idx].item()
                max_token = tokenizer.decode([token_ids[max_idx]])
            else:
                total_prob = 0.0
                max_prob = 0.0
                max_token = ""
            
            op_probs[op] = total_prob
            op_max_prob[op] = max_prob
            op_max_token[op] = max_token
        
        # Determine predicted operation
        predicted_op = max(op_probs.keys(), key=lambda k: op_probs[k])
        confidence = op_probs[predicted_op]
        
        # Margin: difference between top and second
        sorted_probs = sorted(op_probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
    return {
        "predicted_op": predicted_op,
        "confidence": confidence,
        "margin": margin,
        "op_probs": op_probs,
        "op_max_token": op_max_token,
        "op_max_prob": op_max_prob,
        "top5_tokens": top5_tokens,
        "top5_probs": top5_probs.tolist(),
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
    device = next(lm_head.parameters()).device
    
    # Get operation token IDs
    op_token_ids = get_operation_token_ids(tokenizer)
    print(f"Operation token counts: add={len(op_token_ids['add'])}, sub={len(op_token_ids['sub'])}, mul={len(op_token_ids['mul'])}")
    
    # Collect latents
    print(f"\nCollecting latents for {len(problems)} problems...")
    all_latents = []
    for problem in tqdm(problems, desc="Collecting latents"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        all_latents.append(result.latent_vectors)
    
    # Analyze each position
    print("\nAnalyzing operation detection per position...")
    
    results = {
        "config": {
            "n_samples": len(problems),
            "data_seed": data["config"]["seed"],
            "operation_distribution": dict(op_counts),
        },
        "positions": {},
        "summary": {},
    }
    
    for pos in range(6):
        pos_name = f"z{pos+1}"
        print(f"\n--- Position {pos_name} ---")
        
        pos_results = {
            "position": pos,
            "position_name": pos_name,
            "total": len(problems),
            "correct": 0,
            "per_operation": {op: {"total": 0, "correct": 0, "predictions": []} for op in ["add", "sub", "mul"]},
            "predictions": [],
            "avg_confidence": 0.0,
            "avg_margin": 0.0,
        }
        
        confidences = []
        margins = []
        
        for i, (latent_set, problem) in enumerate(zip(all_latents, problems)):
            latent = latent_set[pos]
            true_op = problem["operation"]
            
            analysis = analyze_latent_for_operation(latent, lm_head, tokenizer, op_token_ids, device)
            
            pred_op = analysis["predicted_op"]
            is_correct = pred_op == true_op
            
            pos_results["per_operation"][true_op]["total"] += 1
            if is_correct:
                pos_results["correct"] += 1
                pos_results["per_operation"][true_op]["correct"] += 1
            
            confidences.append(analysis["confidence"])
            margins.append(analysis["margin"])
            
            # Store prediction details
            prediction = {
                "idx": i,
                "true_op": true_op,
                "pred_op": pred_op,
                "correct": is_correct,
                "confidence": analysis["confidence"],
                "margin": analysis["margin"],
                "op_probs": analysis["op_probs"],
                "top5_tokens": analysis["top5_tokens"],
                "top5_probs": analysis["top5_probs"],
            }
            pos_results["predictions"].append(prediction)
            pos_results["per_operation"][true_op]["predictions"].append(prediction)
        
        # Compute aggregates
        pos_results["accuracy"] = pos_results["correct"] / pos_results["total"] * 100
        pos_results["avg_confidence"] = sum(confidences) / len(confidences)
        pos_results["avg_margin"] = sum(margins) / len(margins)
        
        for op in ["add", "sub", "mul"]:
            op_data = pos_results["per_operation"][op]
            if op_data["total"] > 0:
                op_data["accuracy"] = op_data["correct"] / op_data["total"] * 100
            else:
                op_data["accuracy"] = 0.0
        
        results["positions"][pos_name] = pos_results
        
        # Print summary
        print(f"  Overall accuracy: {pos_results['accuracy']:.1f}%")
        print(f"  Avg confidence: {pos_results['avg_confidence']:.4f}")
        print(f"  Avg margin: {pos_results['avg_margin']:.4f}")
        for op in ["add", "sub", "mul"]:
            op_data = pos_results["per_operation"][op]
            print(f"  {op}: {op_data['accuracy']:.1f}% ({op_data['correct']}/{op_data['total']})")
    
    # Summary across positions
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = []
    for pos in range(6):
        pos_name = f"z{pos+1}"
        pos_data = results["positions"][pos_name]
        summary.append({
            "position": pos_name,
            "accuracy": pos_data["accuracy"],
            "avg_confidence": pos_data["avg_confidence"],
            "avg_margin": pos_data["avg_margin"],
            "add_acc": pos_data["per_operation"]["add"]["accuracy"],
            "sub_acc": pos_data["per_operation"]["sub"]["accuracy"],
            "mul_acc": pos_data["per_operation"]["mul"]["accuracy"],
        })
        print(f"{pos_name}: {pos_data['accuracy']:5.1f}% (add={pos_data['per_operation']['add']['accuracy']:.1f}%, sub={pos_data['per_operation']['sub']['accuracy']:.1f}%, mul={pos_data['per_operation']['mul']['accuracy']:.1f}%)")
    
    results["summary"] = summary
    
    # Find best position
    best_pos = max(summary, key=lambda x: x["accuracy"])
    print(f"\nBest position: {best_pos['position']} ({best_pos['accuracy']:.1f}%)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
