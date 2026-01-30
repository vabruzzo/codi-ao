#!/usr/bin/env python3
"""
Phase 0: Exploratory Logit Lens Analysis

Analyze ALL 6 CODI latent positions to discover what information is
linearly decodable from each position.

Tests:
1. Numeric extraction - does argmax give a meaningful number?
2. Operation tokens - are add/sub/mul tokens high probability?
3. Top-k tokens - what are the highest probability tokens overall?

This runs BEFORE training to understand what's in CODI's latent space.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Synthetic Problem Generation
# =============================================================================

ENTITIES = [
    ("team", "members", "recruits", "trains"),
    ("company", "employees", "hires", "promotes"),
    ("school", "students", "enrolls", "graduates"),
    ("farm", "animals", "buys", "sells"),
    ("store", "items", "stocks", "sells"),
]

TEMPLATES_ADD_MUL = [
    "A {entity} has {X} {things}. They {action1} {Y} more. Then they multiply the total by {Z}. How many {things} are there now?",
    "A {entity} starts with {X} {things}. They {action1} {Y} additional {things}. Then each {thing} produces {Z} more. How many {things} total?",
]

TEMPLATES_SUB_MUL = [
    "A {entity} has {X} {things}. They lose {Y} of them. Then each remaining {thing} produces {Z} offspring. How many {things} total?",
    "A {entity} starts with {X} {things}. After {action2}ing {Y}, each remaining {thing} is multiplied by {Z}. How many {things}?",
]


def generate_problem(seed=None):
    """Generate a synthetic math problem with known intermediate results."""
    if seed is not None:
        random.seed(seed)
    
    entity, things, action1, action2 = random.choice(ENTITIES)
    thing = things[:-1] if things.endswith('s') else things
    
    X = random.randint(1, 10)
    Y = random.randint(1, 10)
    Z = random.randint(1, 10)
    
    # Decide operation type
    if random.random() < 0.5:
        template = random.choice(TEMPLATES_ADD_MUL)
        step1 = X + Y
        operation = "add"
    else:
        template = random.choice(TEMPLATES_SUB_MUL)
        # Ensure X > Y for subtraction
        if X <= Y:
            X, Y = Y + 1, X
        step1 = X - Y
        operation = "sub"
    
    step2 = step1 * Z
    final = step1 + step2  # This is what CODI computes but doesn't store
    
    prompt = template.format(
        entity=entity,
        things=things,
        thing=thing,
        action1=action1,
        action2=action2,
        X=X, Y=Y, Z=Z
    )
    
    return {
        "prompt": prompt,
        "X": X, "Y": Y, "Z": Z,
        "step1": step1,
        "step2": step2,
        "final": final,
        "operation": operation,
    }


# =============================================================================
# CODI Model Wrapper
# =============================================================================

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


# =============================================================================
# Logit Lens Analysis
# =============================================================================

def get_top_tokens(latent, model, tokenizer, k=20):
    """Project latent to vocab space and get top-k tokens."""
    # Get the language model head (unembedding matrix)
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        lm_head = model.get_output_embeddings()
    else:
        raise ValueError("Cannot find lm_head")
    
    # Project latent to logits: (hidden_dim,) @ (hidden_dim, vocab_size) -> (vocab_size,)
    with torch.no_grad():
        if hasattr(lm_head, 'weight'):
            logits = latent @ lm_head.weight.T
        else:
            logits = lm_head(latent.unsqueeze(0)).squeeze(0)
        
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(k)
    
    results = []
    for prob, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
        token_str = tokenizer.decode([tok_id])
        results.append({
            "token": token_str,
            "token_id": tok_id,
            "prob": prob,
        })
    
    return results


def check_numeric_tokens(top_tokens, target_number):
    """Check if the target number appears in top tokens."""
    target_str = str(target_number)
    
    for i, tok in enumerate(top_tokens):
        # Check various representations of the number
        tok_str = tok["token"].strip()
        if tok_str == target_str or tok_str == f" {target_str}":
            return {
                "found": True,
                "rank": i + 1,
                "prob": tok["prob"],
                "token": tok["token"],
            }
    
    return {"found": False, "rank": None, "prob": 0.0, "token": None}


def check_operation_tokens(latent, model, tokenizer):
    """Check probability mass on operation-related tokens."""
    # Get logits
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    else:
        lm_head = model.get_output_embeddings()
    
    with torch.no_grad():
        if hasattr(lm_head, 'weight'):
            logits = latent @ lm_head.weight.T
        else:
            logits = lm_head(latent.unsqueeze(0)).squeeze(0)
        probs = F.softmax(logits, dim=-1)
    
    # Define operation token sets
    operation_tokens = {
        "add": ["add", "Add", "ADD", "addition", "plus", "Plus", "+", " add", " Add", " plus", " Plus"],
        "sub": ["sub", "Sub", "subtract", "Subtract", "minus", "Minus", "-", " sub", " subtract", " minus"],
        "mul": ["mul", "Mul", "multiply", "Multiply", "times", "Times", "*", "×", " mul", " multiply", " times"],
    }
    
    results = {}
    for op_name, op_words in operation_tokens.items():
        total_prob = 0.0
        found_tokens = []
        
        for word in op_words:
            try:
                tok_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(tok_ids) == 1:
                    prob = probs[tok_ids[0]].item()
                    if prob > 1e-6:
                        total_prob += prob
                        found_tokens.append((word, prob))
            except:
                pass
        
        results[op_name] = {
            "total_prob": total_prob,
            "tokens": sorted(found_tokens, key=lambda x: -x[1])[:5],
        }
    
    # Which operation has highest probability?
    best_op = max(results.keys(), key=lambda k: results[k]["total_prob"])
    results["predicted"] = best_op
    results["confidence"] = results[best_op]["total_prob"]
    
    return results


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_latent_position(latents, position, model, tokenizer, problems):
    """Analyze what a specific latent position encodes across all problems."""
    results = {
        "position": position,
        "numeric_step1": {"correct": 0, "total": 0, "avg_rank": 0, "ranks": []},
        "numeric_step2": {"correct": 0, "total": 0, "avg_rank": 0, "ranks": []},
        "operation": {"correct": 0, "total": 0, "predictions": []},
        "top_tokens_examples": [],
    }
    
    for i, (latent_set, problem) in enumerate(zip(latents, problems)):
        latent = latent_set[position]  # Get this position's latent
        
        # Get top tokens
        top_tokens = get_top_tokens(latent, model, tokenizer, k=20)
        
        # Check for step1 value
        step1_check = check_numeric_tokens(top_tokens, problem["step1"])
        results["numeric_step1"]["total"] += 1
        if step1_check["found"]:
            results["numeric_step1"]["correct"] += 1
            results["numeric_step1"]["ranks"].append(step1_check["rank"])
        
        # Check for step2 value
        step2_check = check_numeric_tokens(top_tokens, problem["step2"])
        results["numeric_step2"]["total"] += 1
        if step2_check["found"]:
            results["numeric_step2"]["correct"] += 1
            results["numeric_step2"]["ranks"].append(step2_check["rank"])
        
        # Check for operation
        op_check = check_operation_tokens(latent, model, tokenizer)
        results["operation"]["total"] += 1
        if op_check["predicted"] == problem["operation"]:
            results["operation"]["correct"] += 1
        results["operation"]["predictions"].append({
            "true": problem["operation"],
            "pred": op_check["predicted"],
            "confidence": op_check["confidence"],
            "probs": {k: v["total_prob"] for k, v in op_check.items() if k not in ["predicted", "confidence"]},
        })
        
        # Store some examples
        if i < 5:
            results["top_tokens_examples"].append({
                "problem": problem["prompt"][:100] + "...",
                "step1": problem["step1"],
                "step2": problem["step2"],
                "operation": problem["operation"],
                "top_5_tokens": [t["token"] for t in top_tokens[:5]],
                "top_5_probs": [t["prob"] for t in top_tokens[:5]],
            })
    
    # Compute averages
    if results["numeric_step1"]["ranks"]:
        results["numeric_step1"]["avg_rank"] = sum(results["numeric_step1"]["ranks"]) / len(results["numeric_step1"]["ranks"])
    if results["numeric_step2"]["ranks"]:
        results["numeric_step2"]["avg_rank"] = sum(results["numeric_step2"]["ranks"]) / len(results["numeric_step2"]["ranks"])
    
    # Clean up for JSON
    del results["numeric_step1"]["ranks"]
    del results["numeric_step2"]["ranks"]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Explore CODI latent positions")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of problems to analyze")
    parser.add_argument("--output", type=str, default="results/latent_exploration.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 0: Exploratory Logit Lens Analysis")
    print("=" * 60)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate problems
    print(f"\nGenerating {args.n_samples} synthetic problems...")
    problems = [generate_problem(seed=args.seed + i) for i in range(args.n_samples)]
    
    # Count operations
    op_counts = defaultdict(int)
    for p in problems:
        op_counts[p["operation"]] += 1
    print(f"Operation distribution: {dict(op_counts)}")
    
    # Load CODI model
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    # Get the base model for logit lens
    model = codi.model
    tokenizer = codi.tokenizer
    
    # Collect all latents
    print(f"\nRunning CODI on {args.n_samples} problems...")
    all_latents = []
    for problem in tqdm(problems, desc="Collecting latents"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        # result.latent_vectors is a list of tensors, one per position
        all_latents.append(result.latent_vectors)
    
    # Analyze each position
    print("\nAnalyzing each latent position...")
    results = {
        "config": {
            "n_samples": args.n_samples,
            "seed": args.seed,
            "num_positions": 6,
        },
        "positions": {},
        "summary": {},
    }
    
    for pos in range(6):
        print(f"\n--- Position z{pos+1} (index {pos}) ---")
        pos_results = analyze_latent_position(all_latents, pos, model, tokenizer, problems)
        results["positions"][f"z{pos+1}"] = pos_results
        
        # Print summary
        s1_acc = pos_results["numeric_step1"]["correct"] / pos_results["numeric_step1"]["total"] * 100
        s2_acc = pos_results["numeric_step2"]["correct"] / pos_results["numeric_step2"]["total"] * 100
        op_acc = pos_results["operation"]["correct"] / pos_results["operation"]["total"] * 100
        
        print(f"  Step 1 numeric (logit lens): {s1_acc:.1f}%")
        print(f"  Step 2 numeric (logit lens): {s2_acc:.1f}%")
        print(f"  Operation prediction: {op_acc:.1f}%")
        
        if args.verbose and pos_results["top_tokens_examples"]:
            ex = pos_results["top_tokens_examples"][0]
            print(f"  Example top tokens: {ex['top_5_tokens']}")
    
    # Summary across positions
    print("\n" + "=" * 60)
    print("SUMMARY: What does each position encode?")
    print("=" * 60)
    
    summary = []
    for pos in range(6):
        pos_key = f"z{pos+1}"
        pos_data = results["positions"][pos_key]
        
        s1_acc = pos_data["numeric_step1"]["correct"] / pos_data["numeric_step1"]["total"] * 100
        s2_acc = pos_data["numeric_step2"]["correct"] / pos_data["numeric_step2"]["total"] * 100
        op_acc = pos_data["operation"]["correct"] / pos_data["operation"]["total"] * 100
        
        # Determine what this position likely encodes
        encoding = "unknown"
        if s1_acc > 80:
            encoding = "STEP 1 RESULT"
        elif s2_acc > 80:
            encoding = "STEP 2 RESULT"
        elif op_acc > 60:
            encoding = "OPERATION TYPE"
        elif s1_acc > 40 or s2_acc > 40:
            encoding = "partial numeric"
        
        summary.append({
            "position": pos_key,
            "step1_acc": s1_acc,
            "step2_acc": s2_acc,
            "operation_acc": op_acc,
            "likely_encoding": encoding,
        })
        
        print(f"{pos_key}: Step1={s1_acc:5.1f}%, Step2={s2_acc:5.1f}%, Op={op_acc:5.1f}% -> {encoding}")
    
    results["summary"] = summary
    
    # Check if operation is findable anywhere
    best_op_pos = max(summary, key=lambda x: x["operation_acc"])
    print(f"\nBest position for operation: {best_op_pos['position']} ({best_op_pos['operation_acc']:.1f}%)")
    if best_op_pos["operation_acc"] < 50:
        print("WARNING: Operation type may not be linearly decodable from any position!")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
