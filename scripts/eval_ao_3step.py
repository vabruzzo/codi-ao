#!/usr/bin/env python3
"""
Evaluation script for 3-step Activation Oracle.

Tests BOTH single-latent and multi-latent performance:
- Single z2: step1 extraction, operation, operands
- Single z4: step2 extraction  
- Single z6: step3/final answer extraction
- Multi (all 6): all of the above

This enables direct comparison of what each latent encodes individually
vs. whether having all latents improves performance.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_codi_model(config_path="configs/default.yaml"):
    """Load the CODI model."""
    from src.codi_wrapper import CODIWrapper
    
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


def load_ao_model(checkpoint_path: str, config_path: str = "configs/default.yaml"):
    """Load the trained Activation Oracle model."""
    from src.activation_oracle import ActivationOracle, AOConfig, AOPrompt
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ao_config = AOConfig(
        base_model=config["model"]["ao_base_model"],
        latent_dim=config["model"]["ao_latent_dim"],
        inject_layer=config["model"]["ao_inject_layer"],
        lora_r=config["model"]["ao_lora_r"],
        lora_alpha=config["model"]["ao_lora_alpha"],
        device=device,
    )
    
    ao = ActivationOracle(ao_config)
    ao.load_checkpoint(checkpoint_path)
    ao.model.eval()
    
    return ao, AOPrompt


def ao_generate(ao, AOPrompt, latent_vectors, positions, question: str):
    """Generate response from AO."""
    # Ensure tensors on correct device
    device = ao.config.device
    latents_tensor = []
    for lat in latent_vectors:
        if isinstance(lat, list):
            lat = torch.tensor(lat, dtype=torch.float32)
        latents_tensor.append(lat.to(device))
    
    num_latents = len(latents_tensor)
    placeholders = " ?" * num_latents
    prompt_text = f"Layer 50%:{placeholders} {question}"
    
    prompt = AOPrompt(
        text=prompt_text,
        latent_vectors=latents_tensor,
        latent_positions=positions,
    )
    
    response = ao.generate(prompt, max_new_tokens=32, temperature=0)
    return response.strip()


def extract_number(text: str) -> int | None:
    """Extract first number from text."""
    match = re.search(r'-?\d+', text)
    return int(match.group()) if match else None


def extract_calculation(text: str):
    """
    Extract calculation from response.
    Returns (X, op, Y, result) or None.
    """
    text = text.strip()
    
    # Pattern: X op Y or X op Y = Z
    pattern = r'(-?\d+)\s*([+\-*×x])\s*(-?\d+)(?:\s*=\s*(-?\d+))?'
    match = re.search(pattern, text)
    
    if match:
        X = int(match.group(1))
        op_char = match.group(2)
        Y = int(match.group(3))
        
        # Normalize operator
        op_map = {'+': '+', '-': '-', '*': '*', '×': '*', 'x': '*'}
        op = op_map.get(op_char, op_char)
        
        # Calculate result
        if op == '+':
            result = X + Y
        elif op == '-':
            result = X - Y
        elif op == '*':
            result = X * Y
        else:
            result = None
        
        return (X, op, Y, result)
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--problems", type=str, default="data/synthetic_problems_3step.json")
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/ao_3step_evaluation.json")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle latents for sanity check")
    args = parser.parse_args()
    
    print("=" * 60)
    print("3-Step Activation Oracle Evaluation")
    print("=" * 60)
    
    if args.shuffle:
        print("\n*** SHUFFLE MODE: Latents will be mismatched (sanity check) ***\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load problems
    print(f"\nLoading problems from {args.problems}...")
    with open(args.problems) as f:
        data = json.load(f)
    all_problems = data["problems"]
    
    # Use last N as test set
    test_problems = all_problems[-args.n_test:]
    print(f"Testing on {len(test_problems)} held-out problems")
    
    # Load models
    print("\nLoading CODI model...")
    codi = load_codi_model()
    
    print(f"Loading Activation Oracle from {args.checkpoint}...")
    ao, AOPrompt = load_ao_model(args.checkpoint)
    
    # Initialize results
    results = {
        # Single-latent extraction
        "extraction_step1_single": {"correct": 0, "total": 0, "examples": []},
        "extraction_step2_single": {"correct": 0, "total": 0, "examples": []},
        "extraction_step3_single": {"correct": 0, "total": 0, "examples": []},
        
        # Multi-latent extraction
        "extraction_step1_multi": {"correct": 0, "total": 0, "examples": []},
        "extraction_step2_multi": {"correct": 0, "total": 0, "examples": []},
        "extraction_step3_multi": {"correct": 0, "total": 0, "examples": []},
        
        # Operation (single and multi)
        "operation_single": {"correct": 0, "total": 0, "by_op": {}},
        "operation_multi": {"correct": 0, "total": 0, "by_op": {}},
        
        # Operands (single and multi)
        "operand_first_single": {"correct": 0, "total": 0},
        "operand_first_multi": {"correct": 0, "total": 0},
        "operand_second_single": {"correct": 0, "total": 0},
        "operand_second_multi": {"correct": 0, "total": 0},
        
        # Full calculation (single and multi)
        "full_calc_single_strict": {"correct": 0, "total": 0},
        "full_calc_single_semantic": {"correct": 0, "total": 0},
        "full_calc_multi_strict": {"correct": 0, "total": 0},
        "full_calc_multi_semantic": {"correct": 0, "total": 0},
        
        # Comparison (multi only)
        "comparison_multi": {"correct": 0, "total": 0},
        
        # Config
        "config": {
            "checkpoint": args.checkpoint,
            "n_test": args.n_test,
            "shuffle": args.shuffle,
        }
    }
    
    # Collect all latents first (for shuffle mode)
    print("\nCollecting latents...")
    all_latents_list = []
    for problem in tqdm(test_problems, desc="Collecting"):
        latent_result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        if len(latent_result.latent_vectors) >= 6:
            all_latents_list.append(latent_result.latent_vectors[:6])
        else:
            all_latents_list.append(None)
    
    # Shuffle if requested
    if args.shuffle:
        import random
        random.seed(42)
        valid_indices = [i for i, lat in enumerate(all_latents_list) if lat is not None]
        shuffled_indices = valid_indices.copy()
        random.shuffle(shuffled_indices)
        
        # Create mapping
        shuffle_map = dict(zip(valid_indices, shuffled_indices))
    
    # Question templates
    step1_q = "What was calculated in the first step?"
    step2_q = "What was calculated in the second step?"
    step3_q = "What is the final answer?"
    op_q = "What operation was performed in step 1?"
    first_op_q = "What was the first number in step 1?"
    second_op_q = "What was the second number in step 1?"
    full_calc_q = "What calculation was performed in step 1?"
    compare_q = "Which step produced a larger result: step 1 or step 2?"
    
    op_names = {"add": "addition", "sub": "subtraction", "mul": "multiplication"}
    op_symbols = {"add": "+", "sub": "-", "mul": "*"}
    
    # Run evaluations
    print("\nRunning evaluations...")
    print("-" * 60)
    
    for i, problem in enumerate(tqdm(test_problems, desc="Evaluating")):
        latents = all_latents_list[i]
        if latents is None:
            continue
        
        # Get potentially shuffled latents
        if args.shuffle and i in shuffle_map:
            latents = all_latents_list[shuffle_map[i]]
        
        step1 = problem["step1"]
        step2 = problem["step2"]
        step3 = problem["step3"]
        op = problem["operation"]
        X = problem["X"]
        Y = problem["Y"]
        
        z2 = latents[1]  # Position 1
        z4 = latents[3]  # Position 3
        z6 = latents[5]  # Position 5
        
        # =====================================================================
        # STEP 1 EXTRACTION
        # =====================================================================
        
        # Single z2
        resp = ao_generate(ao, AOPrompt, [z2], [1], step1_q)
        pred = extract_number(resp)
        correct = (pred == step1)
        results["extraction_step1_single"]["total"] += 1
        if correct:
            results["extraction_step1_single"]["correct"] += 1
        
        # Multi (all 6)
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], step1_q)
        pred = extract_number(resp)
        correct = (pred == step1)
        results["extraction_step1_multi"]["total"] += 1
        if correct:
            results["extraction_step1_multi"]["correct"] += 1
        
        # =====================================================================
        # STEP 2 EXTRACTION
        # =====================================================================
        
        # Single z4
        resp = ao_generate(ao, AOPrompt, [z4], [3], step2_q)
        pred = extract_number(resp)
        correct = (pred == step2)
        results["extraction_step2_single"]["total"] += 1
        if correct:
            results["extraction_step2_single"]["correct"] += 1
        
        # Multi (all 6)
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], step2_q)
        pred = extract_number(resp)
        correct = (pred == step2)
        results["extraction_step2_multi"]["total"] += 1
        if correct:
            results["extraction_step2_multi"]["correct"] += 1
        
        # =====================================================================
        # STEP 3 / FINAL ANSWER EXTRACTION
        # =====================================================================
        
        # Single z6
        resp = ao_generate(ao, AOPrompt, [z6], [5], step3_q)
        pred = extract_number(resp)
        correct = (pred == step3)
        results["extraction_step3_single"]["total"] += 1
        if correct:
            results["extraction_step3_single"]["correct"] += 1
        
        # Multi (all 6)
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], step3_q)
        pred = extract_number(resp)
        correct = (pred == step3)
        results["extraction_step3_multi"]["total"] += 1
        if correct:
            results["extraction_step3_multi"]["correct"] += 1
        
        # =====================================================================
        # OPERATION DETECTION
        # =====================================================================
        
        expected_op = op_names[op]
        
        # Single z2
        resp = ao_generate(ao, AOPrompt, [z2], [1], op_q)
        correct = expected_op.lower() in resp.lower()
        results["operation_single"]["total"] += 1
        if correct:
            results["operation_single"]["correct"] += 1
        results["operation_single"]["by_op"].setdefault(op, {"correct": 0, "total": 0})
        results["operation_single"]["by_op"][op]["total"] += 1
        if correct:
            results["operation_single"]["by_op"][op]["correct"] += 1
        
        # Multi
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], op_q)
        correct = expected_op.lower() in resp.lower()
        results["operation_multi"]["total"] += 1
        if correct:
            results["operation_multi"]["correct"] += 1
        results["operation_multi"]["by_op"].setdefault(op, {"correct": 0, "total": 0})
        results["operation_multi"]["by_op"][op]["total"] += 1
        if correct:
            results["operation_multi"]["by_op"][op]["correct"] += 1
        
        # =====================================================================
        # OPERAND EXTRACTION
        # =====================================================================
        
        # First operand - single
        resp = ao_generate(ao, AOPrompt, [z2], [1], first_op_q)
        pred = extract_number(resp)
        results["operand_first_single"]["total"] += 1
        if pred == X:
            results["operand_first_single"]["correct"] += 1
        
        # First operand - multi
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], first_op_q)
        pred = extract_number(resp)
        results["operand_first_multi"]["total"] += 1
        if pred == X:
            results["operand_first_multi"]["correct"] += 1
        
        # Second operand - single
        resp = ao_generate(ao, AOPrompt, [z2], [1], second_op_q)
        pred = extract_number(resp)
        results["operand_second_single"]["total"] += 1
        if pred == Y:
            results["operand_second_single"]["correct"] += 1
        
        # Second operand - multi
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], second_op_q)
        pred = extract_number(resp)
        results["operand_second_multi"]["total"] += 1
        if pred == Y:
            results["operand_second_multi"]["correct"] += 1
        
        # =====================================================================
        # FULL CALCULATION
        # =====================================================================
        
        expected_op_sym = op_symbols[op]
        
        # Single
        resp = ao_generate(ao, AOPrompt, [z2], [1], full_calc_q)
        parsed = extract_calculation(resp)
        
        results["full_calc_single_strict"]["total"] += 1
        results["full_calc_single_semantic"]["total"] += 1
        
        if parsed:
            pX, pOp, pY, pResult = parsed
            # Strict: exact operands
            strict_correct = (pX == X and pY == Y and pOp == expected_op_sym)
            # Semantic: correct operation and result
            semantic_correct = (pOp == expected_op_sym and pResult == step1)
            
            if strict_correct:
                results["full_calc_single_strict"]["correct"] += 1
            if semantic_correct:
                results["full_calc_single_semantic"]["correct"] += 1
        
        # Multi
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], full_calc_q)
        parsed = extract_calculation(resp)
        
        results["full_calc_multi_strict"]["total"] += 1
        results["full_calc_multi_semantic"]["total"] += 1
        
        if parsed:
            pX, pOp, pY, pResult = parsed
            strict_correct = (pX == X and pY == Y and pOp == expected_op_sym)
            semantic_correct = (pOp == expected_op_sym and pResult == step1)
            
            if strict_correct:
                results["full_calc_multi_strict"]["correct"] += 1
            if semantic_correct:
                results["full_calc_multi_semantic"]["correct"] += 1
        
        # =====================================================================
        # COMPARISON (multi only)
        # =====================================================================
        
        resp = ao_generate(ao, AOPrompt, latents, [0,1,2,3,4,5], compare_q)
        expected = "step 2" if step2 > step1 else "step 1"
        correct = expected in resp.lower()
        results["comparison_multi"]["total"] += 1
        if correct:
            results["comparison_multi"]["correct"] += 1
    
    # =========================================================================
    # PRINT RESULTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    def pct(d):
        if d["total"] == 0:
            return "N/A"
        return f'{100 * d["correct"] / d["total"]:.1f}%'
    
    print("\n--- Extraction (Single vs Multi) ---")
    print(f"Step 1 (z2 only):  {pct(results['extraction_step1_single'])} ({results['extraction_step1_single']['correct']}/{results['extraction_step1_single']['total']})")
    print(f"Step 1 (all 6):    {pct(results['extraction_step1_multi'])} ({results['extraction_step1_multi']['correct']}/{results['extraction_step1_multi']['total']})")
    print()
    print(f"Step 2 (z4 only):  {pct(results['extraction_step2_single'])} ({results['extraction_step2_single']['correct']}/{results['extraction_step2_single']['total']})")
    print(f"Step 2 (all 6):    {pct(results['extraction_step2_multi'])} ({results['extraction_step2_multi']['correct']}/{results['extraction_step2_multi']['total']})")
    print()
    print(f"Step 3 (z6 only):  {pct(results['extraction_step3_single'])} ({results['extraction_step3_single']['correct']}/{results['extraction_step3_single']['total']})")
    print(f"Step 3 (all 6):    {pct(results['extraction_step3_multi'])} ({results['extraction_step3_multi']['correct']}/{results['extraction_step3_multi']['total']})")
    
    print("\n--- Operation Detection ---")
    print(f"Single (z2):  {pct(results['operation_single'])}")
    print(f"Multi (all):  {pct(results['operation_multi'])}")
    
    print("\n--- Operand Extraction ---")
    print(f"First operand (z2):   {pct(results['operand_first_single'])}")
    print(f"First operand (all):  {pct(results['operand_first_multi'])}")
    print(f"Second operand (z2):  {pct(results['operand_second_single'])}")
    print(f"Second operand (all): {pct(results['operand_second_multi'])}")
    
    print("\n--- Full Calculation ---")
    print(f"Strict (z2):    {pct(results['full_calc_single_strict'])}")
    print(f"Strict (all):   {pct(results['full_calc_multi_strict'])}")
    print(f"Semantic (z2):  {pct(results['full_calc_single_semantic'])}")
    print(f"Semantic (all): {pct(results['full_calc_multi_semantic'])}")
    
    print("\n--- Comparison (multi only) ---")
    print(f"Comparison: {pct(results['comparison_multi'])}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
