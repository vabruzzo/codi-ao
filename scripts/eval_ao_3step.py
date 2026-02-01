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
from collections import Counter
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_training_frequencies(all_problems, n_train=1000):
    """Compute value frequencies in training set."""
    train = all_problems[:n_train]
    return {
        "step1": Counter(p["step1"] for p in train),
        "step3": Counter(p["step3"] for p in train),
    }


def categorize_rarity(value, counts, rare_threshold, common_threshold):
    """Categorize a value as rare, common, or medium based on training frequency."""
    count = counts.get(value, 0)
    if count <= rare_threshold:
        return "rare"
    elif count >= common_threshold:
        return "common"
    return "medium"


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


def load_ao_model(checkpoint_path: str):
    """Load the trained Activation Oracle model."""
    from src.activation_oracle import ActivationOracle, AOConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = AOConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device=device,
        lora_r=64,
        lora_alpha=128,
    )
    
    ao = ActivationOracle.from_pretrained(config=config, lora_path=checkpoint_path)
    ao.eval_mode()
    
    return ao


def ao_generate(ao, latent_vectors: list, question: str, max_new_tokens: int = 32):
    """Generate response from AO."""
    # Convert latent vectors to tensors if needed
    vectors = []
    for v in latent_vectors:
        if isinstance(v, torch.Tensor):
            vectors.append(v)
        elif isinstance(v, list):
            vectors.append(torch.tensor(v, dtype=torch.float32))
        else:
            vectors.append(v)
    
    prompt = ao.create_prompt(question=question, activation_vectors=vectors)
    response = ao.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0)
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
    ao = load_ao_model(args.checkpoint)
    
    # Initialize results
    results = {
        # Single-latent extraction
        "extraction_step1_single": {"correct": 0, "total": 0, "examples": []},
        "extraction_step2_single": {"correct": 0, "total": 0, "examples": []},
        "extraction_step3_z5_single": {"correct": 0, "total": 0, "examples": []},
        "extraction_step3_z6_single": {"correct": 0, "total": 0, "examples": []},
        "extraction_step3_z2z4": {"correct": 0, "total": 0},  # z2+z4 only (test if AO computes)
        
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
        
        # CODI correctness breakdown for step3 extraction
        "codi_analysis": {
            "codi_correct_count": 0,
            "codi_total": 0,
            # When CODI is correct, does AO extract correctly?
            "ao_step3_given_codi_correct": {"correct": 0, "total": 0},
            # When CODI is wrong, does AO extract CODI's (wrong) answer?
            "ao_matches_codi_given_codi_wrong": {"matches": 0, "total": 0},
            # Does AO extraction match CODI output (regardless of correctness)?
            "ao_matches_codi_output": {"matches": 0, "total": 0},
            # Step 1 and Step 2 accuracy broken down by CODI correctness
            "ao_step1_given_codi_correct": {"correct": 0, "total": 0},
            "ao_step1_given_codi_wrong": {"correct": 0, "total": 0},
            "ao_step2_given_codi_correct": {"correct": 0, "total": 0},
            "ao_step2_given_codi_wrong": {"correct": 0, "total": 0},
            # Store examples where CODI was wrong for detailed analysis
            "codi_wrong_examples": [],
        },
        
        # Rarity analysis: does AO perform differently on values seen rarely in training?
        "rarity_analysis": {
            # Step 1 extraction by rarity (rare = ≤15 occurrences, common = ≥50)
            "step1_rare": {"correct": 0, "total": 0},
            "step1_common": {"correct": 0, "total": 0},
            "step1_medium": {"correct": 0, "total": 0},
            # Step 3 extraction by rarity (rare = ≤10 occurrences, common = ≥20)
            "step3_rare": {"correct": 0, "total": 0},
            "step3_common": {"correct": 0, "total": 0},
            "step3_medium": {"correct": 0, "total": 0},
            # Thresholds used
            "thresholds": {
                "step1_rare": 15, "step1_common": 50,
                "step3_rare": 10, "step3_common": 20,
            },
        },
        
        # Config
        "config": {
            "checkpoint": args.checkpoint,
            "n_test": args.n_test,
            "shuffle": args.shuffle,
        }
    }
    
    # Compute training value frequencies for rarity analysis
    n_train = len(all_problems) - args.n_test
    train_freqs = compute_training_frequencies(all_problems, n_train)
    print(f"\nTraining set: {n_train} problems")
    print(f"Unique step1 values in train: {len(train_freqs['step1'])}")
    print(f"Unique step3 values in train: {len(train_freqs['step3'])}")
    
    # Collect all latents first (for shuffle mode)
    # Also capture CODI's actual predictions for correctness analysis
    print("\nCollecting latents...")
    all_latents_list = []
    codi_predictions = []  # Store CODI's actual outputs
    codi_correct_list = []  # Store whether CODI was correct
    
    for problem in tqdm(test_problems, desc="Collecting"):
        ground_truth = str(problem["step3"])  # Final answer as ground truth
        latent_result = codi.collect_latents(
            problem["prompt"], 
            ground_truth_answer=ground_truth,
            return_hidden_states=False
        )
        
        if len(latent_result.latent_vectors) >= 6:
            all_latents_list.append(latent_result.latent_vectors[:6])
            codi_predictions.append(latent_result.predicted_answer)
            codi_correct_list.append(latent_result.is_correct)
        else:
            all_latents_list.append(None)
            codi_predictions.append(None)
            codi_correct_list.append(None)
    
    # Report CODI accuracy
    valid_codi = [c for c in codi_correct_list if c is not None]
    codi_accuracy = sum(valid_codi) / len(valid_codi) * 100 if valid_codi else 0
    print(f"\nCODI accuracy on test set: {codi_accuracy:.1f}% ({sum(valid_codi)}/{len(valid_codi)})")
    
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
        z5 = latents[4]  # Position 4
        z6 = latents[5]  # Position 5
        
        # =====================================================================
        # STEP 1 EXTRACTION
        # =====================================================================
        
        # Problem context for examples
        problem_ctx = {
            "prompt": problem["prompt"][:150],
            "X": X, "Y": Y, "Z": problem.get("Z"),
            "op": op, "step1": step1, "step2": step2, "step3": step3
        }
        
        # Single z2
        resp = ao_generate(ao, [z2], step1_q)
        pred = extract_number(resp)
        correct = (pred == step1)
        results["extraction_step1_single"]["total"] += 1
        if correct:
            results["extraction_step1_single"]["correct"] += 1
        if len(results["extraction_step1_single"]["examples"]) < 10:
            results["extraction_step1_single"]["examples"].append({
                "problem": problem_ctx, "question": step1_q, "latent_positions": [1],
                "true": step1, "response": resp, "parsed": pred, "correct": correct
            })
        
        # Multi (all 6)
        resp_multi = ao_generate(ao, latents, step1_q)
        pred_multi = extract_number(resp_multi)
        step1_multi_correct = (pred_multi == step1)  # Store for CODI analysis
        results["extraction_step1_multi"]["total"] += 1
        if step1_multi_correct:
            results["extraction_step1_multi"]["correct"] += 1
        if len(results["extraction_step1_multi"]["examples"]) < 10:
            results["extraction_step1_multi"]["examples"].append({
                "problem": problem_ctx, "question": step1_q, "latent_positions": [0,1,2,3,4,5],
                "true": step1, "response": resp_multi, "parsed": pred_multi, "correct": step1_multi_correct
            })
        
        # Rarity tracking for step1 (using single z2 result)
        step1_rarity = categorize_rarity(
            step1, train_freqs["step1"], 
            results["rarity_analysis"]["thresholds"]["step1_rare"],
            results["rarity_analysis"]["thresholds"]["step1_common"]
        )
        results["rarity_analysis"][f"step1_{step1_rarity}"]["total"] += 1
        if correct:  # correct refers to single z2 result
            results["rarity_analysis"][f"step1_{step1_rarity}"]["correct"] += 1
        
        # =====================================================================
        # STEP 2 EXTRACTION
        # =====================================================================
        
        # Single z4
        resp = ao_generate(ao, [z4], step2_q)
        pred = extract_number(resp)
        correct = (pred == step2)
        results["extraction_step2_single"]["total"] += 1
        if correct:
            results["extraction_step2_single"]["correct"] += 1
        if len(results["extraction_step2_single"]["examples"]) < 10:
            results["extraction_step2_single"]["examples"].append({
                "problem": problem_ctx, "question": step2_q, "latent_positions": [3],
                "true": step2, "response": resp, "parsed": pred, "correct": correct
            })
        
        # Multi (all 6)
        resp = ao_generate(ao, latents, step2_q)
        pred = extract_number(resp)
        step2_multi_correct = (pred == step2)  # Store for CODI analysis
        results["extraction_step2_multi"]["total"] += 1
        if step2_multi_correct:
            results["extraction_step2_multi"]["correct"] += 1
        if len(results["extraction_step2_multi"]["examples"]) < 10:
            results["extraction_step2_multi"]["examples"].append({
                "problem": problem_ctx, "question": step2_q, "latent_positions": [0,1,2,3,4,5],
                "true": step2, "response": resp, "parsed": pred, "correct": correct
            })
        
        # =====================================================================
        # STEP 3 / FINAL ANSWER EXTRACTION
        # =====================================================================
        
        # Single z5
        resp = ao_generate(ao, [z5], step3_q)
        pred = extract_number(resp)
        correct = (pred == step3)
        results["extraction_step3_z5_single"]["total"] += 1
        if correct:
            results["extraction_step3_z5_single"]["correct"] += 1
        if len(results["extraction_step3_z5_single"]["examples"]) < 10:
            results["extraction_step3_z5_single"]["examples"].append({
                "problem": problem_ctx, "question": step3_q, "latent_positions": [4],
                "true": step3, "response": resp, "parsed": pred, "correct": correct
            })
        
        # Single z6
        resp = ao_generate(ao, [z6], step3_q)
        pred = extract_number(resp)
        correct = (pred == step3)
        results["extraction_step3_z6_single"]["total"] += 1
        if correct:
            results["extraction_step3_z6_single"]["correct"] += 1
        if len(results["extraction_step3_z6_single"]["examples"]) < 10:
            results["extraction_step3_z6_single"]["examples"].append({
                "problem": problem_ctx, "question": step3_q, "latent_positions": [5],
                "true": step3, "response": resp, "parsed": pred, "correct": correct
            })
        
        # z2+z4 only (step1 and step2 latents only) - test if AO computes step3
        resp_z2z4 = ao_generate(ao, [z2, z4], step3_q)
        pred_z2z4 = extract_number(resp_z2z4)
        correct_z2z4 = (pred_z2z4 == step3)
        results["extraction_step3_z2z4"]["total"] += 1
        if correct_z2z4:
            results["extraction_step3_z2z4"]["correct"] += 1
        
        # Multi (all 6)
        resp = ao_generate(ao, latents, step3_q)
        pred = extract_number(resp)
        correct = (pred == step3)
        results["extraction_step3_multi"]["total"] += 1
        if correct:
            results["extraction_step3_multi"]["correct"] += 1
        if len(results["extraction_step3_multi"]["examples"]) < 10:
            codi_pred_for_example = codi_predictions[i]
            codi_pred_num_for_example = extract_number(codi_pred_for_example) if codi_pred_for_example else None
            results["extraction_step3_multi"]["examples"].append({
                "problem": problem_ctx, "question": step3_q, "latent_positions": [0,1,2,3,4,5],
                "true": step3, 
                "codi_output": codi_pred_for_example,
                "codi_output_parsed": codi_pred_num_for_example,
                "codi_correct": codi_correct_list[i],
                "ao_response": resp, 
                "ao_parsed": pred, 
                "ao_correct_vs_truth": correct,
                "ao_matches_codi": (pred == codi_pred_num_for_example) if codi_pred_num_for_example is not None else None
            })
        
        # Rarity tracking for step3 (using multi-latent result)
        step3_rarity = categorize_rarity(
            step3, train_freqs["step3"],
            results["rarity_analysis"]["thresholds"]["step3_rare"],
            results["rarity_analysis"]["thresholds"]["step3_common"]
        )
        results["rarity_analysis"][f"step3_{step3_rarity}"]["total"] += 1
        if correct:  # correct refers to multi-latent result
            results["rarity_analysis"][f"step3_{step3_rarity}"]["correct"] += 1
        
        # =====================================================================
        # CODI CORRECTNESS ANALYSIS (for step3 multi)
        # =====================================================================
        codi_pred = codi_predictions[i]
        codi_is_correct = codi_correct_list[i]
        
        if codi_pred is not None:
            # Extract number from CODI's prediction
            codi_pred_num = extract_number(codi_pred)
            
            results["codi_analysis"]["codi_total"] += 1
            if codi_is_correct:
                results["codi_analysis"]["codi_correct_count"] += 1
            
            # Does AO's extraction match CODI's actual output?
            ao_matches_codi = (pred == codi_pred_num)
            results["codi_analysis"]["ao_matches_codi_output"]["total"] += 1
            if ao_matches_codi:
                results["codi_analysis"]["ao_matches_codi_output"]["matches"] += 1
            
            # When CODI is correct, does AO extract correctly?
            if codi_is_correct:
                # Step 3 (final answer) accuracy
                results["codi_analysis"]["ao_step3_given_codi_correct"]["total"] += 1
                if correct:  # AO matches ground truth for step3
                    results["codi_analysis"]["ao_step3_given_codi_correct"]["correct"] += 1
                
                # Step 1 and Step 2 accuracy when CODI is correct
                results["codi_analysis"]["ao_step1_given_codi_correct"]["total"] += 1
                if step1_multi_correct:
                    results["codi_analysis"]["ao_step1_given_codi_correct"]["correct"] += 1
                results["codi_analysis"]["ao_step2_given_codi_correct"]["total"] += 1
                if step2_multi_correct:
                    results["codi_analysis"]["ao_step2_given_codi_correct"]["correct"] += 1
            else:
                # When CODI is wrong, does AO extract CODI's (wrong) answer?
                results["codi_analysis"]["ao_matches_codi_given_codi_wrong"]["total"] += 1
                if ao_matches_codi:
                    results["codi_analysis"]["ao_matches_codi_given_codi_wrong"]["matches"] += 1
                
                # Step 1 and Step 2 accuracy when CODI is wrong
                results["codi_analysis"]["ao_step1_given_codi_wrong"]["total"] += 1
                if step1_multi_correct:
                    results["codi_analysis"]["ao_step1_given_codi_wrong"]["correct"] += 1
                results["codi_analysis"]["ao_step2_given_codi_wrong"]["total"] += 1
                if step2_multi_correct:
                    results["codi_analysis"]["ao_step2_given_codi_wrong"]["correct"] += 1
                
                # Store examples where CODI was wrong (up to 20)
                if len(results["codi_analysis"]["codi_wrong_examples"]) < 20:
                    results["codi_analysis"]["codi_wrong_examples"].append({
                        "problem": problem_ctx,
                        "ground_truth": step3,
                        "codi_output": codi_pred,
                        "codi_output_parsed": codi_pred_num,
                        "ao_response": resp,
                        "ao_parsed": pred,
                        "ao_matches_codi": ao_matches_codi,
                        "ao_matches_truth": correct,
                        "ao_step1_correct": step1_multi_correct,
                        "ao_step2_correct": step2_multi_correct,
                    })
        
        # =====================================================================
        # OPERATION DETECTION
        # =====================================================================
        
        expected_op = op_names[op]
        
        # Single z2
        resp = ao_generate(ao, [z2], op_q)
        correct = expected_op.lower() in resp.lower()
        results["operation_single"]["total"] += 1
        if correct:
            results["operation_single"]["correct"] += 1
        results["operation_single"]["by_op"].setdefault(op, {"correct": 0, "total": 0})
        results["operation_single"]["by_op"][op]["total"] += 1
        if correct:
            results["operation_single"]["by_op"][op]["correct"] += 1
        
        # Multi
        resp = ao_generate(ao, latents, op_q)
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
        resp = ao_generate(ao, [z2], first_op_q)
        pred = extract_number(resp)
        results["operand_first_single"]["total"] += 1
        if pred == X:
            results["operand_first_single"]["correct"] += 1
        
        # First operand - multi
        resp = ao_generate(ao, latents, first_op_q)
        pred = extract_number(resp)
        results["operand_first_multi"]["total"] += 1
        if pred == X:
            results["operand_first_multi"]["correct"] += 1
        
        # Second operand - single
        resp = ao_generate(ao, [z2], second_op_q)
        pred = extract_number(resp)
        results["operand_second_single"]["total"] += 1
        if pred == Y:
            results["operand_second_single"]["correct"] += 1
        
        # Second operand - multi
        resp = ao_generate(ao, latents, second_op_q)
        pred = extract_number(resp)
        results["operand_second_multi"]["total"] += 1
        if pred == Y:
            results["operand_second_multi"]["correct"] += 1
        
        # =====================================================================
        # FULL CALCULATION
        # =====================================================================
        
        expected_op_sym = op_symbols[op]
        true_calc = f"{X} {expected_op_sym} {Y} = {step1}"
        
        # Single
        resp = ao_generate(ao, [z2], full_calc_q)
        parsed = extract_calculation(resp)
        
        results["full_calc_single_strict"]["total"] += 1
        results["full_calc_single_semantic"]["total"] += 1
        
        strict_correct = False
        semantic_correct = False
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
        
        # Store examples for single
        if "examples" not in results["full_calc_single_strict"]:
            results["full_calc_single_strict"]["examples"] = []
        if len(results["full_calc_single_strict"]["examples"]) < 10:
            results["full_calc_single_strict"]["examples"].append({
                "problem": problem_ctx, "question": full_calc_q, "latent_positions": [1],
                "true": true_calc, "response": resp, "parsed": parsed,
                "strict": strict_correct, "semantic": semantic_correct
            })
        
        # Multi
        resp_multi = ao_generate(ao, latents, full_calc_q)
        parsed_multi = extract_calculation(resp_multi)
        
        results["full_calc_multi_strict"]["total"] += 1
        results["full_calc_multi_semantic"]["total"] += 1
        
        strict_correct_multi = False
        semantic_correct_multi = False
        if parsed_multi:
            pX, pOp, pY, pResult = parsed_multi
            strict_correct_multi = (pX == X and pY == Y and pOp == expected_op_sym)
            semantic_correct_multi = (pOp == expected_op_sym and pResult == step1)
            
            if strict_correct_multi:
                results["full_calc_multi_strict"]["correct"] += 1
            if semantic_correct_multi:
                results["full_calc_multi_semantic"]["correct"] += 1
        
        # Store examples for multi
        if "examples" not in results["full_calc_multi_strict"]:
            results["full_calc_multi_strict"]["examples"] = []
        if len(results["full_calc_multi_strict"]["examples"]) < 10:
            results["full_calc_multi_strict"]["examples"].append({
                "problem": problem_ctx, "question": full_calc_q, "latent_positions": [0,1,2,3,4,5],
                "true": true_calc, "response": resp_multi, "parsed": parsed_multi,
                "strict": strict_correct_multi, "semantic": semantic_correct_multi
            })
        
        # =====================================================================
        # COMPARISON (multi only)
        # =====================================================================
        
        resp = ao_generate(ao, latents, compare_q)
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
    print(f"Step 3 (z5 only):  {pct(results['extraction_step3_z5_single'])} ({results['extraction_step3_z5_single']['correct']}/{results['extraction_step3_z5_single']['total']})")
    print(f"Step 3 (z6 only):  {pct(results['extraction_step3_z6_single'])} ({results['extraction_step3_z6_single']['correct']}/{results['extraction_step3_z6_single']['total']})")
    print(f"Step 3 (z2+z4):    {pct(results['extraction_step3_z2z4'])} ({results['extraction_step3_z2z4']['correct']}/{results['extraction_step3_z2z4']['total']})  <-- step1+step2 latents only")
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
    
    # CODI correctness analysis
    codi_analysis = results["codi_analysis"]
    print("\n--- CODI Correctness Analysis ---")
    print(f"CODI accuracy: {codi_analysis['codi_correct_count']}/{codi_analysis['codi_total']} ({codi_analysis['codi_correct_count']/codi_analysis['codi_total']*100:.1f}%)" if codi_analysis['codi_total'] > 0 else "CODI accuracy: N/A")
    
    # AO extraction given CODI correct
    print("\nWhen CODI is CORRECT:")
    ao_s3_cc = codi_analysis["ao_step3_given_codi_correct"]
    ao_s1_cc = codi_analysis["ao_step1_given_codi_correct"]
    ao_s2_cc = codi_analysis["ao_step2_given_codi_correct"]
    if ao_s3_cc["total"] > 0:
        print(f"  AO Step 1 accuracy: {ao_s1_cc['correct']}/{ao_s1_cc['total']} ({ao_s1_cc['correct']/ao_s1_cc['total']*100:.1f}%)")
        print(f"  AO Step 2 accuracy: {ao_s2_cc['correct']}/{ao_s2_cc['total']} ({ao_s2_cc['correct']/ao_s2_cc['total']*100:.1f}%)")
        print(f"  AO Step 3 accuracy: {ao_s3_cc['correct']}/{ao_s3_cc['total']} ({ao_s3_cc['correct']/ao_s3_cc['total']*100:.1f}%)")
    
    # AO extraction given CODI wrong
    print("\nWhen CODI is WRONG:")
    ao_s1_cw = codi_analysis["ao_step1_given_codi_wrong"]
    ao_s2_cw = codi_analysis["ao_step2_given_codi_wrong"]
    ao_codi_wrong = codi_analysis["ao_matches_codi_given_codi_wrong"]
    if ao_codi_wrong["total"] > 0:
        print(f"  AO Step 1 accuracy: {ao_s1_cw['correct']}/{ao_s1_cw['total']} ({ao_s1_cw['correct']/ao_s1_cw['total']*100:.1f}%)")
        print(f"  AO Step 2 accuracy: {ao_s2_cw['correct']}/{ao_s2_cw['total']} ({ao_s2_cw['correct']/ao_s2_cw['total']*100:.1f}%)")
        print(f"  AO matches CODI's wrong answer: {ao_codi_wrong['matches']}/{ao_codi_wrong['total']} ({ao_codi_wrong['matches']/ao_codi_wrong['total']*100:.1f}%)")
    else:
        print("  (No cases where CODI was wrong)")
    
    ao_matches = codi_analysis["ao_matches_codi_output"]
    if ao_matches["total"] > 0:
        print(f"\nAO matches CODI output (overall): {ao_matches['matches']}/{ao_matches['total']} ({ao_matches['matches']/ao_matches['total']*100:.1f}%)")
    
    # Rarity analysis
    rarity = results["rarity_analysis"]
    print("\n--- Rarity Analysis (Memorization Test) ---")
    print("Does AO perform differently on values seen rarely vs commonly in training?")
    print()
    
    def rarity_pct(d):
        if d["total"] == 0:
            return "N/A (n=0)"
        return f'{100 * d["correct"] / d["total"]:.1f}% (n={d["total"]})'
    
    print(f"Step 1 extraction (z2 only):")
    print(f"  Rare (≤{rarity['thresholds']['step1_rare']} train occurrences):   {rarity_pct(rarity['step1_rare'])}")
    print(f"  Medium:                                {rarity_pct(rarity['step1_medium'])}")
    print(f"  Common (≥{rarity['thresholds']['step1_common']} train occurrences):  {rarity_pct(rarity['step1_common'])}")
    print()
    print(f"Step 3 extraction (all 6 latents):")
    print(f"  Rare (≤{rarity['thresholds']['step3_rare']} train occurrences):   {rarity_pct(rarity['step3_rare'])}")
    print(f"  Medium:                                {rarity_pct(rarity['step3_medium'])}")
    print(f"  Common (≥{rarity['thresholds']['step3_common']} train occurrences):  {rarity_pct(rarity['step3_common'])}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
