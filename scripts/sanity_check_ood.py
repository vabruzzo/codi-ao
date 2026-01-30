#!/usr/bin/env python3
"""
Quick sanity check: Test AO on out-of-distribution data.

Tests:
1. Larger numbers (1-100 instead of 1-10)
2. Novel entity types (zoo, library, etc. not in training)
3. Edge cases (result=0, result=1, large results)
4. Novel question phrasings
5. Real GSM8k problems (optional)

This tests whether the AO learned actual latent interpretation or just memorized
patterns from the narrow training distribution.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.codi_wrapper import CODIWrapper
from src.activation_oracle import ActivationOracle


# ============================================================================
# OOD TEST 1: Larger Numbers
# ============================================================================

def create_large_number_prompts(n: int, min_val: int = 1, max_val: int = 100, seed: int = 999) -> list[dict]:
    """Create test prompts with larger number ranges."""
    random.seed(seed)
    
    ADDITION_TEMPLATES = [
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. They hire {Y} more employees. Then each current employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. They enroll {Y} new students. Then each current student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
    ]
    
    SUBTRACTION_TEMPLATES = [
        "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. {Y} employees resign. Then each remaining employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. {Y} students transfer out. Then each remaining student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
    ]
    
    prompts = []
    for i in range(n):
        is_addition = random.random() < 0.5
        
        if is_addition:
            template = random.choice(ADDITION_TEMPLATES)
            x = random.randint(min_val, max_val)
            y = random.randint(min_val, max_val)
            z = random.randint(min_val, max_val)
            step1 = x + y
            step2 = (x + y) * z
            final = (x + y) + (x + y) * z
            op_type = "addition"
        else:
            template = random.choice(SUBTRACTION_TEMPLATES)
            x = random.randint(min_val + 1, max_val)
            y = random.randint(min_val, x - 1)
            z = random.randint(min_val, max_val)
            step1 = x - y
            step2 = (x - y) * z
            final = (x - y) + (x - y) * z
            op_type = "subtraction"
        
        prompt = template.format(X=x, Y=y, Z=z)
        
        prompts.append({
            "prompt": prompt,
            "type": op_type,
            "x": x, "y": y, "z": z,
            "step1_result": str(step1),
            "step2_result": str(step2),
            "final_answer": str(final),
            "ood_type": "large_numbers",
        })
    
    return prompts


# ============================================================================
# OOD TEST 2: Novel Entity Types (not in training data)
# ============================================================================

def create_novel_entity_prompts(n: int, seed: int = 888) -> list[dict]:
    """Create prompts with entity types NOT in the training data."""
    random.seed(seed)
    
    # These entities are NOT in the original templates
    NOVEL_ADDITION_TEMPLATES = [
        "A zoo starts with {X} animals. They acquire {Y} more animals. Then each current animal attracts {Z} additional visitors. How many visitors are there now at the zoo? Give the answer only and nothing else.",
        "A library starts with {X} books. They purchase {Y} more books. Then each current book inspires {Z} additional donations. How many donations are there now at the library? Give the answer only and nothing else.",
        "A garden starts with {X} plants. They plant {Y} more plants. Then each current plant produces {Z} additional seeds. How many seeds are there now in the garden? Give the answer only and nothing else.",
        "A hospital starts with {X} patients. They admit {Y} more patients. Then each current patient requires {Z} additional staff. How many staff are there now at the hospital? Give the answer only and nothing else.",
        "A farm starts with {X} cows. They buy {Y} more cows. Then each current cow produces {Z} gallons of milk. How many gallons of milk are there now on the farm? Give the answer only and nothing else.",
    ]
    
    NOVEL_SUBTRACTION_TEMPLATES = [
        "A zoo starts with {X} animals. {Y} animals are relocated. Then each remaining animal attracts {Z} additional visitors. How many visitors are there now at the zoo? Give the answer only and nothing else.",
        "A library starts with {X} books. {Y} books are lost. Then each remaining book inspires {Z} additional donations. How many donations are there now at the library? Give the answer only and nothing else.",
        "A garden starts with {X} plants. {Y} plants wilt. Then each remaining plant produces {Z} additional seeds. How many seeds are there now in the garden? Give the answer only and nothing else.",
        "A hospital starts with {X} patients. {Y} patients are discharged. Then each remaining patient requires {Z} additional staff. How many staff are there now at the hospital? Give the answer only and nothing else.",
        "A farm starts with {X} cows. {Y} cows are sold. Then each remaining cow produces {Z} gallons of milk. How many gallons of milk are there now on the farm? Give the answer only and nothing else.",
    ]
    
    prompts = []
    for i in range(n):
        is_addition = random.random() < 0.5
        
        if is_addition:
            template = random.choice(NOVEL_ADDITION_TEMPLATES)
            x = random.randint(1, 10)
            y = random.randint(1, 10)
            z = random.randint(1, 10)
            step1 = x + y
            step2 = (x + y) * z
            final = (x + y) + (x + y) * z
            op_type = "addition"
        else:
            template = random.choice(NOVEL_SUBTRACTION_TEMPLATES)
            x = random.randint(2, 10)
            y = random.randint(1, x - 1)
            z = random.randint(1, 10)
            step1 = x - y
            step2 = (x - y) * z
            final = (x - y) + (x - y) * z
            op_type = "subtraction"
        
        prompt = template.format(X=x, Y=y, Z=z)
        
        prompts.append({
            "prompt": prompt,
            "type": op_type,
            "x": x, "y": y, "z": z,
            "step1_result": str(step1),
            "step2_result": str(step2),
            "final_answer": str(final),
            "ood_type": "novel_entity",
        })
    
    return prompts


# ============================================================================
# OOD TEST 3: Edge Cases
# ============================================================================

def create_edge_case_prompts(seed: int = 777) -> list[dict]:
    """Create prompts with edge case values."""
    random.seed(seed)
    
    TEMPLATE_ADD = "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."
    TEMPLATE_SUB = "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."
    
    prompts = []
    
    # Edge case: step1 = 1 (minimal intermediate)
    prompts.append({
        "prompt": TEMPLATE_SUB.format(X=2, Y=1, Z=5),
        "type": "subtraction", "x": 2, "y": 1, "z": 5,
        "step1_result": "1", "step2_result": "5", "final_answer": "6",
        "ood_type": "edge_step1_is_1",
    })
    
    # Edge case: step2 = step1 (z=1)
    prompts.append({
        "prompt": TEMPLATE_ADD.format(X=3, Y=4, Z=1),
        "type": "addition", "x": 3, "y": 4, "z": 1,
        "step1_result": "7", "step2_result": "7", "final_answer": "14",
        "ood_type": "edge_step2_equals_step1",
    })
    
    # Edge case: large result (max possible with 1-10)
    prompts.append({
        "prompt": TEMPLATE_ADD.format(X=10, Y=10, Z=10),
        "type": "addition", "x": 10, "y": 10, "z": 10,
        "step1_result": "20", "step2_result": "200", "final_answer": "220",
        "ood_type": "edge_large_result",
    })
    
    # Edge case: all same number
    prompts.append({
        "prompt": TEMPLATE_ADD.format(X=5, Y=5, Z=5),
        "type": "addition", "x": 5, "y": 5, "z": 5,
        "step1_result": "10", "step2_result": "50", "final_answer": "60",
        "ood_type": "edge_all_same",
    })
    
    # Edge case: step1 = step2 result (requires specific numbers)
    # (X+Y) = (X+Y)*Z means Z=1, covered above
    # Let's do X-Y=1, Z=1 so step1=step2=1
    prompts.append({
        "prompt": TEMPLATE_SUB.format(X=3, Y=2, Z=1),
        "type": "subtraction", "x": 3, "y": 2, "z": 1,
        "step1_result": "1", "step2_result": "1", "final_answer": "2",
        "ood_type": "edge_both_steps_1",
    })
    
    # Edge case: small numbers (1,1,1)
    prompts.append({
        "prompt": TEMPLATE_ADD.format(X=1, Y=1, Z=1),
        "type": "addition", "x": 1, "y": 1, "z": 1,
        "step1_result": "2", "step2_result": "2", "final_answer": "4",
        "ood_type": "edge_minimal",
    })
    
    return prompts


# ============================================================================
# OOD TEST 4: Novel Question Phrasings
# ============================================================================

NOVEL_QUESTIONS = [
    # These are NOT in the training data
    "Tell me the number stored here.",
    "What value does this represent?",
    "Decode this activation.",
    "What number is encoded?",
    "Extract the numerical value.",
    "What did the model compute?",
    "Read the latent representation.",
]

TRAINED_QUESTION = "What is the intermediate calculation result?"


# ============================================================================
# OOD TEST 5: GSM8k (optional, requires datasets library)
# ============================================================================

def load_gsm8k_prompts(n: int, seed: int = 666) -> list[dict]:
    """Load real GSM8k problems."""
    try:
        from datasets import load_dataset
        import re
    except ImportError:
        print("  [!] datasets library not installed, skipping GSM8k test")
        return []
    
    print("  Loading GSM8k dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n * 2, len(dataset)))
    
    prompts = []
    for idx in indices:
        if len(prompts) >= n:
            break
            
        item = dataset[idx]
        question = item["question"]
        answer = item["answer"]
        
        # Parse intermediate steps
        calc_pattern = r"<<[^>]*=\s*([^>]+)>>"
        intermediate_results = re.findall(calc_pattern, answer)
        intermediate_results = [r.strip().replace(",", "") for r in intermediate_results]
        
        # Need at least 2 steps
        if len(intermediate_results) < 2:
            continue
        
        final_match = re.search(r"####\s*(.+)", answer)
        final_answer = final_match.group(1).strip().replace(",", "") if final_match else ""
        
        prompt = f"{question} Give the answer only and nothing else."
        
        prompts.append({
            "prompt": prompt,
            "step1_result": intermediate_results[0],
            "step2_result": intermediate_results[1] if len(intermediate_results) > 1 else None,
            "final_answer": final_answer,
            "num_steps": len(intermediate_results),
            "ood_type": "gsm8k",
        })
    
    return prompts


def evaluate_prompts(wrapper, ao, prompts, test_name: str, use_novel_questions: bool = False):
    """Evaluate a set of prompts and return results."""
    
    z2_ao_correct = 0
    z4_ao_correct = 0
    z2_ll_correct = 0
    z4_ll_correct = 0
    codi_correct = 0
    total = 0
    errors = []
    
    # Cross-analysis counters
    latent_ok_codi_ok = 0
    latent_ok_codi_fail = 0
    latent_fail_codi_ok = 0
    latent_fail_codi_fail = 0
    
    question = random.choice(NOVEL_QUESTIONS) if use_novel_questions else TRAINED_QUESTION
    
    # Debug: show first few CODI outputs
    debug_count = 0
    
    for item in tqdm(prompts, desc=f"  {test_name}"):
        result = wrapper.collect_latents(
            prompt=item["prompt"],
            ground_truth_answer=item.get("final_answer", "0"),
        )
        
        # Debug: print first 3 CODI outputs to verify extraction
        if debug_count < 3:
            # Extract just the generated part (after the prompt)
            full_output = result.predicted_answer or ""
            prompt_text = item["prompt"]
            generated = full_output[len(prompt_text):].strip() if full_output.startswith(prompt_text) else full_output
            tqdm.write(f"    DEBUG: GT={item.get('final_answer')}, CODI output='{generated[:50]}', is_correct={result.is_correct}")
            debug_count += 1
        
        # Track CODI's final answer correctness
        codi_is_correct = result.is_correct
        if codi_is_correct:
            codi_correct += 1
        
        # Test z2 (step 1)
        z2_latent = result.latent_vectors[1]
        
        # Logit lens for z2
        z2_ll_result = wrapper.logit_lens(z2_latent)
        z2_ll_pred, _ = z2_ll_result.get_top1_at_final_layer()
        z2_ll_pred = z2_ll_pred.strip() if z2_ll_pred else ""
        z2_ll_ok = z2_ll_pred == item["step1_result"]
        if z2_ll_ok:
            z2_ll_correct += 1
        
        # AO for z2
        z2_ao_prompt = ao.create_prompt(question=question, activation_vectors=[z2_latent])
        z2_ao_output = ao.generate(z2_ao_prompt).strip()
        z2_ao_ok = z2_ao_output == item["step1_result"]
        if z2_ao_ok:
            z2_ao_correct += 1
        
        # Test z4 (step 2) if available
        z4_ll_pred = "N/A"
        z4_ao_output = "N/A"
        z4_ao_ok = True
        z4_ll_ok = True
        if item.get("step2_result"):
            z4_latent = result.latent_vectors[3]
            
            # Logit lens for z4
            z4_ll_result = wrapper.logit_lens(z4_latent)
            z4_ll_pred, _ = z4_ll_result.get_top1_at_final_layer()
            z4_ll_pred = z4_ll_pred.strip() if z4_ll_pred else ""
            z4_ll_ok = z4_ll_pred == item["step2_result"]
            if z4_ll_ok:
                z4_ll_correct += 1
            
            # AO for z4
            z4_ao_prompt = ao.create_prompt(question=question, activation_vectors=[z4_latent])
            z4_ao_output = ao.generate(z4_ao_prompt).strip()
            z4_ao_ok = z4_ao_output == item["step2_result"]
            if z4_ao_ok:
                z4_ao_correct += 1
        
        total += 1
        
        # Cross-analysis: latent correctness (AO) vs CODI correctness
        latent_ok = z2_ao_ok and z4_ao_ok
        if codi_is_correct is not None:
            if latent_ok and codi_is_correct:
                latent_ok_codi_ok += 1
            elif latent_ok and not codi_is_correct:
                latent_ok_codi_fail += 1
            elif not latent_ok and codi_is_correct:
                latent_fail_codi_ok += 1
            else:
                latent_fail_codi_fail += 1
        
        if not z2_ao_ok or not z4_ao_ok:
            errors.append({
                "prompt": item["prompt"][:60] + "...",
                "ood_type": item.get("ood_type", "unknown"),
                "codi_correct": codi_is_correct,
                "step1_expected": item["step1_result"],
                "step1_ao": z2_ao_output,
                "step1_ll": z2_ll_pred,
                "step2_expected": item.get("step2_result", "N/A"),
                "step2_ao": z4_ao_output if item.get("step2_result") else "N/A",
                "step2_ll": z4_ll_pred if item.get("step2_result") else "N/A",
                "z2_ao_ok": z2_ao_ok,
                "z4_ao_ok": z4_ao_ok,
            })
    
    return {
        "z2_ao": z2_ao_correct,
        "z4_ao": z4_ao_correct,
        "z2_ll": z2_ll_correct,
        "z4_ll": z4_ll_correct,
        "codi_correct": codi_correct,
        "total": total,
        "errors": errors,
        # Cross-analysis
        "latent_ok_codi_ok": latent_ok_codi_ok,
        "latent_ok_codi_fail": latent_ok_codi_fail,
        "latent_fail_codi_ok": latent_fail_codi_ok,
        "latent_fail_codi_fail": latent_fail_codi_fail,
    }


def print_results(name: str, results: dict):
    """Print results for a test."""
    total = results["total"]
    if total == 0:
        print(f"  {name}: No samples")
        return
    
    z2_ao_pct = 100 * results["z2_ao"] / total
    z4_ao_pct = 100 * results["z4_ao"] / total
    z2_ll_pct = 100 * results["z2_ll"] / total
    z4_ll_pct = 100 * results["z4_ll"] / total
    codi_pct = 100 * results["codi_correct"] / total
    
    combined_ao = (results["z2_ao"] + results["z4_ao"]) / (total * 2) * 100
    combined_ll = (results["z2_ll"] + results["z4_ll"]) / (total * 2) * 100
    
    print(f"  {name}:")
    print(f"    CODI final answer: {codi_pct:.1f}%")
    print(f"    z2: AO={z2_ao_pct:.1f}%, LL={z2_ll_pct:.1f}%")
    print(f"    z4: AO={z4_ao_pct:.1f}%, LL={z4_ll_pct:.1f}%")
    print(f"    Combined latent: AO={combined_ao:.1f}%, LL={combined_ll:.1f}%")
    
    # Cross-analysis
    cross_total = (results["latent_ok_codi_ok"] + results["latent_ok_codi_fail"] + 
                   results["latent_fail_codi_ok"] + results["latent_fail_codi_fail"])
    if cross_total > 0:
        print(f"    Cross-analysis:")
        print(f"      Latent ✓, CODI ✓: {results['latent_ok_codi_ok']:3d} ({100*results['latent_ok_codi_ok']/cross_total:.1f}%)")
        print(f"      Latent ✓, CODI ✗: {results['latent_ok_codi_fail']:3d} ({100*results['latent_ok_codi_fail']/cross_total:.1f}%)  <- Right latent, wrong answer!")
        print(f"      Latent ✗, CODI ✓: {results['latent_fail_codi_ok']:3d} ({100*results['latent_fail_codi_ok']/cross_total:.1f}%)")
        print(f"      Latent ✗, CODI ✗: {results['latent_fail_codi_fail']:3d} ({100*results['latent_fail_codi_fail']/cross_total:.1f}%)")
    
    return combined_ao


def run_all_ood_tests(n_samples: int = 50, max_val: int = 100, include_gsm8k: bool = False, ao_path: str = "checkpoints/ao"):
    """Run all OOD tests."""
    
    print(f"\n{'='*70}")
    print("OUT-OF-DISTRIBUTION SANITY CHECK")
    print(f"{'='*70}\n")
    
    # Load models once
    print("Loading CODI model...")
    wrapper = CODIWrapper.from_pretrained()
    
    print(f"Loading Activation Oracle from {ao_path}...")
    ao = ActivationOracle.from_pretrained(lora_path=ao_path)
    
    all_results = {}
    
    # -------------------------------------------------------------------------
    # TEST 1: Baseline (1-10, trained distribution)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("TEST 1: BASELINE (numbers 1-10, trained question)")
    print(f"{'-'*70}")
    prompts = create_large_number_prompts(n_samples, min_val=1, max_val=10, seed=111)
    all_results["baseline"] = evaluate_prompts(wrapper, ao, prompts, "Baseline 1-10")
    print_results("Baseline (1-10)", all_results["baseline"])
    
    # -------------------------------------------------------------------------
    # TEST 2: Large Numbers
    # -------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"TEST 2: LARGE NUMBERS (1-{max_val})")
    print(f"{'-'*70}")
    prompts = create_large_number_prompts(n_samples, min_val=1, max_val=max_val, seed=222)
    all_results["large_numbers"] = evaluate_prompts(wrapper, ao, prompts, f"Numbers 1-{max_val}")
    print_results(f"Large numbers (1-{max_val})", all_results["large_numbers"])
    
    # -------------------------------------------------------------------------
    # TEST 3: Novel Entity Types
    # -------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("TEST 3: NOVEL ENTITY TYPES (zoo, library, etc.)")
    print(f"{'-'*70}")
    prompts = create_novel_entity_prompts(n_samples, seed=333)
    all_results["novel_entities"] = evaluate_prompts(wrapper, ao, prompts, "Novel entities")
    print_results("Novel entities", all_results["novel_entities"])
    
    # -------------------------------------------------------------------------
    # TEST 4: Edge Cases
    # -------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("TEST 4: EDGE CASES")
    print(f"{'-'*70}")
    prompts = create_edge_case_prompts(seed=444)
    all_results["edge_cases"] = evaluate_prompts(wrapper, ao, prompts, "Edge cases")
    print_results("Edge cases", all_results["edge_cases"])
    
    # Show edge case details
    print("\n  Edge case details:")
    for item in create_edge_case_prompts(seed=444):
        print(f"    {item['ood_type']}: step1={item['step1_result']}, step2={item['step2_result']}")
    
    # -------------------------------------------------------------------------
    # TEST 5: Novel Question Phrasings
    # -------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("TEST 5: NOVEL QUESTION PHRASINGS")
    print(f"{'-'*70}")
    prompts = create_large_number_prompts(n_samples, min_val=1, max_val=10, seed=555)
    all_results["novel_questions"] = evaluate_prompts(
        wrapper, ao, prompts, "Novel questions", use_novel_questions=True
    )
    print_results("Novel questions", all_results["novel_questions"])
    print(f"\n  Sample novel questions used: {NOVEL_QUESTIONS[:3]}")
    
    # -------------------------------------------------------------------------
    # TEST 6: GSM8k (optional)
    # -------------------------------------------------------------------------
    if include_gsm8k:
        print(f"\n{'-'*70}")
        print("TEST 6: REAL GSM8K PROBLEMS")
        print(f"{'-'*70}")
        prompts = load_gsm8k_prompts(n_samples, seed=666)
        if prompts:
            all_results["gsm8k"] = evaluate_prompts(wrapper, ao, prompts, "GSM8k")
            print_results("GSM8k", all_results["gsm8k"])
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    baseline_ao = (all_results["baseline"]["z2_ao"] + all_results["baseline"]["z4_ao"]) / (all_results["baseline"]["total"] * 2) * 100
    baseline_codi = 100 * all_results["baseline"]["codi_correct"] / all_results["baseline"]["total"]
    
    print(f"{'Test':<25} {'CODI Final':<12} {'AO Latent':<12} {'vs Baseline':<12}")
    print(f"{'-'*65}")
    
    for name, results in all_results.items():
        if results["total"] == 0:
            continue
        ao_acc = (results["z2_ao"] + results["z4_ao"]) / (results["total"] * 2) * 100
        codi_acc = 100 * results["codi_correct"] / results["total"]
        diff = ao_acc - baseline_ao
        diff_str = f"{diff:+.1f}%" if name != "baseline" else "-"
        print(f"{name:<25} {codi_acc:.1f}%{'':<5} {ao_acc:.1f}%{'':<5} {diff_str}")
    
    # Cross-analysis summary
    print(f"\n{'='*70}")
    print("CROSS-ANALYSIS SUMMARY (Latent correct vs CODI correct)")
    print(f"{'='*70}\n")
    
    print(f"{'Test':<20} {'L✓C✓':<10} {'L✓C✗':<10} {'L✗C✓':<10} {'L✗C✗':<10}")
    print(f"{'-'*60}")
    for name, results in all_results.items():
        if results["total"] == 0:
            continue
        t = results["total"]
        lok_cok = results["latent_ok_codi_ok"]
        lok_cf = results["latent_ok_codi_fail"]
        lf_cok = results["latent_fail_codi_ok"]
        lf_cf = results["latent_fail_codi_fail"]
        print(f"{name:<20} {lok_cok:>3} ({100*lok_cok/t:4.1f}%) {lok_cf:>3} ({100*lok_cf/t:4.1f}%) {lf_cok:>3} ({100*lf_cok/t:4.1f}%) {lf_cf:>3} ({100*lf_cf/t:4.1f}%)")
    
    print("\n  L✓C✗ = Latent has correct intermediate, but CODI outputs wrong final answer (interesting!)")
    
    # Show sample errors
    print(f"\n{'='*70}")
    print("SAMPLE ERRORS (up to 5 per test)")
    print(f"{'='*70}\n")
    
    for name, results in all_results.items():
        if results["errors"]:
            print(f"\n{name}:")
            for err in results["errors"][:5]:
                codi_status = "CODI✓" if err.get('codi_correct') else "CODI✗"
                print(f"  [{err['ood_type']}] {codi_status} | step1: exp={err['step1_expected']}, ao={err['step1_ao']} {'✓' if err['z2_ao_ok'] else '✗'}")
                print(f"              step2: exp={err['step2_expected']}, ao={err['step2_ao']} {'✓' if err['z4_ao_ok'] else '✗'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOD Sanity Check for Activation Oracle")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of test prompts per test")
    parser.add_argument("--max_val", type=int, default=100, help="Maximum number for large number test")
    parser.add_argument("--gsm8k", action="store_true", help="Include GSM8k test (requires datasets library)")
    parser.add_argument("--ao_path", type=str, default="checkpoints/ao", help="Path to AO checkpoint")
    args = parser.parse_args()
    
    run_all_ood_tests(
        n_samples=args.n_samples,
        max_val=args.max_val,
        include_gsm8k=args.gsm8k,
        ao_path=args.ao_path,
    )
