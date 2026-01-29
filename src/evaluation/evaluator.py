"""
Evaluation harness for CODI Activation Oracle.

Supports evaluation of:
1. Intermediate result extraction (z2, z4)
2. Operation classification
3. Comparison with baselines (logit lens, linear probe)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .baselines import LogitLensBaseline, LinearProbeBaseline, BaselineResult


@dataclass
class EvaluationResult:
    """Result of evaluating on a single example."""
    
    prompt: str  # Original CODI prompt (truncated)
    full_prompt: str = ""  # Full original CODI prompt
    ground_truth: str = ""
    
    # AO input/output
    ao_input_prompt: Optional[str] = None  # Full prompt sent to AO
    ao_prediction: Optional[str] = None  # Raw AO output
    ao_is_correct: Optional[bool] = None
    
    # Baseline predictions
    logit_lens_prediction: Optional[str] = None
    logit_lens_is_correct: Optional[bool] = None
    
    linear_probe_prediction: Optional[str] = None
    linear_probe_is_correct: Optional[bool] = None
    
    # Metadata
    latent_position: int = -1
    question_type: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    
    # Overall metrics
    total_examples: int = 0
    
    # AO metrics
    ao_accuracy: float = 0.0
    ao_correct: int = 0
    
    # Baseline metrics
    logit_lens_accuracy: float = 0.0
    logit_lens_correct: int = 0
    
    linear_probe_accuracy: float = 0.0
    linear_probe_correct: int = 0
    
    # Per-position metrics
    z2_ao_accuracy: float = 0.0
    z4_ao_accuracy: float = 0.0
    z2_logit_lens_accuracy: float = 0.0
    z4_logit_lens_accuracy: float = 0.0
    
    # Detailed results
    results: list[EvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d
    
    def save(self, path: str):
        """Save summary to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class CODIAOEvaluator:
    """
    Evaluator for CODI Activation Oracle.
    
    Usage:
        evaluator = CODIAOEvaluator(
            codi_wrapper=codi_wrapper,
            activation_oracle=ao,
        )
        
        summary = evaluator.evaluate_intermediate_results(
            test_prompts=test_data,
        )
        
        print(f"AO accuracy: {summary.ao_accuracy:.2%}")
        print(f"Logit Lens accuracy: {summary.logit_lens_accuracy:.2%}")
    """
    
    def __init__(
        self,
        codi_wrapper,
        activation_oracle=None,
        linear_probe: Optional[LinearProbeBaseline] = None,
    ):
        self.codi_wrapper = codi_wrapper
        self.ao = activation_oracle
        self.logit_lens = LogitLensBaseline.from_codi_wrapper(codi_wrapper)
        self.linear_probe = linear_probe
        
        # Position to step mapping
        self.position_to_step = {
            1: 0,  # z2 → first intermediate result
            3: 1,  # z4 → second intermediate result
        }
    
    def evaluate_intermediate_results(
        self,
        test_prompts: list[dict],
        positions: list[int] = [1, 3],
        verbose: bool = True,
    ) -> EvaluationSummary:
        """
        Evaluate intermediate result extraction on test prompts.
        
        Args:
            test_prompts: List of dicts with 'prompt' and step results
            positions: Latent positions to evaluate (default: z2 and z4)
            verbose: Show progress
        
        Returns:
            EvaluationSummary with all metrics
        """
        results = []
        
        # Counters
        ao_correct = {"total": 0, 1: 0, 3: 0}
        ll_correct = {"total": 0, 1: 0, 3: 0}
        lp_correct = {"total": 0, 1: 0, 3: 0}
        totals = {"total": 0, 1: 0, 3: 0}
        
        iterator = tqdm(test_prompts, desc="Evaluating", disable=not verbose)
        
        for item in iterator:
            prompt = item["prompt"]
            
            # Collect latent vectors
            codi_result = self.codi_wrapper.collect_latents(prompt)
            
            if len(codi_result.latent_vectors) < max(positions) + 1:
                continue
            
            for pos in positions:
                step_idx = self.position_to_step.get(pos, 0)
                
                # Get ground truth
                # pos == 1 is z2 (step 1), pos == 3 is z4 (step 2)
                if "step1_result" in item and pos == 1:
                    gt = str(item["step1_result"])
                elif "step2_result" in item and pos == 3:
                    gt = str(item["step2_result"])
                elif "results" in item and step_idx < len(item["results"]):
                    gt = str(item["results"][step_idx])
                else:
                    continue
                
                latent_vec = codi_result.latent_vectors[pos]
                
                # Logit Lens prediction
                ll_result = self.logit_lens.predict(latent_vec)
                ll_result.ground_truth = gt
                ll_correct_flag = ll_result._check_correct()
                
                # AO prediction (if available)
                ao_pred = None
                ao_correct_flag = None
                ao_input_prompt = None
                if self.ao is not None:
                    # Use ao.create_prompt() to ensure placeholder token consistency
                    ao_input_prompt = self.ao.create_prompt(
                        question="What is the intermediate calculation result?",
                        activation_vectors=[latent_vec],
                    )
                    ao_pred = self.ao.generate(ao_input_prompt)
                    ao_correct_flag = self._check_match(ao_pred, gt)
                
                # Linear Probe prediction (if available)
                lp_pred = None
                lp_correct_flag = None
                if self.linear_probe is not None:
                    lp_result = self.linear_probe.predict(latent_vec)
                    lp_pred = lp_result.prediction
                    lp_result.ground_truth = gt
                    lp_correct_flag = lp_result._check_correct()
                
                # Record result
                eval_result = EvaluationResult(
                    prompt=prompt[:100] + "...",
                    full_prompt=prompt,
                    ground_truth=gt,
                    ao_input_prompt=ao_input_prompt,
                    ao_prediction=ao_pred,
                    ao_is_correct=ao_correct_flag,
                    logit_lens_prediction=ll_result.prediction,
                    logit_lens_is_correct=ll_correct_flag,
                    linear_probe_prediction=lp_pred,
                    linear_probe_is_correct=lp_correct_flag,
                    latent_position=pos,
                    question_type="intermediate_result",
                )
                results.append(eval_result)
                
                # Update counters
                totals["total"] += 1
                totals[pos] += 1
                
                if ll_correct_flag:
                    ll_correct["total"] += 1
                    ll_correct[pos] += 1
                
                if ao_correct_flag:
                    ao_correct["total"] += 1
                    ao_correct[pos] += 1
                
                if lp_correct_flag:
                    lp_correct["total"] += 1
                    lp_correct[pos] += 1
        
        # Compute summary
        summary = EvaluationSummary(
            total_examples=totals["total"],
            ao_accuracy=ao_correct["total"] / totals["total"] if totals["total"] > 0 else 0,
            ao_correct=ao_correct["total"],
            logit_lens_accuracy=ll_correct["total"] / totals["total"] if totals["total"] > 0 else 0,
            logit_lens_correct=ll_correct["total"],
            linear_probe_accuracy=lp_correct["total"] / totals["total"] if totals["total"] > 0 else 0,
            linear_probe_correct=lp_correct["total"],
            z2_ao_accuracy=ao_correct[1] / totals[1] if totals[1] > 0 else 0,
            z4_ao_accuracy=ao_correct[3] / totals[3] if totals[3] > 0 else 0,
            z2_logit_lens_accuracy=ll_correct[1] / totals[1] if totals[1] > 0 else 0,
            z4_logit_lens_accuracy=ll_correct[3] / totals[3] if totals[3] > 0 else 0,
            results=results,
        )
        
        if verbose:
            self._print_summary(summary)
        
        return summary
    
    def evaluate_classification(
        self,
        test_examples,
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate yes/no classification accuracy.
        
        Args:
            test_examples: List of dicts or ClassificationExample objects with:
                - latent_vectors (list of lists) or latent_vector (single list)
                - question: str
                - answer: "Yes" or "No"
                - question_type or classification_type: str
            verbose: Show progress
        
        Returns:
            Dict with accuracy metrics
        """
        if self.ao is None:
            raise ValueError("Activation Oracle not provided")
        
        correct = 0
        total = 0
        results_by_type = {}
        examples = []  # Track individual examples
        
        iterator = tqdm(test_examples, desc="Evaluating classification", disable=not verbose)
        
        for ex in iterator:
            # Handle both dict and object formats
            if isinstance(ex, dict):
                # New format: latent_vectors (list of lists)
                if "latent_vectors" in ex:
                    latent_vecs = [torch.tensor(v) for v in ex["latent_vectors"]]
                elif "latent_vector" in ex:
                    latent_vecs = [torch.tensor(ex["latent_vector"])]
                else:
                    continue
                question = ex.get("question", "")
                ground_truth = ex.get("answer", "").lower()
                ctype = ex.get("question_type", ex.get("classification_type", "unknown"))
            else:
                # Old format: ClassificationExample object
                if hasattr(ex, "latent_vectors"):
                    latent_vecs = [torch.tensor(v) for v in ex.latent_vectors]
                elif hasattr(ex, "latent_vector"):
                    latent_vecs = [torch.tensor(ex.latent_vector)]
                else:
                    continue
                question = ex.question
                ground_truth = ex.answer.lower()
                ctype = getattr(ex, "classification_type", getattr(ex, "question_type", "unknown"))
            
            # Use ao.create_prompt() to ensure placeholder token consistency
            ao_prompt = self.ao.create_prompt(
                question=question,
                activation_vectors=latent_vecs,
            )
            
            prediction = self.ao.generate(ao_prompt).strip().lower()
            
            # Check if prediction contains yes/no
            is_correct = False
            if "yes" in prediction and ground_truth == "yes":
                is_correct = True
            elif "no" in prediction and ground_truth == "no":
                is_correct = True
            
            if is_correct:
                correct += 1
            total += 1
            
            # Track by type
            if ctype not in results_by_type:
                results_by_type[ctype] = {"correct": 0, "total": 0}
            results_by_type[ctype]["total"] += 1
            if is_correct:
                results_by_type[ctype]["correct"] += 1
            
            # Track individual example
            examples.append({
                "question": question,
                "question_type": ctype,
                "ground_truth": ground_truth,
                "ao_prediction": prediction,
                "is_correct": is_correct,
            })
        
        # Compute per-type accuracies
        for ctype in results_by_type:
            r = results_by_type[ctype]
            r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0
        
        summary = {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "by_type": results_by_type,
            "examples": examples,  # Include all examples
        }
        
        if verbose:
            print(f"\nClassification Accuracy: {summary['accuracy']:.2%} ({correct}/{total})")
            print("\nBy type:")
            for ctype, r in sorted(results_by_type.items()):
                print(f"  {ctype}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")
            
            # Print some examples
            print("\n" + "-" * 60)
            print("CLASSIFICATION EXAMPLES")
            print("-" * 60)
            for i, ex in enumerate(examples[:20]):  # Show first 20
                status = "✓" if ex["is_correct"] else "✗"
                print(f"\n[{i+1}] {ex['question_type']}")
                print(f"    Q: {ex['question']}")
                print(f"    Expected: {ex['ground_truth']}  |  AO: {ex['ao_prediction']}  {status}")
        
        return summary
    
    def evaluate_multi_latent_qa(
        self,
        test_prompts: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate multi-latent QA where all 6 latent vectors are provided.
        
        Tests questions like "What was the first step result?" when given all latents.
        
        Args:
            test_prompts: List of dicts with 'prompt', 'step1_result', 'step2_result', 'final_answer'
            verbose: Show progress
        
        Returns:
            Dict with accuracy metrics
        """
        if self.ao is None:
            raise ValueError("Activation Oracle not provided")
        
        from ..datasets.latent_qa import MULTI_LATENT_TEMPLATES
        import random
        
        results = {
            "step1": {"correct": 0, "total": 0},
            "step2": {"correct": 0, "total": 0},
            "final": {"correct": 0, "total": 0},
            "overall": {"correct": 0, "total": 0},
        }
        examples = []  # Track individual examples
        
        iterator = tqdm(test_prompts, desc="Evaluating multi-latent QA", disable=not verbose)
        
        for item in iterator:
            # Collect all 6 latent vectors
            codi_result = self.codi_wrapper.collect_latents(item["prompt"])
            
            if len(codi_result.latent_vectors) < 6:
                continue
            
            all_latents = [v for v in codi_result.latent_vectors[:6]]
            
            # Test different question types
            test_cases = [
                ("What was calculated in the first step?", str(item.get("step1_result", "")), "step1"),
                ("What was the result of the second calculation?", str(item.get("step2_result", "")), "step2"),
                ("What is the final answer?", str(item.get("final_answer", "")), "final"),
            ]
            
            for question, expected, key in test_cases:
                if not expected:
                    continue
                    
                ao_prompt = self.ao.create_prompt(
                    question=question,
                    activation_vectors=all_latents,
                )
                
                prediction = self.ao.generate(ao_prompt)
                is_correct = self._check_match(prediction, expected)
                
                results[key]["total"] += 1
                results["overall"]["total"] += 1
                
                if is_correct:
                    results[key]["correct"] += 1
                    results["overall"]["correct"] += 1
                
                # Track individual example
                examples.append({
                    "codi_prompt": item["prompt"][:100] + "...",
                    "question": question,
                    "question_type": key,
                    "ground_truth": expected,
                    "ao_prediction": prediction,
                    "is_correct": is_correct,
                })
        
        # Compute accuracies
        for key in results:
            r = results[key]
            r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0
        
        results["examples"] = examples  # Include examples
        
        if verbose:
            print(f"\nMulti-Latent QA Accuracy: {results['overall']['accuracy']:.2%} ({results['overall']['correct']}/{results['overall']['total']})")
            print("\nBy question type:")
            for key in ["step1", "step2", "final"]:
                r = results[key]
                print(f"  {key}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")
            
            # Print examples
            print("\n" + "-" * 60)
            print("MULTI-LATENT EXAMPLES")
            print("-" * 60)
            for i, ex in enumerate(examples[:15]):  # Show first 15
                status = "✓" if ex["is_correct"] else "✗"
                print(f"\n[{i+1}] {ex['question_type']}")
                print(f"    CODI: {ex['codi_prompt']}")
                print(f"    Q: {ex['question']}")
                print(f"    Expected: {ex['ground_truth']}  |  AO: {ex['ao_prediction']}  {status}")
        
        return results
    
    def _check_match(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        import re
        
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()
        
        # Try numeric comparison
        try:
            pred_nums = re.findall(r'[-]?\d+\.?\d*', pred)
            gt_nums = re.findall(r'[-]?\d+\.?\d*', gt)
            if pred_nums and gt_nums:
                return float(pred_nums[-1]) == float(gt_nums[-1])
        except (ValueError, IndexError):
            pass
        
        return pred == gt or pred in gt or gt in pred
    
    def _print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total examples: {summary.total_examples}")
        print()
        print("Overall Accuracy:")
        print(f"  Logit Lens:   {summary.logit_lens_accuracy:.2%} ({summary.logit_lens_correct}/{summary.total_examples})")
        if summary.ao_correct > 0 or summary.ao_accuracy > 0:
            print(f"  AO:           {summary.ao_accuracy:.2%} ({summary.ao_correct}/{summary.total_examples})")
        if summary.linear_probe_correct > 0 or summary.linear_probe_accuracy > 0:
            print(f"  Linear Probe: {summary.linear_probe_accuracy:.2%} ({summary.linear_probe_correct}/{summary.total_examples})")
        print()
        print("By Position:")
        print(f"  z2 Logit Lens: {summary.z2_logit_lens_accuracy:.2%}")
        print(f"  z4 Logit Lens: {summary.z4_logit_lens_accuracy:.2%}")
        if summary.z2_ao_accuracy > 0 or summary.z4_ao_accuracy > 0:
            print(f"  z2 AO:         {summary.z2_ao_accuracy:.2%}")
            print(f"  z4 AO:         {summary.z4_ao_accuracy:.2%}")
        print("=" * 60)
    
    def _print_all_examples(self, summary: EvaluationSummary):
        """Print all evaluation examples with predictions."""
        print("\n" + "=" * 80)
        print("ALL EVALUATION EXAMPLES")
        print("=" * 80)
        
        for i, result in enumerate(summary.results):
            pos_name = f"z{result.latent_position + 1}" if result.latent_position >= 0 else "?"
            ao_status = "✓" if result.ao_is_correct else "✗" if result.ao_is_correct is not None else "-"
            ll_status = "✓" if result.logit_lens_is_correct else "✗" if result.logit_lens_is_correct is not None else "-"
            
            print(f"\n{'─' * 80}")
            print(f"[{i+1}] Position: {pos_name}  |  Result: {ao_status}")
            print(f"{'─' * 80}")
            
            # Full original prompt (what CODI was asked)
            print(f"\n📝 CODI PROMPT:")
            print(f"   {result.full_prompt or result.prompt}")
            
            # Ground truth
            print(f"\n🎯 GROUND TRUTH: {result.ground_truth}")
            
            # Logit Lens baseline
            print(f"\n🔍 LOGIT LENS: {result.logit_lens_prediction} {ll_status}")
            
            # Full AO input prompt
            if result.ao_input_prompt is not None:
                print(f"\n📨 AO INPUT PROMPT:")
                # Show the prompt, replacing placeholder token with visible marker
                # The actual placeholder is " ?" (space + question mark)
                ao_display = result.ao_input_prompt.replace(" ?", " [LATENT] ")
                print(f"   {ao_display}")
            
            # Full AO output
            if result.ao_prediction is not None:
                print(f"\n📤 AO OUTPUT: {result.ao_prediction} {ao_status}")
            
            # Linear probe if available
            if result.linear_probe_prediction is not None:
                lp_status = "✓" if result.linear_probe_is_correct else "✗"
                print(f"\n📊 LINEAR PROBE: {result.linear_probe_prediction} {lp_status}")
        
        print(f"\n{'=' * 80}")
    
    def get_examples_as_dicts(self, summary: EvaluationSummary) -> list[dict]:
        """Convert evaluation results to list of dicts for JSON serialization."""
        examples = []
        for result in summary.results:
            examples.append({
                "codi_prompt": result.full_prompt or result.prompt,
                "ground_truth": result.ground_truth,
                "latent_position": result.latent_position,
                "ao_input_prompt": result.ao_input_prompt,
                "ao_output": result.ao_prediction,
                "ao_is_correct": result.ao_is_correct,
                "logit_lens_prediction": result.logit_lens_prediction,
                "logit_lens_is_correct": result.logit_lens_is_correct,
                "linear_probe_prediction": result.linear_probe_prediction,
                "linear_probe_is_correct": result.linear_probe_is_correct,
            })
        return examples


def run_mvp_evaluation(
    codi_wrapper,
    test_prompts: list[dict],
    verbose: bool = True,
) -> EvaluationSummary:
    """
    Run MVP evaluation (logit lens baseline only, no AO).
    
    This is the first validation step before training AO.
    
    Args:
        codi_wrapper: CODIWrapper instance
        test_prompts: Test prompts with ground truth
        verbose: Show progress
    
    Returns:
        EvaluationSummary
    """
    evaluator = CODIAOEvaluator(
        codi_wrapper=codi_wrapper,
        activation_oracle=None,
    )
    
    return evaluator.evaluate_intermediate_results(
        test_prompts=test_prompts,
        positions=[1, 3],  # z2 and z4
        verbose=verbose,
    )


if __name__ == "__main__":
    print("Evaluator module - run with actual data via scripts/evaluate.py")
