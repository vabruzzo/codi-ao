"""
Evaluation harness for CODI Activation Oracle.

Supports evaluation of:
1. Intermediate result extraction (z3, z5)
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
    
    prompt: str
    ground_truth: str
    
    # AO prediction
    ao_prediction: Optional[str] = None
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
    z3_ao_accuracy: float = 0.0
    z5_ao_accuracy: float = 0.0
    z3_logit_lens_accuracy: float = 0.0
    z5_logit_lens_accuracy: float = 0.0
    
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
            2: 0,  # z3 → first intermediate result
            4: 1,  # z5 → second intermediate result
        }
    
    def evaluate_intermediate_results(
        self,
        test_prompts: list[dict],
        positions: list[int] = [2, 4],
        verbose: bool = True,
    ) -> EvaluationSummary:
        """
        Evaluate intermediate result extraction on test prompts.
        
        Args:
            test_prompts: List of dicts with 'prompt' and step results
            positions: Latent positions to evaluate (default: z3 and z5)
            verbose: Show progress
        
        Returns:
            EvaluationSummary with all metrics
        """
        results = []
        
        # Counters
        ao_correct = {"total": 0, 2: 0, 4: 0}
        ll_correct = {"total": 0, 2: 0, 4: 0}
        lp_correct = {"total": 0, 2: 0, 4: 0}
        totals = {"total": 0, 2: 0, 4: 0}
        
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
                if "step1_result" in item and pos == 2:
                    gt = str(item["step1_result"])
                elif "step2_result" in item and pos == 4:
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
                if self.ao is not None:
                    # Use ao.create_prompt() to ensure placeholder token consistency
                    ao_prompt = self.ao.create_prompt(
                        question="What is the intermediate calculation result?",
                        activation_vectors=[latent_vec],
                        layer_percent=50,
                    )
                    ao_pred = self.ao.generate(ao_prompt)
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
                    ground_truth=gt,
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
            z3_ao_accuracy=ao_correct[2] / totals[2] if totals[2] > 0 else 0,
            z5_ao_accuracy=ao_correct[4] / totals[4] if totals[4] > 0 else 0,
            z3_logit_lens_accuracy=ll_correct[2] / totals[2] if totals[2] > 0 else 0,
            z5_logit_lens_accuracy=ll_correct[4] / totals[4] if totals[4] > 0 else 0,
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
            test_examples: List of ClassificationExample objects
            verbose: Show progress
        
        Returns:
            Dict with accuracy metrics
        """
        if self.ao is None:
            raise ValueError("Activation Oracle not provided")
        
        correct = 0
        total = 0
        results_by_type = {}
        
        iterator = tqdm(test_examples, desc="Evaluating classification", disable=not verbose)
        
        for ex in iterator:
            latent_vec = torch.tensor(ex.latent_vector)
            
            # Use ao.create_prompt() to ensure placeholder token consistency
            ao_prompt = self.ao.create_prompt(
                question=ex.question,
                activation_vectors=[latent_vec],
                layer_percent=ex.layer_percent,
            )
            
            prediction = self.ao.generate(ao_prompt).strip().lower()
            ground_truth = ex.answer.lower()
            
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
            ctype = ex.classification_type
            if ctype not in results_by_type:
                results_by_type[ctype] = {"correct": 0, "total": 0}
            results_by_type[ctype]["total"] += 1
            if is_correct:
                results_by_type[ctype]["correct"] += 1
        
        # Compute per-type accuracies
        for ctype in results_by_type:
            r = results_by_type[ctype]
            r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0
        
        summary = {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "by_type": results_by_type,
        }
        
        if verbose:
            print(f"\nClassification Accuracy: {summary['accuracy']:.2%}")
            print("\nBy type:")
            for ctype, r in results_by_type.items():
                print(f"  {ctype}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")
        
        return summary
    
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
        print(f"  z3 Logit Lens: {summary.z3_logit_lens_accuracy:.2%}")
        print(f"  z5 Logit Lens: {summary.z5_logit_lens_accuracy:.2%}")
        if summary.z3_ao_accuracy > 0 or summary.z5_ao_accuracy > 0:
            print(f"  z3 AO:         {summary.z3_ao_accuracy:.2%}")
            print(f"  z5 AO:         {summary.z5_ao_accuracy:.2%}")
        print("=" * 60)


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
        positions=[2, 4],
        verbose=verbose,
    )


if __name__ == "__main__":
    print("Evaluator module - run with actual data via scripts/evaluate.py")
