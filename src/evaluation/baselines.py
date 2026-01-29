"""
Baseline methods for comparison with Activation Oracle.

Baselines:
1. Logit Lens: Project latent vectors to vocabulary space
2. Linear Probe: Train a simple classifier on latent vectors
3. PatchScopes: Zero-shot AO (no LoRA training)
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    
    method: str
    prediction: str
    confidence: float
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    
    def __post_init__(self):
        if self.ground_truth is not None:
            self.is_correct = self._check_correct()
    
    def _check_correct(self) -> bool:
        """Check if prediction matches ground truth."""
        pred = self.prediction.strip().lower()
        gt = self.ground_truth.strip().lower()
        
        # Try numeric comparison
        try:
            pred_num = float(re.findall(r'[-]?\d+\.?\d*', pred)[-1])
            gt_num = float(re.findall(r'[-]?\d+\.?\d*', gt)[-1])
            return pred_num == gt_num
        except (IndexError, ValueError):
            pass
        
        # String comparison
        return pred == gt or pred in gt or gt in pred


class LogitLensBaseline:
    """
    Logit Lens baseline: project latent vectors to vocabulary space.
    
    This replicates the analysis from the LessWrong post, which found
    that intermediate results appear with high probability in logit lens.
    """
    
    def __init__(
        self,
        lm_head: torch.nn.Module,
        layer_norm: Optional[torch.nn.Module],
        tokenizer,
        device: str = "cuda",
    ):
        self.lm_head = lm_head
        self.layer_norm = layer_norm
        self.tokenizer = tokenizer
        self.device = device
    
    @classmethod
    def from_codi_wrapper(cls, wrapper) -> "LogitLensBaseline":
        """Create LogitLens from a CODIWrapper instance."""
        return cls(
            lm_head=wrapper._lm_head,
            layer_norm=wrapper._layer_norm,
            tokenizer=wrapper.tokenizer,
            device=wrapper.device,
        )
    
    def predict(
        self,
        latent_vector: torch.Tensor,
        top_k: int = 1,
        apply_layer_norm: bool = True,
    ) -> BaselineResult:
        """
        Apply logit lens to predict the most likely token.
        
        Args:
            latent_vector: Latent vector of shape (hidden_dim,)
            top_k: Number of top predictions to consider
            apply_layer_norm: Whether to apply layer norm before projection
        
        Returns:
            BaselineResult with prediction
        """
        vec = latent_vector.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            if apply_layer_norm and self.layer_norm is not None:
                vec = self.layer_norm(vec)
            
            logits = self.lm_head(vec)
            probs = F.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            # Get top prediction
            top_token = self.tokenizer.decode([top_indices[0, 0].item()])
            confidence = top_probs[0, 0].item()
        
        return BaselineResult(
            method="logit_lens",
            prediction=top_token.strip(),
            confidence=confidence,
        )
    
    def evaluate(
        self,
        latent_vectors: list[torch.Tensor],
        ground_truths: list[str],
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate logit lens on a set of examples.
        
        Returns:
            Dict with accuracy and per-example results
        """
        results = []
        correct = 0
        
        iterator = tqdm(
            zip(latent_vectors, ground_truths),
            total=len(latent_vectors),
            desc="Logit Lens",
            disable=not verbose,
        )
        
        for vec, gt in iterator:
            result = self.predict(vec)
            result.ground_truth = gt
            result.is_correct = result._check_correct()
            results.append(result)
            
            if result.is_correct:
                correct += 1
        
        accuracy = correct / len(results) if results else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "results": results,
        }


class LinearProbeBaseline:
    """
    Linear Probe baseline: train a simple classifier/regressor on latent vectors.
    
    For classification: Uses logistic regression
    For regression (numeric results): Uses ridge regression
    """
    
    def __init__(self, task_type: str = "classification"):
        """
        Args:
            task_type: "classification" or "regression"
        """
        self.task_type = task_type
        self.model = None
        self.label_encoder = None
        self.is_fitted = False
    
    def fit(
        self,
        latent_vectors: list[torch.Tensor],
        labels: list[str],
        verbose: bool = True,
    ):
        """
        Train the linear probe.
        
        Args:
            latent_vectors: List of latent vectors
            labels: List of string labels (will be encoded for classification)
        """
        # Convert to numpy
        X = np.stack([v.cpu().numpy() for v in latent_vectors])
        
        if self.task_type == "classification":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            # For regression, try to convert labels to floats
            y = np.array([self._to_float(l) for l in labels])
            self.model = Ridge(alpha=1.0)
        
        if verbose:
            print(f"Training linear probe on {len(X)} examples...")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        if verbose:
            train_acc = self.model.score(X, y)
            print(f"Training accuracy: {train_acc:.4f}")
    
    def _to_float(self, s: str) -> float:
        """Convert string to float, handling various formats."""
        try:
            nums = re.findall(r'[-]?\d+\.?\d*', s)
            return float(nums[-1]) if nums else 0.0
        except (ValueError, IndexError):
            return 0.0
    
    def predict(self, latent_vector: torch.Tensor) -> BaselineResult:
        """
        Predict using the trained linear probe.
        """
        if not self.is_fitted:
            raise ValueError("Linear probe not fitted. Call fit() first.")
        
        X = latent_vector.cpu().numpy().reshape(1, -1)
        
        if self.task_type == "classification":
            pred_idx = self.model.predict(X)[0]
            prediction = self.label_encoder.inverse_transform([pred_idx])[0]
            proba = self.model.predict_proba(X)[0]
            confidence = proba.max()
        else:
            prediction = str(self.model.predict(X)[0])
            confidence = 1.0  # No natural confidence for regression
        
        return BaselineResult(
            method="linear_probe",
            prediction=prediction,
            confidence=confidence,
        )
    
    def evaluate(
        self,
        latent_vectors: list[torch.Tensor],
        ground_truths: list[str],
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate linear probe on a test set.
        """
        results = []
        correct = 0
        
        iterator = tqdm(
            zip(latent_vectors, ground_truths),
            total=len(latent_vectors),
            desc="Linear Probe",
            disable=not verbose,
        )
        
        for vec, gt in iterator:
            result = self.predict(vec)
            result.ground_truth = gt
            result.is_correct = result._check_correct()
            results.append(result)
            
            if result.is_correct:
                correct += 1
        
        accuracy = correct / len(results) if results else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "results": results,
        }


class RandomBaseline:
    """
    Random baseline for comparison.
    
    For classification: Random choice from label set
    For regression: Random number in observed range
    """
    
    def __init__(self, labels: Optional[list[str]] = None, value_range: tuple = (0, 100)):
        self.labels = labels
        self.value_range = value_range
    
    def predict(self) -> BaselineResult:
        if self.labels:
            import random
            prediction = random.choice(self.labels)
        else:
            import random
            prediction = str(random.randint(*self.value_range))
        
        return BaselineResult(
            method="random",
            prediction=prediction,
            confidence=1.0 / len(self.labels) if self.labels else 0.0,
        )
    
    def evaluate(
        self,
        latent_vectors: list[torch.Tensor],
        ground_truths: list[str],
        n_trials: int = 5,
    ) -> dict:
        """
        Evaluate random baseline (average over multiple trials).
        """
        accuracies = []
        
        for _ in range(n_trials):
            correct = 0
            for gt in ground_truths:
                result = self.predict()
                result.ground_truth = gt
                if result._check_correct():
                    correct += 1
            accuracies.append(correct / len(ground_truths))
        
        return {
            "accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "correct": int(np.mean(accuracies) * len(ground_truths)),
            "total": len(ground_truths),
        }


def compare_baselines(
    latent_vectors: list[torch.Tensor],
    ground_truths: list[str],
    codi_wrapper,
    train_split: float = 0.8,
    verbose: bool = True,
) -> dict:
    """
    Compare all baselines on the same data.
    
    Args:
        latent_vectors: List of latent vectors
        ground_truths: Corresponding ground truth labels
        codi_wrapper: CODIWrapper instance for logit lens
        train_split: Fraction of data for training linear probe
        verbose: Show progress
    
    Returns:
        Dict with results for each baseline
    """
    # Split data
    n_train = int(len(latent_vectors) * train_split)
    train_vecs = latent_vectors[:n_train]
    train_labels = ground_truths[:n_train]
    test_vecs = latent_vectors[n_train:]
    test_labels = ground_truths[n_train:]
    
    results = {}
    
    # Logit Lens
    if verbose:
        print("\n--- Logit Lens ---")
    logit_lens = LogitLensBaseline.from_codi_wrapper(codi_wrapper)
    results["logit_lens"] = logit_lens.evaluate(test_vecs, test_labels, verbose)
    
    # Linear Probe
    if verbose:
        print("\n--- Linear Probe ---")
    linear_probe = LinearProbeBaseline(task_type="regression")
    linear_probe.fit(train_vecs, train_labels, verbose)
    results["linear_probe"] = linear_probe.evaluate(test_vecs, test_labels, verbose)
    
    # Random Baseline
    if verbose:
        print("\n--- Random Baseline ---")
    unique_labels = list(set(ground_truths))
    random_baseline = RandomBaseline(labels=unique_labels if len(unique_labels) < 20 else None)
    results["random"] = random_baseline.evaluate(test_vecs, test_labels)
    
    # Summary
    if verbose:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for name, res in results.items():
            print(f"{name}: {res['accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing baselines with synthetic data...")
    
    # Create fake data
    n_samples = 100
    hidden_dim = 2048
    
    latent_vectors = [torch.randn(hidden_dim) for _ in range(n_samples)]
    ground_truths = [str(i % 10) for i in range(n_samples)]  # Labels 0-9
    
    # Test linear probe
    print("\nTraining linear probe...")
    probe = LinearProbeBaseline(task_type="classification")
    probe.fit(latent_vectors[:80], ground_truths[:80])
    
    print("\nEvaluating linear probe...")
    result = probe.evaluate(latent_vectors[80:], ground_truths[80:])
    print(f"Accuracy: {result['accuracy']:.4f}")
    
    # Test random baseline
    print("\nEvaluating random baseline...")
    random_bl = RandomBaseline(labels=list(set(ground_truths)))
    result = random_bl.evaluate(latent_vectors[80:], ground_truths[80:])
    print(f"Accuracy: {result['accuracy']:.4f} ± {result.get('accuracy_std', 0):.4f}")
