#!/usr/bin/env python3
"""
Train and evaluate a linear probe for operation type detection.

Outputs precise metrics for charting, comparable to logit lens results.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


class OperationProbe(nn.Module):
    """Simple linear probe for operation classification."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class LatentOperationDataset(Dataset):
    """Dataset of latent vectors with operation labels."""
    
    def __init__(self, latents: list, labels: list, position: int):
        self.latents = latents
        self.labels = labels
        self.position = position
        self.op_to_idx = {"add": 0, "sub": 1, "mul": 2}
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx][self.position]
        label = self.op_to_idx[self.labels[idx]]
        return latent, label


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


def train_probe(probe, train_loader, device, epochs=10, lr=1e-3):
    """Train the linear probe."""
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for latents, labels in train_loader:
            latents = latents.to(device).float()  # Convert to float32
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(latents)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        acc = correct / total * 100
        # print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={acc:.1f}%")
    
    return probe


def evaluate_probe(probe, eval_loader, device, problems, position):
    """Evaluate probe with detailed metrics."""
    probe.eval()
    idx_to_op = {0: "add", 1: "sub", 2: "mul"}
    
    results = {
        "position": position,
        "position_name": f"z{position+1}",
        "total": 0,
        "correct": 0,
        "per_operation": {op: {"total": 0, "correct": 0, "predictions": []} for op in ["add", "sub", "mul"]},
        "predictions": [],
    }
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for latents, labels in eval_loader:
            latents = latents.to(device).float()  # Convert to float32
            labels = labels.to(device)
            
            logits = probe(latents)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Compute metrics
    for i, (pred_idx, label_idx, probs) in enumerate(zip(all_preds, all_labels, all_probs)):
        pred_op = idx_to_op[pred_idx]
        true_op = idx_to_op[label_idx]
        is_correct = pred_op == true_op
        
        results["total"] += 1
        results["per_operation"][true_op]["total"] += 1
        
        if is_correct:
            results["correct"] += 1
            results["per_operation"][true_op]["correct"] += 1
        
        confidence = probs[pred_idx]
        margin = probs[pred_idx] - sorted(probs, reverse=True)[1]
        
        prediction = {
            "idx": i,
            "true_op": true_op,
            "pred_op": pred_op,
            "correct": is_correct,
            "confidence": confidence,
            "margin": margin,
            "op_probs": {"add": probs[0], "sub": probs[1], "mul": probs[2]},
        }
        results["predictions"].append(prediction)
        results["per_operation"][true_op]["predictions"].append(prediction)
    
    # Aggregates
    results["accuracy"] = results["correct"] / results["total"] * 100
    results["avg_confidence"] = sum(p["confidence"] for p in results["predictions"]) / len(results["predictions"])
    results["avg_margin"] = sum(p["margin"] for p in results["predictions"]) / len(results["predictions"])
    
    for op in ["add", "sub", "mul"]:
        op_data = results["per_operation"][op]
        if op_data["total"] > 0:
            op_data["accuracy"] = op_data["correct"] / op_data["total"] * 100
        else:
            op_data["accuracy"] = 0.0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate linear probe for operation detection")
    parser.add_argument("--data", type=str, default="data/synthetic_problems.json", help="Input data")
    parser.add_argument("--output", type=str, default="results/linear_probe_operation.json")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/test split")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Linear Probe Operation Detection")
    print("=" * 60)
    
    # Set seeds
    torch.manual_seed(args.seed)
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Collect latents
    print(f"\nCollecting latents for {len(problems)} problems...")
    all_latents = []
    for problem in tqdm(problems, desc="Collecting latents"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        all_latents.append(result.latent_vectors)
    
    # Get hidden dim
    hidden_dim = all_latents[0][0].shape[0]
    print(f"Hidden dimension: {hidden_dim}")
    
    # Get labels
    labels = [p["operation"] for p in problems]
    
    # Train/test split
    split_idx = int(len(problems) * args.train_split)
    train_latents = all_latents[:split_idx]
    train_labels = labels[:split_idx]
    test_latents = all_latents[split_idx:]
    test_labels = labels[split_idx:]
    test_problems = problems[split_idx:]
    
    print(f"Train: {len(train_latents)}, Test: {len(test_latents)}")
    
    # Results storage
    results = {
        "config": {
            "n_samples": len(problems),
            "train_samples": len(train_latents),
            "test_samples": len(test_latents),
            "data_seed": data["config"]["seed"],
            "probe_seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "operation_distribution": dict(op_counts),
        },
        "positions": {},
        "summary": [],
    }
    
    # Train and evaluate probe for each position
    print("\nTraining and evaluating probes per position...")
    
    for pos in range(6):
        pos_name = f"z{pos+1}"
        print(f"\n--- Position {pos_name} ---")
        
        # Create datasets
        train_dataset = LatentOperationDataset(train_latents, train_labels, pos)
        test_dataset = LatentOperationDataset(test_latents, test_labels, pos)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create and train probe
        probe = OperationProbe(hidden_dim, num_classes=3).to(device)
        print(f"  Training probe ({args.epochs} epochs)...")
        probe = train_probe(probe, train_loader, device, epochs=args.epochs, lr=args.lr)
        
        # Evaluate
        pos_results = evaluate_probe(probe, test_loader, device, test_problems, pos)
        results["positions"][pos_name] = pos_results
        
        # Print summary
        print(f"  Test accuracy: {pos_results['accuracy']:.1f}%")
        print(f"  Avg confidence: {pos_results['avg_confidence']:.4f}")
        for op in ["add", "sub", "mul"]:
            op_data = pos_results["per_operation"][op]
            print(f"    {op}: {op_data['accuracy']:.1f}% ({op_data['correct']}/{op_data['total']})")
        
        # Save probe checkpoint
        probe_path = Path(f"checkpoints/linear_probes/{pos_name}_operation.pt")
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), probe_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for pos in range(6):
        pos_name = f"z{pos+1}"
        pos_data = results["positions"][pos_name]
        results["summary"].append({
            "position": pos_name,
            "accuracy": pos_data["accuracy"],
            "avg_confidence": pos_data["avg_confidence"],
            "avg_margin": pos_data["avg_margin"],
            "add_acc": pos_data["per_operation"]["add"]["accuracy"],
            "sub_acc": pos_data["per_operation"]["sub"]["accuracy"],
            "mul_acc": pos_data["per_operation"]["mul"]["accuracy"],
        })
        print(f"{pos_name}: {pos_data['accuracy']:5.1f}% (add={pos_data['per_operation']['add']['accuracy']:.1f}%, sub={pos_data['per_operation']['sub']['accuracy']:.1f}%, mul={pos_data['per_operation']['mul']['accuracy']:.1f}%)")
    
    # Find best position
    best_pos = max(results["summary"], key=lambda x: x["accuracy"])
    print(f"\nBest position: {best_pos['position']} ({best_pos['accuracy']:.1f}%)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
