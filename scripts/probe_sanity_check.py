#!/usr/bin/env python3
"""
Sanity checks for linear probe results.

Tests:
1. Shuffled labels - measures overfitting baseline
2. Novel templates - tests generalization to different surface forms
3. Neutral templates - removes lexical operation cues
"""

import argparse
import json
import random
import sys
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
    def __init__(self, latents, labels, position):
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


# Novel templates with DIFFERENT lexical cues
NOVEL_TEMPLATES_ADD = [
    "A collection contains {X} objects. {Y} objects join the collection. The total is then scaled by {Z}. What is the result?",
    "Starting with {X} units, {Y} units arrive. Everything is then {Z}x. Final count?",
    "Count: {X}. Increase: {Y}. Multiplier: {Z}. What's the final number?",
]

NOVEL_TEMPLATES_SUB = [
    "A collection contains {X} objects. {Y} objects depart. The remainder is then scaled by {Z}. What is the result?",
    "Starting with {X} units, {Y} units leave. Everything is then {Z}x. Final count?",
    "Count: {X}. Decrease: {Y}. Multiplier: {Z}. What's the final number?",
]

NOVEL_TEMPLATES_MUL = [
    "A collection contains {X} objects. Each becomes {Y} objects. Then {Z} more objects arrive. What is the result?",
    "Starting with {X} units, each splits into {Y}. Then {Z} units join. Final count?",
    "Count: {X}. Scale factor: {Y}. Offset: {Z}. What's the final number?",
]

# Neutral templates - deliberately AVOID operation-specific words
NEUTRAL_TEMPLATES = [
    "Numbers: {X}, {Y}, {Z}. Apply operation 1, then operation 2. Result?",
    "Given {X}, {Y}, {Z}. Two transformations occur. Final value?",
    "Input: {X}, {Y}, {Z}. Process step 1, then step 2. Output?",
]


def generate_novel_problems(n_samples: int, seed: int = 123):
    """Generate problems with novel templates."""
    random.seed(seed)
    problems = []
    
    ops = ["add", "sub", "mul"]
    templates = {
        "add": NOVEL_TEMPLATES_ADD,
        "sub": NOVEL_TEMPLATES_SUB,
        "mul": NOVEL_TEMPLATES_MUL,
    }
    
    for i in range(n_samples):
        op = ops[i % 3]
        X = random.randint(1, 10)
        Y = random.randint(1, 10)
        Z = random.randint(2, 10)
        
        if op == "sub" and X <= Y:
            X, Y = max(X, Y) + 1, min(X, Y)
        
        template = random.choice(templates[op])
        prompt = template.format(X=X, Y=Y, Z=Z)
        
        if op == "add":
            step1 = X + Y
        elif op == "sub":
            step1 = X - Y
        else:
            step1 = X * Y
        
        problems.append({
            "seed": seed + i,
            "prompt": prompt,
            "X": X, "Y": Y, "Z": Z,
            "operation": op,
            "step1": step1,
        })
    
    return problems


def generate_neutral_problems(n_samples: int, seed: int = 456):
    """Generate problems with neutral (non-indicative) templates."""
    random.seed(seed)
    problems = []
    
    ops = ["add", "sub", "mul"]
    
    for i in range(n_samples):
        op = ops[i % 3]
        X = random.randint(1, 10)
        Y = random.randint(1, 10)
        Z = random.randint(2, 10)
        
        if op == "sub" and X <= Y:
            X, Y = max(X, Y) + 1, min(X, Y)
        
        template = random.choice(NEUTRAL_TEMPLATES)
        prompt = template.format(X=X, Y=Y, Z=Z)
        
        if op == "add":
            step1 = X + Y
        elif op == "sub":
            step1 = X - Y
        else:
            step1 = X * Y
        
        problems.append({
            "seed": seed + i,
            "prompt": prompt,
            "X": X, "Y": Y, "Z": Z,
            "operation": op,
            "step1": step1,
        })
    
    return problems


def load_codi_model(config_path="configs/default.yaml"):
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


def train_probe(probe, train_loader, device, epochs=20, lr=1e-3):
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for latents, labels in train_loader:
            latents = latents.to(device).float()
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(latents)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    return probe


def evaluate_probe(probe, eval_loader, device):
    probe.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for latents, labels in eval_loader:
            latents = latents.to(device).float()
            labels = labels.to(device)
            
            logits = probe(latents)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total * 100


def collect_latents(codi, problems):
    all_latents = []
    for problem in tqdm(problems, desc="Collecting latents"):
        result = codi.collect_latents(problem["prompt"], return_hidden_states=False)
        all_latents.append(result.latent_vectors)
    return all_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic_problems.json")
    parser.add_argument("--output", type=str, default="results/probe_sanity_check.json")
    parser.add_argument("--n_novel", type=int, default=300, help="Novel template problems")
    parser.add_argument("--n_neutral", type=int, default=300, help="Neutral template problems")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Linear Probe Sanity Checks")
    print("=" * 60)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Load original data
    print(f"\nLoading original data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    problems = data["problems"]
    
    # Generate novel and neutral problems
    print(f"Generating {args.n_novel} novel template problems...")
    novel_problems = generate_novel_problems(args.n_novel, seed=123)
    
    print(f"Generating {args.n_neutral} neutral template problems...")
    neutral_problems = generate_neutral_problems(args.n_neutral, seed=456)
    
    # Load model
    print("\nLoading CODI model...")
    codi = load_codi_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Collect all latents
    print("\nCollecting latents for original problems...")
    orig_latents = collect_latents(codi, problems)
    
    print("Collecting latents for novel template problems...")
    novel_latents = collect_latents(codi, novel_problems)
    
    print("Collecting latents for neutral template problems...")
    neutral_latents = collect_latents(codi, neutral_problems)
    
    hidden_dim = orig_latents[0][0].shape[0]
    
    # Prepare labels
    orig_labels = [p["operation"] for p in problems]
    novel_labels = [p["operation"] for p in novel_problems]
    neutral_labels = [p["operation"] for p in neutral_problems]
    
    # Shuffled labels for overfitting test
    shuffled_labels = orig_labels.copy()
    random.shuffle(shuffled_labels)
    
    # Train/test split for original
    split_idx = int(len(problems) * 0.8)
    train_latents = orig_latents[:split_idx]
    train_labels = orig_labels[:split_idx]
    test_latents = orig_latents[split_idx:]
    test_labels = orig_labels[split_idx:]
    
    results = {
        "config": {
            "n_original": len(problems),
            "n_novel": len(novel_problems),
            "n_neutral": len(neutral_problems),
            "epochs": args.epochs,
        },
        "tests": {},
    }
    
    print("\n" + "=" * 60)
    print("Running sanity checks for each position...")
    print("=" * 60)
    
    for pos in range(6):
        pos_name = f"z{pos+1}"
        print(f"\n--- Position {pos_name} ---")
        
        pos_results = {}
        
        # Test 1: Normal training and eval (baseline)
        print("  Test 1: Normal train/test...")
        train_dataset = LatentOperationDataset(train_latents, train_labels, pos)
        test_dataset = LatentOperationDataset(test_latents, test_labels, pos)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        probe = OperationProbe(hidden_dim).to(device)
        probe = train_probe(probe, train_loader, device, epochs=args.epochs)
        normal_acc = evaluate_probe(probe, test_loader, device)
        pos_results["normal_accuracy"] = normal_acc
        print(f"    Normal accuracy: {normal_acc:.1f}%")
        
        # Test 2: Shuffled labels (overfitting test)
        print("  Test 2: Shuffled labels (overfitting test)...")
        shuffled_train = shuffled_labels[:split_idx]
        shuffled_test = shuffled_labels[split_idx:]
        
        train_dataset_shuf = LatentOperationDataset(train_latents, shuffled_train, pos)
        test_dataset_shuf = LatentOperationDataset(test_latents, shuffled_test, pos)
        train_loader_shuf = DataLoader(train_dataset_shuf, batch_size=32, shuffle=True)
        test_loader_shuf = DataLoader(test_dataset_shuf, batch_size=32)
        
        probe_shuf = OperationProbe(hidden_dim).to(device)
        probe_shuf = train_probe(probe_shuf, train_loader_shuf, device, epochs=args.epochs)
        shuffled_acc = evaluate_probe(probe_shuf, test_loader_shuf, device)
        pos_results["shuffled_accuracy"] = shuffled_acc
        print(f"    Shuffled accuracy: {shuffled_acc:.1f}% (expect ~33% if not overfitting)")
        
        # Test 3: Novel templates (generalization test)
        print("  Test 3: Novel templates (generalization)...")
        # Use probe trained on original data
        novel_dataset = LatentOperationDataset(novel_latents, novel_labels, pos)
        novel_loader = DataLoader(novel_dataset, batch_size=32)
        novel_acc = evaluate_probe(probe, novel_loader, device)
        pos_results["novel_accuracy"] = novel_acc
        print(f"    Novel template accuracy: {novel_acc:.1f}%")
        
        # Test 4: Neutral templates (hardest test)
        print("  Test 4: Neutral templates (no lexical cues)...")
        neutral_dataset = LatentOperationDataset(neutral_latents, neutral_labels, pos)
        neutral_loader = DataLoader(neutral_dataset, batch_size=32)
        neutral_acc = evaluate_probe(probe, neutral_loader, device)
        pos_results["neutral_accuracy"] = neutral_acc
        print(f"    Neutral template accuracy: {neutral_acc:.1f}%")
        
        results["tests"][pos_name] = pos_results
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Position':<10} {'Normal':<10} {'Shuffled':<10} {'Novel':<10} {'Neutral':<10}")
    print("-" * 50)
    
    summary = []
    for pos in range(6):
        pos_name = f"z{pos+1}"
        r = results["tests"][pos_name]
        print(f"{pos_name:<10} {r['normal_accuracy']:<10.1f} {r['shuffled_accuracy']:<10.1f} {r['novel_accuracy']:<10.1f} {r['neutral_accuracy']:<10.1f}")
        summary.append({
            "position": pos_name,
            "normal": r["normal_accuracy"],
            "shuffled": r["shuffled_accuracy"],
            "novel": r["novel_accuracy"],
            "neutral": r["neutral_accuracy"],
        })
    
    results["summary"] = summary
    
    print("\nInterpretation:")
    print("- Shuffled ~33%: Probe is NOT overfitting")
    print("- Shuffled >50%: Probe may be overfitting or memorizing")
    print("- Novel high: Generalizes to different wording")
    print("- Neutral low: Relies on lexical cues, not true operation encoding")
    print("- Neutral high: Operation IS encoded in latents (strong evidence!)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
