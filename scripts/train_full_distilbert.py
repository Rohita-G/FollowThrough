#!/usr/bin/env python3
"""Fine-tune DistilBERT on full weak-labeled corpus manifest with resumable checkpoints."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")
RESULTS_DIR = Path("results/full_corpus_experiment")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


class ManifestStreamDataset(IterableDataset):
    def __init__(self, manifest_path: Path, split: str = "train", max_records: Optional[int] = None):
        self.manifest_path = manifest_path
        self.split = split
        self.max_records = max_records

    def __iter__(self):
        count = 0
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if self.max_records and count >= self.max_records:
                    break
                record = json.loads(line)
                if record.get("split") == self.split and not record.get("excluded"):
                    text = record.get("text", "")
                    labels = [float(record.get("weak_request", 0)), float(record.get("weak_commitment", 0))]
                    count += 1
                    yield text, labels


def collate_batch(batch, tokenizer, max_len=128):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    enc["labels"] = labels
    return enc


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names=["request", "commitment"]):
    exact_match = float(accuracy_score(y_true, y_pred))
    h_loss = float(hamming_loss(y_true, y_pred))
    per_label = {}
    tp_tot, fp_tot, fn_tot, tn_tot = 0, 0, 0, 0
    for idx, name in enumerate(label_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        tn_tot += tn
        per_label[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(np.sum(yt)),
            "predicted_positive_count": int(np.sum(yp)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    macro_f1 = float(np.mean([per_label[n]["f1"] for n in label_names]))
    micro_p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    micro_r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    return {
        "title": "Weak-label agreement—not independently verified accuracy",
        "exact_match_accuracy": exact_match,
        "hamming_loss": h_loss,
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "labels": per_label,
    }


def evaluate(model, tokenizer, split="validation", max_eval_records=None, thresholds=None):
    model.eval()
    dataset = ManifestStreamDataset(MANIFEST_PATH, split=split, max_records=max_eval_records)
    loader = DataLoader(dataset, batch_size=16, collate_fn=lambda b: collate_batch(b, tokenizer))

    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch["labels"].numpy())

    if not all_probs:
        return {}, {}, {}

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    if thresholds is None:
        # Tune thresholds on validation
        best_th = {}
        for idx, name in enumerate(["request", "commitment"]):
            best_f1, best_t = 0.0, 0.5
            for th in np.arange(0.2, 0.8, 0.05):
                _, _, f1, _ = precision_recall_fscore_support(targets_arr[:, idx], (probs_arr[:, idx] >= th).astype(int), average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_t = float(th)
            best_th[name] = round(best_t, 2)
        thresholds = best_th

    preds_arr = np.column_stack([
        (probs_arr[:, 0] >= thresholds["request"]).astype(int),
        (probs_arr[:, 1] >= thresholds["commitment"]).astype(int),
    ])

    metrics = compute_metrics(targets_arr, preds_arr)
    return metrics, thresholds, (probs_arr, targets_arr)


def train(max_steps=None, lr=2e-5, save_every=2000):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Initializing DistilBERT training on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=2,
        problem_type="multi_label_classification",
    )
    model.to(DEVICE)

    # Check for existing checkpoint
    latest_ckpt = CHECKPOINT_DIR / "latest.pt"
    start_step = 0
    if latest_ckpt.exists():
        print(f"Resuming from checkpoint {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)

    dataset = ManifestStreamDataset(MANIFEST_PATH, split="train")
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda b: collate_batch(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    step = start_step
    start_time = time.perf_counter()
    running_loss = 0.0

    print(f"Starting training loop from step {step}...")
    for batch in loader:
        step += 1
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 500 == 0:
            avg_loss = running_loss / 500.0
            elapsed = time.perf_counter() - start_time
            print(f"Step {step} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            running_loss = 0.0

        if step % save_every == 0:
            torch.save({"step": step, "model_state_dict": model.state_dict()}, CHECKPOINT_DIR / "latest.pt")
            print(f"Saved checkpoint at step {step}")

        if max_steps and step >= max_steps:
            print(f"Reached target max steps ({max_steps}).")
            break

    # Save final model
    torch.save({"step": step, "model_state_dict": model.state_dict()}, CHECKPOINT_DIR / "latest.pt")
    model_save_path = RESULTS_DIR / "distilbert_fine_tuned"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Evaluation
    print("\nRunning Validation evaluation...")
    val_metrics, selected_thresholds, _ = evaluate(model, tokenizer, split="validation", max_eval_records=2000)
    print(f"Validation selected thresholds: {selected_thresholds}")

    print("\nRunning Test evaluation...")
    test_metrics, _, _ = evaluate(model, tokenizer, split="test", max_eval_records=2000, thresholds=selected_thresholds)

    report = {
        "model": "Model B: Fine-tuned DistilBERT",
        "dataset": "Full Enron Subset v2 Weak-Labeled Corpus (100k Manifest)",
        "total_steps": step,
        "device": str(DEVICE),
        "thresholds": selected_thresholds,
        "runtime_seconds": round(time.perf_counter() - start_time, 2),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_file = RESULTS_DIR / "distilbert_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved DistilBERT metrics to {metrics_file}")
    print(f"Test Exact Match: {test_metrics['exact_match_accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Request F1: {test_metrics['labels']['request']['f1']:.4f}")
    print(f"Test Commitment F1: {test_metrics['labels']['commitment']['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--save_every", type=int, default=2000, help="Checkpoint interval")
    args = parser.parse_args()
    train(max_steps=args.max_steps, save_every=args.save_every)
