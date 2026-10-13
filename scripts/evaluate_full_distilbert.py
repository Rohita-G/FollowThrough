#!/usr/bin/env python3
"""Evaluate existing fine-tuned DistilBERT checkpoint on the entire held-out test split (N=9,657)."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")
MODEL_PATH = Path("results/full_corpus_experiment/distilbert_fine_tuned")
THRESHOLDS_PATH = Path("results/full_corpus_experiment/distilbert_metrics.json")
OUTPUT_PATH = Path("results/full_corpus_experiment/distilbert_metrics_full_test.json")


class ManifestStreamDataset(IterableDataset):
    def __init__(self, manifest_path: Path, split: str = "test"):
        self.manifest_path = manifest_path
        self.split = split

    def __iter__(self):
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("split") == self.split and not record.get("excluded"):
                    text = record.get("text", "")
                    labels = [float(record.get("weak_request", 0)), float(record.get("weak_commitment", 0))]
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


def main():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    # Load thresholds selected from validation
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as f:
        prior_data = json.load(f)
    thresholds = prior_data.get("thresholds", {"request": 0.35, "commitment": 0.2})
    print(f"Using validation thresholds: {thresholds}")

    dataset = ManifestStreamDataset(MANIFEST_PATH, split="test")
    loader = DataLoader(dataset, batch_size=32, collate_fn=lambda b: collate_batch(b, tokenizer))

    all_probs = []
    all_targets = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch["labels"].numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    elapsed = time.perf_counter() - t0
    print(f"Inference on {len(probs_arr)} test records completed in {elapsed:.2f}s")

    preds_arr = np.column_stack([
        (probs_arr[:, 0] >= thresholds["request"]).astype(int),
        (probs_arr[:, 1] >= thresholds["commitment"]).astype(int),
    ])

    test_metrics = compute_metrics(targets_arr, preds_arr)

    output_data = {
        "model": "Model B: Fine-tuned DistilBERT (5,000 steps)",
        "dataset": "Full Enron Subset v2 Weak-Labeled Corpus (100k Manifest)",
        "total_test_records_evaluated": len(probs_arr),
        "thresholds": thresholds,
        "eval_runtime_seconds": round(elapsed, 2),
        "test_metrics": test_metrics,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Also update the primary distilbert_metrics.json so test_metrics reflects the full 9,657 set
    prior_data["full_test_metrics"] = test_metrics
    prior_data["full_test_evaluated_count"] = len(probs_arr)
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(prior_data, f, indent=2)

    print(f"Results saved to {OUTPUT_PATH}")
    print(f"Test Exact Match: {test_metrics['exact_match_accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Request F1: {test_metrics['labels']['request']['f1']:.4f} (Support: {test_metrics['labels']['request']['support']}, TP: {test_metrics['labels']['request']['tp']}, FP: {test_metrics['labels']['request']['fp']}, FN: {test_metrics['labels']['request']['fn']})")
    print(f"Test Commitment F1: {test_metrics['labels']['commitment']['f1']:.4f} (Support: {test_metrics['labels']['commitment']['support']}, TP: {test_metrics['labels']['commitment']['tp']}, FP: {test_metrics['labels']['commitment']['fp']}, FN: {test_metrics['labels']['commitment']['fn']})")


if __name__ == "__main__":
    main()
