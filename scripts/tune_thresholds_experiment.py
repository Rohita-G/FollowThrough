#!/usr/bin/env python3
"""Script to evaluate DistilBERT probabilities, tune thresholds, and compute PR metrics."""

import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
)
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")
MODEL_DIR = Path("models/distilbert_fine_tuned")

class ManifestSplitDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str):
        self.samples = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("split") == split and not record.get("excluded"):
                    self.samples.append((
                        record.get("text", ""),
                        [float(record.get("weak_request", 0)), float(record.get("weak_commitment", 0))]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer, max_len=128):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    enc["labels"] = labels
    return enc

def get_predictions(model, tokenizer, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
    all_probs = []
    all_targets = []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch["labels"].numpy())
    elapsed = time.time() - t0
    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    print(f"Computed {len(probs_arr)} predictions in {elapsed:.2f}s ({len(probs_arr)/elapsed:.1f} samples/s)")
    return probs_arr, targets_arr

def main():
    print(f"Loading model from {MODEL_DIR} on {DEVICE}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("Loading validation dataset...")
    val_dataset = ManifestSplitDataset(MANIFEST_PATH, "validation")
    print(f"Validation size: {len(val_dataset)}")

    print("Loading test dataset...")
    test_dataset = ManifestSplitDataset(MANIFEST_PATH, "test")
    print(f"Test size: {len(test_dataset)}")

    print("\n--- INFERENCE ON VALIDATION ---")
    val_probs, val_targets = get_predictions(model, tokenizer, val_dataset)

    print("\n--- INFERENCE ON TEST ---")
    test_probs, test_targets = get_predictions(model, tokenizer, test_dataset)

    # Label index: 0 = Request, 1 = Commitment
    labels = ["Request", "Commitment"]

    for idx, name in enumerate(labels):
        yt_val = val_targets[:, idx]
        yp_val = val_probs[:, idx]
        yt_test = test_targets[:, idx]
        yp_test = test_probs[:, idx]

        ap_val = average_precision_score(yt_val, yp_val)
        ap_test = average_precision_score(yt_test, yp_test)

        print(f"\n==============================")
        print(f"Label: {name.upper()}")
        print(f"Validation Positive Support: {int(np.sum(yt_val))}/{len(yt_val)} ({np.mean(yt_val)*100:.2f}%)")
        print(f"Validation Average Precision (PR-AUC): {ap_val:.4f}")
        print(f"Test Positive Support: {int(np.sum(yt_test))}/{len(yt_test)} ({np.mean(yt_test)*100:.2f}%)")
        print(f"Test Average Precision (PR-AUC): {ap_test:.4f}")

        # Probability distribution breakdown
        pos_probs_val = yp_val[yt_val == 1]
        neg_probs_val = yp_val[yt_val == 0]
        print(f"Validation Positive class prob percentiles (p10, p50, p90, max): "
              f"{np.percentile(pos_probs_val, [10, 50, 90, 100]) if len(pos_probs_val) else 'N/A'}")
        print(f"Validation Negative class prob percentiles (p50, p90, p99, max): "
              f"{np.percentile(neg_probs_val, [50, 90, 99, 100])}")

    # Threshold sweeping for Commitment on validation data
    print("\n==============================")
    print("COMMITMENT THRESHOLD SWEEP ON VALIDATION SET")
    print(f"{'Threshold':<10} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}")
    print("-" * 75)

    best_f1_th = 0.5
    best_f1 = 0.0
    sweep_results = []

    # Sweep from very low (0.01) to 0.50
    for th in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        preds = (val_probs[:, 1] >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_targets[:, 1], preds, average="binary", zero_division=0)
        tp = int(np.sum((val_targets[:, 1] == 1) & (preds == 1)))
        fp = int(np.sum((val_targets[:, 1] == 0) & (preds == 1)))
        fn = int(np.sum((val_targets[:, 1] == 1) & (preds == 0)))
        tn = int(np.sum((val_targets[:, 1] == 0) & (preds == 0)))
        print(f"{th:<10.3f} | {p:<8.4f} | {r:<8.4f} | {f1:<8.4f} | {tp:<6} | {fp:<6} | {fn:<6} | {tn:<6}")
        sweep_results.append((th, p, r, f1, tp, fp, fn, tn))
        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = th

    print(f"\nBest Validation Commitment F1 = {best_f1:.4f} at Threshold = {best_f1_th:.3f}")

    # Also tune Request threshold on validation
    best_req_th = 0.5
    best_req_f1 = 0.0
    for th in np.arange(0.20, 0.80, 0.05):
        preds = (val_probs[:, 0] >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_targets[:, 0], preds, average="binary", zero_division=0)
        if f1 > best_req_f1:
            best_req_f1 = f1
            best_req_th = float(th)

    print(f"Best Validation Request F1 = {best_req_f1:.4f} at Threshold = {best_req_th:.2f}")

    # Evaluate multiple candidate thresholds on Test Set
    print("\n==============================")
    print("EVALUATION ON TEST SET (N=9,657) WITH CANDIDATE COMMITMENT THRESHOLDS")
    candidates = [
        ("Run 1: Original Benchmark", 0.35, 0.20),
        ("Run 2A: Tuned Max-F1", best_req_th, best_f1_th),
        ("Run 2B: Balanced/Recall-Oriented", best_req_th, 0.05 if best_f1_th > 0.05 else 0.02),
        ("Run 2C: High-Recall Oriented", best_req_th, 0.02),
    ]

    for name, req_t, com_t in candidates:
        yp_req = (test_probs[:, 0] >= req_t).astype(int)
        yp_com = (test_probs[:, 1] >= com_t).astype(int)
        test_preds = np.column_stack([yp_req, yp_com])

        em = accuracy_score(test_targets, test_preds)
        hl = hamming_loss(test_targets, test_preds)

        p_r, r_r, f1_r, _ = precision_recall_fscore_support(test_targets[:, 0], yp_req, average="binary", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(test_targets[:, 1], yp_com, average="binary", zero_division=0)

        tp_c = int(np.sum((test_targets[:, 1] == 1) & (yp_com == 1)))
        fp_c = int(np.sum((test_targets[:, 1] == 0) & (yp_com == 1)))
        fn_c = int(np.sum((test_targets[:, 1] == 1) & (yp_com == 0)))
        tn_c = int(np.sum((test_targets[:, 1] == 0) & (yp_com == 0)))

        macro_f1 = (f1_r + f1_c) / 2.0

        print(f"\nConfiguration: {name} (Req Th={req_t:.2f}, Com Th={com_t:.3f})")
        print(f"  Exact Match Acc: {em:.4f} | Hamming Loss: {hl:.4f} | Macro-F1: {macro_f1:.4f}")
        print(f"  Request:    Precision={p_r:.4f}, Recall={r_r:.4f}, F1={f1_r:.4f}")
        print(f"  Commitment: Precision={p_c:.4f}, Recall={r_c:.4f}, F1={f1_c:.4f} | TP={tp_c}, FP={fp_c}, FN={fn_c}, TN={tn_c}")

    # Save validation and test probabilities to cache for rapid re-analysis
    np.savez_compressed("evaluation/results/distilbert_val_test_probs.npz",
                        val_probs=val_probs, val_targets=val_targets,
                        test_probs=test_probs, test_targets=test_targets)
    print("\nSaved predictions to evaluation/results/distilbert_val_test_probs.npz")

if __name__ == "__main__":
    main()
