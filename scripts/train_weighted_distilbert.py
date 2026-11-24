#!/usr/bin/env python3
"""Run 3 & 4: Fine-tune DistilBERT with Class-Weighted BCE Loss (pos_weight) and evaluate."""

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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")
MODELS_DIR = Path("models")
WEIGHTED_MODEL_DIR = MODELS_DIR / "distilbert_weighted"

class MemoryManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, max_samples=None):
        self.samples = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("split") == split and not record.get("excluded"):
                    self.samples.append((
                        record.get("text", ""),
                        [float(record.get("weak_request", 0)), float(record.get("weak_commitment", 0))]
                    ))
                    if max_samples and len(self.samples) >= max_samples:
                        break

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

def evaluate_model(model, tokenizer, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
    all_probs = []
    all_targets = []
    model.eval()
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
    return probs_arr, targets_arr

def train_weighted(steps=500, lr=2e-5, pos_weight_com=20.0):
    WEIGHTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("models/distilbert_fine_tuned")
    model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_fine_tuned").to(DEVICE)

    # Load training data
    print("Loading train samples into memory for balanced sampling...")
    train_records = []
    com_indices = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            if record.get("split") == "train" and not record.get("excluded"):
                item = (record.get("text", ""), [float(record.get("weak_request", 0)), float(record.get("weak_commitment", 0))])
                if item[1][1] == 1.0:
                    com_indices.append(len(train_records))
                train_records.append(item)

    print(f"Total train records: {len(train_records)}, Commitment records: {len(com_indices)}")

    # Class-weighted loss
    pos_weight = torch.tensor([1.2, pos_weight_com]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train with 50% commitment oversampling per batch to prevent batch starvation
    batch_size = 16
    n_com_per_batch = 4
    n_reg_per_batch = batch_size - n_com_per_batch

    model.train()
    t0 = time.time()
    running_loss = 0.0

    print(f"Starting {steps} training steps with pos_weight={pos_weight_com} and oversampled batches...")
    for s in range(1, steps + 1):
        # sample balanced batch
        batch_com = [train_records[np.random.choice(com_indices)] for _ in range(n_com_per_batch)]
        batch_reg = [train_records[np.random.randint(len(train_records))] for _ in range(n_reg_per_batch)]
        batch = batch_com + batch_reg

        batch_collated = collate_fn(batch, tokenizer)
        input_ids = batch_collated["input_ids"].to(DEVICE)
        attention_mask = batch_collated["attention_mask"].to(DEVICE)
        labels = batch_collated["labels"].to(DEVICE)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if s % 100 == 0:
            avg_l = running_loss / 100.0
            print(f"Step {s}/{steps} | Loss: {avg_l:.4f} | Time: {time.time()-t0:.1f}s")
            running_loss = 0.0

    # Save fine-tuned weighted model
    model.save_pretrained(WEIGHTED_MODEL_DIR)
    tokenizer.save_pretrained(WEIGHTED_MODEL_DIR)
    print(f"Saved weighted model to {WEIGHTED_MODEL_DIR}")

    # Evaluate on Validation
    print("Evaluating on Validation set...")
    val_dataset = MemoryManifestDataset(MANIFEST_PATH, "validation")
    val_probs, val_targets = evaluate_model(model, tokenizer, val_dataset)

    # Evaluate on Test
    print("Evaluating on Test set...")
    test_dataset = MemoryManifestDataset(MANIFEST_PATH, "test")
    test_probs, test_targets = evaluate_model(model, tokenizer, test_dataset)

    # Check AP
    ap_req_val = average_precision_score(val_targets[:, 0], val_probs[:, 0])
    ap_com_val = average_precision_score(val_targets[:, 1], val_probs[:, 1])
    ap_req_test = average_precision_score(test_targets[:, 0], test_probs[:, 0])
    ap_com_test = average_precision_score(test_targets[:, 1], test_probs[:, 1])

    print("\n--- WEIGHTED MODEL RESULTS ---")
    print(f"Validation Request AP: {ap_req_val:.4f} | Commitment AP: {ap_com_val:.4f}")
    print(f"Test Request AP: {ap_req_test:.4f} | Commitment AP: {ap_com_test:.4f}")

    # Sweep thresholds on validation
    best_com_th, best_com_f1 = 0.5, 0.0
    for th in np.arange(0.10, 0.90, 0.05):
        _, _, f1, _ = precision_recall_fscore_support(val_targets[:, 1], (val_probs[:, 1] >= th).astype(int), average="binary", zero_division=0)
        if f1 > best_com_f1:
            best_com_f1 = f1
            best_com_th = float(th)

    best_req_th, best_req_f1 = 0.5, 0.0
    for th in np.arange(0.20, 0.80, 0.05):
        _, _, f1, _ = precision_recall_fscore_support(val_targets[:, 0], (val_probs[:, 0] >= th).astype(int), average="binary", zero_division=0)
        if f1 > best_req_f1:
            best_req_f1 = f1
            best_req_th = float(th)

    print(f"Optimal Validation Thresholds: Request={best_req_th:.2f} (F1={best_req_f1:.4f}), Commitment={best_com_th:.2f} (F1={best_com_f1:.4f})")

    # Evaluate Run 3 (Default threshold 0.50)
    yp_r_def = (test_probs[:, 0] >= 0.50).astype(int)
    yp_c_def = (test_probs[:, 1] >= 0.50).astype(int)
    p_r3, r_r3, f1_r3, _ = precision_recall_fscore_support(test_targets[:, 0], yp_r_def, average="binary", zero_division=0)
    p_c3, r_c3, f1_c3, _ = precision_recall_fscore_support(test_targets[:, 1], yp_c_def, average="binary", zero_division=0)
    tp_c3 = int(np.sum((test_targets[:, 1] == 1) & (yp_c_def == 1)))
    fp_c3 = int(np.sum((test_targets[:, 1] == 0) & (yp_c_def == 1)))
    fn_c3 = int(np.sum((test_targets[:, 1] == 1) & (yp_c_def == 0)))
    tn_c3 = int(np.sum((test_targets[:, 1] == 0) & (yp_c_def == 0)))
    em3 = accuracy_score(test_targets, np.column_stack([yp_r_def, yp_c_def]))

    # Evaluate Run 4 (Tuned thresholds)
    yp_r_opt = (test_probs[:, 0] >= best_req_th).astype(int)
    yp_c_opt = (test_probs[:, 1] >= best_com_th).astype(int)
    p_r4, r_r4, f1_r4, _ = precision_recall_fscore_support(test_targets[:, 0], yp_r_opt, average="binary", zero_division=0)
    p_c4, r_c4, f1_c4, _ = precision_recall_fscore_support(test_targets[:, 1], yp_c_opt, average="binary", zero_division=0)
    tp_c4 = int(np.sum((test_targets[:, 1] == 1) & (yp_c_opt == 1)))
    fp_c4 = int(np.sum((test_targets[:, 1] == 0) & (yp_c_opt == 1)))
    fn_c4 = int(np.sum((test_targets[:, 1] == 1) & (yp_c_opt == 0)))
    tn_c4 = int(np.sum((test_targets[:, 1] == 0) & (yp_c_opt == 0)))
    em4 = accuracy_score(test_targets, np.column_stack([yp_r_opt, yp_c_opt]))

    print(f"\nRUN 3: Weighted Loss Only (Default 0.50 Threshold)")
    print(f"  Exact Match: {em3:.4f} | Macro-F1: {(f1_r3+f1_c3)/2:.4f}")
    print(f"  Request:    P={p_r3:.4f}, R={r_r3:.4f}, F1={f1_r3:.4f}")
    print(f"  Commitment: P={p_c3:.4f}, R={r_c3:.4f}, F1={f1_c3:.4f} | TP={tp_c3}, FP={fp_c3}, FN={fn_c3}, TN={tn_c3}")

    print(f"\nRUN 4: Weighted Loss + Tuned Threshold (Req={best_req_th:.2f}, Com={best_com_th:.2f})")
    print(f"  Exact Match: {em4:.4f} | Macro-F1: {(f1_r4+f1_c4)/2:.4f}")
    print(f"  Request:    P={p_r4:.4f}, R={r_r4:.4f}, F1={f1_r4:.4f}")
    print(f"  Commitment: P={p_c4:.4f}, R={r_c4:.4f}, F1={f1_c4:.4f} | TP={tp_c4}, FP={fp_c4}, FN={fn_c4}, TN={tn_c4}")

    results = {
        "run3": {
            "name": "Weighted Loss Only (th=0.50)",
            "req": {"p": p_r3, "r": r_r3, "f1": f1_r3},
            "com": {"p": p_c3, "r": r_c3, "f1": f1_c3, "tp": tp_c3, "fp": fp_c3, "fn": fn_c3, "tn": tn_c3},
            "macro_f1": (f1_r3+f1_c3)/2,
            "exact_match": em3,
        },
        "run4": {
            "name": f"Weighted Loss + Tuned Threshold (Req={best_req_th:.2f}, Com={best_com_th:.2f})",
            "req": {"p": p_r4, "r": r_r4, "f1": f1_r4},
            "com": {"p": p_c4, "r": r_c4, "f1": f1_c4, "tp": tp_c4, "fp": fp_c4, "fn": fn_c4, "tn": tn_c4},
            "macro_f1": (f1_r4+f1_c4)/2,
            "exact_match": em4,
            "best_req_th": best_req_th,
            "best_com_th": best_com_th,
        },
        "ap": {
            "req_val": ap_req_val, "com_val": ap_com_val,
            "req_test": ap_req_test, "com_test": ap_com_test,
        }
    }
    with open("evaluation/results/weighted_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    train_weighted(steps=400, lr=2e-5, pos_weight_com=20.0)
