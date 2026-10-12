#!/usr/bin/env python3
"""Run bounded training and evaluation experiment on pilot provisional dataset."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results/experiment_run_20261012")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: Any, max_len: int = 128):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    exact_match = float(accuracy_score(y_true, y_pred))
    h_loss = float(hamming_loss(y_true, y_pred))

    # Binary metrics per label
    p_req, r_req, f1_req, _ = precision_recall_fscore_support(
        y_true[:, 0], y_pred[:, 0], average="binary", zero_division=0
    )
    p_com, r_com, f1_com, _ = precision_recall_fscore_support(
        y_true[:, 1], y_pred[:, 1], average="binary", zero_division=0
    )
    macro_f1 = float((f1_req + f1_com) / 2.0)

    # Micro F1
    _, _, micro_f1, _ = precision_recall_fscore_support(
        y_true.flatten(), y_pred.flatten(), average="binary", zero_division=0
    )

    # Confusion matrix elements
    # Label 0: Request
    tp_req = int(np.sum((y_true[:, 0] == 1) & (y_pred[:, 0] == 1)))
    fp_req = int(np.sum((y_true[:, 0] == 0) & (y_pred[:, 0] == 1)))
    fn_req = int(np.sum((y_true[:, 0] == 1) & (y_pred[:, 0] == 0)))
    tn_req = int(np.sum((y_true[:, 0] == 0) & (y_pred[:, 0] == 0)))

    # Label 1: Commitment
    tp_com = int(np.sum((y_true[:, 1] == 1) & (y_pred[:, 1] == 1)))
    fp_com = int(np.sum((y_true[:, 1] == 0) & (y_pred[:, 1] == 1)))
    fn_com = int(np.sum((y_true[:, 1] == 1) & (y_pred[:, 1] == 0)))
    tn_com = int(np.sum((y_true[:, 1] == 0) & (y_pred[:, 1] == 0)))

    return {
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "hamming_loss": h_loss,
        "request": {
            "precision": float(p_req),
            "recall": float(r_req),
            "f1": float(f1_req),
            "support_pos": int(np.sum(y_true[:, 0] == 1)),
            "support_neg": int(np.sum(y_true[:, 0] == 0)),
            "pred_pos": int(np.sum(y_pred[:, 0] == 1)),
            "tp": tp_req,
            "fp": fp_req,
            "fn": fn_req,
            "tn": tn_req,
        },
        "commitment": {
            "precision": float(p_com),
            "recall": float(r_com),
            "f1": float(f1_com),
            "support_pos": int(np.sum(y_true[:, 1] == 1)),
            "support_neg": int(np.sum(y_true[:, 1] == 0)),
            "pred_pos": int(np.sum(y_pred[:, 1] == 1)),
            "tp": tp_com,
            "fp": fp_com,
            "fn": fn_com,
            "tn": tn_com,
        },
    }


def main():
    start_time = time.perf_counter()

    # 1. Load data and recover split assignments
    data_path = Path("data/annotations/pilot_labeling_batch.provisional_train.csv")
    with data_path.open("r", encoding="utf-8") as f:
        all_rows = {r["example_id"]: r for r in csv.DictReader(f)}

    with open("results/action_detector_baseline/tfidf/split_assignments.json") as f:
        orig = json.load(f)

    test_ids = [x["example_id"] for x in orig["test"]]
    train_ids = [x["example_id"] for x in orig["train"]]

    test_rows = [all_rows[i] for i in test_ids]
    train_pool = [all_rows[i] for i in train_ids]

    train_threads = [r["thread_id"] for r in train_pool]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, val_idx = next(gss.split(train_pool, groups=train_threads))

    sub_train = [train_pool[i] for i in tr_idx]
    val_rows = [train_pool[i] for i in val_idx]

    # Save recovered split assignments
    split_info = {
        "train": [{"example_id": r["example_id"], "thread_id": r["thread_id"]} for r in sub_train],
        "validation": [{"example_id": r["example_id"], "thread_id": r["thread_id"]} for r in val_rows],
        "test": [{"example_id": r["example_id"], "thread_id": r["thread_id"]} for r in test_rows],
    }
    with (RESULTS_DIR / "split_assignments.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Convert targets to numpy arrays
    y_train = np.array([[int(r["request"]), int(r["commitment"])] for r in sub_train])
    y_val = np.array([[int(r["request"]), int(r["commitment"])] for r in val_rows])
    y_test = np.array([[int(r["request"]), int(r["commitment"])] for r in test_rows])

    train_texts = [r["text"] for r in sub_train]
    val_texts = [r["text"] for r in val_rows]
    test_texts = [r["text"] for r in test_rows]

    # Thresholds
    req_thresh = 0.3
    com_thresh = 0.5

    # 2. Train Model A: TF-IDF + Logistic Regression
    tfidf_start = time.perf_counter()
    vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)

    clf_req = LogisticRegression(random_state=SEED)
    clf_req.fit(X_train, y_train[:, 0])

    clf_com = LogisticRegression(random_state=SEED)
    clf_com.fit(X_train, y_train[:, 1])
    tfidf_runtime = time.perf_counter() - tfidf_start

    # Predictions TF-IDF
    val_scores_req_tfidf = clf_req.predict_proba(X_val)[:, 1]
    val_scores_com_tfidf = clf_com.predict_proba(X_val)[:, 1]
    val_pred_tfidf = np.column_stack([
        (val_scores_req_tfidf >= req_thresh).astype(int),
        (val_scores_com_tfidf >= com_thresh).astype(int),
    ])

    test_scores_req_tfidf = clf_req.predict_proba(X_test)[:, 1]
    test_scores_com_tfidf = clf_com.predict_proba(X_test)[:, 1]
    test_pred_tfidf = np.column_stack([
        (test_scores_req_tfidf >= req_thresh).astype(int),
        (test_scores_com_tfidf >= com_thresh).astype(int),
    ])

    val_metrics_tfidf = compute_detailed_metrics(y_val, val_pred_tfidf)
    test_metrics_tfidf = compute_detailed_metrics(y_test, test_pred_tfidf)

    # 3. Train Model B: DistilBERT from pretrained checkpoint
    dbert_start = time.perf_counter()
    tokenizer_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name,
        num_labels=2,
        problem_type="multi_label_classification",
    )
    model.to(DEVICE)

    train_ds = TextDataset(train_texts, y_train, tokenizer, max_len=128)
    val_ds = TextDataset(val_texts, y_val, tokenizer, max_len=128)
    test_ds = TextDataset(test_texts, y_test, tokenizer, max_len=128)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    training_history = []

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation evaluation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)

        val_probs_arr = np.vstack(val_probs)
        val_preds_arr = np.column_stack([
            (val_probs_arr[:, 0] >= req_thresh).astype(int),
            (val_probs_arr[:, 1] >= com_thresh).astype(int),
        ])
        epoch_val_metrics = compute_detailed_metrics(y_val, val_preds_arr)

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_macro_f1": epoch_val_metrics["macro_f1"],
            "val_exact_match": epoch_val_metrics["exact_match"],
            "val_req_f1": epoch_val_metrics["request"]["f1"],
            "val_com_f1": epoch_val_metrics["commitment"]["f1"],
        })

        if epoch_val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = epoch_val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    model.load_state_dict(best_state_dict)
    model.to(DEVICE)
    model.eval()

    # Save best checkpoint
    ckpt_dir = RESULTS_DIR / "distilbert" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    dbert_runtime = time.perf_counter() - dbert_start

    # Final predictions with best DistilBERT model
    def get_bert_probs(loader):
        probs_list = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
        return np.vstack(probs_list)

    val_scores_dbert = get_bert_probs(val_loader)
    test_scores_dbert = get_bert_probs(test_loader)

    val_pred_dbert = np.column_stack([
        (val_scores_dbert[:, 0] >= req_thresh).astype(int),
        (val_scores_dbert[:, 1] >= com_thresh).astype(int),
    ])
    test_pred_dbert = np.column_stack([
        (test_scores_dbert[:, 0] >= req_thresh).astype(int),
        (test_scores_dbert[:, 1] >= com_thresh).astype(int),
    ])

    val_metrics_dbert = compute_detailed_metrics(y_val, val_pred_dbert)
    test_metrics_dbert = compute_detailed_metrics(y_test, test_pred_dbert)

    # Check truncation counts
    def count_truncated(texts):
        return sum(1 for t in texts if len(tokenizer.encode(t, truncation=False)) > 128)

    trunc_counts = {
        "train": count_truncated(train_texts),
        "validation": count_truncated(val_texts),
        "test": count_truncated(test_texts),
    }

    # Save training_history.csv
    with (RESULTS_DIR / "training_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_macro_f1", "val_exact_match", "val_req_f1", "val_com_f1"])
        writer.writeheader()
        writer.writerows(training_history)

    # Save predictions.csv
    pred_rows = []
    # Validation predictions
    for idx, r in enumerate(val_rows):
        pred_rows.append({
            "example_id": r["example_id"],
            "split": "validation",
            "thread_id": r["thread_id"],
            "text": r["text"].replace("\n", " ").strip()[:140],
            "y_true_req": y_val[idx, 0],
            "y_true_com": y_val[idx, 1],
            "tfidf_score_req": round(float(val_scores_req_tfidf[idx]), 4),
            "tfidf_pred_req": int(val_pred_tfidf[idx, 0]),
            "tfidf_score_com": round(float(val_scores_com_tfidf[idx]), 4),
            "tfidf_pred_com": int(val_pred_tfidf[idx, 1]),
            "dbert_score_req": round(float(val_scores_dbert[idx, 0]), 4),
            "dbert_pred_req": int(val_pred_dbert[idx, 0]),
            "dbert_score_com": round(float(val_scores_dbert[idx, 1]), 4),
            "dbert_pred_com": int(val_pred_dbert[idx, 1]),
        })
    # Test predictions
    for idx, r in enumerate(test_rows):
        pred_rows.append({
            "example_id": r["example_id"],
            "split": "test",
            "thread_id": r["thread_id"],
            "text": r["text"].replace("\n", " ").strip()[:140],
            "y_true_req": y_test[idx, 0],
            "y_true_com": y_test[idx, 1],
            "tfidf_score_req": round(float(test_scores_req_tfidf[idx]), 4),
            "tfidf_pred_req": int(test_pred_tfidf[idx, 0]),
            "tfidf_score_com": round(float(test_scores_com_tfidf[idx]), 4),
            "tfidf_pred_com": int(test_pred_tfidf[idx, 1]),
            "dbert_score_req": round(float(test_scores_dbert[idx, 0]), 4),
            "dbert_pred_req": int(test_pred_dbert[idx, 0]),
            "dbert_score_com": round(float(test_scores_dbert[idx, 1]), 4),
            "dbert_pred_com": int(test_pred_dbert[idx, 1]),
        })

    with (RESULTS_DIR / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(pred_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    # Save comparison.csv
    comparison_rows = [
        {
            "model": "TF-IDF + LogisticRegression",
            "split": "validation",
            "exact_match": val_metrics_tfidf["exact_match"],
            "macro_f1": val_metrics_tfidf["macro_f1"],
            "micro_f1": val_metrics_tfidf["micro_f1"],
            "hamming_loss": val_metrics_tfidf["hamming_loss"],
            "req_p": val_metrics_tfidf["request"]["precision"],
            "req_r": val_metrics_tfidf["request"]["recall"],
            "req_f1": val_metrics_tfidf["request"]["f1"],
            "req_tp": val_metrics_tfidf["request"]["tp"],
            "req_fp": val_metrics_tfidf["request"]["fp"],
            "req_fn": val_metrics_tfidf["request"]["fn"],
            "req_tn": val_metrics_tfidf["request"]["tn"],
            "com_p": val_metrics_tfidf["commitment"]["precision"],
            "com_r": val_metrics_tfidf["commitment"]["recall"],
            "com_f1": val_metrics_tfidf["commitment"]["f1"],
            "com_tp": val_metrics_tfidf["commitment"]["tp"],
            "com_fp": val_metrics_tfidf["commitment"]["fp"],
            "com_fn": val_metrics_tfidf["commitment"]["fn"],
            "com_tn": val_metrics_tfidf["commitment"]["tn"],
        },
        {
            "model": "TF-IDF + LogisticRegression",
            "split": "test",
            "exact_match": test_metrics_tfidf["exact_match"],
            "macro_f1": test_metrics_tfidf["macro_f1"],
            "micro_f1": test_metrics_tfidf["micro_f1"],
            "hamming_loss": test_metrics_tfidf["hamming_loss"],
            "req_p": test_metrics_tfidf["request"]["precision"],
            "req_r": test_metrics_tfidf["request"]["recall"],
            "req_f1": test_metrics_tfidf["request"]["f1"],
            "req_tp": test_metrics_tfidf["request"]["tp"],
            "req_fp": test_metrics_tfidf["request"]["fp"],
            "req_fn": test_metrics_tfidf["request"]["fn"],
            "req_tn": test_metrics_tfidf["request"]["tn"],
            "com_p": test_metrics_tfidf["commitment"]["precision"],
            "com_r": test_metrics_tfidf["commitment"]["recall"],
            "com_f1": test_metrics_tfidf["commitment"]["f1"],
            "com_tp": test_metrics_tfidf["commitment"]["tp"],
            "com_fp": test_metrics_tfidf["commitment"]["fp"],
            "com_fn": test_metrics_tfidf["commitment"]["fn"],
            "com_tn": test_metrics_tfidf["commitment"]["tn"],
        },
        {
            "model": "DistilBERT (Tuned)",
            "split": "validation",
            "exact_match": val_metrics_dbert["exact_match"],
            "macro_f1": val_metrics_dbert["macro_f1"],
            "micro_f1": val_metrics_dbert["micro_f1"],
            "hamming_loss": val_metrics_dbert["hamming_loss"],
            "req_p": val_metrics_dbert["request"]["precision"],
            "req_r": val_metrics_dbert["request"]["recall"],
            "req_f1": val_metrics_dbert["request"]["f1"],
            "req_tp": val_metrics_dbert["request"]["tp"],
            "req_fp": val_metrics_dbert["request"]["fp"],
            "req_fn": val_metrics_dbert["request"]["fn"],
            "req_tn": val_metrics_dbert["request"]["tn"],
            "com_p": val_metrics_dbert["commitment"]["precision"],
            "com_r": val_metrics_dbert["commitment"]["recall"],
            "com_f1": val_metrics_dbert["commitment"]["f1"],
            "com_tp": val_metrics_dbert["commitment"]["tp"],
            "com_fp": val_metrics_dbert["commitment"]["fp"],
            "com_fn": val_metrics_dbert["commitment"]["fn"],
            "com_tn": val_metrics_dbert["commitment"]["tn"],
        },
        {
            "model": "DistilBERT (Tuned)",
            "split": "test",
            "exact_match": test_metrics_dbert["exact_match"],
            "macro_f1": test_metrics_dbert["macro_f1"],
            "micro_f1": test_metrics_dbert["micro_f1"],
            "hamming_loss": test_metrics_dbert["hamming_loss"],
            "req_p": test_metrics_dbert["request"]["precision"],
            "req_r": test_metrics_dbert["request"]["recall"],
            "req_f1": test_metrics_dbert["request"]["f1"],
            "req_tp": test_metrics_dbert["request"]["tp"],
            "req_fp": test_metrics_dbert["request"]["fp"],
            "req_fn": test_metrics_dbert["request"]["fn"],
            "req_tn": test_metrics_dbert["request"]["tn"],
            "com_p": test_metrics_dbert["commitment"]["precision"],
            "com_r": test_metrics_dbert["commitment"]["recall"],
            "com_f1": test_metrics_dbert["commitment"]["f1"],
            "com_tp": test_metrics_dbert["commitment"]["tp"],
            "com_fp": test_metrics_dbert["commitment"]["fp"],
            "com_fn": test_metrics_dbert["commitment"]["fn"],
            "com_tn": test_metrics_dbert["commitment"]["tn"],
        },
    ]

    with (RESULTS_DIR / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Save metrics.json
    total_runtime = time.perf_counter() - start_time
    full_metrics = {
        "metadata": {
            "dataset": "data/annotations/pilot_labeling_batch.provisional_train.csv",
            "total_examples": len(all_rows),
            "train_examples": len(sub_train),
            "val_examples": len(val_rows),
            "test_examples": len(test_rows),
            "seed": SEED,
            "device": str(DEVICE),
            "best_epoch": best_epoch,
            "thresholds": {"request": req_thresh, "commitment": com_thresh},
            "runtime_seconds": {
                "tfidf": tfidf_runtime,
                "distilbert": dbert_runtime,
                "total": total_runtime,
            },
            "truncation_counts_max128": trunc_counts,
        },
        "tfidf": {
            "validation": val_metrics_tfidf,
            "test": test_metrics_tfidf,
        },
        "distilbert": {
            "validation": val_metrics_dbert,
            "test": test_metrics_dbert,
        },
    }
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Save config.json
    config = {
        "seed": SEED,
        "device": str(DEVICE),
        "batch_size": 2,
        "lr": 2e-5,
        "max_length": 128,
        "epochs": 3,
        "thresholds": {"request": req_thresh, "commitment": com_thresh},
        "models": ["tfidf_logistic_regression", "distilbert-base-uncased"],
    }
    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Experiment completed successfully.")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
