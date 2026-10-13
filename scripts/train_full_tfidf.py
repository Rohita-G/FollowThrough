#!/usr/bin/env python3
"""Train TF-IDF + Logistic Regression baseline on the full weak-labeled corpus manifest."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)

MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("evaluation")


def load_split_data(manifest_path: Path, split_name: str) -> Tuple[List[str], np.ndarray, List[str]]:
    texts: List[str] = []
    labels: List[List[int]] = []
    msg_ids: List[str] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") == split_name and not rec.get("excluded"):
                texts.append(rec.get("text", ""))
                labels.append([int(rec.get("weak_request", 0)), int(rec.get("weak_commitment", 0))])
                msg_ids.append(rec.get("message_id", ""))

    return texts, np.array(labels, dtype=np.int32), msg_ids


def compute_weak_agreement_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ["request", "commitment"],
) -> Dict[str, Any]:
    exact_match = float(accuracy_score(y_true, y_pred))
    h_loss = float(hamming_loss(y_true, y_pred))

    # Per-label metrics
    per_label: Dict[str, Any] = {}
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0

    for idx, name in enumerate(label_names):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        per_label[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(np.sum(y_t)),
            "predicted_positive_count": int(np.sum(y_p)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    macro_f1 = float(np.mean([per_label[name]["f1"] for name in label_names]))
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "title": "Weak-label agreement—not independently verified accuracy",
        "exact_match_accuracy": exact_match,
        "hamming_loss": h_loss,
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "labels": per_label,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.is_file():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        print("Run scripts/build_full_corpus_manifest.py first.")
        return

    print("Loading datasets from full corpus manifest...")
    t0 = time.perf_counter()
    train_texts, train_labels, _ = load_split_data(MANIFEST_PATH, "train")
    val_texts, val_labels, _ = load_split_data(MANIFEST_PATH, "validation")
    test_texts, test_labels, _ = load_split_data(MANIFEST_PATH, "test")
    load_time = time.perf_counter() - t0
    print(f"Loaded: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)} in {load_time:.2f}s")

    # Fit TF-IDF
    print("Fitting TfidfVectorizer (max_features=10000, ngram_range=(1,2))...")
    t1 = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    vectorize_time = time.perf_counter() - t1
    print(f"Vectorized features shape: {X_train.shape} in {vectorize_time:.2f}s")

    # Train Logistic Regression models
    print("Training Logistic Regression models for request and commitment...")
    t2 = time.perf_counter()
    clf_req = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_com = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)

    clf_req.fit(X_train, train_labels[:, 0])
    clf_com.fit(X_train, train_labels[:, 1])
    train_time = time.perf_counter() - t2
    print(f"Training complete in {train_time:.2f}s")

    # Validation evaluation & threshold tuning
    print("Evaluating on Validation set...")
    val_probs_req = clf_req.predict_proba(X_val)[:, 1]
    val_probs_com = clf_com.predict_proba(X_val)[:, 1]

    # Threshold selection on validation to maximize F1
    best_th_req, best_f1_req = 0.5, 0.0
    for th in np.arange(0.2, 0.8, 0.05):
        _, _, f1, _ = precision_recall_fscore_support(val_labels[:, 0], (val_probs_req >= th).astype(int), average="binary", zero_division=0)
        if f1 > best_f1_req:
            best_f1_req = float(f1)
            best_th_req = float(th)

    best_th_com, best_f1_com = 0.5, 0.0
    for th in np.arange(0.2, 0.8, 0.05):
        _, _, f1, _ = precision_recall_fscore_support(val_labels[:, 1], (val_probs_com >= th).astype(int), average="binary", zero_division=0)
        if f1 > best_f1_com:
            best_f1_com = float(f1)
            best_th_com = float(th)

    selected_thresholds = {"request": round(best_th_req, 2), "commitment": round(best_th_com, 2)}
    print(f"Selected Validation Thresholds: {selected_thresholds}")

    val_preds = np.column_stack([
        (val_probs_req >= selected_thresholds["request"]).astype(int),
        (val_probs_com >= selected_thresholds["commitment"]).astype(int),
    ])
    val_metrics = compute_weak_agreement_metrics(val_labels, val_preds)

    # Test evaluation
    print("Evaluating on Test set using validation-selected thresholds...")
    t3 = time.perf_counter()
    test_probs_req = clf_req.predict_proba(X_test)[:, 1]
    test_probs_com = clf_com.predict_proba(X_test)[:, 1]
    test_eval_time = time.perf_counter() - t3

    test_preds = np.column_stack([
        (test_probs_req >= selected_thresholds["request"]).astype(int),
        (test_probs_com >= selected_thresholds["commitment"]).astype(int),
    ])
    test_metrics = compute_weak_agreement_metrics(test_labels, test_preds)

    # Save models and disk usage
    model_path = MODELS_DIR / "tfidf_baseline.joblib"
    joblib.dump({"vectorizer": vectorizer, "clf_req": clf_req, "clf_com": clf_com, "thresholds": selected_thresholds}, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    total_time = time.perf_counter() - t0

    output_report = {
        "model": "Model A: TF-IDF + Logistic Regression",
        "dataset": "Full Enron Subset v2 Weak-Labeled Corpus (100k Manifest)",
        "train_examples": len(train_texts),
        "validation_examples": len(val_texts),
        "test_examples": len(test_texts),
        "thresholds": selected_thresholds,
        "runtime_seconds": {
            "load_data": round(load_time, 2),
            "vectorization": round(vectorize_time, 2),
            "training": round(train_time, 2),
            "test_eval": round(test_eval_time, 2),
            "total": round(total_time, 2),
        },
        "disk_usage_mb": round(model_size_mb, 2),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_file = RESULTS_DIR / "tfidf_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    print(f"\nSaved metrics to {metrics_file}")
    print(f"Test Exact Match: {test_metrics['exact_match_accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Request F1: {test_metrics['labels']['request']['f1']:.4f}")
    print(f"Test Commitment F1: {test_metrics['labels']['commitment']['f1']:.4f}")


if __name__ == "__main__":
    main()
