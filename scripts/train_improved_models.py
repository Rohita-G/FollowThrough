#!/usr/bin/env python3
"""Train improved models with thread-separated validation split, threshold selection, and test evaluation."""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    exact_match = float(accuracy_score(y_true, y_pred))
    p_req, r_req, f1_req, _ = precision_recall_fscore_support(y_true[:, 0], y_pred[:, 0], average="binary", zero_division=0)
    p_com, r_com, f1_com, _ = precision_recall_fscore_support(y_true[:, 1], y_pred[:, 1], average="binary", zero_division=0)
    macro_f1 = float((f1_req + f1_com) / 2.0)
    return {
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "request": {"precision": float(p_req), "recall": float(r_req), "f1": float(f1_req), "support": int(np.sum(y_true[:, 0]))},
        "commitment": {"precision": float(p_com), "recall": float(r_com), "f1": float(f1_com), "support": int(np.sum(y_true[:, 1]))},
    }


def main():
    start_total_time = time.perf_counter()

    # 1. Load data & preserved test split
    with open("data/annotations/pilot_labeling_batch.provisional_train.csv", "r", encoding="utf-8") as f:
        all_rows_dict = {r["example_id"]: r for r in csv.DictReader(f)}

    with open("results/action_detector_baseline/tfidf/split_assignments.json", "r", encoding="utf-8") as f:
        orig_splits = json.load(f)

    test_ids = set(x["example_id"] for x in orig_splits["test"])
    train_ids = [x["example_id"] for x in orig_splits["train"]]

    test_rows = [all_rows_dict[tid] for tid in test_ids]
    train_pool = [all_rows_dict[tid] for tid in train_ids]

    # Convert labels
    for r in train_pool + test_rows:
        r["req_int"] = int(r["request"])
        r["com_int"] = int(r["commitment"])

    # 2. Thread-separated validation split from train_pool
    train_threads = [r["thread_id"] for r in train_pool]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, val_idx = next(gss.split(train_pool, groups=train_threads))

    sub_train = [train_pool[i] for i in tr_idx]
    val_rows = [train_pool[i] for i in val_idx]

    # Check leakage
    sub_tr_texts = {r["text"] for r in sub_train}
    val_leaks = [r["example_id"] for r in val_rows if r["text"] in sub_tr_texts]
    assert len(val_leaks) == 0, f"Leakage detected between train and val: {val_leaks}"

    # Label support report
    val_req_pos = sum(r["req_int"] for r in val_rows)
    val_req_neg = len(val_rows) - val_req_pos
    val_com_pos = sum(r["com_int"] for r in val_rows)
    val_com_neg = len(val_rows) - val_com_pos

    print(f"Dataset splits: Train-subset={len(sub_train)}, Validation={len(val_rows)}, Test={len(test_rows)}")
    print(f"Validation label support: Request [pos={val_req_pos}, neg={val_req_neg}], Commitment [pos={val_com_pos}, neg={val_com_neg}]")

    # 3. Truncation evidence check
    tokenizer_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    req_ev_lost, com_ev_lost = 0, 0
    for r in list(all_rows_dict.values()):
        trunc_text = tokenizer.decode(tokenizer.encode(r["text"], truncation=True, max_length=128), skip_special_tokens=True).lower()
        if r.get("request_evidence"):
            clean_ev = " ".join(r["request_evidence"].split()).lower()
            if clean_ev not in " ".join(trunc_text.split()):
                req_ev_lost += 1
        if r.get("commitment_evidence"):
            clean_ev = " ".join(r["commitment_evidence"].split()).lower()
            if clean_ev not in " ".join(trunc_text.split()):
                com_ev_lost += 1
    print(f"Truncation evidence lost (max_length=128): Request={req_ev_lost}, Commitment={com_ev_lost}")

    # 4. Train TF-IDF baseline on sub_train only
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_tr_vec = vec.fit_transform([r["text"] for r in sub_train])
    X_val_vec = vec.transform([r["text"] for r in val_rows])
    y_tr = np.array([[r["req_int"], r["com_int"]] for r in sub_train])
    y_val = np.array([[r["req_int"], r["com_int"]] for r in val_rows])

    # Compare default vs balanced
    baseline_candidates = {}
    for weight_opt in (None, "balanced"):
        clfs = []
        val_preds = np.zeros_like(y_val)
        for i in range(2):
            unique_c = np.unique(y_tr[:, i])
            if len(unique_c) < 2:
                constant_val = int(unique_c[0])
                clfs.append(constant_val)
                val_preds[:, i] = constant_val
            else:
                clf = LogisticRegression(C=1.0, max_iter=1000, class_weight=weight_opt, random_state=SEED)
                clf.fit(X_tr_vec, y_tr[:, i])
                clfs.append(clf)
                val_preds[:, i] = clf.predict(X_val_vec)

        m = compute_metrics(y_val, val_preds)
        baseline_candidates[str(weight_opt)] = (m, clfs)
        print(f"TF-IDF baseline (class_weight={weight_opt}): Val Macro-F1 = {m['macro_f1']:.4f}")

    best_weight_opt = max(baseline_candidates.keys(), key=lambda k: baseline_candidates[k][0]["macro_f1"])
    best_base_m, best_clfs = baseline_candidates[best_weight_opt]
    print(f"Selected TF-IDF baseline: class_weight={best_weight_opt} (Val Macro-F1: {best_base_m['macro_f1']:.4f})")

    # 5. Fine-tune DistilBERT on sub_train
    class TorchDataset(Dataset):
        def __init__(self, data_rows):
            self.texts = [r["text"] for r in data_rows]
            self.labels = np.array([[r["req_int"], r["com_int"]] for r in data_rows], dtype=np.float32)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        return enc, torch.tensor(np.array(labels), dtype=torch.float32)

    tr_loader = DataLoader(TorchDataset(sub_train), batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(TorchDataset(val_rows), batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, num_labels=2, problem_type="multi_label_classification"
    ).to(dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_macro_f1 = -1.0
    best_epoch = 1
    best_state_dict = None
    best_val_probs = None

    for epoch in range(1, 4):
        bert_model.train()
        for enc, labels in tr_loader:
            optimizer.zero_grad()
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            labels = labels.to(DEVICE)
            logits = bert_model(**enc).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        # Val evaluation
        bert_model.eval()
        val_probs_list = []
        with torch.no_grad():
            for enc, _ in val_loader:
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                logits = bert_model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_list.append(probs)
        val_probs = np.vstack(val_probs_list)
        val_preds = (val_probs >= 0.5).astype(int)
        val_metrics = compute_metrics(y_val, val_preds)

        print(f"DistilBERT Epoch {epoch}/3: Val Macro-F1 = {val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in bert_model.state_dict().items()}
            best_val_probs = val_probs

    print(f"Best DistilBERT epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}")
    bert_model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_dict.items()})

    # 6. Threshold Selection on Validation Set Only
    candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_req_thresh = 0.5
    best_com_thresh = 0.5

    # Tune request if val has both pos and neg
    if val_req_pos > 0 and val_req_neg > 0:
        best_req_f1 = -1.0
        for th in candidate_thresholds:
            p_th = (best_val_probs[:, 0] >= th).astype(int)
            _, _, f1_th, _ = precision_recall_fscore_support(y_val[:, 0], p_th, average="binary", zero_division=0)
            if f1_th > best_req_f1:
                best_req_f1 = f1_th
                best_req_thresh = th
        print(f"Selected Request threshold on validation: {best_req_thresh} (Val F1: {best_req_f1:.4f})")
    else:
        print("Validation lacks positive/negative balance for Request; keeping threshold 0.5.")

    # Tune commitment if val has both pos and neg
    if val_com_pos > 0 and val_com_neg > 0:
        best_com_f1 = -1.0
        for th in candidate_thresholds:
            p_th = (best_val_probs[:, 1] >= th).astype(int)
            _, _, f1_th, _ = precision_recall_fscore_support(y_val[:, 1], p_th, average="binary", zero_division=0)
            if f1_th > best_com_f1:
                best_com_f1 = f1_th
                best_com_thresh = th
        print(f"Selected Commitment threshold on validation: {best_com_thresh}")
    else:
        print(f"Validation lacks positive Commitment examples (pos={val_com_pos}); keeping threshold 0.5.")

    # 7. Final Evaluation on Unchanged 17-message Test Set
    X_test_vec = vec.transform([r["text"] for r in test_rows])
    y_test = np.array([[r["req_int"], r["com_int"]] for r in test_rows])

    # Evaluate Baseline on Test
    test_preds_base = np.zeros_like(y_test)
    for i in range(2):
        if hasattr(best_clfs[i], "predict"):
            test_preds_base[:, i] = best_clfs[i].predict(X_test_vec)
        else:
            test_preds_base[:, i] = best_clfs[i]
    test_metrics_base = compute_metrics(y_test, test_preds_base)

    # Evaluate DistilBERT on Test with chosen thresholds
    test_loader = DataLoader(TorchDataset(test_rows), batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_probs_list = []
    with torch.no_grad():
        for enc, _ in test_loader:
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = bert_model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs_list.append(probs)
    test_probs_bert = np.vstack(test_probs_list)
    test_preds_bert = np.zeros_like(y_test)
    test_preds_bert[:, 0] = (test_probs_bert[:, 0] >= best_req_thresh).astype(int)
    test_preds_bert[:, 1] = (test_probs_bert[:, 1] >= best_com_thresh).astype(int)
    test_metrics_bert = compute_metrics(y_test, test_preds_bert)

    elapsed_total = time.perf_counter() - start_total_time

    # 8. Save Artifacts
    out_dir = Path("results/action_detector_tuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "distilbert" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bert_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    base_out = out_dir / "tfidf"
    base_out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifiers": best_clfs, "class_weight": best_weight_opt}, base_out / "model.joblib")

    thresholds_config = {
        "request_threshold": best_req_thresh,
        "commitment_threshold": best_com_thresh,
        "best_epoch": best_epoch,
        "best_class_weight": best_weight_opt,
        "truncation_lost_evidence": {"request": req_ev_lost, "commitment": com_ev_lost},
    }
    with (out_dir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(thresholds_config, f, indent=2)

    comparison_report = {
        "dataset_notice": "Tiny provisional AI-assisted dataset; test set N=17.",
        "baseline": test_metrics_base,
        "distilbert_tuned": test_metrics_bert,
        "runtime_seconds": round(elapsed_total, 2),
    }
    with (out_dir / "evaluation_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    print("\n=== FINAL TEST EVALUATION COMPARISON (N=17) ===")
    print(f"TF-IDF Baseline: ExactMatch={test_metrics_base['exact_match']:.3f}, Macro-F1={test_metrics_base['macro_f1']:.3f}")
    print(f"  Req F1: {test_metrics_base['request']['f1']:.3f} | Com F1: {test_metrics_base['commitment']['f1']:.3f}")
    print(f"DistilBERT Tuned: ExactMatch={test_metrics_bert['exact_match']:.3f}, Macro-F1={test_metrics_bert['macro_f1']:.3f}")
    print(f"  Req F1: {test_metrics_bert['request']['f1']:.3f} | Com F1: {test_metrics_bert['commitment']['f1']:.3f}")
    print(f"Total training & eval runtime: {elapsed_total:.2f}s")


if __name__ == "__main__":
    main()
