#!/usr/bin/env python3
"""Train baseline and DistilBERT models for message-level Request/Commitment detection.

Supported architectures:
1. TF-IDF + Logistic Regression baseline (with constant-label handling)
2. DistilBERT (distilbert/distilbert-base-uncased) with 2-label BCEWithLogitsLoss

Ensures:
- Local Mac execution (MPS / CPU, no CUDA required).
- Threads are kept strictly separated across train/val/test splits (GroupSplit).
- Exact duplicate text across splits is checked and reported.
- Reports per-label precision, recall, F1, and exact-match accuracy.
- Enforces resolved labels; preserves label provenance metadata.
- Saves split assignments, config, predictions, metrics, and model checkpoint.
- Measures and reports actual elapsed training time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit


def load_and_validate_labeled_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load binary labels from CSV and validate completeness.

    Raises ValueError if any row has blank/unresolved labels or is marked needs_review=1.
    Preserves provenance metadata (label_source, human_reviewed).
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labeling CSV not found: {csv_path}")

    rows: List[Dict[str, Any]] = []
    unresolved_rows: List[str] = []

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, 1):
            ex_id = r.get("example_id", f"row-{idx}")
            req_str = str(r.get("request", "")).strip()
            com_str = str(r.get("commitment", "")).strip()
            needs_rev = str(r.get("needs_review", "0")).strip()

            if req_str == "" or com_str == "" or needs_rev == "1":
                unresolved_rows.append(f"Row {idx} ({ex_id}): req={req_str!r}, com={com_str!r}, needs_review={needs_rev!r}")
                continue

            try:
                req_val = int(req_str)
                com_val = int(com_str)
            except ValueError:
                raise ValueError(
                    f"Row {idx} ({ex_id}) has non-integer label: "
                    f"request={req_str!r}, commitment={com_str!r}. Expected 0 or 1."
                )

            if req_val not in (0, 1) or com_val not in (0, 1):
                raise ValueError(
                    f"Row {idx} has invalid label value. Expected 0 or 1, got {req_val}, {com_val}."
                )

            rows.append({
                "example_id": ex_id,
                "thread_id": r.get("thread_id", ""),
                "source_message_id": r.get("source_message_id", ""),
                "text": str(r.get("text", "")).strip(),
                "request": req_val,
                "commitment": com_val,
                "label_source": r.get("label_source", "human"),
                "human_reviewed": r.get("human_reviewed", "1"),
            })

    if unresolved_rows:
        raise ValueError(
            f"Cannot train with unresolved labels ({len(unresolved_rows)} rows unresolved/flagged for review):\n"
            + "\n".join(f"  - {u}" for u in unresolved_rows[:10])
            + (f"\n  ... and {len(unresolved_rows) - 10} more." if len(unresolved_rows) > 10 else "")
            + "\nPlease finalize all labels before training."
        )

    if not rows:
        raise ValueError(f"No labeled rows found in {csv_path}.")

    return rows


def check_split_leakage(train_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> List[str]:
    """Check for exact duplicate text appearing in both train and test splits."""
    train_texts = {r["text"] for r in train_rows if r["text"]}
    leaks = []
    for r in test_rows:
        if r["text"] in train_texts:
            leaks.append(f"example_id={r['example_id']} in thread {r['thread_id']}")
    return leaks


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute per-label and exact-match metrics."""
    labels = ["request", "commitment"]
    metrics: Dict[str, Any] = {}

    exact_match = accuracy_score(y_true, y_pred)
    metrics["exact_match_accuracy"] = float(exact_match)

    for i, name in enumerate(labels):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        support = int(np.sum(y_true[:, i]))
        metrics[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": support,
        }

    macro_f1 = np.mean([metrics["request"]["f1"], metrics["commitment"]["f1"]])
    metrics["macro_f1"] = float(macro_f1)
    return metrics


def train_eval_tfidf_baseline(
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    ngram_max: int = 2,
    max_features: int = 1000,
    c_reg: float = 1.0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Train TF-IDF + Logistic Regression with constant-label fallback."""
    start_time = time.perf_counter()
    X_train_text = [r["text"] for r in train_rows]
    X_test_text = [r["text"] for r in test_rows]

    y_train = np.array([[r["request"], r["commitment"]] for r in train_rows])
    y_test = np.array([[r["request"], r["commitment"]] for r in test_rows])

    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    y_pred = np.zeros_like(y_test)
    label_names = ["request", "commitment"]
    classifiers = []

    for i, name in enumerate(label_names):
        unique_classes = np.unique(y_train[:, i])
        if len(unique_classes) < 2:
            constant_class = int(unique_classes[0]) if len(unique_classes) == 1 else 0
            print(
                f"[Warning] Training split has insufficient class diversity for {name!r} "
                f"(only class {constant_class} present). Defaulting predictions to {constant_class}.",
                file=sys.stderr,
            )
            y_pred[:, i] = constant_class
            classifiers.append(constant_class)
        else:
            clf = LogisticRegression(C=c_reg, max_iter=1000, class_weight="balanced")
            clf.fit(X_train_vec, y_train[:, i])
            y_pred[:, i] = clf.predict(X_test_vec)
            classifiers.append(clf)

    metrics = compute_multilabel_metrics(y_test, y_pred)
    elapsed = time.perf_counter() - start_time
    metrics["elapsed_seconds"] = round(elapsed, 4)
    model_bundle = {
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "label_names": label_names,
    }
    return metrics, y_pred, model_bundle




def verify_mps_device(torch_module: Any) -> bool:
    """Verify Apple Silicon MPS device via a small forward and backward test."""
    if not (hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available() and torch_module.backends.mps.is_built()):
        return False
    try:
        device = torch_module.device("mps")
        x = torch_module.randn(2, 4, device=device, requires_grad=True)
        y = (x * 2.0).sum()
        y.backward()
        return True
    except Exception as exc:
        print(f"[Notice] MPS verification failed: {exc}", file=sys.stderr)
        return False


def resolve_local_device(torch_module: Any, requested_device: Optional[str] = None) -> Any:
    """Resolve compute device strictly for local Mac environment (MPS or CPU).

    Does not depend on CUDA. If requested device fails, raises an error without silent fallback.
    """
    if requested_device:
        req = requested_device.lower().strip()
        if req == "mps":
            if not verify_mps_device(torch_module):
                raise RuntimeError("Requested device 'mps' failed verification test or is not available.")
            return torch_module.device("mps")
        elif req == "cpu":
            return torch_module.device("cpu")
        else:
            raise ValueError(f"Unsupported device '{requested_device}'. Allowed local devices: 'mps', 'cpu'.")

    # Auto-detection
    if verify_mps_device(torch_module):
        print("Compute device: Apple Silicon Metal Performance Shaders (MPS) verified.", file=sys.stderr)
        return torch_module.device("mps")

    print("Compute device: Local CPU.", file=sys.stderr)
    return torch_module.device("cpu")


def train_eval_distilbert(
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    max_length: int = 128,
    batch_size: int = 2,
    epochs: int = 1,
    lr: float = 3e-5,
    seed: int = 42,
    device_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Any, Any]:
    """Train DistilBERT with BCEWithLogitsLoss using conservative local Mac settings."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(f"DistilBERT requires PyTorch and HuggingFace Transformers: {exc}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_local_device(torch, device_name)

    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="multi_label_classification",
    )
    # Ensure standard float32
    model.to(dtype=torch.float32, device=device)

    # Check truncation count
    all_texts = [r["text"] for r in train_rows + test_rows]
    truncated_count = 0
    for text in all_texts:
        tokens = tokenizer.encode(text, truncation=False)
        if len(tokens) > max_length:
            truncated_count += 1
    pct_truncated = (truncated_count / max(1, len(all_texts))) * 100
    print(
        f"Sequence length report (max_length={max_length}): "
        f"{truncated_count}/{len(all_texts)} messages truncated ({pct_truncated:.1f}%).",
        file=sys.stderr,
    )

    class TextDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]]):
            self.texts = [r["text"] for r in rows]
            self.labels = np.array([[r["request"], r["commitment"]] for r in rows], dtype=np.float32)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc, torch.tensor(np.array(labels), dtype=torch.float32)

    # num_workers=0 for safe, conservative local Mac DataLoader
    train_loader = DataLoader(
        TextDataset(train_rows), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        TextDataset(test_rows), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    start_time = time.perf_counter()
    model.train()
    total_steps = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for enc, labels in train_loader:
            optimizer.zero_grad()
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            logits = model(**enc).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{epochs} - avg loss: {avg_loss:.4f}", file=sys.stderr)

    elapsed_training = time.perf_counter() - start_time
    sec_per_step = elapsed_training / max(1, total_steps)

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for enc, labels in test_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_targets.append(labels.numpy().astype(int))

    y_test = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    metrics = compute_multilabel_metrics(y_test, y_pred)
    metrics["elapsed_training_seconds"] = round(elapsed_training, 3)
    metrics["seconds_per_step"] = round(sec_per_step, 4)
    metrics["total_steps"] = total_steps
    metrics["messages_truncated"] = truncated_count
    metrics["device"] = str(device)

    return metrics, y_pred, model, tokenizer


def save_training_artifacts(
    output_dir: Path,
    config: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    y_pred: np.ndarray,
    metrics: Dict[str, Any],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> None:
    """Save split assignments, configuration, predictions, metrics, and model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. Splits
    splits_record = {
        "train": [{"example_id": r["example_id"], "thread_id": r["thread_id"]} for r in train_rows],
        "test": [{"example_id": r["example_id"], "thread_id": r["thread_id"]} for r in test_rows],
    }
    with (output_dir / "split_assignments.json").open("w", encoding="utf-8") as f:
        json.dump(splits_record, f, indent=2)

    # 3. Predictions
    preds_record = []
    for r, pred in zip(test_rows, y_pred):
        preds_record.append({
            "example_id": r["example_id"],
            "thread_id": r["thread_id"],
            "true_request": r["request"],
            "true_commitment": r["commitment"],
            "pred_request": int(pred[0]),
            "pred_commitment": int(pred[1]),
            "label_source": r.get("label_source", "unknown"),
        })
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(preds_record, f, indent=2)

    # 4. Metrics
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 5. Model Checkpoint (PyTorch model or joblib bundle)
    if model is not None:
        if hasattr(model, "save_pretrained"):
            ckpt_dir = output_dir / "checkpoint"
            model.save_pretrained(ckpt_dir)
            if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(ckpt_dir)
        elif isinstance(model, dict) and "vectorizer" in model:
            import joblib
            joblib.dump(model, output_dir / "model.joblib")

    print(f"Artifacts successfully saved to: {output_dir}")



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/annotations/pilot_labeling_batch.csv", help="Path to labeling CSV")
    parser.add_argument("--model", choices=["tfidf", "distilbert", "both"], default="tfidf")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction (thread-grouped)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length for DistilBERT")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (conservative default: 2)")
    parser.add_argument("--epochs", type=int, default=1, help="DistilBERT epochs (conservative smoke-run: 1)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for DistilBERT")
    parser.add_argument("--device", default=None, help="Local device: 'mps' or 'cpu'")
    parser.add_argument("--output-dir", default=None, help="Directory to save training artifacts")
    args = parser.parse_args(argv)

    try:
        rows = load_and_validate_labeled_data(Path(args.data))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1

    threads = [r["thread_id"] for r in rows]
    unique_threads = sorted(set(threads))
    print(f"Loaded {len(rows)} fully resolved labeled examples across {len(unique_threads)} unique threads.")

    # Group split by thread_id
    if len(unique_threads) < 2:
        print("Warning: Only 1 thread present in data; cannot do grouped split. Evaluating on training data.", file=sys.stderr)
        train_rows = rows
        test_rows = rows
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(gss.split(rows, groups=threads))
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]

    train_threads = set(r["thread_id"] for r in train_rows)
    test_threads = set(r["thread_id"] for r in test_rows)
    print(f"Train split: {len(train_rows)} examples ({len(train_threads)} threads)")
    print(f"Test split:  {len(test_rows)} examples ({len(test_threads)} threads)")

    # Check leakage
    leaks = check_split_leakage(train_rows, test_rows)
    if leaks:
        print(f"[Warning] Exact duplicate text found across splits ({len(leaks)} instances):", file=sys.stderr)
        for leak in leaks[:5]:
            print(f"  {leak}", file=sys.stderr)
    else:
        print("Split check: Zero duplicate text leakage detected across train/test splits.")

    # Small-data limitation warning
    if len(rows) < 100:
        print(
            f"[Notice] Small-data limitation: Total sample size ({len(rows)}) is small. "
            "Results indicate technical feasibility and baseline variance, not final generalized model quality.",
            file=sys.stderr,
        )

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/run_{args.model}_{int(time.time())}")

    # Run models
    if args.model in ("tfidf", "both"):
        print("\n=== TF-IDF + Logistic Regression Baseline ===")
        metrics, y_pred, model_bundle = train_eval_tfidf_baseline(train_rows, test_rows)
        _print_metrics(metrics)
        config = {
            "model": "tfidf_logreg",
            "seed": args.seed,
            "test_size": args.test_size,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "label_source": list(set(r.get("label_source", "unknown") for r in rows)),
        }
        save_training_artifacts(out_dir / "tfidf", config, train_rows, test_rows, y_pred, metrics, model=model_bundle)


    if args.model in ("distilbert", "both"):
        print("\n=== DistilBERT (2-label Multi-label) ===")
        try:
            metrics_bert, y_pred_bert, model, tokenizer = train_eval_distilbert(
                train_rows,
                test_rows,
                max_length=args.max_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                device_name=args.device,
                output_dir=out_dir / "distilbert",
            )
            _print_metrics(metrics_bert)
            config = {
                "model": "distilbert-base-uncased",
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
                "precision": "float32",
                "device": metrics_bert.get("device"),
                "train_examples": len(train_rows),
                "test_examples": len(test_rows),
                "label_source": list(set(r.get("label_source", "unknown") for r in rows)),
            }
            save_training_artifacts(
                out_dir / "distilbert", config, train_rows, test_rows, y_pred_bert, metrics_bert, model, tokenizer
            )
        except Exception as exc:
            print(f"DistilBERT execution failed: {exc}", file=sys.stderr)
            return 1

    return 0


def _print_metrics(metrics: Dict[str, Any]) -> None:
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
    print(f"Macro F1:             {metrics['macro_f1']:.3f}")
    if "elapsed_seconds" in metrics:
        print(f"Elapsed Time:         {metrics['elapsed_seconds']:.4f}s")
    if "elapsed_training_seconds" in metrics:
        print(f"Training Time:        {metrics['elapsed_training_seconds']:.2f}s ({metrics.get('seconds_per_step', 0):.3f}s/step across {metrics.get('total_steps', 0)} steps)")
    print("Per-label metrics:")
    for label in ("request", "commitment"):
        m = metrics[label]
        print(f"  {label:<12} Precision: {m['precision']:.3f} | Recall: {m['recall']:.3f} | F1: {m['f1']:.3f} | Support: {m['support']}")


if __name__ == "__main__":
    raise SystemExit(main())
