# Models & Classifier Training Guide

This directory documents the model architectures, training procedures, and inference loading patterns for FollowThrough.

> **Note on Version Control:** Large binary weights (`*.safetensors`, `*.bin`, `*.joblib`, `*.pt`) are strictly excluded from Git via `.gitignore`. Checkpoints are saved locally during training or downloaded from approved model registries.

---

## 1. Model Architectures

FollowThrough deploys three NLP models:

| Component | Architecture / Backbone | Objective / Role | Output |
|---|---|---|---|
| **Primary Classifier** | `distilbert-base-uncased` + Linear Head (2 units) | Fine-tuned multi-label request & commitment speech act classification | Sigmoid probabilities [0.0, 1.0] for Request and Commitment |
| **Baseline Classifier** | TF-IDF (10,000 sublinear n-grams) + Logistic Regression | Classical linear baseline comparison | Probabilities and binary classifications |
| **Follow-up Linker** | Pretrained `sentence-transformers/all-MiniLM-L6-v2` | Dense bi-encoder semantic retrieval | Cosine similarity ranking [0.0, 1.0] |

---

## 2. Training Models

### A. Training the TF-IDF Baseline
To train the TF-IDF vectorizer and logistic regression models:
```bash
python3 scripts/train_full_tfidf.py
```
- **Inputs:** `data/processed/full_corpus_manifest/manifest.jsonl` (Train split: 78,385 messages).
- **Hyperparameters:** `max_features=10000`, `ngram_range=(1, 2)`, `sublinear_tf=True`, `C=1.0`.
- **Validation Tuning:** Automatically scans threshold space [0.20, 0.80] on validation messages to select F1-optimal decision thresholds (`request: 0.45`, `commitment: 0.65`).
- **Output:** Serialized model bundle saved to `models/tfidf_baseline.joblib`.

### B. Fine-Tuning DistilBERT
To fine-tune DistilBERT on local hardware (Apple Silicon MPS or CUDA GPU):
```bash
python3 scripts/train_full_distilbert.py --max_steps 5000 --save_every 2000
```
- **Inputs:** Streamed batch iterable over `manifest.jsonl`.
- **Loss Function:** `nn.BCEWithLogitsLoss()` over 2 output logits.
- **Optimizer:** `AdamW` with learning rate `2e-5`.
- **Checkpointing:** Resumable checkpoints saved periodically to `models/distilbert_checkpoints/latest.pt`.
- **Final Model:** Exported to Hugging Face format in `models/distilbert_fine_tuned/`.

---

## 3. Loading Models in Code

The `followthrough_core.classifier` package provides unified, clean wrappers:

```python
from pathlib import Path
from followthrough_core.classifier import DistilBERTClassifier, TFIDFBaselineClassifier

# 1. DistilBERT Classifier
bert_clf = DistilBERTClassifier(
    model_path_or_name="models/distilbert_fine_tuned",  # or checkpoint directory
    max_length=128,
)
res_bert = bert_clf.predict("Please send the quarterly numbers by Thursday.", req_thresh=0.35, com_thresh=0.20)
print(res_bert)
# Output: {'req_score': 0.82, 'com_score': 0.18, 'req_pred': 1, 'com_pred': 0, ...}

# 2. For messages exceeding 128 tokens, use sliding-window mode:
res_long = bert_clf.predict(long_email_text, use_sliding_window=True, stride=64)

# 3. TF-IDF Baseline Classifier
tfidf_clf = TFIDFBaselineClassifier(
    model_bundle_path="models/tfidf_baseline.joblib"
)
res_tfidf = tfidf_clf.predict("I will complete the audit tomorrow afternoon.")
print(res_tfidf)
# Output: {'req_score': 0.12, 'com_score': 0.74, 'req_pred': 0, 'com_pred': 1}
```

---

## 4. Role in the Pipeline

Both sequence classifiers serve as **supporting advisory signals**. They surface likelihood indicators to guide user attention and do not gate or block task extraction. Even if a model outputs a low score on an edge case, the rule-based extractor processes the entire message text to identify candidate action items with exact character offsets.
