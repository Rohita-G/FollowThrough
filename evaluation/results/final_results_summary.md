# FollowThrough: Final Verified Evaluation Results Summary

This document presents the official, verified evaluation results of the FollowThrough natural language processing pipeline.

---

## 1. Corpus Scale & Dataset Summary

FollowThrough is grounded in the full processed **Enron Email Corpus (v2)**:
- **Total Parsed Emails:** 100,000
- **Retained Records:** **97,988** (after excluding empty and <10-character fragments)
- **Cleaned Sentences:** **1,442,694**
- **Cleaned Tokens:** **29,556,736**
- **Conversation Threads:** 42,317

### Disjoint Partitioning (Zero Cluster Leakage)
Splits were constructed using connected-component graph clustering across duplicate groups and conversation threads:
- **Train Split (~80%):** 78,385 messages (Request: 35,141, Commitment: 3,201)
- **Validation Split (~10%):** 9,946 messages (Request: 4,549, Commitment: 392)
- **Test Split (~10%):** 9,657 messages (Request: 4,270, Commitment: 363)

---

## 2. Weak-Label Full-Corpus Evaluation ($N=9,657$ Test Messages)

Both models were evaluated on the identical held-out test split of 9,657 messages using validation-tuned decision thresholds:

| Evaluation Metric | Model A: TF-IDF Baseline | Model B: Fine-Tuned DistilBERT (5,000 steps) |
|---|---|---|
| **Exact Match Accuracy** | **0.8801** | 0.7145 |
| **Hamming Loss** | **0.0630** | 0.1479 |
| **Macro-F1** | **0.6874** | 0.3656 |
| **Micro-F1** | **0.8708** | 0.7027 |
| **Request Precision** | **0.8996** | 0.6801 |
| **Request Recall** | **0.9169** | 0.7906 |
| **Request F1** | **0.9081** | 0.7312 |
| **Commitment Precision** | **0.4286** | 0.0000 |
| **Commitment Recall** | **0.5124** | 0.0000 |
| **Commitment F1** | **0.4668** | 0.0000 |
| **Inference Time (Test)** | **0.02s** | 57.57s (Apple MPS) |
| **Model Disk Size** | **0.52 MB** | 255.4 MB |

### Detailed Confusion Matrices ($N=9,657$ Test Set)

#### Request Detection:
- **TF-IDF:** TP=3915, FP=437, FN=355, TN=4950 (Support: 4270)
- **DistilBERT:** TP=3376, FP=1588, FN=894, TN=3799 (Support: 4270)

#### Commitment Detection:
- **TF-IDF:** TP=186, FP=248, FN=177, TN=9046 (Support: 363)
- **DistilBERT:** TP=0, FP=11, FN=363, TN=9283 (Support: 363)

### Scientific Analysis:
1. **Weak-Rule Alignment:** The TF-IDF linear baseline achieves higher agreement on programmatic weak labels because its n-gram features directly memorize the surface keyword forms from the weak supervision rules.
2. **Class Imbalance in DistilBERT:** Commitments account for only ~3.8% of training samples. DistilBERT fine-tuned for 5,000 steps under-predicted rare commitments without heavy loss reweighting.
3. **Non-Gating System Design:** These findings justify FollowThrough's core architectural principle: sequence classifiers supply advisory likelihood scores, while conservative linguistic extraction independently parses the complete text.

---

## 3. Human Ground-Truth Evaluation Status

- **Held-out Review Sheet:** `data/annotations/test_human_review_sample.csv` ($N=50$) and `data/annotations/evaluation_review_sheet.csv` ($N=38$).
- **Status:** **Not yet available: human labels pending**.
- **Explanation:** In strict adherence to scientific integrity rules, human ground-truth metrics are marked as pending rather than fabricated. The human review protocol will establish an unbiased evaluation of real-world task extraction precision and semantic follow-up link accuracy.

---

## 4. Pipeline Component Summary
- **Classification:** Dual models (DistilBERT + TF-IDF) with 128-token sliding-window inference.
- **Extraction:** Rule-based heuristic extraction with character-level evidence tracking `[start_offset:end_offset]`.
- **Follow-up Linking:** Dense bi-encoder cosine similarity via `all-MiniLM-L6-v2`.
- **Persistence:** SQLite task ledger with immutable append-only state transition history.
