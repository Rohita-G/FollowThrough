# FollowThrough-Final

FollowThrough is an evidence-backed conversational task extraction and state-tracking natural language processing (NLP) application. It transforms unstructured emails and workplace messages into verifiable, evidence-backed action items, connects subsequent follow-up messages using dense bi-encoder semantic similarity (`all-MiniLM-L6-v2`), and maintains an immutable audit ledger of user-confirmed state changes in SQLite.

---

## Team Information
- **Team Members:** Rohita Gangishetty
- **Course:** Natural Language Processing Capstone Project
- **Term:** Fall 2026

---

## Problem Statement
In modern asynchronous workplace communication (e.g., email threads, enterprise messaging), team members frequently make commitments ("I will file the brief tomorrow") and issue requests ("Can you review this draft by Thursday?"). Over long conversational threads, these obligations often get buried, slip past deadlines, or get re-assigned without clear visibility. Conventional NLP tools either perform isolated sentence sentiment classification or use generative models prone to hallucinating obligations, non-existent assignees, or fabricated deadlines. FollowThrough resolves this by combining multi-label sequence classification, conservative linguistic extraction with exact character offsets, semantic retrieval, and a human-in-the-loop audit ledger.

---

## Features
- **Speech Act Signal Classification:** Dual classifiers (Fine-tuned DistilBERT and TF-IDF + Logistic Regression baseline) predict Request and Commitment speech act probabilities.
- **Sliding-Window Inference:** Supports messages of any length with overlapping 128-token sliding windows and token-to-character offset alignment.
- **Conservative Linguistic Extraction:** Identifies candidate task descriptions, explicit assignees, and literal deadline expressions without generative hallucination.
- **Exact Character-Level Evidence Offsets:** Retains `[start_offset:end_offset]` character slices enabling users to verify exact source context.
- **Dense Semantic Follow-up Linking:** Uses `sentence-transformers/all-MiniLM-L6-v2` to rank open tasks against incoming follow-up messages.
- **Rule-Based Update Detection:** Detects proposed deadline revisions, owner reassignments, and completions while rejecting pure courtesies (e.g., "Thanks!").
- **SQLite Audit Ledger:** Persists confirmed tasks with SHA-256 deduplication and an append-only change history table.
- **Interactive Streamlit Interface:** Three-tab application for pasting messages, reviewing tasks, managing stored tasks, and linking follow-up messages.

---

## Architecture Overview

```
[ Email / Message Input ]
           │
           ├───► [ Sequence Classifier ] (DistilBERT / TF-IDF Advisory Scores)
           │
           └───► [ Task Extractor ] (Conservative Linguistic Patterns & Character Offsets)
                       │
                       ▼
             [ Streamlit UI Review ] ◄── (Mandatory User Gate)
                       │
                       ▼
             [ SQLite Task Ledger ] (Tasks & Append-Only History)
                       │
[ Follow-up Message ] ─┼───► [ MiniLM Bi-Encoder ] (Semantic Similarity Ranking)
                       │
                       └───► [ Update Detector ] (Revisions: Deadlines, Owners, Status)
                                   │
                                   ▼
                         [ Confirmed Ledger Update ]
```

---

## Installation Instructions

### Prerequisites
- Python 3.10+
- macOS, Linux, or Windows

### Setup
1. Clone or navigate to the repository directory:
   ```bash
   cd FollowThrough-Final
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Reproduce the project

To reproduce the complete FollowThrough project from scratch:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --full
python scripts/train_full_tfidf.py
python scripts/train_full_distilbert.py
python scripts/evaluate_models.py
python scripts/evaluate_extraction_linking.py
python scripts/generate_final_results.py
streamlit run app/app.py
```

> [!TIP]
> For rapid verification without processing the full 100,000 emails, you can run a non-destructive smoke test:
> ```bash
> python scripts/prepare_data.py --smoke
> ```

---

## How to Run the Streamlit App

Launch the interactive web application from the project root:
```bash
streamlit run app/app.py
```
Then navigate to `http://localhost:8501` in your browser.

### Application Workflow:
1. **Tab 1 (New Task):** Paste an email or message, click **Analyze Message**, review detected signals, verify the exact evidence substring and character offsets, edit if needed, and click **Confirm & Save Task**.
2. **Tab 2 (My Tasks):** Browse stored tasks, inspect original evidence slices, view change histories, and record verified status updates with supporting notes.
3. **Tab 3 (Follow-up):** Paste a follow-up message, click **Analyze Follow-up & Rank Matches**, inspect top semantic candidates, review proposed deadline or owner updates, and confirm the revision into the task ledger.

---

## How to Run Evaluation

To display the verified benchmark comparison table across both models on the held-out test split ($N=9,657$):
```bash
python scripts/evaluate_models.py
```

To evaluate task extraction and semantic linking against human review sheets:
```bash
python scripts/evaluate_extraction_linking.py
```

To generate the complete set of 8 official evaluation result artifacts in `evaluation/results/`:
```bash
python scripts/generate_final_results.py
```

---

## How to Run Unit Tests

Execute the test suite using `pytest`:
```bash
pytest tests/
```

---

## Repository Structure

```
FollowThrough-Final/
├── data_pipeline/                 # Reproducible data-provenance modules
│   ├── __init__.py                # Package exports
│   ├── download_enron.py          # Enron corpus tarball download & extraction
│   ├── select_emails.py           # Mailbox-prioritized email file selection
│   ├── parse_emails.py            # RFC 822 MIME header and body parser
│   ├── clean_emails.py            # Boundary stripping, quote removal, body cleaning
│   ├── build_threads.py           # Subject normalization & thread reconstruction
│   ├── build_corpus_manifest.py   # Disjoint graph splitting (zero leakage) & manifest builder
│   └── generate_weak_labels.py    # Transparent regex rules for request & commitment pseudo-labels
├── followthrough_core/            # Core NLP & state tracking modules
│   ├── __init__.py                # Package exports
│   ├── classifier.py              # DistilBERT and TF-IDF classifier wrappers
│   ├── long_message_classifier.py # Sliding-window tokenization and inference
│   ├── task_extractor.py          # Conservative linguistic rule extractor & offsets
│   ├── temporal_normalizer.py     # Calendar grounding engine for relative temporal expressions
│   ├── followup_linker.py         # MiniLM semantic ranking & update detector
│   └── task_store.py              # SQLite task ledger & append-only history
├── scripts/                       # Executable CLI workflows & evaluation scripts
│   ├── prepare_data.py            # Unified data preparation (--full and --smoke)
│   ├── train_full_tfidf.py        # TF-IDF baseline training & threshold optimization
│   ├── train_full_distilbert.py   # DistilBERT fine-tuning script with checkpoints
│   ├── train_weighted_distilbert.py # Class-weighted training & oversampling experiments
│   ├── tune_thresholds_experiment.py # Threshold sweeping on validation PR curves
│   ├── evaluate_models.py         # Comparative classification model evaluation
│   ├── evaluate_extraction_linking.py # Extraction and linking evaluation on review sheet
│   ├── generate_final_results.py  # Generates canonical artifacts in evaluation/results/
│   ├── generate_paper_pdf.py      # Automated IEEE report PDF compilation
│   └── build_full_corpus_manifest.py # Full corpus manifest & disjoint split builder
├── app/                           # Streamlit user interface
│   └── app.py                     # Interactive 3-tab application
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_task_workflow.py      # Offsets, deduplication, and semantic ranking
│   ├── test_long_message_classifier.py # Sliding window coverage & token bounds
│   ├── test_classifier.py         # Classifier wrapper unit tests
│   └── test_temporal_normalizer.py# Relative deadline normalization unit tests
├── evaluation/
│   ├── README.md                  # Evaluation guidelines and dataset notes
│   └── results/                   # Final evaluation result artifacts & comparison
├── data/
│   ├── README.md                  # Enron corpus acquisition & preprocessing guide
│   └── annotations/               # Stratified test review sheets for human audit
├── models/
│   └── README.md                  # Model architectures, training, and loading guide
├── docs/
│   ├── figures/                   # High-resolution system architecture diagrams
│   │   └── system_architecture.png# End-to-end architecture & workflow diagram
│   ├── project_report.pdf         # Compiled IEEE conference research paper (PDF)
│   ├── project_report.tex         # Complete IEEEtran LaTeX source code for Overleaf
│   └── project_report.md          # Multi-section markdown project documentation
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project overview & documentation
```

---

## Capabilities & Limitations
1. **Temporal Grounding Engine:** Relative deadlines (e.g., "for next Monday at 10 AM", "by Friday afternoon", "tomorrow EOD") are anchored to the message origin timestamp and projected into verified ISO calendar dates and visual calendar badges.
2. **Linguistic Heuristic Precision:** The rule-based extractor uses conservative patterns to prevent hallucinations; implicit or colloquial commitments (e.g., "Consider it done") can be managed or augmented through the interactive Task Board.
3. **Advisory Classifiers:** Classification scores provide advisory indicators to guide user attention and do not gate extraction. Weak-supervision label bias was observed on rare commitments in DistilBERT at 5,000 steps without heavy class reweighting.
