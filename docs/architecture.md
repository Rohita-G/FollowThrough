# FollowThrough System Architecture & Pipeline Design

FollowThrough is an end-to-end natural language processing (NLP) application that turns workplace emails and messages into **evidence-backed tasks**, links subsequent follow-up messages to those tasks using **dense semantic similarity**, and tracks user-confirmed state changes in an **audit ledger**.

---

## 1. High-Level Architectural Diagram

```mermaid
graph TD
    subgraph UI ["User Interface (Streamlit)"]
        InputMsg["Pasted Email / Message"]
        UI_Tab1["Tab 1: New Task"]
        UI_Tab2["Tab 2: My Tasks"]
        UI_Tab3["Tab 3: Follow-up"]
        UserReview{{"User Confirmation & Edit (Mandatory Gate)"}}
    end

    subgraph NLP_Inference ["NLP Inference Engine"]
        subgraph Classification ["Trained Classification (Non-Gating)"]
            DistilBERT["Fine-Tuned DistilBERT (MPS/CPU)<br/>Multi-label Binary Classification"]
            SlidingWindow["Sliding-Window Inference (128-tok windows)<br/>Experimental Max-Score Aggregation"]
            TFIDF["TF-IDF + Logistic Regression<br/>Baseline Classifier"]
        end
        subgraph Extraction ["Rule-Based Extraction (Not Trained NER)"]
            TaskExtractor["Conservative Linguistic Extractor<br/>Action Verbs, Explicit Owners, Literal Deadlines"]
            OffsetTracker["Exact Character Offsets & Uncertainty Flags"]
        end
        subgraph SemanticLinking ["Pretrained Semantic Linking"]
            MiniLM["sentence-transformers/all-MiniLM-L6-v2<br/>Dense Cosine Similarity Ranking (Top-3)"]
            UpdateDetector["Transparent Update Detector<br/>Rule-based Revisions (Deadline, Owner, Status)"]
        end
    end

    subgraph Persistence ["Persistence Layer"]
        SQLiteDB[("SQLite Store (followthrough_tasks.db)<br/>tasks & task_history tables<br/>Content Hash Deduplication")]
    end

    InputMsg --> DistilBERT
    DistilBERT --> SlidingWindow
    SlidingWindow -.->|"Advisory Scores"| UI_Tab1
    InputMsg -.-> TFIDF
    TFIDF -.->|"Advisory Scores"| UI_Tab1

    InputMsg --> TaskExtractor
    TaskExtractor --> OffsetTracker
    OffsetTracker --> UI_Tab1

    UI_Tab1 --> UserReview
    UserReview -->|"Confirmed Task Record"| SQLiteDB

    SQLiteDB --> UI_Tab2

    SQLiteDB --> MiniLM
    InputMsg -->|"Follow-up Message"| UI_Tab3
    UI_Tab3 --> MiniLM
    UI_Tab3 --> UpdateDetector
    MiniLM -->|"Top-3 Ranked Matches"| UserReview
    UpdateDetector -->|"Proposed Revisions"| UserReview
    UserReview -->|"Confirmed Revisions / Notes"| SQLiteDB
```

---

## 2. Core Architectural Principles

1. **Non-Gating Classifier Design:**
   Sequence classifiers provide advisory likelihood signals for Request and Commitment speech acts. They guide user attention but never gate or block candidate task extraction.
2. **Conservative Linguistic Extraction:**
   Rather than allowing a generative language model to hallucinate commitments or invent dates, task extraction uses explicit linguistic patterns anchored to modal verbs, proper names, and literal surface date expressions.
3. **Exact Character-Level Evidence Offsets:**
   Every extracted task retains `[start_offset:end_offset]`, enabling the user to verify the exact text slice in the original email.
4. **Dense Semantic Retrieval with Human Confirmation:**
   Follow-up messages are retrieved via `all-MiniLM-L6-v2` dense cosine similarity against open tasks. High similarity indicates semantic proximity, but user confirmation is strictly required to link the messages.
5. **Immutable Append-Only Audit Ledger:**
   The SQLite persistence store tracks every creation, reassignment, deadline revision, and status change in a dedicated `task_history` table.

---

## 3. Data Pipeline & Experiment Design

```mermaid
graph TD
    subgraph Ingestion ["Data Ingestion & Cleaning"]
        EnronRaw["Raw Enron Maildir (100k Emails)"]
        CleanScript["Header Parsing, Signature/Quote Stripping"]
        CleanCorpus["1.44M Sentences / 29.56M Tokens / 97,988 retained"]
    end

    subgraph Partitioning ["Graph Clustering & Weak Supervision"]
        UnionFind["Union-Find Thread & Duplicate Clustering"]
        WeakLabeler["Transparent Linguistic Rules (Request/Commitment)"]
        ManifestSplit["Thread-Disjoint Splits<br/>Train: 78,385 | Val: 9,946 | Test: 9,657"]
    end

    subgraph Modeling ["Model Training & Evaluation"]
        TFIDFTrain["Model A: TF-IDF (10k n-grams) + Logistic Regression"]
        DistilBERTTrain["Model B: DistilBERT (5,000 steps, BCEWithLogitsLoss)"]
        EvaluationSuite["Comparative Weak-Label & Human Evaluation"]
    end

    EnronRaw --> CleanScript
    CleanScript --> CleanCorpus
    CleanCorpus --> UnionFind
    UnionFind --> WeakLabeler
    WeakLabeler --> ManifestSplit
    ManifestSplit --> TFIDFTrain
    ManifestSplit --> DistilBERTTrain
    TFIDFTrain --> EvaluationSuite
    DistilBERTTrain --> EvaluationSuite
```

---

## 4. Component Summary

- **`followthrough_core/classifier.py`**: Clean object-oriented wrappers for DistilBERT and TF-IDF classifiers with unified prediction APIs.
- **`followthrough_core/long_message_classifier.py`**: Overlapping 128-token sliding-window inference with token-to-character offset tracking.
- **`followthrough_core/task_extractor.py`**: Rule-based linguistic candidate extraction with character offsets, uncertainty flags, and forwarded-text boundary detection.
- **`followthrough_core/followup_linker.py`**: Bi-encoder similarity ranking using `all-MiniLM-L6-v2` and conservative revision detection (deadline changes, reassignments, completions).
- **`followthrough_core/task_store.py`**: SQLite database managing deduplicated tasks and an append-only change history ledger.
- **`app/app.py`**: Interactive Streamlit application orchestrating the complete workflow.
