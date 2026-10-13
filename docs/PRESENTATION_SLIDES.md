# FollowThrough: 10–15 Minute Capstone Presentation Script & Slide Outline

**Topic:** Evidence-Backed Conversational Task Extraction and State Tracking Across Message Threads  
**Course:** NLP Capstone Project  
**Duration:** 10–15 minutes  

---

## Slide 1: Title & Overview (1 min)
- **Title:** FollowThrough
- **Subtitle:** Evidence-Backed Conversational Task Extraction and State Tracking Across Message Threads
- **Speaker:** Student NLP Project
- **Core Value Proposition:**
  - Modern work happens in email and chats, but action items get lost and deadlines shift across replies.
  - Generative AI often hallucinates owners or dates; silent automated writes destroy user trust.
  - FollowThrough provides an evidence-grounded, human-in-the-loop task state ledger.

---

## Slide 2: Problem & Motivation (1.5 min)
- **The Operational Challenge:**
  - *Dynamic Obligation Drift:* "Sam, please draft the brief by Thursday" followed next day by "Priya will take over, deadline pushed to Friday."
  - *Static NLP Limitation:* Traditional tools extract isolated sentences and forget previous commitments.
  - *The Hallucination Danger:* Autonomous LLMs hallucinate dates and infer unstated owners without boundary checks.
- **System Design Principles:**
  - Never silently invent owners, dates, or completions.
  - Always surface exact character offsets and evidence snippets.
  - Require explicit user confirmation before persisting any task or applying any state revision.

---

## Slide 3: System Architecture (1.5 min)
- **Three-Tier Architecture:**
  1. **UI Layer (Streamlit):** Three interactive workflows (New Task, My Tasks, Follow-up).
  2. **NLP Inference Engine:**
     - *Classification:* TF-IDF and Fine-Tuned DistilBERT for Request/Commitment signal detection (supporting information only).
     - *Extraction:* Rule-based linguistic parser extracting action descriptions, explicit owners, literal dates, and offsets.
     - *Semantic Linking:* Pretrained `all-MiniLM-L6-v2` dense embeddings for matching follow-ups to open tasks.
     - *Update Detection:* Transparent rules detecting revisions (deadlines, owners, status).
  3. **Persistence Layer:**
     - SQLite relational persistence (`tasks`, `task_history` audit ledger with content hash deduplication).

---

## Slide 4: Data & Preprocessing: Enron Corpus v2 (1.5 min)
- **Corpus Scale:**
  - 100,000 parsed emails from Enron.
  - 1,442,694 cleaned sentences | 29,556,736 tokens.
- **Rigorous Data Hygiene:**
  - Excluded 2,012 uninformative records (1,370 empty bodies, 642 under 10 chars).
  - 97,988 clean records retained in canonical manifest.
- **Leakage Prevention via Graph Partitioning:**
  - 42,317 distinct threads and duplicate email groups clustered via Disjoint-Set (Union-Find).
  - Deterministic hash partitioning: 80% Train (78,385), 10% Validation (9,946), 10% Test (9,657).
  - Zero cluster overlap across splits.

---

## Slide 5: NLP Methods & Weak Supervision (1.5 min)
- **Transparent Linguistic Weak Supervision:**
  - Programmatic, interpretable heuristics for Request and Commitment signals.
  - Explicit filtering for negation ("will not"), past actions ("already sent"), permission queries, and quotes.
  - Local Python execution; corporate emails were not sent to commercial LLM APIs.
- **Model Architectures:**
  - *Model A:* Sublinear TF-IDF (10,000 features, 1-2 n-grams) + Class-balanced Logistic Regression.
  - *Model B:* DistilBERT (`distilbert-base-uncased`) fine-tuned for 5,000 steps with BCEWithLogitsLoss on local MPS.
  - *Retrieval:* Pretrained MiniLM embeddings for semantic similarity scoring.

---

## Slide 6: Training & Evaluation Protocol (2 min)
- **Critical Evaluation Caveat:**
  - Metrics reflect *weak-label agreement*, not independently verified ground-truth human accuracy.
- **Empirical Comparison on Identical Held-Out Test Split (N = 9,657):**
  - **Model A: TF-IDF Baseline:**
    - Exact Match: 88.01% | Macro-F1: 0.6874 | Request F1: 0.9081 | Commitment F1: 0.4668
    - Model Size: 8.4 MB | Training Runtime: 21.8 seconds.
  - **Model B: Fine-Tuned DistilBERT (5,000 steps):**
    - Exact Match: 71.45% | Macro-F1: 0.3656 | Request F1: 0.7312 | Commitment F1: 0.0000
    - Model Size: 255.6 MB | Training Runtime: 16.3 minutes (local MPS).
- **Key Finding:** TF-IDF provides an exceptionally competitive baseline with low latency and effective class weighting on sparse commitment labels (~4% support).

---

## Slide 7: Live Application Walkthrough (3 min)
*(Demonstrate the live Streamlit application step-by-step)*
1. **New Task:** Paste email requesting report submission by Friday.
2. **Review Extracted Task:** View non-gating Request score, extracted action, owner, literal deadline, and character offsets.
3. **User Confirmation:** Confirm and save to SQLite ledger.
4. **My Tasks Ledger:** Review saved task card, inspect expandable evidence offsets, and view initial history entry.
5. **Process Follow-up:** Paste reply: "Priya will take over, please push deadline to Monday."
6. **Semantic Linking & Review:** MiniLM retrieves prior task (Top-1 match); system detects owner and deadline revision.
7. **User Confirms Update:** Confirm revision; inspect chronological audit history in SQLite showing prior state preserved.

---

## Slide 8: Limitations & Ethical Disclosure (1.5 min)
- **System Limitations:**
  - Formulaic polite noise ("Please let me know if you have questions") triggers false request signals.
  - Extreme commitment class sparsity (~4%) resulted in DistilBERT predicting zero test commitment positives without class weighting.
  - Relative temporal anaphora ("by Monday") remains literal text without absolute date resolution.
- **AI-Assistance Disclosure:**
  - AI used for script scaffolding, regex drafting, and provisional pilot screening.
  - Weak supervision generated via deterministic local rules.
  - Test human review sheets are preserved with blank gold fields pending manual verification.

---

## Slide 9: Future Work & Conclusion (1 min)
- **Future Directions:**
  - Complete manual annotation on the held-out human review sample (`test_human_review_sample.csv`).
  - Class-weighted loss and focal loss to improve commitment classification in DistilBERT.
  - Hybrid active learning to refine weak supervision heuristics.
- **Conclusion:**
  - Reconstructed dynamic task state across multi-turn email threads.
  - Validated evidence-grounded extraction and semantic retrieval on 100,000 real-world emails.
  - Preserved user trust through strict confirmation gates.
