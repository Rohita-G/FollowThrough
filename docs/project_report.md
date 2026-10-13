# FollowThrough: Evidence-Backed Conversational Task Extraction and State Tracking

**Course:** NLP Capstone Project  
**Author / Team:** FollowThrough Project Team  
**Date:** October 2026  
**Format:** IEEE Multi-Section Research & Engineering Report  

---

## Abstract
Modern workplace communication frequently establishes operational commitments and requests that become lost across asynchronous message threads. While conventional natural language processing (NLP) pipelines classify isolated sentences or extract static action items, they rarely model dynamic conversational revisions where subsequent messages alter assignees, shift deadlines, or resolve dependencies. In this paper, we present **FollowThrough**, an evidence-backed task extraction and state tracking system combining sequence classification, conservative linguistic extraction, dense semantic sentence retrieval, and user-confirmed state tracking. We ground our evaluation in an active real-world corpus of 100,000 parsed emails from the Enron dataset (comprising 97,988 retained records, 42,317 threads, 1.44 million cleaned sentences, and 29.56 million tokens). Using transparent rule-based weak supervision and connected-thread graph partitioning, we train and evaluate a TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT multilabel classifier (5,000 steps) on identical held-out test splits ($N=9,657$). In addition, we deploy pretrained sentence embeddings (`all-MiniLM-L6-v2`) for follow-up message retrieval and candidate change detection. The system enforces a strict human-in-the-loop verification paradigm: no task or state transition is committed to the persistent SQLite ledger without explicit user confirmation, preventing hallucinated obligations. We report comprehensive weak-label agreement metrics, establish the held-out human evaluation framework, and analyze systemic failure modes.

---

## I. Introduction
Asynchronous workplace collaboration relies heavily on text-based communication channels such as email and enterprise chat. Within these streams, commitments ("I will draft the brief by Thursday") and requests ("Can you review the spreadsheet?") define operational momentum. However, conversational commitments are rarely static: deadlines slip, responsibilities shift, and prior agreements are superseded by subsequent correspondence.

Traditional natural language understanding (NLU) systems treat action-item detection as an isolated sentence classification or named entity recognition (NER) problem. This paradigm exhibits critical limitations:
1. **Isolated Context:** Models classify isolated utterances without maintaining thread continuity, failing to recognize when an earlier task is updated or cancelled.
2. **Hallucination Risk:** Generative language models frequently invent task owners, exact dates, or completion statuses when prompting lacks strict boundary constraints.
3. **Premature Automation:** Silent automated writes to task databases without user verification lead to severe operational errors and distrust.

To resolve these challenges, FollowThrough operationalizes a bounded, verifiable workflow:
- **Signal Classification:** Identifying request and commitment speech acts to guide user attention.
- **Evidence Extraction:** Extracting task descriptions, literal deadlines, and explicit owners alongside character offsets and uncertainty flags using conservative linguistic rules.
- **Semantic Thread Linking:** Aligning follow-up messages to prior open tasks via sentence embeddings.
- **State Revision:** Proposing structured updates (owner reassignment, date modifications, state transitions) that require explicit user confirmation before persistent SQLite ledger commits.

---

## II. Objectives
The technical objectives of the FollowThrough project comprise:
1. **Corpus Scale & Processing:** Ingest and process a large-scale real-world email dataset exceeding 50,000 sentences (achieving 1.44M sentences / 29.56M tokens across 100,000 parsed emails and 97,988 included records) with rigorous body normalization, header parsing, and conservative duplicate grouping.
2. **Multi-Task NLP Integration:** Formulate a cohesive pipeline combining trained sequence classification, conservative rule-based linguistic extraction, and dense pretrained sentence retrieval.
3. **Model Comparison:** Implement and empirically compare a classical linear baseline (TF-IDF + Logistic Regression) against an advanced fine-tuned transformer (DistilBERT with BCEWithLogitsLoss, 5,000 steps) across identical held-out test splits ($N=9,657$).
4. **Transparent Evaluation:** Disentangle weak-label agreement on programmatic labels from human-verified ground-truth performance, providing reproducible metrics and error analyses.
5. **Human-in-the-Loop Verifiability:** Engineer an interactive Streamlit application integrated with SQLite persistence where consequential state transitions require explicit human authorization.

---

## III. Approaches & Methods
FollowThrough coordinates four distinct NLP components:

### A. Request & Commitment Classification
We formulate utterance-level signal detection as a multi-label binary classification problem with two non-mutually exclusive targets:
- **Request:** An imperative, interrogative, or deontic solicitation directed toward a collaborator.
- **Commitment:** An explicit commissive promise, assurance, or undertaking made by the sender.

We evaluate two distinct architectures:
- **Model A (TF-IDF Baseline):** Sublinear TF-IDF word and character n-gram extraction (10,000 features, n-gram range 1–2) paired with independent Logistic Regression classifiers with balanced class weighting.
- **Model B (Fine-Tuned DistilBERT):** Pretrained `distilbert-base-uncased` configured with a two-unit classification head trained for 5,000 steps with binary cross-entropy loss (`BCEWithLogitsLoss`). For messages exceeding 128 tokens, we implement an experimental sliding-window tokenization mechanism with 64-token stride.

### B. Rule-Based Task & Event Extraction
To avoid unconstrained generative hallucinations, candidate task descriptions, explicit owners, and literal temporal expressions are identified via conservative linguistic patterns:
- Action descriptions are anchored to active modal verbs (`need to`, `please`, `will`, `should`).
- Owners are extracted strictly when explicit proper names or speaker pronouns appear in proximity to the verb phrase.
- Deadlines and temporal anchors are preserved as literal surface strings (e.g., "by Friday 5 PM", "tomorrow morning") rather than speculative projections.
- Character-level offsets and heuristic uncertainty flags (`missing_explicit_owner`, `missing_explicit_deadline`, `quoted_or_forwarded_text`) are surfaced directly to the user.

### C. Pretrained Semantic Follow-Up Linking
To retrieve the open parent task relevant to a new follow-up message, candidate tasks are embedded using `sentence-transformers/all-MiniLM-L6-v2`. FollowThrough computes cosine similarity between the incoming follow-up query vector and each stored action description:
$$\text{Sim}(q, t) = \frac{\mathbf{e}_q \cdot \mathbf{e}_t}{\|\mathbf{e}_q\| \|\mathbf{e}_t\|}$$
The top-3 candidate tasks are surfaced to the user. Crucially, semantic similarity is treated as a ranking indicator, not proof of a relationship: explicit human confirmation is required to link messages.

### D. Conservative Update Detection
Rule-based patterns scan the follow-up text for explicit revisions:
- **Deadline Revisions:** Temporal shift markers (`postponed to`, `moved to`, `new deadline is`).
- **Owner Reassignments:** Delegative markers (`reassigned to`, `handing over to`).
- **Completions & Cancellations:** Explicit fulfillment phrases (`I have completed`, `finished the review`, `cancel this`). Courtesies like "Thanks!" are explicitly rejected from triggering completion.

---

## IV. System Workflow & Data Ledger

The end-to-end workflow comprises four phases:
1. **Input & Advisory Inference:** The user pastes a message. The sequence classifier computes request and commitment probabilities. Concurrently, the linguistic extractor processes the full text to produce candidate tasks with exact offsets.
2. **Review & Confirmation:** The user reviews suggested tasks, modifies any field, and confirms. The confirmed record is written to SQLite with a content hash (`SHA256(source || action)`) to prevent duplication.
3. **Follow-up Alignment:** A subsequent message is analyzed. Dense embeddings retrieve candidate parent tasks; update rules detect potential revisions.
4. **State Revision Ledger:** Upon user approval, the database updates the current task record and appends a structured entry to `task_history` (timestamp, update type, old value, new value, exact evidence, contextual snippet).

---

## V. Dataset & Corpus Preprocessing

### A. Corpus Scale
The underlying dataset is derived from the Carnegie Mellon University Enron email archive:
- **Total Ingested Messages:** 100,000 raw messages across employee mailboxes.
- **Retained Records:** 97,988 emails (after filtering empty bodies and <10-character fragments).
- **Cleaned Sentences:** 1,442,694 sentences.
- **Token Volume:** 29.56 million tokens.
- **Conversation Threads:** 42,317 threads.

### B. Thread-Disjoint Partitioning via Connected Graphs
Email archives contain duplicate messages across mailboxes. To prevent data leakage:
1. Duplicate keys are generated: `SHA256(body) || sender || normalized_subject`.
2. A DisjointSet (Union-Find) structure unions each thread ID and duplicate group into connected clusters.
3. Cluster-level deterministic hashing allocates:
   - **Train Split (80%):** 78,385 messages
   - **Validation Split (10%):** 9,946 messages
   - **Test Split (10%):** 9,657 messages
Zero thread or duplicate overlap exists across splits.

---

## VI. Hyperparameters & Experimental Setup

| Parameter / Setting | Model A: TF-IDF + Logistic Regression | Model B: Fine-Tuned DistilBERT |
|---|---|---|
| **Feature Space / Backbone** | 10,000 n-grams (1–2), sublinear TF | `distilbert-base-uncased` (66M params) |
| **Loss Function** | Log-loss with balanced class weights | `BCEWithLogitsLoss` (multi-label) |
| **Optimizer & Learning Rate** | L-BFGS, max iter = 1,000 | AdamW, lr = $2 \times 10^{-5}$ |
| **Batch Size & Steps** | Full-batch sparse matrix | Batch size = 2, 5,000 steps |
| **Decision Thresholds** | Request: 0.45, Commitment: 0.65 | Request: 0.35, Commitment: 0.20 |
| **Hardware** | Apple Silicon M-series (CPU) | Apple Silicon MPS (Metal Performance Shaders) |

---

## VII. Evaluation & Discussion

### A. Empirical Weak-Label Test Results ($N=9,657$)
Both models were evaluated on the identical held-out test split of 9,657 messages:

| Metric | TF-IDF Baseline | DistilBERT (5k steps) |
|---|---|---|
| **Exact Match Accuracy** | **0.8801** | 0.7145 |
| **Hamming Loss** | **0.0630** | 0.1479 |
| **Macro-F1** | **0.6874** | 0.3656 |
| **Micro-F1** | **0.8708** | 0.7027 |
| **Request Precision / Recall / F1** | 0.8996 / 0.9169 / **0.9081** | 0.6801 / 0.7906 / 0.7312 |
| **Commitment Precision / Recall / F1**| 0.4286 / 0.5124 / **0.4668** | 0.0000 / 0.0000 / 0.0000 |
| **Inference Time (Test Split)** | **0.02s** | 57.57s |
| **Disk Footprint** | **0.52 MB** | 255.4 MB |

### B. Analysis of Findings
1. **Weak-Supervision Alignment:** The TF-IDF model achieved superior agreement with programmatic pseudo-labels (Macro-F1 0.6874 vs. 0.3656). Because weak labels were derived from keyword and regex rules, linear n-gram features readily memorize those exact surface forms.
2. **Commitment Class Imbalance:** Commitments appear in only ~3.8% of messages. In DistilBERT, 5,000 gradient updates without extreme focal weighting caused the model to under-predict the rare commitment class, leading to zero true positives on the test pseudo-label.
3. **The Necessity of Human Evaluation:** These findings highlight that programmatic weak labels can introduce systemic distribution bias. Although FollowThrough uses the full preprocessed Enron corpus (97,988 retained emails, 1.44M sentences, 29.56M tokens) for preprocessing, model training, and large-scale evaluation, a stratified human-annotated test set of 200 messages was created to provide an unbiased estimate of real-world task-detection performance.

---

## VIII. Limitations & Future Work
1. **Linguistic Heuristic Boundaries:** Rule-based extraction can miss nuanced commitments phrased colloquially or indirectly (e.g., "Consider that handled").
2. **Relative Temporal Grounding:** Literal temporal extraction captures "by Friday" accurately as a surface string, but does not ground the date into an absolute ISO calendar date without sender timezone and message header context.
3. **Future Directions:** Transitioning to token-level sequence labeling (BIO NER tagging) trained on high-agreement human annotations, incorporating conversational thread dialogue acts, and exploring calibrated thresholding across diverse enterprise domains.

---

## IX. Conclusion
FollowThrough demonstrates that practical conversational NLP systems must extend beyond isolated sentence classification. By integrating sequence classification, conservative linguistic extraction with exact evidence offsets, bi-encoder semantic linking, and persistent state ledgering within an interactive human-in-the-loop application, FollowThrough provides a robust, verifiable solution to operational commitment tracking in email and workplace messaging.

---

## References
1. B. Klimt and Y. Yang, "The Enron Corpus: A New Dataset for Email Classification Research," in *European Conference on Machine Learning (ECML)*, 2004, pp. 217–226.
2. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv preprint arXiv:1910.01108*, 2019.
3. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *EMNLP*, 2019, pp. 3982–3992.
4. P. Turney, "Thumbs up or thumbs down? Semantic orientation applied to unsupervised classification of reviews," in *ACL*, 2002, pp. 417–424.
5. A. Ratner et al., "Snorkel: Rapid Training Data Creation with Weak Supervision," *VLDB*, vol. 11, no. 3, pp. 269–282, 2017.
