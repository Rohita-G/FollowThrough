# FollowThrough: Evidence-Backed Conversational Task Extraction and State Tracking

**Course:** NLP Capstone Project  
**Author:** Student NLP Project  
**Date:** October 2026  
**Format:** IEEE Standard Multi-Section Report  

---

## Abstract
Modern asynchronous workplace communication frequently establishes operational commitments and requests that become stale or lost across multi-turn email threads. While conventional natural language processing (NLP) pipelines extract static action items from isolated utterances, they rarely model dynamic conversational revisions where subsequent messages alter assignees, shift deadlines, or resolve dependencies. In this paper, we introduce **FollowThrough**, an evidence-backed task tracking system that integrates sequence classification, linguistic extraction, dense semantic sentence retrieval, and user-confirmed state tracking. We leverage an active real-world corpus of 100,000 parsed emails from the Enron dataset (comprising 97,988 included records, 42,317 threads, 1.44 million cleaned sentences, and 29.56 million tokens). Using transparent, rule-based weak supervision and connected-thread graph partitioning, we train and evaluate a traditional TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT multilabel classifier (5,000 steps) on local Apple Silicon hardware. In addition, we deploy pretrained sentence embeddings (`all-MiniLM-L6-v2`) for follow-up message retrieval and candidate change detection. Crucially, the system enforces a strict human-in-the-loop verification paradigm: no task or state transition is committed without explicit user confirmation, preventing hallucinated obligations. We report comprehensive weak-label agreement metrics across all 9,657 held-out test messages, outline the pending human reference evaluation protocol, and discuss systemic failure modes.

---

## I. Introduction
Asynchronous workplace collaboration relies heavily on text-based communication channels such as email and enterprise chat. Within these streams, commitments ("I will draft the brief by Thursday") and requests ("Can you review the spreadsheet?") define project momentum. However, conversational interactions are non-static: deadlines slip, responsibilities shift, and prior agreements are superseded by subsequent correspondence.

Traditional natural language understanding (NLU) systems treat action-item detection as an isolated sentence classification or named entity recognition (NER) problem. This paradigm exhibits critical limitations:
1. **Isolated Context:** Models classify isolated sentences without maintaining thread continuity, failing to recognize when an earlier task is updated or cancelled.
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
- **Model B (Fine-Tuned DistilBERT):** Pretrained `distilbert-base-uncased` configured with a two-unit classification head trained for 5,000 steps with binary cross-entropy loss (`BCEWithLogitsLoss`). For messages exceeding 128 tokens, we implement an experimental sliding-window tokenization mechanism with 32-token overlap.

### B. Rule-Based Task & Event Extraction
To avoid unconstrained generative hallucinations, candidate task descriptions, explicit owners, and literal temporal expressions are identified via conservative linguistic patterns and dependency structures:
- Action descriptions are anchored to active modal verbs (`need to`, `please`, `will`, `should`).
- Owners are extracted strictly when explicit proper names or second-person vocatives appear in proximity to the verb phrase.
- Deadlines and temporal anchors are preserved as literal surface strings (e.g., "by Friday 5 PM", "tomorrow morning") rather than speculative calendar projections.
- Character-level offsets and heuristic uncertainty flags (`UNCERTAIN_OWNER`, `UNCERTAIN_DEADLINE`, `FORWARDED_CONTENT`) are surfaced directly to the user.

### C. Pretrained Semantic Follow-Up Linking
When a user inputs a follow-up message into an active thread, FollowThrough retrieves prior open tasks from SQLite. Using `sentence-transformers/all-MiniLM-L6-v2`, the system computes dense 384-dimensional semantic embeddings for the follow-up text and open task descriptions:
$$\text{Sim}(u, t) = \frac{\mathbf{e}_u \cdot \mathbf{e}_t}{\|\mathbf{e}_u\| \|\mathbf{e}_t\|}$$
Tasks are ranked by cosine similarity, presenting the Top-3 candidates to the user.

### D. Candidate Change Detection & State Revision
A secondary rule engine scans the follow-up text for explicit revision cues:
- Date modifications (e.g., "push to Monday", "extended to the 15th").
- Ownership reassignment (e.g., "Priya will take this over").
- Completion signals (e.g., "done", "finished", "completed", "cancelled").
The proposed state transition remains provisional until confirmed in the UI.

---

## IV. System Workflow & Architecture
The system architecture follows a clean three-tiered decoupling:
1. **Presentation Layer (Streamlit):**
   - *Tab 1 (New Task):* Allows pasting messages, inspects classification scores, displays editable task cards with highlighted evidence spans.
   - *Tab 2 (My Tasks):* Manages SQLite records, displays expandable source evidence, and tracks chronological revision histories.
   - *Tab 3 (Follow-up):* Takes reply messages, retrieves Top-3 matching tasks via MiniLM embeddings, flags supported changes, and applies confirmed updates.
2. **Storage Layer:**
   - SQLite relational store (`followthrough_tasks.db`) tracking tasks, metadata, content hashes, and immutable change logs (`task_history`).
3. **Offline Experimentation & Evaluation Engine:**
   - Reproducible batch pipelines for manifest compilation, weak supervision, model training, and metric scoring.

---

## V. Datasets & Manifest Partitioning

### A. Corpus Volume & Characteristics
The project utilizes the `enron_subset_v2` corpus:
- **Raw Messages Analyzed:** 100,000 emails.
- **Cleaned Sentences:** 1,442,694 sentences.
- **Token Volume:** 29,556,736 tokens.
- **Average Sentences per Email:** 14.43.
- **Thread Count:** 42,317 distinct threads (20,506 singletons, 21,811 multi-message threads).

### B. Deduplication & Manifest Filtering
Mailbox folder redundancy was handled conservatively:
- Normalization on body SHA-256 hashes, sender addresses, and subjects.
- Exclusion of uninformative records: 2,012 messages excluded (1,370 empty or whitespace bodies; 642 messages under 10 characters).
- Included active dataset: **97,988 messages**.

### C. Connected Thread & Cluster Partitioning
To guarantee zero data leakage between training, validation, and test splits:
- Duplicate groups and threaded messages were merged using a Disjoint Set (Union-Find) algorithm across all 42,317 threads.
- Partitioning was executed deterministically via MD5 hash buckets on cluster roots:
  - **Train Split:** 78,385 messages (79.99%)
  - **Validation Split:** 9,946 messages (10.15%)
  - **Test Split:** 9,657 messages (9.86%)
- Verification confirmed zero cluster overlap across partitions.

### D. Transparent Weak Supervision
Weak labels were generated locally using deterministic linguistic heuristics:
- **Request Rules:** Interrogative modals (`can you`, `could you`), polite imperatives (`please send`, `kindly provide`), deontic obligation markers (`need you to`), and request verb dependencies.
- **Commitment Rules:** First-person future commissives (`I will`, `we will`, `I'll make sure`), promises, and guarantees.
- Negation patterns (`will not`, `can't`), past tense forms (`sent`, `already provided`), and quoted forward blocks were filtered out.
- **Label Distributions:**
  - Train: 35,141 Request (44.8%), 3,201 Commitment (4.1%)
  - Validation: 4,549 Request (45.7%), 392 Commitment (3.9%)
  - Test: 4,270 Request (44.2%), 363 Commitment (3.8%)

---

## VI. Experimental Parameters & Setup
- **Hardware:** Apple Silicon (MPS backend) and multicore CPU.
- **TF-IDF Baseline:** Maximum features = 10,000, sublinear TF scaling, n-gram range (1, 2), L2 regularization C=1.0.
- **DistilBERT Architecture:** `distilbert-base-uncased`, 6 transformer layers, 768 hidden dimensions, 12 attention heads, 66M parameters.
- **Optimization:** AdamW optimizer, learning rate $\eta = 2 \times 10^{-5}$, batch size = 2, sequence length = 128 tokens, BCEWithLogitsLoss. Fine-tuned for 5,000 steps with saved checkpoints.
- **Threshold Selection:** Tuned on the validation split to maximize macro-averaged F1 score ($F_{1,\text{macro}}$). Selected thresholds: Request = 0.35, Commitment = 0.20.

---

## VII. Evaluation & Discussion

### A. Critical Metric Distinctions
We emphasize a fundamental distinction in our findings:
> **Notice:** Metrics reported against programmatic weak labels measure *algorithmic alignment and weak-label agreement*, **not** independently verified ground-truth human accuracy.

### B. Two-Model Empirical Comparison
Both models were evaluated on the **identical full held-out test split** ($N = 9,657$ messages):

| Metric | Model A: TF-IDF + Logistic Reg | Model B: Fine-Tuned DistilBERT (5k steps) |
| :--- | :---: | :---: |
| **Test Examples Evaluated** | **9,657** | **9,657** |
| **Exact Match Accuracy** | **0.8801** | *0.7145* |
| **Hamming Loss** | **0.0632** | *0.1479* |
| **Macro-F1** | **0.6874** | *0.3656* |
| **Micro-F1** | **0.8715** | *0.7027* |
| **Request Precision** | **0.8996** | *0.6801* |
| **Request Recall** | **0.9169** | *0.7906* |
| **Request F1** | **0.9081** | *0.7312* |
| **Commitment Precision** | **0.4286** | *0.0000* |
| **Commitment Recall** | **0.5124** | *0.0000* |
| **Commitment F1** | **0.4668** | *0.0000* |
| **Model Size on Disk** | **8.4 MB** | **255.6 MB** |
| **Training Run** | **21.8 s** | **16.3 min (5,000 steps on MPS)** |

### C. Analysis of Model Behavior
1. **Strong Agreement on Requests:** Both models achieve solid agreement on requests (F1 = 0.9081 for TF-IDF; F1 = 0.7312 for DistilBERT) due to salient surface lexical markers (`please`, `can you`, imperative verbs).
2. **Sparsity in Commitments:** Commitment signals exhibit severe class imbalance (only 363 positive examples in test, ~3.8%). TF-IDF with balanced class weighting maintained recall (0.5124) and achieved an F1 of 0.4668. In contrast, unweighted DistilBERT with 5,000 steps predicted zero true positives on the test split, demonstrating that standard cross-entropy without focal or class-weighted loss collapses to the majority class on highly skewed corpora.
3. **Compute vs Performance:** The linear TF-IDF baseline performs remarkably well under weak supervision while training in under 22 seconds with an 8.4 MB disk footprint.

### D. Human Reference Protocol & Pending Evaluation
To evaluate task extraction and semantic linking independently of weak classification labels:
- A held-out human review sample (`data/annotations/test_human_review_sample.csv`, $N=50$) has been compiled with blank ground-truth fields.
- A pilot review sheet (`data/annotations/evaluation_review_sheet.csv`, $N=38$) has been formatted with blank human correction columns.
- **Evaluation Status:** Human extraction precision/recall and semantic linking MRR remain **pending human annotation**. In keeping with strict academic integrity standards, numerical extraction and linking scores will be calculated only after manual verification is complete.

---

## VIII. Error Analysis & Limitations
1. **Formulaic Polite Noise:** Routine closing phrases such as "Please let me know if you have questions" trigger false positive request labels despite carrying negligible task content.
2. **Ambiguous Temporal Anaphora:** Deictic expressions such as "by next Tuesday" or "by tomorrow" depend on message transmission headers and require human contextual confirmation.
3. **Commitment Sparsity:** Severe label scarcity for commitments demands specialized loss functions (e.g., Focal Loss, class-weighted BCE) when fine-tuning transformers.
4. **Weak Label Noise:** Because programmatic heuristics generate the training targets, model errors often reflect the intrinsic imperfections of the weak labeling functions rather than learning failures.

---

## IX. Conclusion
FollowThrough demonstrates that combining lightweight classification signals, conservative linguistic rules, dense semantic retrieval, and strict human confirmation yields an effective and trustworthy assistant for managing commitments across asynchronous conversations. By grounding every action in verifiable textual evidence and refusing to perform silent automated writes, the system eliminates hallucination risks inherent in end-to-end generative approaches. Future work will center on completing human-annotated evaluations and introducing class-weighted objectives for commitment modeling.

---

## References
1. V. R. Carvalho and W. W. Cohen, "On the collective classification of email speech acts," in *Proc. 28th Annual Int. ACM SIGIR Conf.*, 2005.
2. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in *NeurIPS EMC^2 Workshop*, 2019.
3. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019.
4. B. Klimt and Y. Yang, "The Enron corpus: A new dataset for email classification research," in *Proc. ECML*, 2004.
5. A. Ratner, C. M. De Sa, S. Wu, D. Sontag, and C. Ré, "Snorkel: Rapid training data creation with weak supervision," in *Proc. VLDB Endowment*, vol. 11, no. 3, 2017.
