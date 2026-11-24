# FollowThrough: Evidence-Backed Conversational Task Extraction and State Tracking in Asynchronous Workplace Communication

**Author:** Rohita Gangishetty  
**Course:** Natural Language Processing Capstone Project  
**Term:** Fall Term 2026  
**Format:** IEEE Multi-Section Research & Engineering Report  

---

## Abstract
In modern enterprise environments, asynchronous communication channels such as email threads and collaborative chat streams function as the primary substrate for organizational coordination. Within these textual interactions, team members continually issue requests, negotiate commitments, shift responsibilities, and modify project milestones. However, because commitments and requests are dispersed across fragmented conversational threads, critical action items frequently slip past deadlines, remain unassigned, or are silently invalidated by subsequent correspondence. Existing natural language processing (NLP) approaches typically frame task management as either isolated sentence-level classification, standard Named Entity Recognition (NER), or open-ended generative summarization via Large Language Models (LLMs). These approaches suffer from operational shortcomings: isolated classifiers fail to track conversational state evolution across multi-turn exchanges, whereas unconstrained generative models are prone to hallucinating obligations, inventing non-existent assignees, or misrepresenting literal temporal deadlines.

To address these challenges, this paper presents **FollowThrough**, an evidence-backed conversational task extraction and dynamic state-tracking system. FollowThrough integrates four complementary processing stages: multi-label speech act signal classification, conservative rule-based linguistic extraction with character-level evidence grounding, dense bi-encoder semantic follow-up linking, and a human-in-the-loop audit ledger backed by SQLite with write-ahead logging (WAL). The pipeline is empirically evaluated using a curated real-world corpus of 100,000 raw emails from the Carnegie Mellon University Enron dataset, yielding 97,988 retained records, 42,317 distinct conversation threads, 1,442,694 cleaned sentences, and 29,556,736 tokens. Using a disjoint-set graph partitioning algorithm based on connected components, the dataset is segmented into thread-isolated training (78,385 messages), validation (9,946 messages), and test (9,657 messages) sets, designed to prevent data leakage across conversation threads and duplicate message groups.

We systematically compare a classical Term Frequency-Inverse Document Frequency (TF-IDF) linear baseline with balanced class weighting against a fine-tuned DistilBERT transformer trained with multi-label binary cross-entropy loss (`BCEWithLogitsLoss`) for 5,000 optimization steps. We also introduce a 128-token sliding-window inference architecture with a 64-token stride to process arbitrarily long messages without token truncation. For follow-up resolution, the system uses pretrained bi-encoder embeddings (`sentence-transformers/all-MiniLM-L6-v2`) to rank open tasks via cosine similarity alongside deterministic rule patterns that detect deadline revisions, reassignments, and task completion. Crucially, FollowThrough enforces a non-gating, human-in-the-loop paradigm wherein candidate extractions and semantic links serve as advisory guidance, requiring explicit human authorization before state mutations are committed to an immutable append-only database. Experimental evaluation on the held-out test split ($N=9,657$) demonstrates that the TF-IDF baseline achieves an exact match accuracy of 0.8801 and a macro-F1 of 0.6874 against programmatic weak labels, outperforming standard fine-tuned DistilBERT (exact match 0.7145, macro-F1 0.3656) due to weak-supervision surface lexical alignment and severe commitment class imbalance (4.08% training prevalence vs. 3.76% test prevalence). A combined intervention of class-weighted loss ($20.0\times$ commitment weight) and batch oversampling recovers test commitment recall to 0.7879, albeit with a low precision of 0.0955 (2,709 false positives), demonstrating a severe recall–precision trade-off on programmatic pseudo-labels.

---

## 1. Introduction

Asynchronous, text-based communication represents the operational backbone of modern knowledge work. In enterprise organizations, platforms such as email clients (e.g., Microsoft Outlook, Gmail) and messaging ecosystems (e.g., Slack, Microsoft Teams) host thousands of daily micro-coordinations. Within these exchanges, collaborators routinely establish two fundamental speech acts that govern workplace momentum: **requests** (solicitations for action, information, or review directed toward others) and **commitments** (explicit undertakings, pledges, or promises made by the author).

Despite their ubiquity, conversational obligations are inherently fluid and difficult to manage. Unlike structured project management systems (e.g., Jira, Asana) where tasks are defined as formal schema-constrained database entities, conversational tasks emerge spontaneously within unstructured natural language. Over the lifecycle of a typical email thread, initial commitments are frequently renegotiated: deadlines slip from "tomorrow morning" to "next Friday," tasks are delegated to third-party colleagues, project scopes expand or contract, and prior requests are superseded or rendered obsolete by subsequent messages. In corporate environments characterized by high communication volume and distributed teams, individuals experience cognitive overload. Consequently, actionable obligations become buried beneath conversational noise, leading to forgotten deliverables, missed project milestones, duplicate effort, and operational friction.

The broader research area of natural language understanding (NLU) has long investigated computational techniques to extract structure from unstructured text. Historically, researchers have applied text classification, intent recognition, information extraction, and semantic parsing to automate task detection. In recent years, the rapid advancement of pretrained transformers and Large Language Models (LLMs) has prompted widespread interest in end-to-end task automation. However, applying NLU to operational task tracking exposes three primary technical challenges:

1. **The Static Context Fallacy:** Traditional machine learning approaches treat action-item detection as an isolated, single-utterance classification or sequence-tagging task. By processing individual sentences in isolation, these models cannot maintain contextual continuity across multi-turn threads, remaining blind to subsequent conversational turns that revise, reassign, or cancel previously agreed-upon tasks.
2. **Generative Hallucination and Unverifiable Outputs:** While generative LLMs can summarize conversations and identify action items, their unconstrained autoregressive nature frequently induces hallucinations. In enterprise settings, an LLM might infer an unstated deadline, fabricate an assignee based on associative bias, or hallucinate a task that was never agreed upon. Furthermore, generative outputs do not provide native character-level provenance linking the extracted entity back to the exact source span in the original communication.
3. **The Premature Automation Dilemma:** Autonomous agents that programmatically write tasks directly into enterprise management software without human oversight introduce organizational risk. False positives pollute issue trackers with phantom tickets, while false negatives cause missed obligations. Without verifiable, evidence-backed human oversight, users quickly lose confidence in automated systems.

The primary objective of the **FollowThrough** project is to design, implement, and evaluate an evidence-backed conversational task extraction and state-tracking system that reconciles natural language processing with operational verifiability. Specifically, this research addresses four primary research questions:
- *RQ1:* How effectively can multi-label sequence classification models (both classical linear baselines and fine-tuned transformer architectures) detect request and commitment speech acts within large-scale, noisy workplace email corpora?
- *RQ2:* Can conservative linguistic extraction combined with exact character-level evidence tracking mitigate generative hallucinations while capturing actionable task descriptions, explicit assignees, and literal deadlines?
- *RQ3:* How well can dense bi-encoder semantic sentence representations (`all-MiniLM-L6-v2`) link subsequent conversational follow-up messages to open parent tasks across multi-turn threads?
- *RQ4:* How can a human-in-the-loop user interface and an immutable, append-only persistence layer be structured to guarantee auditability and operational integrity?

---

## 2. Problem Statement

To formally define the operational challenge addressed by FollowThrough, consider an asynchronous conversation thread $T = \{m_1, m_2, \dots, m_K\}$ comprising an ordered sequence of $K$ messages exchanged among a set of participants $P = \{p_1, p_2, \dots, p_J\}$. Each message $m_k$ consists of metadata (sender $s_k \in P$, recipients $R_k \subseteq P$, timestamp $\tau_k$) and an unstructured body string $C_k$. Within any message $m_k$, an author may articulate one or more forward-looking task instances $\tau = (a, o, d, \sigma)$, where $a$ denotes the action description, $o \in P \cup \{\text{Unspecified}\}$ denotes the assigned owner, $d$ denotes the temporal deadline or fulfillment target, and $\sigma \in \{\text{Pending}, \text{Completed}, \text{Cancelled}\}$ denotes the operational status.

The core problem is to continuously detect, extract, and update the set of active tasks $\mathcal{T}$ across the evolution of thread $T$, while satisfying three fundamental constraints:
1. **Linguistic Verifiability:** For every extracted task component $(a, o, d)$, there must exist an exact character span $[c_{\text{start}}, c_{\text{end}}]$ within message body $C_k$ such that the extracted information is directly grounded in the text:
   $$\text{Substring}(C_k, c_{\text{start}}, c_{\text{end}}) \equiv \text{Evidence}$$
   The extraction mechanism must not generate, interpolate, or extrapolate entities that lack explicit textual grounding.
2. **Conversational Dynamic Alignment:** When a subsequent message $m_{k+j}$ ($j \ge 1$) arrives containing an update (such as "I've completed the slides" or "We need to push the review to Friday"), the system must correctly map $m_{k+j}$ to its corresponding antecedent task $\tau \in \mathcal{T}$ and propose the appropriate state transition $\sigma \to \sigma'$ or parameter revision $d \to d'$, rather than creating a redundant, disconnected task.
3. **Human Authorization and Auditability:** Because conversational language is inherently ambiguous, sarcastic, or tentative, autonomous systems cannot be permitted to make irrevocable modifications to organizational task ledgers. Every state change must pass through an interactive verification gate and be recorded in an immutable ledger $\mathcal{L}$ that logs the operator, timestamp, previous state, new state, and supporting contextual evidence.

---

## 3. Project Architecture

The architecture of FollowThrough is engineered around the principle of **modular separation of concerns with human-in-the-loop verification**. Rather than deploying a monolithic end-to-end model, the system divides the problem into three decoupled subsystems: the Data Ingestion and Lineage Pipeline, the Multi-Stage NLP Inference Engine, and the Persistent State Ledger.

1. **Ingestion and Non-Gating Advisory Classification:** When an unstructured email or chat message enters the system via Tab 1 of the Streamlit interface, it is simultaneously routed to the Sequence Classification Subsystem and the Linguistic Extraction Subsystem. The classifiers do not serve as a hard filtering gate: they provide advisory likelihood scores. If an input message exceeds 128 tokens, an overlapping sliding-window inference engine evaluates 126-token chunks with a 64-token stride.
2. **Deterministic Evidence Extraction and Temporal Grounding:** Concurrently, the `TaskExtractor` evaluates sentences against conservative syntactic templates anchored by modal verb phrases (*please review*, *I will deliver*). Exact character coordinates $[c_{\text{start}}, c_{\text{end}}]$ are preserved. If a temporal expression is present, the `TemporalNormalizer` anchors it to the RFC 822 `Date` header to resolve a standardized ISO-8601 calendar date.
3. **Interactive Human Review Gate:** The candidate tasks and evidence offsets are surfaced in an editable Streamlit form. Only explicit human confirmation commits tasks to the database.
4. **Cryptographic Storage and Audit Ledger:** Confirmed records are fingerprinted via $h = \text{SHA-256}(C_{\text{source}} \parallel a)$ to reject exact duplicates. The `task_history` table records all subsequent updates with full provenance.
5. **Dense Semantic Follow-Up Linking:** Follow-up messages are compared against open tasks via cosine similarity using `all-MiniLM-L6-v2` dense embeddings, while rule patterns propose deadline revisions, reassignments, or completions.

---

## 4. Tools and Technologies

- **Python Language (v3.10+):** Core runtime environment providing MIME parsing (`email.parser`), regular expressions (`re`), and embedded database support (`sqlite3`).
- **PyTorch (v2.0+):** Primary computational graph framework for fine-tuning DistilBERT and executing sliding-window tensor operations with Apple Metal Performance Shaders (MPS) acceleration.
- **Hugging Face Transformers (v4.30+):** Supplies `DistilBertForSequenceClassification`, tokenizer offset mappings, and AdamW optimization schedules.
- **Sentence-Transformers (`all-MiniLM-L6-v2`):** Pretrained 384-dimensional dense bi-encoder mapping follow-up queries and candidate tasks into metric space for real-time cosine retrieval.
- **Scikit-Learn (v1.3+):** Implements sublinear `TfidfVectorizer`, multi-label `LogisticRegression` baseline, and evaluation metrics.
- **SQLite3 (Native WAL Mode):** Embedded relational database engine; zero-configuration deployment with ACID guarantees and high-concurrency WAL journaling.
- **Streamlit (v1.25+):** Rapid reactive web application framework powering the 3-tab user interface.
- **Hardware:** Apple Silicon M-series (8-core CPU, MPS GPU, 16.0 GB Unified Memory, 512 GB PCIe NVMe SSD).

---

## 5. Methodology

### 5.1 Corpus Lineage and Class Prevalence
The corpus is derived from the public Carnegie Mellon University Enron Email Dataset (May 7, 2015 archive release, `enron_mail_20150507.tar.gz`). Mailbox prioritization selects 100,000 files prioritizing core organizational participants (`kaminski-v`, `dasovich-j`, `farmer-d`, `shankman-j`). After RFC 822 parsing, quote stripping, signature removal, and fragment exclusion (<10 characters), **97,988 high-quality records** are retained (1,442,694 sentences, 29,556,736 tokens).

Crucially, commitment speech acts exhibit severe class imbalance across the partitioned corpus:
- **Training Split ($N=78,385$):** 35,141 requests (44.83%) and 3,201 commitments (**4.08% prevalence**).
- **Validation Split ($N=9,946$):** 4,549 requests (45.74%) and 392 commitments (**3.94% prevalence**).
- **Test Split ($N=9,657$):** 4,270 requests (44.22%) and 363 commitments (**3.76% prevalence**).

Distinguishing training prevalence (4.08%) from test prevalence (3.76%) is critical when assessing class weighting ratios and baseline expectations.

### 5.2 Thread-Disjoint Partitioning
To prevent data leakage, a Disjoint-Set (Union-Find) algorithm groups messages sharing normalized subjects and duplicate body hashes ($K_{\text{dup}} = \text{SHA-256}(C_{\text{body}}) \parallel \text{Sender} \parallel \text{Subject}$) into connected components. Deterministic hashing (`seed=42`) allocates complete clusters into 78,385 train, 9,946 validation, and 9,657 test messages, designed to eliminate thread and duplicate leakage across splits.

### 5.3 Sequence Classification Modeling
- **Model A (TF-IDF Baseline):** 10,000 sublinear word/character n-grams (1–2), balanced class weights, L-BFGS logistic regression. Decision thresholds: Request = 0.45, Commitment = 0.65.
- **Model B (Fine-Tuned DistilBERT):** `distilbert-base-uncased` (66M params), multi-label binary cross-entropy loss (`BCEWithLogitsLoss`), AdamW optimizer (lr = $2 \times 10^{-5}$, weight decay = 0.01), batch size = 2, 5,000 steps. Default decision thresholds: Request = 0.50, Commitment = 0.50; validation-tuned thresholds: Request = 0.35, Commitment = 0.20.

---

## 6. Results

### 6.1 Weak-Label Test Split Evaluation ($N=9,657$)

| Evaluation Metric | Model A: TF-IDF Baseline | Model B: DistilBERT (5k steps) | Performance Differential ($\Delta$) |
|---|---|---|---|
| **Exact Match Accuracy** | **0.8801** | 0.7145 | $-0.1656$ (TF-IDF +16.56%) |
| **Hamming Loss** | **0.0630** | 0.1479 | $+0.0849$ (TF-IDF $-57.4\%$ error) |
| **Macro-Averaged F1** | **0.6874** | 0.3656 | $-0.3218$ (TF-IDF +32.18%) |
| **Micro-Averaged F1** | **0.8708** | 0.7027 | $-0.1681$ (TF-IDF +16.81%) |
| **Request Precision** | **0.8996** | 0.6801 | $-0.2195$ |
| **Request Recall** | **0.9169** | 0.7906 | $-0.1263$ |
| **Request F1-Score** | **0.9081** | 0.7312 | $-0.1769$ |
| **Commitment Precision** | **0.4286** | 0.0000 | $-0.4286$ |
| **Commitment Recall** | **0.5124** | 0.0000 | $-0.5124$ |
| **Commitment F1-Score** | **0.4668** | 0.0000 | $-0.4668$ |
| **Test Inference Time** | **0.02 s** | 57.57 s | $2,878\times$ faster (TF-IDF) |
| **Persistent Model Size** | **0.52 MB** | 255.4 MB | $491\times$ smaller (TF-IDF) |

#### Confusion Matrices ($N=9,657$ Test Set):
- **Request Detection (Support = 4,270):**  
  - TF-IDF: TP = 3,915, FP = 437, FN = 355, TN = 4,950  
  - DistilBERT: TP = 3,376, FP = 1,588, FN = 894, TN = 3,799  
- **Commitment Detection (Support = 363):**  
  - TF-IDF: TP = 186, FP = 248, FN = 177, TN = 9,046  
  - DistilBERT: TP = 0, FP = 11, FN = 363, TN = 9,283  

### 6.2 Evaluation of Combined Reweighting & Threshold Tuning Experiments

To investigate minority-class collapse, we conducted experiments combining class-weighted loss and batch oversampling in `scripts/train_weighted_distilbert.py`, followed by threshold sweeping in `scripts/tune_thresholds_experiment.py`. Specifically, we configured a commitment positive weight of $\text{pos\_weight} = 20.0$ (and $1.2$ for requests) within $\text{BCEWithLogitsLoss}$, combined with batch oversampling (4 commitment examples per batch of 16).

| Experimental Run | Configuration Details | Request F1 | Commitment Recall | Commitment Precision | Commitment F1 | Macro-F1 | Exact Match |
|---|---|---|---|---|---|---|---|
| **Run 1: Default DistilBERT** | Standard BCE, $\theta = (0.50, 0.50)$ | 0.7312 | 0.0000 | 0.0000 | 0.0000 | 0.3656 | 0.7145 |
| **Run 2: Threshold-Only** | Standard BCE, $\theta = (0.35, 0.20)$ | 0.7312 | 0.0000 | 0.0000 | 0.0000 | 0.3656 | 0.7145 |
| **Run 3: Combined Intervention** | Weighted Loss ($20.0\times$) + Oversampling, $\theta = (0.50, 0.50)$ | 0.7374 | **1.0000** (363/363) | 0.0495 | 0.0943 | 0.4158 | 0.2574 |
| **Run 4: Combined + Tuned $\theta$** | Weighted + Oversampling, $\theta_{\text{req}} = 0.40, \theta_{\text{com}} = 0.85$ | **0.7603** | **0.7879** (286/363) | **0.0955** (2,709 FP) | **0.1703** | **0.4653** | **0.5495** |

### 6.3 Analysis of Experimental Progression
1. **Why Threshold Tuning Alone Failed (Run 2):** On the unweighted model, all predicted commitment probabilities on the test set were near zero. Sweeping the threshold on validation PR curves down to $\theta_{\text{com}} = 0.20$ yielded only 11 positive predictions on test, all 11 of which were false positives (0 TP, 11 FP, 363 FN). Consequently, precision, recall, and F1 remained at 0.0000. Threshold adjustment alone without loss or sampling intervention was fundamentally unable to overcome minority-class suppression.
2. **Combined Intervention Recovery and Trade-Offs (Runs 3 and 4):** Applying class weighting ($20.0\times$) and batch oversampling jointly forced the network to predict positive commitments, identifying all 363 test commitments at $\theta=0.50$ (100% recall, 6,975 FP). When thresholds were subsequently tuned on validation PR curves ($\theta_{\text{req}}=0.40, \theta_{\text{com}}=0.85$), Run 4 achieved a commitment recall of **78.79%** (286 TP out of 363), raising Request F1 to 0.7603 and Macro-F1 to 0.4653.
3. **Realistic Assessment of Precision:** Crucially, Run 4 produced 2,709 false positives on the test set against programmatic weak labels, yielding a commitment precision of **9.55%**. In practice, this corresponds to approximately one correct prediction for every ten commitment alerts. While this demonstrates that class reweighting can successfully recover recall on an imbalanced corpus, the severe recall–precision trade-off confirms that the classifier is not standalone production-grade. TF-IDF remains the superior and more balanced model on the weak-label benchmark.

---

## 7. Discussion

- **Non-Gating Advisory Paradigm:** Decoupling classification from extraction prevents downstream error propagation. Because classification scores are treated as advisory indicators rather than hard filters, low classifier precision does not invalidate downstream candidate extraction.
- **Linguistic Extraction with Provenance:** Deterministic linguistic extraction anchored to deontic modals preserves exact character offsets ($[c_{\text{start}}, c_{\text{end}}]$), providing verifiable provenance and mitigating generative hallucination risks.
- **Cryptographic Auditability:** Content hashing prevents duplicate tasks, and the append-only `task_history` table maintains full operational accountability.
- **Limitations:** Rule-derived pseudo-labels contain inherent noise. Conservative linguistic patterns miss colloquial undertakings ("Consider it done"). Relative dates require sender timezone context.

---

## 8. Reproducibility & Artifact Lineage

All scripts, models, and artifacts are organized within the repository structure:
- **Training Scripts:** `scripts/train_full_distilbert.py` (Run 1) and `scripts/train_weighted_distilbert.py` (Runs 3–4).
- **Threshold Tuning Script:** `scripts/tune_thresholds_experiment.py`.
- **Results Artifacts:** `evaluation/results/weighted_experiment_results.json` and `evaluation/results/distilbert_val_test_probs.npz`.
- **Model Checkpoints:** Saved under `models/distilbert_fine_tuned/` and `models/distilbert_weighted/`.

---

## 9. References

1. B. Klimt and Y. Yang, "The Enron Corpus: A New Dataset for Email Classification Research," in *Proc. 15th ECML*, Pisa, Italy, 2004, pp. 217–226.
2. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv:1910.01108*, 2019.
3. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP*, Hong Kong, 2019, pp. 3982–3992.
4. A. Ratner, C. M. De Sa, S. Wu, D. Selsam, and C. Ré, "Snorkel: Rapid Training Data Creation with Weak Supervision," *Proc. VLDB*, vol. 11, no. 3, pp. 269–282, 2017.
5. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, Minneapolis, MN, 2019, pp. 4171–4186.
6. J. R. Searle, *Speech Acts: An Essay in the Philosophy of Language*. Cambridge, UK: Cambridge Univ. Press, 1969.
7. J. L. Austin, *How to Do Things with Words*. Oxford, UK: Oxford Univ. Press, 1962.
8. V. R. Carvalho and W. W. Cohen, "On the collective classification of email speech acts," in *Proc. 28th ACM SIGIR*, Salvador, Brazil, 2005, pp. 345–352.
9. P. S. Turney, "Thumbs up or thumbs down? Semantic orientation applied to unsupervised classification of reviews," in *Proc. 40th ACL*, Philadelphia, PA, 2002, pp. 417–424.
10. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
11. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *NeurIPS 32*, Vancouver, Canada, 2019, pp. 8024–8035.
12. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *ICLR*, New Orleans, LA, 2019.
13. T. Mikolov, K. Chen, G. S. Corrado, and J. Dean, "Efficient Estimation of Word Representations in Vector Space," in *Workshop at ICLR*, Scottsdale, AZ, 2013.
14. D. Corney, D. R. Stent, and H. Di Fabbrizio, "What are you going to do? Detecting task-oriented emails," in *Proc. 37th ACM SIGIR*, Gold Coast, Australia, 2014, pp. 1007–1010.
15. *RFC 822: Standard for the Format of ARPA Internet Text Messages*, Internet Engineering Task Force (IETF), Aug. 1982.
