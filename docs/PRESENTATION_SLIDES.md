# FollowThrough: 10–15 Minute Capstone Presentation Script & Slide Outline

**Topic:** Evidence-Backed Conversational Task Extraction and State Tracking Across Message Threads  
**Course:** NLP Capstone Project  
**Duration:** 10–15 minutes  

---

## Slide 1: Title & Overview (1 min)
- **Title:** FollowThrough
- **Subtitle:** Turning Conversational Threads into Evidence-Backed Tasks and Verified Calendar Events
- **Speaker:** Student NLP Project
- **Core Value Proposition:**
  - Modern work happens in email and chats, but action items get lost and deadlines shift across replies.
  - Generative AI often hallucinates owners or dates; silent calendar writes destroy trust.
  - FollowThrough provides an evidence-grounded, human-in-the-loop solution.

---

## Slide 2: Problem & Motivation (1.5 min)
- **The Operational Challenge:**
  - *Dynamic Obligation Drift:* "Sam, please draft the brief by Thursday" followed next day by "Priya will take over, deadline pushed to Friday."
  - *Static NLP Limitation:* Traditional tools extract isolated sentences and forget previous commitments.
  - *The Hallucination Danger:* Autonomous LLMs hallucinate dates, infer unstated owners, and write blindly to production databases.
- **System Design Principles:**
  - Never silently invent owners, dates, or completions.
  - Always surface exact character offsets and evidence snippets.
  - Require explicit user confirmation for any persistence or external write.

---

## Slide 3: System Architecture (1.5 min)
- **Three-Tier Architecture:**
  1. **UI Layer (Streamlit):** Three interactive workflows (Task Creation, Saved Tasks/Calendar, Follow-up Linking).
  2. **NLP Engine:**
     - *Classification:* TF-IDF and Fine-Tuned DistilBERT for Request/Commitment detection.
     - *Extraction:* Rule-based linguistic parser extracting action, explicit owner, and literal dates with offsets.
     - *Semantic Linking:* Pretrained `all-MiniLM-L6-v2` dense embeddings for matching follow-ups to open tasks.
  3. **Storage & Integrations:**
     - SQLite relational persistence (`tasks`, `task_history`).
     - Google Calendar API v3 (local desktop OAuth, idempotent event sync).

---

## Slide 4: Data & Preprocessing: Enron Corpus v2 (1.5 min)
- **Corpus Scale:**
  - 100,000 parsed emails from Enron.
  - 1,442,694 cleaned sentences | 29.56 million tokens.
- **Rigorous Data Hygiene:**
  - Excluded 2,012 uninformative records (1,370 empty bodies, 642 under 10 chars).
  - 97,988 clean records retained in canonical manifest.
- **Leakage Prevention via Graph Partitioning:**
  - Duplicate email grouping + connected thread clustering via Disjoint-Set (Union-Find).
  - 80% Train (78,385), 10% Validation (9,946), 10% Test (9,657).
  - Zero cluster overlap across splits.

---

## Slide 5: NLP Methods & Weak Supervision (1.5 min)
- **Transparent Linguistic Weak Supervision:**
  - Programmatic, interpretable heuristics for Request and Commitment signals.
  - Explicit filtering for negation ("will not"), past actions ("already sent"), permission, and quotes.
  - Avoided sending private corporate text to commercial LLM APIs.
- **Model Comparison Architectures:**
  - *Model A:* Sublinear TF-IDF (10k n-grams) + Class-balanced Logistic Regression.
  - *Model B:* Fine-Tuned DistilBERT (`distilbert-base-uncased`) with multilabel BCE loss on Apple Silicon MPS.
  - *Retrieval:* Pretrained MiniLM embeddings for semantic similarity scoring.

---

## Slide 6: Training & Evaluation Protocol (2 min)
- **Critical Evaluation Caveat:**
  - Metrics reflect *weak-label agreement*, not independently verified ground-truth accuracy.
- **Empirical Comparison on Test Split (N = 9,657):**
  - **TF-IDF Baseline:**
    - Exact Match: 88.01% | Macro-F1: 0.6874 | Request F1: 0.9081 | Commitment F1: 0.4668
    - Model Size: 8.4 MB | Training: 21.8 seconds.
  - **Fine-Tuned DistilBERT (5,000 steps):**
    - Exact Match: 73.60% | Macro-F1: 0.3893 | Request F1: 0.7787 | Commitment F1: 0.0000
    - Model Size: 255.6 MB | Training: 16.3 min (Apple Silicon MPS).
- **Key Finding:** TF-IDF provides an exceptionally competitive baseline with low latency and compute.

---

## Slide 7: Live Application Walkthrough (4 min)
*(Demonstrate the live Streamlit application step-by-step)*
1. **Paste Message:** Enter email requesting draft submission by Friday.
2. **Review Extracted Task:** View Request score (0.92), extracted action, owner, literal deadline, and character offsets.
3. **User Confirmation:** Confirm and save to SQLite.
4. **Calendar Sync:** Preview Google Calendar event card, review time/date, and perform OAuth-backed event creation.
5. **Process Follow-up:** Paste reply: "Priya will take over, please push deadline to Monday."
6. **Semantic Linking & Review:** MiniLM retrieves prior task (Cosine Sim: 0.84); system detects owner and deadline revision.
7. **User Confirms Update:** Save revision; verify change history in SQLite and updated event in Google Calendar.

---

## Slide 8: Limitations & Ethical Disclosure (1 min)
- **System Limitations:**
  - Formulaic polite noise ("Please feel free to reach out") triggers false request alerts.
  - Commitment class is sparse (~4%), making high-precision commitment extraction challenging.
  - Complex temporal resolution across varying user timezones requires manual validation.
- **AI-Assistance Disclosure:**
  - Fully transparent disclosure: AI used for script scaffolding and provisional pilot screening.
  - Weak supervision generated via deterministic local rules.
  - Human review files preserved with blank gold fields pending manual verification.

---

## Slide 9: Conclusion & Q&A (1 min)
- **Summary:**
  - Demonstrated an evidence-backed conversational task assistant on 100k real-world emails.
  - Compared classical linear models against fine-tuned transformers.
  - Enforced verifiability, preventing silent hallucinations and unwanted writes.
- **Thank you! Questions?**
