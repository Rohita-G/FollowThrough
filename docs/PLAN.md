# FollowThrough build plan (Oct 11 to Dec 3, 2026)

## One system, two modules
Shared core: a conversation state tracker. Module 1 (build first) tracks tasks.
Module 2 (stretch) adds decisions, reusing the same Item/event machinery
(PROPOSE, AGREE, REVISE) plus a question-answering layer ("what did we decide?").

## NLP tasks (course needs several)
1. Text classification: action sentence detection; message event type
   (REQUEST, COMMIT, DEADLINE_CHANGE, REASSIGN, CANCEL, COMPLETE, NONE).
2. Token classification (NER style): owner, deadline, task phrase spans.
3. Semantic similarity / linking: cross-encoder scoring (new message, open task).
4. Coreference-lite: resolve "I", "you", "he", "that" to people using speaker + turn context.
5. (Module 2) Question answering over current state, with evidence.

## Data
- Enron (CMU, about 0.5M emails): bulk text for preprocessing/cleaning, plus public
  sentence-level action/speech-act annotations for the action detector.
- AMI Meeting Corpus (CC BY 4.0): multi-party dialogue with dialogue acts, decision
  points and ACTIONS/DECISIONS in abstractive summaries. Natural fit for both modules.
- Synthetic threads (simulate.py + LLM paraphrase): large training set with known gold.
- Hand-annotated real test set: aim 150 to 200 threads from AMI/Enron, double-check
  a 20% subset yourself two days apart to report agreement.
Course needs one dataset of 50,000+ sentences "or equivalent": Enron plus AMI clearly exceed
this in size, but the labeled parts are smaller. Ask the instructor whether
corpus size or labeled size counts, early.

## Model comparison (needs 2+ variants)
- Variant 1: DistilBERT or MiniLM (fast, laptop-friendly)
- Variant 2: RoBERTa-base or DeBERTa-v3-base (GPU session)
- Systems: A message-level, B thread-aware, C thread-aware + evidence/uncertainty check.

## Metrics (needs 3+)
Action F1, owner/deadline span F1, event macro-F1, active-list F1,
stale-task rate (headline), calibration error for the uncertainty check.

## Schedule
| Week | Dates | Goal |
|---|---|---|
| 1 | Oct 11-17 | Run scaffold. Download Enron + AMI. Write annotation guideline (1 page). |
| 2 | Oct 18-24 | Clean/segment real data. Start annotating real test threads (50). |
| 3 | Oct 25-31 | Paraphrase synthetic data with an LLM; train Variant 1 action + event classifiers. |
| 4 | Nov 1-7 | GPU session 1: fine-tune Variant 2. Cross-encoder linker. |
| 5 | Nov 8-14 | Finish test set (150+). Full A/B/C comparison, error analysis. |
| 6 | Nov 15-21 | Local web UI (paste thread -> table with evidence). Demo script. Freeze code. |
| 7 | Nov 22-28 | Report in IEEE format, diagrams. DecisionLedger only if on schedule. |
| 8 | Nov 29-Dec 3 | Slides, rehearse 10-15 min demo, GitHub README, submit. |
Weekly GPU access: week 4 (fine-tune) and week 5 (final runs) are the critical sessions.

## Known risks
- Rule baseline scores 1.0 on current synthetic data: templates are too easy and the regexes
  match them. Paraphrase diversity (week 3) is what makes the benchmark meaningful.
- Synthetic-to-real gap: always report real-test results separately.
- Enron is old corporate email; do not claim it generalizes to modern chat.
