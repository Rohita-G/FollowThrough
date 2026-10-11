# Annotation Pilot Screening (20261011)

This is a bounded screening sample, not 40 guaranteed task–message examples. It contains 40 message candidates from 15 threads. The fixed seed is `20261011`. Thread selection strata: `{"subject_only:11+": 3, "subject_only:2": 2, "subject_only:3-5": 4, "subject_only:6-10": 3, "unlinked:1": 3}`. Message strata: `{"cue_negative": 19, "cue_positive": 21}`.

The sample contains 21 cue-positive and 19 cue-negative messages. It includes 12 request/commitment cue rows, 8 deadline-change cue rows, and 5 completion cue rows (cue counts can overlap). It contains **zero** cancellation cues, reassignment cues, or raw-only cleaning-disagreement rows; these were not fabricated or forced into the sample. Thread link categories available in the source were `{"subject_only": 22087, "unlinked": 10183}`; this pilot selected `{"subject_only": 12, "unlinked": 3}`. Header-linked/mixed categories are absent and cannot be sampled. Unmet planned strata: `[]`.

## Review workflow

1. Open `pilot_screening.20261011.csv`. `candidate_text` is upstream `cleaned_body`; `original_candidate_text` is the stored first-pass body. Compare them because production body line breaks were flattened before cleaning. The cleaner can leave reply headers/signatures/disclaimers and can erase cue-bearing quoted/forwarded content.
2. Use `thread_id` to inspect every row of that thread in `data/processed/enron_subset_threads/messages.jsonl` and its thread summary in `data/processed/enron_subset_threads/threads.jsonl`. `previous_message_id` is neighboring context only, not a task reference.
3. Start with `pilot_screening_drafts.20261011.jsonl`. These `screening_draft` rows intentionally have blank task IDs/text/source IDs and all labels/evidence blank. Do not fill a task link until a human identifies the task from the full thread. Convert the reviewed row to `task_update` or `task_creation` per `docs/annotation_schema_v1.md`.
4. Validate completed annotations with:

```sh
python scripts/validate_annotations.py data/annotations/pilot_annotations.completed.jsonl \
  --messages data/processed/enron_subset_threads/messages.jsonl \
  --threads data/processed/enron_subset_threads/threads.jsonl
```

Cue flags are screening hints, not labels. The sample is not a random population estimate and is not suitable by itself for model training/evaluation.
