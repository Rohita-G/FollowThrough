# Annotation Schema v1.0

## Separate Data Stages

1. **Screening candidate:** one message selected by heuristic cues. The candidate CSV is only a review queue: labels are blank, and it is not a training example.
2. **Screening draft:** optional JSONL scaffold using `example_kind: "screening_draft"` and `annotation_status: "draft"`. It references a message/thread, but task ID/text/source IDs, labels, and evidence remain blank. It is not a task–message pair; after human review it must be replaced by a `task_update` or `task_creation` record.
3. **Annotated task–message example:** one existing task paired with one later message. One message may produce several examples when it updates several tasks.
4. **Training example:** only a validated, completed, real-origin annotation is eligible for a later conversion step. This repository does not currently convert screening CSV rows, train a model, or split data.

The annotated JSONL uses `example_kind: "task_update"` for the pair classifier. A genuinely new task is recorded separately as `example_kind: "task_creation"`, where `task_source_message_ids` contains the creation message itself and `task_text` is the literal task phrase. A creation example is not an update to a pre-existing task and must not be fed to the task–message update classifier. A `screening_draft` is only a scaffold and must never be treated as a pair or training record.

## Fields

Each JSONL line is one object. All fields below are required; unknown extra fields are permitted for forward-compatible provenance.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string, exactly `1.0` | Contract version. |
| `example_id` | non-empty string | Unique annotation-example identifier. |
| `example_kind` | `screening_draft`, `task_update`, or `task_creation` | Screening scaffold versus existing-task pair versus creation. |
| `example_origin` | `real` or `synthetic` | Keeps synthetic demonstrations out of real annotation counts. |
| `annotation_status` | `draft` or `complete` | Drafts are wholly unlabeled; complete rows carry human labels. |
| `thread_id` | non-empty string | Stable reference for full-thread review. |
| `task_id` | string | Local task identity; repeat it across pairs for the same task. Empty only in a screening draft. |
| `task_source_message_ids` | string array | Messages establishing a task; use `message_id` or `path:<source_relative_path>` fallback. Empty only in a screening draft. |
| `task_text` | string | Minimal literal task description for a pair; empty only in a screening draft. |
| `task_owner_text` | string | Literal existing owner expression, or empty. |
| `task_deadline_text` | string | Literal existing deadline expression, or empty. |
| `task_deadline_normalized` | ISO date string or null | Optional normalization, separate from the literal expression. |
| `update_message_id` | non-empty string | The later message being evaluated; use the exact `message_id`, or `path:<source_relative_path>` when the header ID is absent. |
| `update_text` | string | Exact `cleaned_body` of the referenced message at annotation time; may be empty only for an empty-body screening draft. |
| `message_act_labels` | unique string array | Multi-label message acts: `request`, `commitment`. |
| `relation_labels` | unique string array | Multi-label pair relations: `deadline_change`, `reassignment`, `cancellation`, `completion`, `unrelated`, `ambiguous`. |
| `updated_owner_text` | string | Literal changed/new owner evidence, or empty. |
| `updated_deadline_text` | string | Literal changed/new deadline expression, or empty. |
| `updated_deadline_normalized` | ISO date string or null | Optional normalization; never replaces the literal expression. |
| `evidence_spans` | array of span objects | Exact supporting text spans (definition below). |
| `ambiguous` | boolean | Whether this pair/interpretation remains uncertain. |
| `ambiguity_reason` | string | Required when ambiguous; otherwise empty. |
| `annotator_id` | string | Required for completed rows; empty in drafts. |
| `annotation_notes` | string | Concise annotation note; avoid unnecessary personal data. |

### Label and combination rules

`message_act_labels` and `relation_labels` are separate multi-label axes. A message can request a deadline change, commit to reassigned work, or both change an owner and a deadline. Do not force these compatible labels into a softmax category. `unrelated` is exclusive with other task-pair relation labels, but can coexist with a message-level act: for example, a request about a different task can be `request` plus `unrelated` for the current pair. `ambiguous` is an uncertainty relation and can co-occur with a specifically observed relation when the relation is only partly resolved; it requires `ambiguous: true` and a reason. A new task uses `task_creation` plus a message act and has no task-update relation labels.

A message that updates two tasks yields two `task_update` JSONL rows with separate `task_id` and task provenance but the same `thread_id`, `update_message_id`, and exact `update_text`. This is intentional pair-level supervision; later training should account for the repeated message so it is not accidentally overweighted.

### Evidence offsets

Each span is `{ "text_field": "task_text"|"update_text", "start_char": integer, "end_char": integer, "supports_label": string }`. Offsets are zero-based, half-open Python string indices `[start_char, end_char)`, measured in Unicode code points (not UTF-8 bytes or grapheme clusters), against the exact corresponding field in that JSONL object. `end_char` is excluded. The selected substring is the evidence; full source messages and threads are retrieved by IDs rather than copied into the annotation record.

### Drafts and completed examples

A `screening_draft` is wholly unlabeled: task ID/text/source IDs and owner/deadline fields are empty, act/relation labels and evidence are empty arrays, `ambiguous` is false, ambiguity reason and annotator fields are empty, and normalized dates are null. It may retain an empty exact `update_text` only when the referenced cleaned body is empty. Do not create a `task_update` draft until a human identifies the task and its source messages. A `complete` task pair requires an annotator ID, at least one act or relation label, and evidence. Do not treat screening CSV rows or drafts as training data.

## Future Classifier Implications

The update model input is `(task_text, update_text)` for `task_update` examples, with thread context available by `thread_id` where needed. It should use sigmoid multi-label outputs, not a single softmax, for compatible update relations. `unrelated` can be trained as an exclusive relation outcome; `ambiguous` should be represented as an uncertainty target and may accompany a partially observed relation. Message-act outputs are a separate multi-label head. Since one update message can appear in multiple task pairs, message-level act loss should be deduplicated or weighted by the number of task pairs; pair-specific relation loss remains per example. `task_creation` examples belong to a separate new-task/action-detection target and are excluded from this pair classifier. No model or loss is implemented here.

The validator is dependency-free: `python scripts/validate_annotations.py annotations.jsonl --messages <messages.jsonl> --threads <threads.jsonl>`. It validates references against only the explicitly supplied files and never infers labels from heuristic cues.