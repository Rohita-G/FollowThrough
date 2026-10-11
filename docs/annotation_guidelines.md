# FollowThrough Annotation Guidelines

The versioned JSONL contract is [annotation_schema_v1.md](annotation_schema_v1.md). Keep the three data stages distinct:

1. **Screening:** one message per candidate CSV row, selected by transparent phrase cues. It is a review queue, not a label or training example. `previous_message_id` is only the preceding conversational message, never an existing-task reference. Use `thread_id` to retrieve the full thread.
2. **Annotation:** one JSONL record per task–message pair. A task update message may create multiple records when it updates multiple tasks. A genuinely new task is a `task_creation` example, not an update to an existing task.
3. **Training:** only validated `complete` annotations with `example_origin: real` are eligible. Drafts and all synthetic worked examples must be excluded.

## Action Detection and Task Updates

Action detection asks whether a message requests or commits to work. Task-update labeling asks whether a later message changes a particular existing task. They are separate questions: a request can introduce a new task, update an existing task, or be unrelated to the task in a particular pair.

Distinguish a request (“Could you send the draft?”) from a commitment (“I will send the draft”). Treat `request` and `commitment` as compatible multi-label acts where warranted. Do not label from cue words alone.

For each task–message pair, use any supported relation labels: `deadline_change`, `reassignment`, `cancellation`, `completion`, `unrelated`, and/or `ambiguous`. Deadline change, reassignment, cancellation, and completion can co-occur. `unrelated` stands alone for a pair that does not concern the task. `ambiguous` is an uncertainty marker; it can accompany a partly supported relation when the unresolved part is documented.

Proposals and suggestions are not commitments unless the writer accepts or promises the work. Conditional promises (“I can do it if…”) remain conditional; record the commitment only when the condition is satisfied or the promise is otherwise explicit. Negation (“I will not send it”) is not a commitment; distinguish a denial from cancellation of work already underway. Hypothetical or future-perfect wording (“once the review is complete”, “it would be finished”) is not evidence that work has actually completed.

## Context and Existing Tasks

Review all messages reachable by `thread_id`, not just the CSV's neighboring message. Link an update only when task identity is supported by a repeated deliverable, explicit reference, or clear continuation. The immediately previous message is context, not proof of task identity. If the message may refer to two existing tasks, record the candidate pair as ambiguous rather than choosing silently. If the message introduces new work, represent it as `task_creation`; do not fabricate an existing task link.

Quoted text was removed by upstream regex cleaning, which can remove live text or miss unusual quote formats. Consult the retained original `body` and full thread when allowed. If relevant context is missing from the subset or a reference points outside it, use `ambiguous`/insufficient-context notes rather than infer a link.

## Owners, Deadlines, and Evidence

Record owner and deadline expressions literally and only when supported by message/thread evidence. Do not infer an owner from sender/recipient alone or a due date from normal workflow. Preserve relative phrases such as “Friday” and “next week”; put any supported normalized calendar date in the separate optional normalized field. Normalize only when the timestamp, timezone, and context suffice. Evidence spans use zero-based, half-open Python string indices (Unicode code points) into the exact `task_text` or `update_text` stored in that annotation record.

For disagreement, keep independent annotator records, cite the competing evidence in notes, mark ambiguity where appropriate, and adjudicate in a separate reviewed record. Do not overwrite one annotator's judgment with another's. A pilot should include cue-positive and cue-negative messages, multiple thread sizes, and several link-confidence types; have two annotators label a small batch independently, review disagreements, refine the guide, then repeat a pilot before scaling up.

## Sampling and Limitations

Heuristic-positive-only sampling creates selection bias: it overrepresents familiar phrase patterns and misses indirect requests, unusual language, and cue-negative actions. A later annotation sample should include cue-positive and randomly sampled cue-negative messages; single- and multi-message threads; ambiguous references; and header-linked, mixed header/subject, subject-only, and unlinked/isolated messages. Stratify across mailbox and thread-size bands where possible, record sampling probabilities/strata, and reserve a separately sampled evaluation set. This is a documented future strategy only; no production corpus was accessed for it.

Enron is historical corporate email. Results here do not establish performance on modern student chats. Mailbox selection can omit related messages, and thread reconstruction can create false positives or incomplete threads. Use only necessary personal information; avoid copying unrelated names, addresses, phone numbers, account details, or private text into reports/screenshots. Follow course and institutional privacy requirements.