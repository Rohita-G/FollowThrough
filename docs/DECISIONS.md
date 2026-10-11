# Data Selection Decisions

## Enron Subset

The current subset is selected from the already-extracted `data/raw/maildir` tree. Mailboxes are processed sequentially in this order when present: `kaminski-v`, `dasovich-j`, `shackleton-s`, `germany-c`, `beck-s`, followed by every remaining mailbox in alphabetical order. Within each mailbox, files are traversed by sorted source-relative path. Parsing stops as soon as 100,000 successful email records have been written; failed files are counted but do not consume that success budget.

This is a deterministic mailbox-based convenience sample, not a random or thread-complete sample. A mailbox-based subset may omit related messages stored in other mailboxes, and subject-based relationships may cross mailbox boundaries. The 100,000-record cap does not guarantee any particular sentence count or complete conversation threads. Those quantities must be measured from the resulting records.

The parser writes the original first-pass email fields to `data/processed/enron_subset/by_user/<mailbox>.jsonl`, with a maildir-relative `source_relative_path`. Failures are recorded separately without email bodies. `manifest.json` describes mailbox ordering, counts, stop reason, and completion; `checkpoint.json` is advisory and is reconciled against validated success and failure ledgers on resume. The subset output is separate from prior full-corpus caches.

## Deduplication and Thread Links

The thread-building step keeps the first message for each normalized Message-ID across all input mailboxes. Messages without an ID use a SHA-256 key over normalized sender, date, normalized subject, and body. Duplicate provenance is recorded as the set of mailboxes containing copies. Header links use matching `In-Reply-To` and `References` IDs; normalized subjects can additionally join messages only when their timestamps are within 30 days. Messages are ordered by date within reconstructed threads, with undated messages placed last.

These are heuristic links, not ground truth. Message IDs may be malformed or absent, subject-only links can join unrelated conversations, and header references whose target is outside the selected subset cannot be followed. Conversely, mailbox selection may omit related messages. Cleaning truncates at recognizable reply, forward, signature, and disclaimer markers; it can remove useful text or miss unusual formatting, so the original body is retained alongside the cleaned version. No train/validation/test split is created by this step.

## Course AI-Assistance Policy

The course's assignment-specific permission for generative-AI-assisted coding and any required disclosure has not been verified. Check the syllabus or instructor guidance before submitting this work. Do not describe AI-assisted code as independently authored; disclose assistance as the course requires.