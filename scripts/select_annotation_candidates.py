#!/usr/bin/env python3
"""Select human annotation candidates from reconstructed thread messages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# High-recall cues only. These indicate text worth human review, not a gold label.
CUE_FAMILY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "request_or_commitment": (
        r"\bplease\b", r"\bcan you\b", r"\bcould you\b", r"\bwould you\b",
        r"\bwill\s+(?:send|do|take|review|prepare|provide|handle)\b",
        r"\bi['’]ll\s+(?:send|do|take|review|prepare|provide|handle|follow)\b",
        r"\bi will\s+(?:send|do|take|review|prepare|provide|handle|follow)\b",
        r"\bwe will\s+(?:send|do|take|review|prepare|provide|handle|follow)\b",
    ),
    "deadline_change": (
        r"\bdeadline\b", r"\bdue\s+(?:date|by|on)\b", r"\bby\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:extend|postpone|delay|move|push|reschedule)\w*\b.{0,80}\b(?:deadline|due|date|until|to)\b",
        r"\bnew\s+(?:deadline|due date|date)\b",
    ),
    "cancellation": (
        r"\bcancel(?:led|ed|ing|lation)?\b", r"\bcall\s+off\b", r"\bno longer need\b",
        r"\bnot going ahead\b", r"\bwithdraw\w*\b",
    ),
    "completion": (
        r"\bcompleted\b", r"\bfinished\b", r"\btask is done\b", r"\ball done\b",
        r"\bi have (?:sent|submitted|delivered|completed|finished)\b", r"\bhas been completed\b",
    ),
    "reassignment": (
        r"\breassign\w*\b", r"\bassign\w*\b.{0,80}\bto\b", r"\bhand\w*\b.{0,50}\bover to\b",
        r"\btake over\b", r"\binstead of me\b", r"\btransfer\w*\b.{0,80}\bto\b",
    ),
}
CUE_REGEXES = {
    family: tuple(re.compile(pattern, re.I) for pattern in patterns)
    for family, patterns in CUE_FAMILY_PATTERNS.items()
}
ANNOTATOR_COLUMNS = (
    "annotator_actionable",
    "annotator_existing_task_reference",
    "annotator_update_type",
    "annotator_owner_text",
    "annotator_deadline_text",
    "annotator_evidence",
    "annotator_ambiguous",
    "annotator_notes",
)
CSV_COLUMNS = (
    "candidate_id",
    "thread_id",
    "message_id",
    "position",
    "previous_message_id",
    "candidate_text",
    "previous_message_text",
    "heuristic_cues",
    "link_rules",
    "needs_thread_review",
    *ANNOTATOR_COLUMNS,
)


def cue_families(text: str) -> List[str]:
    return [
        family for family, regexes in CUE_REGEXES.items()
        if any(pattern.search(text or "") for pattern in regexes)
    ]


def _read_messages(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete JSONL line in {path} at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            yield record


def select_candidates(thread_output: Path, csv_output: Path, max_candidates: int = 5000) -> Dict[str, Any]:
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    messages_path = thread_output / "messages.jsonl"
    threads_path = thread_output / "threads.jsonl"
    if not messages_path.is_file() or not threads_path.is_file():
        raise FileNotFoundError(f"Expected messages.jsonl and threads.jsonl under {thread_output}")

    with tempfile.TemporaryDirectory(prefix="followthrough_candidates_") as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "candidates.sqlite3")
        connection.execute("""
            CREATE TABLE messages (
                thread_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                message_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                cleaned_body TEXT NOT NULL,
                link_rules TEXT NOT NULL,
                PRIMARY KEY (thread_id, position)
            )
        """)
        count = 0
        for record in _read_messages(messages_path):
            thread_id = record.get("thread_id")
            position = record.get("position")
            source_path = record.get("source_relative_path")
            cleaned_body = record.get("cleaned_body")
            link_rules = record.get("link_rules")
            if not isinstance(thread_id, str) or not thread_id:
                raise ValueError(f"Message record {count + 1} has invalid thread_id")
            if not isinstance(position, int) or isinstance(position, bool) or position < 1:
                raise ValueError(f"Message record {count + 1} has invalid position")
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(f"Message record {count + 1} has invalid source_relative_path")
            if not isinstance(cleaned_body, str):
                raise ValueError(f"Message record {count + 1} has invalid cleaned_body")
            if not isinstance(link_rules, list) or any(not isinstance(rule, str) for rule in link_rules):
                raise ValueError(f"Message record {count + 1} has invalid link_rules")
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, position, str(record.get("message_id") or ""), source_path,
                 cleaned_body, json.dumps(link_rules)),
            )
            count += 1
        connection.commit()
        connection.execute("CREATE INDEX messages_thread_idx ON messages(thread_id, position)")
        connection.execute("""
            CREATE TEMP TABLE thread_sizes AS
            SELECT thread_id, COUNT(*) AS message_count FROM messages GROUP BY thread_id
        """)

        query = connection.execute("""
            SELECT current.thread_id, current.position, current.message_id,
                   current.source_path, current.cleaned_body, current.link_rules,
                   previous.message_id, previous.cleaned_body, sizes.message_count
            FROM messages AS current
            JOIN thread_sizes AS sizes ON sizes.thread_id = current.thread_id
            LEFT JOIN messages AS previous
                ON previous.thread_id = current.thread_id
               AND previous.position = current.position - 1
            ORDER BY sizes.message_count DESC, current.thread_id, current.position
        """)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = csv_output.with_name(f".{csv_output.name}.tmp")
        selected = 0
        cue_counts: Counter[str] = Counter()
        multi_message_candidates = 0
        try:
            with temp_output.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for row in query:
                    (thread_id, position, message_id, source_path, candidate_text,
                     raw_link_rules, previous_message_id, previous_message_text, thread_size) = row
                    families = cue_families(candidate_text)
                    if not families or selected >= max_candidates:
                        continue
                    cue_counts.update(families)
                    if thread_size > 1:
                        multi_message_candidates += 1
                    identity = f"{thread_id}\n{source_path}\n{position}\n{'|'.join(families)}"
                    candidate_id = "cand-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                    writer.writerow({
                        "candidate_id": candidate_id,
                        "thread_id": thread_id,
                        "message_id": message_id,
                        "position": position,
                        "previous_message_id": previous_message_id or "",
                        "candidate_text": candidate_text,
                        "previous_message_text": previous_message_text or "",
                        "heuristic_cues": ";".join(families),
                        "link_rules": ";".join(json.loads(raw_link_rules)),
                        "needs_thread_review": "true" if thread_size > 1 else "false",
                        **{column: "" for column in ANNOTATOR_COLUMNS},
                    })
                    selected += 1
                target.flush()
                os.fsync(target.fileno())
            temp_output.replace(csv_output)
        finally:
            temp_output.unlink(missing_ok=True)
            connection.close()

    stats = {
        "input_messages": count,
        "candidates_written": selected,
        "multi_message_thread_candidates": multi_message_candidates,
        "candidate_counts_by_heuristic_family": dict(sorted(cue_counts.items())),
        "maximum_candidates": max_candidates,
        "selection_order": "thread size descending, then thread_id and position ascending; deterministic truncation at the configured maximum; no random sampling",
        "text_source": "cleaned_body only; quote-removal limitations inherit from the upstream cleaner",
        "previous_message_id_meaning": "conversational predecessor only; not an annotation of an existing task reference",
        "full_thread_review": "thread_id can be used to retrieve the complete thread; needs_thread_review is true when the thread has multiple messages",
        "labels_created": False,
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_output_dir", help="Directory containing messages.jsonl and threads.jsonl")
    parser.add_argument("--output", required=True, help="Explicit CSV output path")
    parser.add_argument("--max-candidates", type=int, default=5000, help="Deterministic output row bound")
    args = parser.parse_args()
    try:
        select_candidates(Path(args.thread_output_dir), Path(args.output), args.max_candidates)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())