#!/usr/bin/env python3
"""Deduplicate, clean, link, and summarize an existing Enron subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Truncate at common reply/forward boundaries; remove quoted lines, signature
# separators/mobile signatures, and recognizable legal/confidentiality footers.
BODY_CUT_PATTERNS = (
    re.compile(r"(?im)^\s*[-_]{2,}\s*(?:original|forwarded) message\s*[-_]{2,}.*$"),
    re.compile(r"(?im)^\s*begin forwarded message\s*:\s*$"),
    re.compile(r"(?im)^on .{1,240}\bwrote\s*:\s*$"),
    re.compile(r"(?im)^\s*from:\s*.+\n\s*(?:sent|to|cc|subject):\s*.+$"),
    re.compile(r"(?im)^\s*--\s*$"),
    re.compile(r"(?im)^\s*sent from my (?:iphone|ipad|android|blackberry).*$"),
    re.compile(r"(?im)^\s*(?:confidentiality notice|disclaimer)\s*:?\s*$"),
    re.compile(r"(?im)^\s*(?:this (?:e-?mail|message) (?:and any attachments )?(?:is|are) confidential|if you are not the intended recipient).*$"),
)
QUOTED_LINE = re.compile(r"(?m)^\s*>.*(?:\n|$)")

# Broad high-recall candidate cues for annotation, not a gold action classifier.
ACTION_LANGUAGE_PATTERNS = (
    r"\bplease\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwill send\b",
    r"\bi['’]ll\b",
    r"\bby\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bdeadline\b",
    r"\bdue\b",
    r"\breassign\w*\b",
    r"\binstead\b",
    r"\baction item\b",
    r"\bneed to\b",
)
ACTION_LANGUAGE = re.compile("|".join(f"(?:{pattern})" for pattern in ACTION_LANGUAGE_PATTERNS), re.I)
SUBJECT_PREFIX = re.compile(r"^(?:(?:re|fw|fwd)\s*:\s*)+", re.I)
MESSAGE_ID_PATTERN = re.compile(r"<([^<>]+)>")
DAY_SECONDS = 30 * 24 * 60 * 60


def _text(record: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None:
            return " ".join(str(value).split())
    return ""


def normalize_subject(subject: str) -> str:
    value = " ".join((subject or "").split())
    while True:
        normalized = SUBJECT_PREFIX.sub("", value).strip()
        if normalized == value:
            return value.casefold()
        value = normalized


def _normalized_value(value: str) -> str:
    return " ".join(value.split()).casefold()


def _message_id_key(message_id: str) -> str:
    return _normalized_value(message_id.strip().strip("<>"))


def dedupe_key(record: Dict[str, Any]) -> str:
    message_id = _text(record, "message_id", "Message-ID")
    if message_id.strip():
        return "id:" + _message_id_key(message_id)
    fallback = (
        _normalized_value(_text(record, "from", "From")),
        _normalized_value(_text(record, "date", "Date")),
        normalize_subject(_text(record, "subject", "Subject")),
        _normalized_value(_text(record, "body", "Body")),
    )
    digest = hashlib.sha256(json.dumps(fallback, ensure_ascii=False).encode("utf-8")).hexdigest()
    return "hash:" + digest


def clean_body(body: str) -> str:
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    cut_at = len(text)
    for pattern in BODY_CUT_PATTERNS:
        match = pattern.search(text)
        if match:
            cut_at = min(cut_at, match.start())
    text = text[:cut_at]
    text = QUOTED_LINE.sub("", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _parse_date(value: str) -> Optional[float]:
    if not value.strip():
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _header_ids(value: str) -> List[str]:
    found = MESSAGE_ID_PATTERN.findall(value or "")
    if found:
        return [_message_id_key(item) for item in found if item.strip()]
    return [_message_id_key(part.strip(" <>\t,;")) for part in re.split(r"[\s,;]+", value or "") if part.strip(" <>\t,;")]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def reconstruct_threads(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return per-message thread annotations and thread-level summaries."""
    union_find = UnionFind(len(records))
    rules: List[set[str]] = [set() for _ in records]
    message_ids: Dict[str, int] = {}
    for index, record in enumerate(records):
        message_id = _message_id_key(_text(record, "message_id", "Message-ID"))
        if message_id:
            message_ids.setdefault(message_id, index)

    for index, record in enumerate(records):
        header_refs = _header_ids(_text(record, "in_reply_to", "In-Reply-To"))
        header_refs.extend(_header_ids(_text(record, "references", "References")))
        for header_ref in header_refs:
            target = message_ids.get(header_ref)
            if target is not None and target != index:
                union_find.union(index, target)
                rules[index].add("header")
                rules[target].add("header")

    subjects: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        subject = normalize_subject(_text(record, "subject", "Subject"))
        if subject:
            subjects[subject].append(index)
    for indices in subjects.values():
        dated = sorted(
            (index for index in indices if records[index].get("_date_epoch") is not None),
            key=lambda index: (records[index]["_date_epoch"], index),
        )
        for offset, left in enumerate(dated):
            right_offset = offset + 1
            while right_offset < len(dated):
                right = dated[right_offset]
                if records[right]["_date_epoch"] - records[left]["_date_epoch"] > DAY_SECONDS:
                    break
                union_find.union(left, right)
                rules[left].add("subject")
                rules[right].add("subject")
                right_offset += 1

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union_find.find(index)].append(index)

    annotations: List[Dict[str, Any]] = [{} for _ in records]
    thread_summaries: List[Dict[str, Any]] = []
    for indices in components.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                records[index].get("_date_epoch") is None,
                records[index].get("_date_epoch") or 0,
                _text(records[index], "source_relative_path"),
            ),
        )
        identity = "\n".join(sorted(_text(records[index], "source_relative_path") for index in indices))
        thread_id = "thread-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        linked_by = sorted(set().union(*(rules[index] for index in indices)))
        for position, index in enumerate(ordered, start=1):
            annotations[index] = {"thread_id": thread_id, "position": position, "link_rules": sorted(rules[index])}
        thread_summaries.append({
            "thread_id": thread_id,
            "message_count": len(indices),
            "source_relative_paths": [_text(records[index], "source_relative_path") for index in ordered],
            "message_ids": [_text(records[index], "message_id", "Message-ID") for index in ordered],
            "link_rules": linked_by,
            "start_date": _text(records[ordered[0]], "date", "Date"),
            "end_date": _text(records[ordered[-1]], "date", "Date"),
            "action_language_candidate": any(records[index].get("_action_language_candidate", False) for index in indices),
        })
    return annotations, thread_summaries


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            dedupe_key TEXT NOT NULL UNIQUE,
            message_id TEXT NOT NULL,
            in_reply_to TEXT NOT NULL,
            references_header TEXT NOT NULL,
            subject TEXT NOT NULL,
            normalized_subject TEXT NOT NULL,
            date TEXT NOT NULL,
            date_epoch REAL,
            sender TEXT NOT NULL,
            recipients TEXT NOT NULL,
            cc TEXT NOT NULL,
            body TEXT NOT NULL,
            cleaned_body TEXT NOT NULL,
            mailbox TEXT NOT NULL,
            source_relative_path TEXT NOT NULL,
            duplicate_mailboxes TEXT NOT NULL
        )
    """)
    return connection


def _insert_record(connection: sqlite3.Connection, record: Dict[str, Any]) -> bool:
    mailbox = _text(record, "mailbox")
    relative_path = _text(record, "source_relative_path")
    if not mailbox or not relative_path:
        raise ValueError("Parsed input record must include mailbox and source_relative_path")
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError(f"Unsafe source-relative path: {relative_path!r}")
    if Path(relative_path).parts[0] != mailbox:
        raise ValueError(f"Mailbox does not match source path: {relative_path!r}")

    message_id = _text(record, "message_id", "Message-ID")
    in_reply_to = _text(record, "in_reply_to", "In-Reply-To")
    references = _text(record, "references", "References")
    subject = _text(record, "subject", "Subject")
    date = _text(record, "date", "Date")
    sender = _text(record, "from", "From")
    recipients = _text(record, "to", "To")
    cc = _text(record, "cc", "Cc")
    body = str(record.get("body", record.get("Body", "")) or "")
    key = dedupe_key(record)
    existing = connection.execute(
        "SELECT id, mailbox, duplicate_mailboxes FROM messages WHERE dedupe_key = ?", (key,)
    ).fetchone()
    if existing:
        row_id, first_mailbox, raw_mailboxes = existing
        duplicate_mailboxes = json.loads(raw_mailboxes)
        for duplicate_mailbox in (first_mailbox, mailbox):
            if duplicate_mailbox not in duplicate_mailboxes:
                duplicate_mailboxes.append(duplicate_mailbox)
        connection.execute(
            "UPDATE messages SET duplicate_mailboxes = ? WHERE id = ?",
            (json.dumps(duplicate_mailboxes), row_id),
        )
        return False

    cleaned = clean_body(body)
    connection.execute("""
        INSERT INTO messages (
            dedupe_key, message_id, in_reply_to, references_header, subject,
            normalized_subject, date, date_epoch, sender, recipients, cc, body,
            cleaned_body, mailbox, source_relative_path, duplicate_mailboxes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        key, message_id, in_reply_to, references, subject, normalize_subject(subject),
        date, _parse_date(date), sender, recipients, cc, body, cleaned, mailbox,
        relative_path, "[]",
    ))
    return True


def _read_input_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("rb") as source:
        line_number = 0
        while True:
            raw = source.readline()
            if not raw:
                break
            line_number += 1
            if not raw.endswith(b"\n"):
                raise RuntimeError(f"Incomplete final JSONL line in {path} at line {line_number}; input was not modified")
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected a JSON object in {path} at line {line_number}")
            yield value


def _count_failures(failures_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not failures_path.exists():
        return counts
    for record in _read_input_jsonl(failures_path):
        counts[str(record.get("reason") or "unknown")]+= 1
    return counts


def build(subset_dir: Path, stats_path: Path) -> Dict[str, Any]:
    by_user = subset_dir / "by_user"
    if not by_user.is_dir():
        raise FileNotFoundError(f"Subset by_user directory not found: {by_user}")
    output_dir = subset_dir.with_name(f"{subset_dir.name}_threads")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_db = output_dir / ".build_threads.sqlite3"
    temp_db.unlink(missing_ok=True)
    connection = _create_database(temp_db)
    parsed_count = 0
    input_files = sorted(by_user.glob("*.jsonl"), key=lambda path: path.name)
    try:
        for input_path in input_files:
            mailbox = input_path.stem
            for record in _read_input_jsonl(input_path):
                record.setdefault("mailbox", mailbox)
                record.setdefault("source_relative_path", f"{mailbox}/unknown/{parsed_count}")
                if record["mailbox"] != mailbox:
                    raise RuntimeError(f"Mailbox mismatch in {input_path}: {record['mailbox']!r}")
                _insert_record(connection, record)
                parsed_count += 1
                if parsed_count % 10000 == 0:
                    connection.commit()
                    print(f"input records parsed={parsed_count}", flush=True)
            connection.commit()

        failures_by_reason = _count_failures(subset_dir / "failures.jsonl")
        unique_count = int(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        duplicates_removed = parsed_count - unique_count
        unique_message_ids = int(connection.execute(
            "SELECT COUNT(DISTINCT lower(trim(message_id, ' <>'))) FROM messages WHERE trim(message_id) != ''"
        ).fetchone()[0])

        # Thinc probes optional PyTorch on import; this workspace's installed torch
        # aborts during that probe. Suppress only the optional bridge in this process.
        sys.modules["torch"] = None
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        sentence_count = 0
        token_count = 0
        cursor = connection.execute("SELECT id, cleaned_body FROM messages ORDER BY id")
        while True:
            batch = cursor.fetchmany(512)
            if not batch:
                break
            for (row_id, cleaned_body), doc in zip(batch, nlp.pipe((row[1] for row in batch), batch_size=512)):
                sentence_count += sum(1 for _ in doc.sents)
                token_count += sum(1 for token in doc if not token.is_space)

        records: List[Dict[str, Any]] = []
        for row in connection.execute("""
            SELECT id, message_id, in_reply_to, references_header, subject, date,
                   sender, recipients, cc, cleaned_body, source_relative_path, date_epoch
            FROM messages ORDER BY id
        """):
            records.append({
                "_row_id": row[0], "message_id": row[1], "in_reply_to": row[2],
                "references": row[3], "subject": row[4], "date": row[5],
                "from": row[6], "to": row[7], "cc": row[8],
                "_action_language_candidate": bool(ACTION_LANGUAGE.search(row[9])),
                "source_relative_path": row[10], "_date_epoch": row[11],
            })
        annotations, thread_summaries = reconstruct_threads(records)
        annotation_by_row = {record["_row_id"]: annotation for record, annotation in zip(records, annotations)}
        candidate_threads = sum(
            summary["message_count"] >= 3 and summary["action_language_candidate"]
            for summary in thread_summaries
        )
        threads_3_plus = sum(summary["message_count"] >= 3 for summary in thread_summaries)
        counts = Counter(summary["message_count"] for summary in thread_summaries)
        distribution = {
            "1": counts.get(1, 0),
            "2": counts.get(2, 0),
            "3-5": sum(count for size, count in counts.items() if 3 <= size <= 5),
            "6-10": sum(count for size, count in counts.items() if 6 <= size <= 10),
            "11+": sum(count for size, count in counts.items() if size >= 11),
        }

        messages_temp = output_dir / ".messages.jsonl.tmp"
        with messages_temp.open("w", encoding="utf-8") as output:
            query = connection.execute("""
                SELECT id, message_id, in_reply_to, references_header, subject, date,
                       sender, recipients, cc, body, cleaned_body, mailbox,
                       source_relative_path, duplicate_mailboxes
                FROM messages ORDER BY id
            """)
            for row in query:
                (row_id, message_id, in_reply_to, references, subject, date,
                 sender, recipients, cc, body, cleaned_body, mailbox,
                 source_relative_path, duplicate_mailboxes) = row
                result = {
                    "message_id": message_id,
                    "in_reply_to": in_reply_to,
                    "references": references,
                    "subject": subject,
                    "date": date,
                    "from": sender,
                    "to": recipients,
                    "cc": cc,
                    "body": body,
                    "cleaned_body": cleaned_body,
                    "mailbox": mailbox,
                    "source_relative_path": source_relative_path,
                    "duplicate_mailboxes": json.loads(duplicate_mailboxes),
                    **annotation_by_row[row_id],
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            import os
            os.fsync(output.fileno())
        threads_temp = output_dir / ".threads.jsonl.tmp"
        with threads_temp.open("w", encoding="utf-8") as output:
            for summary in sorted(thread_summaries, key=lambda item: item["thread_id"]):
                output.write(json.dumps(summary, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        messages_temp.replace(output_dir / "messages.jsonl")
        threads_temp.replace(output_dir / "threads.jsonl")

        failure_count = sum(failures_by_reason.values())
        reconciles = parsed_count == unique_count + duplicates_removed
        if not reconciles:
            print(
                f"WARNING: parsed ({parsed_count}) != unique ({unique_count}) + duplicates removed ({duplicates_removed})",
                flush=True,
            )
        stats = {
            "source_subset": str(subset_dir.relative_to(ROOT)) if subset_dir.is_relative_to(ROOT) else str(subset_dir),
            "parsed_count": parsed_count,
            "failure_count": failure_count,
            "failure_counts_by_reason": dict(sorted(failures_by_reason.items())),
            "unique_records": unique_count,
            "unique_message_ids": unique_message_ids,
            "duplicates_removed": duplicates_removed,
            "parsed_reconciles_to_unique_plus_duplicates": reconciles,
            "thread_count": len(thread_summaries),
            "messages_per_thread": distribution,
            "sentence_count_cleaned_body": sentence_count,
            "token_count_cleaned_body": token_count,
            "threads_with_3_or_more_messages": threads_3_plus,
            "threads_with_3_or_more_and_action_language": candidate_threads,
            "action_language_patterns": list(ACTION_LANGUAGE_PATTERNS),
            "sentence_segmenter": "spaCy blank English pipeline with sentencizer component",
            "outputs": {
                "messages": str((output_dir / "messages.jsonl").relative_to(ROOT)),
                "threads": str((output_dir / "threads.jsonl").relative_to(ROOT)),
            },
        }
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_temp = stats_path.with_name(f".{stats_path.name}.tmp")
        with stats_temp.open("w", encoding="utf-8") as output:
            json.dump(stats, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            import os
            os.fsync(output.fileno())
        stats_temp.replace(stats_path)
        print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
        return stats
    finally:
        connection.close()
        temp_db.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset_dir", help="Subset directory, such as enron_subset_smoke or data/processed/enron_subset_smoke")
    parser.add_argument("--stats-output", default="results/data_stats.json", help="Statistics JSON path; override this for smoke runs.")
    args = parser.parse_args()
    subset_dir = Path(args.subset_dir)
    if not subset_dir.is_absolute():
        if subset_dir.parts and subset_dir.parts[0] == "data":
            subset_dir = ROOT / subset_dir
        else:
            subset_dir = ROOT / "data" / "processed" / subset_dir
    stats_path = Path(args.stats_output)
    if not stats_path.is_absolute():
        stats_path = ROOT / stats_path
    try:
        build(subset_dir.resolve(), stats_path.resolve())
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())