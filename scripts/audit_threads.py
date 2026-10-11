#!/usr/bin/env python3
"""Read-only schema and consistency audit for reconstructed thread JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_threads import ACTION_LANGUAGE, _parse_date, dedupe_key  # noqa: E402

MESSAGE_REQUIRED_TYPES = {
    "message_id": str,
    "in_reply_to": str,
    "references": str,
    "subject": str,
    "date": str,
    "from": str,
    "to": str,
    "cc": str,
    "body": str,
    "cleaned_body": str,
    "mailbox": str,
    "source_relative_path": str,
    "thread_id": str,
    "position": int,
    "duplicate_mailboxes": list,
    "link_rules": list,
}
THREAD_REQUIRED_TYPES = {
    "thread_id": str,
    "message_count": int,
    "source_relative_paths": list,
    "message_ids": list,
    "link_rules": list,
    "start_date": str,
    "end_date": str,
    "action_language_candidate": bool,
}
VALID_LINK_RULES = {"header", "subject"}
TOKEN_PATTERN = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete JSONL line in {path} at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            yield value


def audit_records(
    message_records: Iterable[Dict[str, Any]],
    thread_records: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []

    def issue(code: str, severity: str, location: str, detail: str) -> None:
        issues.append({"code": code, "severity": severity, "location": location, "detail": detail})

    observed_by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_identifiers: Dict[str, str] = {}
    missingness = Counter()
    link_rule_messages = Counter()
    malformed_dates = 0
    missing_dates = 0
    empty_cleaned_bodies = 0
    short_cleaned_bodies = 0
    sentence_count = 0
    token_count = 0
    candidate_messages = 0
    message_count = 0

    for record in message_records:
        message_count += 1
        path_value = record.get("source_relative_path", f"message[{message_count}]")
        location = str(path_value)
        for field, expected_type in MESSAGE_REQUIRED_TYPES.items():
            value = record.get(field)
            valid = isinstance(value, expected_type)
            if expected_type is int and isinstance(value, bool):
                valid = False
            if not valid:
                issue("message_field_type", "error", location, f"{field} must be {expected_type.__name__}")

        for field in ("message_id", "in_reply_to", "references", "subject", "date", "from", "to", "cc", "body", "cleaned_body"):
            if _is_missing(record.get(field)):
                missingness[field] += 1
        if _is_missing(record.get("from")) or _is_missing(record.get("to")):
            missingness["participants_either_sender_or_recipient"] += 1

        cleaned = record.get("cleaned_body")
        if not isinstance(cleaned, str):
            cleaned = ""
        if not cleaned.strip():
            empty_cleaned_bodies += 1
            issue("empty_cleaned_body", "warning", location, "cleaned_body is empty")
        elif len(cleaned.strip()) < 20:
            short_cleaned_bodies += 1
        sentence_count += len([part for part in SENTENCE_SPLIT.split(cleaned.strip()) if part.strip()]) if cleaned.strip() else 0
        token_count += len(TOKEN_PATTERN.findall(cleaned))
        if ACTION_LANGUAGE.search(cleaned):
            candidate_messages += 1

        date_value = record.get("date")
        timestamp = _parse_date(date_value) if isinstance(date_value, str) else None
        if _is_missing(date_value):
            missing_dates += 1
        elif timestamp is None:
            malformed_dates += 1
            issue("unparseable_date", "warning", location, "date is non-empty but not parseable")

        try:
            identity = dedupe_key(record)
        except (TypeError, ValueError) as exc:
            identity = f"invalid:{message_count}"
            issue("identifier_unavailable", "error", location, str(exc))
        if identity in seen_identifiers:
            issue("duplicate_message_identifier", "error", location, f"identifier duplicates {seen_identifiers[identity]}")
        else:
            seen_identifiers[identity] = location

        rules = record.get("link_rules")
        if isinstance(rules, list):
            if any(not isinstance(rule, str) or rule not in VALID_LINK_RULES for rule in rules):
                issue("invalid_message_link_rule", "error", location, "link_rules must contain only 'header' or 'subject'")
            for rule in set(rule for rule in rules if isinstance(rule, str)):
                link_rule_messages[rule] += 1
        else:
            issue("invalid_message_link_rule", "error", location, "link_rules must be a list")

        duplicate_mailboxes = record.get("duplicate_mailboxes")
        if isinstance(duplicate_mailboxes, list):
            if any(not isinstance(mailbox, str) or not mailbox for mailbox in duplicate_mailboxes):
                issue("invalid_duplicate_provenance", "error", location, "duplicate_mailboxes must contain non-empty strings")
            elif len(duplicate_mailboxes) != len(set(duplicate_mailboxes)):
                issue("invalid_duplicate_provenance", "error", location, "duplicate_mailboxes contains repeated mailbox names")
            elif duplicate_mailboxes and record.get("mailbox") not in duplicate_mailboxes:
                issue("invalid_duplicate_provenance", "error", location, "duplicate_mailboxes must include the retained record's mailbox")
        else:
            issue("invalid_duplicate_provenance", "error", location, "duplicate_mailboxes must be a list")

        thread_id = record.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            observed_by_thread[thread_id].append({
                "position": record.get("position"),
                "source_relative_path": path_value,
                "message_id": record.get("message_id", ""),
                "date": date_value,
                "timestamp": timestamp,
                "link_rules": rules if isinstance(rules, list) else [],
            })
        else:
            issue("missing_thread_reference", "error", location, "thread_id must be a non-empty string")

    thread_count = 0
    declared_by_id: Dict[str, Dict[str, Any]] = {}
    subject_only_threads = 0
    mixed_rule_threads = 0
    thread_rule_counts = Counter()
    for thread in thread_records:
        thread_count += 1
        thread_id = thread.get("thread_id", f"thread[{thread_count}]")
        location = str(thread_id)
        for field, expected_type in THREAD_REQUIRED_TYPES.items():
            value = thread.get(field)
            valid = isinstance(value, expected_type)
            if expected_type is int and isinstance(value, bool):
                valid = False
            if not valid:
                issue("thread_field_type", "error", location, f"{field} must be {expected_type.__name__}")
        if not isinstance(thread_id, str) or not thread_id:
            continue
        if thread_id in declared_by_id:
            issue("duplicate_thread_id", "error", location, "thread_id appears more than once")
            continue
        declared_by_id[thread_id] = thread
        declared_rules = thread.get("link_rules", [])
        if isinstance(declared_rules, list):
            if any(not isinstance(rule, str) or rule not in VALID_LINK_RULES for rule in declared_rules):
                issue("invalid_thread_link_rule", "error", location, "thread link_rules contains an unsupported value")
            rule_set = set(rule for rule in declared_rules if isinstance(rule, str))
            for rule in rule_set:
                thread_rule_counts[rule] += 1
            if rule_set == {"subject"}:
                subject_only_threads += 1
            if rule_set == {"header", "subject"}:
                mixed_rule_threads += 1
            observed_rules = {
                rule
                for member in observed_by_thread.get(thread_id, [])
                for rule in member["link_rules"]
                if isinstance(rule, str)
            }
            if rule_set != observed_rules:
                issue("thread_link_rules_mismatch", "error", location, "thread link_rules must equal the union of member message link_rules")
        else:
            issue("invalid_thread_link_rule", "error", location, "thread link_rules must be a list")
        for field in ("source_relative_paths", "message_ids"):
            values = thread.get(field)
            if isinstance(values, list) and any(not isinstance(value, str) for value in values):
                issue("thread_membership_type", "error", location, f"{field} must contain only strings")

        members = observed_by_thread.get(thread_id, [])
        declared_count = thread.get("message_count")
        if isinstance(declared_count, int) and not isinstance(declared_count, bool) and declared_count != len(members):
            issue("thread_count_mismatch", "error", location, f"declared {declared_count}, observed {len(members)}")
        positions = [member["position"] for member in members if isinstance(member["position"], int) and not isinstance(member["position"], bool)]
        if len(positions) != len(members) or sorted(positions) != list(range(1, len(members) + 1)):
            issue("thread_positions_invalid", "error", location, "positions must be unique and contiguous from 1")
        if len(positions) != len(set(positions)):
            issue("thread_positions_duplicate", "error", location, "position repeats within thread")

        ordered = sorted(members, key=lambda member: (member["position"] if isinstance(member["position"], int) else float("inf")))
        declared_paths = thread.get("source_relative_paths")
        if isinstance(declared_paths, list):
            if declared_paths != [member["source_relative_path"] for member in ordered]:
                issue("thread_membership_mismatch", "error", location, "source_relative_paths disagrees with message records")
        declared_ids = thread.get("message_ids")
        if isinstance(declared_ids, list):
            if declared_ids != [member["message_id"] for member in ordered]:
                issue("thread_message_ids_mismatch", "error", location, "message_ids disagrees with message records")

        chronology_keys = []
        for member in ordered:
            timestamp = member["timestamp"]
            chronology_keys.append((timestamp is None, timestamp or 0, str(member["source_relative_path"])))
        if chronology_keys != sorted(chronology_keys):
            issue("thread_chronology_invalid", "error", location, "positions do not follow date then source-path ordering; missing/unparseable dates sort last")

    for thread_id in sorted(set(observed_by_thread) - set(declared_by_id)):
        issue("orphan_message_thread", "error", thread_id, "messages reference a thread absent from threads.jsonl")

    observed_sizes = [len(members) for members in observed_by_thread.values()]
    size_counts = Counter(observed_sizes)
    distribution = {
        "1": size_counts.get(1, 0),
        "2": size_counts.get(2, 0),
        "3-5": sum(count for size, count in size_counts.items() if 3 <= size <= 5),
        "6-10": sum(count for size, count in size_counts.items() if 6 <= size <= 10),
        "11+": sum(count for size, count in size_counts.items() if size >= 11),
    }
    issue_counts = Counter(item["code"] for item in issues)
    error_count = sum(item["severity"] == "error" for item in issues)
    stats = {
        "message_count": message_count,
        "thread_record_count": thread_count,
        "observed_thread_count_from_messages": len(observed_by_thread),
        "unique_builder_message_identifiers": len(seen_identifiers),
        "messages_per_thread": distribution,
        "threads_with_at_least_3_messages": sum(size >= 3 for size in observed_sizes),
        "missingness": dict(sorted(missingness.items())),
        "missing_dates": missing_dates,
        "unparseable_dates": malformed_dates,
        "empty_cleaned_bodies": empty_cleaned_bodies,
        "short_cleaned_bodies_under_20_characters": short_cleaned_bodies,
        "message_counts_by_link_rule": dict(sorted(link_rule_messages.items())),
        "thread_counts_by_link_rule": dict(sorted(thread_rule_counts.items())),
        "subject_only_threads": subject_only_threads,
        "mixed_header_and_subject_threads": mixed_rule_threads,
        "candidate_messages_with_task_update_cues": candidate_messages,
        "cleaned_body_sentence_count": sentence_count,
        "cleaned_body_token_count": token_count,
        "text_counting_method": "sentences: split on whitespace after '.', '!' or '?'; tokens: Unicode regex \\b[\\w’'-]+\\b; counts are heuristics, not linguistic gold.",
        "audit_error_count": error_count,
        "audit_warning_count": len(issues) - error_count,
        "issue_counts_by_code": dict(sorted(issue_counts.items())),
        "semantic_thread_correctness_proven": False,
        "interpretation": "Structural consistency checks do not establish that reconstructed threads are semantically correct.",
    }
    return stats, issues


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def audit_directory(thread_output: Path, audit_output: Path, issues_output: Path) -> Dict[str, Any]:
    messages_path = thread_output / "messages.jsonl"
    threads_path = thread_output / "threads.jsonl"
    if not messages_path.is_file() or not threads_path.is_file():
        raise FileNotFoundError(f"Expected messages.jsonl and threads.jsonl under {thread_output}")
    stats, issues = audit_records(_read_jsonl(messages_path), _read_jsonl(threads_path))
    _atomic_json(audit_output, stats)
    issues_output.parent.mkdir(parents=True, exist_ok=True)
    temp = issues_output.with_name(f".{issues_output.name}.tmp")
    with temp.open("w", encoding="utf-8") as output:
        for item in issues:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temp.replace(issues_output)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_output_dir", help="Directory containing messages.jsonl and threads.jsonl")
    parser.add_argument("--audit-output", required=True, help="Explicit audit summary JSON path")
    parser.add_argument("--issues-output", required=True, help="Explicit structural issue JSONL path")
    args = parser.parse_args()
    try:
        stats = audit_directory(Path(args.thread_output_dir), Path(args.audit_output), Path(args.issues_output))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0 if stats["audit_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())