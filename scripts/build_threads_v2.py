#!/usr/bin/env python3
"""Conservatively link v2 Enron records without body-only deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTION_LANGUAGE_PATTERNS = (
    r"\bplease\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bdeadline\b",
    r"\bdue\b",
    r"\bcancel(?:led|ing|lation)?\b",
    r"\breassign\w*\b",
    r"\binstead\b",
    r"\bneed to\b",
)
ACTION_LANGUAGE = re.compile("|".join(f"(?:{pattern})" for pattern in ACTION_LANGUAGE_PATTERNS), re.I)
SUBJECT_PREFIX = re.compile(r"^(?:(?:re|fw|fwd)\s*:\s*)+", re.I)
MESSAGE_ID_PATTERN = re.compile(r"<([^<>]+)>")
TOKEN_PATTERN = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
GENERIC_SUBJECTS = {
    "hi", "hello", "test", "meeting", "question", "update", "fyi",
    "re", "fw", "fwd", "no subject", "(no subject)",
}
SUBJECT_PAIR_WINDOW_SECONDS = 7 * 24 * 60 * 60
SUBJECT_COMPONENT_SPAN_SECONDS = 14 * 24 * 60 * 60


def _text(record: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None:
            return " ".join(str(value).split())
    return ""


def normalize_subject(subject: str) -> str:
    value = " ".join((subject or "").split())
    while True:
        updated = SUBJECT_PREFIX.sub("", value).strip()
        if updated == value:
            return value.casefold()
        value = updated


def meaningful_subject(subject: str) -> str:
    normalized = normalize_subject(subject)
    words = TOKEN_PATTERN.findall(normalized)
    if normalized in GENERIC_SUBJECTS or len(normalized) < 5 or len(words) < 2:
        return ""
    return normalized


def _message_id_key(message_id: str) -> str:
    return " ".join((message_id or "").strip().strip("<>").split()).casefold()


def _header_ids(value: str) -> List[str]:
    found = MESSAGE_ID_PATTERN.findall(value or "")
    if found:
        return [_message_id_key(item) for item in found if item.strip()]
    return [_message_id_key(part.strip(" <>\t,;")) for part in re.split(r"[\s,;]+", value or "") if part.strip(" <>\t,;")]


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


def _participants(record: Dict[str, Any]) -> Set[str]:
    values = [_text(record, "from"), _text(record, "to"), _text(record, "cc")]
    return {
        address.casefold()
        for _, address in getaddresses(values)
        if address and "@" in address
    }


def _body_hash(record: Dict[str, Any]) -> str:
    body = str(record.get("body", "") or "")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _metadata_signature(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        _text(record, "from").casefold(),
        _text(record, "to").casefold(),
        _text(record, "cc").casefold(),
        _text(record, "subject").casefold(),
        _text(record, "date").casefold(),
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


def classify_duplicates(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotations = [{"duplicate_category": "unique", "duplicate_group_id": "", "duplicate_provenance": []} for _ in records]
    stats = Counter()

    by_id: Dict[str, List[int]] = defaultdict(list)
    by_body: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        message_id = _message_id_key(_text(record, "message_id"))
        if message_id:
            by_id[message_id].append(index)
        body_hash = _body_hash(record)
        if body_hash and str(record.get("body", "") or "").strip():
            by_body[body_hash].append(index)

    for message_id, indices in by_id.items():
        if len(indices) < 2:
            continue
        body_hashes = {_body_hash(records[index]) for index in indices}
        signatures = {_metadata_signature(records[index]) for index in indices}
        category = "same_message_id_copy" if len(body_hashes) == 1 and len(signatures) == 1 else "conflicting_message_id"
        group_id = "msgid-" + hashlib.sha1(message_id.encode("utf-8")).hexdigest()[:16]
        provenance = [
            {
                "source_relative_path": _text(records[index], "source_relative_path"),
                "mailbox": _text(records[index], "mailbox"),
            }
            for index in sorted(indices, key=lambda item: _text(records[item], "source_relative_path"))
        ]
        stats[category + "_groups"] += 1
        stats[category + "_records"] += len(indices)
        for index in indices:
            annotations[index] = {
                "duplicate_category": category,
                "duplicate_group_id": group_id,
                "duplicate_provenance": provenance,
            }

    for body_hash, indices in by_body.items():
        if len(indices) < 2:
            continue
        message_ids = {_message_id_key(_text(records[index], "message_id")) for index in indices}
        if len(message_ids) <= 1:
            continue
        group_id = "body-" + body_hash[:16]
        stats["identical_body_different_message_id_groups"] += 1
        stats["identical_body_different_message_id_records"] += len(indices)
        for index in indices:
            if annotations[index]["duplicate_category"] == "unique":
                annotations[index] = {
                    "duplicate_category": "identical_body_different_message_id_review",
                    "duplicate_group_id": group_id,
                    "duplicate_provenance": [
                        {
                            "source_relative_path": _text(records[item], "source_relative_path"),
                            "mailbox": _text(records[item], "mailbox"),
                            "message_id": _text(records[item], "message_id"),
                        }
                        for item in sorted(indices, key=lambda item: _text(records[item], "source_relative_path"))
                    ],
                }

    stats["records_preserved_without_body_only_collapse"] = len(records)
    return annotations, dict(sorted(stats.items()))


def reconstruct_threads_v2(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    union_find = UnionFind(len(records))
    rules: List[Set[str]] = [set() for _ in records]
    link_details: List[List[Dict[str, Any]]] = [[] for _ in records]
    component_dates: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for index, record in enumerate(records):
        record["_participants"] = _participants(record)
        timestamp = record.get("_date_epoch")
        component_dates[index] = (timestamp, timestamp)

    message_ids: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        message_id = _message_id_key(_text(record, "message_id"))
        if message_id:
            message_ids[message_id].append(index)

    for index, record in enumerate(records):
        refs = _header_ids(_text(record, "in_reply_to"))
        refs.extend(_header_ids(_text(record, "references")))
        for ref in refs:
            for target in message_ids.get(ref, []):
                if target == index:
                    continue
                root = union_find.union(index, target)
                rules[index].add("header")
                rules[target].add("header")
                detail = {"rule": "header", "target_source_relative_path": _text(records[target], "source_relative_path")}
                link_details[index].append(detail)
                component_dates[root] = _component_span(union_find, component_dates, index, target)

    subjects: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        subject = meaningful_subject(_text(record, "subject"))
        if subject and record.get("_date_epoch") is not None:
            subjects[subject].append(index)
    for subject, indices in subjects.items():
        dated = sorted(indices, key=lambda index: (records[index]["_date_epoch"], _text(records[index], "source_relative_path")))
        for right_pos in range(1, len(dated)):
            left = dated[right_pos - 1]
            right = dated[right_pos]
            gap = records[right]["_date_epoch"] - records[left]["_date_epoch"]
            if gap > SUBJECT_PAIR_WINDOW_SECONDS:
                continue
            if not (records[left].get("_participants", set()) & records[right].get("_participants", set())):
                continue
            combined_min, combined_max = _component_span(union_find, component_dates, left, right)
            if combined_min is not None and combined_max is not None and combined_max - combined_min > SUBJECT_COMPONENT_SPAN_SECONDS:
                continue
            root = union_find.union(left, right)
            component_dates[root] = (combined_min, combined_max)
            rules[left].add("subject_heuristic")
            rules[right].add("subject_heuristic")
            detail = {
                "rule": "subject_heuristic",
                "subject": subject,
                "target_source_relative_path": _text(records[left], "source_relative_path"),
                "max_pair_gap_days": 7,
                "max_component_span_days": 14,
                "participant_overlap_required": True,
            }
            link_details[right].append(detail)

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union_find.find(index)].append(index)

    annotations: List[Dict[str, Any]] = [{} for _ in records]
    thread_summaries: List[Dict[str, Any]] = []
    for indices in components.values():
        ordered = sorted(indices, key=lambda index: (
            records[index].get("_date_epoch") is None,
            records[index].get("_date_epoch") or 0,
            _text(records[index], "source_relative_path"),
        ))
        identity = "\n".join(sorted(_text(records[index], "source_relative_path") for index in indices))
        thread_id = "thread-v2-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        linked_by = sorted(set().union(*(rules[index] for index in indices)))
        confidence = "strong_header" if "header" in linked_by else ("heuristic_subject" if "subject_heuristic" in linked_by else "singleton")
        for position, index in enumerate(ordered, start=1):
            annotations[index] = {
                "thread_id": thread_id,
                "position": position,
                "link_rules": sorted(rules[index]),
                "link_details": link_details[index],
                "thread_confidence": confidence,
            }
        thread_summaries.append({
            "thread_id": thread_id,
            "message_count": len(indices),
            "source_relative_paths": [_text(records[index], "source_relative_path") for index in ordered],
            "message_ids": [_text(records[index], "message_id") for index in ordered],
            "link_rules": linked_by,
            "thread_confidence": confidence,
            "start_date": _text(records[ordered[0]], "date"),
            "end_date": _text(records[ordered[-1]], "date"),
            "action_language_candidate": any(ACTION_LANGUAGE.search(str(records[index].get("cleaned_body", "") or "")) for index in indices),
        })
    stats = {
        "subject_link_thresholds": {
            "meaningful_subject_min_chars": 5,
            "meaningful_subject_min_words": 2,
            "pair_window_days": 7,
            "component_span_days": 14,
            "participant_overlap_required": True,
        }
    }
    return annotations, thread_summaries, stats


def _component_span(union_find: UnionFind, component_dates: Dict[int, Tuple[Optional[float], Optional[float]]], left: int, right: int) -> Tuple[Optional[float], Optional[float]]:
    left_min, left_max = component_dates.get(union_find.find(left), (None, None))
    right_min, right_max = component_dates.get(union_find.find(right), (None, None))
    values = [value for value in (left_min, left_max, right_min, right_max) if value is not None]
    if not values:
        return None, None
    return min(values), max(values)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(f"Incomplete JSONL line in {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object in {path}:{line_number}")
            yield value


def _load_records(subset_dir: Path) -> List[Dict[str, Any]]:
    by_user = subset_dir / "by_user"
    if not by_user.is_dir():
        raise FileNotFoundError(f"Missing by_user directory: {by_user}")
    records: List[Dict[str, Any]] = []
    for path in sorted(by_user.glob("*.jsonl")):
        for record in _read_jsonl(path):
            record["_date_epoch"] = _parse_date(_text(record, "date"))
            record["_participants"] = _participants(record)
            records.append(record)
    return records


def _sentence_count(text: str) -> int:
    stripped = (text or "").strip()
    return len([part for part in SENTENCE_SPLIT.split(stripped) if part.strip()]) if stripped else 0


def build_v2(subset_dir: Path, output_dir: Path, stats_path: Path) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite v2 thread output directory: {output_dir}")
    records = _load_records(subset_dir)
    duplicate_annotations, duplicate_stats = classify_duplicates(records)
    thread_annotations, threads, thread_stats = reconstruct_threads_v2(records)
    output_dir.mkdir(parents=True, exist_ok=False)

    messages_temp = output_dir / ".messages.jsonl.tmp"
    sentence_count = 0
    token_count = 0
    missing_headers = Counter()
    cleaning_flags = Counter()
    with messages_temp.open("w", encoding="utf-8") as output:
        for record, duplicate_annotation, thread_annotation in zip(records, duplicate_annotations, thread_annotations):
            public = {key: value for key, value in record.items() if not key.startswith("_")}
            public.update(duplicate_annotation)
            public.update(thread_annotation)
            cleaned_body = str(public.get("cleaned_body", "") or "")
            sentence_count += _sentence_count(cleaned_body)
            token_count += len(TOKEN_PATTERN.findall(cleaned_body))
            for field in ("message_id", "in_reply_to", "references", "from", "to", "cc", "subject", "date"):
                if not _text(public, field):
                    missing_headers[field] += 1
            for flag in public.get("cleaning_flags", []):
                cleaning_flags[str(flag)] += 1
            for flag in public.get("cleaning_uncertain", []):
                cleaning_flags["uncertain:" + str(flag)] += 1
            output.write(json.dumps(public, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())

    threads_temp = output_dir / ".threads.jsonl.tmp"
    with threads_temp.open("w", encoding="utf-8") as output:
        for thread in sorted(threads, key=lambda item: item["thread_id"]):
            output.write(json.dumps(thread, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    messages_temp.replace(output_dir / "messages.jsonl")
    threads_temp.replace(output_dir / "threads.jsonl")

    thread_size_counts = Counter(thread["message_count"] for thread in threads)
    link_counter = Counter(tuple(thread["link_rules"]) for thread in threads)
    stats = {
        "schema_version": "enron_threads_v2",
        "source_subset": str(subset_dir.relative_to(ROOT)) if subset_dir.is_relative_to(ROOT) else str(subset_dir),
        "parsed_messages": len(records),
        "failure_count": sum(1 for _ in _read_jsonl(subset_dir / "failures.jsonl")) if (subset_dir / "failures.jsonl").exists() else 0,
        "unique_records_preserved": len(records),
        "duplicate_counts": duplicate_stats,
        "thread_count": len(threads),
        "messages_per_thread": {
            "1": thread_size_counts.get(1, 0),
            "2": thread_size_counts.get(2, 0),
            "3-5": sum(count for size, count in thread_size_counts.items() if 3 <= size <= 5),
            "6-10": sum(count for size, count in thread_size_counts.items() if 6 <= size <= 10),
            "11+": sum(count for size, count in thread_size_counts.items() if size >= 11),
        },
        "thread_link_categories": {
            "singleton": sum(1 for thread in threads if not thread["link_rules"]),
            "header_linked": sum(1 for thread in threads if "header" in thread["link_rules"]),
            "subject_only_heuristic": link_counter.get(("subject_heuristic",), 0),
            "mixed_header_and_subject_heuristic": sum(1 for thread in threads if set(thread["link_rules"]) == {"header", "subject_heuristic"}),
        },
        "missing_header_counts": dict(sorted(missing_headers.items())),
        "sentence_count_cleaned_body": sentence_count,
        "token_count_cleaned_body": token_count,
        "text_counting_method": "sentences split on whitespace after .!?; tokens use Unicode regex \\b[\\w’'-]+\\b",
        "remaining_quote_forward_signature_indicators": dict(sorted(cleaning_flags.items())),
        "action_language_patterns": list(ACTION_LANGUAGE_PATTERNS),
        **thread_stats,
        "outputs": {
            "messages": str((output_dir / "messages.jsonl").relative_to(ROOT)),
            "threads": str((output_dir / "threads.jsonl").relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temp = stats_path.with_name(f".{stats_path.name}.tmp")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(stats, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temp.replace(stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stats-output", required=True)
    args = parser.parse_args()
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)
    stats_path = Path(args.stats_output)
    if not subset_dir.is_absolute():
        subset_dir = ROOT / subset_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not stats_path.is_absolute():
        stats_path = ROOT / stats_path
    try:
        build_v2(subset_dir.resolve(), output_dir.resolve(), stats_path.resolve())
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
