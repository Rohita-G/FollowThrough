#!/usr/bin/env python3
"""Validate versioned task-message annotations against explicit thread JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SCHEMA_VERSION = "1.0"
EXAMPLE_KINDS = {"screening_draft", "task_update", "task_creation"}
ANNOTATION_STATUSES = {"draft", "complete"}
EXAMPLE_ORIGINS = {"real", "synthetic"}
MESSAGE_ACT_LABELS = {"request", "commitment"}
RELATION_LABELS = {
    "deadline_change", "reassignment", "cancellation", "completion", "unrelated", "ambiguous",
}
ALL_LABELS = MESSAGE_ACT_LABELS | RELATION_LABELS
STRING_FIELDS = (
    "schema_version", "example_id", "example_kind", "example_origin", "annotation_status",
    "thread_id", "task_id", "task_text", "task_owner_text", "task_deadline_text",
    "update_message_id", "update_text", "updated_owner_text", "updated_deadline_text",
    "ambiguity_reason", "annotator_id", "annotation_notes",
)
LIST_STRING_FIELDS = ("task_source_message_ids", "message_act_labels", "relation_labels")
NULLABLE_DATE_FIELDS = ("task_deadline_normalized", "updated_deadline_normalized")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            yield item


def message_reference_key(record: Dict[str, Any]) -> Optional[str]:
    message_id = record.get("message_id")
    if isinstance(message_id, str) and message_id.strip():
        return message_id
    source_path = record.get("source_relative_path")
    if isinstance(source_path, str) and source_path.strip():
        return f"path:{source_path}"
    return None


def load_reference_index(messages_path: Path, threads_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]], List[Dict[str, Any]]]:
    messages: Dict[str, Dict[str, Any]] = {}
    threads: Dict[str, Set[str]] = {}
    thread_records: Dict[str, Dict[str, Any]] = {}
    issues: List[Dict[str, Any]] = []
    for line_number, record in enumerate(_read_jsonl(messages_path), start=1):
        message_id = message_reference_key(record)
        if message_id is None:
            issues.append({"code": "invalid_reference_message_id", "line": line_number, "detail": "message needs a Message-ID or source_relative_path"})
            continue
        if message_id in messages:
            issues.append({"code": "duplicate_reference_message_id", "line": line_number, "detail": message_id})
            continue
        if not isinstance(record.get("thread_id"), str) or not record["thread_id"]:
            issues.append({"code": "invalid_reference_message_thread", "line": line_number, "detail": "thread_id must be a non-empty string"})
        if not isinstance(record.get("position"), int) or isinstance(record.get("position"), bool) or record["position"] < 1:
            issues.append({"code": "invalid_reference_message_position", "line": line_number, "detail": "position must be a positive integer"})
        if not isinstance(record.get("source_relative_path"), str) or not record["source_relative_path"]:
            issues.append({"code": "invalid_reference_message_path", "line": line_number, "detail": "source_relative_path must be a non-empty string"})
        if not isinstance(record.get("cleaned_body"), str):
            issues.append({"code": "invalid_reference_message_text", "line": line_number, "detail": "cleaned_body must be a string"})
        messages[message_id] = record
    for line_number, record in enumerate(_read_jsonl(threads_path), start=1):
        thread_id = record.get("thread_id")
        paths = record.get("source_relative_paths")
        if not isinstance(thread_id, str) or not thread_id:
            issues.append({"code": "invalid_reference_thread_id", "line": line_number, "detail": "thread_id must be a non-empty string"})
            continue
        if thread_id in threads:
            issues.append({"code": "duplicate_reference_thread_id", "line": line_number, "detail": thread_id})
            continue
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            issues.append({"code": "invalid_reference_thread_members", "line": line_number, "detail": "source_relative_paths must be a string list"})
            threads[thread_id] = {path for path in paths if isinstance(path, str)} if isinstance(paths, list) else set()
        else:
            threads[thread_id] = set(paths)
        thread_records[thread_id] = record

    paths_to_messages: Dict[str, Dict[str, Any]] = {}
    for line_number, message in enumerate(messages.values(), start=1):
        source_path = message.get("source_relative_path")
        if isinstance(source_path, str) and source_path:
            if source_path in paths_to_messages:
                issues.append({"code": "duplicate_reference_source_path", "line": line_number, "detail": source_path})
            paths_to_messages[source_path] = message
            message_thread = message.get("thread_id")
            if isinstance(message_thread, str) and message_thread not in threads:
                issues.append({"code": "orphan_reference_message_thread", "line": line_number, "detail": message_thread})
    for thread_id, member_paths in threads.items():
        declared_count = thread_records[thread_id].get("message_count")
        if not isinstance(declared_count, int) or isinstance(declared_count, bool) or declared_count != len(member_paths):
            issues.append({"code": "reference_thread_count_mismatch", "line": 0, "detail": thread_id})
        if len(member_paths) != len(thread_records[thread_id].get("source_relative_paths", [])):
            issues.append({"code": "duplicate_reference_thread_member", "line": 0, "detail": thread_id})
        for source_path in member_paths:
            message = paths_to_messages.get(source_path)
            if message is None:
                issues.append({"code": "missing_reference_thread_member", "line": 0, "detail": f"{thread_id}: {source_path}"})
            elif message.get("thread_id") != thread_id:
                issues.append({"code": "reference_membership_mismatch", "line": 0, "detail": f"{thread_id}: {source_path}"})
    for message in messages.values():
        message_thread = message.get("thread_id")
        source_path = message.get("source_relative_path")
        if isinstance(message_thread, str) and isinstance(source_path, str) and source_path not in threads.get(message_thread, set()):
            issues.append({"code": "message_missing_from_reference_thread", "line": 0, "detail": source_path})
    return messages, threads, issues


def validate_examples(
    examples: Iterable[Dict[str, Any]],
    messages: Dict[str, Dict[str, Any]],
    threads: Dict[str, Set[str]],
    initial_issues: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    issues = list(initial_issues or [])
    seen_example_ids: Set[str] = set()
    example_count = 0
    draft_count = 0
    complete_count = 0
    origin_counts: Dict[str, int] = {origin: 0 for origin in sorted(EXAMPLE_ORIGINS)}

    def add(code: str, row_number: int, detail: str, severity: str = "error") -> None:
        issues.append({"code": code, "severity": severity, "line": row_number, "detail": detail})

    for row_number, example in enumerate(examples, start=1):
        example_count += 1
        for field in STRING_FIELDS:
            if not isinstance(example.get(field), str):
                add("field_type", row_number, f"{field} must be a string")
        for field in LIST_STRING_FIELDS:
            value = example.get(field)
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                add("field_type", row_number, f"{field} must be a list of strings")
        if not isinstance(example.get("ambiguous"), bool):
            add("field_type", row_number, "ambiguous must be a boolean")
        if not isinstance(example.get("evidence_spans"), list):
            add("field_type", row_number, "evidence_spans must be a list")
        for field in NULLABLE_DATE_FIELDS:
            if field not in example:
                add("field_missing", row_number, f"{field} is required and must be a string or null")
            value = example.get(field)
            if value is not None and not isinstance(value, str):
                add("field_type", row_number, f"{field} must be a string or null")

        example_id = example.get("example_id")
        if isinstance(example_id, str) and example_id:
            if example_id in seen_example_ids:
                add("duplicate_example_id", row_number, example_id)
            seen_example_ids.add(example_id)
        if example.get("schema_version") != SCHEMA_VERSION:
            add("schema_version", row_number, f"expected {SCHEMA_VERSION}")
        kind = example.get("example_kind")
        if not isinstance(kind, str) or kind not in EXAMPLE_KINDS:
            add("example_kind", row_number, "must be screening_draft, task_update, or task_creation")
        origin = example.get("example_origin")
        if not isinstance(origin, str) or origin not in EXAMPLE_ORIGINS:
            add("example_origin", row_number, "must be real or synthetic")
        elif isinstance(origin, str):
            origin_counts[origin] += 1

        status = example.get("annotation_status")
        if not isinstance(status, str) or status not in ANNOTATION_STATUSES:
            add("annotation_status", row_number, "must be draft or complete")
        elif status == "draft":
            draft_count += 1
        else:
            complete_count += 1
        if kind == "screening_draft" and status != "draft":
            add("screening_draft_status", row_number, "screening_draft must remain an unlabeled draft")

        required_nonempty = ["example_id", "thread_id", "update_message_id"]
        if kind != "screening_draft":
            required_nonempty.extend(("task_id", "task_text"))
        if kind != "screening_draft" or status != "draft":
            required_nonempty.append("update_text")
        for field in required_nonempty:
            value = example.get(field)
            if isinstance(value, str) and not value.strip():
                add("empty_required_value", row_number, f"{field} must be non-empty")
        if isinstance(example.get("task_source_message_ids"), list):
            raw_source_ids = example["task_source_message_ids"]
            source_ids = [item for item in raw_source_ids if isinstance(item, str)]
            requires_sources = isinstance(kind, str) and kind in {"task_update", "task_creation"} and status == "complete"
            if (requires_sources and not source_ids) or len(source_ids) != len(raw_source_ids) or any(not item.strip() for item in source_ids):
                add("task_source_ids", row_number, "must contain at least one non-empty message ID")
            if len(source_ids) != len(set(source_ids)):
                add("task_source_ids", row_number, "must not repeat message IDs")
        else:
            source_ids = []

        raw_acts = example.get("message_act_labels") if isinstance(example.get("message_act_labels"), list) else []
        raw_relations = example.get("relation_labels") if isinstance(example.get("relation_labels"), list) else []
        acts = [label for label in raw_acts if isinstance(label, str)]
        relations = [label for label in raw_relations if isinstance(label, str)]
        if len(acts) != len(raw_acts) or any(label not in MESSAGE_ACT_LABELS for label in acts):
            add("message_act_label", row_number, f"allowed labels: {sorted(MESSAGE_ACT_LABELS)}")
        if len(relations) != len(raw_relations) or any(label not in RELATION_LABELS for label in relations):
            add("relation_label", row_number, f"allowed labels: {sorted(RELATION_LABELS)}")
        if len(acts) != len(set(acts)) or len(relations) != len(set(relations)):
            add("duplicate_label", row_number, "labels must be unique within each label list")
        if "unrelated" in relations and len(relations) != 1:
            add("label_combination", row_number, "unrelated must stand alone among task-pair relation labels")
        if "ambiguous" in relations:
            if example.get("ambiguous") is not True:
                add("ambiguity_mismatch", row_number, "ambiguous relation label requires ambiguous=true")
        if example.get("ambiguous") is True:
            if "ambiguous" not in relations:
                add("ambiguity_mismatch", row_number, "ambiguous=true requires the ambiguous relation label")
            if not str(example.get("ambiguity_reason", "")).strip():
                add("ambiguity_reason", row_number, "ambiguous annotations require a non-empty reason")
        elif str(example.get("ambiguity_reason", "")).strip():
            add("ambiguity_reason", row_number, "must be blank when ambiguous=false")

        if kind == "task_creation":
            if relations:
                add("task_creation_labels", row_number, "task_creation examples use message_act_labels, not task-update relation_labels")
            if example.get("annotation_status") == "complete" and not (set(acts) & MESSAGE_ACT_LABELS):
                add("task_creation_act", row_number, "task_creation requires request or commitment")
        if kind == "task_update" and status == "complete" and not source_ids:
            add("task_update_source", row_number, "task_update requires existing task provenance")

        if status == "draft":
            blank_fields = (
                ("message_act_labels", []), ("relation_labels", []), ("evidence_spans", []),
                ("ambiguous", False), ("ambiguity_reason", ""), ("updated_owner_text", ""),
                ("updated_deadline_text", ""), ("updated_deadline_normalized", None),
                ("annotator_id", ""), ("annotation_notes", ""),
            )
            if kind == "screening_draft":
                blank_fields += (
                    ("task_id", ""), ("task_text", ""), ("task_owner_text", ""),
                    ("task_deadline_text", ""), ("task_deadline_normalized", None),
                    ("task_source_message_ids", []),
                )
            for field, expected in blank_fields:
                if example.get(field) != expected:
                    add("draft_not_blank", row_number, f"{field} must equal {expected!r} in an unlabeled draft")
            if example.get("task_deadline_normalized") is not None:
                add("draft_not_blank", row_number, "task_deadline_normalized must be null in an unlabeled draft")
        elif status == "complete":
            if not str(example.get("annotator_id", "")).strip():
                add("annotator_id", row_number, "completed examples require annotator_id")
            if not acts and not relations:
                add("completed_without_labels", row_number, "completed examples require at least one message-act or relation label")
            if not example.get("evidence_spans"):
                add("completed_without_evidence", row_number, "completed examples require evidence spans")

        owner_literal = str(example.get("updated_owner_text", "")).strip()
        deadline_literal = str(example.get("updated_deadline_text", "")).strip()
        if owner_literal and "reassignment" not in relations:
            add("changed_owner_without_label", row_number, "updated_owner_text requires reassignment")
        if deadline_literal and "deadline_change" not in relations:
            add("changed_deadline_without_label", row_number, "updated_deadline_text requires deadline_change")
        if owner_literal and owner_literal not in str(example.get("update_text", "")):
            add("changed_owner_not_literal", row_number, "updated_owner_text must be a literal expression in update_text")
        if deadline_literal and deadline_literal not in str(example.get("update_text", "")):
            add("changed_deadline_not_literal", row_number, "updated_deadline_text must be a literal expression in update_text")
        for field in NULLABLE_DATE_FIELDS:
            normalized = example.get(field)
            if isinstance(normalized, str) and normalized:
                try:
                    date.fromisoformat(normalized)
                except ValueError:
                    add("normalized_date", row_number, f"{field} must use ISO YYYY-MM-DD")
                literal_field = "task_deadline_text" if field == "task_deadline_normalized" else "updated_deadline_text"
                if not str(example.get(literal_field, "")).strip():
                    add("normalized_date_without_literal", row_number, f"{field} requires its separate literal text field")

        update_id = example.get("update_message_id")
        update_record = messages.get(update_id) if isinstance(update_id, str) else None
        if update_record is None:
            add("unknown_update_message", row_number, str(update_id))
        else:
            if update_record.get("thread_id") != example.get("thread_id"):
                add("update_thread_mismatch", row_number, "update message is not in the declared thread")
            example_thread_id = example.get("thread_id")
            if not isinstance(example_thread_id, str) or example_thread_id not in threads:
                add("unknown_thread", row_number, str(example_thread_id))
            elif update_record.get("source_relative_path") not in threads[example_thread_id]:
                add("update_thread_membership", row_number, "update message is absent from declared thread membership")
            if example.get("update_text") != update_record.get("cleaned_body"):
                add("update_text_mismatch", row_number, "update_text must exactly match the referenced message cleaned_body")

        source_records = []
        for source_id in source_ids:
            source_record = messages.get(source_id)
            if source_record is None:
                add("unknown_task_source_message", row_number, str(source_id))
                continue
            source_records.append((source_id, source_record))
            if source_record.get("thread_id") != example.get("thread_id"):
                add("task_source_thread_mismatch", row_number, f"{source_id} is not in the declared thread")
            example_thread_id = example.get("thread_id")
            if isinstance(example_thread_id, str) and example_thread_id in threads and source_record.get("source_relative_path") not in threads[example_thread_id]:
                add("task_source_thread_membership", row_number, f"{source_id} is absent from declared thread membership")

        if update_record is not None and isinstance(update_record.get("position"), int) and not isinstance(update_record.get("position"), bool):
            update_position = update_record["position"]
            if kind == "screening_draft":
                if source_ids:
                    add("screening_draft_has_task", row_number, "screening_draft must not assert task provenance")
            elif kind == "task_creation":
                if source_ids != [update_id]:
                    add("task_creation_provenance", row_number, "task_creation provenance must be exactly its own update message")
            elif kind == "task_update":
                for source_id, source_record in source_records:
                    source_position = source_record.get("position")
                    if not isinstance(source_position, int) or isinstance(source_position, bool):
                        add("task_source_position", row_number, f"{source_id} has no valid integer position")
                    elif source_position >= update_position:
                        add("task_source_not_earlier", row_number, f"{source_id} must precede update_message_id")
                if update_id in source_ids:
                    add("task_source_is_update", row_number, "task_update provenance cannot be the update message itself")

        evidence_spans = example.get("evidence_spans") if isinstance(example.get("evidence_spans"), list) else []
        labeled_items = set(acts) | set(relations)
        supported_labels: Set[str] = set()
        update_supported_labels: Set[str] = set()
        owner_span_supported = False
        deadline_span_supported = False
        for span_index, span in enumerate(evidence_spans):
            if not isinstance(span, dict):
                add("evidence_span_type", row_number, f"evidence_spans[{span_index}] must be an object")
                continue
            target = span.get("text_field")
            start = span.get("start_char")
            end = span.get("end_char")
            support = span.get("supports_label")
            if not isinstance(target, str) or target not in {"task_text", "update_text"}:
                add("evidence_target", row_number, f"evidence_spans[{span_index}] has invalid text_field")
                continue
            if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
                add("evidence_offsets", row_number, f"evidence_spans[{span_index}] offsets must be integers")
                continue
            text_value = example.get(target)
            if not isinstance(text_value, str) or start < 0 or end <= start or end > len(text_value):
                add("evidence_bounds", row_number, f"evidence_spans[{span_index}] is outside {target}")
            if not isinstance(support, str) or support not in labeled_items:
                add("evidence_label", row_number, f"evidence_spans[{span_index}] must support an assigned label")
            elif isinstance(support, str):
                supported_labels.add(support)
                if target == "update_text":
                    update_supported_labels.add(support)
            if target == "update_text" and isinstance(start, int) and isinstance(end, int) and isinstance(text_value, str) and 0 <= start < end <= len(text_value):
                selected_text = text_value[start:end]
                owner_span_supported |= bool(owner_literal and support == "reassignment" and owner_literal in selected_text)
                deadline_span_supported |= bool(deadline_literal and support == "deadline_change" and deadline_literal in selected_text)
        if example.get("annotation_status") == "complete":
            for label in labeled_items - supported_labels:
                add("label_without_evidence", row_number, f"{label} has no supporting evidence span")
            for label in labeled_items - update_supported_labels:
                add("label_without_update_evidence", row_number, f"{label} has no evidence span in update_text")
        if owner_literal and not owner_span_supported:
            add("changed_owner_without_evidence", row_number, "an updated owner literal must appear inside a reassignment evidence span")
        if deadline_literal and not deadline_span_supported:
            add("changed_deadline_without_evidence", row_number, "an updated deadline literal must appear inside a deadline_change evidence span")

    errors = [issue for issue in issues if issue.get("severity", "error") == "error"]
    return {
        "schema_version": SCHEMA_VERSION,
        "example_count": example_count,
        "draft_count": draft_count,
        "complete_count": complete_count,
        "origin_counts": origin_counts,
        "error_count": len(errors),
        "issues": issues,
    }


def validate_file(annotation_path: Path, messages_path: Path, threads_path: Path) -> Dict[str, Any]:
    messages, threads, reference_issues = load_reference_index(messages_path, threads_path)
    return validate_examples(_read_jsonl(annotation_path), messages, threads, reference_issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", help="Versioned annotation JSONL file")
    parser.add_argument("--messages", required=True, help="Explicit messages.jsonl reference file")
    parser.add_argument("--threads", required=True, help="Explicit threads.jsonl reference file")
    args = parser.parse_args()
    try:
        report = validate_file(Path(args.annotations), Path(args.messages), Path(args.threads))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())