import csv
import json

import pytest

from scripts.create_annotation_pilot import create_pilot
from scripts.validate_annotations import validate_file


def _write_synthetic_thread_source(root):
    root.mkdir()
    sizes = [2, 2, 3, 3, 3, 3, 6, 6, 6, 11, 11, 11, 1, 1, 1]
    messages = []
    threads = []
    for thread_number, size in enumerate(sizes):
        thread_id = f"PILOT-SYN-{thread_number:02d}"
        category = "unlinked" if size == 1 else "subject_only"
        source_paths = []
        message_ids = []
        for position in range(1, size + 1):
            message_id = f"{thread_id}-M{position}"
            source_path = f"synthetic/{thread_id}/{position}"
            source_paths.append(source_path)
            message_ids.append(message_id)
            if position % 4 == 0:
                body = "Please finish the task."
            elif position % 4 == 1:
                body = "The deadline is due Friday."
            elif position % 4 == 2:
                body = "Routine status note."
            else:
                body = "I will send the task."
            cleaned = body
            if position == size and thread_number % 3 == 0:
                body = "Please finish the quoted task."
                cleaned = "Routine status note."
            messages.append({
                "message_id": message_id,
                "thread_id": thread_id,
                "position": position,
                "source_relative_path": source_path,
                "cleaned_body": cleaned,
                "body": body,
                "link_rules": ["subject"] if position > 1 and category == "subject_only" else [],
            })
        threads.append({
            "thread_id": thread_id,
            "message_count": size,
            "source_relative_paths": source_paths,
            "message_ids": message_ids,
            "link_rules": ["subject"] if category == "subject_only" else [],
        })
    (root / "messages.jsonl").write_text("".join(json.dumps(item) + "\n" for item in messages), encoding="utf-8")
    (root / "threads.jsonl").write_text("".join(json.dumps(item) + "\n" for item in threads), encoding="utf-8")


def test_pilot_is_bounded_deterministic_and_has_blank_valid_drafts(tmp_path):
    source = tmp_path / "source"
    _write_synthetic_thread_source(source)
    outputs_a = (tmp_path / "pilot_a.csv", tmp_path / "drafts_a.jsonl", tmp_path / "readme_a.md")
    outputs_b = (tmp_path / "pilot_b.csv", tmp_path / "drafts_b.jsonl", tmp_path / "readme_b.md")

    report_a = create_pilot(source, *outputs_a, seed=1337, target_threads=15, target_messages=40)
    report_b = create_pilot(source, *outputs_b, seed=1337, target_threads=15, target_messages=40)

    assert report_a["selected_threads"] == 15
    assert report_a["selected_messages"] == 40
    assert {"cue_positive", "cue_negative", "cleaning_disagreement"} <= set(report_a["screening_strata"])
    assert outputs_a[0].read_bytes() == outputs_b[0].read_bytes()
    assert outputs_a[1].read_bytes() == outputs_b[1].read_bytes()
    with outputs_a[0].open(encoding="utf-8-sig", newline="") as source_file:
        candidates = list(csv.DictReader(source_file))
    drafts = [json.loads(line) for line in outputs_a[1].read_text(encoding="utf-8").splitlines()]
    assert len(candidates) == len(drafts) == 40
    assert all(row["annotator_actionable"] == "" and row["annotator_update_type"] == "" for row in candidates)
    assert all(draft["example_kind"] == "screening_draft" and draft["task_text"] == "" and not draft["task_source_message_ids"] for draft in drafts)
    validated = validate_file(outputs_a[1], source / "messages.jsonl", source / "threads.jsonl")
    assert validated["error_count"] == 0, validated["issues"]
    with pytest.raises(FileExistsError):
        create_pilot(source, *outputs_a, seed=1337, target_threads=15, target_messages=40)