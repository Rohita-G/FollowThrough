import csv
import hashlib
import json

import pytest

from scripts.select_annotation_candidates import ANNOTATOR_COLUMNS, CSV_COLUMNS, cue_families, select_candidates


@pytest.mark.parametrize(
    ("text", "family"),
    [
        ("Could you send this?", "request_or_commitment"),
        ("The deadline is due Friday.", "deadline_change"),
        ("Please cancel the meeting.", "cancellation"),
        ("I have completed the report.", "completion"),
        ("Please reassign this task to Lee.", "reassignment"),
    ],
)
def test_each_heuristic_family_is_detected(text, family):
    assert family in cue_families(text)


def _write_thread_dir(root, messages, threads):
    root.mkdir(parents=True)
    (root / "messages.jsonl").write_text("".join(json.dumps(row) + "\n" for row in messages), encoding="utf-8")
    (root / "threads.jsonl").write_text("".join(json.dumps(row) + "\n" for row in threads), encoding="utf-8")


def _message(position, text, thread_id="t1"):
    return {
        "message_id": f"<m{position}>",
        "thread_id": thread_id,
        "position": position,
        "source_relative_path": f"mail-a/inbox/{position}",
        "cleaned_body": text,
        "link_rules": ["subject"] if position > 1 else [],
    }


def _thread(thread_id, count):
    return {"thread_id": thread_id, "message_count": count}


def test_selection_is_deterministic_and_annotator_fields_are_blank(tmp_path):
    source = tmp_path / "threads"
    messages = [_message(1, "Can you send the draft?"), _message(2, "I will send it by Friday.")]
    _write_thread_dir(source, messages, [_thread("t1", 2)])
    input_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    first_stats = select_candidates(source, first, max_candidates=20)
    second_stats = select_candidates(source, second, max_candidates=20)

    assert first.read_bytes() == second.read_bytes()
    with first.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert first_stats["candidates_written"] == 2
    assert second_stats["candidates_written"] == 2
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert all(all(row[column] == "" for column in ANNOTATOR_COLUMNS) for row in rows)
    assert all(row["needs_thread_review"] == "true" for row in rows)
    assert rows[1]["previous_message_id"] == "<m1>"
    assert "task reference" in first_stats["previous_message_id_meaning"]
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()} == input_hashes


def test_empty_candidates_still_write_header(tmp_path):
    source = tmp_path / "threads"
    _write_thread_dir(source, [_message(1, "Hello there.")], [_thread("t1", 1)])
    output = tmp_path / "empty.csv"
    stats = select_candidates(source, output)
    with output.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(CSV_COLUMNS)
        assert list(reader) == []
    assert stats["candidates_written"] == 0


def test_zero_candidate_limit_is_valid_and_keeps_header(tmp_path):
    source = tmp_path / "threads"
    _write_thread_dir(source, [_message(1, "Please send the report.")], [_thread("t1", 1)])
    output = tmp_path / "limited.csv"
    stats = select_candidates(source, output, max_candidates=0)
    assert stats["candidates_written"] == 0
    with output.open(encoding="utf-8-sig", newline="") as file:
        assert csv.DictReader(file).fieldnames == list(CSV_COLUMNS)