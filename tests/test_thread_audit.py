import hashlib
import json

from scripts.audit_threads import audit_directory, audit_records


def _message(**overrides):
    record = {
        "message_id": "<m1>",
        "in_reply_to": "",
        "references": "",
        "subject": "Update",
        "date": "Mon, 01 Jan 2024 09:00:00 +0000",
        "from": "a@example.com",
        "to": "b@example.com",
        "cc": "",
        "body": "Please send the document.",
        "cleaned_body": "Please send the document.",
        "mailbox": "mail-a",
        "source_relative_path": "mail-a/inbox/1",
        "duplicate_mailboxes": [],
        "thread_id": "t1",
        "position": 1,
        "link_rules": [],
    }
    record.update(overrides)
    return record


def _thread(**overrides):
    record = {
        "thread_id": "t1",
        "message_count": 1,
        "source_relative_paths": ["mail-a/inbox/1"],
        "message_ids": ["<m1>"],
        "link_rules": [],
        "start_date": "Mon, 01 Jan 2024 09:00:00 +0000",
        "end_date": "Mon, 01 Jan 2024 09:00:00 +0000",
        "action_language_candidate": True,
    }
    record.update(overrides)
    return record


def _codes(issues):
    return {item["code"] for item in issues}


def test_valid_input_reconciles_and_reports_candidate_message():
    stats, issues = audit_records([_message()], [_thread()])
    assert stats["message_count"] == 1
    assert stats["thread_record_count"] == 1
    assert stats["unique_builder_message_identifiers"] == 1
    assert stats["candidate_messages_with_task_update_cues"] == 1
    assert stats["audit_error_count"] == 0
    assert "semantic_thread_correctness_proven" in stats and not stats["semantic_thread_correctness_proven"]
    assert not [item for item in issues if item["severity"] == "error"]


def test_broken_declared_thread_membership_is_reported():
    stats, issues = audit_records([_message()], [_thread(source_relative_paths=["mail-a/inbox/wrong"])])
    assert stats["audit_error_count"] >= 1
    assert "thread_membership_mismatch" in _codes(issues)


def test_duplicate_and_noncontiguous_positions_are_reported():
    first = _message()
    second = _message(message_id="<m2>", source_relative_path="mail-a/inbox/2", position=3)
    thread = _thread(message_count=2, source_relative_paths=["mail-a/inbox/1", "mail-a/inbox/2"], message_ids=["<m1>", "<m2>"])
    _, issues = audit_records([first, second], [thread])
    assert "thread_positions_invalid" in _codes(issues)

    second["position"] = 1
    _, issues = audit_records([first, second], [thread])
    assert "thread_positions_duplicate" in _codes(issues)


def test_missing_and_malformed_dates_are_counted_and_empty_bodies_warn():
    missing = _message(date="", message_id="<missing-date>", source_relative_path="mail-a/inbox/2", position=2, cleaned_body="")
    malformed = _message(date="not a date", message_id="<bad-date>", source_relative_path="mail-a/inbox/3", position=3)
    thread = _thread(
        message_count=3,
        source_relative_paths=["mail-a/inbox/1", "mail-a/inbox/2", "mail-a/inbox/3"],
        message_ids=["<m1>", "<missing-date>", "<bad-date>"],
    )
    stats, issues = audit_records([_message(), missing, malformed], [thread])
    assert stats["missing_dates"] == 1
    assert stats["unparseable_dates"] == 1
    assert stats["empty_cleaned_bodies"] == 1
    assert {"unparseable_date", "empty_cleaned_body"} <= _codes(issues)


def test_audit_does_not_modify_input_files(tmp_path):
    source = tmp_path / "threads"
    source.mkdir()
    messages = source / "messages.jsonl"
    threads = source / "threads.jsonl"
    messages.write_text(json.dumps(_message()) + "\n", encoding="utf-8")
    threads.write_text(json.dumps(_thread()) + "\n", encoding="utf-8")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (messages, threads)}

    audit_directory(source, tmp_path / "audit.json", tmp_path / "issues.jsonl")

    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (messages, threads)} == before