import sqlite3

from scripts.build_threads import _create_database, _insert_record, normalize_subject, reconstruct_threads


def _record(**overrides):
    record = {
        "message_id": "",
        "in_reply_to": "",
        "references": "",
        "subject": "Project update",
        "date": "Mon, 01 Jan 2024 09:00:00 +0000",
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "cc": "",
        "body": "Please send the updated file.",
        "mailbox": "mail-a",
        "source_relative_path": "mail-a/inbox/1",
    }
    record.update(overrides)
    return record


def test_duplicate_message_id_across_two_mailboxes_keeps_first_and_records_both():
    connection = _create_database(":memory:")
    first = _record(message_id="<same-id>")
    second = _record(message_id="<same-id>", mailbox="mail-b", source_relative_path="mail-b/inbox/2")

    assert _insert_record(connection, first) is True
    assert _insert_record(connection, second) is False
    row = connection.execute("SELECT mailbox, duplicate_mailboxes FROM messages").fetchone()
    assert row[0] == "mail-a"
    assert __import__("json").loads(row[1]) == ["mail-a", "mail-b"]
    connection.close()


def test_subject_normalization_removes_repeated_reply_and_forward_prefixes():
    assert normalize_subject(" RE: Fw:  Project   Update ") == "project update"


def test_in_reply_to_links_messages_by_header():
    first = _record(message_id="<root>", source_relative_path="mail-a/inbox/1", _date_epoch=1704099600)
    reply = _record(message_id="<reply>", in_reply_to="<root>", subject="A different subject", source_relative_path="mail-a/inbox/2", _date_epoch=1704103200)
    annotations, threads = reconstruct_threads([first, reply])

    assert annotations[0]["thread_id"] == annotations[1]["thread_id"]
    assert annotations[0]["link_rules"] == ["header"]
    assert annotations[1]["link_rules"] == ["header"]
    assert threads[0]["message_count"] == 2


def test_subject_linking_merges_within_30_days_but_not_after():
    first = _record(subject="Re: Project update", source_relative_path="mail-a/inbox/1", _date_epoch=1704067200)
    within_limit = _record(subject="FW: project UPDATE", source_relative_path="mail-b/inbox/2", _date_epoch=1706659200)
    outside_limit = _record(subject="Project update", source_relative_path="mail-c/inbox/3", _date_epoch=1709337601)
    annotations, threads = reconstruct_threads([first, within_limit, outside_limit])

    assert annotations[0]["thread_id"] == annotations[1]["thread_id"]
    assert annotations[2]["thread_id"] != annotations[0]["thread_id"]
    assert annotations[1]["link_rules"] == ["subject"]
    assert len(threads) == 2