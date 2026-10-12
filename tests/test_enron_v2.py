from pathlib import Path

from scripts.parse_enron_subset_v2 import clean_body_v2, parse_email_v2
from scripts.build_threads_v2 import classify_duplicates, reconstruct_threads_v2


def test_v2_parser_preserves_body_lines_and_captures_available_headers(tmp_path):
    raw = (
        "Message-ID: <v2-1>\r\n"
        "From: Alice Example <alice@example.com>\r\n"
        "To: Bob Example <bob@example.com>\r\n"
        "Cc: Carol Example <carol@example.com>\r\n"
        "Subject: Review plan\r\n"
        "Date: Mon, 01 Jan 2024 09:00:00 +0000\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "First paragraph.\r\n"
        "\r\n"
        "Second paragraph.\r\n"
    )
    path = tmp_path / "email"
    path.write_bytes(raw.encode("utf-8"))

    result = parse_email_v2(path, "user/inbox/1")

    assert result["status"] == "parsed"
    record = result["record"]
    assert record["cc"] == "Carol Example <carol@example.com>"
    assert record["message_id"] == "<v2-1>"
    assert record["body"] == "First paragraph.\n\nSecond paragraph.\n"
    assert record["header_presence"]["Cc"] is True
    assert record["header_presence"]["In-Reply-To"] is False


def test_forwarded_payload_is_preserved_and_flagged_for_review():
    body = "New note.\n----- Forwarded Message -----\nFrom: sender@example.com\nPlease finish the delivery.\n"
    result = clean_body_v2(body)
    assert "Forwarded Message" in result["cleaned_body"]
    assert "Please finish the delivery." in result["cleaned_body"]
    assert "forwarded_payload_preserved_for_review" in result["cleaning_flags"]
    assert result["cleaning_uncertain"]


def test_line_aware_quoted_reply_tail_is_removed_but_new_text_is_kept():
    body = "Please send the revised table.\nOn Mon, 1 Jan 2024, Alex wrote:\n> old request: send the draft\n"
    result = clean_body_v2(body)
    assert result["cleaned_body"] == "Please send the revised table."
    assert "old request" not in result["cleaned_body"]
    assert "quoted_reply_tail_removed" in result["cleaning_flags"]


def test_embedded_reply_headers_are_preserved_and_marked_uncertain():
    body = "New request: please review.\nFrom: prior@example.com\nSent: Monday\nTo: team@example.com\nSubject: old item\nQuoted content."
    result = clean_body_v2(body)
    assert "New request: please review." in result["cleaned_body"]
    assert "Quoted content." in result["cleaned_body"]
    assert "embedded_message_header_preserved" in result["cleaning_uncertain"]


def test_terminal_signature_is_removed_without_dropping_request():
    body = "Could you review this today?\n-- \nAlex Example\n"
    result = clean_body_v2(body)
    assert result["cleaned_body"] == "Could you review this today?"
    assert "terminal_signature_removed" in result["cleaning_flags"]


def _thread_record(**overrides):
    record = {
        "message_id": "<root>",
        "in_reply_to": "",
        "references": "",
        "subject": "Project delivery plan",
        "date": "Mon, 01 Jan 2024 09:00:00 +0000",
        "from": "alice@example.com",
        "to": "bob@example.com",
        "cc": "",
        "body": "Please send the project delivery plan.",
        "cleaned_body": "Please send the project delivery plan.",
        "mailbox": "mail-a",
        "source_relative_path": "mail-a/inbox/1",
        "_date_epoch": 1704099600,
        "_participants": {"alice@example.com", "bob@example.com"},
    }
    record.update(overrides)
    return record


def test_v2_duplicate_same_message_id_copy_preserves_provenance():
    first = _thread_record()
    second = _thread_record(mailbox="mail-b", source_relative_path="mail-b/inbox/1")

    annotations, stats = classify_duplicates([first, second])

    assert annotations[0]["duplicate_category"] == "same_message_id_copy"
    assert annotations[1]["duplicate_category"] == "same_message_id_copy"
    assert [item["mailbox"] for item in annotations[0]["duplicate_provenance"]] == ["mail-a", "mail-b"]
    assert stats["same_message_id_copy_groups"] == 1


def test_v2_duplicate_conflicting_message_id_is_surfaced():
    first = _thread_record()
    second = _thread_record(body="Different body.", cleaned_body="Different body.", source_relative_path="mail-a/inbox/2")

    annotations, stats = classify_duplicates([first, second])

    assert annotations[0]["duplicate_category"] == "conflicting_message_id"
    assert annotations[1]["duplicate_category"] == "conflicting_message_id"
    assert stats["conflicting_message_id_groups"] == 1


def test_v2_identical_body_different_message_ids_are_not_collapsed():
    first = _thread_record(message_id="<one>")
    second = _thread_record(
        message_id="<two>",
        source_relative_path="mail-a/inbox/2",
        **{"from": "carol@example.com", "to": "dan@example.com"},
    )

    annotations, stats = classify_duplicates([first, second])

    assert annotations[0]["duplicate_category"] == "identical_body_different_message_id_review"
    assert annotations[1]["duplicate_category"] == "identical_body_different_message_id_review"
    assert stats["records_preserved_without_body_only_collapse"] == 2


def test_v2_header_links_are_strong_and_subject_links_are_heuristic():
    root = _thread_record(message_id="<root>", subject="Original topic")
    reply = _thread_record(
        message_id="<reply>",
        in_reply_to="<root>",
        subject="Different topic",
        source_relative_path="mail-a/inbox/2",
        _date_epoch=1704103200,
    )
    heuristic = _thread_record(
        message_id="<heuristic>",
        subject="Project delivery plan",
        source_relative_path="mail-a/inbox/3",
        _date_epoch=1704186000,
    )
    heuristic_prior = _thread_record(
        message_id="<heuristic-prior>",
        subject="Re: Project delivery plan",
        source_relative_path="mail-a/inbox/4",
        _date_epoch=1704182400,
    )

    annotations, threads, _ = reconstruct_threads_v2([root, reply, heuristic_prior, heuristic])

    assert annotations[0]["thread_id"] == annotations[1]["thread_id"]
    assert annotations[0]["thread_confidence"] == "strong_header"
    assert annotations[1]["link_rules"] == ["header"]
    assert annotations[2]["thread_id"] == annotations[3]["thread_id"]
    assert annotations[2]["thread_confidence"] == "heuristic_subject"
    assert any(thread["thread_confidence"] == "strong_header" for thread in threads)
    assert any(thread["thread_confidence"] == "heuristic_subject" for thread in threads)


def test_v2_subject_linking_requires_participant_overlap_and_short_span():
    first = _thread_record(message_id="<one>", _date_epoch=1704099600)
    no_overlap = _thread_record(
        message_id="<two>",
        source_relative_path="mail-a/inbox/2",
        _date_epoch=1704186000,
        **{"from": "carol@example.com", "to": "dan@example.com"},
    )
    too_late = _thread_record(
        message_id="<three>",
        source_relative_path="mail-a/inbox/3",
        _date_epoch=1706086800,
    )

    annotations, threads, _ = reconstruct_threads_v2([first, no_overlap, too_late])

    assert len({item["thread_id"] for item in annotations}) == 3
    assert all(thread["thread_confidence"] == "singleton" for thread in threads)
