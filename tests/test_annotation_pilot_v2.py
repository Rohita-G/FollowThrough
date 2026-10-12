"""Focused unit tests for v2 candidate screening, copy grouping, and pilot generation."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.evidence_offset_helper import derive_evidence_span, find_occurrences
from scripts.select_annotation_pilot_v2 import (
    DISPLAY_REPRESENTATIVE_RULE,
    count_distinct_nonempty_raw_bodies,
    group_messages_conservatively,
    normalize_sender,
    normalize_subject,
    normalize_timestamp,
    select_and_generate_pilot,
)
from scripts.validate_annotations import validate_file


def _make_msg(
    thread_id: str = "thread-1",
    position: int = 1,
    message_id: str = "<msg-1@enron>",
    sender: str = "alice@enron.com",
    date: str = "Mon, 14 May 2001 10:00:00 -0700",
    subject: str = "Project Update",
    body: str = "Here is the plan.",
    cleaned_body: str = "Here is the plan.",
    source_path: str = "alice/inbox/1.",
) -> Dict[str, Any]:
    return {
        "thread_id": thread_id,
        "position": position,
        "message_id": message_id,
        "from": sender,
        "date": date,
        "subject": subject,
        "body": body,
        "cleaned_body": cleaned_body,
        "source_relative_path": source_path,
        "raw_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest() if body else "",
        "cleaning_flags": [],
        "cleaning_uncertain": False,
        "link_rules": ["subject"],
    }


class TestCandidateScreening:
    def test_exclude_all_copy_threads(self):
        """Threads where all messages share identical raw body must be excluded (count == 1)."""
        msgs = [
            _make_msg(position=1, source_path="folder_a/1.", body="Identical text"),
            _make_msg(position=2, source_path="folder_b/1.", body="Identical text"),
            _make_msg(position=3, source_path="folder_c/1.", body="Identical text"),
        ]
        assert count_distinct_nonempty_raw_bodies(msgs) == 1

    def test_retain_threads_with_different_substantive_bodies(self):
        """Threads with >= 2 distinct nonempty bodies must be retained."""
        msgs = [
            _make_msg(position=1, body="First turn request"),
            _make_msg(position=2, body="Second turn response"),
        ]
        assert count_distinct_nonempty_raw_bodies(msgs) == 2

    def test_empty_body_handling(self):
        """Whitespace or empty bodies must not count toward distinct nonempty bodies."""
        msgs = [
            _make_msg(position=1, body="Only valid turn"),
            _make_msg(position=2, body="   \n\t  "),
            _make_msg(position=3, body=""),
        ]
        assert count_distinct_nonempty_raw_bodies(msgs) == 1


class TestConservativeCopyGrouping:
    def test_conservative_copy_grouping_identical(self):
        """Identical body, same sender, same normalized date, and compatible subject are grouped."""
        msgs = [
            _make_msg(position=1, source_path="user/all_documents/10.", message_id="<id-1>"),
            _make_msg(position=2, source_path="user/sent/10.", message_id="<id-2>"),
        ]
        groups = group_messages_conservatively(msgs)
        assert len(groups) == 1
        g = groups[0]
        assert g["is_copy_group"] is True
        assert g["copy_count"] == 2
        assert g["representative_position"] == 1
        assert g["representative_message_id"] == "<id-1>"
        assert g["representative_selection_rule"] == DISPLAY_REPRESENTATIVE_RULE
        assert g["member_message_ids"] == ["<id-1>", "<id-2>"]
        assert g["member_source_relative_paths"] == ["user/all_documents/10.", "user/sent/10."]
        assert "source_relative_path" in g["metadata_differences"]

    def test_distinct_senders_remain_separate(self):
        """Identical body but different senders must NEVER be collapsed."""
        msgs = [
            _make_msg(position=1, sender="alice@enron.com", body="Thanks."),
            _make_msg(position=2, sender="bob@enron.com", body="Thanks."),
        ]
        groups = group_messages_conservatively(msgs)
        assert len(groups) == 2
        assert groups[0]["representative_message"]["from"] == "alice@enron.com"
        assert groups[1]["representative_message"]["from"] == "bob@enron.com"

    def test_distinct_dates_remain_separate(self):
        """Identical body and same sender but different dates must NEVER be collapsed."""
        msgs = [
            _make_msg(position=1, date="Mon, 14 May 2001 10:00:00 -0700", body="Reminder"),
            _make_msg(position=2, date="Wed, 16 May 2001 10:00:00 -0700", body="Reminder"),
        ]
        groups = group_messages_conservatively(msgs)
        assert len(groups) == 2
        assert groups[0]["representative_message"]["date"] != groups[1]["representative_message"]["date"]

    def test_empty_bodies_not_grouped_as_copies(self):
        """Empty bodies must not be grouped together as substantive copies."""
        msgs = [
            _make_msg(position=1, body=""),
            _make_msg(position=2, body=""),
        ]
        groups = group_messages_conservatively(msgs)
        assert len(groups) == 2


class TestPilotGenerationAndImmutability:
    def test_deterministic_selection_and_blank_labels(self):
        """Generating pilot must produce deterministic outputs with strictly blank labels."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            thread_dir = tmp_path / "threads_dir"
            thread_dir.mkdir()

            msgs = [
                _make_msg(thread_id="t-1", position=1, message_id="<msg-1@enron>", sender="a@e.com",
                          date="Mon, 14 May 2001 10:00:00 -0700", source_path="alice/inbox/1.",
                          body="Please send report.", cleaned_body="Please send report."),
                _make_msg(thread_id="t-1", position=2, message_id="<msg-2@enron>", sender="b@e.com",
                          date="Mon, 14 May 2001 11:00:00 -0700", source_path="alice/inbox/2.",
                          body="I will send it tomorrow.", cleaned_body="I will send it tomorrow."),
            ]
            original_msgs = copy.deepcopy(msgs)

            (thread_dir / "messages.jsonl").write_text(
                "\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8"
            )
            thread_meta = {
                "thread_id": "t-1",
                "message_count": 2,
                "source_relative_paths": ["alice/inbox/1.", "alice/inbox/2."],
                "link_rules": ["subject"],
                "start_date": "2001-05-14",
                "end_date": "2001-05-14",
            }
            (thread_dir / "threads.jsonl").write_text(json.dumps(thread_meta) + "\n", encoding="utf-8")

            out_csv = tmp_path / "screening.csv"
            out_drafts = tmp_path / "drafts.jsonl"
            out_disp_jsonl = tmp_path / "display.jsonl"
            out_disp_md = tmp_path / "display.md"
            out_readme = tmp_path / "README.md"

            stats = select_and_generate_pilot(
                thread_output_dir=thread_dir,
                output_csv=out_csv,
                output_drafts=out_drafts,
                output_display_jsonl=out_disp_jsonl,
                output_display_md=out_disp_md,
                output_readme=out_readme,
                selected_thread_ids=["t-1"],
                seed=42,
            )

            assert stats["selected_threads"] == 1
            assert stats["distinct_conservative_groups_selected"] == 2
            assert stats["labels_created"] is False

            # Input immutability check
            current_msgs = [json.loads(line) for line in (thread_dir / "messages.jsonl").read_text().strip().split("\n")]
            assert current_msgs == original_msgs

            # Drafts validation
            draft_lines = [json.loads(line) for line in out_drafts.read_text().strip().split("\n")]
            for draft in draft_lines:
                assert draft["schema_version"] == "1.0"
                assert draft["annotation_status"] == "draft"
                assert draft["example_kind"] == "screening_draft"
                assert draft["task_id"] == ""
                assert draft["task_text"] == ""
                assert draft["task_source_message_ids"] == []
                assert draft["message_act_labels"] == []
                assert draft["relation_labels"] == []
                assert draft["evidence_spans"] == []
                assert draft["ambiguous"] is False
                assert draft["annotator_id"] == ""

            # Check that validate_annotations_file passes without errors
            val_results = validate_file(
                annotation_path=out_drafts,
                messages_path=thread_dir / "messages.jsonl",
                threads_path=thread_dir / "threads.jsonl",
            )
            assert val_results["error_count"] == 0
            assert val_results["draft_count"] == 2


class TestEvidenceOffsetHelper:
    def test_derive_evidence_span_single_match(self):
        text = "Please submit the invoice by Friday."
        span = derive_evidence_span(text, "invoice by Friday", source="update_text")
        assert span["start"] == 18
        assert span["end"] == 35
        assert text[span["start"]:span["end"]] == "invoice by Friday"
        assert span["source"] == "update_text"

    def test_derive_evidence_span_multiple_occurrences(self):
        text = "Invoice for team A. Invoice for team B."
        # Omitting occurrence must raise ValueError
        with pytest.raises(ValueError, match="Please specify an explicit --occurrence"):
            derive_evidence_span(text, "Invoice", occurrence=None)

        # Occurrence 1
        span1 = derive_evidence_span(text, "Invoice", occurrence=1)
        assert span1["start"] == 0
        assert span1["end"] == 7

        # Occurrence 2
        span2 = derive_evidence_span(text, "Invoice", occurrence=2)
        assert span2["start"] == 20
        assert span2["end"] == 27
        assert text[span2["start"]:span2["end"]] == "Invoice"

    def test_derive_evidence_span_not_found(self):
        with pytest.raises(ValueError, match="Substring not found"):
            derive_evidence_span("Short text", "Nonexistent phrase")
