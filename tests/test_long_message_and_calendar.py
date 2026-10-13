"""Synthetic tests for sliding-window long message inference and legacy Google Calendar sync.

NOTE: Calendar tests here cover legacy components pending user deletion approval.
Sliding-window long message inference remains active.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.task_store import TaskStore
from followthrough_core.calendar_sync import CalendarSyncManager
from followthrough_core.long_message_classifier import classify_long_message


# ---------------------------------------------------------------------------
# 1. Tests for Long Message Inference & Coverage
# ---------------------------------------------------------------------------

def test_long_message_sliding_window_offsets_and_coverage():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    ckpt_path = Path("results/action_detector_tuned/distilbert/checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    device = torch.device("cpu")

    # Construct a synthetic message longer than 128 tokens
    # Section 1: Background padding
    padding_sentences = [
        f"This is background paragraph number {i} discussing general operations and administrative procedures."
        for i in range(12)
    ]
    # Section 2: Explicit request far beyond the 128-token mark
    action_sentence = "Please submit the finalized financial risk report by Friday afternoon."
    full_message = " ".join(padding_sentences) + " " + action_sentence

    tokens = tokenizer.encode(full_message, truncation=False)
    assert len(tokens) > 130, f"Expected >130 tokens, got {len(tokens)}"

    # Run sliding window classifier
    res = classify_long_message(
        full_message,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_window_length=128,
        stride=64,
    )

    assert res["num_windows"] >= 2
    assert res["total_tokens"] == len(tokenizer.encode(full_message, add_special_tokens=False))

    # Verify per-window special tokens, offsets, and original text slicing
    for w in res["windows"]:
        assert w["token_count"] <= 128
        # Check source text exact character offset match
        assert full_message[w["char_start"]:w["char_end"]] == w["source_text"]
        assert 0.0 <= w["req_score"] <= 1.0
        assert 0.0 <= w["com_score"] <= 1.0

    # Ensure last window catches the action sentence at the end
    last_window = res["windows"][-1]
    assert "financial risk report" in last_window["source_text"]

    # Verify that task extraction extracts from the FULL message independent of 128 truncation
    candidates = extract_candidate_tasks(full_message)
    assert len(candidates) >= 1
    found_task = any("financial risk report" in c["action_description"] for c in candidates)
    assert found_task is True
    # Character offset matches exact source text
    cand = candidates[0]
    assert full_message[cand["start_offset"]:cand["end_offset"]] == cand["evidence_text"]


# ---------------------------------------------------------------------------
# 2. Tests for SQLite Additive Migration & Calendar Linkage
# ---------------------------------------------------------------------------

def test_database_additive_migration_and_calendar_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks_migration_test.db"
        store = TaskStore(db_path)

        task_data = {
            "source_text": "Please review the quarterly audit by next Wednesday.",
            "action_description": "Please review the quarterly audit by next Wednesday.",
            "owner": "Sarah",
            "deadline": "next Wednesday",
            "evidence_text": "Please review the quarterly audit by next Wednesday.",
            "start_offset": 0,
            "end_offset": 53,
        }
        ok, task_id = store.save_task(task_data)
        assert ok is True

        # Check initial calendar fields are None
        task = store.get_task(task_id)
        assert "calendar_id" in task
        assert "calendar_event_id" in task
        assert task["calendar_id"] is None
        assert task["calendar_event_id"] is None

        # Link calendar event
        up_ok = store.update_task_calendar(task_id, "primary", "gcal_event_12345")
        assert up_ok is True

        updated_task = store.get_task(task_id)
        assert updated_task["calendar_id"] == "primary"
        assert updated_task["calendar_event_id"] == "gcal_event_12345"

        # Clear calendar event
        clr_ok = store.clear_task_calendar(task_id)
        assert clr_ok is True
        cleared_task = store.get_task(task_id)
        assert cleared_task["calendar_id"] is None
        assert cleared_task["calendar_event_id"] is None


# ---------------------------------------------------------------------------
# 3. Tests for Google Calendar Payload Validation & Mocked Sync
# ---------------------------------------------------------------------------

def test_calendar_payload_rules():
    # Rule 1: Date-only deadline must remain all-day
    p_allday = CalendarSyncManager.prepare_event_payload(
        title="Quarterly Review",
        description="Task task-123 evidence",
        start_date="2026-10-16",
        is_all_day=True,
    )
    assert "date" in p_allday["start"]
    assert p_allday["start"]["date"] == "2026-10-16"
    assert "dateTime" not in p_allday["start"]

    # Rule 2: Timed event requires explicit start and end time; cannot invent duration
    try:
        CalendarSyncManager.prepare_event_payload(
            title="Quarterly Review",
            description="Task task-123",
            start_date="2026-10-16",
            is_all_day=False,
            start_time="14:00",
            end_time="",  # missing end time
        )
        assert False, "Should have raised ValueError for missing end time"
    except ValueError as e:
        assert "end time / duration" in str(e)

    # Valid timed event
    p_timed = CalendarSyncManager.prepare_event_payload(
        title="Quarterly Review",
        description="Task task-123",
        start_date="2026-10-16",
        is_all_day=False,
        start_time="14:00",
        end_time="15:00",
        timezone="America/Chicago",
    )
    assert p_timed["start"]["dateTime"] == "2026-10-16T14:00:00"
    assert p_timed["start"]["timeZone"] == "America/Chicago"
    assert p_timed["end"]["dateTime"] == "2026-10-16T15:00:00"


def test_calendar_mocked_sync_and_idempotent_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        token_path = Path(tmpdir) / "token.json"
        creds_path = Path(tmpdir) / "credentials.json"
        mgr = CalendarSyncManager(credentials_path=creds_path, token_path=token_path)

        mock_service = MagicMock()
        mock_events = MagicMock()
        mock_service.events.return_value = mock_events

        # 1. First sync -> insert
        mock_events.insert.return_value.execute.return_value = {"id": "evt_abc123"}
        payload = {"summary": "Audit Task", "start": {"date": "2026-10-20"}, "end": {"date": "2026-10-20"}}

        ok, event_id, _ = mgr.sync_event("primary", payload, existing_event_id=None, service=mock_service)
        assert ok is True
        assert event_id == "evt_abc123"
        mock_events.insert.assert_called_once_with(calendarId="primary", body=payload)

        # 2. Second sync with existing_event_id -> update existing event (no duplicate creation)
        mock_events.update.return_value.execute.return_value = {"id": "evt_abc123"}
        updated_payload = {"summary": "Audit Task (Rescheduled)", "start": {"date": "2026-10-22"}, "end": {"date": "2026-10-22"}}

        ok_up, up_event_id, _ = mgr.sync_event("primary", updated_payload, existing_event_id="evt_abc123", service=mock_service)
        assert ok_up is True
        assert up_event_id == "evt_abc123"
        mock_events.update.assert_called_once_with(calendarId="primary", eventId="evt_abc123", body=updated_payload)

        # 3. Deletion with mock
        mock_events.delete.return_value.execute.return_value = {}
        del_ok, _ = mgr.delete_event("primary", "evt_abc123", service=mock_service)
        assert del_ok is True
        mock_events.delete.assert_called_once_with(calendarId="primary", eventId="evt_abc123")


if __name__ == "__main__":
    print("Running Long Message & Calendar Sync Tests...")
    test_long_message_sliding_window_offsets_and_coverage()
    test_database_additive_migration_and_calendar_columns()
    test_calendar_payload_rules()
    test_calendar_mocked_sync_and_idempotent_update()
    print("All Long Message & Google Calendar tests passed successfully!")
