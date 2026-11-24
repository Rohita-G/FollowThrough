"""Synthetic tests for task extraction, SQLite persistence, semantic ranking, and update history."""

import datetime
from pathlib import Path
import tempfile

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.task_store import TaskStore
from followthrough_core.followup_linker import FollowupLinker


def test_evidence_offsets_exact_match():
    text = "Please submit the quarterly risk report by Friday afternoon. Thanks."
    candidates = extract_candidate_tasks(text)
    assert len(candidates) >= 1
    c = candidates[0]
    # Verify exact character slice
    assert text[c["start_offset"]:c["end_offset"]] == c["evidence_text"]
    assert "Please submit" in c["evidence_text"]
    assert c["deadline"] == "by Friday"


def test_persistence_and_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tasks.db"
        store = TaskStore(db_path)

        task_data = {
            "source_text": "Please prepare the presentation for tomorrow.",
            "action_description": "Please prepare the presentation for tomorrow.",
            "owner": "Shirley",
            "deadline": "tomorrow",
            "evidence_text": "Please prepare the presentation for tomorrow.",
            "start_offset": 0,
            "end_offset": 46,
        }

        # First save succeeds
        ok1, task_id1 = store.save_task(task_data)
        assert ok1 is True
        assert task_id1.startswith("task-")

        # Duplicate save is prevented
        ok2, msg2 = store.save_task(task_data)
        assert ok2 is False
        assert "Duplicate task" in msg2

        # Check retrieval
        tasks = store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["action_description"] == task_data["action_description"]


def test_update_history_and_no_thanks_completion():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tasks.db"
        store = TaskStore(db_path)

        task_data = {
            "source_text": "Please review the audit draft by Thursday.",
            "action_description": "Please review the audit draft by Thursday.",
            "owner": "Mike",
            "deadline": "by Thursday",
            "evidence_text": "Please review the audit draft by Thursday.",
            "start_offset": 0,
            "end_offset": 42,
        }
        ok, task_id = store.save_task(task_data)
        assert ok is True

        # Ensure "Thanks" is NOT detected as completion
        proposals_thanks = FollowupLinker.detect_updates("Thanks for the quick turnaround!")
        assert len(proposals_thanks) == 0

        # Detect deadline revision
        followup_text = "We have postponed to Friday afternoon."
        proposals = FollowupLinker.detect_updates(followup_text)
        assert len(proposals) == 1
        p = proposals[0]
        assert p["update_type"] == "deadline_revised"
        assert p["new_value"] == "Friday"

        # Apply update
        up_ok = store.apply_update(
            task_id,
            p["update_type"],
            p["new_value"],
            p["evidence_text"],
            followup_text,
        )
        assert up_ok is True

        # Verify task and history
        updated_task = store.get_task(task_id)
        assert updated_task["deadline"] == "Friday"

        history = store.get_history(task_id)
        assert len(history) == 2  # created + deadline_revised
        assert history[1]["update_type"] == "deadline_revised"
        assert history[1]["old_value"] == "by Thursday"
        assert history[1]["new_value"] == "Friday"
        assert history[1]["evidence_text"] == "postponed to Friday"


def test_semantic_ranking_behavior():
    linker = FollowupLinker()
    tasks = [
        {"task_id": "1", "action_description": "Prepare the financial quarterly budget report"},
        {"task_id": "2", "action_description": "Arrange phone interview with engineering candidate"},
        {"task_id": "3", "action_description": "Send birthday card to team member"},
    ]

    followup = "Any updates on submitting the quarterly budget numbers?"
    ranked = linker.rank_candidates(followup, tasks, top_k=3)

    assert len(ranked) == 3
    # Top candidate should be the budget task
    assert ranked[0]["task"]["task_id"] == "1"
    assert ranked[0]["similarity_score"] > ranked[1]["similarity_score"]
    assert ranked[0]["similarity_score"] > 0.4


def test_update_task_details_audit_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tasks.db"
        store = TaskStore(db_path)

        task_data = {
            "source_text": "Please prepare the presentation for tomorrow.",
            "action_description": "Prepare the presentation",
            "owner": "Shirley",
            "deadline": "2026-10-18 (Sun)",
            "evidence_text": "Please prepare the presentation for tomorrow.",
            "start_offset": 0,
            "end_offset": 46,
        }
        ok, task_id = store.save_task(task_data)
        assert ok is True

        # Manually update task details
        update_ok = store.update_task_details(
            task_id,
            action_description="Prepare the executive board presentation",
            owner="David",
            deadline="2026-10-20 (Tue)",
            edit_reason="Adjusted scope and assignee",
        )
        assert update_ok is True

        # Verify task values
        updated = store.get_task(task_id)
        assert updated["action_description"] == "Prepare the executive board presentation"
        assert updated["owner"] == "David"
        assert updated["deadline"] == "2026-10-20 (Tue)"

        # Verify audit history
        hist = store.get_history(task_id)
        assert len(hist) == 2  # created + task_edited
        edit_entry = hist[1]
        assert edit_entry["update_type"] == "task_edited"
        assert "due: '2026-10-18 (Sun)'" in edit_entry["old_value"]
        assert "due: '2026-10-20 (Tue)'" in edit_entry["new_value"]
        assert edit_entry["evidence_text"] == "Adjusted scope and assignee"


def test_disclaimer_stripping_and_header_date():
    text = (
        "Date: Monday, Oct 19, 2026 09:00 AM\n"
        "From: Alice <alice@example.com>\n"
        "To: Bob <bob@example.com>\n"
        "Subject: Project Kickoff\n\n"
        "Hi Bob,\n\n"
        "Please send over the revised wireframes by Wednesday afternoon.\n\n"
        "CONFIDENTIALITY NOTICE: This transmission is confidential and intended solely for the recipient. "
        "If you have received this message in error, please notify the sender immediately and delete it.\n"
        "Please consider the environment before printing this email."
    )
    candidates = extract_candidate_tasks(text)
    # Exactly 1 real task should be extracted: "Please send over the revised wireframes by Wednesday afternoon."
    # The confidentiality disclaimer and environment notice must NOT be extracted as candidate tasks.
    assert len(candidates) == 1
    c = candidates[0]
    assert "2026-10-21" in c["deadline_normalized"]  # Anchored to header date Oct 19 (Wednesday is Oct 21)
    assert "Wednesday" in c["deadline_literal"]
    assert c["owner"].lower() == "bob"


def test_chat_and_instant_message_extraction():
    # Chat message with @mentions, shorthand leads (can u, plz), and colon speaker prefixes
    chat_log = (
        "@kevin: can you review the pull request #402 before 3 PM today?\n"
        "alex: I will deploy the hotfix patch to staging by tonight.\n"
        "@sarah please verify the staging endpoints by EOD.\n"
        "dan: let's jump on a sync tomorrow at 11am\n"
        "Hey Alex, plz send over the pitch deck by tomorrow afternoon."
    )
    candidates = extract_candidate_tasks(chat_log, reference_date=datetime.date(2026, 10, 17))
    assert len(candidates) == 5

    # Check Kevin's request
    c0 = candidates[0]
    assert c0["owner"].lower() == "kevin"
    assert "pull request" in c0["action_description"].lower()
    assert "15:00" in c0["deadline_normalized"]

    # Check Alex's commitment
    c1 = candidates[1]
    assert c1["owner"].lower() == "alex"
    assert c1["task_type"] == "Commitment"
    assert "staging" in c1["action_description"].lower()

    # Check Sarah's request
    c2 = candidates[2]
    assert c2["owner"].lower() == "sarah"
    assert "staging endpoints" in c2["action_description"].lower()

    # Check Dan's sync request
    c3 = candidates[3]
    assert c3["owner"].lower() == "dan"
    assert "sync" in c3["action_description"].lower()

    # Check Alex's mobile text
    c4 = candidates[4]
    assert c4["owner"].lower() == "alex"
    assert "pitch deck" in c4["action_description"].lower()


def test_task_deletion_and_ledger_purge():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tasks.db"
        store = TaskStore(db_path)

        # Create 3 tasks
        store.save_task({
            "source_text": "Task 1",
            "action_description": "Task 1 action",
            "owner": "Alice",
            "deadline": "tomorrow",
        })
        _, tid2 = store.save_task({
            "source_text": "Task 2",
            "action_description": "Task 2 action",
            "owner": "Bob",
            "deadline": "tomorrow",
        })
        _, tid3 = store.save_task({
            "source_text": "Task 3",
            "action_description": "Task 3 action",
            "owner": "Charlie",
            "deadline": "tomorrow",
        })

        assert len(store.list_tasks()) == 3

        # Mark task 2 completed and task 3 cancelled
        store.apply_update(tid2, "completed", "completed", "done", "finished")
        store.apply_update(tid3, "cancelled", "cancelled", "dropped", "not needed")

        # Purge completed and cancelled tasks
        purged = store.clear_completed_and_cancelled()
        assert purged == 2

        remaining = store.list_tasks()
        assert len(remaining) == 1
        assert remaining[0]["owner"] == "Alice"

        # Delete single task
        del_ok = store.delete_task(remaining[0]["task_id"])
        assert del_ok is True
        assert len(store.list_tasks()) == 0


if __name__ == "__main__":
    test_evidence_offsets_exact_match()
    test_persistence_and_deduplication()
    test_update_history_and_no_thanks_completion()
    test_semantic_ranking_behavior()
    test_update_task_details_audit_log()
    test_disclaimer_stripping_and_header_date()
    test_chat_and_instant_message_extraction()
    test_task_deletion_and_ledger_purge()
    print("All synthetic task workflow tests passed successfully!")


