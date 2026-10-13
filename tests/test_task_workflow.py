"""Synthetic tests for task extraction, SQLite persistence, semantic ranking, and update history."""

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


if __name__ == "__main__":
    test_evidence_offsets_exact_match()
    test_persistence_and_deduplication()
    test_update_history_and_no_thanks_completion()
    test_semantic_ranking_behavior()
    print("All synthetic task workflow tests passed successfully!")
