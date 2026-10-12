#!/usr/bin/env python3
"""End-to-end verification of the FollowThrough task extraction and follow-up linking workflow."""

import tempfile
from pathlib import Path

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.task_store import TaskStore
from followthrough_core.followup_linker import FollowupLinker


def run_e2e_verification():
    print("=" * 60)
    print("RUNNING END-TO-END WORKFLOW VERIFICATION")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "e2e_test_tasks.db"
        store = TaskStore(db_path)
        linker = FollowupLinker()

        # Step 1: Create and confirm a report-submission task from a message
        initial_message = (
            "Please submit the quarterly risk report by Friday afternoon. Shirley will assist."
        )
        print(f"\n1. Source Message:\n   \"{initial_message}\"")

        candidates = extract_candidate_tasks(initial_message)
        assert len(candidates) >= 1, "Expected at least 1 candidate task extracted"
        task_cand = candidates[0]

        print(f"\n   Extracted candidate task:")
        print(f"   - Action: {task_cand['action_description']}")
        print(f"   - Owner: {task_cand['owner']}")
        print(f"   - Deadline: {task_cand['deadline']}")
        print(f"   - Evidence: \"{task_cand['evidence_text']}\" (offsets: {task_cand['start_offset']}:{task_cand['end_offset']})")
        print(f"   - Uncertainty Flags: {task_cand['uncertainty_flags']}")

        # Verify exact offset match against original text
        assert initial_message[task_cand['start_offset']:task_cand['end_offset']] == task_cand['evidence_text']

        # Confirm and save to SQLite
        task_data = {
            "source_text": initial_message,
            "action_description": task_cand["action_description"],
            "owner": task_cand["owner"],
            "deadline": task_cand["deadline"],
            "evidence_text": task_cand["evidence_text"],
            "start_offset": task_cand["start_offset"],
            "end_offset": task_cand["end_offset"],
        }
        saved_ok, task_id = store.save_task(task_data)
        assert saved_ok, f"Failed to save task: {task_id}"
        print(f"\n   [Confirmed & Saved] Task ID: {task_id}")

        # Step 2: Paste a follow-up changing its deadline
        followup_message = (
            "We have postponed the quarterly risk report to next Monday. Please make note."
        )
        print(f"\n2. Pasted Follow-up Message:\n   \"{followup_message}\"")

        # Step 3: Suggest and confirm matching task using semantic similarity
        saved_tasks = store.list_tasks()
        ranked = linker.rank_candidates(followup_message, saved_tasks, top_k=3)
        assert len(ranked) >= 1
        top_match = ranked[0]

        print(f"\n3. Semantic Matching Results (top match):")
        print(f"   - Matched Task ID: {top_match['task']['task_id']}")
        print(f"   - Description: {top_match['task']['action_description']}")
        print(f"   - Similarity Score: {top_match['similarity_score']:.4f}")

        assert top_match["task"]["task_id"] == task_id, "Top match did not match expected task ID"

        # Detect conservative updates
        proposals = FollowupLinker.detect_updates(followup_message)
        print(f"\n   Detected Proposed Updates:")
        for p in proposals:
            print(f"   - Type: {p['update_type']} | New value: '{p['new_value']}' | Evidence: '{p['evidence_text']}'")

        deadline_update = next((p for p in proposals if p["update_type"] == "deadline_revised"), None)
        assert deadline_update is not None, "Failed to detect proposed deadline revision"
        assert deadline_update["new_value"] == "next Monday"

        # Step 4: Save confirmed change and verify persistence
        up_ok = store.apply_update(
            task_id=task_id,
            update_type=deadline_update["update_type"],
            new_value=deadline_update["new_value"],
            evidence_text=deadline_update["evidence_text"],
            followup_text=followup_message,
        )
        assert up_ok, "Failed to apply confirmed update"
        print(f"\n4. Confirmed update successfully applied to SQLite.")

        # Verify persisted task and history
        persisted_task = store.get_task(task_id)
        assert persisted_task["deadline"] == "next Monday", f"Expected 'next Monday', got '{persisted_task['deadline']}'"
        assert persisted_task["status"] == "pending"

        history = store.get_history(task_id)
        assert len(history) == 2, f"Expected 2 history rows (created + deadline_revised), got {len(history)}"

        print(f"\n   Persisted Task State:")
        print(f"   - ID: {persisted_task['task_id']}")
        print(f"   - Deadline: {persisted_task['deadline']}")
        print(f"   - History entries: {len(history)}")
        for h in history:
            print(f"     * [{h['timestamp']}] {h['update_type']}: '{h['old_value']}' -> '{h['new_value']}' (evidence: '{h['evidence_text']}')")

    print("\n" + "=" * 60)
    print("END-TO-END WORKFLOW VERIFIED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_verification()
