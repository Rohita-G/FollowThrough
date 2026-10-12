#!/usr/bin/env python3
"""Prepare the human evaluation review sheet from the existing 12-thread pilot batch.

Creates a single review sheet containing:
1. Task Extraction evaluation rows (messages across pilot threads).
2. Semantic Linking evaluation rows (subsequent messages in threads where prior tasks exist).

Fields for human gold reference are left strictly blank.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.followup_linker import FollowupLinker


def main():
    pilot_csv_path = Path("data/annotations/pilot_labeling_batch.provisional_train.csv")
    output_path = Path("data/annotations/evaluation_review_sheet.csv")

    with pilot_csv_path.open("r", encoding="utf-8") as f:
        pilot_rows = list(csv.DictReader(f))

    linker = FollowupLinker()

    # Group messages by thread in chronological order
    threads: Dict[str, List[Dict[str, Any]]] = {}
    for r in pilot_rows:
        threads.setdefault(r["thread_id"], []).append(r)

    review_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # SECTION 1: Task Extraction Subset (20 messages from pilot)
    # -------------------------------------------------------------
    extraction_sample_ids = [
        "pilot-1277c0cc300136ea",  # Thread 1: Arrange phone interview
        "pilot-dfd9946fb440fd9a",  # Thread 1: We shall arrange interview
        "pilot-5d47a2225242d061",  # Thread 1: Courtesy follow-up
        "pilot-a5b1b376c3a98949",  # Thread 2: Review request
        "pilot-113146af622bd969",  # Thread 3: Informational / no task
        "pilot-0addf740b414d47a",  # Thread 3: Action sentence
        "pilot-76e89387b40fc2d1",  # Thread 4: Informational update
        "pilot-3111db6010db782d",  # Thread 4: Request with deadline
        "pilot-b93eccade5c80986",  # Thread 5: Direct request
        "pilot-25842e51d9365c9e",  # Thread 5: Commitment
        "pilot-a7f35f3685ba5ca4",  # Thread 6: Brief acknowledgment
        "pilot-056ac0e9dd18a4a6",  # Thread 7: Action item
        "pilot-0ad861a6d7979e76",  # Thread 7: Follow-up reply
        "pilot-d8cabd0fbc38cc4e",  # Thread 8: Action directive
        "pilot-3d054e2129e0e0c5",  # Thread 9: Multi-step request
        "pilot-4da725694ba3ab84",  # Thread 9: Detailed action plan
        "pilot-a36a1a5ec7725457",  # Thread 11: Candidate request
        "pilot-520608e3e8ab0a0f",  # Thread 11: Reply
        "pilot-f66b4722397cf576",  # Thread 12: Notification
        "pilot-3d1ccd174a321260",  # Thread 12: Action prompt
    ]

    extraction_rows_map = {r["example_id"]: r for r in pilot_rows if r["example_id"] in extraction_sample_ids}

    for idx, eid in enumerate(extraction_sample_ids, start=1):
        r = extraction_rows_map[eid]
        cands = extract_candidate_tasks(r["text"])

        extracted_task_desc = cands[0]["action_description"] if cands else "NONE_DETECTED"
        extracted_owner = cands[0]["owner"] if cands else "NONE"
        extracted_deadline = cands[0]["deadline"] if cands else "NONE"
        extracted_evidence = cands[0]["evidence_text"] if cands else ""
        extracted_offsets = f"{cands[0]['start_offset']}:{cands[0]['end_offset']}" if cands else ""

        review_rows.append({
            "review_id": f"EXT-{idx:02d}",
            "evaluation_section": "TASK_EXTRACTION",
            "example_id": r["example_id"],
            "thread_id": r["thread_id"],
            "source_message_id": r["source_message_id"],
            "source_text": r["text"].replace("\r\n", "\n").replace("\r", "\n"),
            "system_extracted_task": extracted_task_desc,
            "system_extracted_owner": extracted_owner,
            "system_extracted_deadline": extracted_deadline,
            "system_extracted_evidence": extracted_evidence,
            "system_extracted_offsets": extracted_offsets,
            "prior_available_tasks": "",
            "system_proposed_matches": "",
            # Blank Human Gold Fields
            "human_gold_has_task": "",           # 1 if message contains a real task/event, 0 if not
            "human_gold_action_description": "", # Gold task wording or NONE
            "human_gold_owner": "",              # Gold explicit owner or Unspecified
            "human_gold_deadline": "",           # Gold literal deadline or None
            "human_gold_evidence_span": "",      # Exact matching substring in source
            "human_gold_linked_task_id": "",     # N/A for extraction
            "human_gold_notes_or_exclude": "",   # Leave blank or mark "AMBIGUOUS"
        })

    # -------------------------------------------------------------
    # SECTION 2: Semantic Linking Subset (Real Thread Follow-ups)
    # -------------------------------------------------------------
    # Walk through threads: for any message appearing AFTER an earlier message has a candidate task,
    # generate a linking evaluation query using only prior tasks available before that message.
    linking_idx = 1
    for tid, msgs in threads.items():
        prior_tasks_in_thread: List[Dict[str, Any]] = []

        for m_idx, m in enumerate(msgs):
            # Check if this message is a subsequent turn with prior tasks available
            if prior_tasks_in_thread:
                # Rank prior tasks using FollowupLinker
                ranked = linker.rank_candidates(m["text"], prior_tasks_in_thread, top_k=3)
                top_matches_repr = "; ".join([
                    f"{r['task']['task_id']} ({r['similarity_score']:.3f}: {r['task']['action_description'][:40]}...)"
                    for r in ranked
                ])
                prior_tasks_repr = " | ".join([
                    f"[{t['task_id']}]: {t['action_description'][:50]}"
                    for t in prior_tasks_in_thread
                ])

                review_rows.append({
                    "review_id": f"LNK-{linking_idx:02d}",
                    "evaluation_section": "SEMANTIC_LINKING",
                    "example_id": m["example_id"],
                    "thread_id": m["thread_id"],
                    "source_message_id": m["source_message_id"],
                    "source_text": m["text"].replace("\r\n", "\n").replace("\r", "\n"),
                    "system_extracted_task": "",
                    "system_extracted_owner": "",
                    "system_extracted_deadline": "",
                    "system_extracted_evidence": "",
                    "system_extracted_offsets": "",
                    "prior_available_tasks": prior_tasks_repr,
                    "system_proposed_matches": top_matches_repr,
                    # Blank Human Gold Fields
                    "human_gold_has_task": "",
                    "human_gold_action_description": "",
                    "human_gold_owner": "",
                    "human_gold_deadline": "",
                    "human_gold_evidence_span": "",
                    "human_gold_linked_task_id": "",     # Gold prior task_id that this message links to, or "NONE", or "AMBIGUOUS"
                    "human_gold_notes_or_exclude": "",   # Notes or "AMBIGUOUS"
                })
                linking_idx += 1

            # Extract any task from this message to add to prior tasks pool for subsequent turns
            msg_cands = extract_candidate_tasks(m["text"])
            for c_idx, c in enumerate(msg_cands):
                prior_tasks_in_thread.append({
                    "task_id": f"task-{m['example_id'][:8]}-{c_idx+1}",
                    "action_description": c["action_description"],
                    "owner": c["owner"],
                    "deadline": c["deadline"],
                    "evidence_text": c["evidence_text"],
                    "source_example_id": m["example_id"],
                })

    # Write unified CSV review sheet
    fieldnames = [
        "review_id",
        "evaluation_section",
        "example_id",
        "thread_id",
        "source_message_id",
        "source_text",
        "system_extracted_task",
        "system_extracted_owner",
        "system_extracted_deadline",
        "system_extracted_evidence",
        "system_extracted_offsets",
        "prior_available_tasks",
        "system_proposed_matches",
        "human_gold_has_task",
        "human_gold_action_description",
        "human_gold_owner",
        "human_gold_deadline",
        "human_gold_evidence_span",
        "human_gold_linked_task_id",
        "human_gold_notes_or_exclude",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"Evaluation review sheet created at {output_path}")
    print(f"Total review rows: {len(review_rows)} (Extraction: 20, Linking: {linking_idx - 1})")


if __name__ == "__main__":
    main()
