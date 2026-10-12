#!/usr/bin/env python3
"""Evaluation script for FollowThrough NLP Pipeline.

Evaluates:
1. Task Extraction:
   - Precision, Recall, and F1 comparing rule-extracted tasks with human gold.
   - Owner and Deadline exact-match accuracy (handling unspecified fields explicitly).
2. Semantic Follow-up Linking:
   - Mean Reciprocal Rank (MRR)
   - Recall@1 and Recall@3 on valid annotated task-link queries.

Rule: Metrics are computed only when human annotations are supplied in evaluation_review_sheet.csv.
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: Optional[str]) -> str:
    """Normalize text for consistent comparison by lowercasing and stripping punctuation."""
    if not text:
        return ""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(clean.split())


def compute_token_overlap(str1: str, str2: str) -> float:
    """Compute Jaccard token overlap between two strings."""
    tokens1 = set(normalize_text(str1).split())
    tokens2 = set(normalize_text(str2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def evaluate_pipeline(review_sheet_path: Path = Path("data/annotations/evaluation_review_sheet.csv")):
    if not review_sheet_path.is_file():
        print(f"Error: Review sheet not found at {review_sheet_path}")
        return

    with review_sheet_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ext_rows = [r for r in rows if r.get("evaluation_section") == "TASK_EXTRACTION"]
    lnk_rows = [r for r in rows if r.get("evaluation_section") == "SEMANTIC_LINKING"]

    print("=" * 65)
    print("FOLLOWTHROUGH NLP PIPELINE EVALUATION")
    print("=" * 65)
    print(f"Total rows in review sheet: {len(rows)}")
    print(f"  - Extraction subset: {len(ext_rows)} rows")
    print(f"  - Linking subset:    {len(lnk_rows)} rows\n")

    # ---------------------------------------------------------
    # 1. Evaluate Task Extraction
    # ---------------------------------------------------------
    ext_reviewed = [r for r in ext_rows if r.get("human_gold_has_task", "").strip() != ""]
    print(f"--- Task Extraction Evaluation (Reviewed: {len(ext_reviewed)}/{len(ext_rows)}) ---")

    if not ext_reviewed:
        print("Status: AWAITING HUMAN REVIEW.")
        print("Please fill 'human_gold_has_task' (1 or 0) and target fields in data/annotations/evaluation_review_sheet.csv.\n")
    else:
        tp, fp, fn, tn = 0, 0, 0, 0
        owner_correct, deadline_correct = 0, 0
        evaluable_task_count = 0
        excluded_ext = 0

        for r in ext_reviewed:
            # Check for exclusion due to ambiguity
            if "ambiguous" in r.get("human_gold_notes_or_exclude", "").lower():
                excluded_ext += 1
                continue

            gold_has_task = r["human_gold_has_task"].strip() in ("1", "true", "yes")
            sys_has_task = r["system_extracted_task"].strip() not in ("NONE_DETECTED", "", "None")

            if gold_has_task and sys_has_task:
                # Check semantic alignment of extracted action description
                overlap = compute_token_overlap(r["system_extracted_task"], r.get("human_gold_action_description", ""))
                if overlap >= 0.25:  # Shared key action keywords
                    tp += 1
                    evaluable_task_count += 1

                    # Owner Exact Match
                    norm_sys_owner = normalize_text(r.get("system_extracted_owner", "unspecified"))
                    norm_gold_owner = normalize_text(r.get("human_gold_owner", "unspecified"))
                    if norm_sys_owner == norm_gold_owner or (norm_sys_owner in ("unspecified", "none") and norm_gold_owner in ("unspecified", "none")):
                        owner_correct += 1

                    # Deadline Exact Match
                    norm_sys_deadline = normalize_text(r.get("system_extracted_deadline", "none"))
                    norm_gold_deadline = normalize_text(r.get("human_gold_deadline", "none"))
                    if norm_sys_deadline == norm_gold_deadline or (norm_sys_deadline in ("none", "none specified") and norm_gold_deadline in ("none", "none specified")):
                        deadline_correct += 1
                else:
                    fp += 1
                    fn += 1
            elif gold_has_task and not sys_has_task:
                fn += 1
            elif not gold_has_task and sys_has_task:
                fp += 1
            else:
                tn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        owner_acc = owner_correct / evaluable_task_count if evaluable_task_count > 0 else 0.0
        deadline_acc = deadline_correct / evaluable_task_count if evaluable_task_count > 0 else 0.0

        print(f"Evaluated examples: {len(ext_reviewed) - excluded_ext} (Excluded ambiguous: {excluded_ext})")
        print(f"Contingency: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"Task Extraction Precision: {prec:.4f}")
        print(f"Task Extraction Recall:    {rec:.4f}")
        print(f"Task Extraction F1-Score:  {f1:.4f}")
        print(f"Owner Exact Match Accuracy:    {owner_acc:.4f} ({owner_correct}/{evaluable_task_count})")
        print(f"Deadline Exact Match Accuracy: {deadline_acc:.4f} ({deadline_correct}/{evaluable_task_count})\n")

    # ---------------------------------------------------------
    # 2. Evaluate Semantic Follow-up Linking
    # ---------------------------------------------------------
    lnk_reviewed = [r for r in lnk_rows if r.get("human_gold_linked_task_id", "").strip() != ""]
    print(f"--- Semantic Linking Evaluation (Reviewed: {len(lnk_reviewed)}/{len(lnk_rows)}) ---")

    if not lnk_reviewed:
        print("Status: AWAITING HUMAN REVIEW.")
        print("Please fill 'human_gold_linked_task_id' (task ID, NONE, or AMBIGUOUS) in data/annotations/evaluation_review_sheet.csv.\n")
    else:
        reciprocal_ranks = []
        hits_at_1 = 0
        hits_at_3 = 0
        evaluable_links = 0
        excluded_lnk = 0

        for r in lnk_reviewed:
            gold_target = r["human_gold_linked_task_id"].strip()
            if not gold_target or "ambiguous" in gold_target.lower():
                excluded_lnk += 1
                continue

            if gold_target.upper() in ("NONE", "NO_LINK", "UNRELATED"):
                # Message does not link to any prior task
                continue

            evaluable_links += 1
            # Parse top matches from system_proposed_matches
            raw_matches = r.get("system_proposed_matches", "")
            matched_ids = re.findall(r"(task-[a-z0-9\-]+)", raw_matches)

            if gold_target in matched_ids:
                rank = matched_ids.index(gold_target) + 1
                reciprocal_ranks.append(1.0 / rank)
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
            else:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / evaluable_links if evaluable_links > 0 else 0.0
        r1 = hits_at_1 / evaluable_links if evaluable_links > 0 else 0.0
        r3 = hits_at_3 / evaluable_links if evaluable_links > 0 else 0.0

        print(f"Evaluated linking queries: {evaluable_links} (Excluded: {excluded_lnk})")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"Recall@1:                   {r1:.4f} ({hits_at_1}/{evaluable_links})")
        print(f"Recall@3:                   {r3:.4f} ({hits_at_3}/{evaluable_links})\n")

    print("=" * 65)


if __name__ == "__main__":
    evaluate_pipeline()
