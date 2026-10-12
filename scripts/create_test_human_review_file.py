#!/usr/bin/env python3
"""Create small review file from held-out test split for human reference labels without fabricating labels."""

import csv
import json
from pathlib import Path

MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")
OUTPUT_PATH = Path("data/annotations/test_human_review_sample.csv")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_records = []
    
    # Stratified-style selection: pick 25 weak-positive and 25 weak-negative from test split
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") == "test" and not rec.get("excluded"):
                sample_records.append(rec)

    # Deterministic sampling (50 examples total)
    # 20 with weak_commitment=1, 15 with weak_request=1 only, 15 with neither
    com_samples = [r for r in sample_records if r.get("weak_commitment") == 1][:20]
    req_samples = [r for r in sample_records if r.get("weak_request") == 1 and r.get("weak_commitment") == 0][:15]
    neg_samples = [r for r in sample_records if r.get("weak_request") == 0 and r.get("weak_commitment") == 0][:15]

    selected = com_samples + req_samples + neg_samples

    fieldnames = [
        "sample_id",
        "message_id",
        "thread_id",
        "source_relative_path",
        "weak_request",
        "weak_commitment",
        "human_gold_request",  # BLANK for human annotator
        "human_gold_commitment",  # BLANK for human annotator
        "human_reviewer_notes",  # BLANK for human annotator
        "text",
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(selected, 1):
            writer.writerow({
                "sample_id": f"test_sample_{idx:03d}",
                "message_id": r["message_id"],
                "thread_id": r["thread_id"],
                "source_relative_path": r.get("source_relative_path", ""),
                "weak_request": r["weak_request"],
                "weak_commitment": r["weak_commitment"],
                "human_gold_request": "",  # Pending human review
                "human_gold_commitment": "",  # Pending human review
                "human_reviewer_notes": "",
                "text": r["text"][:1000],  # generous text length for review
            })

    print(f"Created human review sample with {len(selected)} held-out test records at {OUTPUT_PATH}")
    print("Human gold fields left blank per project specification.")


if __name__ == "__main__":
    main()
