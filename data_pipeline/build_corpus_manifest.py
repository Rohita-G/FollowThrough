#!/usr/bin/env python3
"""Build Canonical Experiment Manifest with Weak Labels and Thread-Disjoint Splits."""

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from data_pipeline.generate_weak_labels import compute_weak_labels


class DisjointSet:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_manifest(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build manifest entries with thread-disjoint splits and weak labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "manifest_summary.json"

    ds = DisjointSet()
    processed_records: List[Dict[str, Any]] = []

    total_records = len(records)
    excluded_empty = 0
    excluded_too_short = 0
    included_records = 0

    # Pass 1: Duplicate grouping and disjoint cluster formation
    for idx, rec in enumerate(records):
        cleaned_body = rec.get("cleaned_body", "").strip()
        msg_id = rec.get("message_id", f"msg_{idx}")
        thread_id = rec.get("thread_id", f"singleton_{idx}")

        norm_subj = re.sub(r"^(?:re|fwd|fw):\s*", "", rec.get("subject", "").strip(), flags=re.I).lower()
        sender = rec.get("from", "").strip().lower()
        body_hash = rec.get("raw_body_sha256", hashlib.sha256(cleaned_body.encode("utf-8")).hexdigest())

        dup_group_key = f"{body_hash}||{sender}||{norm_subj}"
        group_id = f"grp_{hashlib.md5(dup_group_key.encode('utf-8')).hexdigest()[:12]}"

        exclusion_reason: Optional[str] = None
        if not cleaned_body:
            exclusion_reason = "empty_or_whitespace_body"
            excluded_empty += 1
        elif len(cleaned_body) < 10:
            exclusion_reason = "too_short_under_10_chars"
            excluded_too_short += 1
        else:
            included_records += 1

        ds.union(f"TH:{thread_id}", f"GRP:{group_id}")

        processed_records.append({
            "message_id": msg_id,
            "thread_id": thread_id,
            "group_id": group_id,
            "source_relative_path": rec.get("source_relative_path", ""),
            "mailbox": rec.get("mailbox", ""),
            "cleaned_body": cleaned_body,
            "excluded": exclusion_reason is not None,
            "exclusion_reason": exclusion_reason,
        })

    # Pass 2: Cluster assignment, split determination, and weak labeling
    split_counts = {"train": 0, "validation": 0, "test": 0, "excluded": 0}
    label_counts = {
        "train": {"req": 0, "com": 0, "total": 0},
        "validation": {"req": 0, "com": 0, "total": 0},
        "test": {"req": 0, "com": 0, "total": 0},
    }

    manifest_entries: List[Dict[str, Any]] = []

    with manifest_path.open("w", encoding="utf-8") as f_out:
        for rec in processed_records:
            if rec["excluded"]:
                rec["split"] = "excluded"
                rec["weak_request"] = 0
                rec["weak_commitment"] = 0
                rec["weak_rules"] = []
                rec["cluster_id"] = "none"
                split_counts["excluded"] += 1
            else:
                cluster_id = ds.find(f"TH:{rec['thread_id']}")
                rec["cluster_id"] = cluster_id

                # Deterministic cluster split
                cluster_hash = int(hashlib.md5(cluster_id.encode("utf-8")).hexdigest()[:8], 16)
                bucket = cluster_hash % 100
                if bucket < 80:
                    split = "train"
                elif bucket < 90:
                    split = "validation"
                else:
                    split = "test"

                rec["split"] = split
                split_counts[split] += 1

                weak_req, weak_com, rules = compute_weak_labels(rec["cleaned_body"])
                rec["weak_request"] = weak_req
                rec["weak_commitment"] = weak_com
                rec["weak_rules"] = rules

                label_counts[split]["total"] += 1
                label_counts[split]["req"] += weak_req
                label_counts[split]["com"] += weak_com

            entry = {
                "message_id": rec["message_id"],
                "thread_id": rec["thread_id"],
                "group_id": rec["group_id"],
                "cluster_id": rec["cluster_id"],
                "split": rec["split"],
                "source_relative_path": rec["source_relative_path"],
                "mailbox": rec["mailbox"],
                "excluded": rec["excluded"],
                "exclusion_reason": rec["exclusion_reason"],
                "weak_request": rec["weak_request"],
                "weak_commitment": rec["weak_commitment"],
                "weak_rules": rec["weak_rules"],
                "text": rec["cleaned_body"],
            }
            manifest_entries.append(entry)
            f_out.write(json.dumps(entry) + "\n")

    summary = {
        "corpus_version": "v2",
        "total_source_records": total_records,
        "included_usable_records": included_records,
        "excluded_records": excluded_empty + excluded_too_short,
        "exclusions_by_reason": {
            "empty_or_whitespace_body": excluded_empty,
            "too_short_under_10_chars": excluded_too_short,
        },
        "split_counts": split_counts,
        "split_percentages": {
            "train": round(split_counts["train"] / max(1, included_records) * 100, 2),
            "validation": round(split_counts["validation"] / max(1, included_records) * 100, 2),
            "test": round(split_counts["test"] / max(1, included_records) * 100, 2),
        },
        "label_counts": label_counts,
        "manifest_path": str(manifest_path),
    }

    with summary_path.open("w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    return manifest_entries, summary
