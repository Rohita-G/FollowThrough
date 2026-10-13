#!/usr/bin/env python3
"""Build Canonical Experiment Manifest, Weak Labels, and Thread-Disjoint Splits for Full 100k Corpus."""

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Weak Labeling Linguistic Rules
# ---------------------------------------------------------------------------

# Request Positive Patterns
REQ_PATTERNS = [
    re.compile(r"\b(?:please|pls)\s+(?:review|send|forward|check|call|contact|provide|arrange|confirm|update|sign|approve|let me know)\b", re.I),
    re.compile(r"\b(?:could you|can you|would you|will you)\s+(?:please\s+)?(?:review|send|forward|check|call|provide|arrange|confirm|update|help)\b", re.I),
    re.compile(r"\b(?:let me know|advise|ensure that|make sure to|need you to|ask you to)\b", re.I),
    re.compile(r"\b(?:please\s+[a-z]+|let\'s\s+[a-z]+)\b", re.I),
]

# Request Negative / Guard Patterns
REQ_GUARDS = [
    re.compile(r"\b(?:please\s+(?:don\'t|do not|ignore|disregard))\b", re.I),
    re.compile(r"\b(?:i\s+(?:asked|requested|forwarded|sent))\b", re.I),  # Reported past action
    re.compile(r"\b(?:no\s+need\s+to|don\'t\s+worry\s+about)\b", re.I),
    re.compile(r"^(?:thanks|thank you|thx)[\s.!,]*$", re.I),              # Pure pleasantry
]

# Commitment Positive Patterns
COM_PATTERNS = [
    re.compile(r"\b(?:i\s+will|we\s+will|i\'ll|we\'ll|i\s+shall|we\s+shall)\s+(?:send|review|call|follow\s+up|provide|arrange|attend|check|submit|file|handle)\b", re.I),
    re.compile(r"\b(?:i\s+promise|we\s+agree(?:d)?\s+to|i\s+can\s+handle|we\s+can\s+handle|i\'m\s+working\s+on)\b", re.I),
    re.compile(r"\b(?:will\s+get\s+back\s+to\s+you|will\s+do|i\s+intend\s+to)\b", re.I),
]

# Commitment Negative / Guard Patterns
COM_GUARDS = [
    re.compile(r"\b(?:i\s+will\s+not|we\s+will\s+not|i\s+won\'t|we\s+won\'t|cannot|can\'t)\b", re.I),
    re.compile(r"\b(?:i\s+(?:have\s+)?(?:already\s+)?(?:sent|submitted|completed|finished|filed))\b", re.I), # Past completed action
    re.compile(r"\b(?:if\s+i\s+(?:can|could)|might\s+be\s+able\s+to)\b", re.I),  # Hypothetical / tentative
]


def compute_weak_labels(text: str) -> Tuple[int, int, List[str]]:
    """Compute transparent linguistic weak labels for request and commitment.
    
    Returns (weak_request, weak_commitment, rules_triggered).
    """
    clean = text.strip()
    if not clean:
        return 0, 0, []

    rules_triggered: List[str] = []

    # 1. Request labeling
    req_pos = False
    for idx, p in enumerate(REQ_PATTERNS):
        if p.search(clean):
            req_pos = True
            rules_triggered.append(f"REQ_POS_{idx}")
            break

    if req_pos:
        for idx, g in enumerate(REQ_GUARDS):
            if g.search(clean):
                req_pos = False
                rules_triggered.append(f"REQ_GUARD_{idx}")
                break

    # 2. Commitment labeling
    com_pos = False
    for idx, p in enumerate(COM_PATTERNS):
        if p.search(clean):
            com_pos = True
            rules_triggered.append(f"COM_POS_{idx}")
            break

    if com_pos:
        for idx, g in enumerate(COM_GUARDS):
            if g.search(clean):
                com_pos = False
                rules_triggered.append(f"COM_GUARD_{idx}")
                break

    return int(req_pos), int(com_pos), rules_triggered


# ---------------------------------------------------------------------------
# Union-Find for Connected Thread & Duplicate Group Clustering
# ---------------------------------------------------------------------------

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


def main():
    start_time = time.perf_counter()
    input_messages_path = Path("data/processed/enron_subset_v2_threads/messages.jsonl")
    output_dir = Path("data/processed/full_corpus_manifest")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_output_path = output_dir / "manifest.jsonl"
    summary_output_path = output_dir / "manifest_summary.json"

    if not input_messages_path.is_file():
        print(f"Input messages file not found at {input_messages_path}")
        print("Please ensure the preprocessed messages JSONL file exists.")
        return

    print("Step 1: Reading records and building duplicate groups...")
    total_records = 0
    excluded_empty = 0
    excluded_too_short = 0
    included_records = 0

    records: List[Dict[str, Any]] = []
    ds = DisjointSet()

    # Pass 1: Parse, deduplicate grouping, and cluster
    with input_messages_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_records += 1
            msg = json.loads(line)

            cleaned_body = msg.get("cleaned_body", "").strip()
            msg_id = msg.get("message_id", f"msg_{total_records}")
            thread_id = msg.get("thread_id", f"singleton_{total_records}")

            norm_subj = re.sub(r"^(?:re|fwd|fw):\s*", "", msg.get("subject", "").strip(), flags=re.I).lower()
            sender = msg.get("from", "").strip().lower()
            body_hash = msg.get("raw_body_sha256", hashlib.sha256(cleaned_body.encode("utf-8")).hexdigest())

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

            # Connect thread_id and group_id so they never split across train/val/test
            ds.union(f"TH:{thread_id}", f"GRP:{group_id}")

            records.append({
                "message_id": msg_id,
                "thread_id": thread_id,
                "group_id": group_id,
                "source_relative_path": msg.get("source_relative_path", ""),
                "mailbox": msg.get("mailbox", ""),
                "cleaned_body": cleaned_body,
                "excluded": exclusion_reason is not None,
                "exclusion_reason": exclusion_reason,
            })

    print(f"Total read: {total_records}")
    print(f"Included: {included_records}, Excluded: {excluded_empty + excluded_too_short} (Empty: {excluded_empty}, Short: {excluded_too_short})")

    # Pass 2: Assign clusters and deterministic split
    print("\nStep 2: Assigning clusters and split partition (~80% Train, ~10% Val, ~10% Test)...")
    split_counts = {"train": 0, "validation": 0, "test": 0, "excluded": 0}
    label_counts = {
        "train": {"req": 0, "com": 0, "total": 0},
        "validation": {"req": 0, "com": 0, "total": 0},
        "test": {"req": 0, "com": 0, "total": 0},
    }

    with manifest_output_path.open("w", encoding="utf-8") as manifest_file:
        for rec in records:
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

                # Deterministic split hashing on cluster_id
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

            manifest_entry = {
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
            manifest_file.write(json.dumps(manifest_entry) + "\n")

    elapsed = time.perf_counter() - start_time
    print(f"\nManifest creation complete in {elapsed:.2f}s!")
    print(f"Split distributions: {split_counts}")

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
        "elapsed_seconds": round(elapsed, 2),
        "manifest_path": str(manifest_output_path),
    }

    with summary_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_output_path}")


if __name__ == "__main__":
    main()
