#!/usr/bin/env python3
"""Screen annotation candidate threads and generate versioned human review pilot packets.

This script implements:
1. Versioned annotation-selection screening requiring >=2 distinct nonempty raw bodies per thread.
   (Excludes single-body / all-copy threads).
2. Conservative copy grouping for annotation display:
   Only groups messages when raw body, sender, normalized timestamp, and subject match.
   Keeps ambiguous repeats (different senders or dates) as separate events.
   Preserves all source identifiers, paths, and metadata differences.
3. Curated human review pilot generation (CSV, JSONL drafts, full thread displays, README).
   All annotation fields remain completely blank (no prefilled labels).
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_annotation_candidates import ANNOTATOR_COLUMNS, CSV_COLUMNS, cue_families
from scripts.validate_annotations import SCHEMA_VERSION, message_reference_key

PILOT_V2_COLUMNS = CSV_COLUMNS + (
    "screening_stratum",
    "thread_size",
    "distinct_body_count",
    "copy_group_size",
    "is_copy_group",
    "original_candidate_text",
    "original_body_cues",
    "cleaning_flags",
    "cleaning_uncertain",
)

DISPLAY_REPRESENTATIVE_RULE = "earliest_position_in_thread"


def normalize_sender(sender: Optional[str]) -> str:
    """Normalize sender email address for conservative copy matching."""
    if not sender:
        return ""
    # Extract email or lowercase stripped string
    _, addr = email.utils.parseaddr(sender)
    if addr:
        return addr.lower().strip()
    return sender.lower().strip()


def normalize_timestamp(date_str: Optional[str]) -> str:
    """Normalize RFC 2822 / ISO date string to UTC ISO string.

    Returns normalized string or stripped lowercase fallback.
    """
    if not date_str:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return date_str.strip().lower()


def normalize_subject(subject: Optional[str]) -> str:
    """Normalize subject line by stripping common reply/forward prefixes."""
    if not subject:
        return ""
    s = subject.strip().lower()
    for prefix in ("re:", "fw:", "fwd:", "[fwd:"):
        while s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s.rstrip("]").strip()


def group_messages_conservatively(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group messages conservatively within a thread.

    Only groups messages when:
    - raw_body is nonempty and identical (by sha256 or exact string)
    - sender matches (normalized)
    - timestamp matches (normalized UTC)
    - subject matches (normalized)

    Different senders or dates are never collapsed (they remain separate events).
    Preserves all member message_ids, source_paths, positions, and records differences.
    """
    sorted_msgs = sorted(
        messages,
        key=lambda m: (m.get("position", 0), m.get("source_relative_path", "")),
    )

    groups: List[Dict[str, Any]] = []

    for msg in sorted_msgs:
        body = (msg.get("body") or "").strip()
        body_sha = msg.get("raw_body_sha256") or (
            hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""
        )
        sender = normalize_sender(msg.get("from"))
        timestamp = normalize_timestamp(msg.get("date"))
        subject = normalize_subject(msg.get("subject"))

        matched_group: Optional[Dict[str, Any]] = None
        # Only group nonempty bodies where all metadata dimensions strictly align
        if body_sha and sender and timestamp:
            for g in groups:
                if (
                    g["_body_sha"] == body_sha
                    and g["_sender"] == sender
                    and g["_timestamp"] == timestamp
                    and g["_subject"] == subject
                ):
                    matched_group = g
                    break

        if matched_group is not None:
            matched_group["members"].append(msg)
        else:
            groups.append({
                "_body_sha": body_sha,
                "_sender": sender,
                "_timestamp": timestamp,
                "_subject": subject,
                "members": [msg],
            })

    # Format output display groups
    display_groups: List[Dict[str, Any]] = []
    for idx, g in enumerate(groups, start=1):
        members = g["members"]
        # Representative chosen deterministically by earliest position in thread
        rep = members[0]
        rep_pos = rep.get("position", 0)
        rep_id = str(rep.get("message_id") or "")
        rep_path = str(rep.get("source_relative_path") or "")

        member_ids = [str(m.get("message_id") or "") for m in members]
        member_paths = [str(m.get("source_relative_path") or "") for m in members]
        member_positions = [m.get("position", 0) for m in members]

        # Calculate metadata differences across members
        metadata_diffs: Dict[str, List[Any]] = {}
        if len(members) > 1:
            paths = sorted(set(member_paths))
            if len(paths) > 1:
                metadata_diffs["source_relative_path"] = paths
            to_addrs = sorted({str(m.get("to") or "") for m in members})
            if len(to_addrs) > 1:
                metadata_diffs["to"] = to_addrs
            cc_addrs = sorted({str(m.get("cc") or "") for m in members})
            if len(cc_addrs) > 1:
                metadata_diffs["cc"] = cc_addrs

        group_record = {
            "group_id": f"group-{rep.get('thread_id', 'unknown')}-pos{rep_pos}",
            "group_index": idx,
            "representative_message_id": rep_id,
            "representative_position": rep_pos,
            "representative_source_relative_path": rep_path,
            "representative_selection_rule": DISPLAY_REPRESENTATIVE_RULE,
            "is_copy_group": len(members) > 1,
            "copy_count": len(members),
            "member_message_ids": member_ids,
            "member_positions": member_positions,
            "member_source_relative_paths": member_paths,
            "metadata_differences": metadata_diffs,
            "representative_message": rep,
            "all_members": members,
        }
        display_groups.append(group_record)

    return display_groups


def count_distinct_nonempty_raw_bodies(messages: Sequence[Dict[str, Any]]) -> int:
    """Return count of distinct non-empty raw message bodies in a thread.

    NOTE: Distinct-body counts are a heuristic screening proxy, not a count of
    confirmed conversational turns.
    """
    distinct = set()
    for m in messages:
        body = (m.get("body") or "").strip()
        if body:
            sha = m.get("raw_body_sha256") or hashlib.sha256(body.encode("utf-8")).hexdigest()
            distinct.add(sha)
    return len(distinct)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete JSONL line in {path} at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            yield record


def select_and_generate_pilot(
    thread_output_dir: Path,
    output_csv: Path,
    output_drafts: Path,
    output_display_jsonl: Path,
    output_display_md: Path,
    output_readme: Path,
    selected_thread_ids: Optional[Sequence[str]] = None,
    seed: int = 20261012,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Screen candidates, build conservative display groups, and write pilot packet."""
    messages_path = thread_output_dir / "messages.jsonl"
    threads_path = thread_output_dir / "threads.jsonl"

    if not messages_path.is_file() or not threads_path.is_file():
        raise FileNotFoundError(f"Missing messages.jsonl or threads.jsonl in {thread_output_dir}")

    output_files = [output_csv, output_drafts, output_display_jsonl, output_display_md, output_readme]
    if not overwrite:
        existing = [str(p) for p in output_files if p.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {', '.join(existing)}")

    # Default curated thread IDs from the bounded review if not explicitly provided
    if selected_thread_ids is None:
        selected_thread_ids = [
            "thread-v2-2728d41088a5a5b7",  # Credible: Recruiting interview arrangement
            "thread-v2-201803453f303a89",  # Credible: Employee retention plan
            "thread-v2-01f8d4d29d2a0ad1",  # Credible: May 5 testimony prep
            "thread-v2-0fa19e507cf18b3d",  # Credible: California filing reminder letter
            "thread-v2-0efc7733bd9dba1b",  # Credible: A/A tutor program question
            "thread-v2-07e38b163b240420",  # Credible: George Huan referral (2 distinct bodies)
            "thread-v2-0d4244a3eab8ad10",  # Credible: Southern Cone master agreement
            "thread-v2-07afbfe327ed7fac",  # Credible: FPL contract inquiry
            "thread-v2-2bb29360714fa8f4",  # Credible: Power swap confirmation
            "thread-v2-03a1a8f4813b5a18",  # Uncertain: Academic paper forward / heavy quote
            "thread-v2-0f5c436187906f5f",  # Uncertain: Wayne Tow resume forward
            "thread-v2-2084b4332b84df1e",  # Deliberative: SoCalGas Capacity Forum
        ]

    thread_filter = set(selected_thread_ids)
    threads_by_id: Dict[str, Dict[str, Any]] = {}
    for t in _read_jsonl(threads_path):
        tid = t.get("thread_id")
        if tid in thread_filter:
            threads_by_id[tid] = t

    messages_by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in _read_jsonl(messages_path):
        tid = m.get("thread_id")
        if tid in thread_filter:
            messages_by_thread[tid].append(m)

    # Process threads in deterministic order of selected_thread_ids
    processed_threads: List[Dict[str, Any]] = []
    screening_csv_rows: List[Dict[str, Any]] = []
    drafts_jsonl_rows: List[Dict[str, Any]] = []

    total_raw_messages = 0
    total_distinct_groups = 0

    for tid in selected_thread_ids:
        msgs = messages_by_thread.get(tid, [])
        if not msgs:
            continue
        total_raw_messages += len(msgs)

        distinct_bodies = count_distinct_nonempty_raw_bodies(msgs)
        if distinct_bodies < 2:
            raise ValueError(f"Thread {tid} has < 2 distinct bodies ({distinct_bodies}); fails screening criterion")

        # Group conservatively
        display_groups = group_messages_conservatively(msgs)
        total_distinct_groups += len(display_groups)

        t_meta = threads_by_id.get(tid, {})
        thread_info = {
            "thread_id": tid,
            "subject": msgs[0].get("subject", ""),
            "raw_message_count": len(msgs),
            "distinct_raw_body_count_proxy": distinct_bodies,
            "display_group_count": len(display_groups),
            "link_rules": t_meta.get("link_rules", []),
            "start_date": t_meta.get("start_date", ""),
            "end_date": t_meta.get("end_date", ""),
            "display_groups": display_groups,
        }
        processed_threads.append(thread_info)

        # For each display group, construct screening queue item and blank draft
        for g_idx, group in enumerate(display_groups):
            rep_msg = group["representative_message"]
            pos = group["representative_position"]
            path = group["representative_source_relative_path"]
            cleaned = rep_msg.get("cleaned_body", "")
            raw = rep_msg.get("body", "")
            clean_cues = cue_families(cleaned)
            raw_cues = cue_families(raw)

            # Determine screening stratum
            if clean_cues:
                stratum = "cue_positive"
            elif raw_cues:
                stratum = "cleaning_disagreement"
            else:
                stratum = "cue_negative"

            candidate_identity = f"{seed}\n{tid}\n{path}\n{pos}"
            candidate_id = "pilot-" + hashlib.sha256(candidate_identity.encode("utf-8")).hexdigest()[:16]

            # Previous turn context: representative of preceding display group
            prev_msg_id = ""
            prev_msg_text = ""
            if g_idx > 0:
                prev_rep = display_groups[g_idx - 1]["representative_message"]
                prev_msg_id = str(message_reference_key(prev_rep) or "")
                prev_msg_text = prev_rep.get("cleaned_body", "")

            ref_id = str(message_reference_key(rep_msg) or "")

            csv_row = {
                "candidate_id": candidate_id,
                "thread_id": tid,
                "message_id": ref_id,
                "position": pos,
                "previous_message_id": prev_msg_id,
                "candidate_text": cleaned,
                "previous_message_text": prev_msg_text,
                "heuristic_cues": ";".join(clean_cues),
                "link_rules": ";".join(t_meta.get("link_rules", [])),
                "needs_thread_review": "true" if len(msgs) > 1 else "false",
                # All annotator columns strictly blank
                **{col: "" for col in ANNOTATOR_COLUMNS},
                "screening_stratum": stratum,
                "thread_size": len(msgs),
                "distinct_body_count": distinct_bodies,
                "copy_group_size": group["copy_count"],
                "is_copy_group": "true" if group["is_copy_group"] else "false",
                "original_candidate_text": raw,
                "original_body_cues": ";".join(raw_cues),
                "cleaning_flags": ";".join(rep_msg.get("cleaning_flags", [])),
                "cleaning_uncertain": "true" if rep_msg.get("cleaning_uncertain") else "false",
            }
            screening_csv_rows.append(csv_row)

            # Blank draft JSONL conforming to schema v1.0
            draft_row = {
                "schema_version": SCHEMA_VERSION,
                "example_id": "draft-" + candidate_id,
                "example_kind": "screening_draft",
                "example_origin": "real",
                "annotation_status": "draft",
                "thread_id": tid,
                "task_id": "",
                "task_source_message_ids": [],
                "task_text": "",
                "task_owner_text": "",
                "task_deadline_text": "",
                "task_deadline_normalized": None,
                "update_message_id": ref_id,
                "update_text": cleaned,
                "message_act_labels": [],
                "relation_labels": [],
                "updated_owner_text": "",
                "updated_deadline_text": "",
                "updated_deadline_normalized": None,
                "evidence_spans": [],
                "ambiguous": False,
                "ambiguity_reason": "",
                "annotator_id": "",
                "annotation_notes": "",
                "screening_candidate_id": candidate_id,
                "heuristic_cues_not_labels": clean_cues,
                # Provenance and copy context
                "copy_group_info": {
                    "is_copy_group": group["is_copy_group"],
                    "copy_count": group["copy_count"],
                    "member_message_ids": group["member_message_ids"],
                    "member_source_relative_paths": group["member_source_relative_paths"],
                    "representative_selection_rule": group["representative_selection_rule"],
                    "metadata_differences": group["metadata_differences"],
                },
            }
            drafts_jsonl_rows.append(draft_row)

    # Write files atomically via temp paths
    for p in output_files:
        p.parent.mkdir(parents=True, exist_ok=True)

    temp_files = [p.with_name(f".{p.name}.{os.getpid()}.tmp") for p in output_files]
    try:
        # 1. CSV
        with temp_files[0].open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PILOT_V2_COLUMNS)
            writer.writeheader()
            writer.writerows(screening_csv_rows)
            f.flush(); os.fsync(f.fileno())

        # 2. JSONL Drafts
        with temp_files[1].open("w", encoding="utf-8") as f:
            for d in drafts_jsonl_rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())

        # 3. Display JSONL
        with temp_files[2].open("w", encoding="utf-8") as f:
            for pt in processed_threads:
                f.write(json.dumps(pt, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())

        # 4. Human-readable Markdown Display
        md_content = _generate_markdown_display(processed_threads)
        temp_files[3].write_text(md_content, encoding="utf-8")
        with temp_files[3].open("rb") as f:
            os.fsync(f.fileno())

        # 5. README Guide
        readme_content = _generate_readme(
            seed=seed,
            thread_count=len(processed_threads),
            msg_count=len(screening_csv_rows),
            raw_msg_count=total_raw_messages,
            csv_name=output_csv.name,
            drafts_name=output_drafts.name,
            display_md_name=output_display_md.name,
            thread_output_dir=thread_output_dir,
        )
        temp_files[4].write_text(readme_content, encoding="utf-8")
        with temp_files[4].open("rb") as f:
            os.fsync(f.fileno())

        # Move to final destinations
        for temp_p, final_p in zip(temp_files, output_files):
            os.replace(temp_p, final_p)

    finally:
        for temp_p in temp_files:
            temp_p.unlink(missing_ok=True)

    stats = {
        "seed": seed,
        "selected_threads": len(processed_threads),
        "raw_messages_in_threads": total_raw_messages,
        "distinct_conservative_groups_selected": len(screening_csv_rows),
        "screening_strata_distribution": dict(Counter(r["screening_stratum"] for r in screening_csv_rows)),
        "copy_groups_count": sum(1 for r in screening_csv_rows if r["is_copy_group"] == "true"),
        "outputs": {
            "csv": str(output_csv),
            "drafts": str(output_drafts),
            "display_jsonl": str(output_display_jsonl),
            "display_md": str(output_display_md),
            "readme": str(output_readme),
        },
        "labels_created": False,
    }
    return stats


def _generate_markdown_display(processed_threads: List[Dict[str, Any]]) -> str:
    """Generate human-readable markdown packet showing full chronological threads with copy grouping."""
    lines = [
        "# FollowThrough Human Annotation Pilot Display Packet (Version 20261012)",
        "",
        "> [!IMPORTANT]",
        "> This packet is a curated pilot for testing annotation feasibility—not a representative benchmark or final test set.",
        "> All identical message copies across mailbox folders have been grouped conservatively based on matching body, sender, normalized timestamp, and subject.",
        "> Different senders or different dates are never collapsed and remain distinct events.",
        "",
        "---",
        "",
    ]

    for t_idx, t in enumerate(processed_threads, 1):
        lines.append(f"## Thread {t_idx}: `{t['thread_id']}`")
        lines.append(f"**Subject**: {t['subject']}  ")
        lines.append(f"**Raw Messages in Thread**: {t['raw_message_count']} | **Distinct Bodies (proxy)**: {t['distinct_raw_body_count_proxy']} | **Distinct Event Groups**: {t['display_group_count']}  ")
        lines.append(f"**Date Range**: {t['start_date']} to {t['end_date']}  ")
        lines.append("")

        for g in t["display_groups"]:
            rep = g["representative_message"]
            pos = g["representative_position"]
            sender = rep.get("from", "")
            date = rep.get("date", "")
            rep_path = g["representative_source_relative_path"]
            is_copy = g["is_copy_group"]
            count = g["copy_count"]
            cleaned = rep.get("cleaned_body", "")
            raw = rep.get("body", "")
            flags = rep.get("cleaning_flags", [])
            uncertain = rep.get("cleaning_uncertain", False)

            lines.append(f"### Event {g['group_index']} (Thread Position {pos})")
            if is_copy:
                lines.append(f"> [!NOTE]")
                lines.append(f"> **Conservative Copy Group**: {count} identical mailbox folder copies identified.")
                lines.append(f"> **Representative Selection Rule**: `{g['representative_selection_rule']}`")
                lines.append(f"> **Representative Source**: `{rep_path}` (Message-ID: `{rep.get('message_id')}`)")
                diffs = g["metadata_differences"]
                if diffs.get("source_relative_path"):
                    lines.append(f"> **All Folders / Paths**: {', '.join(f'`{p}`' for p in diffs['source_relative_path'])}")
                lines.append("")
            else:
                lines.append(f"- **Source**: `{rep_path}` (Message-ID: `{rep.get('message_id')}`)")

            lines.append(f"- **From**: `{sender}`")
            lines.append(f"- **Date**: `{date}`")
            lines.append(f"- **Cleaning Flags**: `{', '.join(flags) if flags else 'none'}`" + (" *(Cleaning Uncertain)*" if uncertain else ""))
            lines.append("")

            lines.append("#### Newly Written Content (Cleaned Body):")
            lines.append("```text")
            lines.append(cleaned if cleaned.strip() else "(Cleaned body is empty)")
            lines.append("```")
            lines.append("")

            # Show raw body with distinguishable forwarded/quoted sections if they differ
            if raw.strip() != cleaned.strip():
                lines.append("<details>")
                lines.append("<summary>View Full Raw Message Body (Preserved Original / Forwarded Context)</summary>")
                lines.append("")
                lines.append("```text")
                lines.append(raw)
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def _generate_readme(
    seed: int,
    thread_count: int,
    msg_count: int,
    raw_msg_count: int,
    csv_name: str,
    drafts_name: str,
    display_md_name: str,
    thread_output_dir: Path,
) -> str:
    """Generate concise annotation guide README."""
    return f"""# Annotation Pilot Screening Guide (Version {seed})

This packet is a **curated pilot for testing human annotation feasibility**—it is **not** a representative benchmark, random population sample, or final test set. It contains {msg_count} distinct conversational event candidates across {thread_count} threads (drawn from {raw_msg_count} raw mailbox records).

---

## 1. Feasibility Goals & Category Disclaimers

### Pilot Feasibility Questions
This pilot is designed specifically to determine:
1. **Task creation feasibility**: Whether initial requests and commitments can be reliably identified and annotated with high agreement.
2. **Update occurrence frequency**: Whether explicit task updates (e.g. status changes, actual progress updates) occur often enough in real multi-turn threads to support pair modeling.
3. **Research scope refinement**: Whether the original ambitious task-state research scope should be retained or narrowed based on empirical human annotation yield.

### Unverified Categories Disclaimer
- **No verified deadline changes, reassignment, cancellation, or completion are claimed**: The bounded review did **not** establish verified instances of these categories in this sample. Annotators must not assume they are present.
- **No manufactured synthetic data**: Rare categories are **not** manufactured with synthetic examples to inflate coverage. Real pilot data remains strictly real (`example_origin: "real"`). Synthetic test fixtures are kept strictly isolated in test suites (`tests/fixtures/`) and never mixed with pilot annotations.

---

## 2. How Copies Are Handled

In the Enron corpus, the same email frequently appears across multiple folders (`sent/`, `all_documents/`, `notes_inbox/`).
- **Conservative grouping**: Messages are grouped into a single event **only** when raw body, sender, normalized timestamp, and subject all strictly match.
- **Ambiguous repeats preserved**: Different senders (e.g. two people writing "Thanks") or different dates (e.g. resends days apart) are **never collapsed** and appear as separate events.
- **Provenance preserved**: Every displayed group shows its representative selection rule (`{DISPLAY_REPRESENTATIVE_RULE}`), all underlying source paths, and folder differences. Canonical message IDs and corpus records remain unchanged.

---

## 3. Annotation Workflow

For each thread in `{display_md_name}`:

### Phase 1: Conversation & Text Inspection
1. **Confirm conversation coherence**: Verify whether the displayed messages belong to a single coherent conversation thread or an accidental subject collision.
2. **Inspect repeats & copies**: Note whether any uncollapsed identical messages are ambiguous repeats across different people or times.
3. **Identify newly written content**: Contrast the `Cleaned Body` with the `Full Raw Message Body` to distinguish the sender's own newly written words from forwarded payloads, signature blocks, or quoted chains.

### Phase 2: Task Identification & Pairing
4. **Identify new tasks & commitments**: Look for substantive requests or affirmative commitments.
5. **Pair an existing task with a later update**: ONLY record an update when a later message clearly refers to the same earlier task and provides an actual update. Do **not** force ambiguous messages into an update category.
6. **Record literal owners, deadlines, and evidence**:
   - Extract literal phrases from the text (do not invent unstated deadlines or assignees).
   - Use `python scripts/evidence_offset_helper.py` to calculate exact `evidence_spans` character offsets.
7. **Mark ambiguity rather than guessing**: If the reference or task meaning is uncertain, set `ambiguous: true` and record the reason in `ambiguity_reason`.

---

## 4. Negative Guidelines & Distinctions to Emphasize

- **Acknowledgment $\\neq$ Completion**: "Thanks", "Thanks a lot", or "Received" is conversational courtesy, not evidence that an assigned task has been completed.
- **New Date $\\neq$ Deadline Change**: Discussing dates, coordinating target schedules, or mentioning a meeting time is not automatically an altered task deadline.
- **Mentioning a Person $\\neq$ Reassignment**: Adding participants, CC'ing someone, or discussing another team member is not automatically transferring task ownership.
- **Forwarded Request $\\neq$ New Request**: An email forwarding someone else's prior request is not necessarily a new request made by the forwarding sender.
- **Inquiry $\\neq$ Commitment**: Phrases like "Let me know what we need to do" are open-ended inquiries/prompts, not affirmative commitments.

---

## 4. Evidence Offset Helper

To avoid manual offset calculations, use the dependency-free helper:

```sh
# Calculate exact character offset from text
python scripts/evidence_offset_helper.py --text "We will send the report tomorrow." --sub "will send"

# When a substring appears multiple times, specify the occurrence explicitly:
python scripts/evidence_offset_helper.py --text "Send to Alice. Send to Bob." --sub "Send" --occurrence 2

# Or directly look up text from a draft entry:
python scripts/evidence_offset_helper.py --drafts data/annotations/{drafts_name} \\
  --example-id draft-pilot-xxx --sub "Please review"
```

The tool outputs a schema-compliant span object:
```json
{{"start": 3, "end": 12, "text": "will send", "source": "update_text"}}
```

---

## 5. Saving Completed Annotations & Validation

1. Save your completed annotations to:
   `data/annotations/pilot_annotations.completed.20261012.jsonl`
2. Run the validator:
   ```sh
   python scripts/validate_annotations.py data/annotations/pilot_annotations.completed.20261012.jsonl \\
     --messages {thread_output_dir / 'messages.jsonl'} \\
     --threads {thread_output_dir / 'threads.jsonl'}
   ```
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_output_dir", help="Path to reconstructed thread output directory (containing messages.jsonl and threads.jsonl)")
    parser.add_argument("--output-csv", default="data/annotations/pilot_screening.20261012.csv")
    parser.add_argument("--output-drafts", default="data/annotations/pilot_screening_drafts.20261012.jsonl")
    parser.add_argument("--output-display-jsonl", default="data/annotations/pilot_thread_display.20261012.jsonl")
    parser.add_argument("--output-display-md", default="data/annotations/pilot_thread_display.20261012.md")
    parser.add_argument("--output-readme", default="data/annotations/README.pilot.20261012.md")
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs")
    args = parser.parse_args(argv)

    try:
        stats = select_and_generate_pilot(
            thread_output_dir=Path(args.thread_output_dir),
            output_csv=Path(args.output_csv),
            output_drafts=Path(args.output_drafts),
            output_display_jsonl=Path(args.output_display_jsonl),
            output_display_md=Path(args.output_display_md),
            output_readme=Path(args.output_readme),
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
