#!/usr/bin/env python3
"""Create a deterministic, human-review-only screening pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_annotation_candidates import ANNOTATOR_COLUMNS, CSV_COLUMNS, cue_families
from scripts.validate_annotations import SCHEMA_VERSION, message_reference_key

PILOT_COLUMNS = CSV_COLUMNS + (
    "screening_stratum",
    "thread_size",
    "original_candidate_text",
    "original_body_cues",
    "cleaning_cue_disagreement",
)


def _link_category(rules: Sequence[str]) -> str:
    rule_set = set(rules)
    if rule_set == {"header", "subject"}:
        return "mixed"
    if rule_set == {"header"}:
        return "header_only"
    if rule_set == {"subject"}:
        return "subject_only"
    if not rule_set:
        return "unlinked"
    return "other"


def _size_band(size: int) -> str:
    if size <= 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    return "11+"


def _message_stratum(cleaned: str, original: str) -> Tuple[str, List[str], List[str]]:
    cleaned_cues = cue_families(cleaned)
    original_cues = cue_families(original)
    if cleaned_cues:
        return "cue_positive", cleaned_cues, original_cues
    if original_cues:
        return "cleaning_disagreement", cleaned_cues, original_cues
    return "cue_negative", cleaned_cues, original_cues


def _reference_id(record: Dict[str, Any]) -> str:
    key = message_reference_key(record)
    if key is None:
        raise ValueError("Input message lacks both message_id and source_relative_path")
    return key


def _choose_thread_ids(
    thread_meta: Dict[str, Dict[str, Any]], rng: random.Random, target_threads: int,
) -> Tuple[List[str], Dict[str, int], List[str]]:
    selected: List[str] = []
    selected_set = set()
    planned: Dict[str, int] = Counter()
    shortfalls: List[str] = []

    def choose(category: str, band: Optional[str], count: int) -> None:
        candidates = [
            thread_id for thread_id, item in thread_meta.items()
            if thread_id not in selected_set
            and item["link_category"] == category
            and (band is None or _size_band(item["message_count"]) == band)
        ]
        candidates.sort()
        rng.shuffle(candidates)
        take = min(count, len(candidates))
        selected.extend(candidates[:take])
        selected_set.update(candidates[:take])
        planned[f"{category}:{band or 'any'}"] += take
        if take < count:
            shortfalls.append(f"{category}/{band or 'any'}: requested {count}, available {len(candidates)}")

    # Preserve rare confidence categories first when the corpus contains them.
    for category in ("header_only", "mixed"):
        if any(item["link_category"] == category for item in thread_meta.values()):
            choose(category, None, 1)

    special_count = len(selected)
    desired_unlinked = min(3, max(0, target_threads - special_count - 12))
    subject_quotas = {"2": 2, "3-5": 4, "6-10": 3, "11+": 3}
    reduce_order = ("11+", "6-10", "3-5", "2")
    while sum(subject_quotas.values()) + desired_unlinked + special_count > target_threads:
        for band in reduce_order:
            if subject_quotas[band] > 0:
                subject_quotas[band] -= 1
                break

    for band, count in subject_quotas.items():
        if count:
            choose("subject_only", band, count)
    if desired_unlinked:
        choose("unlinked", "1", desired_unlinked)

    remaining = target_threads - len(selected)
    if remaining > 0:
        # Transparent fallback only fills the thread target; each unmet planned
        # stratum remains listed in shortfalls and the README.
        for category in ("subject_only", "unlinked", "header_only", "mixed"):
            if remaining <= 0:
                break
            candidates = [tid for tid, item in thread_meta.items() if tid not in selected_set and item["link_category"] == category]
            candidates.sort()
            rng.shuffle(candidates)
            take = min(remaining, len(candidates))
            selected.extend(candidates[:take])
            selected_set.update(candidates[:take])
            planned[f"{category}:fallback"] += take
            remaining -= take
    return selected[:target_threads], dict(planned), shortfalls


def _choose_messages(connection: sqlite3.Connection, thread_ids: List[str], target_messages: int, rng: random.Random) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    per_thread_rows: Dict[str, List[Dict[str, Any]]] = {}
    for thread_id in thread_ids:
        rows = connection.execute("""
            SELECT thread_id, position, message_id, source_path, body, cleaned_body, link_rules
            FROM messages WHERE thread_id = ? ORDER BY position
        """, (thread_id,)).fetchall()
        records = [dict(zip(("thread_id", "position", "message_id", "source_relative_path", "body", "cleaned_body", "link_rules"), row)) for row in rows]
        for record in records:
            record["link_rules"] = json.loads(record["link_rules"])
            stratum, clean_cues, original_cues = _message_stratum(record["cleaned_body"], record["body"])
            record["screening_stratum"] = stratum
            record["heuristic_cues"] = clean_cues
            record["original_body_cues"] = original_cues
        per_thread_rows[thread_id] = records

        # Up to three messages per thread: update-positive, cue-negative, and
        # cleaning-disagreement are preferred; remaining slots are seeded draws.
        priorities = (
            lambda row: row["screening_stratum"] == "cue_positive" and any(cue != "request_or_commitment" for cue in row["heuristic_cues"]),
            lambda row: row["screening_stratum"] == "cue_positive",
            lambda row: row["screening_stratum"] == "cue_negative",
            lambda row: row["screening_stratum"] == "cleaning_disagreement",
        )
        chosen: List[Dict[str, Any]] = []
        for predicate in priorities:
            choices = [row for row in records if row not in chosen and predicate(row)]
            if choices and len(chosen) < min(3, len(records)):
                choices.sort(key=lambda row: (row["position"], row["source_relative_path"]))
                chosen.append(rng.choice(choices))
        remaining = [row for row in records if row not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:max(0, min(3, len(records)) - len(chosen))])
        chosen.sort(key=lambda row: row["position"])
        for row in chosen:
            key = (row["thread_id"], row["position"])
            selected.append(row)
            selected_keys.add(key)

    if len(selected) < target_messages:
        remainder = [
            row for rows in per_thread_rows.values() for row in rows
            if (row["thread_id"], row["position"]) not in selected_keys
        ]
        # Balance any remaining cue classes before seeded fill.
        for stratum in ("cue_positive", "cue_negative", "cleaning_disagreement"):
            group = [row for row in remainder if row["screening_stratum"] == stratum]
            rng.shuffle(group)
            for row in group:
                if len(selected) >= target_messages:
                    break
                selected.append(row)
                selected_keys.add((row["thread_id"], row["position"]))
        if len(selected) < target_messages:
            remainder = [row for row in remainder if (row["thread_id"], row["position"]) not in selected_keys]
            rng.shuffle(remainder)
            selected.extend(remainder[:target_messages - len(selected)])

    rng.shuffle(selected)
    return selected[:target_messages]


def create_pilot(
    thread_output: Path,
    screening_output: Path,
    drafts_output: Path,
    readme_output: Path,
    seed: int = 20261011,
    target_threads: int = 15,
    target_messages: int = 40,
) -> Dict[str, Any]:
    if not 10 <= target_threads <= 20:
        raise ValueError("target_threads must be between 10 and 20")
    if not 30 <= target_messages <= 50:
        raise ValueError("target_messages must be between 30 and 50")
    messages_path = thread_output / "messages.jsonl"
    threads_path = thread_output / "threads.jsonl"
    if not messages_path.is_file() or not threads_path.is_file():
        raise FileNotFoundError(f"Expected messages.jsonl and threads.jsonl under {thread_output}")
    outputs = (screening_output, drafts_output, readme_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing pilot files: " + ", ".join(existing))

    rng = random.Random(seed)
    thread_meta: Dict[str, Dict[str, Any]] = {}
    available_link_categories = Counter()
    with tempfile.TemporaryDirectory(prefix="followthrough_pilot_") as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "pilot.sqlite3")
        connection.execute("""
            CREATE TABLE messages (
                thread_id TEXT NOT NULL, position INTEGER NOT NULL, message_id TEXT NOT NULL,
                source_path TEXT NOT NULL, body TEXT NOT NULL, cleaned_body TEXT NOT NULL,
                link_rules TEXT NOT NULL, PRIMARY KEY(thread_id, position)
            )
        """)
        for record in _read_jsonl(threads_path):
            thread_id = record.get("thread_id")
            count = record.get("message_count")
            if not isinstance(thread_id, str) or not isinstance(count, int) or isinstance(count, bool):
                raise ValueError("threads.jsonl contains an invalid thread_id/message_count")
            category = _link_category(record.get("link_rules", []))
            thread_meta[thread_id] = {"message_count": count, "link_category": category, "link_rules": record.get("link_rules", [])}
            available_link_categories[category] += 1

        input_messages = 0
        with connection:
            for record in _read_jsonl(messages_path):
                thread_id = record.get("thread_id")
                position = record.get("position")
                source_path = record.get("source_relative_path")
                if thread_id not in thread_meta or not isinstance(position, int) or isinstance(position, bool):
                    raise ValueError("messages.jsonl contains a message with invalid thread membership/position")
                if not isinstance(source_path, str) or not source_path:
                    raise ValueError("messages.jsonl contains a message without source_relative_path")
                connection.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (thread_id, position, str(record.get("message_id") or ""), source_path,
                     str(record.get("body") or ""), str(record.get("cleaned_body") or ""),
                     json.dumps(record.get("link_rules") if isinstance(record.get("link_rules"), list) else [])),
                )
                input_messages += 1
        connection.execute("CREATE INDEX pilot_thread_idx ON messages(thread_id, position)")

        selected_thread_ids, thread_strata, shortfalls = _choose_thread_ids(thread_meta, rng, target_threads)
        selected_rows = _choose_messages(connection, selected_thread_ids, target_messages, rng)
        if len(selected_thread_ids) < 10 or len(selected_rows) < 30:
            connection.close()
            raise RuntimeError(
                f"Available strata only support {len(selected_thread_ids)} threads/{len(selected_rows)} messages; "
                "not creating an undersized pilot."
            )

        chosen_keys = {(row["thread_id"], row["position"]) for row in selected_rows}
        previous_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for thread_id, position in chosen_keys:
            if position <= 1:
                continue
            row = connection.execute("""
                SELECT thread_id, position, message_id, source_path, body, cleaned_body, link_rules
                FROM messages WHERE thread_id = ? AND position = ?
            """, (thread_id, position - 1)).fetchone()
            if row:
                previous_by_key[(thread_id, position)] = dict(zip(
                    ("thread_id", "position", "message_id", "source_relative_path", "body", "cleaned_body", "link_rules"), row
                ))
        connection.close()

    selected_rows.sort(key=lambda row: (row["thread_id"], row["position"]))
    stratum_counts = Counter(row["screening_stratum"] for row in selected_rows)
    cue_counts = Counter(cue for row in selected_rows for cue in row["heuristic_cues"])
    thread_category_counts = Counter(thread_meta[thread_id]["link_category"] for thread_id in selected_thread_ids)
    candidate_rows: List[Dict[str, Any]] = []
    drafts: List[Dict[str, Any]] = []
    for row in selected_rows:
        thread_id = row["thread_id"]
        position = row["position"]
        source_path = row["source_relative_path"]
        candidate_id = "pilot-" + hashlib.sha256(f"{seed}\n{thread_id}\n{source_path}\n{position}".encode("utf-8")).hexdigest()[:16]
        previous = previous_by_key.get((thread_id, position), {})
        thread_size = thread_meta[thread_id]["message_count"]
        thread_rules = thread_meta[thread_id]["link_rules"]
        screening_row = {
            "candidate_id": candidate_id,
            "thread_id": thread_id,
            "message_id": _reference_id(row),
            "position": position,
            "previous_message_id": _reference_id(previous) if previous else "",
            "candidate_text": row["cleaned_body"],
            "previous_message_text": previous.get("cleaned_body", ""),
            "heuristic_cues": ";".join(row["heuristic_cues"]),
            "link_rules": ";".join(thread_rules),
            "needs_thread_review": "true" if thread_size > 1 else "false",
            **{column: "" for column in ANNOTATOR_COLUMNS},
            "screening_stratum": row["screening_stratum"],
            "thread_size": thread_size,
            "original_candidate_text": row["body"],
            "original_body_cues": ";".join(row["original_body_cues"]),
            "cleaning_cue_disagreement": "true" if row["screening_stratum"] == "cleaning_disagreement" else "false",
        }
        candidate_rows.append(screening_row)
        drafts.append({
            "schema_version": SCHEMA_VERSION,
            "example_id": "draft-" + candidate_id,
            "example_kind": "screening_draft",
            "example_origin": "real",
            "annotation_status": "draft",
            "thread_id": thread_id,
            "task_id": "",
            "task_source_message_ids": [],
            "task_text": "",
            "task_owner_text": "",
            "task_deadline_text": "",
            "task_deadline_normalized": None,
            "update_message_id": screening_row["message_id"],
            "update_text": row["cleaned_body"],
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
            "heuristic_cues_not_labels": row["heuristic_cues"],
        })

    if not any(row["screening_stratum"] == "cue_positive" for row in candidate_rows):
        raise RuntimeError("Selected pilot has no cue-positive messages; refusing to lower the screening criteria")
    if not any(row["screening_stratum"] == "cue_negative" for row in candidate_rows):
        raise RuntimeError("Selected pilot has no cue-negative messages; refusing to lower the screening criteria")

    screening_output.parent.mkdir(parents=True, exist_ok=True)
    drafts_output.parent.mkdir(parents=True, exist_ok=True)
    readme_output.parent.mkdir(parents=True, exist_ok=True)
    temp_paths = [path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in outputs]
    try:
        with temp_paths[0].open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=PILOT_COLUMNS)
            writer.writeheader()
            writer.writerows(candidate_rows)
            target.flush(); os.fsync(target.fileno())
        with temp_paths[1].open("w", encoding="utf-8") as target:
            for draft in drafts:
                target.write(json.dumps(draft, ensure_ascii=False) + "\n")
            target.flush(); os.fsync(target.fileno())

        report = {
            "source_thread_output": str(thread_output),
            "seed": seed,
            "requested_threads": target_threads,
            "selected_threads": len(selected_thread_ids),
            "requested_messages": target_messages,
            "selected_messages": len(candidate_rows),
            "screening_strata": dict(sorted(stratum_counts.items())),
            "heuristic_cue_counts": dict(sorted(cue_counts.items())),
            "selected_thread_link_categories": dict(sorted(thread_category_counts.items())),
            "available_thread_link_categories": dict(sorted(available_link_categories.items())),
            "planned_thread_strata": thread_strata,
            "unmet_thread_strata": shortfalls,
            "labels_created": False,
            "source_paths": {
                "screening_csv": str(screening_output),
                "blank_screening_drafts": str(drafts_output),
                "readme": str(readme_output),
            },
        }
        readme = f"""# Annotation Pilot Screening ({seed})

This is a bounded screening sample, not 40 guaranteed task–message examples. It contains {len(candidate_rows)} message candidates from {len(selected_thread_ids)} threads. The fixed seed is `{seed}`. Thread selection strata: `{json.dumps(thread_strata, sort_keys=True)}`. Message strata: `{json.dumps(dict(sorted(stratum_counts.items())), sort_keys=True)}`.

The sample includes cue-positive, cue-negative, and cleaner-disagreement messages where present. Thread link categories available in the source were `{json.dumps(dict(sorted(available_link_categories.items())), sort_keys=True)}`; this pilot selected `{json.dumps(dict(sorted(thread_category_counts.items())), sort_keys=True)}`. Header-linked/mixed categories absent in the corpus cannot be sampled. Unmet planned strata: `{json.dumps(shortfalls)}`.

## Review workflow

1. Open `{screening_output.name}`. `candidate_text` is upstream `cleaned_body`; `original_candidate_text` is the stored first-pass body. Compare them because production body line breaks were flattened before cleaning. The cleaner can leave reply headers/signatures/disclaimers and can erase cue-bearing quoted/forwarded content.
2. Use `thread_id` to inspect every row of that thread in `{thread_output / 'messages.jsonl'}` and its thread summary in `{thread_output / 'threads.jsonl'}`. `previous_message_id` is neighboring context only, not a task reference.
3. Start with `{drafts_output.name}`. These `screening_draft` rows intentionally have blank task IDs/text/source IDs and all labels/evidence blank. Do not fill a task link until a human identifies the task from the full thread. Convert the reviewed row to `task_update` or `task_creation` per `docs/annotation_schema_v1.md`.
4. Validate completed annotations with:

```sh
python scripts/validate_annotations.py data/annotations/pilot_annotations.completed.jsonl \\
  --messages {thread_output / 'messages.jsonl'} \\
  --threads {thread_output / 'threads.jsonl'}
```

Cue flags are screening hints, not labels. The sample is not a random population estimate and is not suitable by itself for model training/evaluation.
"""
        temp_paths[2].write_text(readme, encoding="utf-8")
        with temp_paths[2].open("rb") as target:
            os.fsync(target.fileno())
        # The outputs were checked absent before work; recheck to avoid replacing concurrent user files.
        if any(path.exists() for path in outputs):
            raise FileExistsError("A pilot output appeared during generation; refusing to overwrite it")
        for temp_path, final_path in zip(temp_paths, outputs):
            os.replace(temp_path, final_path)
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete JSONL line in {path} at {line_number}")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object in {path} at line {line_number}")
            yield item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_output_dir", help="Directory containing production messages.jsonl and threads.jsonl")
    parser.add_argument("--screening-output", required=True)
    parser.add_argument("--drafts-output", required=True)
    parser.add_argument("--readme-output", required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--target-threads", type=int, default=15)
    parser.add_argument("--target-messages", type=int, default=40)
    args = parser.parse_args()
    try:
        create_pilot(
            Path(args.thread_output_dir), Path(args.screening_output), Path(args.drafts_output), Path(args.readme_output),
            seed=args.seed, target_threads=args.target_threads, target_messages=args.target_messages,
        )
    except (OSError, ValueError, sqlite3.Error, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())