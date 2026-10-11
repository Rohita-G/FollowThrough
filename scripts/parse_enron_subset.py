#!/usr/bin/env python3
"""Create a deterministic, resumable mailbox-prioritized Enron subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.explore_data import _parse_enron_file  # noqa: E402

MAILDIR = ROOT / "data" / "raw" / "maildir"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "enron_subset"
SUCCESS_CAP = 100_000
PRIORITY_MAILBOXES = ("kaminski-v", "dasovich-j", "shackleton-s", "germany-c", "beck-s")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, path)


def _mailbox_order(source: Path) -> List[str]:
    with os.scandir(source) as entries:
        mailboxes = sorted(
            entry.name for entry in entries
            if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False)
        )
    present = set(mailboxes)
    return [name for name in PRIORITY_MAILBOXES if name in present] + [
        name for name in mailboxes if name not in PRIORITY_MAILBOXES
    ]


def _mailbox_files(mailbox_root: Path) -> List[Path]:
    files: List[Path] = []
    pending = [mailbox_root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                dirs: List[Path] = []
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            files.append(Path(entry.path))
                    except OSError as exc:
                        raise RuntimeError(f"Cannot inspect source entry {entry.path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Cannot scan mailbox directory {directory}: {exc}") from exc
        pending.extend(reversed(sorted(dirs, key=lambda path: path.name)))
    return sorted(files, key=lambda path: path.relative_to(mailbox_root).as_posix())


def _read_jsonl(path: Path, expected_mailbox: str | None = None) -> Iterator[Dict[str, Any]]:
    """Validate a JSONL ledger and explicitly recover only an incomplete tail."""
    if not path.exists():
        return
    with path.open("r+b") as source:
        while True:
            line_start = source.tell()
            raw_line = source.readline()
            if not raw_line:
                break
            has_newline = raw_line.endswith(b"\n")
            content = raw_line[:-1] if has_newline else raw_line
            try:
                record = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if not has_newline and source.read(1) == b"":
                    source.seek(line_start)
                    source.truncate()
                    source.flush()
                    os.fsync(source.fileno())
                    print(
                        f"RECOVERY: truncated incomplete final JSONL fragment in {path.name} "
                        f"at byte {line_start}; it was not accepted as a record.",
                        flush=True,
                    )
                    break
                raise RuntimeError(f"Corrupt JSONL record in {path} at byte {line_start}: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"Non-object JSONL record in {path} at byte {line_start}")
            if not isinstance(record.get("source_relative_path"), str):
                raise RuntimeError(f"Record lacks source_relative_path in {path} at byte {line_start}")
            rel_path = Path(record["source_relative_path"])
            if rel_path.is_absolute() or ".." in rel_path.parts or not rel_path.parts:
                raise RuntimeError(f"Unsafe source_relative_path in {path}: {record['source_relative_path']!r}")
            if record.get("mailbox") != rel_path.parts[0]:
                raise RuntimeError(f"Mailbox/source path mismatch in {path}: {record['source_relative_path']!r}")
            if expected_mailbox is not None and record["mailbox"] != expected_mailbox:
                raise RuntimeError(f"Record in {path} belongs to unexpected mailbox {record['mailbox']!r}")
            if not has_newline:
                source.seek(0, os.SEEK_END)
                source.write(b"\n")
                source.flush()
                os.fsync(source.fileno())
                print(f"RECOVERY: added missing newline to valid final record in {path.name}.", flush=True)
            yield record


def _load_ledgers(output_dir: Path, mailboxes: List[str]) -> Tuple[Set[str], Set[str], Counter[str], Counter[str]]:
    success_paths: Set[str] = set()
    failure_paths: Set[str] = set()
    successes_by_mailbox: Counter[str] = Counter()
    failures_by_mailbox: Counter[str] = Counter()
    all_mailboxes = set(mailboxes)
    by_user = output_dir / "by_user"

    if by_user.exists():
        for path in sorted(by_user.glob("*.jsonl")):
            mailbox = path.stem
            if mailbox not in all_mailboxes:
                raise RuntimeError(f"Existing subset output belongs to unknown mailbox: {path}")
            for record in _read_jsonl(path, expected_mailbox=mailbox):
                if "Body" not in record or "Subject" not in record:
                    raise RuntimeError(f"Existing email record is missing expected parsed fields: {path}")
                rel_path = record["source_relative_path"]
                if rel_path in success_paths:
                    raise RuntimeError(f"Duplicate successful source path in subset outputs: {rel_path}")
                success_paths.add(rel_path)
                successes_by_mailbox[mailbox] += 1

    failures_path = output_dir / "failures.jsonl"
    for record in _read_jsonl(failures_path):
        mailbox = record.get("mailbox")
        if mailbox not in all_mailboxes or not record.get("reason"):
            raise RuntimeError(f"Invalid failure record in {failures_path}")
        rel_path = record["source_relative_path"]
        if rel_path in failure_paths or rel_path in success_paths:
            raise RuntimeError(f"Duplicate or conflicting source path in subset ledgers: {rel_path}")
        failure_paths.add(rel_path)
        failures_by_mailbox[mailbox] += 1

    return success_paths, failure_paths, successes_by_mailbox, failures_by_mailbox


def _write_checkpoint(
    output_dir: Path,
    max_successful: int,
    success_paths: Set[str],
    failure_paths: Set[str],
    mailbox_states: Dict[str, str],
    current_mailbox: str | None,
    last_source_path: str | None,
) -> None:
    _atomic_json(output_dir / "checkpoint.json", {
        "version": 1,
        "source_dataset_path": str(MAILDIR.relative_to(ROOT)),
        "maximum_successful_records": max_successful,
        "attempted_files": len(success_paths) + len(failure_paths),
        "successful_records": len(success_paths),
        "failed_files": len(failure_paths),
        "current_mailbox": current_mailbox,
        "last_source_relative_path": last_source_path,
        "mailbox_states": mailbox_states,
        "updated_at": _utc_now(),
    })


def _validate_metadata(output_dir: Path, max_successful: int, success_paths: Set[str], failure_paths: Set[str]) -> Dict[str, Any] | None:
    actual_attempted = len(success_paths) + len(failure_paths)
    actual_successful = len(success_paths)
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Cannot safely resume: invalid checkpoint {checkpoint_path}: {exc}") from exc
        if checkpoint.get("version") != 1 or checkpoint.get("maximum_successful_records") != max_successful:
            raise RuntimeError("Cannot safely resume: checkpoint version or requested record cap differs")
        if checkpoint.get("attempted_files", 0) > actual_attempted:
            raise RuntimeError("Cannot safely resume: checkpoint is ahead of the validated success/failure ledgers")
        if checkpoint.get("successful_records", 0) > actual_successful or checkpoint.get("failed_files", 0) > len(failure_paths):
            raise RuntimeError("Cannot safely resume: checkpoint counts exceed validated ledger counts")
        last_path = checkpoint.get("last_source_relative_path")
        if last_path and last_path not in success_paths and last_path not in failure_paths:
            raise RuntimeError("Cannot safely resume: checkpoint's last source path is absent from both ledgers")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Cannot safely resume: invalid manifest {manifest_path}: {exc}") from exc
        if manifest.get("requested_maximum_successful_records") != max_successful:
            raise RuntimeError("Cannot safely resume: manifest requested record cap differs")
        if manifest.get("attempted_files", 0) > actual_attempted or manifest.get("actual_successful_records", 0) > actual_successful:
            raise RuntimeError("Cannot safely resume: manifest is ahead of the validated ledgers")
        if manifest.get("failed_files", 0) > len(failure_paths):
            raise RuntimeError("Cannot safely resume: manifest failure count exceeds the failure ledger")
        return manifest
    return None


def _progress(mailbox: str, attempted: int, successful: int, failed: int, started: float) -> None:
    elapsed = time.monotonic() - started
    print(
        f"mailbox={mailbox} attempted={attempted} successful={successful} failures={failed} "
        f"elapsed_seconds={elapsed:.1f} cumulative_successful={successful}",
        flush=True,
    )


def run_subset(
    output_dir: Path,
    max_successful: int = SUCCESS_CAP,
    max_attempts: int | None = None,
    max_mailboxes: int | None = None,
) -> Dict[str, Any]:
    if not MAILDIR.is_dir():
        raise FileNotFoundError(f"Expected already-extracted Enron maildir at {MAILDIR}; no extraction was attempted")
    if not 1 <= max_successful <= SUCCESS_CAP:
        raise ValueError(f"max_successful must be between 1 and {SUCCESS_CAP}")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if max_mailboxes is not None and max_mailboxes < 1:
        raise ValueError("max_mailboxes must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_user").mkdir(parents=True, exist_ok=True)
    all_mailboxes = _mailbox_order(MAILDIR)
    success_paths, failure_paths, successes_by_mailbox, failures_by_mailbox = _load_ledgers(output_dir, all_mailboxes)
    if len(success_paths) > max_successful:
        raise RuntimeError("Existing valid subset records exceed the requested successful-record cap")
    previous_manifest = _validate_metadata(output_dir, max_successful, success_paths, failure_paths)

    if len(success_paths) == max_successful and previous_manifest is not None:
        print(
            f"Subset already has {len(success_paths)} valid records at the requested cap; "
            "no records added.",
            flush=True,
        )
        return previous_manifest

    selected_mailboxes = all_mailboxes[:max_mailboxes] if max_mailboxes is not None else all_mailboxes
    mailbox_states: Dict[str, str] = {}
    if previous_manifest:
        mailbox_states.update(previous_manifest.get("mailboxes", {}))
        mailbox_states = {
            name: data.get("status", "not_started") if isinstance(data, dict) else data
            for name, data in mailbox_states.items()
        }
    attempted_this_run = 0
    started = time.monotonic()
    current_mailbox: str | None = None
    last_source_path: str | None = None
    stop_reason: str | None = None

    try:
        for mailbox_index, mailbox in enumerate(selected_mailboxes):
            current_mailbox = mailbox
            mailbox_root = MAILDIR / mailbox
            files = _mailbox_files(mailbox_root)
            mailbox_states[mailbox] = "completed"
            for file_index, path in enumerate(files):
                relative_path = path.relative_to(MAILDIR).as_posix()
                if relative_path in success_paths or relative_path in failure_paths:
                    continue
                if len(success_paths) >= max_successful:
                    stop_reason = "maximum_successful_records_reached"
                    mailbox_states[mailbox] = "stopped_partway"
                    break
                if max_attempts is not None and attempted_this_run >= max_attempts:
                    stop_reason = "smoke_attempt_limit_reached"
                    mailbox_states[mailbox] = "stopped_partway"
                    break

                parsed = _parse_enron_file(path)
                if parsed.get("status") == "parsed":
                    if len(success_paths) >= max_successful:
                        stop_reason = "maximum_successful_records_reached"
                        mailbox_states[mailbox] = "stopped_partway"
                        break
                    record = dict(parsed)
                    record.pop("status", None)
                    record.pop("path", None)
                    record["source_relative_path"] = relative_path
                    record["mailbox"] = mailbox
                    with (output_dir / "by_user" / f"{mailbox}.jsonl").open("a", encoding="utf-8", buffering=1) as output:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                    success_paths.add(relative_path)
                    successes_by_mailbox[mailbox] += 1
                else:
                    reason = str(parsed.get("reason") or "unknown_parse_failure")
                    failure_record = {
                        "source_relative_path": relative_path,
                        "mailbox": mailbox,
                        "reason": reason,
                        "detail": str(parsed.get("detail", ""))[:500],
                    }
                    with (output_dir / "failures.jsonl").open("a", encoding="utf-8", buffering=1) as output:
                        output.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                        output.flush()
                    failure_paths.add(relative_path)
                    failures_by_mailbox[mailbox] += 1

                attempted_this_run += 1
                last_source_path = relative_path
                attempted_total = len(success_paths) + len(failure_paths)
                if attempted_total % 1000 == 0:
                    _progress(mailbox, attempted_total, len(success_paths), len(failure_paths), started)
                    _write_checkpoint(output_dir, max_successful, success_paths, failure_paths, mailbox_states, mailbox, last_source_path)

                if len(success_paths) >= max_successful:
                    stop_reason = "maximum_successful_records_reached"
                    mailbox_states[mailbox] = "completed" if file_index == len(files) - 1 else "stopped_partway"
                    _write_checkpoint(output_dir, max_successful, success_paths, failure_paths, mailbox_states, mailbox, last_source_path)
                    break
                if max_attempts is not None and attempted_this_run >= max_attempts:
                    stop_reason = "smoke_attempt_limit_reached"
                    mailbox_states[mailbox] = "stopped_partway" if file_index < len(files) - 1 else "completed"
                    _write_checkpoint(output_dir, max_successful, success_paths, failure_paths, mailbox_states, mailbox, last_source_path)
                    break

            _progress(mailbox, len(success_paths) + len(failure_paths), len(success_paths), len(failure_paths), started)
            if mailbox_states.get(mailbox) == "completed":
                _write_checkpoint(output_dir, max_successful, success_paths, failure_paths, mailbox_states, mailbox, last_source_path)
            if stop_reason:
                break
            if max_mailboxes is not None and mailbox_index + 1 >= max_mailboxes and len(selected_mailboxes) < len(all_mailboxes):
                stop_reason = "smoke_mailbox_limit_reached"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        if current_mailbox:
            mailbox_states[current_mailbox] = "stopped_partway"
        _write_checkpoint(output_dir, max_successful, success_paths, failure_paths, mailbox_states, current_mailbox, last_source_path)
        print("Interrupted cleanly; validated JSONL records are saved and can be resumed.", flush=True)

    if stop_reason is None:
        stop_reason = "all_mailboxes_exhausted"
    for mailbox in all_mailboxes:
        mailbox_states.setdefault(mailbox, "not_started")

    attempted_total = len(success_paths) + len(failure_paths)
    reason_counts = Counter(
        record["reason"] for record in _read_jsonl(output_dir / "failures.jsonl")
    )
    included = [mailbox for mailbox in all_mailboxes if successes_by_mailbox[mailbox] or failures_by_mailbox[mailbox]]
    per_mailbox = {
        mailbox: {
            "successful_records": successes_by_mailbox[mailbox],
            "failed_files": failures_by_mailbox[mailbox],
            "attempted_files": successes_by_mailbox[mailbox] + failures_by_mailbox[mailbox],
            "status": mailbox_states[mailbox],
        }
        for mailbox in all_mailboxes
    }
    complete = stop_reason in {"maximum_successful_records_reached", "all_mailboxes_exhausted"}
    manifest = {
        "source_dataset_path": str(MAILDIR.relative_to(ROOT)),
        "requested_maximum_successful_records": max_successful,
        "actual_successful_records": len(success_paths),
        "attempted_files": attempted_total,
        "failed_files": len(failure_paths),
        "failure_counts_by_reason": dict(sorted(reason_counts.items())),
        "mailbox_processing_order": all_mailboxes,
        "mailboxes_included": included,
        "mailboxes": per_mailbox,
        "selection_method": "Prioritized mailboxes first (kaminski-v, dasovich-j, shackleton-s, germany-c, beck-s when present), then remaining mailboxes alphabetically; files sorted by source-relative path within each mailbox; sequential parsing stops at the global successful-record cap.",
        "completion_status": "complete" if complete else "paused",
        "stop_reason": stop_reason,
        "updated_at": _utc_now(),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _write_checkpoint(output_dir, max_successful, success_paths, failure_paths, mailbox_states, current_mailbox, last_source_path)
    print(
        f"SUBSET {manifest['completion_status']}: attempted={attempted_total} "
        f"successful={len(success_paths)} failures={len(failure_paths)} stop_reason={stop_reason}",
        flush=True,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Subset output directory (default is the production directory).")
    parser.add_argument("--max-successful", type=int, default=SUCCESS_CAP, help=f"Hard cap, at most {SUCCESS_CAP} successful email records.")
    parser.add_argument("--max-attempts", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-mailboxes", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    try:
        run_subset(output_dir, args.max_successful, args.max_attempts, args.max_mailboxes)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())