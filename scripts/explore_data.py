#!/usr/bin/env python3
"""Phase 1 corpus explorer for FollowThrough.

This script reads the raw Enron, Parakweet, and AMI archives and reports the counts
needed for the project. It is intentionally dependency-light and supports a fast
`--limit N` subset scan so the initial validation can finish in a laptop-friendly
window while the full run is kept as a separate command.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def split_sentences(text: str) -> List[str]:
    text = (text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def normalize_subject(subject: str) -> str:
    s = (subject or "").strip()
    while True:
        updated = re.sub(r"^(?:re|fw|fwd)\s*:\s*", "", s, flags=re.I)
        if updated == s:
            break
        s = updated.strip()
    return re.sub(r"\s+", " ", s).strip()


def thread_key_for_email(msg: Dict[str, str]) -> Tuple[str, str]:
    subject = normalize_subject(msg.get("Subject", ""))
    if subject:
        return ("subject", subject)
    return ("no-subject", str(msg.get("Message-ID") or msg.get("path") or "unknown"))


def _parse_enron_file(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_bytes()
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        if not msg or not any(msg.get(name) for name in ("From", "Date", "Subject", "Message-ID")):
            return {"status": "failure", "path": str(path), "reason": "missing_email_headers"}

        body = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        payload = part.get_payload() or ""
                    if isinstance(payload, bytes):
                        payload = payload.decode("utf-8", errors="replace")
                    if isinstance(payload, str):
                        body = payload
                        break
            if body is None:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            payload = payload.decode("utf-8", errors="replace")
                        elif payload is None:
                            payload = part.get_payload() or ""
                        if isinstance(payload, str):
                            body = payload
                            break
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body = payload.decode("utf-8", errors="replace")
            elif isinstance(payload, str):
                body = payload

        if body is None:
            body = ""

        body = html.unescape(re.sub(r"<[^>]+>", " ", body))
        body = re.sub(r"\s+", " ", body).strip()

        return {
            "status": "parsed",
            "path": str(path),
            "Subject": msg.get("Subject", ""),
            "Message-ID": msg.get("Message-ID", ""),
            "Date": msg.get("Date", ""),
            "From": msg.get("From", ""),
            "To": msg.get("To", ""),
            "Cc": msg.get("Cc", ""),
            "References": msg.get("References", ""),
            "In-Reply-To": msg.get("In-Reply-To", ""),
            "Body": body,
        }
    except Exception as exc:  # pragma: no cover - runtime safety for corrupt mail files
        return {"status": "failure", "path": str(path), "reason": type(exc).__name__, "detail": str(exc)}


def _iter_enron_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            dirs = []
            files = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        files.append(Path(entry.path))
                except OSError:
                    logging.warning("Unable to inspect Enron path %s", entry.path)
            yield from sorted(files, key=str)
            pending.extend(reversed(sorted(dirs, key=str)))


def _load_enron_cache(cache_path: Path) -> set[str]:
    completed_paths: set[str] = set()
    if not cache_path.exists():
        return completed_paths
    with cache_path.open("r+b") as cache_file:
        cache_file.seek(0, os.SEEK_END)
        size = cache_file.tell()
        if size:
            cache_file.seek(-1, os.SEEK_END)
            if cache_file.read(1) != b"\n":
                position = size - 1
                while position >= 0:
                    cache_file.seek(position)
                    if cache_file.read(1) == b"\n":
                        cache_file.truncate(position + 1)
                        break
                    position -= 1
                else:
                    cache_file.truncate(0)
        cache_file.seek(0)
        for line in cache_file:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if record.get("path"):
                completed_paths.add(record["path"])
    return completed_paths


def _summarize_enron_cache(cache_path: Path) -> Dict[str, Any]:
    files_scanned = 0
    parse_failures: Counter[str] = Counter()
    unique_message_ids = set()
    thread_sizes: Counter[Tuple[str, str]] = Counter()
    sentence_count = 0
    token_count = 0
    with cache_path.open("r", encoding="utf-8") as cache_file:
        for line in cache_file:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            files_scanned += 1
            if result.get("status") != "parsed":
                parse_failures[result.get("reason", "unknown_error")] += 1
                continue
            message_id = str(result.get("Message-ID", "")).strip()
            if message_id:
                unique_message_ids.add(message_id)
            thread_sizes[thread_key_for_email(result)] += 1
            sentence_count += len(split_sentences(result.get("Body", "")))
            token_count += len(tokenize(result.get("Body", "")))

    thread_counts = Counter(thread_sizes.values())
    sizes = list(thread_sizes.values())
    return {
        "files_scanned": files_scanned,
        "successfully_parsed": sum(thread_sizes.values()),
        "parse_failures": {
            "total": sum(parse_failures.values()),
            "top_reasons": [{"reason": reason, "count": count} for reason, count in parse_failures.most_common(3)],
        },
        "unique_message_ids": len(unique_message_ids),
        "threads_after_subject_normalization": len(thread_sizes),
        "messages_per_thread": {
            "bins": {
                "1": thread_counts.get(1, 0),
                "2": thread_counts.get(2, 0),
                "3-5": sum(count for size, count in thread_counts.items() if 3 <= size <= 5),
                "6+": sum(count for size, count in thread_counts.items() if size >= 6),
            },
            "median": statistics.median(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
        },
        "first_pass_body_extraction": {"sentence_count": sentence_count, "token_count": token_count},
        "index_jsonl": str(cache_path.resolve().relative_to(ROOT)),
    }


def collect_enron(
    root: Path,
    limit: int | None = None,
    batch_size: int = 10000,
    workers: int | None = None,
    sequential: bool = True,
    cache_path: Path | None = None,
    from_cache: bool = False,
) -> Dict[str, Any]:
    tarball = root / "enron_mail_20150507.tar.gz"
    extracted_dir = root / "enron_mail_20150507"
    maildir = root / "maildir"
    if tarball.exists() and not (extracted_dir.exists() or maildir.exists()):
        logging.info("Extracting Enron archive %s", tarball)
        import tarfile

        with tarfile.open(tarball, "r:gz") as zf:
            zf.extractall(root)

    source_dir = maildir if maildir.exists() else extracted_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"No extracted Enron corpus at {source_dir}. Download it first with scripts/download_data.py")

    cache_path = cache_path or PROCESSED / (f"enron_parse.limit{limit}.jsonl" if limit is not None else "enron_parse.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    completed_paths = _load_enron_cache(cache_path)
    if from_cache:
        if not completed_paths:
            raise FileNotFoundError(f"No Enron parse cache found at {cache_path}")
        return _summarize_enron_cache(cache_path)

    paths = list(_iter_enron_files(source_dir))
    if limit is not None:
        paths = paths[:limit]
    pending_paths = [path for path in paths if str(path) not in completed_paths]
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)

    logging.info(
        "Enron files=%d cached=%d pending=%d workers=%d mode=%s batch_size=%d",
        len(paths), len(completed_paths), len(pending_paths), worker_count, "sequential" if sequential else "pool", batch_size,
    )
    for batch_start in range(0, len(pending_paths), batch_size):
        batch = pending_paths[batch_start:batch_start + batch_size]
        processed_in_batch = 0
        with cache_path.open("a", encoding="utf-8", buffering=1) as output:
            try:
                if sequential:
                    result_iterator = map(_parse_enron_file, batch)
                    for record in result_iterator:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed_paths.add(record["path"])
                        processed_in_batch += 1
                else:
                    with Pool(processes=worker_count) as pool:
                        for record in pool.imap_unordered(_parse_enron_file, batch, chunksize=500):
                            output.write(json.dumps(record, ensure_ascii=False) + "\n")
                            completed_paths.add(record["path"])
                            processed_in_batch += 1
            except KeyboardInterrupt:
                output.flush()
                os.fsync(output.fileno())
                logging.warning("Interrupted; saved %d completed records from the active batch", processed_in_batch)
                raise
            output.flush()
            os.fsync(output.fileno())
        processed = min(batch_start + processed_in_batch, len(pending_paths))
        logging.info("Checkpointed %d/%d pending files (%d total cached)", processed, len(pending_paths), len(completed_paths))

    summary = _summarize_enron_cache(cache_path)
    logging.info("Wrote Enron index cache to %s", cache_path)
    return summary


def benchmark_enron(root: Path, count: int = 20000, workers: int | None = None) -> Dict[str, Any]:
    source_dir = root / "maildir"
    if not source_dir.exists():
        raise FileNotFoundError(f"No extracted Enron maildir at {source_dir}")
    paths = list(_iter_enron_files(source_dir))[:count]
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    measurements = {}
    for mode in ("sequential", "pool_imap_unordered"):
        started = time.perf_counter()
        if mode == "sequential":
            processed = sum(1 for _ in map(_parse_enron_file, paths))
        else:
            with Pool(processes=worker_count) as pool:
                processed = sum(1 for _ in pool.imap_unordered(_parse_enron_file, paths, chunksize=500))
        elapsed = time.perf_counter() - started
        measurements[mode] = {
            "files": processed,
            "seconds": round(elapsed, 3),
            "files_per_second": round(processed / elapsed, 2) if elapsed else 0,
        }
        print(f"{mode}: {processed} files in {elapsed:.2f}s ({processed / elapsed:.2f} files/s)")
    measurements["workers"] = worker_count
    return measurements


def count_ami(root: Path) -> Dict[str, Any]:
    zip_path = root / "ami_public_manual_1.6.2.zip"
    if not (root / "words").exists() and zip_path.exists():
        logging.info("Extracting AMI archive %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)

    def xml_files(directory: Path, pattern: str) -> List[Path]:
        return sorted(directory.glob(pattern)) if directory.exists() else []

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].split(":")[-1]

    words_files = xml_files(root / "words", "*.words.xml")
    segment_files = xml_files(root / "segments", "*.segments.xml")
    dact_files = xml_files(root / "dialogueActs", "*.dialog-act.xml")
    decision_files = xml_files(root / "decision" / "manual", "*.decision.xml")
    summary_files = xml_files(root / "abstractive", "*.abssumm.xml")
    if not words_files:
        raise FileNotFoundError(f"AMI words XML files were not found under {root / 'words'}")

    meeting_ids = {re.match(r"([A-Z]{2}\d+[a-z])", path.name).group(1) for path in words_files if re.match(r"([A-Z]{2}\d+[a-z])", path.name)}

    def count_elements(paths: List[Path], target: str) -> int:
        total = 0
        for path in paths:
            try:
                total += sum(1 for element in ET.parse(path).getroot().iter() if local_name(element.tag) == target)
            except ET.ParseError as exc:
                logging.warning("AMI XML parse failure for %s: %s", path, exc)
        return total

    action_items: List[str] = []
    decision_examples: List[str] = []
    total_words = count_elements(words_files, "w")
    for path in summary_files:
        try:
            root_element = ET.parse(path).getroot()
        except ET.ParseError as exc:
            logging.warning("AMI summary parse failure for %s: %s", path, exc)
            continue
        for section in root_element:
            section_name = local_name(section.tag)
            if section_name not in {"actions", "decisions"}:
                continue
            target = action_items if section_name == "actions" else decision_examples
            target.extend(
                " ".join((sentence.text or "").split())
                for sentence in section.iter()
                if local_name(sentence.tag) == "sentence" and (sentence.text or "").strip()
            )

    rng = random.Random(42)
    return {
        "meetings": len(meeting_ids),
        "utterances": count_elements(segment_files, "segment"),
        "dialogue_acts": count_elements(dact_files, "dact"),
        "decision_annotations": count_elements(decision_files, "decision"),
        "abstractive_action_items": {
            "count": len(action_items),
            "examples": rng.sample(action_items, min(5, len(action_items))),
        },
        "decision_examples": rng.sample(decision_examples, min(5, len(decision_examples))),
        "total_words": total_words,
        "count_sources": {
            "utterances": "segments/*.segments.xml <segment>",
            "dialogue_acts": "dialogueActs/*.dialog-act.xml <dact>",
            "decision_annotations": "decision/manual/*.decision.xml <decision>",
            "action_items": "abstractive/*.abssumm.xml <sentence> inside <actions>",
            "words": "words/*.words.xml <w>",
        },
    }


def count_parakweet(root: Path) -> Dict[str, Any]:
    repo_zip = root / "EmailIntentDataSet-master.zip"
    raw_file = root / "parakweet" / "Ask0729-fixed.txt"
    train_name = "Ask0729-fixed.txt"
    test_name = "testSet-qualifiedBatch-fixed.txt"
    test_lines: List[str] = []
    if repo_zip.exists():
        with zipfile.ZipFile(repo_zip) as zf:
            members = {Path(name).name: name for name in zf.namelist() if "/src/resources/" in name}
            test_member = members.get(test_name)
            if test_member:
                test_lines = zf.read(test_member).decode("utf-8", errors="replace").splitlines()
    train_lines = raw_file.read_text(encoding="utf-8", errors="replace").splitlines() if raw_file.exists() else []

    def parse_rows(lines: List[str]) -> List[Dict[str, str]]:
        rows = []
        for line in lines:
            if not line.strip():
                continue
            label, separator, sentence = line.partition("\t")
            if not separator:
                logging.warning("Skipping malformed Parakweet row: %r", line[:120])
                continue
            rows.append({"label": label.strip(), "sentence": sentence.strip()})
        return rows

    train = parse_rows(train_lines)
    test = parse_rows(test_lines)
    all_rows = train + test
    rng = random.Random(42)
    return {
        "total_sentences": len(all_rows),
        "label_distribution": dict(sorted(Counter(row["label"] for row in all_rows).items())),
        "train": {
            "file": f"src/resources/{train_name}",
            "local_path": str(raw_file.relative_to(ROOT)) if raw_file.exists() else None,
            "sentences": len(train),
            "label_distribution": dict(sorted(Counter(row["label"] for row in train).items())),
        },
        "test": {
            "file": f"src/resources/{test_name}",
            "archive": str(repo_zip.relative_to(ROOT)),
            "sentences": len(test),
            "label_distribution": dict(sorted(Counter(row["label"] for row in test).items())),
        },
        "format": "one UTF-8 row per example: label<TAB>sentence; labels are No and Yes",
        "random_labeled_examples_seed_42": rng.sample(all_rows, min(10, len(all_rows))),
    }


def print_table(rows: List[Tuple[Any, ...]]) -> None:
    if len(rows) < 2:
        return
    header = rows[0]
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(header))]
    print("  ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for row in rows[1:]:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the FollowThrough data sources.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N Enron files for a quick validation run.")
    parser.add_argument("--log", type=str, default="data/processed/explore.log", help="Log file for progress updates.")
    parser.add_argument("--stats-output", type=str, default=None, help="Optional summary JSON path; limited runs default to results/data_stats.limitN.json.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark sequential and multiprocessing parsing on a sample.")
    parser.add_argument("--benchmark-files", type=int, default=20000, help="Number of real Enron files to benchmark.")
    parser.add_argument("--workers", type=int, default=None, help="Pool workers; defaults to CPU count minus one.")
    parser.add_argument("--batch-size", type=int, default=10000, help="Files per durable JSONL checkpoint.")
    parser.add_argument("--parallel", action="store_true", help="Use Pool.imap_unordered instead of the measured faster sequential mode.")
    parser.add_argument("--enron-only", action="store_true", help="Run only the Enron parse and write results/enron_stats.json.")
    parser.add_argument("--from-cache", action="store_true", help="Summarize the Enron JSONL cache without scanning or parsing raw files.")
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )

    if args.benchmark:
        benchmark_enron(RAW, count=args.benchmark_files, workers=args.workers)
        return 0

    enron_cache = PROCESSED / (f"enron_parse.limit{args.limit}.jsonl" if args.limit is not None else "enron_parse.jsonl")
    enron = collect_enron(
        RAW,
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
        sequential=not args.parallel,
        cache_path=enron_cache,
        from_cache=args.from_cache,
    )
    if args.enron_only:
        summary_path = ROOT / args.stats_output if args.stats_output else RESULTS / "enron_stats.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(enron, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(enron, indent=2, ensure_ascii=False))
        print(f"Summary written to {summary_path}")
        return 0

    ami = count_ami(RAW)
    parakweet = count_parakweet(RAW)

    summary = {"enron": enron, "ami": ami, "parakweet": parakweet}
    default_stats_name = f"data_stats.limit{args.limit}.json" if args.limit is not None else "data_stats.json"
    summary_path = ROOT / args.stats_output if args.stats_output else RESULTS / default_stats_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Enron ===")
    print_table([
        ("metric", "value"),
        ("files_scanned", enron["files_scanned"]),
        ("successfully_parsed", enron["successfully_parsed"]),
        ("parse_failures", enron["parse_failures"]["total"]),
        ("top_failure_reasons", json.dumps(enron["parse_failures"]["top_reasons"])),
        ("unique_message_ids", enron["unique_message_ids"]),
        ("threads_after_subject_normalization", enron["threads_after_subject_normalization"]),
        ("thread_size_bins", json.dumps(enron["messages_per_thread"]["bins"])),
        ("median_msgs_per_thread", enron["messages_per_thread"]["median"]),
        ("max_msgs_per_thread", enron["messages_per_thread"]["max"]),
        ("sentence_count_after_first_pass", enron["first_pass_body_extraction"]["sentence_count"]),
        ("token_count_after_first_pass", enron["first_pass_body_extraction"]["token_count"]),
    ])

    print("\n=== AMI ===")
    print_table([
        ("metric", "value"),
        ("meetings", ami["meetings"]),
        ("utterances", ami["utterances"]),
        ("dialogue_acts", ami["dialogue_acts"]),
        ("decision_annotations", ami["decision_annotations"]),
        ("action_items_in_abstractive_summaries", ami["abstractive_action_items"]["count"]),
        ("total_words", ami["total_words"]),
    ])

    print("\n=== Parakweet ===")
    print_table([
        ("metric", "value"),
        ("total_sentences", parakweet["total_sentences"]),
        ("label_distribution", json.dumps(parakweet["label_distribution"], sort_keys=True)),
        ("format", parakweet["format"]),
        ("train_file", parakweet["train"]["file"]),
        ("test_file", parakweet["test"]["file"]),
    ])

    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
