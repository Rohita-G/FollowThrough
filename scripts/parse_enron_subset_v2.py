#!/usr/bin/env python3
"""Parse the existing Enron subset into a separate, line-preserving v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V1_SUBSET = ROOT / "data" / "processed" / "enron_subset"
MAILDIR = ROOT / "data" / "raw" / "maildir"
SUCCESS_LIMIT = 100_000
FORWARD_MARKER = re.compile(r"(?im)^\s*(?:-{2,}\s*forwarded message\s*-{2,}|begin forwarded message\s*:?)\s*$")
ORIGINAL_MARKER = re.compile(r"(?im)^\s*-{2,}\s*original message\s*-{2,}\s*$")
REPLY_INTRO = re.compile(r"(?im)^\s*on .{1,240}\bwrote\s*:\s*$")
QUOTED_LINE = re.compile(r"^\s*>+")
HEADER_LINE = re.compile(r"(?i)^\s*(from|sent|date|to|cc|subject|reply-to):\s*.+$")
SIGNATURE_SEPARATOR = re.compile(r"^\s*--\s*$")
MOBILE_SIGNATURE = re.compile(r"(?i)^\s*sent from my (?:iphone|ipad|android|blackberry).*$")
SIGNOFF = re.compile(r"(?i)^\s*(?:best(?: regards)?|kind regards|regards|sincerely|thanks|thank you),?\s*$")
LEGAL_FOOTER = re.compile(
    r"(?i)^\s*(?:this (?:e-?mail|message)(?: and any attachments)? is confidential|"
    r"if you are not the intended recipient|confidentiality notice|disclaimer)\b.*$"
)
TASK_CUE = re.compile(
    r"(?i)\b(?:please|can you|could you|will send|i['’]ll|deadline|due|cancel|"
    r"complete|completed|reassign|instead|action item|need to)\b"
)
ADDRESS = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
WORD = re.compile(r"\b[\w’'-]+\b", re.UNICODE)


class _HTMLToText(HTMLParser):
    BLOCK_TAGS = {"address", "article", "br", "div", "footer", "h1", "h2", "h3", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _decode_part(part: Any) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_body(message: Any) -> Tuple[str, str]:
    warnings: List[str] = []
    if message.is_multipart():
        plain_parts = [
            part for part in message.walk()
            if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment"
        ]
        if plain_parts:
            return "\n\n".join(_decode_part(part) for part in plain_parts).replace("\r\n", "\n").replace("\r", "\n"), "plain"
        html_parts = [
            part for part in message.walk()
            if part.get_content_type() == "text/html" and part.get_content_disposition() != "attachment"
        ]
        if html_parts:
            parser = _HTMLToText()
            for part in html_parts:
                try:
                    parser.feed(_decode_part(part))
                except Exception:
                    warnings.append("html_conversion_error")
            return "".join(parser.parts).replace("\r\n", "\n").replace("\r", "\n"), "html_fallback"
        return "", "no_text_part"
    if message.get_content_type() == "text/html":
        parser = _HTMLToText()
        parser.feed(_decode_part(message))
        return "".join(parser.parts).replace("\r\n", "\n").replace("\r", "\n"), "html_fallback"
    return _decode_part(message).replace("\r\n", "\n").replace("\r", "\n"), "plain"


def clean_body_v2(body: str) -> Dict[str, Any]:
    """Remove only line-identified quoted tails/lines and strong trailing boilerplate markers.

    Forwarded-message payloads and embedded reply-header blocks are preserved because
    they can contain task evidence; those cases receive explicit review flags instead.
    Signature/disclaimer deletion is restricted to recognizable terminal regions.
    """
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    flags: List[str] = []
    uncertain: List[str] = []

    cut_index = len(lines)
    for index, line in enumerate(lines):
        if ORIGINAL_MARKER.match(line) or REPLY_INTRO.match(line):
            cut_index = index
            flags.append("quoted_reply_tail_removed")
            break
        if FORWARD_MARKER.match(line):
            flags.append("forwarded_payload_preserved_for_review")
            uncertain.append("forwarded_payload_preserved")
        if HEADER_LINE.match(line) and re.match(r"(?i)^\s*from:", line):
            next_headers = sum(1 for candidate in lines[index + 1:index + 5] if HEADER_LINE.match(candidate))
            if next_headers >= 2:
                uncertain.append("embedded_message_header_preserved")

    lines = lines[:cut_index]
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    if nonblank:
        tail_start = nonblank[max(0, int(len(nonblank) * 0.60))]
        for index in range(tail_start, len(lines)):
            if SIGNATURE_SEPARATOR.match(lines[index]) or MOBILE_SIGNATURE.match(lines[index]):
                suffix = "\n".join(lines[index + 1:])
                if not TASK_CUE.search(suffix):
                    lines = lines[:index]
                    flags.append("terminal_signature_removed")
                else:
                    uncertain.append("signature_marker_with_task_cue_preserved")
                break
            if LEGAL_FOOTER.match(lines[index]):
                suffix = "\n".join(lines[index:])
                if index >= tail_start and not re.search(r"(?i)\b(?:please|deadline|due|will send|i['’]ll)\b", suffix):
                    lines = lines[:index]
                    flags.append("terminal_legal_footer_removed")
                else:
                    uncertain.append("legal_footer_candidate_preserved")
                break
            if SIGNOFF.match(lines[index]) and len(lines) - index <= 5:
                suffix = "\n".join(lines[index:])
                if not TASK_CUE.search(suffix):
                    lines = lines[:index]
                    flags.append("terminal_signoff_removed")
                else:
                    uncertain.append("signoff_with_task_cue_preserved")
                break

    kept: List[str] = []
    removed_quote_lines = 0
    for line in lines:
        if QUOTED_LINE.match(line):
            removed_quote_lines += 1
            continue
        kept.append(line.rstrip())
    if removed_quote_lines:
        flags.append("quoted_prefixed_lines_removed")
    cleaned = "\n".join(kept).strip()
    if not cleaned:
        uncertain.append("cleaned_body_empty_review")
    if TASK_CUE.search(text) and not TASK_CUE.search(cleaned):
        uncertain.append("task_cue_removed_or_confined_to_quote_review")
    return {
        "cleaned_body": cleaned,
        "cleaning_flags": sorted(set(flags)),
        "cleaning_uncertain": sorted(set(uncertain)),
        "quoted_lines_removed": removed_quote_lines,
    }


def parse_email_v2(path: Path, source_relative_path: str) -> Dict[str, Any]:
    try:
        message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body, body_format = _extract_body(message)
        cleaned = clean_body_v2(body)
        header_names = ("Message-ID", "In-Reply-To", "References", "From", "To", "Cc", "Subject", "Date")
        result: Dict[str, Any] = {
            "message_id": str(message.get("Message-ID", "") or "").strip(),
            "in_reply_to": str(message.get("In-Reply-To", "") or "").strip(),
            "references": str(message.get("References", "") or "").strip(),
            "from": str(message.get("From", "") or "").strip(),
            "to": str(message.get("To", "") or "").strip(),
            "cc": str(message.get("Cc", "") or "").strip(),
            "subject": str(message.get("Subject", "") or "").strip(),
            "date": str(message.get("Date", "") or "").strip(),
            "body": body,
            **cleaned,
            "body_format": body_format,
            "header_presence": {name: message.get(name) is not None for name in header_names},
            "source_relative_path": source_relative_path,
            "mailbox": source_relative_path.split("/", 1)[0],
            "schema_version": "enron_email_v2",
        }
        result["raw_body_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return {"status": "parsed", "record": result}
    except Exception as exc:
        return {"status": "failure", "source_relative_path": source_relative_path,
                "mailbox": source_relative_path.split("/", 1)[0],
                "reason": type(exc).__name__, "detail": str(exc)[:400]}


def _load_v1_selection(v1_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = v1_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed v1 parser manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completion_status") != "complete" or manifest.get("actual_successful_records") != manifest.get("requested_maximum_successful_records"):
        raise RuntimeError("v1 subset is not marked complete at its requested record cap")
    mailbox_order = manifest.get("mailbox_processing_order", [])
    order_index = {mailbox: index for index, mailbox in enumerate(mailbox_order)}
    selected: List[Dict[str, Any]] = []
    seen_paths = set()
    for path in sorted((v1_dir / "by_user").glob("*.jsonl"), key=lambda item: order_index.get(item.stem, len(order_index))):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                source_path = record.get("source_relative_path")
                if not isinstance(source_path, str) or not source_path or source_path in seen_paths:
                    raise RuntimeError(f"Invalid or duplicate source path in {path}:{line_number}")
                if record.get("mailbox") != path.stem or source_path.split("/", 1)[0] != path.stem:
                    raise RuntimeError(f"V1 mailbox/path provenance mismatch in {path}:{line_number}")
                if not (MAILDIR / source_path).is_file():
                    raise FileNotFoundError(f"Selected raw source is missing: {source_path}")
                seen_paths.add(source_path)
                selected.append({
                    "source_relative_path": source_path,
                    "v1_message_id": record.get("Message-ID", ""),
                    "v1_body": record.get("Body", ""),
                    "v1_cleaned_body": record.get("cleaned_body", ""),
                })
    if len(selected) != manifest.get("actual_successful_records"):
        raise RuntimeError(f"V1 manifest count {manifest.get('actual_successful_records')} does not match source records {len(selected)}")
    return selected, manifest


def _select_smoke(records: List[Dict[str, Any]], size: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    if size < 10:
        raise ValueError("smoke size must be at least 10")
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    body_first: Dict[str, Dict[str, Any]] = {}
    duplicate_body_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for record in records:
        body = record.get("v1_body") or ""
        cleaned = record.get("v1_cleaned_body") or ""
        if re.search(r"(?im)^\s*(?:from|sent|to|cc|subject):\s", body): buckets["reply_headers"].append(record)
        if FORWARD_MARKER.search(body) or re.search(r"(?i)begin forwarded message|forwarded message", body): buckets["forwarded_markers"].append(record)
        if not cleaned.strip(): buckets["empty_cleaned_body"].append(record)
        if body and len(body) > 80 and cleaned and body == cleaned and not any((FORWARD_MARKER.search(body), re.search(r"(?im)^\s*(?:from|sent|to|cc|subject):\s", body))):
            buckets["normal"] .append(record)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        prior = body_first.get(digest)
        if prior is None:
            body_first[digest] = record
        elif prior.get("v1_message_id") != record.get("v1_message_id"):
            if prior.get("v1_body") == body:
                duplicate_body_pairs.append((prior, record))
    if duplicate_body_pairs:
        pair = sorted(duplicate_body_pairs, key=lambda pair: (pair[0]["source_relative_path"], pair[1]["source_relative_path"]))[0]
        buckets["identical_body_different_message_id"].extend(pair)
    selected_by_path: Dict[str, Dict[str, Any]] = {}
    reasons: Dict[str, List[str]] = defaultdict(list)
    for category in ("reply_headers", "forwarded_markers", "empty_cleaned_body", "identical_body_different_message_id", "normal"):
        options = sorted(buckets.get(category, []), key=lambda item: item["source_relative_path"])
        take = min(8 if category != "identical_body_different_message_id" else 2, len(options))
        if take and len(options) > take:
            chosen = rng.sample(options, take)
        else:
            chosen = options[:take]
        for item in chosen:
            selected_by_path[item["source_relative_path"]] = item
            reasons[item["source_relative_path"]].append(category)
    remaining = [item for item in records if item["source_relative_path"] not in selected_by_path]
    rng.shuffle(remaining)
    for item in remaining[:max(0, size - len(selected_by_path))]:
        selected_by_path[item["source_relative_path"]] = item
        reasons[item["source_relative_path"]].append("seeded_fill")
    ordered = sorted(selected_by_path.values(), key=lambda item: item["source_relative_path"])
    if len(ordered) < size:
        raise RuntimeError(f"Only {len(ordered)} eligible source files for requested smoke size {size}")
    return ordered[:size], {path: sorted(set(values)) for path, values in reasons.items() if path in {r["source_relative_path"] for r in ordered[:size]}}


def run_v2(v1_dir: Path, output_dir: Path, smoke_size: Optional[int] = None, seed: int = 20261011) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite v2 output directory: {output_dir}")
    selection, v1_manifest = _load_v1_selection(v1_dir)
    selection_reasons: Dict[str, List[str]] = {}
    mode = "full_recovered_v1_selection"
    if smoke_size is not None:
        selection, selection_reasons = _select_smoke(selection, smoke_size, seed)
        mode = "fixed_seed_stratified_smoke"
    output_dir.mkdir(parents=True, exist_ok=False)
    by_user = output_dir / "by_user"
    by_user.mkdir()
    failures_path = output_dir / "failures.jsonl"
    parse_counts = Counter()
    header_presence = Counter()
    linebreak_records = 0
    empty_cleaned = 0
    uncertain_cleaning = 0
    selected_hash = hashlib.sha256()
    with failures_path.open("w", encoding="utf-8") as failures:
        for index, selected in enumerate(selection, start=1):
            source_relative_path = selected["source_relative_path"]
            selected_hash.update((source_relative_path + "\n").encode("utf-8"))
            raw_path = MAILDIR / source_relative_path
            result = parse_email_v2(raw_path, source_relative_path)
            if result["status"] != "parsed":
                parse_counts["failed"] += 1
                parse_counts["failure_" + result["reason"]] += 1
                failures.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue
            record = result["record"]
            record["selection_reasons"] = selection_reasons.get(source_relative_path, [mode])
            parse_counts["parsed"] += 1
            linebreak_records += "\n" in record["body"]
            empty_cleaned += not bool(record["cleaned_body"].strip())
            uncertain_cleaning += bool(record["cleaning_uncertain"])
            for name, present in record["header_presence"].items():
                header_presence[name + ("_present" if present else "_absent")] += 1
            mailbox_path = by_user / f"{record['mailbox']}.jsonl"
            with mailbox_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(selection):
                print(f"v2 records attempted={index} parsed={parse_counts['parsed']} failed={parse_counts['failed']} elapsed_mode={mode}", flush=True)
    selection_path = output_dir / "selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as output:
        for selected in selection:
            output.write(json.dumps({
                "source_relative_path": selected["source_relative_path"],
                "selection_reasons": selection_reasons.get(selected["source_relative_path"], [mode]),
            }, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "enron_subset_v2",
        "source_subset": str(v1_dir.relative_to(ROOT)) if v1_dir.is_relative_to(ROOT) else str(v1_dir),
        "source_manifest_updated_at": v1_manifest.get("updated_at"),
        "selection_method": mode,
        "selection_seed": seed if smoke_size is not None else None,
        "selection_count": len(selection),
        "selection_sha256": selected_hash.hexdigest(),
        "selected_sources": str(selection_path.relative_to(ROOT)) if selection_path.is_relative_to(ROOT) else selection_path.name,
        "parser_counts": dict(parse_counts),
        "raw_header_presence": dict(sorted(header_presence.items())),
        "records_with_preserved_linebreaks": linebreak_records,
        "empty_cleaned_bodies": empty_cleaned,
        "records_with_cleaning_uncertainty": uncertain_cleaning,
        "selection_reasons": dict(Counter(reason for reasons in selection_reasons.values() for reason in reasons)),
        "completion_status": "complete" if parse_counts["parsed"] + parse_counts["failed"] == len(selection) else "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temp_manifest = output_dir / ".manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp_manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-subset", default=str(V1_SUBSET))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    v1_dir = Path(args.v1_subset)
    output = Path(args.output_dir)
    if not v1_dir.is_absolute(): v1_dir = ROOT / v1_dir
    if not output.is_absolute(): output = ROOT / output
    try:
        run_v2(v1_dir, output, args.smoke_size, args.seed)
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
