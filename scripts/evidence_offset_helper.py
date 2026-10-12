#!/usr/bin/env python3
"""Dependency-free helper to calculate exact character evidence offsets for annotation.

Usage examples:
    python scripts/evidence_offset_helper.py --text "I will send the report tomorrow." --sub "will send"
    python scripts/evidence_offset_helper.py --text "Send to Alice. Send to Bob." --sub "Send" --occurrence 2
    python scripts/evidence_offset_helper.py --drafts data/annotations/pilot_screening_drafts.20261012.jsonl \
        --example-id draft-pilot-xxx --sub "Please arrange"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def find_occurrences(text: str, substring: str) -> List[Tuple[int, int]]:
    """Return all [start, end] char offsets for substring in text."""
    if not substring:
        raise ValueError("Substring cannot be empty")
    occurrences = []
    start = 0
    while True:
        pos = text.find(substring, start)
        if pos == -1:
            break
        occurrences.append((pos, pos + len(substring)))
        start = pos + 1
    return occurrences


def derive_evidence_span(
    target_text: str,
    substring: str,
    occurrence: Optional[int] = 1,
    source: str = "update_text",
    supports_label: str = "",
) -> Dict[str, Any]:
    """Derive an evidence span dictionary for annotation.

    Raises ValueError if substring is not found or if multiple occurrences exist
    and occurrence is not explicitly provided.
    """
    if source not in ("update_text", "task_text"):
        raise ValueError(f"source must be 'update_text' or 'task_text', got {source!r}")
    if not isinstance(target_text, str):
        raise TypeError("target_text must be a string")
    if not isinstance(substring, str) or not substring:
        raise ValueError("substring must be a non-empty string")

    matches = find_occurrences(target_text, substring)
    if not matches:
        raise ValueError(
            f"Substring not found in target text.\n"
            f"Substring: {substring!r}\n"
            f"Target text length: {len(target_text)} characters."
        )

    if len(matches) > 1 and occurrence is None:
        contexts = []
        for idx, (s, e) in enumerate(matches, 1):
            ctx_start = max(0, s - 20)
            ctx_end = min(len(target_text), e + 20)
            snippet = target_text[ctx_start:ctx_end].replace("\n", " ")
            contexts.append(f"  [{idx}] ...{snippet}...")
        raise ValueError(
            f"Substring appears {len(matches)} times in target text. "
            f"Please specify an explicit --occurrence (1 to {len(matches)}):\n"
            + "\n".join(contexts)
        )

    occ = occurrence if occurrence is not None else 1
    if not (1 <= occ <= len(matches)):
        raise ValueError(
            f"Requested occurrence {occ} out of range (found {len(matches)} occurrences)."
        )

    start_offset, end_offset = matches[occ - 1]
    extracted = target_text[start_offset:end_offset]
    assert extracted == substring, f"Offset slice mismatch: {extracted!r} != {substring!r}"

    return {
        "text_field": source,
        "start_char": start_offset,
        "end_char": end_offset,
        "supports_label": supports_label,
        # Backward-compatible convenience keys:
        "start": start_offset,
        "end": end_offset,
        "source": source,
        "text": substring,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Literal target text string to search within")
    parser.add_argument("--sub", required=True, help="Target substring to locate")
    parser.add_argument(
        "--occurrence",
        type=int,
        default=None,
        help="1-based occurrence index if substring appears multiple times",
    )
    parser.add_argument(
        "--source",
        choices=["update_text", "task_text"],
        default="update_text",
        help="Evidence source field name (default: update_text)",
    )
    parser.add_argument(
        "--supports-label",
        default="",
        help="Target label that this evidence span supports (e.g. request, commitment, completion)",
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        help="Path to JSONL drafts file to look up target text by example_id",
    )
    parser.add_argument(
        "--example-id",
        help="example_id in drafts file to fetch target text from",
    )
    args = parser.parse_args(argv)

    target_text = args.text
    if args.drafts:
        if not args.example_id:
            parser.error("--example-id is required when --drafts is provided")
        if not args.drafts.is_file():
            parser.error(f"Drafts file not found: {args.drafts}")
        found = False
        with args.drafts.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("example_id") == args.example_id:
                    target_text = rec.get(args.source, "")
                    found = True
                    break
        if not found:
            parser.error(f"example_id {args.example_id!r} not found in {args.drafts}")

    if target_text is None:
        parser.error("Either --text or (--drafts and --example-id) must be provided")

    try:
        span = derive_evidence_span(
            target_text=target_text,
            substring=args.sub,
            occurrence=args.occurrence,
            source=args.source,
            supports_label=args.supports_label,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    ctx_start = max(0, span["start"] - 40)
    ctx_end = min(len(target_text), span["end"] + 40)
    before = target_text[ctx_start:span["start"]].replace("\n", " ")
    after = target_text[span["end"]:ctx_end].replace("\n", " ")
    print(f"Context: ...{before}[{span['text']}]{after}...", file=sys.stderr)

    print(json.dumps(span, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
