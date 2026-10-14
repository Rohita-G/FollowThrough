#!/usr/bin/env python3
"""Conservative, line-preserving email body cleaning for FollowThrough."""

import hashlib
import re
from typing import Any, Dict, List

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


def clean_body(body: str) -> Dict[str, Any]:
    """Clean email body by removing reply tails, signatures, disclaimers, and quoted lines.
    
    Preserves forwarded payload blocks for human/model review.
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
        "raw_body_sha256": hashlib.sha256((body or "").encode("utf-8")).hexdigest(),
    }
