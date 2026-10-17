#!/usr/bin/env python3
"""Enhanced, robust email body cleaning for FollowThrough.

Handles:
- Reply boundary stripping (Original Message, wrote:, Notes headers)
- Forwarded message handling (preserves author notes, strips forward banners & metadata headers)
- Quoted lines prefix removal ('>')
- Corporate legal disclaimers, asterisks banners, and sign-offs
- Consecutive whitespace and empty line normalization
"""

import hashlib
import re
from typing import Any, Dict, List

# Forward & Reply Boundaries
FORWARD_BANNER = re.compile(
    r"(?im)^\s*[-=]{2,}\s*(?:forwarded by|forwarded message|begin forwarded message)\b.*",
    re.I
)
BANNER_CONTINUATION = re.compile(
    r"^\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*[-=]{2,}\s*$",
    re.I
)
ORIGINAL_MARKER = re.compile(
    r"(?im)^\s*[-=]{2,}\s*(?:original message|reply separator)\b.*",
    re.I
)
REPLY_INTRO = re.compile(r"(?im)^\s*on .{1,240}\bwrote\s*:\s*$")
FROM_ON_REPLY = re.compile(r"(?i)^\s*from:\s*.+\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}\b.*")
NOTES_REPLY = re.compile(r"^\s*[\w\.\s-]+@ENRON\s*$", re.I)

# Headers within forwarded / embedded messages
HEADER_LINE = re.compile(
    r"(?i)^\s*(?:from|sent|date|to|cc|bcc|subject|reply-to|please respond to)\b[:\s].*$"
)
NOTES_FROM_LINE = re.compile(
    r'^\s*["\w\s\.,\'-]+<.*?>\s*on\s+\d{1,2}/\d{1,2}/\d{2,4}',
    re.I
)

# Quotes, Signatures, and Disclaimers
QUOTED_LINE = re.compile(r"^\s*>+")
SIGNATURE_SEPARATOR = re.compile(r"^\s*--\s*$")
MOBILE_SIGNATURE = re.compile(r"(?i)^\s*sent from my (?:iphone|ipad|android|blackberry).*$")
SIGNOFF = re.compile(r"(?i)^\s*(?:best(?: regards)?|kind regards|regards|sincerely|thanks|thank you),?\s*$")
ASTERISK_BORDER = re.compile(r"^\s*\*+\s*$")
LEGAL_FOOTER = re.compile(
    r"(?i)^\s*(?:this (?:e-?mail|message)(?: and any attachments)? is confidential|"
    r"if you are not the intended recipient|confidentiality notice|disclaimer|all rights reserved)\b.*$"
)
TASK_CUE = re.compile(
    r"(?i)\b(?:please|can you|could you|will send|i['’]ll|deadline|due|cancel|"
    r"complete|completed|reassign|instead|action item|need to)\b"
)


def clean_body(body: str) -> Dict[str, Any]:
    """Clean email body by removing reply tails, signatures, disclaimers, and quoted lines.
    
    Robustly handles forward banners and nested reply headers.
    """
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    flags: List[str] = []
    uncertain: List[str] = []

    # 1. Detect Reply & Forward cut boundaries
    fwd_start = None
    orig_start = None

    for i, line in enumerate(lines):
        if orig_start is None:
            if ORIGINAL_MARKER.match(line) or REPLY_INTRO.match(line) or FROM_ON_REPLY.match(line):
                orig_start = i
            elif (
                NOTES_REPLY.match(line)
                and i + 1 < len(lines)
                and re.search(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}", lines[i + 1])
            ):
                orig_start = i

        if fwd_start is None and FORWARD_BANNER.match(line):
            fwd_start = i

    # If there is a reply banner, cut everything below it
    if orig_start is not None:
        lines = lines[:orig_start]
        flags.append("quoted_reply_tail_removed")

    # If there is a forward banner:
    if fwd_start is not None and fwd_start < len(lines):
        text_before = "\n".join(lines[:fwd_start]).strip()
        # If author wrote meaningful text before the forward (e.g., 'FYI' or 'Please review')
        if text_before and len(text_before) > 5 and not text_before.lower().startswith("http"):
            lines = lines[:fwd_start]
            flags.append("forwarded_tail_removed")
        else:
            # Pure forward: strip forward banner line, continuation, and header metadata
            idx = fwd_start + 1
            if idx < len(lines) and BANNER_CONTINUATION.match(lines[idx]):
                idx += 1
            while idx < len(lines):
                cur = lines[idx]
                if (
                    not cur.strip()
                    or HEADER_LINE.match(cur)
                    or NOTES_REPLY.match(cur)
                    or NOTES_FROM_LINE.match(cur)
                    or re.search(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}", cur)
                ):
                    idx += 1
                else:
                    break
            lines = lines[idx:]
            flags.append("forwarded_headers_stripped")

    # 2. Terminal signature & disclaimer removal
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    if nonblank:
        tail_start = nonblank[max(0, int(len(nonblank) * 0.50))]
        for index in range(tail_start, len(lines)):
            line = lines[index]
            if SIGNATURE_SEPARATOR.match(line) or MOBILE_SIGNATURE.match(line):
                suffix = "\n".join(lines[index + 1:])
                if not TASK_CUE.search(suffix):
                    lines = lines[:index]
                    flags.append("terminal_signature_removed")
                else:
                    uncertain.append("signature_marker_with_task_cue_preserved")
                break

            if LEGAL_FOOTER.match(line) or (ASTERISK_BORDER.match(line) and index + 1 < len(lines) and LEGAL_FOOTER.match(lines[index + 1])):
                suffix = "\n".join(lines[index:])
                if index >= tail_start and not re.search(r"(?i)\b(?:please|deadline|due|will send|i['’]ll)\b", suffix):
                    lines = lines[:index]
                    flags.append("terminal_legal_footer_removed")
                else:
                    uncertain.append("legal_footer_candidate_preserved")
                break

            if SIGNOFF.match(line) and len(lines) - index <= 5:
                suffix = "\n".join(lines[index:])
                if not TASK_CUE.search(suffix):
                    lines = lines[:index]
                    flags.append("terminal_signoff_removed")
                else:
                    uncertain.append("signoff_with_task_cue_preserved")
                break

    # 3. Strip quote-prefixed lines ('>')
    kept: List[str] = []
    removed_quote_lines = 0
    for line in lines:
        if QUOTED_LINE.match(line):
            removed_quote_lines += 1
            continue
        kept.append(line.rstrip())

    if removed_quote_lines:
        flags.append("quoted_prefixed_lines_removed")

    # 4. Collapse whitespace and multiple blank lines
    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

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
