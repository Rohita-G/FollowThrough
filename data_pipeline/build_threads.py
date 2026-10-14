#!/usr/bin/env python3
"""Thread reconstruction linking emails by subject, references, and timeline."""

import hashlib
import re
from typing import Any, Dict, List, Optional

SUBJECT_PREFIX = re.compile(r"^(?:(?:re|fw|fwd)\s*:\s*)+", re.I)


def normalize_subject(subject: str) -> str:
    """Normalize email subject by stripping Re/Fw/Fwd prefixes and lowercasing."""
    val = " ".join((subject or "").split())
    while True:
        updated = SUBJECT_PREFIX.sub("", val).strip()
        if updated == val:
            return val.lower()
        val = updated


def build_threads(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign thread_id to records based on Message-ID, In-Reply-To, References, and normalized subject."""
    # Build reply graph or fallback to normalized subject clusters
    thread_map: Dict[str, str] = {}
    threaded_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        norm_subj = normalize_subject(rec.get("subject", ""))
        in_reply_to = rec.get("in_reply_to", "").strip()
        msg_id = rec.get("message_id", f"msg_{idx}").strip()

        if in_reply_to and in_reply_to in thread_map:
            thread_id = thread_map[in_reply_to]
        elif norm_subj:
            thread_id = f"thread_{hashlib.md5(norm_subj.encode('utf-8')).hexdigest()[:12]}"
        else:
            thread_id = f"singleton_{hashlib.md5(msg_id.encode('utf-8')).hexdigest()[:12]}"

        if msg_id:
            thread_map[msg_id] = thread_id

        rec_with_thread = dict(rec)
        rec_with_thread["thread_id"] = thread_id
        threaded_records.append(rec_with_thread)

    return threaded_records
