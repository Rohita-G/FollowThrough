#!/usr/bin/env python3
"""Linguistic rule-based weak supervision for Request and Commitment classification."""

import re
from typing import List, Tuple

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
    clean = (text or "").strip()
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
