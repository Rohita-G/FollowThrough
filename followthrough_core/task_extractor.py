"""Linguistic Rule-Based Candidate Task Extractor for FollowThrough.

Extracts candidate task objects, owners, and deadlines using conservative
linguistic patterns with exact character offset tracking.
Note: This is a rule-based heuristic extractor, not a trained statistical NER model.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# Pre-compiled regex patterns for date and deadline expressions
DATE_PATTERNS = [
    re.compile(
        r"\b(by|before|on|until|due)\s+(next\s+[A-Za-z]+|[A-Z][a-z]+(?:\s+\d{1,2})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?|tomorrow|today|tonight|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)\b",
        re.I,
    ),
    re.compile(r"\b(tomorrow|today|tonight|next\s+week|next\s+month)\b", re.I),
    re.compile(r"\b([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?)\b"),
]

# Patterns detecting action directives and imperative requests/commitments
ACTION_LEAD_PATTERNS = [
    re.compile(
        r"^(?:please\s*,?\s*|could\s+you\s+|can\s+you\s+|let\'s\s+|would\s+you\s+mind\s+|i\s+will\s+|we\s+shall\s+|we\s+will\s+|i\'ll\s+|i\s+can\s+|you\s+should\s+)(.+)$",
        re.I,
    ),
]

# Patterns detecting explicit personal names assigned to actions
NAME_TARGET_PATTERNS = [
    re.compile(r"^([A-Z][a-z]+)\s*,\s*(?:please|could you|can you|let's)", re.I),
    re.compile(r"\b(?:assigned\s+to|ask)\s+([A-Z][a-z]+)\b", re.I),
]


def find_quoted_boundary(text: str) -> int:
    """Find character offset where forwarded or quoted email text begins.
    
    Prevents false attribution of actions contained in quoted reply history.
    """
    patterns = [
        r"-+\s*Forwarded by",
        r"\bFrom:\s+[A-Za-z0-9]",
        r"-----Original Message-----",
    ]
    min_pos = len(text)
    for p in patterns:
        m = re.search(p, text, re.I)
        if m and m.start() < min_pos:
            min_pos = m.start()
    return min_pos


def clean_action_title(sentence: str) -> str:
    """Extract a concise imperative action title from a full sentence.
    
    Strips conversational preambles (e.g., 'Please', 'Could you', 'I will').
    """
    text = sentence.strip().rstrip(".!?")
    text = re.sub(r"^[A-Z][a-z]+,\s*", "", text).strip()
    prefixes = [
        r"^(?:please\s*,?\s*)",
        r"^(?:could\s+you\s+(?:please\s*)?)",
        r"^(?:can\s+you\s+(?:please\s*)?)",
        r"^(?:would\s+you\s+mind\s+)",
        r"^(?:let\'s\s+)",
        r"^(?:i\s+will\s+)",
        r"^(?:we\s+will\s+|we\s+shall\s+)",
        r"^(?:i\'ll\s+)",
        r"^(?:i\s+can\s+)",
        r"^(?:you\s+should\s+)",
        r"^(?:make\s+sure\s+to\s+)",
        r"^(?:be\s+sure\s+to\s+)",
    ]
    for p in prefixes:
        text = re.sub(p, "", text, flags=re.I).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def extract_candidate_tasks(text: str) -> List[Dict[str, Any]]:
    """Extract candidate task/event objects with exact evidence and offsets.

    Processes the full input text independently of classifier token limits.
    Produces:
    - action_description: Concise imperative action phrase
    - task_type: 'Request' (delegated) or 'Commitment' (self-assigned promise)
    - owner: Explicit assignee or speaker pronoun ('Unspecified' if missing)
    - deadline: Literal date/deadline expression ('None specified' if missing)
    - evidence_text: Exact source substring
    - start_offset, end_offset: Exact character positions in the source text
    - uncertainty_flags: Explicit flags for missing owners, deadlines, or quoted text
    """
    quoted_start = find_quoted_boundary(text)
    candidates: List[Dict[str, Any]] = []

    # Detect global greeting recipient and signoff sender for context-aware owner resolution
    m_rec = re.search(r"(?im)^\s*(?:hi|hello|dear|hey)?\s*([A-Z][a-z]+)\s*[,:]\s*$", text)
    greeting_recipient = m_rec.group(1) if m_rec else None

    m_send = re.search(r"(?im)^\s*(?:thanks|thank you|regards|best regards|sincerely),?\s*\n\s*([A-Z][a-z]+)\s*$", text)
    signoff_sender = m_send.group(1) if m_send else None

    # Iterate over sentences while preserving exact character offsets
    sentence_regex = re.compile(r"([^\n.!?]+(?:[\n.!?]+|$))")
    for match in sentence_regex.finditer(text):
        raw_sentence = match.group(1)
        start_char = match.start()
        end_char = match.end()

        # Trim leading and trailing whitespace while adjusting character offsets
        leading_ws = len(raw_sentence) - len(raw_sentence.lstrip())
        trailing_ws = len(raw_sentence) - len(raw_sentence.rstrip())
        clean_sentence = raw_sentence.strip()

        # Skip trivial or empty lines
        if len(clean_sentence) < 8:
            continue

        actual_start = start_char + leading_ws
        actual_end = end_char - trailing_ws
        evidence_substring = text[actual_start:actual_end]

        # 1. Action Directive Check
        is_action = False
        for ap in ACTION_LEAD_PATTERNS:
            if ap.search(clean_sentence):
                is_action = True
                break

        # Check imperative / commitment verbs with modal auxiliary
        if not is_action and re.search(
            r"\b(arrange|review|prepare|schedule|submit|file|provide|call|contact|coordinate)\b",
            clean_sentence,
            re.I,
        ):
            if re.search(r"\b(please|could|will|shall|let\'s|need to|should|must)\b", clean_sentence, re.I):
                is_action = True

        if not is_action:
            continue

        # 2. Task Type & Explicit Owner Extraction
        is_commitment = bool(re.match(r"^(?:i|we)\s+(?:will|shall|can|promise|plan to|’ll|'ll)\b", clean_sentence, re.I))
        task_type = "Commitment" if is_commitment else "Request"

        owner: Optional[str] = None
        for np in NAME_TARGET_PATTERNS:
            m_name = np.search(clean_sentence)
            if m_name:
                owner = m_name.group(1)
                break
        if not owner:
            if is_commitment:
                # First-person commitment assigned to sender
                if signoff_sender:
                    owner = f"Sender ({signoff_sender})"
                else:
                    owner = "Sender (I)" if re.match(r"^i\b", clean_sentence, re.I) else "Sender Group (We)"
            else:
                # Direct request defaults to greeting recipient if present
                if greeting_recipient:
                    owner = greeting_recipient

        # 3. Literal Deadline Extraction
        deadline: Optional[str] = None
        for dp in DATE_PATTERNS:
            m_date = dp.search(clean_sentence)
            if m_date:
                deadline = m_date.group(0).strip()
                break

        # 4. Uncertainty Flags
        flags: List[str] = []
        if not owner:
            flags.append("missing_explicit_owner")
        if not deadline:
            flags.append("missing_explicit_deadline")
        if actual_start >= quoted_start:
            flags.append("quoted_or_forwarded_text")

        action_title = clean_action_title(clean_sentence)

        candidates.append({
            "task_title": action_title,
            "action_description": action_title,
            "raw_sentence": clean_sentence,
            "task_type": task_type,
            "owner": owner or "Unspecified",
            "deadline": deadline or "None specified",
            "evidence_text": evidence_substring,
            "start_offset": actual_start,
            "end_offset": actual_end,
            "uncertainty_flags": flags,
            "is_quoted": actual_start >= quoted_start,
        })

    return candidates
