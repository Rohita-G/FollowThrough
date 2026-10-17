"""Linguistic Rule-Based Candidate Task Extractor for FollowThrough.

Extracts candidate task objects, owners, and deadlines using conservative
linguistic patterns with exact character offset tracking.
Note: This is a rule-based heuristic extractor, not a trained statistical NER model.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

# Pre-compiled regex patterns for date and deadline expressions
DATE_PATTERNS = [
    re.compile(
        rf"\b(by|before|on|until|due)\s+(?:end\s+of\s+day|EOD|COB|close\s+of\s+business|tomorrow(?:\s+morning|\s+afternoon|\s+evening)?|today(?:\s+morning|\s+afternoon|\s+evening)?|tonight|next\s+[A-Za-z]+|{MONTHS}(?:\s+\d{{1,2}})?|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)\b",
        re.I,
    ),
    re.compile(r"\b(tomorrow|today|tonight|next\s+week|next\s+month)\b", re.I),
    re.compile(rf"\b({MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,\s*\d{{4}})?)\b", re.I),
]

# Patterns detecting action directives and imperative requests/commitments
ACTION_LEAD_PATTERNS = [
    re.compile(
        r"^(?:please\s*,?\s*|kindly\s*,?\s*|could\s+you\s+|can\s+you\s+|let\'s\s+|would\s+you\s+mind\s+|i\s+will\s+|we\s+shall\s+|we\s+will\s+|i\'ll\s+|we\'ll\s+|i\s+can\s+|i\s+plan\s+to\s+|you\s+should\s+|make\s+sure\s+to\s+|be\s+sure\s+to\s+)(.+)$",
        re.I,
    ),
]

# Common workplace action verbs
ACTION_VERBS = (
    r"(?:update|review|prepare|schedule|submit|coordinate|investigate|verify|"
    r"check|fix|resolve|send|email|share|provide|deploy|test|audit|monitor|"
    r"draft|compile|implement|deliver|follow\s+up|reach\s+out|organize|call|"
    r"arrange|finalize|confirm|sign|approve|analyze|reconcile|benchmark|merge|push|file)"
)

# Patterns detecting explicit personal names assigned to actions
NAME_TARGET_PATTERNS = [
    re.compile(r"^@?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[,:\-]\s*(?:please|could you|can you|let's|kindly|make sure|be sure|you should)", re.I),
    re.compile(r"^@?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[,:\-]\s*", re.I),
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
    
    Strips discourse connectives, greeting names, and conversational preambles.
    """
    text = sentence.strip().rstrip(".!?")
    # Strip conversational discourse connectives
    text = re.sub(
        r"^(?:also|additionally|furthermore|meanwhile|in the meantime|next|first|finally|then),?\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    # Strip leading addressed name callout: e.g., 'David, ' or 'Elena: '
    text = re.sub(r"^@?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*[,:\-]\s*", "", text).strip()
    prefixes = [
        r"^(?:please\s*,?\s*)",
        r"^(?:kindly\s*,?\s*)",
        r"^(?:could\s+you\s+(?:please\s*)?)",
        r"^(?:can\s+you\s+(?:please\s*)?)",
        r"^(?:would\s+you\s+mind\s+)",
        r"^(?:let\'s\s+)",
        r"^(?:i\s+will\s+)",
        r"^(?:we\s+will\s+|we\s+shall\s+)",
        r"^(?:i\'ll\s+|we\'ll\s+)",
        r"^(?:i\s+plan\s+to\s+|we\s+plan\s+to\s+)",
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

    m_send = re.search(r"(?im)^\s*(?:thanks|thank you|regards|best regards|sincerely|cheers),?\s*\n\s*([A-Z][a-z]+)\s*$", text)
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

        # Skip passive generic chitchat
        if re.search(r"^let me know if\b", clean_sentence, re.I):
            continue

        actual_start = start_char + leading_ws
        actual_end = end_char - trailing_ws
        evidence_substring = text[actual_start:actual_end]

        # Normalized sentence stripped of discourse connective for intent matching
        norm_sentence = re.sub(
            r"^(?:also|additionally|furthermore|meanwhile|in the meantime|next|first|finally|then),?\s*",
            "",
            clean_sentence,
            flags=re.I,
        ).strip()

        # Check for explicit addressed name at start: e.g. "David, please..." or "Elena: please..."
        addressed_name: Optional[str] = None
        for np in NAME_TARGET_PATTERNS:
            m_np = np.search(norm_sentence)
            if m_np:
                addressed_name = m_np.group(1)
                break

        # Strip addressed name for action verb checks
        after_name_sentence = re.sub(r"^@?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*[,:\-]\s*", "", norm_sentence).strip()

        # 1. Action Directive / Commitment Check
        is_commitment_lead = bool(re.search(r"^(?:i|we)\s+(?:will|shall|can|promise|plan to|’ll|'ll|need to|must|should)\b", after_name_sentence, re.I))
        is_request_lead = bool(re.search(r"^(?:please|kindly|could you|can you|would you mind|let\'s|you should|make sure to|be sure to)\b", after_name_sentence, re.I))
        has_action_verb = bool(re.search(r"\b" + ACTION_VERBS + r"\b", after_name_sentence, re.I))

        is_action = False
        task_type = "Request"

        if is_commitment_lead and has_action_verb:
            is_action = True
            task_type = "Commitment"
        elif is_request_lead and (has_action_verb or len(after_name_sentence.split()) >= 4):
            is_action = True
            task_type = "Request"
        elif addressed_name and (is_request_lead or bool(re.search(r"^(?:can|could|please|update|review|prepare|schedule|submit|coordinate|check|fix|deploy|test|investigate)\b", after_name_sentence, re.I))):
            is_action = True
            task_type = "Request"
        else:
            # Fallback for standard imperative actions starting with an action verb or directive
            for ap in ACTION_LEAD_PATTERNS:
                if ap.search(after_name_sentence):
                    is_action = True
                    break

        if not is_action:
            continue

        # 2. Owner Resolution
        owner: Optional[str] = addressed_name
        if not owner:
            if task_type == "Commitment":
                if signoff_sender:
                    owner = f"Sender ({signoff_sender})"
                else:
                    owner = "Sender (I)" if re.search(r"\bi\b", after_name_sentence, re.I) else "Sender Group (We)"
            else:
                if greeting_recipient and greeting_recipient.lower() not in {"team", "all", "everyone"}:
                    owner = greeting_recipient
                elif greeting_recipient:
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
