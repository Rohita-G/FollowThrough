import re
import datetime
from typing import Any, Dict, Optional, Tuple

DAY_NAME_TO_INT = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTH_NAME_TO_INT = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_header_date(text: str) -> Optional[datetime.date]:
    """Extract reference date from standard email headers like 'Date:' or 'Sent:'."""
    # Example: "Date: Friday, Oct 17, 2026" or "Sent: 05/07/2001" or "Date: 17 Oct 2026"
    m = re.search(r"(?im)^\s*(?:date|sent):\s*([^\n]+)", text)
    if not m:
        return None
    raw = m.group(1).strip()
    
    # Try ISO YYYY-MM-DD
    m_iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", raw)
    if m_iso:
        try:
            return datetime.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            pass

    # Try MM/DD/YYYY
    m_us = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", raw)
    if m_us:
        try:
            year = int(m_us.group(3))
            if year < 100:
                year += 2000
            return datetime.date(year, int(m_us.group(1)), int(m_us.group(2)))
        except ValueError:
            pass

    # Try "Month DD, YYYY" or "DD Month YYYY"
    for m_name, m_num in MONTH_NAME_TO_INT.items():
        pattern1 = rf"\b{m_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b"
        m1 = re.search(pattern1, raw, re.I)
        if m1:
            try:
                return datetime.date(int(m1.group(2)), m_num, int(m1.group(1)))
            except ValueError:
                pass
        pattern2 = rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{m_name},?\s+(\d{{4}})\b"
        m2 = re.search(pattern2, raw, re.I)
        if m2:
            try:
                return datetime.date(int(m2.group(2)), m_num, int(m2.group(1)))
            except ValueError:
                pass

    return None


def normalize_deadline(
    literal_phrase: str,
    anchor_date: Optional[datetime.date] = None,
) -> Dict[str, Any]:
    """Ground and normalize a literal deadline phrase into an unambiguous calendar date.
    
    Args:
        literal_phrase: Raw substring extracted from text (e.g. 'by Wednesday', 'tomorrow', 'before 2 PM')
        anchor_date: Reference base date (defaults to today)
        
    Returns:
        Dict containing:
        - 'literal_phrase': Original extracted phrase
        - 'iso_date': Resolved YYYY-MM-DD or None
        - 'display_date': Human-readable formatted string (e.g., '2026-10-21 (Wed)')
        - 'anchor_date': Reference date used
        - 'is_relative': True if resolved from relative expression
        - 'time_str': Extracted time expression if present (e.g. '14:00')
    """
    if not literal_phrase or literal_phrase.strip().lower() in {"none specified", "none", ""}:
        return {
            "literal_phrase": "None specified",
            "iso_date": None,
            "display_date": "None specified",
            "anchor_date": anchor_date.isoformat() if anchor_date else None,
            "is_relative": False,
            "time_str": None,
        }

    anchor = anchor_date or datetime.date.today()
    cleaned = literal_phrase.strip()
    norm = re.sub(r"^(?:by|before|on|until|due)\s+", "", cleaned, flags=re.I).strip()

    # Detect optional time: e.g. "2 PM", "5:30 PM", "14:00"
    time_str: Optional[str] = None
    m_time = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", cleaned, re.I)
    if m_time:
        hr = int(m_time.group(1))
        minute = int(m_time.group(2) or 0)
        ampm = m_time.group(3).lower()
        if ampm == "pm" and hr < 12:
            hr += 12
        elif ampm == "am" and hr == 12:
            hr = 0
        time_str = f"{hr:02d}:{minute:02d}"

    resolved_date: Optional[datetime.date] = None
    is_relative = True

    # 1. Tomorrow / Today / Tonight / EOD
    if re.search(r"\btomorrow\b", norm, re.I):
        resolved_date = anchor + datetime.timedelta(days=1)
    elif re.search(r"\b(today|tonight|end\s+of\s+day|eod|cob|close\s+of\s+business)\b", norm, re.I):
        resolved_date = anchor
    elif re.search(r"\bnext\s+week\b", norm, re.I):
        resolved_date = anchor + datetime.timedelta(days=7)
    elif re.search(r"\bnext\s+month\b", norm, re.I):
        resolved_date = anchor + datetime.timedelta(days=30)

    # 2. Day of week: e.g. "Wednesday", "next Monday", "before Friday"
    if not resolved_date:
        for dname, target_idx in DAY_NAME_TO_INT.items():
            if re.search(rf"\bnext\s+{dname}\b", norm, re.I):
                # Next week's occurrence
                days_ahead = (target_idx - anchor.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                else:
                    days_ahead += 7
                resolved_date = anchor + datetime.timedelta(days=days_ahead)
                break
            elif re.search(rf"\b{dname}\b", norm, re.I):
                # Upcoming occurrence
                days_ahead = (target_idx - anchor.weekday()) % 7
                if days_ahead == 0:
                    # Same day of week: if time specified and later, today, else 7 days ahead
                    days_ahead = 7
                resolved_date = anchor + datetime.timedelta(days=days_ahead)
                break

    # 3. Explicit Calendar Month & Day: e.g. "Nov 15", "October 24th"
    if not resolved_date:
        for mname, mnum in MONTH_NAME_TO_INT.items():
            m_cal = re.search(rf"\b{mname}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?\b", norm, re.I)
            if m_cal:
                day_num = int(m_cal.group(1))
                year_num = int(m_cal.group(2)) if m_cal.group(2) else anchor.year
                # If month has already passed in anchor year and no explicit year given, assume next year
                if not m_cal.group(2) and (mnum < anchor.month or (mnum == anchor.month and day_num < anchor.day)):
                    year_num += 1
                try:
                    resolved_date = datetime.date(year_num, mnum, day_num)
                    is_relative = False
                    break
                except ValueError:
                    pass

    # 4. Numeric dates: e.g. "10/24", "10/24/2026"
    if not resolved_date:
        m_num = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", norm)
        if m_num:
            m_val = int(m_num.group(1))
            d_val = int(m_num.group(2))
            y_val = int(m_num.group(3)) if m_num.group(3) else anchor.year
            if y_val < 100:
                y_val += 2000
            try:
                resolved_date = datetime.date(y_val, m_val, d_val)
                is_relative = False
            except ValueError:
                pass

    # 5. Time-only deadline: e.g. "before 2 PM" without day defaults to today
    if not resolved_date and time_str:
        resolved_date = anchor

    if resolved_date:
        day_str = resolved_date.strftime("%a")
        iso_str = resolved_date.isoformat()
        if time_str:
            display_str = f"{iso_str} ({day_str}) at {time_str}"
        else:
            display_str = f"{iso_str} ({day_str})"
    else:
        iso_str = None
        display_str = cleaned

    return {
        "literal_phrase": cleaned,
        "iso_date": iso_str,
        "display_date": display_str,
        "anchor_date": anchor.isoformat(),
        "is_relative": is_relative,
        "time_str": time_str,
    }
