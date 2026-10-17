"""
Calendar Sync & Export Module for FollowThrough.

Provides 100% free, zero-dependency calendar integration:
1. Google Calendar 1-Click Web Links (pre-filled with title, dates, evidence notes, origin thread)
2. iCalendar (.ics) RFC 5545 export compatible with Apple Calendar, Microsoft Outlook, and Google Calendar.
"""

from __future__ import annotations

import datetime
import re
import urllib.parse
from typing import Any, Dict, Optional, Tuple


def parse_deadline_to_datetimes(
    deadline_str: Optional[str],
    default_anchor: Optional[datetime.date] = None,
) -> Tuple[datetime.datetime, datetime.datetime, bool]:
    """
    Parse a normalized deadline display string (e.g. '2026-10-26 (Mon) at 10:00'
    or '2026-10-21 (Wed)') into (start_dt, end_dt, is_all_day).

    If deadline_str is None or 'None specified', defaults to next day 09:00 - 10:00.
    """
    if not default_anchor:
        default_anchor = datetime.date(2026, 10, 17)

    if not deadline_str or deadline_str.strip().lower() in ("none specified", "none", ""):
        # Default fallback: next day 9:00 AM - 10:00 AM
        start_dt = datetime.datetime.combine(default_anchor + datetime.timedelta(days=1), datetime.time(9, 0))
        end_dt = start_dt + datetime.timedelta(hours=1)
        return start_dt, end_dt, False

    # Check for specific date + time pattern: "YYYY-MM-DD (...) at HH:MM"
    timed_match = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[^\d]+)?at\s+(\d{1,2}):(\d{2})", deadline_str, re.I)
    if timed_match:
        year, month, day, hour, minute = map(int, timed_match.groups())
        start_dt = datetime.datetime(year, month, day, hour, minute)
        end_dt = start_dt + datetime.timedelta(hours=1)
        return start_dt, end_dt, False

    # Check for date-only pattern: "YYYY-MM-DD"
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", deadline_str)
    if date_match:
        year, month, day = map(int, date_match.groups())
        # All-day event: start at midnight, end next day midnight
        start_dt = datetime.datetime(year, month, day, 9, 0)
        end_dt = datetime.datetime(year, month, day, 10, 0)
        return start_dt, end_dt, True

    # Fallback to default
    start_dt = datetime.datetime.combine(default_anchor + datetime.timedelta(days=1), datetime.time(9, 0))
    end_dt = start_dt + datetime.timedelta(hours=1)
    return start_dt, end_dt, False


def generate_google_calendar_url(
    task: Dict[str, Any],
    anchor_date: Optional[datetime.date] = None,
) -> str:
    """
    Generate a 1-click Google Calendar web creation URL pre-populated with:
    - Title / summary
    - Event start & end timestamps
    - Detailed description containing original email evidence and audit thread info
    """
    action_desc = task.get("action_description", "FollowThrough Task")
    deadline_str = task.get("deadline", "")
    evidence = task.get("evidence_text", "")
    owner = task.get("owner", "Unspecified")
    thread_title = task.get("thread_title", "")
    task_id = task.get("task_id", "")

    start_dt, end_dt, is_all_day = parse_deadline_to_datetimes(deadline_str, default_anchor=anchor_date)

    if is_all_day:
        # All day format: YYYYMMDD/YYYYMMDD (next day)
        next_day = start_dt.date() + datetime.timedelta(days=1)
        dates_param = f"{start_dt.strftime('%Y%m%d')}/{next_day.strftime('%Y%m%d')}"
    else:
        # Timed format: YYYYMMDDTHHMMSS/YYYYMMDDTHHMMSS
        dates_param = f"{start_dt.strftime('%Y%m%d%T%H%M%S').replace(':', '')}/{end_dt.strftime('%Y%m%d%T%H%M%S').replace(':', '')}"

    details_parts = [
        f"Task: {action_desc}",
        f"Assignee: {owner}",
        f"Due: {deadline_str or 'None specified'}",
    ]
    if thread_title:
        details_parts.append(f"Origin Thread: {thread_title}")
    if evidence:
        details_parts.append(f"\nEmail Evidence Quote:\n\"{evidence}\"")
    if task_id:
        details_parts.append(f"\nRef ID: {task_id}")
    details_parts.append("\nManaged via FollowThrough Task Intelligence.")

    details_str = "\n".join(details_parts)

    params = {
        "action": "TEMPLATE",
        "text": action_desc,
        "dates": dates_param,
        "details": details_str,
    }

    base_url = "https://calendar.google.com/calendar/render"
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def generate_ics_content(
    task: Dict[str, Any],
    anchor_date: Optional[datetime.date] = None,
) -> str:
    """
    Generate RFC 5545 iCalendar (.ics) content for importing into:
    - Apple Calendar (macOS / iOS)
    - Microsoft Outlook
    - Google Calendar
    - Any standard iCalendar client
    """
    task_id = task.get("task_id", f"task-{abs(hash(task.get('action_description', ''))):08x}")
    action_desc = task.get("action_description", "FollowThrough Task")
    deadline_str = task.get("deadline", "")
    evidence = task.get("evidence_text", "")
    owner = task.get("owner", "Unspecified")
    thread_title = task.get("thread_title", "")

    start_dt, end_dt, is_all_day = parse_deadline_to_datetimes(deadline_str, default_anchor=anchor_date)
    now_utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Clean text fields for RFC 5545 escaping
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    summary_escaped = _escape(action_desc)

    desc_lines = [
        f"Task: {action_desc}",
        f"Assignee: {owner}",
        f"Deadline: {deadline_str or 'None specified'}",
    ]
    if thread_title:
        desc_lines.append(f"Origin Thread: {thread_title}")
    if evidence:
        desc_lines.append(f"Email Evidence Quote: \"{evidence}\"")
    desc_lines.append(f"Ref ID: {task_id}")
    desc_lines.append("Managed via FollowThrough Task Intelligence.")
    description_escaped = _escape("\n".join(desc_lines))

    if is_all_day:
        dtstart_line = f"DTSTART;VALUE=DATE:{start_dt.strftime('%Y%m%d')}"
        next_day = start_dt.date() + datetime.timedelta(days=1)
        dtend_line = f"DTEND;VALUE=DATE:{next_day.strftime('%Y%m%d')}"
    else:
        dtstart_line = f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}"
        dtend_line = f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}"

    ics_content = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//FollowThrough//Task Intelligence Engine//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{task_id}@followthrough.local",
        f"DTSTAMP:{now_utc}",
        dtstart_line,
        dtend_line,
        f"SUMMARY:{summary_escaped}",
        f"DESCRIPTION:{description_escaped}",
        "STATUS:CONFIRMED",
        "TRANSP:OPAQUE",
        "SEQUENCE:0",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ])

    return ics_content
