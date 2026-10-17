"""
Unit tests for FollowThrough Calendar Sync and .ics Export.
"""

import datetime
from followthrough_core.calendar_sync import (
    parse_deadline_to_datetimes,
    generate_google_calendar_url,
    generate_ics_content,
)


def test_parse_timed_deadline():
    deadline = "2026-10-26 (Mon) at 10:00"
    start_dt, end_dt, is_all_day = parse_deadline_to_datetimes(deadline)
    assert not is_all_day
    assert start_dt == datetime.datetime(2026, 10, 26, 10, 0)
    assert end_dt == datetime.datetime(2026, 10, 26, 11, 0)


def test_parse_date_only_deadline():
    deadline = "2026-10-21 (Wed)"
    start_dt, end_dt, is_all_day = parse_deadline_to_datetimes(deadline)
    assert is_all_day
    assert start_dt.date() == datetime.date(2026, 10, 21)


def test_parse_unspecified_deadline():
    anchor = datetime.date(2026, 10, 17)
    start_dt, end_dt, is_all_day = parse_deadline_to_datetimes("None specified", default_anchor=anchor)
    assert not is_all_day
    assert start_dt.date() == datetime.date(2026, 10, 18)


def test_generate_google_calendar_url():
    task = {
        "task_id": "task-test1234",
        "action_description": "Schedule the product roadmap sync for next Monday at 10 AM",
        "deadline": "2026-10-26 (Mon) at 10:00",
        "owner": "Team",
        "evidence_text": "Also, let's schedule the product roadmap sync for next Monday at 10 AM.",
        "thread_title": "Product Roadmap Sync",
    }
    url = generate_google_calendar_url(task)
    assert "calendar.google.com/calendar/render" in url
    assert "action=TEMPLATE" in url
    assert "Schedule+the+product+roadmap+sync" in url or "Schedule%20the%20product%20roadmap%20sync" in url
    assert "20261026" in url


def test_generate_ics_content():
    task = {
        "task_id": "task-test1234",
        "action_description": "Review the Q3 financial report by Thursday afternoon",
        "deadline": "2026-10-22 (Thu)",
        "owner": "Finance Team",
        "evidence_text": "Please review the Q3 financial report by Thursday afternoon.",
        "thread_title": "Q3 Financials",
    }
    ics = generate_ics_content(task)
    assert "BEGIN:VCALENDAR" in ics
    assert "VERSION:2.0" in ics
    assert "BEGIN:VEVENT" in ics
    assert "UID:task-test1234@followthrough.local" in ics
    assert "SUMMARY:Review the Q3 financial report" in ics
    assert "DTSTART;VALUE=DATE:20261022" in ics
    assert "END:VEVENT" in ics
    assert "END:VCALENDAR" in ics
