import datetime
from followthrough_core.temporal_normalizer import normalize_deadline, parse_header_date


def test_parse_header_date():
    header_text = "Date: Friday, Oct 17, 2026\nFrom: Marcus\nSubject: Sprint Review"
    d = parse_header_date(header_text)
    assert d == datetime.date(2026, 10, 17)

    header_text_us = "Sent: 05/07/2001 12:28 PM\nTo: Vince"
    d2 = parse_header_date(header_text_us)
    assert d2 == datetime.date(2001, 5, 7)


def test_normalize_relative_deadlines():
    # Anchor date: Saturday, Oct 17, 2026
    anchor = datetime.date(2026, 10, 17)

    # 1. "tomorrow" -> Sunday, Oct 18, 2026
    res = normalize_deadline("tomorrow", anchor_date=anchor)
    assert res["iso_date"] == "2026-10-18"
    assert "2026-10-18 (Sun)" in res["display_date"]
    assert res["is_relative"] is True

    # 2. "by Wednesday" -> Wednesday, Oct 21, 2026
    res2 = normalize_deadline("by Wednesday", anchor_date=anchor)
    assert res2["iso_date"] == "2026-10-21"
    assert "2026-10-21 (Wed)" in res2["display_date"]

    # 3. "before Friday at 5 PM" -> Friday, Oct 23, 2026 at 17:00
    res3 = normalize_deadline("before Friday at 5 PM", anchor_date=anchor)
    assert res3["iso_date"] == "2026-10-23"
    assert "2026-10-23 (Fri) at 17:00" in res3["display_date"]

    # 4. "next Monday" -> Monday, Oct 26, 2026 (since Oct 19 is upcoming Monday, next Monday is Oct 26)
    res4 = normalize_deadline("next Monday", anchor_date=anchor)
    assert res4["iso_date"] == "2026-10-26"
    assert "2026-10-26 (Mon)" in res4["display_date"]

    # 5. "by end of day" -> Saturday, Oct 17, 2026
    res5 = normalize_deadline("by end of day", anchor_date=anchor)
    assert res5["iso_date"] == "2026-10-17"

    # 6. "before 2 PM" -> Saturday, Oct 17, 2026 at 14:00
    res6 = normalize_deadline("before 2 PM", anchor_date=anchor)
    assert res6["iso_date"] == "2026-10-17"
    assert res6["time_str"] == "14:00"


def test_normalize_absolute_deadlines():
    anchor = datetime.date(2026, 10, 17)

    # Specific calendar date
    res = normalize_deadline("Nov 15th", anchor_date=anchor)
    assert res["iso_date"] == "2026-11-15"
    assert res["is_relative"] is False

    # Numeric date
    res2 = normalize_deadline("12/01/2026", anchor_date=anchor)
    assert res2["iso_date"] == "2026-12-01"


def test_missing_deadline():
    res = normalize_deadline("None specified")
    assert res["iso_date"] is None
    assert res["display_date"] == "None specified"
