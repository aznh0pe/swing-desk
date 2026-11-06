"""NYSE trading calendar helpers (Pacific-time aware).

Hard-coded holiday / early-close tables for 2024-2028 with an optional
pandas_market_calendars upgrade when that package is importable.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")

HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04",
    "2028-09-04", "2028-11-23", "2028-12-25",
}

# Market closes at 1:00 PM ET (10:00 AM PT) on these days.
EARLY_CLOSES = {
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
    "2028-07-03", "2028-11-24",
}

HOLIDAY_NAMES = {
    "2026-11-26": "Thanksgiving Day", "2026-12-25": "Christmas Day",
    "2027-01-01": "New Year's Day", "2027-01-18": "Martin Luther King Jr. Day", "2027-02-15": "Presidents Day",
    "2027-03-26": "Good Friday", "2027-05-31": "Memorial Day", "2027-06-18": "Juneteenth (observed)",
    "2027-07-05": "Independence Day (observed)", "2027-09-06": "Labor Day", "2027-11-25": "Thanksgiving Day",
    "2027-12-24": "Christmas Day (observed)", "2028-01-17": "Martin Luther King Jr. Day",
    "2028-02-21": "Presidents Day", "2028-04-14": "Good Friday", "2028-05-29": "Memorial Day",
    "2028-06-19": "Juneteenth", "2028-07-04": "Independence Day", "2028-09-04": "Labor Day",
    "2028-11-23": "Thanksgiving Day", "2028-12-25": "Christmas Day",
}


def closed_reason(day) -> str:
    """Why the market is shut on a day ('' when it trades)."""
    day = _d(day)
    if is_trading_day(day):
        return ""
    if day.weekday() >= 5:
        return "weekend"
    return HOLIDAY_NAMES.get(day.isoformat(), "market holiday")


def hours_label(day) -> str:
    o, c = session_times_pt(day)
    f = lambda t: t.strftime("%I:%M %p").lstrip("0")
    return f"{f(o)}–{f(c)} PT" + (" (half day)" if is_early_close(day) else "")


try:  # optional, more complete source
    import pandas_market_calendars as _pmc  # type: ignore

    _NYSE = _pmc.get_calendar("XNYS")
except Exception:  # pragma: no cover - fine without it
    _NYSE = None


def _d(x) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x)[:10])


def is_trading_day(day) -> bool:
    day = _d(day)
    if day.weekday() >= 5:
        return False
    if _NYSE is not None:
        try:
            sched = _NYSE.schedule(start_date=day, end_date=day)
            return len(sched) == 1
        except Exception:
            pass
    return day.isoformat() not in HOLIDAYS


def is_early_close(day) -> bool:
    return _d(day).isoformat() in EARLY_CLOSES


def next_trading_day(day, n: int = 1) -> dt.date:
    day = _d(day)
    count = 0
    while count < n:
        day += dt.timedelta(days=1)
        if is_trading_day(day):
            count += 1
    return day


def prev_trading_day(day, n: int = 1) -> dt.date:
    day = _d(day)
    count = 0
    while count < n:
        day -= dt.timedelta(days=1)
        if is_trading_day(day):
            count += 1
    return day


def trading_days_between(start, end) -> int:
    """Trading days in (start, end] — i.e. sessions elapsed after `start` up to `end`."""
    start, end = _d(start), _d(end)
    if end <= start:
        return 0
    n, day = 0, start
    while day < end:
        day += dt.timedelta(days=1)
        if is_trading_day(day):
            n += 1
    return n


def session_times_pt(day):
    """(open, close) as aware PT datetimes for a trading day."""
    day = _d(day)
    open_et = dt.datetime.combine(day, dt.time(9, 30), ET)
    close_et = dt.datetime.combine(day, dt.time(13 if is_early_close(day) else 16, 0), ET)
    return open_et.astimezone(PT), close_et.astimezone(PT)


def now_pt() -> dt.datetime:
    return dt.datetime.now(PT)


def market_phase(now: dt.datetime | None = None) -> str:
    """'closed_day' | 'premarket' | 'open' | 'postclose' for the PT wall clock."""
    now = (now or now_pt()).astimezone(PT)
    today = now.date()
    if not is_trading_day(today):
        return "closed_day"
    o, c = session_times_pt(today)
    if now < o:
        return "premarket"
    if now < c:
        return "open"
    return "postclose"


def last_completed_session(now: dt.datetime | None = None) -> dt.date:
    """Most recent session whose closing bar should exist (allow 25 min after close)."""
    now = (now or now_pt()).astimezone(PT)
    today = now.date()
    if is_trading_day(today):
        _, c = session_times_pt(today)
        if now >= c + dt.timedelta(minutes=25):
            return today
    return prev_trading_day(today)


def next_session(now: dt.datetime | None = None) -> dt.date:
    """The session new entry orders are meant for."""
    now = (now or now_pt()).astimezone(PT)
    today = now.date()
    if is_trading_day(today):
        o, c = session_times_pt(today)
        if now < c:
            return today
    return next_trading_day(today)
