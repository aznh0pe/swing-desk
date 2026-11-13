"""Is the market-data refresh loop on right now?

    python pipeline/keeper_window.py            # exit 0 = keep refreshing, 1 = stop

GitHub runs scheduled workflows about once an hour, however often they are asked for (Nov 13, 2026: one run an
hour all day at 15-minute spacing), so prices on the data branch could be an hour old when a scan ran. The
12:25 PM PT pre-close scan found 55-minute-old prices and had to wait for a refresh. So on a trading day the
workflow keeps itself running from before the pre-market check until after the final daily bars, and refreshes
every 10 minutes instead of waiting for GitHub's schedule.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from engine import calendar as cal  # noqa: E402
except Exception:  # pragma: no cover - the calendar ships with the repo
    cal = None

ET = ZoneInfo("America/New_York")
START = dt.time(6, 30)       # 3:30 AM PT: before the 5:17 AM PT pre-market check (pre-market prices from 4:00 AM ET)
AFTER_CLOSE = 40             # minutes after the close: the final daily bars land 10-20 minutes after it


def window(now_et: dt.datetime) -> tuple[dt.datetime, dt.datetime] | None:
    d = now_et.date()
    trading = cal.is_trading_day(d) if cal else d.weekday() < 5
    if not trading:
        return None
    close = dt.time(13, 0) if (cal and cal.is_early_close(d)) else dt.time(16, 0)
    start = dt.datetime.combine(d, START, ET)
    end = dt.datetime.combine(d, close, ET) + dt.timedelta(minutes=AFTER_CLOSE)
    return start, end


def keep_running(now_et: dt.datetime) -> bool:
    w = window(now_et)
    return bool(w and w[0] <= now_et < w[1])


if __name__ == "__main__":
    now = dt.datetime.now(ET)
    on = keep_running(now)
    print(f"{now:%Y-%m-%d %H:%M} ET: refresh loop {'on' if on else 'off'}")
    sys.exit(0 if on else 1)
