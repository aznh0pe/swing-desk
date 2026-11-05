"""Market-data feed for Swing Desk (runs in GitHub Actions).

    python pipeline/fetch_data.py --mode auto --out out

Modes: auto | daily | intraday | premarket | earnings | history | all
`out/` holds the current contents of the `data` branch; this script updates it in place.
Design rules: never overwrite good data with worse data, retry everything, record what
failed in meta.json so the scanner can see data health.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import json
import pathlib
import sys
import time
import traceback
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import universe as U  # noqa: E402

try:
    from engine import calendar as cal  # noqa: E402
except Exception:  # pragma: no cover
    cal = None

ET = ZoneInfo("America/New_York")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BATCH = 60
errors: list[str] = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- yahoo

def _long_from_yf(df: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = []
    multi = isinstance(df.columns, pd.MultiIndex)
    for t in batch:
        try:
            if multi:
                lvl0 = df.columns.get_level_values(0)
                if t in lvl0:
                    sub = df[t]
                else:  # (field, ticker) layout
                    sub = df.xs(t, axis=1, level=1)
            else:
                sub = df
            sub = sub.rename(columns=str.lower)
            if "close" not in sub:
                continue
            sub = sub.dropna(subset=["close"])
            if sub.empty:
                continue
            idx = pd.to_datetime(sub.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(ET).tz_localize(None)
            out.append(pd.DataFrame({"date": idx, "ticker": t,
                                     "open": sub.get("open"), "high": sub.get("high"),
                                     "low": sub.get("low"), "close": sub["close"],
                                     "volume": sub.get("volume", np.nan)}))
        except Exception as e:
            errors.append(f"parse {t}: {e!r}"[:160])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def yf_download(tickers: list[str], **kw) -> tuple[pd.DataFrame, list[str]]:
    import yfinance as yf

    frames, failed = [], []
    adj = kw.pop("auto_adjust", True)
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        got = pd.DataFrame()
        for attempt in range(4):
            try:
                df = yf.download(batch, group_by="ticker", threads=True, progress=False,
                                 auto_adjust=adj, timeout=30, **kw)
                got = _long_from_yf(df, batch)
                if not got.empty:
                    break
            except Exception as e:
                errors.append(f"yf batch {i}: {e!r}"[:200])
            time.sleep(5 * (attempt + 1))
        have = set(got["ticker"]) if not got.empty else set()
        missing = [t for t in batch if t not in have]
        # retry stragglers one by one
        for t in missing:
            ok = False
            for attempt in range(3):
                try:
                    df = yf.download(t, progress=False, auto_adjust=adj, timeout=30, **kw)
                    one = _long_from_yf(df, [t])
                    if not one.empty:
                        got = pd.concat([got, one], ignore_index=True)
                        ok = True
                        break
                except Exception as e:
                    errors.append(f"yf {t}: {e!r}"[:160])
                time.sleep(2 * (attempt + 1))
            if not ok:
                failed.append(t)
        if not got.empty:
            frames.append(got)
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, failed


def stooq_daily(t: str, start: str) -> pd.DataFrame:
    if t.startswith("^"):
        return pd.DataFrame()
    sym = t.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d&d1={start.replace('-', '')}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df = df.rename(columns=str.lower)
    if "close" not in df:
        return pd.DataFrame()
    return pd.DataFrame({"date": pd.to_datetime(df["date"]), "ticker": t, "open": df["open"],
                         "high": df["high"], "low": df["low"], "close": df["close"],
                         "volume": df.get("volume", np.nan)})


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["close"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float).round(4)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).round().astype("int64")
    df = df[(df["close"] > 0)]
    return df.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])


# ----------------------------------------------------------------------------- modes

def mode_daily(out: pathlib.Path, tickers: list[str], meta: dict, now_et: dt.datetime):
    start = (now_et.date() - dt.timedelta(days=1150)).isoformat()
    log(f"daily: {len(tickers)} tickers from {start}")
    new, failed = yf_download(tickers, start=start, interval="1d")
    if failed:
        log(f"daily: {len(failed)} failed on Yahoo, trying Stooq")
        rec = []
        for t in failed[:80]:
            try:
                d = stooq_daily(t, start)
                if not d.empty:
                    rec.append(d)
            except Exception as e:
                errors.append(f"stooq {t}: {e!r}"[:120])
        if rec:
            new = pd.concat([new] + rec, ignore_index=True)
            got = set(pd.concat(rec)["ticker"])
            failed = [t for t in failed if t not in got]
    new = _clean(new)
    old_p = out / "daily.csv.gz"
    old = pd.read_csv(old_p) if old_p.exists() else pd.DataFrame()
    have = set(new["ticker"]) if not new.empty else set()
    coverage = len(have) / max(1, len(tickers))
    if coverage < 0.6 and not old.empty:
        errors.append(f"daily coverage {coverage:.0%}: kept previous file")
        meta["daily_status"] = "kept_previous"
        return
    if not old.empty:
        keep = old[old["ticker"].isin(set(tickers) - have)]
        if len(keep):
            new = pd.concat([new, _clean(keep)], ignore_index=True)
    new.to_csv(old_p, index=False, compression="gzip")
    last = new["date"].max()
    meta.update({"daily_last_date": str(pd.Timestamp(last).date()), "daily_tickers": int(new["ticker"].nunique()),
                 "daily_failed": failed[:50], "daily_status": "ok" if coverage >= 0.9 else "partial",
                 "daily_updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    # final bars are reliable ~20 minutes after the close
    close_t = dt.time(13, 0) if (cal and cal.is_early_close(now_et.date())) else dt.time(16, 0)
    if pd.Timestamp(last).date() == now_et.date() and now_et.time() >= (dt.datetime.combine(now_et.date(), close_t)
                                                                         + dt.timedelta(minutes=20)).time():
        meta["daily_final_for"] = str(now_et.date())
    log(f"daily: wrote {len(new)} rows, last {last}, coverage {coverage:.0%}")


def mode_intraday(out: pathlib.Path, tickers: list[str], meta: dict, now_et: dt.datetime):
    log("intraday snapshot")
    df, failed = yf_download(tickers, period="5d", interval="1d")
    df = _clean(df)
    if df.empty:
        errors.append("intraday: no data")
        return
    today = pd.Timestamp(now_et.date())
    snap = df[df["date"] == today].copy()
    if snap.empty:
        errors.append("intraday: no bar for today yet")
        return
    snap["asof"] = now_et.isoformat(timespec="minutes")
    snap.to_csv(out / "intraday.csv", index=False)
    meta.update({"intraday_asof": now_et.isoformat(timespec="minutes"), "intraday_tickers": int(len(snap))})
    log(f"intraday: {len(snap)} tickers")


def mode_premarket(out: pathlib.Path, tickers: list[str], meta: dict, now_et: dt.datetime):
    log("pre-market snapshot")
    import yfinance as yf

    rows = []
    for i in range(0, len(tickers), BATCH):
        batch = [t for t in tickers[i:i + BATCH] if not t.startswith("^")]
        for attempt in range(3):
            try:
                df = yf.download(batch, period="1d", interval="5m", prepost=True, group_by="ticker",
                                 threads=True, progress=False, auto_adjust=False, timeout=30)
                lg = _long_from_yf(df, batch)
                break
            except Exception as e:
                errors.append(f"premarket batch {i}: {e!r}"[:160])
                lg = pd.DataFrame()
                time.sleep(4 * (attempt + 1))
        if lg.empty:
            continue
        # _long_from_yf normalised timestamps to naive ET
        for t, g in lg.groupby("ticker"):
            g = g[g["date"].dt.time < dt.time(9, 30)]
            if g.empty:
                continue
            rows.append({"ticker": t, "pm_last": round(float(g["close"].iloc[-1]), 4),
                         "pm_high": round(float(g["high"].max()), 4), "pm_low": round(float(g["low"].min()), 4),
                         "pm_volume": int(pd.to_numeric(g["volume"], errors="coerce").fillna(0).sum()),
                         "asof": now_et.isoformat(timespec="minutes")})
        time.sleep(1)
    if rows:
        pd.DataFrame(rows).to_csv(out / "premarket.csv", index=False)
        meta.update({"premarket_asof": now_et.isoformat(timespec="minutes"), "premarket_tickers": len(rows)})
    log(f"pre-market: {len(rows)} tickers")


def _nasdaq_day(day: dt.date) -> list[dict]:
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                                               "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        js = json.loads(r.read().decode())
    return ((js or {}).get("data") or {}).get("rows") or []


def mode_earnings(out: pathlib.Path, stocks: list[str], meta: dict, now_et: dt.datetime):
    log("earnings calendar")
    res: dict[str, dict] = {}
    day, days, hits = now_et.date() - dt.timedelta(days=7), 0, 0
    want = set(stocks)
    while days < 32:
        if day.weekday() < 5:
            for attempt in range(3):
                try:
                    rows = _nasdaq_day(day)
                    for r in rows:
                        s = (r.get("symbol") or "").upper().replace(".", "-")
                        if s in want:
                            hits += 1
                            when = {"time-pre-market": "bmo", "time-after-hours": "amc"}.get(r.get("time"), "?")
                            ent = res.setdefault(s, {})
                            if day >= now_et.date():
                                ent.setdefault("next", day.isoformat())
                                ent.setdefault("when", when)
                            else:
                                ent["last"] = day.isoformat()
                    break
                except Exception as e:
                    if attempt == 2:
                        errors.append(f"nasdaq earnings {day}: {e!r}"[:160])
                    time.sleep(3 * (attempt + 1))
            time.sleep(0.6)
        day += dt.timedelta(days=1)
        days += 1
    source = "nasdaq"
    if hits < 40:  # Nasdaq blocked or broken: fall back to Yahoo per ticker
        source = "yahoo"
        import yfinance as yf

        def one(t):
            try:
                c = yf.Ticker(t).calendar
                d = c.get("Earnings Date") if isinstance(c, dict) else None
                if d:
                    d0 = d[0] if isinstance(d, (list, tuple)) else d
                    return t, {"next": pd.Timestamp(d0).date().isoformat(), "when": "?"}
            except Exception:
                return t, None
            return t, None

        with cf.ThreadPoolExecutor(6) as ex:
            for t, v in ex.map(one, stocks):
                if v and v["next"] >= now_et.date().isoformat():
                    res[t] = v
    (out / "earnings.json").write_text(json.dumps({"asof": now_et.isoformat(timespec="minutes"),
                                                   "source": source, "tickers": res}, indent=0))
    meta.update({"earnings_asof": now_et.isoformat(timespec="minutes"), "earnings_source": source,
                 "earnings_count": len(res)})
    log(f"earnings: {len(res)} tickers via {source}")


def mode_history(out: pathlib.Path, tickers: list[str], meta: dict, now_et: dt.datetime):
    log("history rebuild from 2008")
    df, failed = yf_download(tickers, start="2008-01-01", interval="1d")
    df = _clean(df)
    if df.empty or df["ticker"].nunique() < 0.6 * len(tickers):
        errors.append("history: too little data, kept previous")
        return
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype("float32")
    df = df.sort_values(["ticker", "date"])
    # split by period so no single file nears GitHub's 50 MB warning size
    for old_f in out.glob("history*.parquet"):
        old_f.unlink()
    yr = df["date"].dt.year
    for name, m in (("history_2008_2014.parquet", yr <= 2014), ("history_2015_2020.parquet", (yr >= 2015) & (yr <= 2020)),
                    ("history_2021_now.parquet", yr >= 2021)):
        df[m].to_parquet(out / name, index=False, compression="zstd")
    meta.update({"history_asof": now_et.isoformat(timespec="minutes"), "history_tickers": int(df["ticker"].nunique()),
                 "history_failed": failed[:50]})
    log(f"history: {len(df)} rows, {df['ticker'].nunique()} tickers")


# ----------------------------------------------------------------------------- driver

def plan_auto(now_et: dt.datetime, meta: dict, out: pathlib.Path) -> list[str]:
    modes = []
    d, t = now_et.date(), now_et.time()
    trading = cal.is_trading_day(d) if cal else d.weekday() < 5
    close_t = dt.time(13, 0) if (cal and cal.is_early_close(d)) else dt.time(16, 0)
    hist_age = None
    if meta.get("history_asof"):
        hist_age = (now_et - dt.datetime.fromisoformat(meta["history_asof"])).days
    if not list(out.glob("history*.parquet")) or (hist_age is not None and hist_age >= 7):
        modes.append("history")
    if not (out / "daily.csv.gz").exists():
        modes.append("daily")
    if trading and dt.time(4, 0) <= t < dt.time(9, 30):
        modes.append("premarket")
    elif trading and dt.time(9, 30) <= t < close_t:
        modes.append("intraday")
    if trading and t >= dt.time(close_t.hour, 10) and meta.get("daily_final_for") != str(d):
        modes.append("daily")
    if meta.get("earnings_asof", "")[:10] != str(d) and t >= dt.time(16, 10):
        modes.append("earnings")
    if not trading and meta.get("daily_last_date") and cal:
        exp = cal.prev_trading_day(d)
        if meta["daily_last_date"] < str(exp):
            modes.append("daily")
    seen = []
    for m in modes:
        if m not in seen:
            seen.append(m)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    now_et = dt.datetime.now(ET)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    items = U.build(out / "universe.json")
    (out / "universe.json").write_text(json.dumps({"asof": now_et.isoformat(timespec="minutes"), "items": items}))
    tickers = [i["ticker"] for i in items]
    stocks = [i["ticker"] for i in items if i["type"] == "stock"]
    if a.mode == "auto":
        modes = plan_auto(now_et, meta, out)
    elif a.mode == "all":
        modes = ["history", "daily", "earnings"] + (["intraday"] if cal and cal.market_phase() == "open" else [])
    else:
        modes = [m.strip() for m in a.mode.split(",")]
    log(f"now {now_et:%Y-%m-%d %H:%M} ET, modes {modes}, universe {len(tickers)}")
    for m in modes:
        try:
            {"daily": lambda: mode_daily(out, tickers, meta, now_et),
             "intraday": lambda: mode_intraday(out, tickers, meta, now_et),
             "premarket": lambda: mode_premarket(out, tickers, meta, now_et),
             "earnings": lambda: mode_earnings(out, stocks, meta, now_et),
             "history": lambda: mode_history(out, tickers, meta, now_et)}[m]()
        except Exception as e:
            errors.append(f"{m} crashed: {e!r}"[:300])
            traceback.print_exc()
    meta.update({"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "generated_et": now_et.isoformat(timespec="minutes"), "modes_run": modes,
                 "universe": len(tickers), "errors": errors[-40:], "version": 1})
    meta_p.write_text(json.dumps(meta, indent=1))
    (out / "README.md").write_text("Swing Desk market-data branch. Rewritten on every run (single commit).\n")
    log("done", json.dumps({k: meta.get(k) for k in ("daily_last_date", "intraday_asof", "earnings_count")}))
    # signal to the workflow whether anything changed
    print(f"::notice::modes={','.join(modes) or 'none'} errors={len(errors)}")


if __name__ == "__main__":
    main()
