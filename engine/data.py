"""Market-data access for the engine.

Data lives on the `data` branch of the user's swing-desk GitHub repo, refreshed by
the market-data GitHub Action. Everything here is defensive: retries, fallbacks,
schema checks and staleness detection so a scan never dies on bad input.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd

from . import calendar as cal

FIELDS = ["open", "high", "low", "close", "volume"]
DATA_FILES = [
    "daily.csv.gz", "intraday.csv", "premarket.csv", "earnings.json",
    "universe.json", "meta.json",
]


# --------------------------------------------------------------------------- fetch

def _run(cmd, timeout=180):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def fetch_branch(repo: str, branch: str, dest: str, files: list[str] | None = None,
                 attempts: int = 4) -> dict:
    """Shallow-clone `branch` of `repo` ("owner/name") into dest.

    Falls back to raw.githubusercontent.com per-file downloads. Returns a report
    dict; raises only if nothing usable could be fetched.
    """
    dest_p = pathlib.Path(dest)
    report = {"method": None, "attempts": 0, "errors": []}
    url = f"https://github.com/{repo}.git"
    for i in range(attempts):
        report["attempts"] += 1
        if dest_p.exists():
            shutil.rmtree(dest_p, ignore_errors=True)
        try:
            r = _run(["git", "clone", "--depth", "1", "--single-branch", "--branch", branch,
                      url, str(dest_p)], timeout=300)
            if r.returncode == 0:
                report["method"] = "git"
                return report
            report["errors"].append(r.stderr.strip()[-300:])
        except Exception as e:  # timeout etc.
            report["errors"].append(repr(e)[:300])
        time.sleep(min(30, 3 * 2 ** i))
    # raw fallback
    dest_p.mkdir(parents=True, exist_ok=True)
    got = 0
    for f in files or DATA_FILES:
        raw = f"https://raw.githubusercontent.com/{repo}/{branch}/{f}"
        for i in range(3):
            try:
                with urllib.request.urlopen(raw, timeout=120) as resp:
                    (dest_p / f).parent.mkdir(parents=True, exist_ok=True)
                    (dest_p / f).write_bytes(resp.read())
                got += 1
                break
            except Exception as e:
                report["errors"].append(f"{f}: {e!r}"[:200])
                time.sleep(2 * (i + 1))
    if got == 0:
        raise RuntimeError(f"Could not fetch branch {branch} of {repo}: {report['errors'][-3:]}")
    report["method"] = f"raw({got} files)"
    return report


# --------------------------------------------------------------------------- load

def _read_json(path, default=None):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except Exception:
        return default


def load_meta(d):
    return _read_json(pathlib.Path(d) / "meta.json", {}) or {}


def load_universe(d) -> pd.DataFrame:
    js = _read_json(pathlib.Path(d) / "universe.json", None)
    if js and js.get("items"):
        u = pd.DataFrame(js["items"])
    else:  # fall back to whatever is in the price file
        u = pd.DataFrame(columns=["ticker", "name", "type", "sector", "sector_etf"])
    for c in ["name", "type", "sector", "sector_etf"]:
        if c not in u:
            u[c] = None
    return u.set_index("ticker", drop=False)


def load_earnings(d) -> dict:
    js = _read_json(pathlib.Path(d) / "earnings.json", {}) or {}
    return js.get("tickers", {})


def _clean_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    need = {"date", "ticker", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"price file missing columns: {need - set(df.columns)}")
    if "volume" not in df:
        df["volume"] = np.nan
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    # repair OHLC inconsistencies instead of dropping
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df = df.drop_duplicates(["date", "ticker"], keep="last")
    return df


def read_prices(path) -> pd.DataFrame:
    path = pathlib.Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _clean_long(df)


def to_wide(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for f in FIELDS:
        w = long_df.pivot(index="date", columns="ticker", values=f).sort_index()
        out[f] = w.astype("float64")
    # keep only tickers that have closes
    cols = out["close"].columns[out["close"].notna().any()]
    return {k: v[cols] for k, v in out.items()}


def load_daily(d) -> dict[str, pd.DataFrame]:
    return to_wide(read_prices(pathlib.Path(d) / "daily.csv.gz"))


def load_history(d) -> dict[str, pd.DataFrame]:
    p = pathlib.Path(d)
    for name in ["history.parquet", "history.csv.gz"]:
        if (p / name).exists():
            return to_wide(read_prices(p / name))
    raise FileNotFoundError("no history file in " + str(p))


def load_intraday(d) -> pd.DataFrame | None:
    p = pathlib.Path(d) / "intraday.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        return _clean_long(df).assign(asof=df.get("asof"))
    except Exception:
        return None


def load_premarket(d) -> pd.DataFrame | None:
    p = pathlib.Path(d) / "premarket.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df.set_index("ticker") if not df.empty else None
    except Exception:
        return None


def append_partial_bar(wide: dict, intra: pd.DataFrame | None) -> tuple[dict, bool]:
    """Add/replace today's partial bar from the intraday snapshot."""
    if intra is None or intra.empty:
        return wide, False
    day = intra["date"].max()
    snap = intra[intra["date"] == day].set_index("ticker")
    last = wide["close"].index.max()
    if day < last:
        return wide, False
    out = {}
    for f in FIELDS:
        w = wide[f]
        if day > last:
            w = pd.concat([w, pd.DataFrame(index=[day], columns=w.columns, dtype="float64")])
        else:
            w = w.copy()
        vals = snap[f].reindex(w.columns)
        mask = vals.notna()
        w.loc[day, mask[mask].index] = vals[mask].values
        out[f] = w
    # tickers with no snapshot keep NaN for the partial day -> forward-fill close-like fields
    for f in ["open", "high", "low", "close"]:
        out[f].iloc[-1] = out[f].iloc[-1].fillna(out["close"].iloc[-2])
    out["volume"].iloc[-1] = out["volume"].iloc[-1].fillna(0)
    return out, True


# --------------------------------------------------------------------------- health

def data_health(wide: dict, meta: dict, mode: str, now=None) -> dict:
    now = now or cal.now_pt()
    notes, status = [], "ok"
    last = wide["close"].index.max().date()
    expected = cal.last_completed_session(now)
    lag = cal.trading_days_between(last, expected)
    if lag > 0:
        status = "stale"
        notes.append(f"Latest daily bar is {last}, expected {expected} ({lag} session(s) behind).")
    coverage = wide["close"].iloc[-1].notna().mean()
    if coverage < 0.85:
        status = "degraded" if status == "ok" else status
        notes.append(f"Only {coverage:.0%} of tickers have a bar for {last}.")
    gen = meta.get("generated_utc")
    if gen:
        try:
            age_min = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(gen)).total_seconds() / 60
            notes.append(f"Data feed last refreshed {age_min:.0f} min ago.")
        except Exception:
            pass
    return {"status": status, "last_bar": str(last), "expected_bar": str(expected),
            "sessions_behind": int(lag), "coverage": round(float(coverage), 3), "notes": notes}
