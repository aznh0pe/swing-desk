"""Synthetic OHLCV generator for smoke tests (NOT for strategy conclusions)."""
import json, pathlib, sys
import numpy as np, pandas as pd
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
dates = pd.bdate_range("2014-01-02", "2026-11-05")
tick = [f"S{i:02d}" for i in range(60)] + ["SPY", "QQQ", "XLK", "XLF", "TQQQ"]
rows = []
mkt = np.cumsum(rng.normal(0.0004, 0.011, len(dates)))
for t in tick:
    beta = rng.uniform(0.6, 1.6)
    idio = np.zeros(len(dates)); 
    for i in range(1, len(dates)):  # mean-reverting idiosyncratic noise + drift
        idio[i] = 0.93 * idio[i-1] + rng.normal(0, 0.012)
    drift = np.cumsum(rng.normal(0.0002, 0.004, len(dates)))
    logp = np.log(rng.uniform(20, 300)) + beta * mkt + idio + drift
    c = np.exp(logp)
    o = c * np.exp(rng.normal(0, 0.006, len(dates)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.008, len(dates))))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.008, len(dates))))
    v = rng.lognormal(15, 0.4, len(dates))
    rows.append(pd.DataFrame({"date": dates, "ticker": t, "open": o, "high": h, "low": l, "close": c, "volume": v}))
vix = 15 * np.exp(np.cumsum(rng.normal(0, 0.05, len(dates))) * 0.2)
rows.append(pd.DataFrame({"date": dates, "ticker": "^VIX", "open": vix, "high": vix*1.03, "low": vix*0.97, "close": vix, "volume": 0}))
df = pd.concat(rows)
df.to_parquet(out / "history.parquet")
df[df.date >= "2023-11-01"].to_csv(out / "daily.csv.gz", index=False)
last = df[df.date == df.date.max()].copy(); last["close"] *= 0.99; last["asof"] = "2026-11-05T12:05:00-08:00"
last.to_csv(out / "intraday.csv", index=False)
items = [{"ticker": t, "name": t, "type": "stock", "sector": "Information Technology"} for t in tick[:60]]
items += [{"ticker": t, "name": t, "type": "etf", "sector": None} for t in ["SPY", "QQQ", "XLK", "XLF"]]
items += [{"ticker": "TQQQ", "name": "TQQQ", "type": "etf_lev", "sector": None}, {"ticker": "^VIX", "name": "VIX", "type": "index", "sector": None}]
(out / "universe.json").write_text(json.dumps({"items": items}))
(out / "earnings.json").write_text(json.dumps({"tickers": {"S01": {"next": "2026-11-09", "when": "amc"}}}))
(out / "meta.json").write_text(json.dumps({"generated_utc": "2026-11-05T21:35:00Z"}))
print("ok", df.shape)
