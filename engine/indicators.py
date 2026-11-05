"""Vectorised indicators on wide (dates x tickers) frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Health Care": "XLV", "Industrials": "XLI", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Utilities": "XLU", "Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}


def wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au, ad = wilder(up, n), wilder(dn, n)
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(ad != 0, 100.0)
    return out.where(au.notna())


def true_range(h, l, c):
    pc = c.shift(1)
    return np.fmax(np.fmax(h - l, (h - pc).abs()), (l - pc).abs())


def streak_down(close: pd.DataFrame) -> pd.DataFrame:
    a = close.values
    out = np.zeros_like(a)
    for i in range(1, a.shape[0]):
        down = a[i] < a[i - 1]
        out[i] = np.where(down, out[i - 1] + 1, 0)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def adx(h, l, c, n=14):
    up = h.diff()
    dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(h, l, c)
    atr_ = wilder(tr, n)
    pdi = 100 * wilder(pdm, n) / atr_
    mdi = 100 * wilder(mdm, n) / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n)


def compute(w: dict, universe: pd.DataFrame | None = None) -> dict:
    """Return a dict of indicator frames aligned to w['close']."""
    o, h, l, c, v = (w[k] for k in ["open", "high", "low", "close", "volume"])
    ind = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    for n in (5, 10, 20, 50, 100, 200):
        ind[f"sma{n}"] = c.rolling(n, min_periods=n).mean()
    ind["ema20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ind["ema10"] = c.ewm(span=10, adjust=False, min_periods=10).mean()
    ind["rsi2"] = rsi(c, 2)
    ind["rsi3"] = rsi(c, 3)
    ind["rsi14"] = rsi(c, 14)
    tr = true_range(h, l, c)
    ind["atr10"] = wilder(tr, 10)
    ind["atr14"] = wilder(tr, 14)
    ind["atr_pct"] = ind["atr10"] / c
    rng = (h - l)
    ind["ibs"] = ((c - l) / rng.replace(0, np.nan)).fillna(0.5)
    sd20 = c.rolling(20, min_periods=20).std()
    ind["pctb"] = (c - (ind["sma20"] - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    ind["min7c"] = c.rolling(7, min_periods=7).min()
    ind["max7c"] = c.rolling(7, min_periods=7).max()
    ind["hh10"] = h.rolling(10, min_periods=10).max()
    ind["hh20"] = h.rolling(20, min_periods=20).max()
    ind["ll20"] = l.rolling(20, min_periods=20).min()
    ind["hh252"] = h.rolling(252, min_periods=120).max()
    ind["mean4"] = c.rolling(4, min_periods=4).mean()
    ind["streak"] = streak_down(c)
    ind["slope200"] = ind["sma200"] / ind["sma200"].shift(20) - 1
    ind["slope50"] = ind["sma50"] / ind["sma50"].shift(10) - 1
    ind["dist200"] = c / ind["sma200"] - 1
    ind["dist50"] = c / ind["sma50"] - 1
    ind["v20"] = v.rolling(20, min_periods=10).mean()
    ind["v5"] = v.rolling(5, min_periods=3).mean()
    ind["dv20"] = (c * v).rolling(20, min_periods=10).mean()
    ind["vol_ratio"] = v / ind["v20"].replace(0, np.nan)
    ind["ret5"] = c / c.shift(5) - 1
    ind["ret63"] = c / c.shift(63) - 1
    ind["ret126"] = c / c.shift(126) - 1
    ind["adx14"] = adx(h, l, c, 14)
    tp = (h + l + c) / 3
    sv = v.rolling(20, min_periods=10).sum().replace(0, np.nan)
    vwap = (tp * v).rolling(20, min_periods=10).sum() / sv
    ex2 = (tp * tp * v).rolling(20, min_periods=10).sum() / sv
    ind["vwap20"] = vwap
    ind["vstd20"] = np.sqrt((ex2 - vwap * vwap).clip(lower=0))
    ind["gap_up"] = o / h.shift(1) - 1

    # --- asset types / sectors
    tickers = list(c.columns)
    types = pd.Series("stock", index=tickers)
    sectors = pd.Series(None, index=tickers, dtype=object)
    if universe is not None and len(universe):
        u = universe.reindex(tickers)
        types = u["type"].fillna("stock")
        sectors = u["sector"]
    ind["types"] = types
    ind["sectors"] = sectors
    is_stock = (types == "stock").values
    ind["is_stock"] = pd.DataFrame(np.broadcast_to(is_stock, c.shape), index=c.index, columns=tickers)
    ind["is_etf"] = ~ind["is_stock"]
    is_lev = types.isin(["etf_lev"]).values
    ind["is_lev"] = pd.DataFrame(np.broadcast_to(is_lev, c.shape), index=c.index, columns=tickers)

    # --- cross-sectional relative strength among stocks
    r = 0.6 * ind["ret126"] + 0.4 * ind["ret63"]
    rs_stock = r.loc[:, is_stock] if is_stock.any() else r
    ind["rs_pct"] = rs_stock.rank(axis=1, pct=True).reindex(columns=tickers)

    # --- market regime
    def series(t):
        return c[t] if t in c.columns else None

    spy = series("SPY")
    if spy is None:
        spy = c.loc[:, is_stock].mean(axis=1) if is_stock.any() else c.mean(axis=1)
    mk = pd.DataFrame(index=c.index)
    mk["spy"] = spy
    mk["spy_sma200"] = spy.rolling(200, min_periods=150).mean()
    mk["spy_sma50"] = spy.rolling(50, min_periods=40).mean()
    mk["spy_above200"] = (spy > mk["spy_sma200"]).astype(float)
    mk["spy_dist200"] = spy / mk["spy_sma200"] - 1
    mk["spy_ret5"] = spy / spy.shift(5) - 1
    mk["spy_rsi2"] = rsi(spy.to_frame(), 2).iloc[:, 0]
    st = c.loc[:, is_stock] if is_stock.any() else c
    mk["breadth50"] = (st > ind["sma50"].loc[:, st.columns]).sum(axis=1) / st.notna().sum(axis=1).replace(0, np.nan)
    mk["breadth200"] = (st > ind["sma200"].loc[:, st.columns]).sum(axis=1) / st.notna().sum(axis=1).replace(0, np.nan)
    vix = series("^VIX")
    if vix is None:
        vix = pd.Series(np.nan, index=c.index)
    mk["vix"] = vix.ffill()
    mk["vix_ratio"] = mk["vix"] / mk["vix"].rolling(20, min_periods=10).mean()
    ind["market"] = mk

    # sector ETF trend per ticker (1 if sector ETF above its 50d)
    sec_up = pd.DataFrame(np.nan, index=c.index, columns=tickers)
    for sec, etf in SECTOR_ETF.items():
        if etf in c.columns:
            up = (c[etf] > ind["sma50"][etf]).astype(float)
            cols = sectors[sectors == sec].index
            for t in cols:
                sec_up[t] = up
    for t in tickers:  # ETFs: their own trend
        if types.get(t, "stock") != "stock":
            sec_up[t] = (c[t] > ind["sma50"][t]).astype(float)
    ind["sector_up"] = sec_up.fillna(0.5)
    return ind
