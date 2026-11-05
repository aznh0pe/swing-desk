"""Trade-level and portfolio-level backtests for the plans in strategies.py.

Conservative fill/exit rules:
* Limit entries fill only if the session's low trades at/below the limit; fill price is
  the better of the open and the limit.
* On the fill day only the stop and close-based exits count (we cannot know whether the
  high came before the fill).
* If a later session touches both stop and target, the stop is assumed first.
* Gaps through the stop/target fill at the open.
* 0.05% slippage on market-style exits (stop / signal / time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import features_at


def _arr(x):
    return x.values if hasattr(x, "values") else x


def simulate(plan: dict, ind: dict, P: dict, last_signal_row: int | None = None,
             include_unfilled: bool = False) -> pd.DataFrame:
    O, H, L, C = (_arr(ind[k]) for k in ["open", "high", "low", "close"])
    ATR = _arr(ind["atr10"])
    SMA5 = _arr(ind["sma5"])
    RSI2 = _arr(ind["rsi2"])
    MAX7 = _arr(ind["max7c"])
    T = C.shape[0]
    mask = _arr(plan["mask"]).copy()
    lim_last = T - 1 if last_signal_row is None else last_signal_row
    mask[lim_last:, :] = False
    t_idx, j_idx = np.nonzero(mask)
    if len(t_idx) == 0:
        return pd.DataFrame()
    LIM = _arr(plan["limit"])
    stop_spec, tgt_spec = plan["stop"], plan["target"]
    STOP_LV = _arr(stop_spec["frame"]) if stop_spec["kind"] == "level" else None
    TGT_LV = _arr(tgt_spec["frame"]) if tgt_spec["kind"] == "level" else None
    entry_close = plan["entry_kind"] == "close"
    valid = max(1, int(plan.get("valid_days", 1)))
    hold = int(plan["max_hold"])
    dyn = plan.get("dyn_exit", "none")
    slip = P["risk"].get("slippage_pct", 0.05) / 100.0

    rows = []
    for t, j in zip(t_idx.tolist(), j_idx.tolist()):
        atr = ATR[t, j]
        if not np.isfinite(atr) or atr <= 0:
            continue
        limit = C[t, j] if entry_close else LIM[t, j]
        if not np.isfinite(limit) or limit <= 0:
            continue
        fill_i, fill = -1, np.nan
        if entry_close:
            fill_i, fill = t, C[t, j]
        else:
            for d in range(t + 1, min(t + valid, T - 1) + 1):
                lo = L[d, j]
                if not np.isfinite(lo):
                    continue
                if lo <= limit:
                    fill_i, fill = d, min(O[d, j], limit)
                    break
        if fill_i < 0:
            if include_unfilled:
                rows.append((t, j, -1, np.nan, -1, np.nan, "unfilled", 0, np.nan, np.nan, limit))
            continue
        # stops / targets
        if STOP_LV is not None:
            stop = STOP_LV[t, j]
            if not np.isfinite(stop) or stop >= fill:
                stop = fill - 2.0 * atr
        else:
            stop = fill - stop_spec["mult"] * atr
        if TGT_LV is not None:
            tgt = TGT_LV[t, j]
            lo_t = fill + tgt_spec.get("min_atr", 0.25) * atr
            hi_t = fill + tgt_spec.get("cap_atr", 4.0) * atr
            tgt = lo_t if not np.isfinite(tgt) else min(max(tgt, lo_t), hi_t)
        else:
            tgt = fill + tgt_spec["mult"] * atr
        exit_i, exit_p, reason = -1, np.nan, "open"
        start = fill_i if not entry_close else t + 1
        held = 0
        for d in range(start, T):
            o, hi, lo, cl = O[d, j], H[d, j], L[d, j], C[d, j]
            if not np.isfinite(cl):
                continue
            held += 1
            same_day = (d == fill_i) and not entry_close
            if same_day:
                if lo <= stop:
                    exit_i, exit_p, reason = d, stop * (1 - slip), "stop"
                    break
            else:
                if o <= stop:
                    exit_i, exit_p, reason = d, o * (1 - slip), "stop"
                    break
                if o >= tgt:
                    exit_i, exit_p, reason = d, o, "target"
                    break
                if lo <= stop:
                    exit_i, exit_p, reason = d, stop * (1 - slip), "stop"
                    break
                if hi >= tgt:
                    exit_i, exit_p, reason = d, tgt, "target"
                    break
            sig = False
            if dyn == "sma5":
                sig = cl > SMA5[d, j]
            elif dyn == "prev_high":
                sig = d > 0 and cl > H[d - 1, j]
            elif dyn == "high7":
                sig = cl >= MAX7[d, j]
            elif dyn == "rsi2_70":
                sig = RSI2[d, j] > 70
            if sig:
                exit_i, exit_p, reason = d, cl * (1 - slip), "signal"
                break
            if held >= hold:
                exit_i, exit_p, reason = d, cl * (1 - slip), "time"
                break
        rows.append((t, j, fill_i, fill, exit_i, exit_p, reason, held, stop, tgt, limit))

    if not rows:
        return pd.DataFrame()
    a = np.array(rows, dtype=object)
    df = pd.DataFrame({
        "t": a[:, 0].astype(int), "j": a[:, 1].astype(int), "fill_i": a[:, 2].astype(int),
        "fill": a[:, 3].astype(float), "exit_i": a[:, 4].astype(int),
        "exit": a[:, 5].astype(float), "reason": a[:, 6], "held": a[:, 7].astype(int),
        "stop": a[:, 8].astype(float), "target": a[:, 9].astype(float),
        "limit": a[:, 10].astype(float),
    })
    dates = ind["close"].index
    tick = ind["close"].columns
    df["strategy"] = plan["name"]
    df["ticker"] = tick[df["j"].values]
    df["signal_date"] = dates[df["t"].values]
    df["filled"] = df["fill_i"] >= 0
    ok = df["filled"] & (df["exit_i"] >= 0)
    df["ret"] = np.where(ok, df["exit"] / df["fill"] - 1, np.nan)
    risk = (df["fill"] - df["stop"]).where(df["fill"] > df["stop"])
    df["r_mult"] = np.where(ok, (df["exit"] - df["fill"]) / risk, np.nan)
    df["win"] = df["ret"] > 0
    df["exit_date"] = pd.NaT
    df.loc[ok, "exit_date"] = dates[df.loc[ok, "exit_i"].values]
    df["fill_date"] = pd.NaT
    df.loc[df["filled"], "fill_date"] = dates[df.loc[df["filled"], "fill_i"].values]
    feats = features_at(ind, df["t"].values, df["j"].values)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return df


def stats(tr: pd.DataFrame) -> dict:
    t = tr[tr["filled"] & tr["ret"].notna()] if len(tr) else tr
    n = int(len(t))
    if n == 0:
        return {"trades": 0}
    wins = t[t["ret"] > 0]["ret"]
    losses = t[t["ret"] <= 0]["ret"]
    gross_w, gross_l = wins.sum(), -losses.sum()
    years = max((t["signal_date"].max() - t["signal_date"].min()).days / 365.25, 0.25)
    by_reason = t["reason"].value_counts(normalize=True).round(3).to_dict()
    streak, worst = 0, 0
    for r in t.sort_values("signal_date")["ret"].values:
        streak = streak + 1 if r <= 0 else 0
        worst = max(worst, streak)
    out = {
        "trades": n,
        "win_rate": round(float((t["ret"] > 0).mean()), 4),
        "avg_ret": round(float(t["ret"].mean()), 5),
        "median_ret": round(float(t["ret"].median()), 5),
        "avg_win": round(float(wins.mean()), 5) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 5) if len(losses) else 0.0,
        "profit_factor": round(float(gross_w / gross_l), 3) if gross_l > 0 else None,
        "avg_r": round(float(t["r_mult"].mean()), 3),
        "avg_hold": round(float(t["held"].mean()), 2),
        "trades_per_year": round(n / years, 1),
        "worst_loss": round(float(t["ret"].min()), 4),
        "max_consec_losses": int(worst),
        "exit_mix": by_reason,
    }
    if "filled" in tr:
        out["fill_rate"] = round(float(tr["filled"].mean()), 3) if len(tr) else None
    return out


def yearly(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr[tr["filled"] & tr["ret"].notna()].copy()
    if t.empty:
        return pd.DataFrame()
    t["year"] = t["signal_date"].dt.year
    g = t.groupby("year")["ret"]
    return pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()).round(3),
                         "avg_ret": g.mean().round(4)})


def portfolio(trades: pd.DataFrame, start_capital=5000.0, max_open=5, risk_pct=1.25,
              max_pos_pct=25.0, score_col="score", min_score=None) -> dict:
    """Day-by-day slot simulation over pre-simulated trades (filled ones only).

    Each signal day, orders are placed for the best-scoring candidates up to the free
    slots; a slot is held from fill to exit. Sizing: risk_pct of capital per trade,
    capped at max_pos_pct of capital.
    """
    tr = trades[trades["filled"] & trades["ret"].notna()].copy()
    if min_score is not None and score_col in tr:
        tr = tr[tr[score_col] >= min_score]
    if tr.empty:
        return {"trades": 0}
    if score_col not in tr:
        tr[score_col] = -tr["rsi2"]
    tr = tr.sort_values(["signal_date", score_col], ascending=[True, False])
    capital = start_capital
    open_pos = []  # (exit_date, pnl)
    taken = []
    curve = []
    for day, grp in tr.groupby("signal_date", sort=True):
        # realise exits up to this day
        still = []
        for ex, pnl in open_pos:
            if ex <= day:
                capital += pnl
                curve.append((ex, capital))
            else:
                still.append((ex, pnl))
        open_pos = still
        free = max_open - len(open_pos)
        if free <= 0:
            continue
        used_tickers = set()
        for _, r in grp.iterrows():
            if free <= 0:
                break
            if r["ticker"] in used_tickers:
                continue
            risk_ps = r["fill"] - r["stop"]
            if risk_ps <= 0:
                continue
            dollars = min(capital * risk_pct / 100 / risk_ps * r["fill"], capital * max_pos_pct / 100)
            pnl = dollars * r["ret"]
            open_pos.append((r["exit_date"], pnl))
            taken.append({"date": day, "ticker": r["ticker"], "strategy": r["strategy"],
                          "ret": r["ret"], "pnl": pnl, "dollars": dollars})
            used_tickers.add(r["ticker"])
            free -= 1
    for ex, pnl in sorted(open_pos, key=lambda x: x[0]):
        capital += pnl
        curve.append((ex, capital))
    tk = pd.DataFrame(taken)
    cv = pd.DataFrame(curve, columns=["date", "equity"]).sort_values("date")
    peak = cv["equity"].cummax()
    dd = (cv["equity"] / peak - 1).min() if len(cv) else 0
    years = max((tk["date"].max() - tk["date"].min()).days / 365.25, 0.25)
    cagr = (capital / start_capital) ** (1 / years) - 1 if capital > 0 else -1
    weeks = tk.groupby(pd.Grouper(key="date", freq="W")).size()
    return {"trades": int(len(tk)), "win_rate": round(float((tk["ret"] > 0).mean()), 4),
            "final": round(float(capital), 2), "cagr": round(float(cagr), 4),
            "max_dd": round(float(dd), 4), "trades_per_week": round(float(len(tk) / (years * 52)), 2),
            "pct_weeks_with_3plus": round(float((weeks >= 3).mean()), 3), "curve": cv}
