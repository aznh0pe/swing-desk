"""Win-probability model and setup score.

p(win) comes from a per-strategy logistic model fit on backtest trades (features at the
signal close), shifted by a live-performance offset that the daily learning loop
maintains. Score (0-100) blends p(win), expected return, regime and setup quality.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .strategies import FEATURES

CLIP = {"rsi2": (0, 100), "depth_atr": (-3, 6), "dist200": (-0.5, 1.5), "slope200": (-0.2, 0.3),
        "atr_pct": (0, 0.2), "rs_pct": (0, 1), "ibs": (0, 1), "streak": (0, 10), "adx14": (0, 80),
        "vol_ratio": (0, 6), "dist50": (-0.4, 0.8), "ret5": (-0.4, 0.4), "spy_above200": (0, 1),
        "spy_dist200": (-0.4, 0.4), "spy_ret5": (-0.2, 0.2), "breadth50": (0, 1), "vix": (8, 90),
        "vix_ratio": (0.5, 3), "sector_up": (0, 1), "is_etf": (0, 1)}


def _prep(X: pd.DataFrame, med: dict | None = None):
    X = X.reindex(columns=FEATURES).astype(float).copy()
    for c, (lo, hi) in CLIP.items():
        if c in X:
            X[c] = X[c].clip(lo, hi)
    if med is None:
        med = {c: (float(X[c].median()) if X[c].notna().any() else 0.0) for c in FEATURES}
    X = X.fillna(med)
    return X, med


def fit(trades: pd.DataFrame, min_trades: int = 150, split_date=None) -> dict:
    t = trades[trades["filled"] & trades["ret"].notna()].copy()
    base = float((t["ret"] > 0).mean()) if len(t) else 0.5
    model = {"kind": "base", "base_rate": round(base, 4), "n": int(len(t))}
    if len(t) > 0:
        wins = t[t["ret"] > 0]["ret"]
        losses = t[t["ret"] <= 0]["ret"]
        model["avg_win"] = round(float(wins.mean()), 5) if len(wins) else 0.0
        model["avg_loss"] = round(float(losses.mean()), 5) if len(losses) else 0.0
    if len(t) < min_trades:
        return model
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
    except Exception:
        return model
    y = (t["ret"] > 0).astype(int).values
    X, med = _prep(t)
    mu = X.mean()
    sd = X.std().replace(0, 1).fillna(1)
    Z = ((X - mu) / sd).values
    val = {}
    if split_date is not None:
        tr_m = (t["signal_date"] < split_date).values
        te_m = ~tr_m
        if tr_m.sum() >= min_trades and te_m.sum() >= 50 and len(set(y[te_m])) == 2:
            lr = LogisticRegression(C=0.3, max_iter=500)
            lr.fit(Z[tr_m], y[tr_m])
            pt = lr.predict_proba(Z[te_m])[:, 1]
            val = {"oos_trades": int(te_m.sum()), "oos_auc": round(float(roc_auc_score(y[te_m], pt)), 3),
                   "oos_brier": round(float(brier_score_loss(y[te_m], pt)), 4),
                   "oos_base_brier": round(float(brier_score_loss(y[te_m], np.full(te_m.sum(), y[tr_m].mean()))), 4)}
            # calibration by quintile of predicted p
            q = pd.qcut(pt, 5, labels=False, duplicates="drop")
            cal = pd.DataFrame({"p": pt, "y": y[te_m], "q": q}).groupby("q").agg(p=("p", "mean"), win=("y", "mean"), n=("y", "size"))
            val["oos_calibration"] = cal.round(3).reset_index(drop=True).to_dict("records")
    if len(set(y)) < 2:
        return model
    lr = LogisticRegression(C=0.3, max_iter=500)
    lr.fit(Z, y)
    model.update({"kind": "logit", "features": FEATURES, "median": med,
                  "mean": {k: float(v) for k, v in mu.items()},
                  "std": {k: float(v) for k, v in sd.items()},
                  "coef": {k: float(v) for k, v in zip(FEATURES, lr.coef_[0])},
                  "intercept": float(lr.intercept_[0]), "validation": val})
    return model


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def predict(model: dict | None, feats: pd.DataFrame, offset: float = 0.0) -> np.ndarray:
    if not model:
        return np.full(len(feats), 0.6)
    if model.get("kind") != "logit":
        b = _logit(model.get("base_rate", 0.6)) + offset
        return np.full(len(feats), 1 / (1 + math.exp(-b)))
    X, _ = _prep(feats, model["median"])
    z = np.zeros(len(X)) + model["intercept"] + offset
    for f in model["features"]:
        z += model["coef"][f] * (X[f].values - model["mean"][f]) / (model["std"][f] or 1)
    return 1 / (1 + np.exp(-z))


def score(p: np.ndarray, feats: pd.DataFrame, model: dict | None, rr: np.ndarray | None = None) -> np.ndarray:
    m = model or {}
    aw, al = m.get("avg_win", 0.025), m.get("avg_loss", -0.035)
    exp_ret = p * aw + (1 - p) * al
    s_p = np.clip((p - 0.50) / (0.88 - 0.50), 0, 1)
    s_e = np.clip(exp_ret / 0.02, 0, 1)
    f = feats.fillna({"spy_above200": 0.5, "sector_up": 0.5, "rs_pct": 0.5, "dist200": 0.0})
    s_reg = 0.5 * f["spy_above200"].values + 0.5 * f["sector_up"].values
    s_q = 0.45 * np.clip(f["dist200"].values / 0.15, 0, 1) + 0.35 * f["rs_pct"].values \
        + 0.20 * (np.clip(rr, 0, 1.5) / 1.5 if rr is not None else 0.5)
    sc = 100 * (0.50 * s_p + 0.20 * s_e + 0.15 * s_reg + 0.15 * s_q)
    return np.round(sc, 1)


def grade(score_v: float, p: float, G: dict) -> str:
    if score_v >= G.get("aplus_score", 80) and p >= G.get("aplus_pwin", 0.70):
        return "A+"
    if score_v >= G.get("aplus_score", 80) - 10 and p >= G.get("aplus_pwin", 0.70) - 0.05:
        return "A"
    return "B"


def live_offset(base_rate: float, wins: int, n: int, prior_n: float = 60.0, max_shift: float = 0.6) -> float:
    """Logit shift that moves a strategy's base win-rate toward its live record."""
    if n <= 0:
        return 0.0
    a = base_rate * prior_n + wins
    b = (1 - base_rate) * prior_n + (n - wins)
    post = a / (a + b)
    return float(np.clip(_logit(post) - _logit(base_rate), -max_shift, max_shift))
