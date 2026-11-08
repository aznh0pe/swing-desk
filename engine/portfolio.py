"""Portfolio state (from the dashboard database export) and position sizing."""
from __future__ import annotations

import json
import math
import pathlib

DEFAULT_SETTINGS = {
    "initial_balance": 5000.0,
    "income_default_pct": 30.0,
    "max_open_positions": 5,
    "risk_pct": 2.0,
    "max_position_pct": 25.0,
}

# Risk governor: the full risk per trade is earned by the live record. Position size steps down to the
# floor while the account is in a drawdown deeper than `dd_step_down` from its peak (realised P&L), or
# while the last `window` closed trades won less than `win_step_down`; it steps back up once the drawdown
# is inside `dd_restore` and the trailing win rate is back above `win_restore`. Nov 2026 backtest, A+ only,
# 5 slots, 2008-2026: 1.25% risk -> +6.0%/yr with a -6.8% worst drawdown; 2.0% -> +9.6%/yr, -10.8%.
# Win rate does not change with size, so the governor only protects the account while the edge is unproven.
GOVERNOR = {"floor_pct": 1.25, "dd_step_down": 0.08, "dd_restore": 0.04, "window": 20,
            "win_step_down": 0.60, "win_restore": 0.65}


def risk_governor(state: dict, capital: float, settings: dict) -> dict:
    """Effective risk per trade for this scan, with the reason, from the account's own record."""
    full = float(settings.get("risk_pct", DEFAULT_SETTINGS["risk_pct"]))
    g = {**GOVERNOR, **(settings.get("governor") or {})}
    floor = min(full, float(g["floor_pct"]))
    led = sorted((v["doc"] for v in state["ledger"].values()), key=lambda d: str(d.get("date") or d.get("at") or ""))
    eq, peak = 0.0, 0.0
    for d in led:
        eq += float(d.get("amount") or 0)
        if d.get("type") in ("initial", "deposit", "withdrawal"):
            peak = max(eq, peak) if d.get("type") != "withdrawal" else max(peak + float(d.get("amount") or 0), eq)
        peak = max(peak, eq)
    if not led:
        eq = peak = capital
    dd = (peak - eq) / peak if peak > 0 else 0.0
    closed = [v["doc"] for v in state["trades"].values()
              if v["doc"].get("status") == "closed" and v["doc"].get("exit_price") and v["doc"].get("entry_price")]
    closed.sort(key=lambda t: str(t.get("exit_date") or t.get("closed_at") or ""))
    recent = closed[-int(g["window"]):]
    wr = (sum(1 for t in recent if float(t["exit_price"]) > float(t["entry_price"])) / len(recent)) if len(recent) >= int(g["window"]) else None
    was_down = bool((state.get("learning", {}).get("governor") or {}).get("doc", {}).get("stepped_down"))
    reasons = []
    if dd >= g["dd_step_down"] or (was_down and dd > g["dd_restore"]):
        reasons.append(f"drawdown {100 * dd:.1f}% from the account's peak")
    if wr is not None and (wr < g["win_step_down"] or (was_down and wr < g["win_restore"])):
        reasons.append(f"last {len(recent)} trades won {100 * wr:.0f}%")
    if reasons:
        return {"risk_pct": floor, "full_pct": full, "stepped_down": True, "drawdown": round(dd, 4), "trailing_win": wr,
                "reason": "stepped down to %.2f%% (%s); back to %.2f%% once the drawdown is under %d%% and the last %d trades win %d%%+"
                % (floor, "; ".join(reasons), full, round(100 * g["dd_restore"]), g["window"], round(100 * g["win_restore"]))}
    return {"risk_pct": full, "full_pct": full, "stepped_down": False, "drawdown": round(dd, 4), "trailing_win": wr,
            "reason": "full size: drawdown %.1f%% from peak%s" % (100 * dd, f", last {len(recent)} trades won {100 * wr:.0f}%" if wr is not None else "")}


def _unwrap(obj):
    """ArtifactData exports may wrap a document; return (doc, version)."""
    if not isinstance(obj, dict):
        return {}, None
    version = obj.get("version") or obj.get("_version") or obj.get("__version__")
    for key in ("data", "fields", "doc", "document", "value"):
        if isinstance(obj.get(key), dict) and len(obj) <= 6:
            return obj[key], version or obj[key].get("version")
    doc = {k: v for k, v in obj.items() if k not in ("version", "_version", "__version__")}
    return doc, version


def load_collection(state_dir, name) -> dict:
    """{doc_id: {'doc': {...}, 'version': n}} for <state_dir>/<name>/*.json."""
    out = {}
    p = pathlib.Path(state_dir) / name
    if not p.exists():
        return out
    for f in sorted(p.glob("*.json")):
        try:
            doc, ver = _unwrap(json.loads(f.read_text()))
        except Exception:
            continue
        doc_id = doc.get("id") or f.stem
        out[doc_id] = {"doc": doc, "version": ver}
    return out


def load_state(state_dir) -> dict:
    sd = pathlib.Path(state_dir) if state_dir else None
    if sd is None or not sd.exists():
        return {"settings": dict(DEFAULT_SETTINGS), "trades": {}, "ledger": {}, "alerts": {}, "scans": {}, "learning": {},
                "params_doc": None, "versions": {}}
    settings = dict(DEFAULT_SETTINGS)
    cfg = load_collection(sd, "config")
    if "settings" in cfg:
        settings.update({k: v for k, v in cfg["settings"]["doc"].items() if v is not None})
    params_doc = None
    strat = load_collection(sd, "strategy")
    if "params" in strat:
        params_doc = strat["params"]["doc"]
    return {"settings": settings, "trades": load_collection(sd, "trades"),
            "ledger": load_collection(sd, "ledger"), "alerts": load_collection(sd, "alerts"),
            "scans": load_collection(sd, "scans"), "learning": load_collection(sd, "learning"), "params_doc": params_doc,
            "versions": {"config/settings": cfg.get("settings", {}).get("version"),
                         "strategy/params": strat.get("params", {}).get("version")}}


def funds(state: dict) -> dict:
    s = state["settings"]
    led = [v["doc"] for v in state["ledger"].values()]
    has_initial = any(d.get("type") == "initial" for d in led)
    total = 0.0 if has_initial else float(s.get("initial_balance", 5000))
    parts = {"initial": 0.0 if has_initial else total, "deposit": 0.0, "withdrawal": 0.0,
             "pnl": 0.0, "income": 0.0}
    for d in led:
        amt = float(d.get("amount") or 0)
        typ = d.get("type")
        total += amt
        if typ in parts:
            parts[typ] += amt
    open_tr = [v["doc"] for v in state["trades"].values() if v["doc"].get("status") == "open"]
    cost = sum(float(t.get("entry_price") or 0) * float(t.get("shares") or 0) for t in open_tr)
    return {"capital": round(total, 2), "open_cost": round(cost, 2),
            "cash": round(total - cost, 2), "open_count": len(open_tr),
            "parts": {k: round(v, 2) for k, v in parts.items()}}


def size_position(entry: float, stop: float, capital: float, cash: float, P: dict,
                  score: float, settings: dict) -> dict:
    risk = P["risk"]
    rp = float(settings.get("risk_pct", risk["risk_pct"]))
    if score >= 90:
        rp = max(rp, float(risk.get("risk_pct_top", rp)))
    max_pct = float(settings.get("max_position_pct", risk["max_position_pct"]))
    per_share = max(entry - stop, 0.01)
    risk_usd = capital * rp / 100
    cap_usd = capital * max_pct / 100
    budget = min(cap_usd, cash) if cash > 0 else 0.0
    by_risk = risk_usd / per_share
    note = None
    if budget <= 0:
        # nothing free right now: show the size it would be once capital frees up
        budget_nominal = cap_usd
        shares = math.floor(min(by_risk, budget_nominal / entry))
        note = "No free cash at the moment: size shown assumes a slot frees up."
    else:
        shares = math.floor(min(by_risk, budget / entry))
    fractional = False
    if shares < 1:
        alloc = min(by_risk * entry, (budget if budget > 0 else cap_usd))
        if alloc >= entry * 0.9 or (per_share <= 1.6 * risk_usd and entry <= (budget if budget > 0 else cap_usd)):
            shares = 1  # one whole share keeps it a limit order; risk stays within 1.6x the target
        else:
            shares = round(alloc / entry, 3)
            fractional = True
            note = ("Share price is above the position budget: fractional shares are needed, "
                    "which your broker only fills during market hours (no limit orders). "
                    "Place it as a market-hours order or skip.")
    total = round(shares * entry, 2)
    return {"shares": shares, "fractional": fractional, "total_cost": total,
            "risk_usd": round(shares * per_share, 2),
            "pct_of_capital": round(100 * total / capital, 1) if capital > 0 else None,
            "risk_pct_used": rp, "note": note}
