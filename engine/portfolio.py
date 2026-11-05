"""Portfolio state (from the dashboard database export) and position sizing."""
from __future__ import annotations

import json
import math
import pathlib

DEFAULT_SETTINGS = {
    "initial_balance": 5000.0,
    "income_default_pct": 30.0,
    "max_open_positions": 5,
    "risk_pct": 1.25,
    "max_position_pct": 25.0,
}


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
        return {"settings": dict(DEFAULT_SETTINGS), "trades": {}, "ledger": {}, "alerts": {},
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
            "params_doc": params_doc,
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
        if alloc >= entry * 0.9:
            shares = 1
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
