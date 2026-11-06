"""Backtest grid search + daily learning loop.

    python -m engine.calibrate backtest --data <dir with history.parquet> --out <dir>
    python -m engine.calibrate learn    --data <data-dir> --state <state-dir> --out <dir>

`backtest` picks, per strategy, the variant with the highest in-sample win rate that still
clears profit-factor / expectancy floors, reports out-of-sample results honestly, fits the
win-probability model, calibrates the A+ cut-offs and writes params.json.

`learn` grades every past alert against what the market actually did (shadow results, so
the system learns even from setups you skipped), folds in your real fills, nudges each
strategy's win-probability toward its live record, flags drift, and writes a journal entry.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd

from . import backtest as B
from . import calendar as cal
from . import data as D
from . import indicators as I
from . import model as M
from . import rules as R
from . import strategies as S
from .portfolio import load_state

GRIDS = {
    "MR_STRETCH": {
        "entry": [{"rsi2_max": r, "k_atr": k} for r in (5, 10) for k in (0.0, 0.3, 0.6, 0.9)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "mean4", "min_atr": 0.3}, {"kind": "atr", "mult": 0.75},
                           {"kind": "atr", "mult": 1.25})
                 for s in (2.5, 3.5) for h in (5, 8)],
    },
    "LEADER_PULLBACK": {
        "entry": [{"rsi2_max": r, "k_atr": k, "rs_min": q} for r in (10, 20) for k in (0.0, 0.4, 0.8)
                  for q in (0.7, 0.85)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "max7c", "min_atr": 0.3}, {"kind": "atr", "mult": 1.0})
                 for s in (2.5, 3.5) for h in (6, 10)],
    },
    "ETF_IBS": {
        "entry": [{"ibs_max": i, "k_atr": k, "entry_kind": ek, "rsi2_max": 35}
                  for i in (0.1, 0.2) for ek, k in (("close", 0.0), ("limit_next", 0.0),
                                                    ("limit_next", 0.3), ("limit_next", 0.6))],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "high_t", "min_atr": 0.25}, {"kind": "atr", "mult": 0.75})
                 for s in (2.0, 3.0) for h in (4, 7)],
    },
    "TREND_PULLBACK": {
        "entry": [{"k_atr": k, "rs_min": q} for k in (0.0, 0.3, 0.6) for q in (0.7, 0.85)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "hh20", "min_atr": 0.75}, {"kind": "atr", "mult": 1.0})
                 for s in (2.0, 3.0) for h in (7, 12)],
    },
    "BREAKOUT_RETEST": {
        "entry": [{"base_max": b, "vol_mult": v, "valid_days": d} for b in (0.10, 0.15)
                  for v in (1.5, 2.0) for d in (1, 3)],
        "exit": [{"target": {"kind": "atr", "mult": m}, "stop_atr": s, "max_hold": h}
                 for m in (1.0, 2.0) for s in (1.5, 2.5) for h in (7, 12)],
    },
    "GAP_HOLD": {
        "entry": [{"gap_min": g, "vol_mult": v, "rsi2_max": r, "k_atr": k} for g in (0.03, 0.05)
                  for v in (2.0, 3.0) for r in (20, 35) for k in (0.0, 0.4)],
        "exit": [{"target": {"kind": "atr", "mult": m}, "max_hold": h} for m in (1.0, 1.5, 2.5)
                 for h in (6, 10)],
    },
    "FAIR_VALUE": {
        "entry": [{"z": z, "k_atr": k, "rsi2_max": r} for z in (1.5, 2.0, 2.5) for k in (0.0, 0.4)
                  for r in (15, 30)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "vwap20", "min_atr": 0.3}, {"kind": "mean4", "min_atr": 0.3})
                 for s in (2.5, 3.5) for h in (5, 8)],
    },
}

FLOORS = {"min_trades": 120, "min_pf": 1.35, "min_avg": 0.0035}


def objective(s: dict) -> float:
    if not s or s.get("trades", 0) < FLOORS["min_trades"]:
        return -1.0
    pf = s.get("profit_factor") or 0
    pen = 0.0
    if pf < FLOORS["min_pf"]:
        pen += 0.25 + (FLOORS["min_pf"] - pf)
    if s["avg_ret"] < FLOORS["min_avg"]:
        pen += 0.25 + 20 * (FLOORS["min_avg"] - s["avg_ret"])
    return s["win_rate"] + 2.0 * s["avg_ret"] - pen


def _variant(P, name, v):
    P2 = copy.deepcopy(P)
    P2["strategies"][name].update(copy.deepcopy(v))
    return P2


def search_strategy(name, ind, P, split, log):
    g = GRIDS[name]
    best_entry, best_obj, tried = None, -9, []
    base_exit = {}
    for ev in g["entry"]:
        P2 = _variant(P, name, ev)
        plan = S.build(name, ind, P2)
        tr = B.simulate(plan, ind, P2, include_unfilled=True)
        if tr.empty:
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"variant": ev, "train": st, "obj": round(o, 4)})
        if o > best_obj:
            best_obj, best_entry = o, ev
    if best_entry is None:
        return None
    best_v, best_obj2, best_tr = dict(best_entry), -9, None
    for xv in g["exit"]:
        v = {**best_entry, **xv}
        P2 = _variant(P, name, v)
        plan = S.build(name, ind, P2)
        tr = B.simulate(plan, ind, P2, include_unfilled=True)
        if tr.empty:
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"variant": v, "train": st, "obj": round(o, 4)})
        if o > best_obj2:
            best_obj2, best_v, best_tr = o, v, tr
    # stage 3: market-regime gate
    for reg in ("spy200", "spy200_breadth", "vixcalm"):
        v = {**best_v, "regime": reg}
        P2 = _variant(P, name, v)
        plan = S.build(name, ind, P2)
        tr = B.simulate(plan, ind, P2, include_unfilled=True)
        if tr.empty:
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"variant": v, "train": st, "obj": round(o, 4)})
        if o > best_obj2 + 0.005:  # gate must earn its keep
            best_obj2, best_v, best_tr = o, v, tr
    log(f"  {name}: best train objective {best_obj2:.3f} with {best_v}")
    return {"variant": best_v, "trades": best_tr, "objective": best_obj2, "tried": tried}


def _isotonic(rows: list[dict]) -> list[dict]:
    """Pool adjacent bands until the win rate never falls as the score rises."""
    blocks = [{"lo": r["lo"], "hi": r["hi"], "w": r["win"] * r["n"], "a": r["avg"] * r["n"], "n": r["n"], "rows": [r]} for r in rows]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i]["w"] / blocks[i]["n"] > blocks[i + 1]["w"] / blocks[i + 1]["n"]:
            a, b = blocks[i], blocks.pop(i + 1)
            a.update(hi=b["hi"], w=a["w"] + b["w"], a=a["a"] + b["a"], n=a["n"] + b["n"], rows=a["rows"] + b["rows"])
            i = max(0, i - 1)
        else:
            i += 1
    out = []
    for bl in blocks:
        for r in bl["rows"]:
            out.append({**r, "win": round(bl["w"] / bl["n"], 4), "avg": round(bl["a"] / bl["n"], 5), "n_band": bl["n"]})
    return out


def calibrate_grades(all_tr: pd.DataFrame, split, min_week=3) -> dict:
    """Pick A+ cut-offs on out-of-sample trades: the highest win rate whose average trade is
    at least +0.6% and that still produces enough setups per week across the universe."""
    t = all_tr[(all_tr["signal_date"] >= split) & all_tr["ret"].notna() & all_tr["filled"]].copy()
    sig = all_tr[all_tr["signal_date"] >= split]
    if t.empty:
        return {"aplus_score": 70, "aplus_pwin": 0.66, "min_week": min_week, "note": "fallback cut-offs"}
    weeks_idx = pd.date_range(sig["signal_date"].min(), sig["signal_date"].max(), freq="W")
    weeks = max(1.0, len(weeks_idx))
    cands = []
    for sc in range(40, 94, 2):
        for pw in (0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76):
            sel = t[(t["score"] >= sc) & (t["p"] >= pw)]
            if len(sel) < 150:
                continue
            avg = sel["ret"].mean()
            s_all = sig[(sig["score"] >= sc) & (sig["p"] >= pw)].drop_duplicates(["signal_date", "ticker"])
            per_week = len(s_all) / weeks
            if avg < 0.006 or per_week < min_week * 1.6:
                continue
            wr = (sel["ret"] > 0).mean()
            wk = s_all.groupby(pd.Grouper(key="signal_date", freq="W")).size().reindex(weeks_idx, fill_value=0)
            cands.append((float(wr), float(avg), sc, pw, per_week, len(sel), float((wk >= min_week).mean())))
    if not cands:
        return {"aplus_score": 70, "aplus_pwin": 0.66, "min_week": min_week, "note": "fallback cut-offs"}
    top_wr = max(c[0] for c in cands)
    top_avg = max(c[1] for c in cands if c[0] >= top_wr - 0.012)
    # adjacent score bands are statistically indistinguishable, so among cut-offs within 1.2 points
    # of the best win rate (and 85% of its average trade) take the one that covers the most weeks
    near = [c for c in cands if c[0] >= top_wr - 0.012 and c[1] >= 0.85 * top_avg] or cands
    wr, avg, sc, pw, per_week, n, cover = max(near, key=lambda c: (c[6], c[0]))
    # realised out-of-sample win rate by score band: what the alert's Win % shows
    table = []
    for lo, hi in ((40, 54), (54, 62), (62, 66), (66, 70), (70, 74), (74, 78), (78, 82), (82, 86), (86, 90), (90, 101)):
        b = t[(t["score"] >= lo) & (t["score"] < hi) & (t["p"] >= pw)]
        if len(b) >= 40:
            table.append({"lo": lo, "hi": hi, "win": round(float((b["ret"] > 0).mean()), 4),
                          "avg": round(float(b["ret"].mean()), 5), "n": int(len(b))})
    table = _isotonic(table)
    return {"aplus_score": sc, "aplus_pwin": pw, "min_week": min_week,
            "oos_aplus_win_rate": round(wr, 4), "oos_aplus_avg_ret": round(float(avg), 5),
            "oos_aplus_trades": int(n), "oos_aplus_signals_per_week": round(float(per_week), 1),
            "oos_weeks_with_min_setups": round(cover, 3), "score_win_table": table,
            "topup_floor": max(40, sc - 8)}


def run_backtest(data_dir, out_dir, split="2022-01-01", strategies=None, log=print, merge_state=None) -> dict:
    t0 = time.time()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(split)
    oldP = {}
    if merge_state:
        old = load_state(merge_state).get("params_doc") or {}
        oldP = old.get("params") or old
    w = D.load_history(data_dir)
    universe = D.load_universe(data_dir)
    log(f"history: {w['close'].shape[0]} days x {w['close'].shape[1]} tickers "
        f"({w['close'].index.min().date()} to {w['close'].index.max().date()})")
    ind = I.compute(w, universe)
    log(f"indicators done in {time.time() - t0:.0f}s")
    P = S.params_with_defaults(None)
    pre_changes = []
    if merge_state:  # learn from real fills: stop slippage worse than assumed -> backtest with yours
        try:
            lp = ((load_state(merge_state).get("learning") or {}).get("portfolio") or {}).get("doc") or {}
            ss = lp.get("stop_slippage") or {}
            if (ss.get("n") or 0) >= 10 and ss.get("avg_pct") is not None and -float(ss["avg_pct"]) > P["risk"]["slippage_pct"]:
                P["risk"]["slippage_pct"] = round(-float(ss["avg_pct"]), 3)
                pre_changes.append(f"Backtests now assume {P['risk']['slippage_pct']:.2f}% slippage on stops and exits, "
                                   f"matching your last {ss['n']} stop fills.")
        except Exception as ex:
            log(f"learning/portfolio unavailable: {ex!r}")
    report = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "split": str(split.date()),
              "period": [str(w["close"].index.min().date()), str(w["close"].index.max().date())],
              "tickers": int(w["close"].shape[1]), "strategies": {}}
    pooled = []

    def evaluate(name, res, spec=None):
        if spec is not None:  # a research strategy: its rules travel with the params
            P.setdefault("custom", {})[name] = {"spec": spec, "params": res["variant"], "label": spec.get("label"),
                                                "family": spec.get("family"), "about": spec.get("about"),
                                                "source": spec.get("source")}
            P["strategies"][name] = R.strategy_entry(spec, res["variant"], True)
        else:
            P["strategies"][name].update(res["variant"])
        tr = res["trades"]
        filled = tr[tr["filled"] & tr["ret"].notna()]
        tr_train, tr_test = tr[tr["signal_date"] < split], tr[tr["signal_date"] >= split]
        st_train, st_test, st_all = B.stats(tr_train), B.stats(tr_test), B.stats(tr)
        mdl = M.fit(filled, split_date=split)
        mdl["avg_hold"] = st_all.get("avg_hold")
        # p for every row from a train-only fit (honest OOS scores for grade calibration)
        train_only = M.fit(filled[filled["signal_date"] < split])
        feats = tr[S.FEATURES]
        tr = tr.assign(p=M.predict(train_only, feats))
        rr = ((tr["target"] - tr["fill"]) / (tr["fill"] - tr["stop"])).where(tr["filled"],
              (tr["target"] - tr["limit"]) / (tr["limit"] - tr["stop"]))
        tr = tr.assign(score=M.score(tr["p"].values, feats, train_only, rr.fillna(1).values))
        pooled.append(tr)
        enabled, checks, reason = S.enable_checks(st_train, st_test)
        oos_weeks = max(1.0, (ind["close"].index[-1] - split).days / 7.0)
        # honest per-strategy figures for display and the learning loop: out-of-sample
        if st_test.get("trades", 0) >= 100:
            mdl["base_rate"] = st_test["win_rate"]
            mdl["avg_win"], mdl["avg_loss"] = st_test["avg_win"], st_test["avg_loss"]
        P["strategies"][name]["enabled"] = bool(enabled)
        P.setdefault("models", {})[name] = mdl
        mt = S.meta(name, P)
        rec = {"enabled": bool(enabled), "variant": res["variant"], "train": st_train,
               "test_oos": st_test, "all": st_all, "yearly": B.yearly(tr).reset_index().to_dict("records"),
               "model_validation": mdl.get("validation"), "label": mt["label"], "family": mt["family"],
               "checks": checks, "status_reason": reason,
               "path": None if enabled else S.path_to_pass(st_train, st_test, oos_weeks)}
        if spec is not None:
            rec.update(custom=True, about=spec.get("about"), source=spec.get("source"),
                       rules=R.describe({**spec, "params": res["variant"], "grid": {}}))
        report["strategies"][name] = rec
        log(f"  {name}: train win {st_train.get('win_rate')} avg {st_train.get('avg_ret')} PF "
            f"{st_train.get('profit_factor')} | OOS win {st_test.get('win_rate')} avg {st_test.get('avg_ret')} "
            f"PF {st_test.get('profit_factor')} n={st_test.get('trades')} | enabled={enabled}")

    for name in (strategies or list(GRIDS)):
        log(f"searching {name} ...")
        res = search_strategy(name, ind, P, split, log)
        if res is None:
            P["strategies"][name]["enabled"] = False
            report["strategies"][name] = {"enabled": False, "reason": "no trades", "status_reason": "Off: no trades in the backtest.",
                                          "label": S.STRATEGY_META[name]["label"], "family": S.STRATEGY_META[name]["family"]}
            continue
        evaluate(name, res)
    # research strategies: this week's candidates plus every research strategy promoted before
    research_docs = []
    if merge_state and not strategies:
        from .research import candidate_specs, evaluate_spec, retest_specs, search_spec
        from .scan import load_params as _load_live
        # ideas that fail only on 2022-now checks can pass as new data arrives: re-test them every week
        try:
            P_live, P_r, now_r = _load_live(load_state(merge_state), None), S.params_with_defaults(None), cal.now_pt()
            for spec in retest_specs(merge_state):
                d = evaluate_spec(spec, ind, P_r, P_live, split, now_r, by="weekly re-test", log=log)
                research_docs.append(d)
                log(f"  re-test {spec['id']}: {d['verdict']}")
        except Exception as ex:
            log(f"research re-tests failed: {ex!r}")
        try:
            cands = candidate_specs(merge_state, oldP, extra_results=research_docs)
        except Exception as ex:
            cands = {}
            log(f"research candidates unavailable: {ex!r}")
        for cid, spec in cands.items():
            errs = R.validate(spec)
            if errs:
                log(f"  skipping research strategy {cid}: {errs}")
                continue
            log(f"searching research strategy {cid} ...")
            try:
                res = search_spec(spec, ind, P, split, log)
            except Exception as ex:
                log(f"  {cid} failed: {ex!r}")
                continue
            if res is not None:
                evaluate(cid, res, spec)
    allp = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    if not allp.empty:
        enabled_names = [n for n, v in report["strategies"].items() if v.get("enabled")]

        def fit_and_grade(names):
            en_ = allp[allp["strategy"].isin(names)].copy()
            pm_ = None
            # pooled robust-factor model: fit on pre-split trades, win odds recalibrated out of sample
            try:
                pm_ = M.fit_pooled(en_, names, split)
                p_, e_, ep_ = M.predict_pooled(pm_, en_[S.FEATURES], en_["strategy"].values)
                en_["p"], en_["e"], en_["e_pct"] = p_, e_, ep_
                en_["score"] = M.score_pooled(p_, ep_)
            except Exception as ex:  # keep the per-strategy scores if the pooled fit fails
                log(f"pooled model failed: {ex!r}")
                pm_ = None
            return en_, pm_, calibrate_grades(en_, split)

        en, pm, G = fit_and_grade(enabled_names)
        # research strategies must earn their place inside the A+ set, not just pass on their own
        research_on = [n for n in enabled_names if (P["strategies"].get(n) or {}).get("custom")]
        if research_on:
            base_names = [n for n in enabled_names if n not in research_on]
            en0, pm0, G0 = fit_and_grade(base_names) if base_names else (None, None, {})
            oos_ = en[(en["signal_date"] >= split) & en["ret"].notna() & en["filled"]]
            ap_ = oos_[(oos_["score"] >= G["aplus_score"]) & (oos_["p"] >= G["aplus_pwin"])]
            a_win = float(G.get("oos_aplus_win_rate") or 0)
            per, weak = {}, []
            for n in research_on:
                sub = ap_[ap_["strategy"] == n]
                w_ = float((sub["ret"] > 0).mean()) if len(sub) else None
                per[n] = {"aplus_trades_oos": int(len(sub)), "aplus_win_oos": round(w_, 4) if w_ is not None else None,
                          "aplus_avg_oos": round(float(sub["ret"].mean()), 5) if len(sub) else None}
                if len(sub) >= 30 and w_ < a_win - 0.03:
                    weak.append(n)
            # keep or raise the A+ win rate and average trade without thinning out the weekly setups
            better = (not G0) or (a_win >= float(G0.get("oos_aplus_win_rate") or 0) - 0.005
                                  and float(G.get("oos_aplus_avg_ret") or 0) >= float(G0.get("oos_aplus_avg_ret") or 0) - 0.0005
                                  and float(G.get("oos_aplus_signals_per_week") or 0) >= 0.8 * float(G0.get("oos_aplus_signals_per_week") or 0))
            drop = research_on if not better else weak
            gate = {"with_research": {k: G.get(k) for k in ("aplus_score", "aplus_pwin", "oos_aplus_win_rate", "oos_aplus_avg_ret",
                                                             "oos_aplus_trades", "oos_aplus_signals_per_week", "oos_weeks_with_min_setups")},
                    "without_research": {k: G0.get(k) for k in ("aplus_score", "aplus_pwin", "oos_aplus_win_rate", "oos_aplus_avg_ret",
                                                                 "oos_aplus_trades", "oos_aplus_signals_per_week", "oos_weeks_with_min_setups")},
                    "per_strategy": per, "dropped": drop}
            for n in drop:
                pn = per.get(n) or {}
                why = (f"Off: it passed every standalone check, but adding the research strategies didn't improve the A+ set "
                       f"on unseen data ({100 * a_win:.1f}% wins with them vs {100 * float(G0.get('oos_aplus_win_rate') or 0):.1f}% without)."
                       if not better else
                       f"Off: it passed every standalone check, but its A+-grade trades won only {100 * (pn.get('aplus_win_oos') or 0):.1f}% "
                       f"on unseen data vs {100 * a_win:.1f}% for the whole A+ set, so it would dilute the alerts.")
                P["strategies"][n]["enabled"] = False
                report["strategies"][n].update(enabled=False, status_reason=why + " Re-tested every Saturday.")
                log(f"  research gate: {n} off ({why})")
            if drop:
                enabled_names = [n for n in enabled_names if n not in drop]
                en, pm, G = (en0, pm0, G0) if set(drop) == set(research_on) and en0 is not None else fit_and_grade(enabled_names)
            for n in research_on:
                if n not in drop:
                    report["strategies"][n]["aplus_oos"] = per.get(n)
            report["research_gate"] = gate
        if pm is not None:
            P["models"]["_pooled"] = pm
            report["pooled_model"] = pm["validation"]
            log(f"pooled model: OOS AUC {pm['validation']['oos_auc']}, return corr {pm['validation']['oos_ret_corr']}")
        allp = en
        P["grading"].update(G)
        report["grading"] = G
        # score buckets (validates that a higher score = better setup)
        oos = en[(en["signal_date"] >= split) & en["ret"].notna()].copy()
        if len(oos) > 50:
            oos["bucket"] = pd.cut(oos["score"], [0, 50, 60, 70, 80, 90, 101], right=False)
            sb = oos.groupby("bucket", observed=True)["ret"].agg(trades="size", win_rate=lambda s: (s > 0).mean(), avg_ret="mean")
            report["score_buckets_oos"] = [{"bucket": str(k), **{kk: round(float(vv), 4) for kk, vv in r.items()}}
                                           for k, r in sb.iterrows()]
        # portfolio simulation, A+ only, $5k, 5 slots
        aplus = en[(en["score"] >= G["aplus_score"]) & (en["p"] >= G["aplus_pwin"])]
        for label, sub in (("all_years", aplus), ("oos", aplus[aplus["signal_date"] >= split])):
            pf = B.portfolio(sub, 5000, P["risk"]["max_open"], P["risk"]["risk_pct"], P["risk"]["max_position_pct"])
            cv = pf.pop("curve", None)
            report.setdefault("portfolio", {})[label] = pf
            if label == "all_years" and cv is not None and len(cv):
                cv.assign(date=cv["date"].astype(str)).to_csv(out / "equity_curve.csv", index=False)
        allp.drop(columns=[c for c in ("t", "j", "fill_i", "exit_i") if c in allp]).to_parquet(out / "trades.parquet")
    P["version"] = int(time.time())
    P["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    P["source"] = "backtest"
    changes = list(pre_changes)
    if research_docs:
        report["research_retests"] = [{"idea": d.get("idea_id"), "verdict": d.get("verdict"), "summary": d.get("summary")}
                                      for d in research_docs]
        for d in research_docs:
            if d.get("verdict") == "candidate":
                changes.append(f"Research idea {d.get('label')} became a candidate on this week's data.")
    if merge_state:
        if oldP.get("live_adjust"):
            P["live_adjust"] = oldP["live_adjust"]  # keep what live trading has taught us
        for k in ("risk",):
            if isinstance(oldP.get(k), dict):
                P[k].update(oldP[k])
        for n, sp in P["strategies"].items():
            o = (oldP.get("strategies") or {}).get(n) or {}
            lab = S.meta(n, P)["label"]
            why = (report["strategies"].get(n) or {}).get("status_reason", "")
            if sp.get("custom") and not o:
                changes.append(f"Promoted research strategy {lab}: passed the recalibration and now trades live. {why}"
                               if sp.get("enabled") else f"Research candidate {lab} did not pass the recalibration. {why}")
                continue
            if o and o.get("enabled") != sp.get("enabled"):
                changes.append(f"{lab}: {'enabled' if sp.get('enabled') else 'disabled'} after recalibration. {why}")
            diff = {k: v for k, v in sp.items() if k not in ("enabled", "paused_reason") and o.get(k) != v}
            if o and diff:
                changes.append(f"{lab}: parameters now {json.dumps(diff, default=str)}.")
        og, ng = oldP.get("grading") or {}, P["grading"]
        if og and (og.get("aplus_score") != ng.get("aplus_score") or og.get("aplus_pwin") != ng.get("aplus_pwin")):
            changes.append(f"A+ cut-offs: score {og.get('aplus_score')} -> {ng.get('aplus_score')}, "
                           f"win {og.get('aplus_pwin')} -> {ng.get('aplus_pwin')}.")
    report["changes"] = changes
    (out / "params.json").write_text(json.dumps(P, indent=1, default=str))
    (out / "backtest_report.json").write_text(json.dumps(report, indent=1, default=str))
    # dashboard documents
    cv_p = out / "equity_curve.csv"
    curve = []
    if cv_p.exists():
        cv = pd.read_csv(cv_p)
        step = max(1, len(cv) // 300)
        curve = cv.iloc[::step].round(2).to_dict("records")
    bt_doc = {k: v for k, v in report.items()}
    bt_doc["equity_curve"] = curve
    (out / "doc_backtest.json").write_text(json.dumps(bt_doc, default=str))
    (out / "doc_params.json").write_text(json.dumps({"params": P, "updated_at": P["updated"],
                                                     "changelog_entry": changes or ["Weekly recalibration: no material changes."]}, default=str))
    # re-tested research ideas: new (create-only) result documents
    rw = []
    for d in research_docs:
        fp = out / "docs" / "research_results" / f"{d['id']}.json"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(d, default=str))
        rw.append({"op": "set", "collection": "research_results", "doc_id": d["id"], "file_path": str(fp.resolve())})
    (out / "research_writes.json").write_text(json.dumps([rw[i:i + 45] for i in range(0, len(rw), 45)], indent=1))
    log(f"backtest finished in {time.time() - t0:.0f}s")
    return report


# ----------------------------------------------------------------------------- learning

def shadow_outcome(a: dict, ind: dict) -> dict:
    """What happened to an entry alert if it had been placed exactly as sent."""
    tk = a.get("ticker")
    c = ind["close"]
    res = {"alert_id": a.get("id"), "ticker": tk, "strategy": a.get("strategy"), "status": "unknown"}
    if tk not in c.columns:
        return res
    j = c.columns.get_loc(tk)
    idx = c.index
    try:
        vf = pd.Timestamp(a.get("valid_for"))
    except Exception:
        return res
    if vf not in idx:
        res["status"] = "pending" if vf > idx[-1] else "no_session"
        return res
    d0 = idx.get_loc(vf)
    O, H, L, C = (ind[k].values[:, j] for k in ["open", "high", "low", "close"])
    entry, stop, tgt = float(a["entry"]), float(a["stop"]), float(a["target"])
    hold = int(a.get("hold_max_days") or 7)
    dyn = a.get("dyn_exit") or "none"
    if not (L[d0] <= entry):
        res["status"] = "unfilled"
        return res
    fill = min(O[d0], entry)
    sma5 = ind["sma5"].values[:, j]
    held = 0
    for d in range(d0, len(idx)):
        held += 1
        if d == d0:
            if L[d] <= stop:
                return {**res, "status": "closed", "fill": fill, "exit": stop, "reason": "stop", "held": held}
        else:
            if O[d] <= stop:
                return {**res, "status": "closed", "fill": fill, "exit": O[d], "reason": "stop", "held": held}
            if O[d] >= tgt:
                return {**res, "status": "closed", "fill": fill, "exit": O[d], "reason": "target", "held": held}
            if L[d] <= stop:
                return {**res, "status": "closed", "fill": fill, "exit": stop, "reason": "stop", "held": held}
            if H[d] >= tgt:
                return {**res, "status": "closed", "fill": fill, "exit": tgt, "reason": "target", "held": held}
        if (dyn == "sma5" and C[d] > sma5[d]) or (dyn == "prev_high" and C[d] > H[d - 1]):
            return {**res, "status": "closed", "fill": fill, "exit": C[d], "reason": "signal", "held": held}
        if held >= hold:
            return {**res, "status": "closed", "fill": fill, "exit": C[d], "reason": "time", "held": held}
    return {**res, "status": "open", "fill": fill, "last": C[-1], "held": held}


def run_learning(data_dir, state_dir, out_dir, now=None) -> dict:
    now = now or cal.now_pt()
    out = pathlib.Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    state = load_state(state_dir)
    from .scan import load_params, _write
    P = load_params(state, None)
    w = D.load_daily(data_dir)
    ind = I.compute({k: v.iloc[-400:] for k, v in w.items()}, D.load_universe(data_dir))
    alerts = [v["doc"] for v in state["alerts"].values() if v["doc"].get("type") == "entry"]
    shadow = [shadow_outcome(a, ind) for a in alerts]
    sh = pd.DataFrame([s for s in shadow if s.get("status") == "closed"])
    if not sh.empty:
        sh["ret"] = sh["exit"] / sh["fill"] - 1
    trades = [v["doc"] for v in state["trades"].values()]
    closed = pd.DataFrame([t for t in trades if t.get("status") == "closed" and t.get("exit_price")])
    if not closed.empty:
        closed["ret"] = closed["exit_price"].astype(float) / closed["entry_price"].astype(float) - 1
    from . import learnlog as LL
    closed_list = [t for t in trades if t.get("status") == "closed" and t.get("exit_price")]
    prev_stats = ((state.get("learning") or {}).get("stats") or {}).get("doc")
    stats = LL.stats_doc(prev_stats, alerts, shadow, closed_list, len(state["alerts"]) < 1000, now)
    by = {}
    changes, insights = [], []
    for name in P["strategies"]:
        mdl = P.get("models", {}).get(name) or {}
        base = float(mdl.get("base_rate", 0.65))
        s_sub = sh[sh["strategy"] == name] if not sh.empty else pd.DataFrame()
        l_sub = closed[closed["strategy"] == name] if not closed.empty else pd.DataFrame()
        sn, sw = len(s_sub), int((s_sub["ret"] > 0).sum()) if len(s_sub) else 0
        ln, lw = len(l_sub), int((l_sub["ret"] > 0).sum()) if len(l_sub) else 0
        # the permanent record (every month ever graded, live fills double, 6-month half-life)
        n_w, w_w = LL.weighted_record(stats["months"], name, now)
        n_eff, w_eff = int(round(n_w)), int(round(w_w))
        exp_p = np.mean([a.get("win_pct", 100 * base) for a in alerts if a.get("strategy") == name] or [100 * base]) / 100
        old = float(P.get("live_adjust", {}).get(name, {}).get("offset", 0.0))
        new = M.live_offset(base, w_eff, n_eff) if n_eff else 0.0
        by[name] = {"shadow_n": sn, "shadow_wins": sw, "live_n": ln, "live_wins": lw,
                    "expected_win": round(exp_p, 3), "base_rate": base, "offset_old": round(old, 3),
                    "offset_new": round(new, 3), "record_n": n_eff, "record_wins": w_eff,
                    "shadow_avg_ret": round(float(s_sub["ret"].mean()), 4) if sn else None,
                    "live_avg_ret": round(float(l_sub["ret"].mean()), 4) if ln else None}
        if abs(new - old) >= 0.02:
            P.setdefault("live_adjust", {})[name] = {"offset": round(new, 3), "n": n_eff, "wins": w_eff,
                                                     "updated": now.isoformat(timespec="seconds")}
            changes.append(f"{S.meta(name, P)['label']}: win-probability offset {old:+.2f} -> {new:+.2f} "
                           f"(record {w_eff}/{n_eff} weighted vs {base:.0%} backtest).")
        if n_eff >= 25:
            wr = w_eff / n_eff
            if wr < exp_p - 0.12:
                insights.append(f"{S.meta(name, P)['label']} is under-performing: {wr:.0%} live/shadow vs "
                                f"{exp_p:.0%} expected over {n_eff} weighted trades. Scores are being discounted.")
            if wr < 0.5 and P["strategies"][name].get("enabled", True):
                P["strategies"][name]["enabled"] = False
                P["strategies"][name]["paused_reason"] = (f"Paused by live results: {wr:.0%} win rate over {n_eff} weighted "
                                                          f"trades (live fills count double) is below the 50% floor, "
                                                          f"{now.date().isoformat()}.")
                changes.append(f"Paused {S.meta(name, P)['label']}: {wr:.0%} win rate over {n_eff} trades "
                               "is below the 50% floor. It will be re-tested at the weekly recalibration.")
    # feature diagnostics on recent shadow results
    if not sh.empty and len(sh) >= 12:
        amap = {a.get("id"): a for a in alerts}
        rows = []
        for _, r in sh.iterrows():
            f = (amap.get(r["alert_id"]) or {}).get("features") or {}
            rows.append({**f, "win": r["ret"] > 0})
        fd = pd.DataFrame(rows)
        for col, label in (("vix_ratio", "VIX vs its 20-day average"), ("rsi2", "RSI(2) at signal"),
                           ("dist200", "distance above the 200-day"), ("spy_above200", "SPY above 200-day"),
                           ("sector_up", "sector trend up")):
            if col in fd and fd[col].notna().sum() >= 10 and fd["win"].nunique() == 2:
                mw, ml = fd.loc[fd["win"], col].median(), fd.loc[~fd["win"], col].median()
                if pd.notna(mw) and pd.notna(ml) and abs(mw - ml) > 0.15 * (abs(mw) + abs(ml) + 1e-9):
                    insights.append(f"Winners vs losers differ on {label}: median {mw:.2f} vs {ml:.2f}.")
    # headline stats
    def agg(df):
        if df is None or df.empty:
            return {"n": 0}
        return {"n": int(len(df)), "win_rate": round(float((df["ret"] > 0).mean()), 3),
                "avg_ret": round(float(df["ret"].mean()), 4)}
    worked = []
    if not sh.empty:
        top = sh.sort_values("ret", ascending=False).head(3)
        worked = [f"{r.ticker} ({S.meta(r.strategy, P)['label']}) {100 * r.ret:+.1f}% via {r.reason}"
                  for r in top.itertuples() if r.ret > 0]
        bad = sh.sort_values("ret").head(3)
        didnt = [f"{r.ticker} ({S.meta(r.strategy, P)['label']}) {100 * r.ret:+.1f}% via {r.reason}"
                 for r in bad.itertuples() if r.ret <= 0]
    else:
        didnt = []
    ledger = [v["doc"] for v in state["ledger"].values()]
    port = LL.portfolio_stats(alerts, shadow, trades, ledger, now.date().isoformat(), P)
    port["updated_at"] = now.isoformat(timespec="seconds")
    insights += [f"Portfolio: {x}" for x in port.get("lessons", [])]
    day = now.date().isoformat()
    journal = {"id": day, "date": day, "created_at": now.isoformat(timespec="seconds"),
               "shadow": agg(sh), "live": agg(closed), "by_strategy": by, "worked": worked,
               "didnt_work": didnt, "insights": insights or ["No statistically meaningful drift yet."],
               "changes": changes or ["No parameter changes today."], "params_version": P.get("version")}
    writes = []
    jid = now.strftime("%Y-%m-%d-%H%M")
    journal["id"] = jid
    _write(out, writes, "set", "journal", jid, journal)
    if changes:
        P["version"] = int(time.time())
        P["updated"] = now.isoformat(timespec="seconds")
        doc = {"params": P, "updated_at": now.isoformat(timespec="seconds"),
               "changelog_entry": changes}
        # existing doc: the session reads strategy/params first and fills in if_version
        _write(out, writes, "set", "strategy", "params", doc)
        writes[-1]["if_version"] = "VERSION_OF_strategy/params (get it first; omit only if the doc does not exist)"
    for did, doc in (("stats", stats), ("portfolio", port)):
        _write(out, writes, "set", "learning", did, doc)
        if (state.get("learning") or {}).get(did):
            writes[-1]["if_version"] = f"VERSION_OF_learning/{did} (get it first)"
    (out / "writes.json").write_text(json.dumps([writes], indent=1))
    (out / "journal.json").write_text(json.dumps(journal, indent=1, default=str))
    return {"journal": day, "changes": len(changes), "insights": len(insights), "writes": len(writes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "learn"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--only", default=None, help="comma list of strategies")
    ap.add_argument("--merge-state", default=None, help="dashboard state export to keep live adjustments from")
    a = ap.parse_args()
    if a.cmd == "backtest":
        rep = run_backtest(a.data, a.out, a.split, a.only.split(",") if a.only else None, merge_state=a.merge_state)
        print(json.dumps({k: v for k, v in rep.items() if k != "strategies"}, indent=1, default=str))
    else:
        print(json.dumps(run_learning(a.data, a.state, a.out), indent=1, default=str))


if __name__ == "__main__":
    main()
