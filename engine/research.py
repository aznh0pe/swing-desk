"""Strategy research lab: new swing ideas tested exactly like the live strategies.

    python -m engine.research next  --state S --out O              which queued idea to test next
    python -m engine.research add   --spec F --state S --out O     validate + queue a new idea
    python -m engine.research test  (--id ID[,ID] | --spec F | --queued N) --data D --state S --out O
    python -m engine.research seed  --state S --out O              queue the starter backlog

Every idea is a JSON rule spec (see engine/rules.py). `test` searches its small parameter grid on
2008-2021 data only, then reports how the chosen variant did on unseen 2022-now data, whether it
clears the live-trading bar, and how many of its signals the live strategies already take.
Verdicts: candidate (goes into Saturday's recalibration, which decides whether it trades live),
redundant, watch, rejected. Outputs are create-only dashboard documents in O/docs plus
O/writes.json (ArtifactData batch chunks) and O/result.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd

from . import backtest as B
from . import calendar as cal
from . import learnlog as LL
from . import model as M
from . import data as D
from . import indicators as I
from . import rules as R
from . import strategies as S
from .portfolio import load_collection, load_state

QUALITY = {"oos_win": 0.62, "oos_pf": 1.15, "oos_trades": 80, "max_overlap": 0.70}
SEED_FILE = pathlib.Path(__file__).resolve().parent / "research_seed.json"
SEED_FILES = sorted(pathlib.Path(__file__).resolve().parent.glob("research_seed*.json"))  # the curated backlog
STAT_KEYS = ("trades", "win_rate", "avg_ret", "profit_factor", "avg_hold", "trades_per_year", "worst_loss",
             "max_consec_losses", "exit_mix", "fill_rate", "avg_win", "avg_loss")


# ----------------------------------------------------------------------------- state
def load_ideas(state_dir) -> dict:
    return {k: v["doc"] for k, v in load_collection(state_dir, "research").items()} if state_dir else {}


def load_results(state_dir) -> dict:
    return {k: v["doc"] for k, v in load_collection(state_dir, "research_results").items()} if state_dir else {}


def latest_results(results: dict) -> dict:
    out = {}
    for r in results.values():
        iid = r.get("idea_id")
        if iid and (iid not in out or str(r.get("tested_at", "")) > str(out[iid].get("tested_at", ""))):
            out[iid] = r
    return out


def retest_specs(state_dir, cap: int = 12) -> list[dict]:
    """Ideas whose verdict can change with new data (they fail only on 2022-now checks): re-tested every Saturday."""
    ideas, latest = load_ideas(state_dir), latest_results(load_results(state_dir))
    out = []
    for iid, r in sorted(latest.items(), key=lambda kv: str(kv[1].get("tested_at", ""))):
        if iid not in ideas or not ideas[iid].get("spec") or r.get("verdict") == "candidate":
            continue
        path = r.get("path") or {}
        if path.get("status") in ("data", "data_slow", "overlap") or (not path and r.get("verdict") in ("watch", "redundant")):
            out.append(ideas[iid]["spec"])
    return out[:cap]


def candidate_specs(state_dir, old_params: dict | None = None, cap: int = 6, extra_results: list | None = None) -> dict:
    """Specs the weekly recalibration should evaluate: every strategy promoted earlier (always, never capped:
    one left out would silently drop out of the live params) plus up to `cap` new research candidates, best
    first. Nov 12, 2026: the cap used to cover both, so once six strategies had been promoted no new candidate
    was ever tested on Saturday, and a seventh promoted strategy would have been dropped without a word."""
    ideas, results = load_ideas(state_dir), load_results(state_dir)
    for r in extra_results or []:
        results[r["id"]] = r
    latest = latest_results(results)
    keep = {iid: cu["spec"] for iid, cu in ((old_params or {}).get("custom") or {}).items() if cu.get("spec")}
    pool = {}
    for iid, r in latest.items():
        if r.get("verdict") == "candidate" and iid in ideas and ideas[iid].get("spec") and iid not in keep:
            sp_ = dict(ideas[iid]["spec"])
            if r.get("graded_ok"):  # judged on its graded setups (see GRADED), in every market condition
                sp_["graded"] = True
                sp_.pop("market_conditions", None)
            elif r.get("market_conditions"):  # a specialist: re-tested restricted to its market conditions
                sp_["market_conditions"] = list(r["market_conditions"])
            win_ = ((r.get("graded") or {}).get("periods", {}).get("oos", {}).get("win_rate") if r.get("graded_ok") else None) \
                or (r.get("specialist") or {}).get("oos", {}).get("win_rate") or (r.get("oos") or {}).get("win_rate") or 0
            pool[iid] = (float(win_), sp_)
    ranked = sorted(pool.items(), key=lambda kv: -kv[1][0])[:cap]
    return {**keep, **{k: v[1] for k, v in ranked}}


def _write(out: pathlib.Path, writes: list, coll: str, did: str, doc: dict):
    p = out / "docs" / coll / f"{did}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, default=str))
    writes.append({"op": "set", "collection": coll, "doc_id": did, "file_path": str(p.resolve())})


def _finish(out: pathlib.Path, writes: list, result: dict):
    out.mkdir(parents=True, exist_ok=True)
    (out / "writes.json").write_text(json.dumps([writes[i:i + 45] for i in range(0, len(writes), 45)], indent=1))
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


def idea_doc(spec: dict, by: str, now, ideas: dict | None = None) -> dict:
    doc = {"id": spec["id"], "label": spec["label"], "family": spec["family"], "about": spec["about"],
           "source": spec.get("source"), "spec": spec, "rules": R.describe(spec), "signature": R.signature(spec),
           "created_at": now.isoformat(timespec="seconds"), "created_by": by}
    if ideas:
        sim = similar_ideas(spec, ideas)
        if sim:
            doc["variant_of"] = sim[:3]
    return doc


def check_new(spec: dict, ideas: dict) -> list[str]:
    errs = R.validate(spec)
    if errs:
        return errs
    if spec["id"] in ideas:
        return [f"{spec['id']} already exists in the research log"]
    sig = R.signature(spec)
    for iid, d in ideas.items():
        if d.get("signature") == sig or (d.get("spec") and R.signature(d["spec"]) == sig):
            return [f"same rules as {iid} ({d.get('label')})"]
    return []


def _cond_set(spec: dict) -> set:
    """Rule shapes with the numbers blanked out, so 'rsi2 < 10' and 'rsi2 < 20' count as the same rule."""
    pr = (R.combos(spec) or [{}])[0]
    return {re.sub(r"(?<![A-Za-z_])\d+(\.\d+)?", "#", re.sub(r"\s+", "", str(R.subst(c, pr)))) for c in spec.get("conditions") or []}


def similar_ideas(spec: dict, ideas: dict, min_jaccard: float = 0.6) -> list[dict]:
    """Ideas in the log whose rules mostly overlap this one (same universe, most conditions shared).
    Not a refusal: a variant is welcome, but it is recorded as one so the log shows what it builds on."""
    mine = _cond_set(spec)
    out = []
    for iid, d in ideas.items():
        sp = d.get("spec") or {}
        if not sp or iid == spec.get("id"):
            continue
        theirs = _cond_set(sp)
        if not mine or not theirs:
            continue
        j = len(mine & theirs) / len(mine | theirs)
        if j >= min_jaccard:
            out.append({"id": iid, "label": d.get("label"), "shared_rules": round(j, 2)})
    return sorted(out, key=lambda x: -x["shared_rules"])


def seed_backlog(ideas: dict) -> list[dict]:
    """Curated ideas (engine/research_seed*.json) not yet in the research log, in file order."""
    out, seen = [], set()
    for f in SEED_FILES:
        try:
            specs = json.loads(f.read_text())
        except Exception:
            continue
        for sp in specs:
            if sp.get("id") in ideas or sp.get("id") in seen or check_new(sp, ideas):
                continue
            seen.add(sp["id"])
            out.append(sp)
    return out


# ----------------------------------------------------------------------------- testing
def search_spec(spec: dict, ind: dict, P: dict, split, log=print) -> dict | None:
    """Grid search on pre-split data only (same objective as the live strategies)."""
    from .calibrate import objective
    b = S.base_masks(ind, P)
    best, tried = None, []
    for params in R.combos(spec):
        plan = R.build_plan(spec, ind, P, b, params)
        tr = B.simulate(plan, ind, P, include_unfilled=True)
        if tr.empty:
            tried.append({"params": params, "train": {"trades": 0}, "obj": -1})
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"params": params, "train": {k: st.get(k) for k in ("trades", "win_rate", "avg_ret", "profit_factor")},
                      "obj": round(o, 4)})
        if best is None or o > best["objective"]:
            best = {"variant": params, "trades": tr, "objective": o, "plan": plan}
    if best is None:
        return None
    best["tried"] = tried
    log(f"  {spec['id']}: best train objective {best['objective']:.3f} with {best['variant']}")
    return best


def grade_preview(tr: pd.DataFrame, P_live: dict, split) -> dict | None:
    """How an idea's unseen-period (2022-now) setups would grade with today's live model and bars: how many
    reach A+, how many fall in the weekly top-up band, and how each group did; plus calm vs stressed market
    days (A+ setups cluster in selloffs, so ideas that win in calm markets are what fills the weekly minimum).
    Approximate: the live model has no term of its own for a new strategy. The weekly recalibration fits it
    properly and decides."""
    pm = (P_live.get("models") or {}).get("_pooled")
    G = P_live.get("grading") or {}
    if not (pm and pm.get("kind") == "pooled") or not G.get("aplus_score"):
        return None
    o = tr[tr["signal_date"] >= pd.Timestamp(split)]
    if o.empty or any(c not in o for c in S.FEATURES):
        return None
    p_, e_, ep_ = M.predict_pooled(pm, o[S.FEATURES], ["__new__"] * len(o))
    o = o.assign(p=p_, score=M.score_pooled(p_, ep_))
    ap = (o["score"] >= float(G["aplus_score"])) & (o["p"] >= float(G["aplus_pwin"]))
    band = ~ap & (o["score"] >= float(G.get("topup_floor", float(G["aplus_score"]) - 8))) & (o["p"] >= float(G.get("topup_pwin", 0)))
    stress = pd.Series([LL.stressed({"vix_ratio": v, "spy_ret5": r}) for v, r in zip(o["vix_ratio"], o["spy_ret5"])], index=o.index)

    def grp(mask):
        f = o[mask & o["filled"] & o["ret"].notna()]
        return {"signals": int(mask.sum()), "share": round(float(mask.mean()), 4) if len(o) else 0.0, "trades": int(len(f)),
                "win_rate": round(float((f["ret"] > 0).mean()), 4) if len(f) else None,
                "avg_ret": round(float(f["ret"].mean()), 5) if len(f) else None}
    return {"aplus": grp(ap), "topup_band": grp(band), "below_bars": grp(~ap & ~band), "calm": grp(~stress),
            "stressed": grp(stress), "bars": {k: G.get(k) for k in ("aplus_score", "aplus_pwin", "topup_floor", "topup_pwin")},
            "tested_topup_win": G.get("oos_topup_win_rate"), "tested_aplus_win": G.get("oos_aplus_win_rate")}


SPECIALIST = {"train_n": 150, "oos_n": 100, "train_win": 0.62, "train_pf": 1.25, "oos_win": 0.66, "oos_pf": 1.30,
              "oos_avg": 0.0025}

# Graded candidates (Nov 12, 2026). The desk never sends an idea's raw signals: it sends only the ones that clear
# the A+ bar or the weekly top-up bar. An idea whose raw signals win often but with small gains can still be a
# strong source of setups once graded, so it is judged on exactly the setups the desk would send, graded by the
# live model it was never fit on. The bars: at least as good as the live weekly top-ups (the weakest grade the
# desk sends) with the same profit bars the top-up rule itself must clear, in both halves of 2022-now, and on
# 2008-2021 too; its raw rules must still not be broken (2008-2021 win rate, 2022-now profit factor, no decay).
GRADED = {"oos_n": 150, "min_win": 0.66, "pf": 1.25, "avg": 0.004, "half_n": 50, "half_slack": 0.03,
          "train_n": 150, "train_win": 0.64}
GRADED_RAW = ("train_win", "oos_pf", "oos_decay")


def _grp_stats(f: pd.DataFrame) -> dict:
    if not len(f):
        return {"trades": 0, "win_rate": None, "avg_ret": None, "profit_factor": None}
    w = f["ret"] > 0
    gl = -f.loc[~w, "ret"].sum()
    return {"trades": int(len(f)), "win_rate": round(float(w.mean()), 4), "avg_ret": round(float(f["ret"].mean()), 5),
            "profit_factor": round(float(f.loc[w, "ret"].sum() / gl), 3) if gl > 0 else None}


def graded_bars(G: dict) -> tuple[float, float, str]:
    """The lowest bar the desk sends at: the weekly top-up bar when top-ups are on, else the A+ bar."""
    if G.get("topup_floor") is not None and G.get("topup_ok", True) is not False and G.get("topup_mode") != "off":
        return float(G["topup_floor"]), float(G.get("topup_pwin") or 0), "topup"
    return float(G["aplus_score"]), float(G["aplus_pwin"]), "aplus"


def graded_check(tr: pd.DataFrame, P_live: dict, split, raw_checks: list, cal_end="2024-01-01") -> dict | None:
    """Judges an idea on the setups the desk would send from it: every signal scored by the live pooled model
    (which never saw this idea, so 2008-2021 is a fair test too) and kept only where it clears the lowest bar
    the desk sends at. Returns the checks, the per-period results and the results by market condition."""
    pm = (P_live.get("models") or {}).get("_pooled")
    G = P_live.get("grading") or {}
    if not (pm and pm.get("kind") == "pooled") or not G.get("aplus_score") or any(c not in tr for c in S.FEATURES):
        return None
    floor, tpw, kind = graded_bars(G)
    p_, e_, ep_ = M.predict_pooled(pm, tr[S.FEATURES], ["__new__"] * len(tr))
    t = tr.assign(p=p_, score=M.score_pooled(p_, ep_))
    sent_sig = t[(t["score"] >= floor) & (t["p"] >= tpw)]
    sent = sent_sig[sent_sig["filled"] & sent_sig["ret"].notna()]
    split, ce = pd.Timestamp(split), pd.Timestamp(cal_end)
    per = {"train": _grp_stats(sent[sent["signal_date"] < split]), "oos": _grp_stats(sent[sent["signal_date"] >= split]),
           "early": _grp_stats(sent[(sent["signal_date"] >= split) & (sent["signal_date"] < ce)]),
           "late": _grp_stats(sent[sent["signal_date"] >= ce])}
    ref = G.get("oos_topup_win_rate") if kind == "topup" else G.get("oos_aplus_win_rate")
    need = max(GRADED["min_win"], float(ref or 0))
    o, tn, e, l_ = per["oos"], per["train"], per["early"], per["late"]
    half_need = need - GRADED["half_slack"]
    checks = [
        {"key": "g_oos_n", "label": "Setups the desk would send, 2022-now", "value": o["trades"],
         "need": f">= {GRADED['oos_n']}", "ok": o["trades"] >= GRADED["oos_n"]},
        {"key": "g_oos_win", "label": "Their win rate 2022-now", "value": o["win_rate"],
         "need": f">= {100 * need:.1f}% (the live {'weekly top-ups' if kind == 'topup' else 'A+ setups'}' tested win rate)",
         "ok": (o["win_rate"] or 0) >= need},
        {"key": "g_oos_pf", "label": "Their profit factor 2022-now", "value": o["profit_factor"], "need": f">= {GRADED['pf']:.2f}",
         "ok": (o["profit_factor"] or 0) >= GRADED["pf"]},
        {"key": "g_oos_avg", "label": "Their average trade 2022-now", "value": o["avg_ret"], "need": f">= +{100 * GRADED['avg']:.2f}%",
         "ok": (o["avg_ret"] or 0) >= GRADED["avg"]},
        {"key": "g_halves", "label": "Their win rate in 2022-23 and in 2024-now", "value": [e["win_rate"], l_["win_rate"]],
         "need": f">= {100 * half_need:.1f}% in both, with a gain",
         "ok": all(x["trades"] >= GRADED["half_n"] and (x["win_rate"] or 0) >= half_need and (x["avg_ret"] or 0) > 0 for x in (e, l_))},
        {"key": "g_train", "label": "Same grading on 2008-2021", "value": tn["win_rate"],
         "need": f">= {100 * GRADED['train_win']:.0f}% of {GRADED['train_n']}+, with a gain",
         "ok": tn["trades"] >= GRADED["train_n"] and (tn["win_rate"] or 0) >= GRADED["train_win"] and (tn["avg_ret"] or 0) > 0},
    ] + [c for c in raw_checks if c["key"] in GRADED_RAW]
    o_sig = sent_sig[sent_sig["signal_date"] >= split]
    weeks = max(1.0, (tr["signal_date"].max() - split).days / 7.0)
    cond = S.condition_series(sent).values if len(sent) else []
    bc = {}
    if len(sent):
        s2 = sent.assign(condition=cond)
        for c_, g in s2[s2["signal_date"] >= split].groupby("condition"):
            bc[c_] = _grp_stats(g)
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "bars": {"score": floor, "pwin": tpw, "kind": kind},
            "need_win": round(need, 4), "periods": per, "by_condition": bc,
            "per_week": round(len(o_sig) / weeks, 2), "share_of_signals": round(len(o_sig) / max(1, int((t["signal_date"] >= split).sum())), 4)}


def graded_sentence(gc: dict, raw_oos: dict) -> str:
    o, e, l_, tn = (gc["periods"][k] for k in ("oos", "early", "late", "train"))
    b = gc["bars"]
    return (f"Candidate on the setups the desk would send: graded by the live model (score {b['score']:.0f}+, model win "
            f"{100 * b['pwin']:.0f}%+), {o['trades']} of its 2022-now setups qualify and {100 * o['win_rate']:.1f}% won "
            f"(profit factor {o['profit_factor']:.2f}, average {100 * o['avg_ret']:+.2f}%): {100 * e['win_rate']:.0f}% in 2022-23, "
            f"{100 * l_['win_rate']:.0f}% in 2024-now, and {100 * tn['win_rate']:.0f}% of {tn['trades']} on 2008-2021, which the "
            f"model never saw. Its raw signals win {100 * (raw_oos.get('win_rate') or 0):.1f}% with small gains (profit factor "
            f"{raw_oos.get('profit_factor') or 0:.2f}), and the desk never sends those: only the graded ones, about "
            f"{gc['per_week']:.1f} a week. The daily promotion check replays the desk with it and puts it live only if the "
            "A+ and top-up win rates hold and the account grows faster; Saturday's recalibration re-checks it.")


def condition_breakdown(tr: pd.DataFrame, split) -> dict:
    """The idea's results in each market condition, 2008-2021 and 2022-now."""
    split = pd.Timestamp(split)
    t = tr.assign(condition=S.condition_series(tr).values)
    f = t[t["filled"] & t["ret"].notna()]
    out = {}
    for cond, g in f.groupby("condition"):
        out[cond] = {"train": _compact(B.stats(g[g["signal_date"] < split])), "oos": _compact(B.stats(g[g["signal_date"] >= split]))}
    return out


def specialist_conditions(bc: dict) -> list[str]:
    """Market conditions where the idea clears specialist bars in both periods (enough trades, win rate,
    profit factor, and a real average trade on 2022-now)."""
    q = SPECIALIST
    ok = []
    for cond, d in bc.items():
        tr_, oo = d.get("train") or {}, d.get("oos") or {}
        if ((tr_.get("trades") or 0) >= q["train_n"] and (oo.get("trades") or 0) >= q["oos_n"]
                and (tr_.get("win_rate") or 0) >= q["train_win"] and (tr_.get("profit_factor") or 0) >= q["train_pf"]
                and (oo.get("win_rate") or 0) >= q["oos_win"] and (oo.get("profit_factor") or 0) >= q["oos_pf"]
                and (oo.get("avg_ret") or 0) >= q["oos_avg"]):
            ok.append(cond)
    return [c for c in S.CONDITIONS if c in ok]


def preview_sentence(gp: dict | None) -> str:
    if not gp:
        return ""
    a, t, c = gp["aplus"], gp["topup_band"], gp["calm"]

    def w(x):
        return f"won {100 * x['win_rate']:.0f}%" if x.get("win_rate") is not None else "none filled"
    return (f" Graded with today's model: {100 * a['share']:.0f}% of its 2022-now signals reach A+ ({w(a)}), "
            f"{100 * t['share']:.0f}% fall in the weekly top-up band ({w(t)}); on calm-market days its setups {w(c)}.")


def condition_sentence(bc: dict | None) -> str:
    if not bc:
        return ""
    bits = []
    for c in S.CONDITIONS:
        o = (bc.get(c) or {}).get("oos") or {}
        if (o.get("trades") or 0) >= 30:
            bits.append(f"{S.CONDITION_LABEL[c].lower()} {100 * o['win_rate']:.0f}% ({o['trades']})")
    return (" By market condition, 2022-now: " + ", ".join(bits) + ".") if bits else ""


def research_focus(state_dir) -> dict | None:
    """What the desk needs most from new strategies right now, from the live grades and their records."""
    try:
        st = load_state(state_dir)
    except Exception:
        return None
    pd_ = st.get("params_doc") or {}
    G = (pd_.get("params") or pd_).get("grading") or {}
    mn = int((st.get("settings") or {}).get("min_setups_week", 5))
    pc_ = (pd_.get("params") or pd_).get("conditions") or {}
    calm_share = (pc_.get("share_of_days") or {}).get("bull_calm")
    calm_hint = ("rules-based long setups that win in calm bull markets"
                 + (f" ({100 * float(calm_share):.0f}% of trading days since 2008, the condition where the live strategies are weakest and A+ setups are rare)" if calm_share else "")
                 + " and still win 66%+ there with a real average trade: for example pullbacks in strong uptrends, relative-strength "
                 "leaders resting on support, quiet-market ETF mean reversion, continuation after strong earnings. The test reports "
                 "every idea per market condition (by_condition): an all-weather idea that holds up in every condition is the "
                 "best find (it trades in any market), and an idea that only works in some conditions can still qualify as a "
                 "specialist for them (market_conditions), so an idea that wins in calm bull markets but not in selloffs still counts")
    if G.get("aplus_watch"):
        return {"grade": "aplus", "why": (G["aplus_watch"] or {}).get("note", "A+ setups trail their test."),
                "look_for": "high-win-rate ideas for market selloffs and sharp dips (where A+ setups come from) that would "
                            "raise the A+ set's win rate"}
    if G.get("topup_ok") is False or G.get("topup_mode") in ("tight", "off"):
        return {"grade": "topup", "why": "the weekly top-ups are the weak grade right now (their own record or test fell "
                                         "below the bar)", "look_for": calm_hint}
    cov, ap = G.get("oos_weeks_at_min"), G.get("oos_weeks_at_min_aplus_only")
    if cov is not None and float(cov) < 0.6:
        return {"grade": "topup", "why": (f"only {100 * float(cov):.0f}% of weeks reach the {mn}-setup minimum with setups "
                                          f"that clear the bars (A+ alone {100 * float(ap or 0):.0f}%)"), "look_for": calm_hint}
    # Never without a focus: the desk always has a gap to close. Point at the live weekly shortfall first, then at
    # the market condition where A+ setups are scarcest compared with how often it occurs.
    live = live_week_gap(st, mn)
    gap = condition_gap(pc_)
    why = []
    if live.get("this_week", 0) < live.get("pace", 0):
        why.append(f"this week has sent {live['this_week']} of the {live['pace']} setups its pace needs by tonight")
    if live.get("short_weeks"):
        why.append(f"{live['short_weeks']} of the last {live['weeks']} weeks sent fewer than {mn} setups")
    if gap:
        why.append(f"A+ setups are scarcest in {S.CONDITION_PLURAL[gap['condition']]}: {100 * gap['share_days']:.0f}% of "
                   f"2022-now trading days but {100 * gap['share_aplus']:.0f}% of A+ setups")
    look = (calm_hint if (not gap or gap["condition"] == "bull_calm") else
            f"high-win-rate setups for {S.CONDITION_PLURAL[gap['condition']]}, ideally all-weather")
    return {"grade": "aplus" if gap else "topup",
            "why": "; ".join(why) or "the weekly minimum depends on top-ups in quiet weeks",
            "look_for": look + ". Prefer a new trigger (support holding, gap reversal, quiet pullback, relative strength) "
                               "over another oversold filter: the oversold family is mostly redundant with the live book.",
            "live": live, "condition_gap": gap}


def live_week_gap(st: dict, mn: int, weeks: int = 4) -> dict:
    """How many setups each recent week actually sent (A+ setups and weekly top-ups, by signal week)."""
    import datetime as _dt
    sent = {}
    for v in (st.get("alerts") or {}).values():
        d = v.get("doc") or {}
        if d.get("type") != "entry" or not d.get("signal_date"):
            continue
        try:
            day = _dt.date.fromisoformat(str(d["signal_date"])[:10])
        except ValueError:
            continue
        wk = day - _dt.timedelta(days=day.weekday())
        sent[wk] = sent.get(wk, 0) + 1
    today = _dt.date.today()
    this = today - _dt.timedelta(days=today.weekday())
    past = [this - _dt.timedelta(weeks=i) for i in range(1, weeks + 1)]
    short = sum(1 for w in past if sent.get(w, 0) < mn and w >= min(sent or {this: 0}))
    pace = min(mn, today.weekday() + 1) if today.weekday() < 5 else mn
    return {"weeks": weeks, "short_weeks": short, "this_week": sent.get(this, 0), "pace": pace,
            "by_week": {str(w): sent.get(w, 0) for w in [this] + past}}


def condition_gap(pc: dict) -> dict | None:
    """The market condition with the biggest shortfall of A+ setups compared with how often it occurs."""
    share = pc.get("share_of_days_oos") or pc.get("share_of_days") or {}
    ap = pc.get("aplus") or {}
    tot = sum(float((ap.get(c) or {}).get("trades") or 0) for c in S.CONDITIONS)
    if not share or not tot:
        return None
    rows = [{"condition": c, "share_days": round(float(share.get(c) or 0), 3),
             "share_aplus": round(float((ap.get(c) or {}).get("trades") or 0) / tot, 3)} for c in S.CONDITIONS]
    best = max(rows, key=lambda r: r["share_days"] - r["share_aplus"])
    return best if best["share_days"] - best["share_aplus"] > 0.05 else None


def _compact(st: dict) -> dict:
    return {k: st.get(k) for k in STAT_KEYS if k in st}


def evaluate_spec(spec: dict, ind: dict, P: dict, P_live: dict, split, now, by="research", log=print) -> dict:
    t0 = time.time()
    rid = f"{spec['id']}-{now.strftime('%Y%m%d-%H%M')}"
    base = {"id": rid, "idea_id": spec["id"], "label": spec.get("label"), "family": spec.get("family"),
            "tested_at": now.isoformat(timespec="seconds"), "by": by,
            "data_period": [str(ind["close"].index.min().date()), str(ind["close"].index.max().date())],
            "split": str(pd.Timestamp(split).date())}
    errs = R.validate(spec)
    if errs:
        return {**base, "verdict": "rejected", "summary": "Rejected: the rules could not be read (" + "; ".join(errs) + ").",
                "checks": [], "runtime_s": round(time.time() - t0, 1)}
    res = search_spec(spec, ind, P, split, log)
    if res is None:
        return {**base, "verdict": "rejected", "summary": "Rejected: the rules never produced a trade on 2008-2026 data.",
                "checks": [], "runtime_s": round(time.time() - t0, 1)}
    tr = res["trades"]
    split = pd.Timestamp(split)
    st_train, st_test = B.stats(tr[tr["signal_date"] < split]), B.stats(tr[tr["signal_date"] >= split])
    ok_live, checks, live_reason = S.enable_checks(st_train, st_test)
    # how much of it the live strategies already trade (unseen period, same ticker and day)
    cand = res["plan"]["mask"]
    rows = cand.index >= split
    live_any = None
    for nm, pl in S.all_plans(ind, P_live).items():
        live_any = pl["mask"] if live_any is None else (live_any | pl["mask"])
    c_oos = cand.loc[rows].values
    n_sig = int(c_oos.sum())
    overlap = float((c_oos & live_any.loc[rows].values).sum() / n_sig) if (n_sig and live_any is not None) else 0.0
    weeks = max(1.0, (cand.index[rows].max() - cand.index[rows].min()).days / 7.0) if rows.any() else 1.0
    per_week = n_sig / weeks
    ow, opf, on = st_test.get("win_rate"), st_test.get("profit_factor"), st_test.get("trades", 0)
    q = [
        {"key": "oos_win", "label": "Win rate 2022-now (unseen data)", "value": ow, "need": f">= {QUALITY['oos_win']:.0%}",
         "ok": ow is not None and ow >= QUALITY["oos_win"]},
        {"key": "oos_pf_q", "label": "Profit factor 2022-now", "value": opf, "need": f">= {QUALITY['oos_pf']:.2f}",
         "ok": opf is not None and opf >= QUALITY["oos_pf"]},
        {"key": "oos_trades", "label": "Trades 2022-now", "value": on, "need": f">= {QUALITY['oos_trades']}",
         "ok": (on or 0) >= QUALITY["oos_trades"]},
        {"key": "overlap", "label": "Signals the live strategies already take", "value": round(overlap, 3),
         "need": f"<= {QUALITY['max_overlap']:.0%}", "ok": overlap <= QUALITY["max_overlap"]},
    ]
    q_ok = {c["key"]: c["ok"] for c in q}
    bc = condition_breakdown(tr, split)
    aw = S.all_weather(bc)
    spec_conds = list(spec.get("market_conditions") or [])
    full_pass = ok_live and all(q_ok.values())
    try:  # judged on the setups the desk would send (see GRADED)
        gc = None if full_pass else graded_check(tr, P_live, split, checks)
    except Exception as ex:  # a guide like the preview: never a reason to lose a test
        gc = {"ok": False, "error": repr(ex)[:200]}
    graded_ok = bool(gc and gc.get("ok")) and q_ok["overlap"] and not spec_conds
    special = None
    if not spec_conds and not full_pass and not graded_ok and not aw["all_weather"]:
        # not good enough everywhere: is it a specialist for some market conditions? (an all-weather idea never is:
        # it holds up everywhere, so narrowing it would only throw good trades away)
        sc_ = specialist_conditions(bc)
        if sc_ and set(sc_) >= set(aw["tested"]):
            sc_ = []  # passes in every condition it was tested in: that is all-weather, not a specialist
        if sc_:
            rows_ = pd.Series(S.condition_series(tr).values, index=tr.index).isin(sc_)
            tr_s = tr[rows_.values]
            st_tr, st_te = B.stats(tr_s[tr_s["signal_date"] < split]), B.stats(tr_s[tr_s["signal_date"] >= split])
            ok_s, checks_s, _ = S.enable_checks(st_tr, st_te)
            cmask = cand & S.condition_gate(ind, sc_, cand)
            n_s = int(cmask.loc[rows].values.sum())
            ov_s = float((cmask.loc[rows].values & live_any.loc[rows].values).sum() / n_s) if (n_s and live_any is not None) else 0.0
            if ok_s and ov_s <= QUALITY["max_overlap"] and (st_te.get("trades") or 0) >= QUALITY["oos_trades"]:
                special = {"conditions": sc_, "train": _compact(st_tr), "oos": _compact(st_te), "overlap": round(ov_s, 3),
                           "per_week": round(n_s / weeks, 2), "checks": checks_s}
    if graded_ok:
        verdict = "candidate"
        summary = graded_sentence(gc, st_test) + f" {100 * overlap:.0f}% of its signals overlap the live strategies."
    elif special:
        verdict = "candidate"
        labs = " and ".join(S.CONDITION_PLURAL[c] for c in special["conditions"])
        so = special["oos"]
        summary = (f"Candidate as a specialist for {labs} only: there {100 * so['win_rate']:.1f}% of {so['trades']} trades won on "
                   f"unseen 2022-now data (profit factor {so['profit_factor']:.2f}, average {100 * so['avg_ret']:+.2f}%, about "
                   f"{special['per_week']:.1f} signals a week), {100 * special['overlap']:.0f}% overlap with the live strategies. "
                   f"Across all conditions it won {100 * (ow or 0):.1f}%, so it would trade only in {labs}. It goes into "
                   "Saturday's recalibration restricted to those conditions.")
    elif ok_live and all(q_ok.values()):
        verdict = "candidate"
        summary = (f"Candidate: {100 * ow:.1f}% of {on} trades won on unseen 2022-now data (profit factor {opf:.2f}, "
                   f"about {per_week:.1f} signals a week), and {100 * overlap:.0f}% of its signals overlap the live strategies. "
                   "It goes into Saturday's recalibration and trades live only if it passes there too.")
    elif ok_live and not q_ok["overlap"] and q_ok["oos_win"] and q_ok["oos_pf_q"]:
        verdict = "redundant"
        summary = (f"Redundant: it passes ({100 * ow:.1f}% wins on unseen data), but {100 * overlap:.0f}% of its signals "
                   "are trades the live strategies already take, so it adds little.")
    else:
        fails = [c for c in checks + q if not c["ok"]]
        if any(c["key"] == "oos_pf" for c in fails):  # one profit-factor line is enough
            fails = [c for c in fails if c["key"] != "oos_pf_q"]
        order = {"oos_win": 0, "oos_pf": 1, "oos_pf_q": 1, "train_win": 2, "train_pf": 3, "oos_decay": 4,
                 "train_avg": 5, "oos_trades": 6, "overlap": 7}
        fails.sort(key=lambda c: order.get(c["key"], 9))
        near = (st_train.get("win_rate") or 0) >= 0.58 and (st_train.get("profit_factor") or 0) >= 1.2 and (opf or 0) >= 0.95
        bits = []
        for c in fails:
            v = c["value"]
            if v is None:
                bits.append(f"{c['label'].lower()}: no data")
            elif c["key"] in ("train_win", "oos_win", "overlap", "oos_decay"):
                bits.append(f"{c['label'].lower()} {100 * v:.1f}{'%' if c['key'] != 'oos_decay' else ' pts'} (needs {c['need']})")
            elif c["key"] == "train_avg":
                bits.append(f"{c['label'].lower()} {100 * v:+.3f}% (needs {c['need']})")
            else:
                bits.append(f"{c['label'].lower()} {v} (needs {c['need']})")
        verdict = "watch" if near else "rejected"
        summary = ("Watch: close to the bar but not there: " if near else "Rejected: ") + "; ".join(bits[:4]) + "."
    try:
        gp = grade_preview(tr, P_live, split)
    except Exception as ex:  # the preview is a guide, never a reason to lose a test
        gp = {"error": repr(ex)[:200]}
    if gp and "error" not in gp:
        summary += preview_sentence(gp)
    summary += condition_sentence(bc)
    aws = S.all_weather_sentence(aw)
    if aws and not special:
        summary += " " + aws
    variant = res["variant"]
    path = None
    if verdict != "candidate":
        path = S.path_to_pass(st_train, st_test, weeks, extra_checks=q,
                              need={"oos_pf": QUALITY["oos_pf"], "oos_win": QUALITY["oos_win"], "oos_trades": QUALITY["oos_trades"]})
    if gc and not graded_ok and verdict != "candidate" and "error" not in gc:
        miss = [c for c in gc["checks"] if not c["ok"] and c["key"].startswith("g_")]
        if miss:  # how close its graded setups came, so the next variant knows what to fix
            o_ = gc["periods"]["oos"]
            summary += (f" On the setups the desk would send ({o_['trades']} graded 2022-now, "
                        + (f"{100 * o_['win_rate']:.1f}% won" if o_["win_rate"] is not None else "none filled")
                        + "), it misses: " + "; ".join(c["label"].lower() + f" (needs {c['need']})" for c in miss[:3]) + ".")
    doc = {**base, "verdict": verdict, "summary": summary, "path": path, "checks": checks + q, "live_rule_ok": bool(ok_live),
           "live_reason": live_reason, "variant": variant, "rules": R.describe({**spec, "params": variant, "grid": {}}),
           "train": _compact(st_train), "oos": _compact(st_test), "overlap_live": round(overlap, 3),
           "oos_signals_per_week": round(per_week, 2), "grade_preview": gp, "by_condition": bc, "all_weather": aw,
           "graded": gc, "graded_ok": graded_ok,
           **({"market_conditions": special["conditions"], "specialist": special} if special else
              ({"market_conditions": spec_conds} if spec_conds else {})),
           "yearly": B.yearly(tr).reset_index().to_dict("records")[-20:],
           "grid_tried": len(res["tried"]), "runtime_s": round(time.time() - t0, 1)}
    return doc


def run_tests(specs: list[dict], data_dir, state_dir, out_dir, split="2022-01-01", by="research", log=print,
              ideas_to_add: list[dict] | None = None) -> dict:
    from .scan import load_params
    now = cal.now_pt()
    out = pathlib.Path(out_dir)
    writes, results = [], []
    t0 = time.time()
    w = D.load_history(data_dir)
    ind = I.compute(w, D.load_universe(data_dir))
    log(f"history + indicators ready in {time.time() - t0:.0f}s ({w['close'].shape[0]} days x {w['close'].shape[1]} tickers)")
    P = S.params_with_defaults(None)
    P_live = load_params(load_state(state_dir), None)
    known = load_ideas(state_dir)
    for spec in ideas_to_add or []:
        _write(out, writes, "research", spec["id"], idea_doc(spec, by, now, known))
    for spec in specs:
        log(f"testing {spec.get('id')} ...")
        try:
            doc = evaluate_spec(spec, ind, P, P_live, split, now, by, log)
        except Exception as e:  # one bad idea never stops the others
            doc = {"id": f"{spec.get('id')}-{now.strftime('%Y%m%d-%H%M')}", "idea_id": spec.get("id"),
                   "label": spec.get("label"), "tested_at": now.isoformat(timespec="seconds"), "verdict": "rejected",
                   "summary": f"Rejected: the test failed ({e!r})"[:300], "checks": []}
        _write(out, writes, "research_results", doc["id"], doc)
        results.append({"idea": doc["idea_id"], "verdict": doc["verdict"], "summary": doc["summary"],
                        "train_win": (doc.get("train") or {}).get("win_rate"), "oos_win": (doc.get("oos") or {}).get("win_rate"),
                        "oos_pf": (doc.get("oos") or {}).get("profit_factor"), "overlap": doc.get("overlap_live")})
        log(f"  -> {doc['verdict']}: {doc['summary']}")
    res = {"tested": len(results), "results": results, "writes": len(writes), "runtime_s": round(time.time() - t0, 1)}
    _finish(out, writes, res)
    return res


# ----------------------------------------------------------------------------- CLI
def cmd_next(state_dir, out_dir):
    """What to test next: a queued idea from the log; otherwise the next untested idea from the curated
    backlog (engine/research_seed*.json), written to <out>/seed_spec.json so `test --spec` can take it.
    Only when both are empty does the daily run have to research a brand-new idea on the web."""
    ideas, latest = load_ideas(state_dir), latest_results(load_results(state_dir))
    queued = sorted([d for i, d in ideas.items() if i not in latest], key=lambda d: str(d.get("created_at", "")))
    known = [{"id": i, "label": d.get("label"), "family": d.get("family"),
              "verdict": (latest.get(i) or {}).get("verdict", "queued")} for i, d in sorted(ideas.items())]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"next": queued[0]["id"] if queued else None, "queued": [d["id"] for d in queued], "known": known}
    backlog = seed_backlog(ideas)
    res["backlog"] = [sp["id"] for sp in backlog]
    if not queued and backlog:
        sp = backlog[0]
        (out / "seed_spec.json").write_text(json.dumps(sp, indent=1))
        res.update(next=sp["id"], from_backlog=True, spec_file=str((out / "seed_spec.json").resolve()),
                   similar=similar_ideas(sp, ideas))
    families = sorted({str(d.get("family", "")).split("/")[0].strip() for d in ideas.values()} - {""})
    res["families_covered"] = families
    focus = research_focus(state_dir)
    if focus:
        res["focus"] = focus  # when designing a new idea, serve this first
    (out / "next.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "add", "test", "seed"])
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--spec", default=None, help="JSON file with one spec (or a list of specs)")
    ap.add_argument("--id", default=None, help="queued idea id(s), comma separated")
    ap.add_argument("--queued", type=int, default=0, help="test the N oldest queued ideas")
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--by", default="daily research")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.cmd == "next":
        return cmd_next(a.state, a.out)
    ideas = load_ideas(a.state)
    now = cal.now_pt()
    if a.cmd in ("add", "seed"):
        specs = json.loads(pathlib.Path(a.spec).read_text()) if a.cmd == "add" else json.loads(SEED_FILE.read_text())
        specs = specs if isinstance(specs, list) else [specs]
        writes, added, refused = [], [], []
        for sp in specs:
            errs = check_new(sp, ideas)
            if errs:
                refused.append({"id": sp.get("id"), "why": errs})
                continue
            _write(out, writes, "research", sp["id"], idea_doc(sp, "seed backlog" if a.cmd == "seed" else a.by, now))
            ideas[sp["id"]] = {"signature": R.signature(sp)}
            added.append(sp["id"])
        return _finish(out, writes, {"added": added, "refused": refused, "writes": len(writes)})
    # test
    if not a.data:
        ap.error("test needs --data")
    specs, to_add = [], []
    if a.spec:
        raw = json.loads(pathlib.Path(a.spec).read_text())
        for sp in (raw if isinstance(raw, list) else [raw]):
            if sp.get("id") in ideas:
                specs.append(ideas[sp["id"]].get("spec") or sp)
                continue
            errs = check_new(sp, ideas)
            if errs:
                print(json.dumps({"refused": sp.get("id"), "why": errs}, indent=1))
                continue
            specs.append(sp)
            to_add.append(sp)
    if a.id:
        for iid in a.id.split(","):
            iid = iid.strip()
            if iid in ideas and ideas[iid].get("spec"):
                specs.append(ideas[iid]["spec"])
            else:
                print(json.dumps({"missing": iid}))
    if a.queued:
        latest = latest_results(load_results(a.state))
        queued = sorted([d for i, d in ideas.items() if i not in latest and d.get("spec")],
                        key=lambda d: str(d.get("created_at", "")))
        specs += [d["spec"] for d in queued[:a.queued]]
    if not specs:
        return _finish(out, [], {"tested": 0, "results": [], "writes": 0, "note": "nothing to test"})
    run_tests(specs, a.data, a.state, a.out, a.split, a.by, ideas_to_add=to_add)


if __name__ == "__main__":
    main()
