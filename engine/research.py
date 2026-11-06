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
import time

import numpy as np
import pandas as pd

from . import backtest as B
from . import calendar as cal
from . import data as D
from . import indicators as I
from . import rules as R
from . import strategies as S
from .portfolio import load_collection, load_state

QUALITY = {"oos_win": 0.62, "oos_pf": 1.15, "oos_trades": 80, "max_overlap": 0.70}
SEED_FILE = pathlib.Path(__file__).resolve().parent / "research_seed.json"
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
    """Specs the weekly recalibration should evaluate: research candidates plus strategies promoted earlier."""
    ideas, results = load_ideas(state_dir), load_results(state_dir)
    for r in extra_results or []:
        results[r["id"]] = r
    latest = latest_results(results)
    pool = {}
    for iid, cu in ((old_params or {}).get("custom") or {}).items():
        if cu.get("spec"):
            pool[iid] = (1.0, cu["spec"])  # already promoted: always re-tested
    for iid, r in latest.items():
        if r.get("verdict") == "candidate" and iid in ideas and ideas[iid].get("spec") and iid not in pool:
            pool[iid] = (float((r.get("oos") or {}).get("win_rate") or 0), ideas[iid]["spec"])
    ranked = sorted(pool.items(), key=lambda kv: -kv[1][0])[:cap]
    return {k: v[1] for k, v in ranked}


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


def idea_doc(spec: dict, by: str, now) -> dict:
    return {"id": spec["id"], "label": spec["label"], "family": spec["family"], "about": spec["about"],
            "source": spec.get("source"), "spec": spec, "rules": R.describe(spec), "signature": R.signature(spec),
            "created_at": now.isoformat(timespec="seconds"), "created_by": by}


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
    if ok_live and all(q_ok.values()):
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
    variant = res["variant"]
    path = None
    if verdict != "candidate":
        path = S.path_to_pass(st_train, st_test, weeks, extra_checks=q,
                              need={"oos_pf": QUALITY["oos_pf"], "oos_win": QUALITY["oos_win"], "oos_trades": QUALITY["oos_trades"]})
    doc = {**base, "verdict": verdict, "summary": summary, "path": path, "checks": checks + q, "live_rule_ok": bool(ok_live),
           "live_reason": live_reason, "variant": variant, "rules": R.describe({**spec, "params": variant, "grid": {}}),
           "train": _compact(st_train), "oos": _compact(st_test), "overlap_live": round(overlap, 3),
           "oos_signals_per_week": round(per_week, 2),
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
    for spec in ideas_to_add or []:
        _write(out, writes, "research", spec["id"], idea_doc(spec, by, now))
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
    ideas, latest = load_ideas(state_dir), latest_results(load_results(state_dir))
    queued = sorted([d for i, d in ideas.items() if i not in latest], key=lambda d: str(d.get("created_at", "")))
    known = [{"id": i, "label": d.get("label"), "family": d.get("family"),
              "verdict": (latest.get(i) or {}).get("verdict", "queued")} for i, d in sorted(ideas.items())]
    res = {"next": queued[0]["id"] if queued else None, "queued": [d["id"] for d in queued], "known": known}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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
