"""Owner, Nov 23, 2026: "when there are a ton of A+ alerts in a given day, make sure you show the best optimal ones
... prioritize which ones to execute first ... collapse the remaining ones as optional remaining A+ set ups ... any
triggered alert should still learn and be tested."

One rule, used by the scan and by the news check's re-rank (scan.feature_block): best score first, at most 2 per
sector (open positions count), at most 3 per strategy, 8 at most; the first three are marked "place first". Every
other A+ setup stays its own alert, listed as optional, and the learning run grades all three groups every day."""
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import calibrate as C  # noqa: E402
from engine import scan as SC  # noqa: E402
from engine import strategies as S  # noqa: E402


def _pick(cands, held=None):
    """cands: (ticker, sector, strategy) in rank order -> (featured tickers, {ticker: reason})."""
    held, fsec, fstrat, feat, why = dict(held or {}), {}, {}, [], {}
    for tk, sec, st in cands:
        b = SC.feature_block(sec, st, len(feat), held, fsec, fstrat, st.title())
        if b is None:
            feat.append(tk)
            fsec[sec] = fsec.get(sec, 0) + 1
            fstrat[st] = fstrat.get(st, 0) + 1
        else:
            why[tk] = b
    return feat, why


def test_one_strategy_firing_on_dozens_of_tickers_gets_three_top_picks_not_eight():
    # Nov 20's night in miniature: one strategy on 30 tickers across 30 sectors, then two setups from other strategies
    cands = [(f"T{i}", f"Sector {i}", "panic") for i in range(30)] + [("U1", "Sector 40", "washout"), ("U2", "Sector 41", "pullback")]
    feat, why = _pick(cands)
    assert feat == ["T0", "T1", "T2", "U1", "U2"]
    assert why["T3"] == "3 top picks already from Panic: more would be the same bet"
    assert (SC.FEATURED, SC.PER_SECTOR, SC.PER_STRATEGY, SC.PLACE_FIRST) == (8, 2, 3, 3)


def test_the_sector_cap_still_counts_open_positions_and_the_list_stops_at_eight():
    feat, why = _pick([("A", "Tech", "s1"), ("B", "Tech", "s2"), ("C", "Energy", "s3")], held={"Tech": 1})
    assert feat == ["A", "C"] and why["B"] == "2 positions/top picks already in Tech"
    many = [(f"T{i}", f"Sec{i}", f"s{i}") for i in range(12)]
    feat, why = _pick(many)
    assert len(feat) == 8 and why["T8"] == "outside tonight's top 8"


def test_the_scan_and_the_news_rerank_share_the_rule_and_the_wording():
    scan_src = (ROOT / "engine" / "scan.py").read_text()
    post_src = (ROOT / "engine" / "postprocess.py").read_text()
    assert "blocked = feature_block(sc, cd[\"name\"], rank, sec_count, featured_sec, featured_strat, meta[\"label\"])" in scan_src
    assert "blocked = feature_block(sk, e.get(\"strategy\"), len(feat), held, fsec, fstrat, e.get(\"strategy_label\"))" in post_src
    assert "FEATURED, PER_SECTOR = 8, 2" not in post_src          # one set of numbers, in scan.py
    note = SC.optional_note("outside tonight's top 8")
    assert note.startswith("Optional A+ setup (outside tonight's top 8).") and SC.is_optional_note(note)
    assert SC.is_optional_note("Also qualified (2 already in Financials).")   # cards written before Nov 23
    assert not SC.is_optional_note("Ranked #4 tonight and you have 0 free position slot(s).")


def _entry(tk, rank=None, **kw):
    e = {"id": f"e_{tk}", "ticker": tk, "name": tk, "strategy": "X", "strategy_label": "Dip", "grade": "A+", "score": 99.0,
         "win_pct": 78.0, "exp_return_pct": 1.5, "entry": 100.0, "stop": 90.0, "target": 105.0, "rr": 0.5, "shares": 5,
         "total_cost": 500.0, "pct_of_capital": 10.0, "risk_usd": 50.0, "hold_max_days": 7, "hold_typical_days": 3,
         "valid_for": "2026-11-24", "order": "Buy LIMIT $100.00", "exit_signal": "Sell above the 5-day average.",
         "why": ["RSI(2) 4"], "earnings_next": None, "news_check": "pending", "rank": rank}
    if rank is None:
        e.update(also_qualified=True, capacity_note=SC.optional_note("outside tonight's top 8"))
    e.update(kw)
    return e


def test_the_email_says_what_to_place_first_and_lists_the_rest_as_optional():
    entries = [_entry(f"F{i}", i) for i in range(1, 6)] + [_entry(f"O{i}") for i in range(1, 4)]
    scan = {"mode": "postclose", "market": {"label": "Correction", "spy": 1, "spy_vs_200d_pct": 1, "spy_5d_pct": 1,
            "breadth_above_50d_pct": 50, "vix": 26}, "funds": {"capital": 5000, "cash": 900, "open_count": 7},
            "data": {"status": "ok", "notes": []}, "alert_updates": {}, "fill_checks": [], "counts": {}}
    txt = SC.make_digest(scan, entries, [], [], [], 0, S.params_with_defaults(None), dt.datetime(2026, 11, 23, 14, 6))
    assert "(5 featured, 3 more below)" in txt and SC.PICK_RULE in txt
    assert "The other 3 A+ setups tonight are optional" in txt
    assert txt.count("PRIORITY Place first: one of tonight's top three.") == 3
    assert txt.count("PRIORITY Next in line: place it after #1-#3") == 2
    head = txt.index(SC.OPTIONAL_HEAD)
    assert all(txt.index(f"F{i} |") < head for i in range(1, 6)) and all(txt.index(f"O{i}") > head for i in range(1, 4))
    assert "ALSO QUALIFIED" not in txt
    quiet = SC.make_digest(scan, [_entry("F1", 1)], [], [], [], 0, S.params_with_defaults(None), dt.datetime(2026, 11, 23, 14, 6))
    assert SC.PICK_RULE not in quiet and SC.OPTIONAL_HEAD not in quiet    # a normal night needs no explanation


def test_every_sent_setup_is_graded_and_the_three_groups_are_compared_on_live_results():
    alerts = ([{"id": f"f{i}", "rank": 1 + i % 3} for i in range(12)] + [{"id": f"n{i}", "rank": 4 + i % 5} for i in range(10)]
              + [{"id": f"o{i}", "rank": None, "also_qualified": True} for i in range(40)])
    shadow = ([{"alert_id": f"f{i}", "status": "closed", "fill": 100.0, "exit": 103.0 if i < 9 else 96.0} for i in range(12)]
              + [{"alert_id": f"n{i}", "status": "closed", "fill": 100.0, "exit": 101.0 if i < 7 else 97.0} for i in range(10)]
              + [{"alert_id": f"o{i}", "status": "closed", "fill": 100.0, "exit": 101.0 if i < 28 else 96.0} for i in range(40)]
              + [{"alert_id": "o39", "status": "pending"}])
    sc = C.pick_scorecard(alerts, shadow)
    assert sc["first"] == {"n": 12, "wins": 9, "win_rate": 0.75, "avg_ret": 0.0125}
    assert sc["next"]["n"] == 10 and sc["optional"]["n"] == 40 and sc["optional"]["win_rate"] == 0.7
    line = C.pick_scorecard_line(sc)
    assert line.startswith("Selection check") and "top picks #1-3 won 75% of 12" in line and "optional A+ setups won 70% of 40" in line
    assert "up for review" not in line
    assert C.pick_scorecard_line({"first": {"n": 3, "wins": 2, "win_rate": 0.667, "avg_ret": 0.01}}) is None   # too few to say
    bad = {"first": {"n": 40, "wins": 24, "win_rate": 0.6, "avg_ret": -0.002}, "next": {"n": 0},
           "optional": {"n": 60, "wins": 45, "win_rate": 0.75, "avg_ret": 0.01}}
    assert "up for review" in C.pick_scorecard_line(bad)                 # the rule is watched, not trusted
    src = (ROOT / "engine" / "calibrate.py").read_text()
    assert '"by_pick": picks,' in src and "picks = pick_scorecard(alerts, shadow)" in src


def test_the_dashboard_marks_the_first_three_and_collapses_the_optional_ones():
    html = (ROOT / "dashboard" / "index.html").read_text()
    for needle in ("Optional remaining A+ setups (${extra.length})", "Place first", "Next in line", "const PLACE_FIRST = 3;",
                   "Why not a top pick", "Selection check (every sent setup, graded as if placed)"):
        assert needle in html, needle
    assert "Also qualified tonight" not in html
