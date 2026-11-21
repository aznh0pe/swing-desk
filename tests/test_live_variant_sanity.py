"""Sat Nov 21, 2026: the weekly recalibration kept the live setup, but backtest_report.json still carried the
refit's figures for ETF Close-at-Low Snapback (a limit-order variant, 76.6% -> 66.9% on unseen data), so the
sanity check flagged the live buy-at-the-close variant, which re-tested at 76.2% / 73.0%. When the live setup is
kept, live_variant_stats re-tests each live built-in strategy whose parameters differ and puts those figures in
the report, with the refit's moved to rec["refit"]."""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import calibrate as C  # noqa: E402


def _trades(n_train, w_train, n_test, w_test):
    rows = []
    for i in range(n_train):
        rows.append({"signal_date": pd.Timestamp("2015-01-05") + pd.Timedelta(days=i), "ret": 0.02 if i < w_train else -0.03,
                     "filled": True})
    for i in range(n_test):
        rows.append({"signal_date": pd.Timestamp("2023-01-05") + pd.Timedelta(days=i), "ret": 0.02 if i < w_test else -0.03,
                     "filled": True})
    return pd.DataFrame(rows)


def test_live_variant_replaces_refit_figures(monkeypatch):
    report = {"strategies": {
        "ETF_IBS": {"enabled": True, "variant": {"entry_kind": "limit_next"},
                    "train": {"win_rate": 0.766}, "test_oos": {"win_rate": 0.669}},
        "FAIR_VALUE": {"enabled": True, "variant": {"entry_kind": "limit_next"},
                       "train": {"win_rate": 0.7}, "test_oos": {"win_rate": 0.7}},
        "R_X": {"enabled": True, "custom": True, "variant": {"k": 1}, "train": {}, "test_oos": {}}}}
    P_live = {"strategies": {"ETF_IBS": {"enabled": True, "entry_kind": "close"},
                             "FAIR_VALUE": {"enabled": True, "entry_kind": "limit_next"},
                             "R_X": {"enabled": True}}}
    seen = []
    monkeypatch.setattr(C.S, "all_plans", lambda ind, P: {"ETF_IBS": "plan", "FAIR_VALUE": "plan", "R_X": "plan"})
    monkeypatch.setattr(C.B, "simulate", lambda plan, ind, P, include_unfilled=True: seen.append(plan) or _trades(100, 76, 60, 44))
    monkeypatch.setattr(C.B, "stats", lambda t: {"win_rate": round(float((t["ret"] > 0).mean()), 3) if len(t) else None,
                                                 "trades": len(t)})
    done = C.live_variant_stats(report, {}, P_live, "2022-01-01", log=lambda *a: None)
    assert done == ["ETF_IBS"]   # unchanged variant and research strategies are left alone
    rec = report["strategies"]["ETF_IBS"]
    assert rec["stats_of"] == "live variant" and rec["variant"] == {"entry_kind": "close"}
    assert rec["train"]["win_rate"] == 0.76 and rec["test_oos"]["win_rate"] == round(44 / 60, 3)
    assert rec["refit"]["test_oos"]["win_rate"] == 0.669
    assert "refit" not in report["strategies"]["FAIR_VALUE"]


def test_ensure_swap_never_raises(tmp_path, monkeypatch):
    """Sat Nov 21, 2026: the tier check ran an 8 GB machine out of memory; run_tiers now adds swap itself."""
    big = tmp_path / "big"
    big.write_text("MemTotal: 32000000 kB\nMemAvailable: 30000000 kB\nSwapTotal: 0 kB\n")
    assert "nothing to add" in C.ensure_swap(meminfo=str(big))
    small = tmp_path / "small"
    small.write_text("MemTotal: 8000000 kB\nMemAvailable: 7000000 kB\nSwapTotal: 0 kB\n")

    class R:
        returncode, stdout, stderr = 1, "", "Operation not permitted"
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    out = C.ensure_swap(meminfo=str(small), path=str(tmp_path / "sw"))
    assert "could not add swap" in out
    assert "swap check skipped" in C.ensure_swap(meminfo=str(tmp_path / "missing"))
