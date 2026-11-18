"""Buy-near-the-close setups: a limit at the alert price, placed late, never chased.

Nov 17, 2026: the pre-close scan sent XOP ($177.25) and XLE ($63.16), both A+. The owner placed both with limits
at the alert price in the last minutes. XOP filled. XLE lifted off its low and closed at $63.36, so it didn't fill.
The review then said XLE "should have filled: a day limit order at the alert price, placed before the open", which
is the wrong story for a buy near the close, and the grader counted XLE as bought at $63.36. On the final bar XLE sat
at 19% of its day's range (the rule needs under 10%): it no longer qualified, so no fill was the right result.
"""
import datetime as dt
import pathlib

import pandas as pd

from engine import calibrate as CB
from engine import learnlog as LL
from engine import scan as SC
from engine import strategies as S

IDX = pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"])


def _ind():
    f = lambda xle, xop: pd.DataFrame({"XLE": xle, "XOP": xop}, index=IDX)
    ind = {"open": f([64.52, 64.09, 63.5, 64.0], [182.59, 180.88, 178.0, 180.0]),
           "high": f([64.69, 64.57, 64.2, 64.9], [182.94, 182.75, 181.0, 183.5]),
           "low": f([63.5, 63.07, 63.2, 63.9], [179.46, 177.04, 177.5, 179.5]),
           "close": f([63.69, 63.36, 64.0, 64.8], [179.98, 177.20, 180.5, 183.2])}
    for k in ("sma5", "rsi2", "max7c"):
        ind[k] = f([999.0] * 4, [999.0] * 4) if k != "rsi2" else f([10.0] * 4, [10.0] * 4)
    return ind


def test_the_order_line_gives_a_limit_and_says_not_to_chase():
    t = SC.close_order_text(63.16, 59.5, 64.3)
    assert t.startswith("Buy near the close")          # the grader and the fill check recognise the kind by this prefix
    assert "LIMIT at $63.16" in t and "last 10 minutes" in t and "Never chase" in t
    assert "stop $59.50" in t and "target $64.30" in t
    assert "limit at about the last price" not in pathlib.Path(SC.__file__).read_text().split("def close_order_text")[0]
    src = pathlib.Path(SC.__file__).read_text()
    assert 'close_order_text(cd["entry"], cd["stop"], cd["target"]) if cd["plan"]["entry_kind"] == "close"' in src
    assert "never chase above it" in src               # the email's line above the pre-close setups


def test_the_grader_does_not_buy_a_close_setup_that_lifted_off_its_low():
    ind = _ind()
    xle = {"id": "e1", "ticker": "XLE", "strategy": "ETF_IBS", "valid_for": "2026-11-17", "entry": 63.16, "stop": 59.5,
           "target": 64.3, "hold_max_days": 7, "dyn_exit": "none", "entry_kind": "close",
           "order": SC.close_order_text(63.16, 59.5, 64.3)}
    assert CB.shadow_outcome(xle, ind)["status"] == "unfilled"          # closed $63.36, above the $63.16 limit
    # a close at or under the alert price is bought at the close, as in the backtest
    xop = {**xle, "id": "e2", "ticker": "XOP", "entry": 177.25, "stop": 160.56, "target": 181.43}
    r = CB.shadow_outcome(xop, ind)
    assert r["status"] == "closed" and r["fill"] == 177.20 and r["reason"] == "target"
    # the kind is recognised from the old order line too, and from entry_kind alone
    old = {**xle, "order": "Buy near the close today (limit at about the last price)"}
    old.pop("entry_kind")
    assert CB.shadow_outcome(old, ind)["status"] == "unfilled"
    assert CB.shadow_outcome({**xle, "order": ""}, ind)["status"] == "unfilled"


def test_the_fill_check_says_what_the_close_did_against_the_limit():
    ind = _ind()
    a = {"id": "e1", "ticker": "XLE", "type": "entry", "status": "placed", "valid_for": "2026-11-17", "entry": 63.16,
         "entry_kind": "close", "stop": 59.5, "target": 64.3}
    fc = SC.order_fill_check(a, ind, dt.date(2026, 11, 18), "premarket", False)
    assert fc["at_close"] and fc["filled"] is None and fc["close"] == 63.36      # a hint, never a verdict
    assert "closed at $63.36, above the $63.16 limit" in fc["text"] and "right result" in fc["text"]
    assert "press Didn't fill." in fc["text"]
    ok = SC.order_fill_check({**a, "ticker": "XOP", "entry": 177.25, "status": "new"}, ind, dt.date(2026, 11, 18), "premarket", False)
    assert "closed at $177.20, at or under the $177.25 limit" in ok["text"] and "probably filled" in ok["text"]
    assert "press I didn't place it, or Didn't see it in time" in ok["text"]
    # no price data for the day: the question is still asked, without the hint
    none = SC.order_fill_check(a, {}, dt.date(2026, 11, 18), "premarket", False)
    assert none["at_close"] and "close" not in none and "Only your brokerage knows for sure" in none["text"]


def test_the_review_tells_the_two_kinds_of_unfilled_order_apart():
    P = S.params_with_defaults(None)
    base = {"type": "entry", "status": "unfilled", "strategy": "ETF_IBS", "grade": "A+"}
    alerts = [{**base, "id": "L1", "ticker": "AAA", "valid_for": "2026-11-16", "entry": 10.0, "order": "Buy LIMIT $10.00, day order"},
              {**base, "id": "C1", "ticker": "XOP", "valid_for": "2026-11-17", "entry": 177.25, "entry_kind": "close",
               "order": SC.close_order_text(177.25, 160.56, 181.43)}]
    shadow = [{"alert_id": "L1", "status": "closed", "fill": 10.0, "exit": 10.5},
              {"alert_id": "C1", "status": "closed", "fill": 177.2, "exit": 181.43}]
    doc = LL.portfolio_stats(alerts, shadow, [], [], "2026-11-19", P)
    lim = [x for x in doc["lessons"] if "placed before the open" in x]
    cl = [x for x in doc["lessons"] if "buy-near-the-close order(s) recorded as not filled" in x]
    assert len(lim) == 1 and "AAA" in lim[0] and "XOP" not in lim[0]
    assert len(cl) == 1 and "XOP 2026-11-17" in cl[0] and "last 10 minutes" in cl[0]
    # XLE on Nov 17: the grader says unfilled (it closed above the limit), so the review has nothing to flag
    xle = [{**base, "id": "C2", "ticker": "XLE", "valid_for": "2026-11-17", "entry": 63.16, "entry_kind": "close",
            "order": SC.close_order_text(63.16, 59.5, 64.3)}]
    doc = LL.portfolio_stats(xle, [{"alert_id": "C2", "status": "unfilled"}], [], [], "2026-11-19", P)
    assert not any("recorded as not filled" in x for x in doc["lessons"])
