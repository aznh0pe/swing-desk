"""Browser test: every view of an open position gives the same instruction.

Nov 11, 2026: the 12:26 PM pre-close scan said "sell at the close" for two positions. At 2:07 PM the post-close
scan re-checked them (one: sell at the open; the other: keep holding), but the Alerts tab kept listing the old
"sell at close" alerts while the position cards (Overview and Alerts) showed the new instructions. This test
replays that day with made-up positions and checks that the Overview cards, the Alerts tab's action list, the
Alerts tab cards and the tab badge all agree, and that a superseded instruction never shows.
"""
import json
import pathlib

import pytest

pw = pytest.importorskip("playwright.sync_api")

ROOT = pathlib.Path(__file__).resolve().parent.parent
MOCK = """
window.claude = { use: async (n) => {
  const data = window.__DATA;
  if (n === "mcp") return { callTool: async () => ({ payload: {} }) };
  if (n === "user") return { isOwner: () => true, canEdit: () => true };
  if (n !== "db") return null;
  const snapDoc = (p) => ({ exists: !!data[p], data: () => data[p], id: p.split("/").pop(), metadata: { fromCache: false } });
  const snapCol = (c) => ({ docs: (data[c] || []).map((d) => ({ id: d.id, data: () => d, exists: true })), metadata: { fromCache: false } });
  const col = (c) => { const q = { orderBy: () => q, limit: () => q, where: () => q,
    onSnapshot: (cb) => { setTimeout(() => cb(snapCol(c)), 5); return () => {}; }, doc: (id) => docRef(c + "/" + id) }; return q; };
  const docRef = (p) => ({ onSnapshot: (cb) => { setTimeout(() => cb(snapDoc(p)), 5); return () => {}; },
    set: async () => {}, update: async () => {}, delete: async () => {} });
  return { doc: docRef, collection: col };
} };
"""


def _pos(tid, tk, last, status, signals, entry=100.0):
    return {"trade_id": tid, "ticker": tk, "asof": "2026-11-11", "entry": entry, "entry_date": "2026-11-06",
            "last": last, "unrealized": round(last - entry, 2), "unrealized_pct": round(100 * (last / entry - 1), 2),
            "shares": 1, "stop": 85.0, "target": 110.0, "max_hold": 8, "days_held": 4, "days_left": 4,
            "dyn_exit": "sma5", "sma5": 101.0, "signals": signals, "status": status}


def _sell_close(tk):
    return {"code": "EXIT_RULE", "label": "SELL AT TODAY'S CLOSE (before 1:00 PM PT): EXIT RULE MET", "tone": "sell",
            "when": "close", "detail": f"{tk}: exit rule met: the close is above its 5-day average."}


def fixture(latest_b=None):
    trades = [{"id": f"t_{tk}", "ticker": tk, "status": "open", "entry_price": 100.0, "shares": 1, "stop": 85.0,
               "target": 110.0, "entry_date": "2026-11-06", "max_hold": 8, "dyn_exit": "sma5", "strategy": "FAIR_VALUE",
               "strategy_label": "Below Fair Value (VWAP-20)"} for tk in ("AAA", "BBB")]
    sig = lambda txt, act, urg: [{"code": "EXIT_RULE", "level": "alert", "action": act, "urgency": urg, "text": txt}]
    pre = {"id": "20261111-1226-preclose", "mode": "preclose", "started_at": "2026-11-11T12:26:00-08:00",
           "status": "ok", "session_day": "2026-11-11", "run_issues": [],
           "positions": [_pos(f"t_{tk}", tk, 104.0, _sell_close(tk), sig("sell before the close", "SELL AT CLOSE", "today"))
                         for tk in ("AAA", "BBB")]}
    post_b = latest_b or ({"code": "HOLD", "label": "KEEP HOLDING", "tone": "hold", "when": "none",
                           "detail": "Nothing to do today."}, [])
    post = {"id": "20261111-1407-postclose", "mode": "postclose", "started_at": "2026-11-11T14:07:00-08:00",
            "status": "ok", "session_day": "2026-11-11", "run_issues": [],
            "positions": [_pos("t_AAA", "AAA", 104.7, {"code": "EXIT_RULE", "label": "SELL AT THE OPEN: EXIT RULE MET",
                                                         "tone": "sell", "when": "open", "detail": "Take the gain at the next open."},
                               sig("take the gain at the next open", "SELL AT OPEN", "next")),
                          _pos("t_BBB", "BBB", 99.4, post_b[0], post_b[1])]}
    # the open exit alerts the pre-close scan wrote (never overwritten; later scans only send their view of them)
    alerts = [{"id": f"x_2026-11-11_t_{tk}_EXIT_RULE", "type": "exit", "status": "new", "trade_id": f"t_{tk}",
               "ticker": tk, "code": "EXIT_RULE", "action": "SELL AT CLOSE", "urgency": "today",
               "text": "Take the gain before the 1:00 PM PT close today.", "created_at": "2026-11-11T12:26:00-08:00",
               "scan_id": pre["id"], "last": 104.0, "unrealized": 4.0, "days_held": 4, "days_left": 4}
              for tk in ("AAA", "BBB")]
    return {"config/settings": {"initial_balance": 5000, "max_open_positions": 5, "max_position_pct": 25, "risk_pct": 2,
                                "income_default_pct": 30, "timezone": "America/Los_Angeles"},
            "alerts": alerts, "trades": trades,
            "ledger": [{"id": "initial", "type": "initial", "amount": 5000, "date": "2026-11-05"}],
            "scans": [post, pre], "journal": [], "learning": [], "research": [], "research_results": [], "runs": [],
            "strategy/params": {"params": {"grading": {"aplus_score": 86, "aplus_pwin": 0.62}, "strategies": {}}},
            "strategy/backtest": {}}


READ = """() => {
  const out = { cards: {}, actions: {}, badge: Number((document.querySelector('#tab-alerts .count') || {}).textContent || 0),
                text: document.querySelector('#main').innerText };
  document.querySelectorAll('article.pos-card').forEach((c) => {
    out.cards[c.querySelector('.tk').textContent.trim()] = c.querySelector('.pc-status .lbl').textContent.trim(); });
  document.querySelectorAll('[data-pos-action], .act').forEach((a) => {
    out.actions[a.querySelector('.tk').textContent.trim()] = a.querySelector('.what').textContent.trim(); });
  return out; }"""


def render(fx):
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + (ROOT / "dashboard" / "index.html").read_text()
    import datetime as dt
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # no browser on this machine
            pytest.skip(f"chromium unavailable: {e}")
        ctx = b.new_context(viewport={"width": 1180, "height": 900}, timezone_id="America/Los_Angeles")
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.add_init_script(f"window.__DATA = {json.dumps(fx)};" + MOCK)
        page.clock.set_fixed_time(dt.datetime.fromisoformat("2026-11-11T17:05:00-08:00"))
        page.route("http://dash.test/", lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        page.goto("http://dash.test/")
        page.wait_for_timeout(700)
        page.click("#tab-overview")
        page.wait_for_timeout(200)
        ov = page.evaluate(READ)
        page.click("#tab-alerts")
        page.wait_for_timeout(200)
        al = page.evaluate(READ)
        b.close()
    assert not errs, errs
    return ov, al


def test_overview_and_alerts_give_each_position_the_same_instruction():
    ov, al = render(fixture())
    # the latest scan decides: AAA sell at the open, BBB keep holding, on every card in both tabs
    assert ov["cards"] == al["cards"] == {"AAA": "SELL AT THE OPEN: EXIT RULE MET", "BBB": "KEEP HOLDING"}
    # the Alerts tab's action list holds exactly the positions whose latest instruction needs action, worded the same
    assert al["actions"] == {"AAA": "SELL AT THE OPEN: EXIT RULE MET"}
    assert al["badge"] == 1
    # the superseded pre-close instruction shows nowhere
    assert "SELL AT CLOSE" not in al["text"] and "SELL AT TODAY'S CLOSE" not in al["text"]
    assert "SELL AT TODAY'S CLOSE" not in ov["text"]


def test_a_stop_to_raise_is_listed_with_the_same_wording_everywhere():
    raise_ = ({"code": "RAISE_STOP", "label": "KEEP HOLDING · RAISE THE STOP TO $101.00", "tone": "act", "when": "next",
               "detail": "Raise the stop before the next open, then press Stop moved."},
              [{"code": "RAISE_STOP", "level": "info", "action": "RAISE STOP", "urgency": "next", "new_stop": 101.0,
                "text": "Raise the stop to $101.00."}])
    ov, al = render(fixture(latest_b=raise_))
    assert ov["cards"] == al["cards"] and ov["cards"]["BBB"] == raise_[0]["label"]
    assert al["actions"] == {"AAA": "SELL AT THE OPEN: EXIT RULE MET", "BBB": raise_[0]["label"]}
    assert al["badge"] == 2
