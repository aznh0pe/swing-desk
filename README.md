# Swing Desk

A rules-based swing-trading desk for a small account: it scans ~600 liquid US stocks and
ETFs several times a day, sends only A+ setups (limit entry, stop, target, size, hold
window, exit rule), watches open positions for early exits, and learns from every alert.

**Not financial advice.** These are model-generated signals from backtested rules. Win
percentages are estimates and can be wrong. You place and manage every order yourself.

## How it fits together

| Piece | Where | What it does |
|---|---|---|
| Market-data feed | `.github/workflows/market-data.yml`, `pipeline/` | GitHub Action, asked every 15 min 3-4 AM to 3-4 PM PT on weekdays (GitHub delays or skips some scheduled runs, so the effective cadence is about every 30 min), plus a late-evening safety pass; a Saturday keepalive commit stops GitHub from pausing the schedule after 60 quiet days: daily bars (3 yrs), intraday snapshot, pre-market snapshot, earnings calendar; Saturday: full history since 2008. Writes the `data` branch (one squashed commit, no history bloat). |
| Engine | `engine/` | Indicators, 7 built-in strategies plus research strategies from JSON rule specs, conservative backtester, win-probability model, scoring/grading, position sizing, exit monitor, daily learning, weekly recalibration. |
| Scans | Claude scheduled tasks (`ops/schedule.json`) | Pre-market 5:50 AM PT, pre-close 12:25 PM PT (9:20 AM on half days), post-close 2:05 PM PT, learning 3:12 PM PT, recalibration Sat 10:14 AM PT. Each attaches this repo (or, when a session can't, runs the identical engine copy published with the dashboard), clones the `data` branch, reads the dashboard state, runs the engine, news-checks the setups, writes alerts, confirms the scan document saved, and emails the digest. |
| Backup checks | Claude scheduled tasks | 6:02 AM, 12:38 PM, 2:53 PM, 4:23 PM on weekdays and Sun 10:23 AM. Each confirms its run saved results (and waits if it is still running); if it failed or never ran, the backup carries out the full run itself, so no run stays failed. |
| Dashboard | Claude artifact | Overview (monthly / QTD / historical), Alerts (exits, new A+ setups, open positions), Trade log, Funds & income, Scans (with Scan now), Strategy lab (backtests + learning journal). |

## Strategies

| Key | Name | Family |
|---|---|---|
| MR_STRETCH | RSI(2) Stretch Dip | Mean reversion: uptrend name, RSI(2) washed out, limit below the close |
| LEADER_PULLBACK | Leader 7-Day Pullback | Momentum pullback in top relative-strength stocks |
| ETF_IBS | ETF Close-at-Low Snapback | Mean reversion / auction "poor low" in index and sector ETFs |
| TREND_PULLBACK | Trend Pullback to 20-EMA | Price action / trend continuation |
| BREAKOUT_RETEST | Tight-Base Breakout Retest | Breakout, bought on the retest |
| GAP_HOLD | Power-Gap Hold | Event drift after a high-volume gap |
| FAIR_VALUE | Below Fair Value (VWAP-20) | Fair value / auction rotation back to value |

Each strategy's parameters are picked by grid search on 2008-2021 and judged on
2022-present out-of-sample data (`python -m engine.calibrate backtest`). A strategy trades
live only while it passes five checks (`engine/strategies.py` ENABLE_RULE): 2008-2021 win
rate >= 58%, profit factor >= 1.30 and average trade > +0.20%, then out-of-sample profit
factor >= 1.05 with a win rate no more than 8 points below in-sample. The weekly
recalibration re-tests all of them; the daily review pauses one whose live record drops
below 50%. The reasons are written into the backtest report and shown in the Strategy lab.

## Strategy research

New ideas are plain JSON rule specs (`engine/rules.py` documents the format and the fields
you can use), so they can be written, tested and promoted without code changes.

```bash
python -m engine.research next --state state --out rs                    # next queued idea
python -m engine.research test --spec idea.json --data data --state state --out rs
python -m engine.research test --queued 3 --data data --state state --out rs
```

`test` tunes the idea's small grid on 2008-2021 only, then reports unseen 2022-now results,
the live-trading checks and how many of its signals the live strategies already take.
Verdicts: candidate (>= 62% wins and PF >= 1.15 on unseen data, mostly new trades), watch,
redundant, rejected. The daily learning task tests one or two ideas (a queued one, or a new
one it researched that day); `engine/research_seed.json` is the starter backlog. Saturday's
recalibration evaluates the candidates with the live strategies and keeps one on only if it
passes the live checks and keeps or raises the A+ win rate and average trade.

## Dashboard refresh and Scan now

Alerts are create-only, so a later scan never overwrites what you did on a card. Each scan
document carries `alert_updates` (rank, score, news verdict, sizing, withdrawn) for alerts
that already exist; the dashboard applies the newest one. Pressing Scan now writes
`runs/<token>-s0`, fires the on-demand task with the token, the task records steps 1-5 and
the scanner adds step 6 (done) to its results batch, which drives the progress bar.

## Keeping runs reliable

* **Two ways to get the engine.** Scheduled sessions attach this repo with `add_repo`; when a session lacks that tool they read the identical copy published with the dashboard (`engine/*` plus `engine/MANIFEST.json`). After any engine change run `python ops/publish_engine.py` and republish the dashboard with the printed files (`tests/test_engine_manifest.py` fails until the manifest matches the code). The daily review flags drift between the two.
* **Backups.** Every scheduled run has a backup check (see `ops/schedule.json`). Backup prompts embed their task's prompt, so after editing a template run `python ops/build_prompts.py --artifact <dashboard url> --out ops/generated` and update both the task and its backup (`tests/test_prompts.py` fails while `ops/generated` is stale).
* **Closed days and stale data.** Scheduled scans on weekends and holidays record a "closed" scan and stop. Intraday or pre-market prices from another day are never treated as today's; the digest and email headline start with a data warning whenever bars are behind.
* **Saved-or-not.** Every run re-reads what it wrote; scans resend their results once and warn in the email if they still didn't save.
* **Dashboard.** The Scans tab shows today's runs (ran / running / backup running / missed, with Run now), the data age and the engine version. Scan now retries a stopped scan once by itself and offers Try again.
* **Half days.** One-shot pre-close tasks exist for Nov 27 and Dec 24, 2026. Add one for each later half day (Nov 26 2027, Jul 3 2028, Nov 24 2028) before it comes.

## Running locally

```bash
pip install pandas numpy scikit-learn pyarrow
git clone --depth 1 -b data https://github.com/<you>/swing-desk data
python -m engine.calibrate backtest --data data --out bt      # needs history.parquet
python -m engine.scan --mode postclose --data data --out out  # writes out/digest.md
```
