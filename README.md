# Swing Desk

A rules-based swing-trading desk for a small account: it scans ~600 liquid US stocks and
ETFs several times a day, sends only A+ setups (limit entry, stop, target, size, hold
window, exit rule), watches open positions for early exits, and learns from every alert.

**Not financial advice.** These are model-generated signals from backtested rules. Win
percentages are estimates and can be wrong. You place and manage every order yourself.

## How it fits together

| Piece | Where | What it does |
|---|---|---|
| Market-data feed | `.github/workflows/market-data.yml`, `pipeline/` | GitHub Action, every 30 min 4 AM-3 PM PT on weekdays: daily bars (3 yrs), intraday snapshot, pre-market snapshot, earnings calendar; Saturday: full history since 2008. Writes the `data` branch (one squashed commit, no history bloat). |
| Engine | `engine/` | Indicators, 7 strategies, conservative backtester, win-probability model, scoring/grading, position sizing, exit monitor, daily learning, weekly recalibration. |
| Scans | Claude scheduled tasks | Pre-market 5:50 AM PT, pre-close 12:25 PM PT, post-close 2:05 PM PT, learning 3:12 PM PT, recalibration Sat 10:14 AM PT. Each clones this repo + the `data` branch, reads the dashboard state, runs the engine, news-checks the setups, writes alerts, and emails the digest. |
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
2022-present out-of-sample data (`python -m engine.calibrate backtest`). A strategy only
goes live if its in-sample win rate is at least 58% with profit factor >= 1.3.

## Running locally

```bash
pip install pandas numpy scikit-learn pyarrow
git clone --depth 1 -b data https://github.com/<you>/swing-desk data
python -m engine.calibrate backtest --data data --out bt      # needs history.parquet
python -m engine.scan --mode postclose --data data --out out  # writes out/digest.md
```
