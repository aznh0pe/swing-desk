# Which A+ setups to show first on a crowded day (Nov 23, 2026)

The study behind `engine/scan.py: feature_block` (top picks: best score first, at most 2 per sector, at most 3 per
strategy, 8 at most; the first three are "place first").

Input: `trades.parquet` from a full backtest (`engine.calibrate.run_backtest`), here the Nov 14, 2026 run: 5,654 A+
setups, 2008-now; 146 days with 9 or more A+ setups (95 before 2022, 51 since). Run in this order from a folder
that holds the backtest output one level up (`../bt_base/trades.parquet`):

1. `prep.py`: the A+ setups, one per ticker and day.
2. `rules.py`, `terc.py`: does any single reading pick better setups inside a crowded day? (None did in both periods.)
3. `strat.py`: ranking by each strategy's earlier record, walk-forward; per-strategy caps.
4. `rank.py`: results by rank, what the per-strategy cap drops, and a near-duplicate (correlation) filter.

A rule was adopted only where it held both before 2022 and since. The live check of the rule runs every day in the
learning run (`engine.calibrate.pick_scorecard`).
