# Prompt rollout, Nov 11 2026 (engine-copy-first setup, ISSUES reporting, phone alert)
Each scheduled task must carry the matching file in ops/generated/. Done = updated with update_trigger.
- [x] postclose  trig_01VJiBMjH2dJJnBrAaTr9BcX
- [x] premarket  trig_01BC6UyGsrA3mesQsU6pFbRP
- [x] preclose   trig_015vu682N2v147NyT2kAdSLJ
- [x] weekend    trig_01XGcnEVgDkEYRGBoFe9xijx
- [x] manual     trig_01812Le9m7LTp5vm9HhfZSvN
- [x] learning   trig_01V9Hsf7VTMtsod6AoHQnyiP
- [x] weekly     trig_01QZiuM65EDHm187zYcSY7Kk
- [x] halfday Nov 27  trig_01Au1pGF52XisNisf1Y94NTW (preclose_halfday.md)
- [x] halfday Dec 24  trig_01KsB1UAjVRRvPmyWFLo6BgE (preclose_halfday.md)
- [x] backup premarket trig_01WA17hiGscrK1sLsnnpwjEW
- [x] backup preclose  trig_01G67H8Efdaw2V1wPGevQH4X
- [x] backup postclose trig_01PY6eHAikqDnNCUU4QR2q51
- [x] backup learning  trig_01GShKJnc3qaLTWHnZzqHg7o
- [x] backup weekly    trig_01XCwqgr1TMFQuHDDY1drXFa
- [x] backup weekend   trig_01TYGCAK2FcGJmaF9wXMKPLE
All 15 scheduled tasks carry the Nov 11 prompts (finished Nov 11, 7:53 AM PT).

## Nov 11 midday: two grades (A+ and weekly top-ups) tracked, tested and tuned separately
Only these four prompts changed (daily review exports scans for watchlist grading and follows the research "focus";
the recalibration journals the top-ups on their own): all updated 10:35 AM PT.
- [x] learning        trig_01V9Hsf7VTMtsod6AoHQnyiP
- [x] backup learning trig_01GShKJnc3qaLTWHnZzqHg7o
- [x] weekly          trig_01QZiuM65EDHm187zYcSY7Kk
- [x] backup weekly   trig_01XCwqgr1TMFQuHDDY1drXFa
Scan prompts are unchanged (the scanner itself carries the new top-up rule). Live params re-tuned with
`python3 -m engine.calibrate topups` (strategy/params v5); learning/stats rewritten with the per-grade record (v4).

## Nov 13 2026: pre-market check moved earlier, late-start guard

The scheduler left the 5:50 AM pre-market check and its 6:02 AM backup waiting past 6:26 AM. Both tasks now
run earlier, and the pre-market prompt starts with an "already done?" step (`PREMARKET_GUARD` in
build_prompts.py), so a run that starts late after a backup or a dashboard catch-up stops with one line.
- [x] premarket         trig_01BC6UyGsrA3mesQsU6pFbRP  5:17 AM PT (was 5:50), prompt regenerated
- [x] backup premarket  trig_01WA17hiGscrK1sLsnnpwjEW  5:47 AM PT (was 6:02), prompt regenerated
Other prompts are unchanged.
