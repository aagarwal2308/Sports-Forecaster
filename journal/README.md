# Research journal

- `experiments.jsonl` is written automatically: one line per `backtest` / `forecast` run,
  with the run id, data snapshot hash, code version, and metrics.
- Add your own dated notes below: what you tried, what you expected, what happened, and why.
  Write down hypotheses **before** running an experiment.

## 2026-10-13: Milestone 1 baselines
- Expectation: home_rate beats coin_flip slightly, because home teams win about 55–58% of games.
- Result (validation 2016–2020): Brier 0.2470 vs 0.2500. It beat the coin flip in 2016–18 and lost in 2019–20.
- Why: home-win share fell to 52% (2019) and 50% (2020, mostly empty stadiums). The
  all-history average was slow to adapt. Question for M2: should home advantage use a recent window?

## 2026-10-13: Milestone 2, Elo
- Settings search on TRAIN 2002–2015: best K=20, HFA=55, revert=0.5 (Brier 0.2187). The 538-style defaults
  (20/48/0.33) scored 0.2192, a difference that is basically noise.
- Validation: Brier 0.2195 vs home_rate 0.2470, accuracy 64.4%. Elo beat home_rate in all five seasons.
- Division map checked against the source's div_game flag for every 2002–2026 regular-season game: 0 mismatches.

## 2026-10-13: Milestone 3, season simulator
- Hypothesis: Elo-based simulation should beat a standings-only coin flip.
- Result (validation, from week 8): playoff Brier 0.112 vs 0.119, wins MAE 1.39 vs 1.65. Better at weeks 4 and 12 too, but the intervals overlap.
- Decision made on validation: "hot" sims (ratings updating inside each run) put 90% of actual win totals inside the
  10–90% range, so the ranges were too wide. "Cold" put 81% inside (target 80%) with the same Brier. Default is now cold.
- Open question: in-season Elo is fairly static. Would a QB-adjusted rating help? (Needs a player-data source.)

## 2026-10-13: Launching the live test (v0.2.0)
- Added: UTC kickoff times and a kickoff guard; model fingerprints; all models filed together; `scorecard`; CSV export for git.
- Re-ran the validation backtest after these changes: identical numbers (elo Brier 0.2195), so the probabilities are unchanged.
- Wrote PREREGISTRATION.md, then filed the first forecasts: run 20261013T054836Z-2812c7, 15 games (Oct 15–22) × 3 models.
- Noticed the source alternating between two versions within the hour (sha f2bbc6… vs bbaf0b…). The only difference was
  one game's betting moneyline, which the cleaner drops, so model inputs were identical. Snapshots handle this by design.
