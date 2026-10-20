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

## 2026-10-13: Calibration (finishes Milestone 2)
- Elo on validation: all 9 bins' 95% Wilson intervals include the diagonal; ECE 0.022.
- Slight lean in the 0.3–0.6 bins (home teams won a bit less than predicted). Hypothesis: the fixed home-field
  bonus (55) is too large for the post-2019 era. To test as a NEW model version, not by changing elo-1.0+208703b4.
- home_rate ECE 0.018 is "better" only because it never says anything but 57–58%. ECE must be read alongside Brier.

## 2026-10-20: Week 6 review and the learned home-field test
- Live Week 6: Elo Brier 0.275 on 14 games (coin flip 0.250, home_rate 0.259); favorites won 6/14 vs 9.3 expected.
  Under Elo's own probabilities, a week this bad happens 8.8% of the time; 6% of 2016–2020 weeks were as bad. Not evidence of a broken model.
- Tested the 2026-10-13 hypothesis (fixed home bonus too big) as candidate elo_hfa-0.1 (home bonus learns from results).
  Train 2002–2015: worse at every learning speed. Validation 2016–2020: about 0.0003 better, every 95% range includes 0. The learned bonus
  rose before the 2019–20 drop, so it adapts too slowly. **Rejected**; code kept for reproducibility. Live model unchanged.
- Note: this candidate was designed after seeing Week 6, but it was scored only on 2002–2020 data. Week 6 was not used to tune anything.
- Next idea: QB-adjusted Elo (needs a licensed starting-QB data source).
- Full report: reports/2026_week06_review.md
