# Week 6 review: 2026 NFL live forecast test

*Written 2026-10-20 from data snapshot `75d7da6451ae` (downloaded 2026-10-20 09:14 ET).
All forecasts were filed before kickoff (most recently 2026-10-15 16:53 ET). Estimates for research, not betting advice.*

## Summary

- **Elo had a bad week:** favorites won **6 of 14** games, and Elo scored **worse than a coin flip** (Brier 0.275 vs 0.250).
- **It was unlucky, not broken.** If Elo's probabilities were exactly right, a week this bad would still happen about **1 time in 11**.
  In the 2016–2020 replay, **6%** of weeks were this bad or worse.
- **14 games settle nothing.** The 95% range on Elo's Brier this week runs from 0.185 to 0.376. It includes the pre-registered expectation (about 0.22),
  and it includes "worse than a coin flip". The verdict is made at season end, as pre-registered.
- **No change to the live model.** A pre-season idea, a learned home-field advantage, was tested on past seasons and **rejected**: see "Refinement" below.

## Every game

| Kickoff (ET) | Game | Final | Elo's favorite | Elo chance | Favorite won? |
|---|---|---|---|---|---|
| Thu 10-15 20:15 | SEA @ DEN | 14–17 | SEA | 51% | no |
| Sun 10-18 09:30 | HOU @ JAX (neutral) | 34–31 | JAX | 64% | no |
| Sun 10-18 13:00 | NYJ @ NE | 19–26 | NE | 79% | **yes** |
| Sun 10-18 13:00 | BAL @ CLE | 10–13 | BAL | 57% | no |
| Sun 10-18 13:00 | TEN @ IND | 26–20 | IND | 76% | no |
| Sun 10-18 13:00 | CAR @ PHI | 25–14 | PHI | 60% | no |
| Sun 10-18 13:00 | CHI @ ATL | 31–23 | ATL | 51% | no |
| Sun 10-18 13:00 | PIT @ TB | 26–19 | PIT | 53% | **yes** |
| Sun 10-18 13:00 | NO @ NYG | 34–27 | NYG | 67% | no |
| Sun 10-18 16:05 | ARI @ LA | 18–16 | LA | 78% | no |
| Sun 10-18 16:25 | LAC @ KC | 6–22 | KC | 76% | **yes** |
| Sun 10-18 16:25 | BUF @ LV | 33–30 | BUF | 65% | **yes** |
| Sun 10-18 20:20 | DAL @ GB | 13–23 | GB | 66% | **yes** |
| Mon 10-19 20:15 | WAS @ SF | 17–23 | SF | 82% | **yes** |

Elo expected its favorites to win about **9.3** of these 14; they won **6**. All eight upsets were decided by 11 points or fewer, and six of them by 7 points or fewer.
The biggest misses: ARI over LA (Elo gave ARI 22%) and TEN over IND (24%).

## Scorecard so far (14 scored games)

| Model | Brier (lower is better) [95%] | Accuracy |
|---|---|---|
| coin_flip | 0.250 | 50% |
| home_rate | 0.259 [0.226, 0.292] | 46% |
| **elo (frozen)** | **0.275** [0.185, 0.376] | 43% |

All three models had a poor week, because **home teams won only 6 of 14** (Elo's average forecast for home teams was 62.5%).
Small samples swing hard: one more home win would have changed the ordering.

## Is this surprising?

| Check | Result |
|---|---|
| If Elo's probabilities were exactly right, chance of a week at least this bad | **8.8%** (about 1 in 11) |
| ...chance that favorites win 6 or fewer of these 14 | **5.7%** |
| Elo's 2016–2020 replay: weeks worse than a coin flip | **19%** (about 1 week in 5) |
| Elo's 2016–2020 replay: weeks with Brier 0.275 or worse | **6%** (worst: 0.315) |

**Conclusion:** an unusual week, roughly a once-or-twice-a-season event, but well within what a well-calibrated model produces.
It is not evidence that the model broke.

## Refinement: what I tested, and why the live model stays unchanged

**Rule:** the pre-registration freezes `elo-1.0+208703b4` for the whole season. Improvements are tested on **past seasons only**.
Tuning on this weekend's 14 games would be fitting noise.

**Idea tested** (logged on 2026-10-13, *before* the season): the fixed home-field bonus (55 Elo points) may be too big for
recent seasons. Candidate `elo_hfa-0.1` lets the bonus learn from results, the same way team ratings do.

| Test | Result |
|---|---|
| Learning switched off | Identical to elo-1.0 (sanity check passed) |
| Train seasons 2002–2015 (choosing the learning speed) | **Worse at every learning speed** (0.21880–0.22028 vs 0.21873) |
| Validation seasons 2016–2020 | Tiny gain, about 0.0003 Brier; every 95% range includes zero |
| What it learned | The bonus *rose* to about 65 just before 2019–2020, when home advantage collapsed. It adapts too slowly to help. |

**Decision: rejected.** The code (`src/forecast_lab/elo_hfa.py`) and its tests are kept so the result can be reproduced.
It is not added to the live forecasts.

## What could actually improve the model

The model's biggest known blind spot remains **quarterback and injury information**. Elo only sees final scores.
This data set doesn't record who started at quarterback, so we can't tell whether starter changes
played a part in any of this week's upsets.
Next step: find a properly licensed source of starting-QB data (nflverse publishes player data; its license needs checking),
then build a QB-adjusted Elo as a **new** version, validated on 2016–2020 before it runs alongside the frozen model.

## Housekeeping

- Today's scheduled run (Tue 10-20, 09:14 ET) downloaded data, recorded every Week 6 result and filed Week 7 forecasts.
  It then paused waiting for a permission approval, so its commit hasn't happened yet.
- No forecast was missed. Every Week 6 game had a pre-kickoff forecast.
