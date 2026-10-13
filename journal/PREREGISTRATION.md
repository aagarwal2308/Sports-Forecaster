# Pre-registration: 2026 NFL live forecast test

**Written 2026-10-13, before any forecast from the frozen model was filed.**
After it is committed, this file must not be edited. Corrections go in a new dated section at the bottom.

> Forecasts are statistical estimates for research and education. This test measures forecast
> accuracy; it is not connected to, and does not recommend, betting.

## What is being tested

| Role | Model version | Settings |
|---|---|---|
| **Primary (frozen)** | `elo-1.0+208703b4` | K = 20, home field = 55, season reversion = 0.5 |
| Baseline | `home_rate-1.0+b8576a29` | Beta(10,10) prior |
| Baseline | `coin_flip-1.0+51e7c6c7` | 50/50 |

The `+xxxxxxxx` fingerprint covers each model's settings and source code. If code or settings
change, the version changes, and that new version is a *different* model with its own record.
**`elo-1.0+208703b4` will not be modified during the test.** Improved models may run alongside it
under their own versions, but they don't replace it.

## Period and data

- Games from **Week 7 (first kickoff Thu 2026-10-15)** through the Super Bowl, including playoffs.
- Data: nflverse `games.csv` (CC BY 4.0), downloaded with `forecast-lab ingest` before every forecast run.
- Routine: forecasts are filed on Tuesday (after Monday Night Football) and refreshed on Thursday before kickoff.
  A missed run is noted below. It cannot be made up afterwards, because the database refuses forecasts filed after kickoff.

## Scoring rule (fixed now)

1. Only forecasts with `predicted_at_utc` **before** `kickoff_utc` count.
2. For each game and model version, the **latest** pre-kickoff forecast is scored.
3. Ties are not scored, but they are counted.
4. The 15 `home_rate-1.0` records filed by v0.1 (no kickoff time stored) are excluded.

**Primary metric:** Brier score. **Secondary:** log loss, accuracy, and calibration (favorite's stated probability vs. how often favorites won).

## Expectations (written before seeing any results)

- **H1:** Elo's Brier is lower than the coin flip's (0.250) over the full test period.
- **H2:** Elo's Brier is lower than home_rate's over the full test period.
- **Expected level:** an Elo Brier of about 0.22 (validation 2016–2020: 0.2195, 95% interval 0.212–0.228)
  and accuracy of about 62–67%.
- **Sample-size caveat:** with about 190 games, the 95% interval on Brier will be about ±0.02. That is enough to test H1,
  maybe not H2, and certainly not small differences between variants.
- **Judgment is made at season end.** Weekly results are recorded, but no conclusion or model change is made from them.

## Known weaknesses going in

No injury or quarterback information. Elo reacts slowly to big in-season changes. And nothing about
roster moves is used.

## Log of deviations / missed runs
_(append dated entries below; never edit the sections above)_
