# Multi-Sport AI Forecasting Lab: Plan (First Deliverable)

*Written 2026-10-13. Data sources were checked on that date; re-check before relying on them.*

> This is an educational research simulator. It outputs probability **estimates**.
> It does not place bets, connect to betting accounts, or recommend wagers.

---

## 1. Recommended first sport: NFL (American football)

| Reason | Detail |
|---|---|
| **Cleanest licensed data** | One tidy file (`games.csv`) from nflverse covers every game since 1999, and the nflverse-data repo is licensed **CC BY 4.0** (we checked its `LICENSE.md`). No scraping, no API keys. |
| **Manageable problem** | 32 teams, about 285 games a season, and one game per team per week. The whole history is about 7,500 rows, which fits easily in pandas on a laptop. |
| **Live season right now** | The 2026 season is in Week 6. Each week you can file real **pre-game** forecasts and score them after the games, which is the most honest test a model can get. |
| **Teaches humility** | Small samples and noisy outcomes mean even good models land around 60–67% accuracy. You learn early why uncertainty intervals matter. |
| **Clear baseline effects** | Home advantage is measurable, and it *changed*: it nearly vanished in 2020 (empty stadiums) and has stayed weaker since. That's a good first lesson that the past doesn't always predict the future. |

**Trade-offs, stated honestly:** NFL seasons are short, so per-season metrics have wide intervals. Quarterback injuries swing results, and Milestone 1 can't see them. The NBA has about 4× more games, but its main free data path (`stats.nba.com` via `nba_api`) is unofficial, and SportsDataverse notes it needs TLS-fingerprint workarounds. We avoid that kind of access, so NBA comes later through a properly licensed source.

## 2. Free data-source assessment

"Verified" means I read the source's page or license file on 2026-10-13. "Not verified" means you must check it before use.

| Source | What | License / terms | Status | Decision |
|---|---|---|---|---|
| **nflverse-data releases**: `github.com/nflverse/nflverse-data/releases/download/schedules/games.csv` | NFL games 1999–present, including the scheduled 2026 games | Repo `LICENSE.md` is **CC BY 4.0** (attribution required). The nflreadr docs add that the underlying NFL data "belong to their respective owners" and is governed by their terms. | Verified. Download tested: 7,548 games. | **Primary source for Milestone 1** |
| nflverse/nfldata `data/games.csv` | A near-identical mirror of the same file | **No license stated** in that repo | Verified (no license) | Not used. We prefer the licensed release copy. |
| **nflreadpy** (Python port of nflreadr) | Loaders for schedules, player stats, rosters, and more | Code MIT. README says data is "broadly CC-BY 4.0", except FTN charting data, which is CC-BY-SA 4.0. Marked *experimental*; returns Polars data frames. | Verified (README) | Candidate for **Phase 5 player stats**. Re-verify then. |
| **SportsDataverse** (`sportsdataverse-py`) | Many leagues (NFL, NBA, college, NHL, MLB, soccer…) | Code **MIT**. Data comes mostly from ESPN and league APIs, and **no data terms are stated**, so each upstream provider's terms apply. Marked experimental. | Verified (README) | Useful for research leads. Use only datasets whose upstream terms you've checked. |
| `stats.nba.com` / `nba_api` | NBA stats | Unofficial endpoints. Needs anti-bot workarounds (per the SportsDataverse README). | Partly verified | **Avoid.** Bypassing bot protection is off-limits for this project. |
| Kaggle datasets | Many sports | Varies per dataset | Not verified | Case by case. Read each dataset's license. |
| Retrosheet (MLB), football-data.co.uk (soccer) | Historical results | Not checked this session | **Not verified** | Possible future sports. Verify first. |

**Note on betting columns:** `games.csv` includes betting-market lines. They stay in the raw snapshot (unchanged), but the cleaner **drops them** from the clean data. A market line could be a useful accuracy benchmark later, but that is a deliberate future decision for you to make, not a default.

## 3. Architecture

```
                    ┌──────────────────────────┐
  nflverse (CC BY)  │ sources.py               │  URL + license + attribution
  games.csv ───────▶│ ingest.py                │  download → raw snapshot (read-only)
                    └────────────┬─────────────┘  + manifest {sha256, time, license}
                                 ▼
                    data/raw/<source>/<time>_<sha>.csv  ← never edited
                                 │  load_snapshot() re-checks sha256
                                 ▼
                    ┌──────────────────────────┐
                    │ clean.py + teams.py      │  validate: errors → excluded + logged
                    │                          │            warnings → kept + flagged
                    └────────────┬─────────────┘  franchise ids (OAK→LV, SD→LAC, STL→LA)
                                 ▼
                    data/clean/<source>/<sha>/games.csv, issues.csv, manifest.json
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ models (baseline.py)     │  walk-forward: uses only games
                    │  M1 coin_flip, home_rate │  BEFORE each game's date
                    │  M2 elo   M3 logistic    │
                    └──────┬────────────┬──────┘
                           ▼            ▼
              ┌────────────────┐  ┌───────────────────────────┐
              │ metrics.py     │  │ store.py  (SQLite)        │
              │ Brier, logloss │  │ append-only predictions   │
              │ accuracy, CIs  │  │ 'backtest' vs 'prospective'│
              │ TEST SET LOCK  │  │ outcomes filled in later  │
              └───────┬────────┘  └─────────────┬─────────────┘
                      ▼                         ▼
              journal/experiments.jsonl   ──▶  Streamlit dashboard (M4)
```

## 4. Project directory

```
forecast_lab/
├── pyproject.toml            dependencies + the `forecast-lab` command
├── README.md                 install, usage, troubleshooting, file guide
├── docs/PLAN.md              this document
├── src/forecast_lab/
│   ├── config.py             paths + season splits (train / validation / LOCKED test)
│   ├── sources.py            registry of data sources with license info
│   ├── ingest.py             download → immutable raw snapshot + manifest
│   ├── teams.py              team codes → stable franchise ids
│   ├── clean.py              validation + standardization
│   ├── baseline.py           coin_flip + home_rate models           (M1)
│   ├── metrics.py            scoring, bootstrap intervals, test lock
│   ├── store.py              SQLite prediction records
│   ├── cli.py                command-line interface
│   ├── elo.py                Elo ratings                            (M2)
│   ├── models.py             model registry
│   ├── nfl_structure.py      divisions, conferences, playoff format
│   ├── simulate.py           Monte Carlo season simulator           (M3)
│   ├── calibration.py        reliability tables + plots             (M2)
│   ├── scorecard.py          live forecast scoring
│   ├── dashboard_app.py      Streamlit dashboard (read-only)        (M4)
│   ├── dashboard_data.py     read-only loaders for the dashboard
│   ├── logistic.py           scikit-learn model                     (M3, planned)
│   └── players/              player-performance module               (Phase 5, planned)
├── tests/                    pytest acceptance tests
├── journal/                  experiments.jsonl (automatic) + your notes
└── data/                     raw/, clean/, lab.sqlite (created at runtime)
```

## 5. Two-week beginner plan (about 1–2 hours a day)

| Day | Goal | Done when |
|---|---|---|
| 1 | Install; run tests; run `ingest` → `clean` → `backtest` | 39 tests pass; you see the validation table |
| 2 | Read `ingest.py` and `clean.py`; open the raw and clean CSVs side by side | You can explain why raw is never edited |
| 3 | Read `baseline.py` and `test_baseline.py`; write your first journal note | You can explain the leakage test in your own words |
| 4–5 | **M2: Elo ratings.** Per-franchise ratings, home bonus, K-factor, regression to the mean between seasons | Elo beats home_rate on *validation* Brier, or we write down why it doesn't |
| 6 | **M2: Calibration.** Reliability table plus a matplotlib plot | When the model says 70%, roughly 70% of those games are home wins (within the interval) |
| 7 | Weekly routine: `ingest`, `forecast`, record outcomes | Your first scored pre-game forecasts |
| 8–9 | **M3: Logistic regression** (scikit-learn) on Elo difference, rest difference, neutral site, and shrunk recent form. Fit on train, compare on validation | Paired comparison against Elo with intervals |
| 10 | Performance by season and by team; explain the wins and losses | Written explanation in the journal |
| 11–12 | **M4: Streamlit dashboard v1**: explorer, model comparison, calibration, records, data-quality panel | Runs locally; every forecast is labeled "estimate" |
| 13 | Pre-register the final model and settings in the journal | Choices written down *before* the test run |
| 14 | One-time `--unlock-test` evaluation; reflect | Test results recorded, with no tuning afterwards |

Phase 5 (player performance) starts after this, once the game model is stable.

## 6. Status
- Milestone 1 (data pipeline + baselines): **done**
- Milestone 2 (Elo + calibration plots): **done**
- Milestone 3 (season simulator): **done** (added on request; it replaces part of days 6–7)
- Streamlit dashboard (Milestone 4): **done**
- Next: logistic regression (must beat Elo on validation)
