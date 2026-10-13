"""Milestone 3: Monte Carlo season simulator.

Idea: we can't know how the rest of the season will go, but we have a
probability for every remaining game. So we "play" the rest of the season
thousands of times with random draws, and count how often each thing happens
(a team reaches 11 wins, wins its division, makes the playoffs...).

    games before `cutoff` with results -> fixed (they already happened)
    every other regular-season game    -> simulated from model probabilities

Ratings are frozen within each simulated season by default ("cold").
"Hot" mode (hot=True) keeps updating Elo after every simulated game, so a
team that gets lucky early is rated higher for the rest of that simulation.
Validation check (2016-2020, from week 8): cold 10-90% win ranges held 81%
of actual results (target 80%); hot held 90%, meaning its ranges were too
wide. Playoff Brier was the same (0.112). Hence cold is the default.

What this does NOT capture (be honest when reading results):
  * NFL tiebreakers: teams level on wins are separated AT RANDOM, not by
    head-to-head, division record, etc.
  * Ties (about 0.2% of games) are not simulated.
  * Injuries, trades, and rest decisions late in the season.
  * Uncertainty in the Elo parameters themselves.
The `mc_se` column is Monte Carlo noise only (from using finite simulations),
not the model's total uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .elo import EloParams, run_elo, win_probability
from .metrics import bootstrap_ci, brier, check_test_lock
from .nfl_structure import DIVISIONS, FIRST_SEASON, TEAM_DIVISION, TEAMS, check_divisions, conference_of, wild_cards

SIM_MODELS = ("elo", "coin_flip")


@dataclass
class SimulationResult:
    table: pd.DataFrame  # one row per team
    meta: dict           # everything needed to reproduce / interpret the run


def _season_ratings(games: pd.DataFrame, season: int, cutoff: date, params: EloParams):
    """Elo state using only games completed before `cutoff`, moved into `season`."""
    _, state = run_elo(games[games["gameday"] < cutoff], params)
    for t in TEAMS:
        state.start_season(t, season, params)
    return state


def simulate_season(
    games: pd.DataFrame,
    season: int,
    cutoff: date,
    model: str = "elo",
    params: EloParams = EloParams(),
    n_sims: int = 10_000,
    seed: int = 0,
    hot: bool = False,
) -> SimulationResult:
    if season < FIRST_SEASON:
        raise ValueError(f"Simulator supports seasons from {FIRST_SEASON} (current division structure).")
    if model not in SIM_MODELS:
        raise ValueError(f"model must be one of {SIM_MODELS}")

    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    known = reg[(reg["gameday"] < cutoff) & (reg["status"] == "final")]
    remaining = reg[~reg["game_id"].isin(known["game_id"])].sort_values(["gameday", "game_id"])

    warnings = check_divisions(games, season)[:5]
    unresolved_past = int((remaining["gameday"] < cutoff).sum())
    if unresolved_past:
        warnings.append(f"{unresolved_past} game(s) dated before the cutoff have no result and were simulated")

    idx = {t: i for i, t in enumerate(TEAMS)}
    n_teams = len(TEAMS)
    w, l, t = np.zeros(n_teams), np.zeros(n_teams), np.zeros(n_teams)
    for r in known.itertuples(index=False):
        h, a = idx[r.home_franchise], idx[r.away_franchise]
        if r.outcome == "home_win":
            w[h] += 1; l[a] += 1
        elif r.outcome == "away_win":
            w[a] += 1; l[h] += 1
        else:
            t[h] += 1; t[a] += 1

    state = _season_ratings(games, season, cutoff, params)
    ratings_now = np.array([state.rating(team) for team in TEAMS])

    rng = np.random.default_rng(seed)
    R = np.tile(ratings_now, (n_sims, 1))
    wins = np.tile(w + 0.5 * t, (n_sims, 1))  # standings points: a tie counts as half a win
    step = params.k * state.mean_margin_multiplier

    for r in remaining.itertuples(index=False):
        h, a = idx[r.home_franchise], idx[r.away_franchise]
        if model == "elo":
            p = win_probability(R[:, h] - R[:, a] + (0.0 if r.neutral_site else params.home_field))
        else:
            p = np.full(n_sims, 0.5)
        home_won = rng.random(n_sims) < p
        wins[:, h] += home_won
        wins[:, a] += ~home_won
        if hot and model == "elo":
            delta = step * (home_won - p)
            R[:, h] += delta
            R[:, a] -= delta

    # --- standings: random tiebreak among teams level on wins -------------------
    score = wins + rng.random((n_sims, n_teams)) * 1e-3
    rows = np.arange(n_sims)
    div_winner = np.zeros((n_sims, n_teams), dtype=bool)
    for members in DIVISIONS.values():
        cols = np.array([idx[m] for m in members])
        div_winner[rows, cols[np.argmax(score[:, cols], axis=1)]] = True

    playoffs = div_winner.copy()
    top_seed = np.zeros((n_sims, n_teams), dtype=bool)
    for conf in ("AFC", "NFC"):
        cols = np.array([idx[x] for x in TEAMS if conference_of(x) == conf])
        sc, dw = score[:, cols], div_winner[:, cols]
        wc = np.argsort(-np.where(dw, -np.inf, sc), axis=1)[:, : wild_cards(season)]
        playoffs[rows[:, None], cols[wc]] = True
        top_seed[rows, cols[np.argmax(np.where(dw, sc, -np.inf), axis=1)]] = True

    p_playoffs = playoffs.mean(axis=0)
    table = pd.DataFrame({
        "team": TEAMS,
        "conference": [conference_of(x) for x in TEAMS],
        "division": [TEAM_DIVISION[x] for x in TEAMS],
        "wins": w.astype(int), "losses": l.astype(int), "ties": t.astype(int),
        "elo_now": ratings_now.round(1),
        "mean_wins": wins.mean(axis=0),
        "wins_p10": np.quantile(wins, 0.10, axis=0),
        "wins_p50": np.quantile(wins, 0.50, axis=0),
        "wins_p90": np.quantile(wins, 0.90, axis=0),
        "p_division": div_winner.mean(axis=0),
        "p_playoffs": p_playoffs,
        "p_top_seed": top_seed.mean(axis=0),
        "mc_se_playoffs": np.sqrt(p_playoffs * (1 - p_playoffs) / n_sims),
    }).sort_values(["conference", "division", "mean_wins"], ascending=[True, True, False]).reset_index(drop=True)

    meta = {
        "season": season, "cutoff": str(cutoff), "model": model, "hot": hot, "n_sims": n_sims, "seed": seed,
        "elo_params": params.to_dict(), "sim_rating_step": round(step, 3),
        "games_known": len(known), "games_simulated": len(remaining), "warnings": warnings,
        "limitations": "random tiebreaks; no ties simulated; no injuries; parameter uncertainty not included",
    }
    return SimulationResult(table, meta)


def actual_outcomes(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """What really happened: regular-season wins (ties = 1/2) and playoff qualification
    (a team made the playoffs if it appears in any postseason game that season)."""
    reg = games[(games["season"] == season) & (games["game_type"] == "REG") & (games["status"] == "final")]
    pts = {x: 0.0 for x in TEAMS}
    for r in reg.itertuples(index=False):
        hp = {"home_win": 1.0, "away_win": 0.0, "tie": 0.5}[r.outcome]
        pts[r.home_franchise] += hp
        pts[r.away_franchise] += 1 - hp
    post = games[(games["season"] == season) & (games["game_type"] != "REG")]
    in_post = set(post["home_franchise"]) | set(post["away_franchise"])
    return pd.DataFrame({"team": TEAMS, "actual_wins": [pts[x] for x in TEAMS],
                         "made_playoffs": [x in in_post for x in TEAMS]})


def backtest_simulator(games: pd.DataFrame, seasons, after_week: int, params: EloParams = EloParams(),
                       n_sims: int = 2000, seed: int = 0, unlock_test: bool = False) -> pd.DataFrame:
    """Re-run the simulator as it would have run after `after_week` of past seasons,
    using only information available then, and attach what actually happened."""
    seasons = list(seasons)
    check_test_lock(seasons, unlock_test)
    rows = []
    for s in seasons:
        reg = games[(games["season"] == s) & (games["game_type"] == "REG")]
        cutoff = reg.loc[reg["week"] > after_week, "gameday"].min()
        actual = actual_outcomes(games, s)
        for model in SIM_MODELS:
            res = simulate_season(games, s, cutoff, model=model, params=params, n_sims=n_sims, seed=seed)
            rows.append(res.table.merge(actual, on="team").assign(season=s, model=model, cutoff=cutoff))
    return pd.concat(rows, ignore_index=True)


def summarize_backtest(bt: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    out = []
    for model, d in bt.groupby("model"):
        y = d["made_playoffs"].to_numpy(dtype=float)
        p = d["p_playoffs"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(brier, y, p, n_boot=n_boot, seed=seed)
        err = np.abs(d["mean_wins"] - d["actual_wins"]).to_numpy()
        out.append({"model": model, "team_seasons": len(d), "playoff_brier": brier(y, p),
                    "playoff_brier_ci95": (round(lo, 4), round(hi, 4)),
                    "wins_mae": err.mean(),
                    "actual_wins_inside_p10_p90": float(((d["actual_wins"] >= d["wins_p10"]) &
                                                         (d["actual_wins"] <= d["wins_p90"])).mean())})
    return pd.DataFrame(out)
