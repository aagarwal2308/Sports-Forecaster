"""Milestone 1 baseline models. Deliberately simple: they are the bar every
later model must beat on held-out data.

coin_flip   P(home win) = 0.5 for every game. "Knows nothing."
home_rate   P(home win) = league-wide home win rate over all completed games
            played on EARLIER dates, shrunk toward 0.5 by a Beta prior.
            Neutral-site games get 0.5 (no home team advantage).

Outcome space: {home_win, away_win}. NFL ties (~0.2% of games) are not
modeled yet; they are excluded from the rate and from scoring, and counted
in every evaluation report so the exclusion stays visible.

No-leakage rule: a prediction for a game on date D may only use games
completed before D. Games on the same date never see each other.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys

import numpy as np
import pandas as pd

MODEL_VERSIONS = {
    "coin_flip": "coin_flip-1.0",
    "home_rate": "home_rate-1.0",
    "elo": "elo-1.0",
}

PREDICTION_COLUMNS = [
    "game_id", "season", "week", "gameday", "kickoff_utc", "game_type", "status",
    "model_version", "p_home_win", "p_away_win",
    "features_json", "warnings_json", "actual_outcome",
]


# Which module holds each model's code (baseline.py itself is always included).
_MODEL_MODULES = {"elo": "forecast_lab.elo"}


def model_version(model: str, params: dict | None = None) -> str:
    """Readable name + fingerprint, e.g. 'elo-1.0+3f9a1c2e'.

    The fingerprint hashes the model's settings AND the source code of the
    files that compute it. Change a setting or edit the code -> new version,
    so a live forecast record can never silently mix two different models.
    """
    source = inspect.getsource(sys.modules[__name__])
    if model in _MODEL_MODULES:
        source += inspect.getsource(importlib.import_module(_MODEL_MODULES[model]))
    blob = json.dumps({"model": model, "params": params or {}}, sort_keys=True) + source
    return f"{MODEL_VERSIONS[model]}+{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def _kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """Kickoff from the clean data; if absent, the conservative start of gameday (US Eastern)."""
    if "kickoff_utc" in games.columns:
        return pd.to_datetime(games["kickoff_utc"], utc=True)
    day_start = pd.to_datetime(games["gameday"]).dt.tz_localize("America/New_York")
    return day_start.dt.tz_convert("UTC")


def _row_warnings(games: pd.DataFrame) -> list[list[str]]:
    return [[w for w in str(s).split(";") if w] for s in games["dq_warnings"]]


def _finish(games: pd.DataFrame, model: str, p_home: np.ndarray, features: list[dict],
            warnings: list[list[str]], params: dict | None = None) -> pd.DataFrame:
    out = games[["game_id", "season", "week", "gameday", "game_type", "status"]].copy()
    out["kickoff_utc"] = _kickoff_utc(games).to_numpy()
    out["model_version"] = model_version(model, params)
    out["p_home_win"] = p_home.astype(float)
    out["p_away_win"] = 1.0 - out["p_home_win"]
    out["features_json"] = [json.dumps(f, sort_keys=True) for f in features]
    out["warnings_json"] = [json.dumps(w) for w in warnings]
    # The real result is attached only for completed games; it is never an input.
    out["actual_outcome"] = games["outcome"].where(games["status"] == "final", pd.NA)
    return out[PREDICTION_COLUMNS].reset_index(drop=True)


def predict_coin_flip(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return _finish(games, "coin_flip", np.full(len(games), 0.5), [{} for _ in range(len(games))], _row_warnings(games))


def predict_home_rate(
    games: pd.DataFrame,
    prior_home_wins: float = 10.0,
    prior_games: float = 20.0,
    min_games_warning: int = 50,
) -> pd.DataFrame:
    """Walk-forward home-win-rate baseline.

    The prior acts like `prior_games` imaginary games with `prior_home_wins`
    home wins (default Beta(10,10) -> starts at 0.5 and moves as evidence
    arrives). This guards against wild estimates from tiny samples.
    """
    g = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    decided = g["outcome"].isin(["home_win", "away_win"])  # excludes ties and unplayed
    eligible = (g["status"] == "final") & ~g["neutral_site"] & decided
    home_won = eligible & g["outcome"].isin(["home_win"])

    daily = (
        pd.DataFrame({"gameday": g["gameday"], "wins": home_won.astype(int), "games": eligible.astype(int)})
        .groupby("gameday").sum()
    )
    # cumsum through each date, shifted one date down -> totals strictly BEFORE that date
    before = daily.cumsum().shift(1, fill_value=0)
    wins_before = g["gameday"].map(before["wins"]).to_numpy()
    games_before = g["gameday"].map(before["games"]).to_numpy()

    rate = (wins_before + prior_home_wins) / (games_before + prior_games)
    p_home = np.where(g["neutral_site"].to_numpy(), 0.5, rate)

    features, warnings = [], _row_warnings(g)
    for i in range(len(g)):
        features.append({
            "home_wins_before": int(wins_before[i]),
            "games_before": int(games_before[i]),
            "prior": f"Beta({prior_home_wins:g},{prior_games - prior_home_wins:g})",
            "neutral_site": bool(g.at[i, "neutral_site"]),
        })
        if games_before[i] < min_games_warning:
            warnings[i].append("few_prior_games")
    params = {"prior_home_wins": prior_home_wins, "prior_games": prior_games, "min_games_warning": min_games_warning}
    return _finish(g, "home_rate", p_home, features, warnings, params)

