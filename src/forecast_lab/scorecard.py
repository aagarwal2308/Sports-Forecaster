"""Score live (prospective) forecasts once games are final.

Scoring rule (pre-registered in journal/PREREGISTRATION.md):
  * Only forecasts whose predicted_at_utc is BEFORE kickoff_utc count.
  * For each game and model version, the LATEST such forecast is scored
    (it used the freshest information available before kickoff).
  * Ties are not scored (the models don't predict them) but are counted.
  * Records without a kickoff time (filed by v0.1, before kickoff checks
    existed) can't be verified as pre-game, so they're excluded and counted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DB_PATH
from .metrics import METRICS, bootstrap_ci
from .store import load_predictions


def eligible_forecasts(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, dict]:
    df = load_predictions(db_path, "prospective")
    notes = {"unverifiable_no_kickoff": int(df["kickoff_utc"].isna().sum())}
    df = df[df["kickoff_utc"].notna()].copy()
    df["predicted_at"] = pd.to_datetime(df["predicted_at_utc"], utc=True, format="ISO8601")
    df["kickoff"] = pd.to_datetime(df["kickoff_utc"], utc=True, format="ISO8601")
    after = df["predicted_at"] >= df["kickoff"]
    notes["filed_after_kickoff"] = int(after.sum())  # should always be 0; the database refuses these
    df = df[~after]
    latest = df.sort_values("predicted_at").groupby(["game_id", "model_version"], as_index=False).last()
    return latest, notes


def scorecard(db_path: Path = DB_PATH, n_boot: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    latest, notes = eligible_forecasts(db_path)
    rows = []
    for version, d in latest.groupby("model_version"):
        decided = d[d["actual_outcome"].isin(["home_win", "away_win"])]
        row = {"model_version": version, "forecast_games": len(d), "scored": len(decided),
               "pending": int(d["actual_outcome"].isna().sum()), "ties": int((d["actual_outcome"] == "tie").sum())}
        if len(decided):
            y = (decided["actual_outcome"] == "home_win").to_numpy(dtype=float)
            p = decided["p_home_win"].to_numpy(dtype=float)
            for name, fn in METRICS.items():
                row[name] = fn(y, p)
                row[f"{name}_ci95"] = bootstrap_ci(fn, y, p, n_boot=n_boot, seed=seed) if len(decided) >= 10 else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows), notes


def calibration_table(db_path: Path = DB_PATH, model_prefix: str = "elo-") -> pd.DataFrame:
    """Group forecasts by the probability given to the FAVORITE; compare with how often favorites won."""
    latest, _ = eligible_forecasts(db_path)
    d = latest[latest["model_version"].str.startswith(model_prefix) & latest["actual_outcome"].isin(["home_win", "away_win"])]
    if d.empty:
        return pd.DataFrame()
    fav_p = np.maximum(d["p_home_win"], d["p_away_win"])
    fav_won = np.where(d["p_home_win"] >= 0.5, d["actual_outcome"] == "home_win", d["actual_outcome"] == "away_win")
    bins = pd.cut(fav_p, [0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    return (pd.DataFrame({"bin": bins, "p": fav_p, "won": fav_won})
            .groupby("bin", observed=True).agg(games=("won", "size"), said=("p", "mean"), happened=("won", "mean")))
