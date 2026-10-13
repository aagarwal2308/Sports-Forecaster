"""AT-8: live forecasts are scored by the pre-registered rule."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from forecast_lab.scorecard import eligible_forecasts, scorecard
from forecast_lab.store import record_outcomes, save_predictions


def preds(p_home, kickoff="2026-10-18T17:00:00Z", game="G1"):
    """SYNTHETIC prediction rows for one scheduled game."""
    return pd.DataFrame([{
        "game_id": game, "season": 2026, "week": 7, "gameday": "2026-10-18", "kickoff_utc": pd.Timestamp(kickoff),
        "game_type": "REG", "status": "scheduled", "model_version": "elo-1.0+test",
        "p_home_win": p_home, "p_away_win": 1 - p_home, "features_json": "{}", "warnings_json": "[]",
        "actual_outcome": pd.NA}])


def t(day, hour):
    return datetime(2026, 10, day, hour, tzinfo=timezone.utc)


def test_latest_pre_kickoff_forecast_is_scored(tmp_path):
    db = tmp_path / "lab.sqlite"
    save_predictions(preds(0.60), "prospective", "r1", "snapA", db, now=t(13, 6))
    save_predictions(preds(0.70), "prospective", "r2", "snapB", db, now=t(16, 6))   # newer data, still pre-game
    record_outcomes(pd.DataFrame({"game_id": ["G1"], "status": ["final"], "outcome": ["home_win"]}), db, now=t(19, 6))
    latest, notes = eligible_forecasts(db)
    assert len(latest) == 1 and latest["p_home_win"].iloc[0] == pytest.approx(0.70)
    table, _ = scorecard(db, n_boot=50)
    row = table.iloc[0]
    assert (row["scored"], row["pending"]) == (1, 0)
    assert row["brier"] == pytest.approx(0.09)


def test_pending_and_tie_games_are_counted_not_scored(tmp_path):
    db = tmp_path / "lab.sqlite"
    save_predictions(pd.concat([preds(0.6, game="G1"), preds(0.6, game="G2")]), "prospective", "r1", "s", db, now=t(13, 6))
    record_outcomes(pd.DataFrame({"game_id": ["G1"], "status": ["final"], "outcome": ["tie"]}), db, now=t(19, 6))
    row = scorecard(db, n_boot=50)[0].iloc[0]
    assert (row["scored"], row["ties"], row["pending"]) == (0, 1, 1)
