"""AT-5: forecasts made before an event are kept separate from hindsight analyses."""

import sqlite3
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from conftest import synthetic_clean_games
from forecast_lab.baseline import predict_home_rate
from forecast_lab.store import ProspectiveForecastError, connect, load_predictions, record_outcomes, save_predictions

NOW = datetime(2011, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def games_with_future():
    g = synthetic_clean_games(n_dates=20)  # synthetic; move the last date's games after NOW, unplayed
    last = g["gameday"] == g["gameday"].max()
    g.loc[last, "gameday"] = date(2011, 1, 9)
    g.loc[last, ["status", "outcome"]] = ["scheduled", pd.NA]
    return g


def test_backtest_records_store_all_required_fields(tmp_path, games_with_future):
    db = tmp_path / "lab.sqlite"
    preds = predict_home_rate(games_with_future)
    assert save_predictions(preds, "backtest", "run1", "abc123", db, now=NOW) == len(preds)
    rec = load_predictions(db)
    for col in ("game_id", "predicted_at_utc", "model_version", "p_home_win", "p_away_win",
                "features_json", "warnings_json", "actual_outcome", "data_snapshot_sha256", "prediction_kind"):
        assert col in rec.columns
    assert (rec["prediction_kind"] == "backtest").all()


def test_prospective_forecast_refused_for_played_games(tmp_path, games_with_future):
    preds = predict_home_rate(games_with_future)
    with pytest.raises(ProspectiveForecastError):
        save_predictions(preds, "prospective", "run1", "abc123", tmp_path / "db.sqlite", now=NOW)


def test_prospective_lifecycle_and_outcome_recording(tmp_path, games_with_future):
    db = tmp_path / "lab.sqlite"
    preds = predict_home_rate(games_with_future)
    upcoming = preds[preds["status"] == "scheduled"]
    assert save_predictions(upcoming, "prospective", "run1", "abc123", db, now=NOW) == len(upcoming)
    assert save_predictions(upcoming, "prospective", "run2", "abc123", db, now=NOW) == 0  # no overwrite
    assert load_predictions(db, "prospective")["actual_outcome"].isna().all()

    finished = games_with_future.copy()  # the games are later played
    finished.loc[finished["status"] == "scheduled", ["status", "outcome"]] = ["final", "away_win"]
    later = datetime(2011, 1, 10, tzinfo=timezone.utc)
    assert record_outcomes(finished, db, now=later) == len(upcoming)
    rec = load_predictions(db, "prospective")
    assert (rec["actual_outcome"] == "away_win").all()
    assert (rec["predicted_at_utc"] < rec["outcome_recorded_at_utc"]).all()  # forecast came first


def test_stored_forecasts_cannot_be_edited(tmp_path, games_with_future):
    db = tmp_path / "lab.sqlite"
    save_predictions(predict_home_rate(games_with_future), "backtest", "run1", "abc123", db, now=NOW)
    with pytest.raises(sqlite3.IntegrityError, match="cannot be edited"):
        with connect(db) as con:
            con.execute("UPDATE predictions SET p_home_win = 0.99")


def test_forecast_after_kickoff_on_game_day_is_refused(tmp_path, games_with_future):
    preds = predict_home_rate(games_with_future)
    upcoming = preds[preds["status"] == "scheduled"].copy()
    upcoming["kickoff_utc"] = pd.Timestamp("2011-01-09T18:00:00Z")
    db = tmp_path / "lab.sqlite"
    before = datetime(2011, 1, 9, 17, 59, tzinfo=timezone.utc)
    after = datetime(2011, 1, 9, 18, 1, tzinfo=timezone.utc)   # same date, 1 minute after kickoff
    assert save_predictions(upcoming, "prospective", "r1", "abc", db, now=before) == len(upcoming)
    with pytest.raises(ProspectiveForecastError):
        save_predictions(upcoming, "prospective", "r2", "def", db, now=after)
    assert (load_predictions(db, "prospective")["kickoff_utc"] == "2011-01-09T18:00:00Z").all()


def test_old_database_is_migrated_without_losing_rows(tmp_path):
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)  # a v0.1-style table without kickoff_utc
    con.execute("""CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
        prediction_kind TEXT NOT NULL, game_id TEXT NOT NULL, season INTEGER NOT NULL, gameday TEXT NOT NULL,
        predicted_at_utc TEXT NOT NULL, data_snapshot_sha256 TEXT NOT NULL, model_version TEXT NOT NULL,
        p_home_win REAL NOT NULL, p_away_win REAL NOT NULL, features_json TEXT NOT NULL, warnings_json TEXT NOT NULL,
        actual_outcome TEXT, outcome_recorded_at_utc TEXT)""")
    con.execute("INSERT INTO predictions VALUES (NULL,'r','prospective','G1',2026,'2026-10-15','t','s','m',0.5,0.5,'{}','[]',NULL,NULL)")
    con.commit(); con.close()
    rec = load_predictions(db)
    assert len(rec) == 1 and "kickoff_utc" in rec.columns and pd.isna(rec.loc[0, "kickoff_utc"])
