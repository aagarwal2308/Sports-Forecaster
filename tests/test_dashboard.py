"""AT-10: the dashboard reads data correctly, never shows locked test seasons,
and renders without errors (with data and with an empty data folder)."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import raw_csv_bytes, raw_row, synthetic_league_season
from forecast_lab import dashboard_data as dd
from forecast_lab.clean import clean_snapshot, load_clean_games
from forecast_lab.config import TEST_SEASONS
from forecast_lab.elo import predict_elo
from forecast_lab.ingest import snapshot_source
from forecast_lab.simulate import simulate_season
from forecast_lab.sources import NFLVERSE_SCHEDULES
from forecast_lab.store import record_outcomes, save_predictions

APP = str(Path(__file__).resolve().parents[1] / "src" / "forecast_lab" / "dashboard_app.py")
FILED = datetime(2020, 9, 1, tzinfo=timezone.utc)


@pytest.fixture
def lab(tmp_path):
    """A SYNTHETIC lab folder: one cleaned snapshot, live + backtest records, a saved simulation."""
    data = tmp_path / "data"
    rows = [
        raw_row("2020_01_KC_BUF", "2020-09-13", "KC", "BUF", 20, 27),
        raw_row("2020_01_OAK_DEN", "2020-09-13", "OAK", "DEN", 24, 17),
        raw_row("2020_02_NE_MIA", "2020-09-20", "NE", "MIA", 10, 13),
        raw_row("2020_03_GB_CHI", "2020-09-27", "GB", "CHI", 21, 21),   # a tie
        raw_row("2020_05_DAL_NYG", "2020-10-11", "DAL", "NYG"),         # not played yet
    ]
    manifest, _ = snapshot_source(NFLVERSE_SCHEDULES, data / "raw", lambda url: raw_csv_bytes(rows),
                                  now=datetime(2020, 10, 1, tzinfo=timezone.utc))
    cm = clean_snapshot(manifest, data / "raw", data / "clean")
    games = load_clean_games(Path(cm["path"]) / "games.csv")
    db = data / "lab.sqlite"

    before = games.copy()  # how the schedule looked on Sept 1, before any game was played
    before.loc[:, ["status", "outcome"]] = ["scheduled", pd.NA]
    save_predictions(predict_elo(before), "prospective", "r1", manifest["sha256"], db, now=FILED)
    record_outcomes(games, db, now=datetime(2020, 10, 1, tzinfo=timezone.utc))
    save_predictions(predict_elo(games), "backtest", "r2", manifest["sha256"], db, now=FILED)

    sim = simulate_season(synthetic_league_season(), 2020, datetime(2020, 11, 5).date(), n_sims=200)
    (data / "simulations").mkdir()
    sim.table.to_csv(data / "simulations" / "20201105T000000Z-abc.csv", index=False)
    (data / "simulations" / "20201105T000000Z-abc.json").write_text(pd.Series(sim.meta).to_json())
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "README.md").write_text("# notes\n")
    return {"data": data, "db": db, "games": games, "journal": journal}


def test_load_clean_finds_latest_snapshot(lab):
    games, manifest, folder = dd.load_clean(lab["data"])
    assert len(games) == 5 and manifest["n_errors"] == 0 and (folder / "issues.csv").exists()


def test_upcoming_shows_only_games_not_yet_started(lab):
    v = dd.current_versions()["elo"]
    early = dd.upcoming_forecasts(lab["db"], lab["games"], pd.Timestamp("2020-09-01", tz="UTC"), v)
    late = dd.upcoming_forecasts(lab["db"], lab["games"], pd.Timestamp("2020-09-30", tz="UTC"), v)
    assert len(early) == 5
    assert list(late["home"]) == ["NYG"]  # only the game still to come


def test_finished_forecasts_and_running_brier(lab):
    fin = dd.finished_forecasts(lab["db"], lab["games"], dd.current_versions())
    assert set(fin["model"]) == {"elo"} and len(fin) == 3  # tie and unplayed game not scored
    rb = dd.running_brier(fin)
    assert list(rb["games scored"]) == [1, 2, 3]
    assert rb["cumulative Brier"].iloc[-1] == pytest.approx(fin["brier"].mean())


def test_history_never_includes_locked_test_seasons(lab):
    v = dd.current_versions()
    db = lab["db"]
    leaked = predict_elo(lab["games"]).assign(season=TEST_SEASONS[0])
    leaked["game_id"] = "T_" + leaked["game_id"]  # distinct games in a locked season
    save_predictions(leaked, "backtest", "r3", "other-snapshot", db, now=FILED)
    hist = dd.history_by_season(db, v)
    assert not set(hist["season"]) & set(TEST_SEASONS)
    assert set(hist["season"]) == {2020}


def test_latest_simulation_and_runs(lab):
    sim, meta = dd.latest_simulation(lab["data"])
    assert len(sim) == 32 and meta["n_sims"] == 200
    assert dd.recent_runs(lab["journal"]).empty


def run_app(monkeypatch, data, journal):
    monkeypatch.setenv("FORECAST_LAB_DATA_DIR", str(data))
    monkeypatch.setenv("FORECAST_LAB_JOURNAL_DIR", str(journal))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_renders_with_data(monkeypatch, lab):
    at = run_app(monkeypatch, lab["data"], lab["journal"])
    assert not at.exception
    assert "estimate" in at.info[0].value  # the disclaimer is always shown first
    assert [t.label for t in at.tabs] == ["This week", "Live scorecard", "Model history",
                                          "Season simulator", "Data explorer", "Journal"]
    assert not at.button  # read-only: no buttons that could change anything


def test_app_explains_empty_lab(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "empty", tmp_path)
    assert not at.exception
    assert "forecast-lab ingest" in at.warning[0].value
