"""AT-2: validation catches bad records, never silently fixes them, and is reproducible."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from conftest import raw_csv_bytes, raw_frame, raw_row
from forecast_lab.clean import SchemaError, clean_games, clean_snapshot, load_clean_games
from forecast_lab.ingest import snapshot_source
from forecast_lab.sources import NFLVERSE_SCHEDULES
from forecast_lab.teams import franchise_of


def checks(result, severity):
    return set(result.issues.loc[result.issues["severity"] == severity, "check"])


def test_valid_rows_are_kept_with_correct_status_and_outcome(valid_rows, as_of):
    res = clean_games(raw_frame(valid_rows), as_of)
    assert len(res.games) == 6 and res.issues.empty
    g = res.games.set_index("game_id")
    assert g.loc["2020_01_KC_BUF", "outcome"] == "home_win"
    assert g.loc["2020_01_OAK_DEN", "outcome"] == "away_win"
    assert g.loc["2020_03_GB_CHI", "outcome"] == "tie"
    assert g.loc["2020_05_DAL_NYG", "status"] == "scheduled"
    assert pd.isna(g.loc["2020_05_DAL_NYG", "outcome"])
    assert g.loc["2020_02_TB_SF", "neutral_site"]
    assert list(res.games["gameday"]) == sorted(res.games["gameday"])  # chronological


def test_relocated_team_gets_stable_franchise_id(valid_rows, as_of):
    g = clean_games(raw_frame(valid_rows), as_of).games.set_index("game_id")
    assert g.loc["2020_01_OAK_DEN", "away_team"] == "OAK"       # source value preserved
    assert g.loc["2020_01_OAK_DEN", "away_franchise"] == "LV"   # stable id for models
    assert franchise_of("SD") == "LAC" and franchise_of("STL") == "LA" and franchise_of("KC") == "KC"
    with pytest.raises(KeyError):
        franchise_of("XYZ")


@pytest.mark.parametrize("bad_row, expected_check", [
    (raw_row("B1", "2020-09-13", "KC", "KC", 1, 2), "same_team"),
    (raw_row("B2", "2020-09-13", "KC", "XXX", 1, 2), "unknown_home_team"),
    (raw_row("B3", "2020-09-13", "KC", "BUF", -1, 2, result="3"), "negative_away_score"),
    (raw_row("B4", "2020-09-13", "KC", "BUF", "ten", 2, result=""), "bad_away_score"),
    (raw_row("B5", "2020-09-13", "KC", "BUF", 10, "", result=""), "partial_score"),
    (raw_row("B6", "13/09/2020", "KC", "BUF", 1, 2), "bad_gameday"),
    (raw_row("B7", "2020-09-13", "KC", "BUF", 1, 2, location="Moon"), "bad_location"),
    (raw_row("B8", "2020-12-25", "KC", "BUF", 1, 2), "score_for_future_game"),
    (raw_row("B9", "2020-09-13", "KC", "BUF", 1, 2, game_type="PRE"), "bad_game_type"),
])
def test_bad_rows_are_excluded_and_logged(valid_rows, as_of, bad_row, expected_check):
    res = clean_games(raw_frame(valid_rows + [bad_row]), as_of)
    assert expected_check in checks(res, "error")
    assert bad_row["game_id"] not in set(res.games["game_id"])
    assert len(res.games) == len(valid_rows)  # good rows untouched


def test_duplicate_ids_exclude_all_copies(valid_rows, as_of):
    dup = raw_row("2020_01_KC_BUF", "2020-09-13", "KC", "BUF", 99, 0)
    res = clean_games(raw_frame(valid_rows + [dup]), as_of)
    assert "duplicate_game_id" in checks(res, "error")
    assert "2020_01_KC_BUF" not in set(res.games["game_id"])


def test_past_game_without_score_is_flagged_not_invented(valid_rows, as_of):
    row = raw_row("2020_02_LAC_CIN", "2020-09-20", "LAC", "CIN")  # before as_of, no score
    res = clean_games(raw_frame(valid_rows + [row]), as_of)
    g = res.games.set_index("game_id").loc["2020_02_LAC_CIN"]
    assert "missing_result" in checks(res, "warning")
    assert g["status"] == "missing_result" and pd.isna(g["home_score"]) and pd.isna(g["outcome"])
    assert g["dq_warnings"] == "missing_result"


def test_result_mismatch_is_warned(valid_rows, as_of):
    row = raw_row("2020_02_LAC_CIN", "2020-09-20", "LAC", "CIN", 10, 20, result="7")
    res = clean_games(raw_frame(valid_rows + [row]), as_of)
    assert "result_mismatch" in checks(res, "warning")
    assert res.games.set_index("game_id").loc["2020_02_LAC_CIN", "home_margin"] == 10  # scores win


def test_missing_required_column_raises(valid_rows, as_of):
    with pytest.raises(SchemaError):
        clean_games(raw_frame(valid_rows).drop(columns=["home_score"]), as_of)


def test_betting_columns_are_not_carried_into_clean_data(valid_rows, as_of):
    assert "spread_line" not in clean_games(raw_frame(valid_rows), as_of).games.columns


def test_cleaning_is_byte_reproducible_and_round_trips(tmp_path, valid_rows):
    raw_dir, clean_dir = tmp_path / "raw", tmp_path / "clean"
    manifest, _ = snapshot_source(NFLVERSE_SCHEDULES, raw_dir, lambda url: raw_csv_bytes(valid_rows),
                                  now=datetime(2020, 10, 1, tzinfo=timezone.utc))
    first = clean_snapshot(manifest, raw_dir, clean_dir)
    games_csv = Path(first["path"]) / "games.csv"
    bytes1 = games_csv.read_bytes()
    second = clean_snapshot(manifest, raw_dir, clean_dir)
    assert first["clean_sha256"] == second["clean_sha256"]
    assert games_csv.read_bytes() == bytes1
    assert first["as_of"] == "2020-10-01"  # from the snapshot, not today's date
    loaded = load_clean_games(games_csv)
    assert loaded["home_score"].dtype == "Int64" and loaded["neutral_site"].dtype == bool
    assert loaded.set_index("game_id").loc["2020_03_GB_CHI", "outcome"] == "tie"


def test_kickoff_converted_from_eastern_to_utc(as_of):
    rows = [raw_row("K1", "2020-09-13", "KC", "BUF", 1, 2), raw_row("K2", "2020-12-13", "KC", "BUF")]
    rows[1]["gametime"] = "20:20"
    g = clean_games(raw_frame(rows), as_of).games.set_index("game_id")
    assert g.loc["K1", "kickoff_utc"] == "2020-09-13T17:00:00Z"  # EDT = UTC-4
    assert g.loc["K2", "kickoff_utc"] == "2020-12-14T01:20:00Z"  # EST = UTC-5, next day in UTC


def test_unplayed_game_without_time_uses_start_of_day(as_of):
    res = clean_games(raw_frame([raw_row("K3", "2020-12-13", "KC", "BUF", gametime="")]), as_of)
    assert res.games.loc[0, "kickoff_utc"] == "2020-12-13T05:00:00Z"  # midnight Eastern
    assert "kickoff_time_unknown" in res.games.loc[0, "dq_warnings"]
