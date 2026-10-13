"""AT-3: baselines output valid probability distributions with no look-ahead."""

import json

import numpy as np
import pandas as pd

from conftest import synthetic_clean_games
from forecast_lab.baseline import PREDICTION_COLUMNS, predict_coin_flip, predict_home_rate


def test_outputs_are_valid_distributions_with_required_fields():
    games = synthetic_clean_games()
    for model in (predict_coin_flip, predict_home_rate):
        p = model(games)
        assert list(p.columns) == PREDICTION_COLUMNS
        assert ((p["p_home_win"] > 0) & (p["p_home_win"] < 1)).all()
        assert np.allclose(p["p_home_win"] + p["p_away_win"], 1.0)
        assert p["game_id"].is_unique and len(p) == len(games)
        p["features_json"].map(json.loads)  # features are recorded and parseable
        p["warnings_json"].map(json.loads)


def test_future_results_cannot_change_past_predictions():
    """The core leakage test: rewrite every result after a cutoff date and
    check that no prediction on or before the cutoff moves."""
    games = synthetic_clean_games()
    cutoff = sorted(games["gameday"].unique())[20]
    altered = games.copy()
    later = altered["gameday"] > cutoff
    altered.loc[later, "outcome"] = "home_win"  # pretend home teams won every later game

    a, b = predict_home_rate(games), predict_home_rate(altered)
    upto = a["gameday"] <= cutoff
    pd.testing.assert_series_equal(a.loc[upto, "p_home_win"], b.loc[upto, "p_home_win"])
    assert not np.allclose(a.loc[~upto, "p_home_win"], b.loc[~upto, "p_home_win"])  # sanity: the test can fail


def test_same_day_games_do_not_see_each_other():
    games = synthetic_clean_games()
    p = predict_home_rate(games)
    assert p.groupby("gameday")["p_home_win"].nunique().max() == 1
    first_day = p[p["gameday"] == p["gameday"].min()]
    assert (first_day["p_home_win"] == 0.5).all()  # nothing known yet -> prior mean
    assert "few_prior_games" in first_day["warnings_json"].iloc[0]


def test_home_rate_learns_and_is_shrunk_by_prior():
    games = synthetic_clean_games(n_dates=200)  # true home win share = 2/3
    p = predict_home_rate(games)
    last = p["p_home_win"].iloc[-1]
    assert 0.6 < last < 2 / 3   # close to 2/3 but pulled slightly toward 0.5
    assert p["p_home_win"].iloc[0] == 0.5


def test_neutral_site_gets_no_home_edge_and_ties_excluded():
    games = synthetic_clean_games()
    games.loc[50, "neutral_site"] = True
    games.loc[10, ["outcome", "home_margin"]] = ["tie", 0]
    p = predict_home_rate(games).set_index("game_id")
    assert p.loc[games.loc[50, "game_id"], "p_home_win"] == 0.5
    feats = json.loads(p["features_json"].iloc[-1])
    assert feats["games_before"] == len(games) - 4 - 2  # minus last date's games, the tie, the neutral game


def test_unplayed_games_have_no_actual_outcome():
    games = synthetic_clean_games()
    games.loc[len(games) - 1, ["status", "outcome"]] = ["scheduled", pd.NA]
    p = predict_home_rate(games)
    assert pd.isna(p["actual_outcome"].iloc[-1])


def test_model_version_fingerprints_settings():
    from forecast_lab.baseline import model_version
    from forecast_lab.elo import EloParams, predict_elo
    games = synthetic_clean_games()
    v1 = predict_elo(games)["model_version"].iloc[0]
    v2 = predict_elo(games, EloParams(k=25))["model_version"].iloc[0]
    assert v1.startswith("elo-1.0+") and v1 != v2
    assert v1 == model_version("elo", EloParams().to_dict())  # stable for identical settings + code
