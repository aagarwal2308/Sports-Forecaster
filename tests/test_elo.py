"""AT-6: Elo ratings behave as designed and never look ahead."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_clean_games, synthetic_league_season
from forecast_lab.elo import MEAN_RATING, EloParams, margin_multiplier, predict_elo, run_elo, win_probability


def test_probability_formula():
    assert win_probability(0) == pytest.approx(0.5)
    assert win_probability(400) == pytest.approx(10 / 11)
    assert win_probability(-100) == pytest.approx(1 - win_probability(100))


def test_first_games_use_home_field_only():
    p = predict_elo(synthetic_clean_games()).iloc[0]
    assert p["p_home_win"] == pytest.approx(float(win_probability(EloParams().home_field)))
    assert "early_ratings" in json.loads(p["warnings_json"])


def test_ratings_are_zero_sum_and_winners_gain():
    games = synthetic_league_season(played_weeks=17)
    _, state = run_elo(games)
    ratings = np.array(list(state.ratings.values()))
    assert ratings.mean() == pytest.approx(MEAN_RATING)
    assert state.ratings["KC"] == ratings.max() and state.ratings["NYJ"] == ratings.min()


def test_bigger_margins_count_more_but_favorites_are_damped():
    assert margin_multiplier(21, 0) > margin_multiplier(3, 0)
    assert margin_multiplier(14, 200) < margin_multiplier(14, 0)  # favorite winning big: smaller update
    assert margin_multiplier(0, 0) > 0                           # ties still move ratings


def test_ratings_revert_toward_mean_between_seasons():
    one = synthetic_league_season(season=2020, played_weeks=17)
    two = synthetic_league_season(season=2021, played_weeks=1, seed=1)
    _, end_2020 = run_elo(one)
    preds = predict_elo(pd.concat([one, two]))
    kc_first_2021 = preds[(preds["season"] == 2021) & preds["game_id"].str.contains("KC")].iloc[0]
    f = json.loads(kc_first_2021["features_json"])
    kc_elo = f["home_elo"] if kc_first_2021["game_id"].endswith("_KC") else f["away_elo"]
    expected = MEAN_RATING + (end_2020.ratings["KC"] - MEAN_RATING) * (1 - EloParams().season_revert)
    assert kc_elo == pytest.approx(expected, abs=0.1)


def test_future_results_cannot_change_past_elo_predictions():
    games = synthetic_clean_games()
    cutoff = sorted(games["gameday"].unique())[20]
    altered = games.copy()
    later = altered["gameday"] > cutoff
    altered.loc[later, ["outcome", "home_margin"]] = ["home_win", 30]
    a, b = predict_elo(games), predict_elo(altered)
    upto = a["gameday"] <= cutoff
    pd.testing.assert_series_equal(a.loc[upto, "p_home_win"], b.loc[upto, "p_home_win"])
    assert not np.allclose(a.loc[~upto, "p_home_win"], b.loc[~upto, "p_home_win"])


def test_neutral_site_removes_home_field():
    games = synthetic_clean_games()
    games.loc[0, "neutral_site"] = True
    p = predict_elo(games).set_index("game_id")
    assert p.loc[games.loc[0, "game_id"], "p_home_win"] == pytest.approx(0.5)
