"""AT-11: the learned-home-field candidate is a clean, leak-free variant of Elo,
and the frozen live model can never change by accident."""

import json

import numpy as np
import pandas as pd

from conftest import synthetic_clean_games
from forecast_lab.baseline import model_version
from forecast_lab.elo import EloParams, predict_elo
from forecast_lab.elo_hfa import EloHfaParams, predict_elo_hfa, run_elo_hfa

FROZEN = "elo-1.0+208703b4"  # pre-registered in journal/PREREGISTRATION.md


def test_frozen_live_model_is_unchanged():
    """If this fails, someone edited elo.py or baseline.py (or Elo's settings) during the live test.
    Revert that change; improvements belong in a NEW model file."""
    assert model_version("elo", EloParams().to_dict()) == FROZEN


def test_zero_learning_reproduces_elo_exactly():
    games = synthetic_clean_games()
    a = predict_elo(games)
    b = predict_elo_hfa(games, EloHfaParams(hfa_k=0))
    np.testing.assert_allclose(a["p_home_win"], b["p_home_win"])
    assert b["model_version"].iloc[0].startswith("elo_hfa-0.1+")


def test_home_advantage_learns_in_the_right_direction():
    games = synthetic_clean_games()  # synthetic: home teams win 2/3 of games
    _, _, grows = run_elo_hfa(games, EloHfaParams(hfa_k=2))
    flipped = games.copy()
    flipped["outcome"] = flipped["outcome"].map({"home_win": "away_win", "away_win": "home_win"})
    flipped["home_margin"] = -flipped["home_margin"]
    _, _, shrinks = run_elo_hfa(flipped, EloHfaParams(hfa_k=2))
    assert grows > 55 > shrinks


def test_no_look_ahead():
    games = synthetic_clean_games()
    cutoff = sorted(games["gameday"].unique())[20]
    altered = games.copy()
    altered.loc[altered["gameday"] > cutoff, ["outcome", "home_margin"]] = ["away_win", -30]
    a, b = predict_elo_hfa(games, EloHfaParams(hfa_k=4)), predict_elo_hfa(altered, EloHfaParams(hfa_k=4))
    upto = a["gameday"] <= cutoff
    pd.testing.assert_series_equal(a.loc[upto, "p_home_win"], b.loc[upto, "p_home_win"])


def test_neutral_sites_get_no_bonus_and_features_record_it():
    games = synthetic_clean_games()
    games.loc[30, "neutral_site"] = True
    p = predict_elo_hfa(games).set_index("game_id")
    assert json.loads(p.loc[games.loc[30, "game_id"], "features_json"])["home_field"] == 0
