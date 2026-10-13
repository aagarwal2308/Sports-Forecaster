"""AT-7: the season simulator is consistent, reproducible, leak-free and honest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_league_season
from forecast_lab.config import TEST_SEASONS
from forecast_lab.metrics import TestSetLockedError
from forecast_lab.simulate import actual_outcomes, backtest_simulator, simulate_season

CUTOFF = date(2020, 11, 5)  # after week 8 of the synthetic 2020 season


@pytest.fixture(scope="module")
def season():
    return synthetic_league_season(played_weeks=8)


@pytest.fixture(scope="module")
def result(season):
    return simulate_season(season, 2020, CUTOFF, n_sims=3000, seed=1)


def test_probabilities_add_up_to_available_spots(result):
    t = result.table
    assert t["p_playoffs"].sum() == pytest.approx(14)   # 7 per conference since 2020
    assert t["p_division"].sum() == pytest.approx(8)
    assert t["p_top_seed"].sum() == pytest.approx(2)
    assert ((t["p_division"] <= t["p_playoffs"]) & (t["p_top_seed"] <= t["p_division"])).all()


def test_season_win_totals_are_conserved(result, season):
    t = result.table
    assert t["mean_wins"].sum() == pytest.approx(len(season))  # every game produces exactly one win
    assert (t["wins_p10"] <= t["wins_p50"]).all() and (t["wins_p50"] <= t["wins_p90"]).all()
    assert result.meta["games_known"] == 128 and result.meta["games_simulated"] == 144


def test_same_seed_same_answer(season, result):
    again = simulate_season(season, 2020, CUTOFF, n_sims=3000, seed=1)
    pd.testing.assert_frame_equal(result.table, again.table)


def test_results_after_cutoff_are_never_used(season, result):
    """Even if the later games' real results are in the data, the simulator must ignore them."""
    leaked = synthetic_league_season(played_weeks=17)  # same pairings, all weeks 'final'
    res = simulate_season(leaked, 2020, CUTOFF, n_sims=3000, seed=1)
    pd.testing.assert_frame_equal(result.table, res.table)


def test_dominant_team_ranks_above_winless_team(result):
    t = result.table.set_index("team")
    assert t.loc["KC", "wins"] == 8 and t.loc["NYJ", "wins"] == 0
    assert t.loc["KC", "p_playoffs"] > 0.9 > 0.1 > t.loc["NYJ", "p_playoffs"]


def test_finished_season_has_no_uncertainty_in_wins(season):
    done = synthetic_league_season(played_weeks=17)
    t = simulate_season(done, 2020, date(2021, 2, 1), n_sims=500).table.set_index("team")
    actual = actual_outcomes(done, 2020).set_index("team").loc[t.index]
    assert (t["mean_wins"] == actual["actual_wins"]).all()
    assert (t["wins_p10"] == t["wins_p90"]).all()


def test_hot_simulation_spreads_outcomes_more_than_cold(season):
    hot = simulate_season(season, 2020, CUTOFF, n_sims=4000, seed=2, hot=True).table
    cold = simulate_season(season, 2020, CUTOFF, n_sims=4000, seed=2, hot=False).table
    assert (hot["wins_p90"] - hot["wins_p10"]).mean() >= (cold["wins_p90"] - cold["wins_p10"]).mean()


def test_coin_flip_mode_ignores_ratings(season):
    t = simulate_season(season, 2020, CUTOFF, model="coin_flip", n_sims=4000, seed=3).table.set_index("team")
    remaining = 17 - 8
    assert t.loc["KC", "mean_wins"] == pytest.approx(8 + remaining / 2, abs=0.15)


def test_rejects_old_seasons_and_locked_test(season):
    with pytest.raises(ValueError):
        simulate_season(season, 2001, CUTOFF)
    with pytest.raises(TestSetLockedError):
        backtest_simulator(season, TEST_SEASONS, after_week=8)
