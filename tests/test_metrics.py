"""AT-4: scoring rules are correct, intervals are reported, and the test set is locked."""

import math

import numpy as np
import pandas as pd
import pytest

from forecast_lab.config import TEST_SEASONS, VALIDATION_SEASONS
from forecast_lab.metrics import TestSetLockedError, accuracy, bootstrap_ci, brier, evaluate, log_loss

y = np.array([1, 0, 1, 1], dtype=float)


def test_known_values():
    half = np.full(4, 0.5)
    assert brier(y, half) == pytest.approx(0.25)
    assert log_loss(y, half) == pytest.approx(math.log(2))
    assert accuracy(y, half) == pytest.approx(0.5)       # 0.5 forecasts get half credit
    assert brier(y, y) == 0.0
    assert accuracy(y, np.array([0.9, 0.2, 0.4, 0.7])) == pytest.approx(0.75)


def test_confident_mistakes_are_punished_more_by_log_loss():
    p_mild, p_bold = np.array([0.6]), np.array([0.99])
    assert log_loss(np.array([0.0]), p_bold) > 5 * log_loss(np.array([0.0]), p_mild)


def test_bootstrap_interval_is_deterministic_and_contains_estimate():
    rng = np.random.default_rng(1)
    yy = rng.integers(0, 2, 500).astype(float)
    pp = rng.uniform(0.3, 0.7, 500)
    lo, hi = bootstrap_ci(brier, yy, pp, seed=0)
    assert lo < brier(yy, pp) < hi
    assert (lo, hi) == bootstrap_ci(brier, yy, pp, seed=0)


def _preds(season, outcomes, p=0.6):
    return pd.DataFrame({"season": season, "model_version": "m", "p_home_win": p, "actual_outcome": outcomes})


def test_evaluate_reports_sizes_and_excludes_ties_visibly():
    season = VALIDATION_SEASONS[0]
    preds = _preds(season, ["home_win", "away_win", "tie", None, "home_win"])
    r = evaluate(preds, [season], n_boot=200)
    assert (r["n_games"], r["n_ties_excluded"], r["n_unresolved_excluded"]) == (3, 1, 1)
    assert set(r) >= {"accuracy", "brier", "log_loss", "brier_ci95"}


def test_test_seasons_are_locked():
    preds = _preds(TEST_SEASONS[0], ["home_win", "away_win"])
    with pytest.raises(TestSetLockedError):
        evaluate(preds, TEST_SEASONS)
    assert evaluate(preds, TEST_SEASONS, unlock_test=True, n_boot=50)["n_games"] == 2


def test_splits_are_chronological_and_disjoint():
    from forecast_lab.config import TRAIN_SEASONS
    assert max(TRAIN_SEASONS) < min(VALIDATION_SEASONS) < max(VALIDATION_SEASONS) < min(TEST_SEASONS)
