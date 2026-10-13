"""AT-9: calibration tables and intervals are computed correctly."""

import numpy as np
import pandas as pd
import pytest

from forecast_lab.calibration import expected_calibration_error, plot_reliability, reliability_table, wilson_interval
from forecast_lab.config import TEST_SEASONS, VALIDATION_SEASONS
from forecast_lab.metrics import TestSetLockedError

S = VALIDATION_SEASONS[0]


def synthetic_preds(p, y, season=S):
    """SYNTHETIC forecasts p with outcomes y (1 = home win)."""
    return pd.DataFrame({"season": season, "p_home_win": p,
                         "actual_outcome": np.where(np.asarray(y) == 1, "home_win", "away_win")})


def test_wilson_interval_known_value_and_bounds():
    lo, hi = wilson_interval(np.array([5]), np.array([10]))
    assert (lo[0], hi[0]) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = wilson_interval(np.array([0, 10]), np.array([10, 10]))
    assert lo[0] == pytest.approx(0, abs=1e-12) and hi[1] == pytest.approx(1, abs=1e-12)  # stays in [0, 1]


def test_perfectly_calibrated_forecasts_have_zero_ece():
    # 10 games at 0.3 with 3 home wins; 10 at 0.8 with 8 home wins
    p = [0.3] * 10 + [0.8] * 10
    y = [1] * 3 + [0] * 7 + [1] * 8 + [0] * 2
    t = reliability_table(synthetic_preds(p, y), [S])
    assert list(t["n"]) == [10, 10]
    assert expected_calibration_error(t) == pytest.approx(0)
    assert t["diagonal_inside_ci"].all()


def test_overconfident_forecasts_are_detected():
    p = [0.9] * 200
    y = [1] * 100 + [0] * 100  # says 90%, happens 50%
    t = reliability_table(synthetic_preds(p, y), [S])
    assert expected_calibration_error(t) == pytest.approx(0.4)
    assert not t["diagonal_inside_ci"].iloc[0]


def test_edges_and_ties_handled():
    t = reliability_table(pd.concat([synthetic_preds([0.0, 1.0, 0.5], [0, 1, 1]),
                                     pd.DataFrame({"season": [S], "p_home_win": [0.5], "actual_outcome": ["tie"]})]), [S])
    assert t["n"].sum() == 3  # the tie is not scored


def test_test_seasons_locked():
    with pytest.raises(TestSetLockedError):
        reliability_table(synthetic_preds([0.5], [1], season=TEST_SEASONS[0]), TEST_SEASONS)


def test_plot_is_written(tmp_path):
    t = reliability_table(synthetic_preds([0.3] * 10 + [0.8] * 10, [1, 0] * 10), [S])
    out = plot_reliability({"elo": t, "home_rate": t}, "test", tmp_path / "cal.png")
    assert out.exists() and out.stat().st_size > 10_000
