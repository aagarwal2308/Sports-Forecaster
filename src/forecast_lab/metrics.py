"""Scoring rules, uncertainty intervals, and the locked test set.

Lower is better for Brier score and log loss; higher is better for accuracy.
Reference points for a 50/50 guess: Brier = 0.25, log loss = ln 2 = 0.693.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import TEST_SEASONS


class TestSetLockedError(PermissionError):
    """Raised when code tries to score the held-out test seasons without unlocking."""

    __test__ = False  # tell pytest this is not a test class


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Share of games where the favored side won. A 0.5 forecast gets half credit."""
    correct = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float))
    return float(np.mean(correct))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability. 0 = perfect, 0.25 = coin flip."""
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Average surprise. Punishes confident wrong forecasts very hard."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "brier": brier,
    "log_loss": log_loss,
}


def bootstrap_ci(metric, y, p, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling games with replacement.

    Caveat: games are treated as independent. Games in the same week share
    conditions, so true uncertainty is somewhat larger than this shows.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y[idx], p[idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def check_test_lock(seasons: Iterable[int], unlock_test: bool) -> None:
    touched = sorted(set(seasons) & set(TEST_SEASONS))
    if touched and not unlock_test:
        raise TestSetLockedError(
            f"Seasons {touched} are the locked test set. Tune and compare on validation seasons; "
            "unlock the test set only for a final, pre-declared evaluation (--unlock-test)."
        )


def scoreable(preds: pd.DataFrame) -> pd.DataFrame:
    """Games with a decided result (home or away win)."""
    return preds[preds["actual_outcome"].isin(["home_win", "away_win"])]


def evaluate(preds: pd.DataFrame, seasons: Iterable[int], unlock_test: bool = False,
             n_boot: int = 1000, seed: int = 0) -> dict:
    seasons = list(seasons)
    check_test_lock(seasons, unlock_test)
    in_scope = preds[preds["season"].isin(seasons)]
    scored = scoreable(in_scope)
    report = {
        "model_version": in_scope["model_version"].iloc[0] if len(in_scope) else None,
        "seasons": f"{min(seasons)}-{max(seasons)}",
        "n_games": int(len(scored)),
        "n_ties_excluded": int((in_scope["actual_outcome"] == "tie").sum()),
        "n_unresolved_excluded": int(in_scope["actual_outcome"].isna().sum()),
    }
    if len(scored) == 0:
        return report
    y = (scored["actual_outcome"] == "home_win").to_numpy(dtype=float)
    p = scored["p_home_win"].to_numpy(dtype=float)
    for name, fn in METRICS.items():
        lo, hi = bootstrap_ci(fn, y, p, n_boot=n_boot, seed=seed)
        report[name] = fn(y, p)
        report[f"{name}_ci95"] = (lo, hi)
    return report


def evaluate_by_season(preds: pd.DataFrame, seasons: Iterable[int], unlock_test: bool = False) -> pd.DataFrame:
    seasons = list(seasons)
    check_test_lock(seasons, unlock_test)
    rows = []
    for s in seasons:
        scored = scoreable(preds[preds["season"] == s])
        if len(scored) == 0:
            continue
        y = (scored["actual_outcome"] == "home_win").to_numpy(dtype=float)
        p = scored["p_home_win"].to_numpy(dtype=float)
        rows.append({"season": s, "n_games": len(scored), "home_win_share": y.mean(),
                     **{name: fn(y, p) for name, fn in METRICS.items()}})
    return pd.DataFrame(rows)
