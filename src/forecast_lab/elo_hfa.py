"""Candidate model: Elo with a LEARNED home-field advantage ("elo_hfa").

Hypothesis (journal, 2026-10-13, written before the live season): the fixed
home-field bonus of 55 Elo points is too large for recent seasons. NFL home
advantage has shrunk, and Elo's validation calibration leaned that way.

Change vs. elo-1.0: home advantage is no longer a constant. It starts at
`home_field` and, after every completed non-neutral game, moves a little:
    home_field += hfa_k * (actual - expected)
If home teams keep winning less often than predicted, the bonus shrinks;
if more often, it grows. Everything else is identical to elo-1.0.

This lives in its own file on purpose: elo.py and baseline.py are covered by
the frozen live model's fingerprint (elo-1.0+208703b4) and must not change.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import baseline, elo
from .baseline import _finish, _row_warnings
from .elo import EloState, margin_multiplier, win_probability

NAME = "elo_hfa-0.1"


@dataclass(frozen=True)
class EloHfaParams:
    k: float = 20.0
    home_field: float = 55.0       # starting value in 1999; learned afterwards
    season_revert: float = 0.5
    hfa_k: float = 1.0             # how fast home advantage learns (0 = fixed, i.e. elo-1.0)

    def to_dict(self) -> dict:
        return asdict(self)


def version(params: EloHfaParams) -> str:
    """Readable name + fingerprint of settings and of every source file used."""
    source = "".join(inspect.getsource(mod) for mod in (sys.modules[__name__], elo, baseline))
    blob = json.dumps(params.to_dict(), sort_keys=True) + source
    return f"{NAME}+{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def run_elo_hfa(games: pd.DataFrame, params: EloHfaParams = EloHfaParams()) -> tuple[pd.DataFrame, EloState, float]:
    """Walk forward exactly like elo.run_elo, but with a learned home advantage.
    Returns (predictions, final ratings, final home advantage)."""
    g = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    state = EloState()
    as_elo = elo.EloParams(k=params.k, home_field=params.home_field, season_revert=params.season_revert)
    n = len(g)
    home_pre, away_pre, diff_pre, hfa_pre = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    hfa = params.home_field

    home, away = g["home_franchise"].to_numpy(), g["away_franchise"].to_numpy()
    season, neutral = g["season"].to_numpy(), g["neutral_site"].to_numpy()
    final, margin = (g["status"] == "final").to_numpy(), g["home_margin"].to_numpy()

    for _, idx in g.groupby("gameday", sort=True).indices.items():
        for i in idx:  # predict with information from earlier dates only
            state.start_season(home[i], int(season[i]), as_elo)
            state.start_season(away[i], int(season[i]), as_elo)
            home_pre[i], away_pre[i] = state.rating(home[i]), state.rating(away[i])
            hfa_pre[i] = 0.0 if neutral[i] else hfa
            diff_pre[i] = home_pre[i] - away_pre[i] + hfa_pre[i]
        hfa_change = 0.0
        for i in idx:  # then learn from this date's results
            if not final[i]:
                continue
            m = int(margin[i])
            actual = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
            expected = float(win_probability(diff_pre[i]))
            winner_diff = diff_pre[i] if m > 0 else -diff_pre[i] if m < 0 else 0.0
            change = params.k * margin_multiplier(m, winner_diff) * (actual - expected)
            state.ratings[home[i]] = state.rating(home[i]) + change
            state.ratings[away[i]] = state.rating(away[i]) - change
            if not neutral[i]:
                hfa_change += params.hfa_k * (actual - expected)
        hfa += hfa_change  # applied after the whole date, so same-day games don't see each other

    features = [{"home_elo": round(float(h), 1), "away_elo": round(float(a), 1), "home_field": round(float(f), 2),
                 "elo_diff": round(float(d), 1), "params": params.to_dict()}
                for h, a, f, d in zip(home_pre, away_pre, hfa_pre, diff_pre)]
    out = _finish(g, "elo", np.asarray(win_probability(diff_pre)), features, _row_warnings(g), params.to_dict())
    out["model_version"] = version(params)
    return out, state, float(hfa)


def predict_elo_hfa(games: pd.DataFrame, params: EloHfaParams = EloHfaParams()) -> pd.DataFrame:
    return run_elo_hfa(games, params)[0]
