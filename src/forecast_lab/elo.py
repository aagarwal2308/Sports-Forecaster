"""Milestone 2: Elo team ratings (in the style popularized by FiveThirtyEight's NFL Elo).

Every franchise has a rating (average = 1500). Before a game:
    rating_diff = home_rating - away_rating + home_field_bonus   (bonus = 0 at neutral sites)
    P(home win) = 1 / (1 + 10 ** (-rating_diff / 400))
After the game, both teams move by the same amount in opposite directions:
    change = K * margin_multiplier * (actual - expected)
where actual is 1 (home win), 0.5 (tie) or 0 (away win). The margin multiplier
lets blowouts count more than narrow wins, and is damped for heavy favorites
(the "autocorrelation" correction) so good teams don't inflate forever.
Between seasons, each rating is pulled part of the way back toward 1500,
because rosters change.

No-leakage rule (same as the baselines): every game on date D is predicted
with ratings that include only results from dates before D.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .baseline import _finish, _row_warnings

MEAN_RATING = 1500.0


@dataclass(frozen=True)
class EloParams:
    """Defaults chosen by `tune_elo` on TRAIN seasons 2002-2015 only (see journal, 2026-10-13)."""
    k: float = 20.0                # how fast ratings react
    home_field: float = 55.0       # rating points added for the home team
    season_revert: float = 0.5     # share of the gap to 1500 removed between seasons

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EloState:
    """Ratings after processing every completed game before some date."""
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: dict[str, int] = field(default_factory=dict)
    mean_margin_multiplier: float = 1.0  # average multiplier seen in history (used by simulations)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, MEAN_RATING)

    def start_season(self, team: str, season: int, params: EloParams) -> None:
        """Apply between-season reversion the first time a team appears in a new season."""
        if self.last_season.get(team) != season:
            if team in self.ratings:
                self.ratings[team] = MEAN_RATING + (self.ratings[team] - MEAN_RATING) * (1 - params.season_revert)
            self.last_season[team] = season


def win_probability(rating_diff: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(rating_diff) / 400.0))


def margin_multiplier(margin: int, winner_rating_diff: float) -> float:
    """ln(|margin| + 1) scaled down when the favorite wins (ties count as a 1-point margin)."""
    if margin == 0:
        return math.log(2.0)
    return math.log(abs(margin) + 1.0) * (2.2 / (winner_rating_diff * 0.001 + 2.2))


def run_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> tuple[pd.DataFrame, EloState]:
    """Walk forward through all games. Returns (predictions, final state).

    Only rows with status == 'final' update ratings. Scheduled games still get
    a prediction using the latest ratings, which is how upcoming games are forecast.
    """
    g = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    state = EloState()
    n = len(g)
    home_pre, away_pre, diff_pre = np.empty(n), np.empty(n), np.empty(n)
    multipliers = []

    home = g["home_franchise"].to_numpy()
    away = g["away_franchise"].to_numpy()
    season = g["season"].to_numpy()
    neutral = g["neutral_site"].to_numpy()
    final = (g["status"] == "final").to_numpy()
    margin = g["home_margin"].to_numpy()

    for _, idx in g.groupby("gameday", sort=True).indices.items():
        # 1) predict every game on this date with ratings from earlier dates only
        for i in idx:
            state.start_season(home[i], int(season[i]), params)
            state.start_season(away[i], int(season[i]), params)
            home_pre[i], away_pre[i] = state.rating(home[i]), state.rating(away[i])
            diff_pre[i] = home_pre[i] - away_pre[i] + (0.0 if neutral[i] else params.home_field)
        # 2) then learn from this date's results
        for i in idx:
            if not final[i]:
                continue
            m = int(margin[i])
            actual = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
            expected = float(win_probability(diff_pre[i]))
            winner_diff = diff_pre[i] if m > 0 else -diff_pre[i] if m < 0 else 0.0
            mult = margin_multiplier(m, winner_diff)
            multipliers.append(mult)
            change = params.k * mult * (actual - expected)
            state.ratings[home[i]] = state.rating(home[i]) + change
            state.ratings[away[i]] = state.rating(away[i]) - change

    state.mean_margin_multiplier = float(np.mean(multipliers)) if multipliers else 1.0
    p_home = win_probability(diff_pre)
    features = [
        {"home_elo": round(float(h), 1), "away_elo": round(float(a), 1),
         "home_field": 0.0 if nt else params.home_field, "elo_diff": round(float(d), 1), "params": params.to_dict()}
        for h, a, d, nt in zip(home_pre, away_pre, diff_pre, neutral)
    ]
    warnings = _row_warnings(g)
    for i in range(n):
        # A team's first season has no rating history; flag those predictions.
        if (g.at[i, "season"] == g["season"].min()) or (g.at[i, "season"] <= 2002 and "HOU" in (home[i], away[i])):
            warnings[i].append("early_ratings")
    return _finish(g, "elo", np.asarray(p_home), features, warnings, params.to_dict()), state


def predict_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    return run_elo(games, params)[0]


def tune_elo(games: pd.DataFrame, seasons, grid: dict[str, list[float]]) -> pd.DataFrame:
    """Score every parameter combination by Brier on `seasons` (pass TRAIN seasons only)."""
    from itertools import product
    from .metrics import brier, scoreable

    rows = []
    for values in product(*grid.values()):
        params = EloParams(**dict(zip(grid.keys(), values)))
        scored = scoreable(predict_elo(games, params))
        scored = scored[scored["season"].isin(list(seasons))]
        y = (scored["actual_outcome"] == "home_win").to_numpy(dtype=float)
        rows.append({**params.to_dict(), "n_games": len(scored), "brier": brier(y, scored["p_home_win"].to_numpy())})
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
