"""Shared test helpers.

All tables built here are SYNTHETIC TEST FIXTURES with made-up scores, used
only to exercise code paths. They are never mixed with real data.
"""

from datetime import date

import pandas as pd
import pytest

RAW_HEADER = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score",
              "home_team", "home_score", "location", "result", "spread_line"]


def raw_row(game_id, gameday, away, home, away_score="", home_score="", location="Home",
            season="2020", week="1", game_type="REG", result=None, spread_line="-3", gametime="13:00"):
    if result is None:
        result = str(int(home_score) - int(away_score)) if home_score != "" and away_score != "" else ""
    return dict(zip(RAW_HEADER, [game_id, season, game_type, week, gameday, gametime, away, str(away_score),
                                 home, str(home_score), location, result, spread_line]))


def raw_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_HEADER).astype(str)


def raw_csv_bytes(rows) -> bytes:
    return raw_frame(rows).to_csv(index=False, lineterminator="\n").encode()


@pytest.fixture
def as_of():
    return date(2020, 10, 1)


@pytest.fixture
def valid_rows():
    """Synthetic: 3 played games, 1 neutral-site game, 1 tie, 1 future game."""
    return [
        raw_row("2020_01_KC_BUF", "2020-09-13", "KC", "BUF", 20, 27),
        raw_row("2020_01_OAK_DEN", "2020-09-13", "OAK", "DEN", 24, 17),
        raw_row("2020_02_NE_MIA", "2020-09-20", "NE", "MIA", 10, 13),
        raw_row("2020_02_TB_SF", "2020-09-20", "TB", "SF", 30, 27, location="Neutral"),
        raw_row("2020_03_GB_CHI", "2020-09-27", "GB", "CHI", 21, 21),
        raw_row("2020_05_DAL_NYG", "2020-10-11", "DAL", "NYG", week="5"),
    ]


def synthetic_clean_games(n_dates=40, games_per_date=4, home_win_every=3, start=date(2010, 9, 5), season=2010):
    """Synthetic clean-format table: deterministic pattern, ~2/3 home wins."""
    from datetime import timedelta
    rows, k = [], 0
    teams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE"]
    for d in range(n_dates):
        day = start + timedelta(days=7 * d)
        for j in range(games_per_date):
            home_win = (k % home_win_every) != 0
            rows.append({
                "game_id": f"S{d:03d}_{j}", "season": season + d // 20, "game_type": "REG", "week": d % 20 + 1,
                "gameday": day, "home_team": teams[2 * j], "away_team": teams[2 * j + 1],
                "home_franchise": teams[2 * j], "away_franchise": teams[2 * j + 1],
                "home_score": 24 if home_win else 17, "away_score": 17 if home_win else 24,
                "home_margin": 7 if home_win else -7, "outcome": "home_win" if home_win else "away_win",
                "status": "final", "neutral_site": False, "dq_warnings": "",
            })
            k += 1
    return pd.DataFrame(rows)


def synthetic_league_season(season=2020, n_weeks=17, played_weeks=8, seed=0, strong="KC", weak="NYJ"):
    """SYNTHETIC full 32-team season with made-up results. Teams are randomly
    paired each week. In played weeks, `strong` always wins and `weak` always
    loses; other games are decided by a fixed coin. Later weeks are unplayed."""
    import numpy as np
    from datetime import timedelta
    from forecast_lab.nfl_structure import TEAM_DIVISION, TEAMS

    rng = np.random.default_rng(seed)
    start = date(season, 9, 10)
    rows = []
    for wk in range(1, n_weeks + 1):
        day = start + timedelta(days=7 * (wk - 1))
        order = rng.permutation(TEAMS)
        for j in range(16):
            home, away = order[2 * j], order[2 * j + 1]
            played = wk <= played_weeks
            if strong in (home, away):
                home_win = home == strong
            elif weak in (home, away):
                home_win = away == weak
            else:
                home_win = bool(rng.random() < 0.57)
            rows.append({
                "game_id": f"{season}_{wk:02d}_{away}_{home}", "season": season, "game_type": "REG", "week": wk,
                "gameday": day, "home_team": home, "away_team": away, "home_franchise": home, "away_franchise": away,
                "home_score": (24 if home_win else 17) if played else pd.NA,
                "away_score": (17 if home_win else 24) if played else pd.NA,
                "home_margin": (7 if home_win else -7) if played else pd.NA,
                "outcome": ("home_win" if home_win else "away_win") if played else pd.NA,
                "status": "final" if played else "scheduled", "neutral_site": False,
                "div_game": "1" if TEAM_DIVISION[home] == TEAM_DIVISION[away] else "0", "dq_warnings": "",
            })
    df = pd.DataFrame(rows)
    for c in ("home_score", "away_score", "home_margin"):
        df[c] = df[c].astype("Int64")
    return df
