"""NFL league structure needed to turn simulated games into standings.

Divisions have been unchanged since the 2002 realignment (32 teams, 8 divisions
of 4). Keyed by franchise id (see teams.py). This is checked against the source
data's own `div_game` flag by `check_divisions` (run on every simulation), so a
future realignment can't slip through silently.

Playoff format: 4 division winners + 2 wild cards per conference through 2019;
4 + 3 since 2020 (assumed unchanged for 2026).
"""

from __future__ import annotations

import pandas as pd

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIVISION = {t: d for d, teams in DIVISIONS.items() for t in teams}
TEAMS = sorted(TEAM_DIVISION)
FIRST_SEASON = 2002


def conference_of(team: str) -> str:
    return TEAM_DIVISION[team][:3]


def wild_cards(season: int) -> int:
    return 3 if season >= 2020 else 2


def check_divisions(games: pd.DataFrame, season: int) -> list[str]:
    """Compare our division map with the source's div_game flag for one season.
    Returns a list of mismatches (empty = consistent)."""
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    problems = []
    for r in reg.itertuples(index=False):
        ours = TEAM_DIVISION.get(r.home_franchise) == TEAM_DIVISION.get(r.away_franchise)
        theirs = str(r.div_game).strip() == "1"
        if ours != theirs:
            problems.append(f"{r.game_id}: div_game={r.div_game!r} but our map says same_division={ours}")
    return problems
