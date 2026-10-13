"""Standardized team identifiers.

The source uses the abbreviation a team had *at the time of the game*
(e.g. OAK before 2020, LV after). Ratings need a stable identity across
relocations, so every game gets both:
    team code      -> what the source said (kept for auditability)
    franchise_id   -> stable id used by models
"""

# Relocated franchises: historical code -> current code.
RELOCATIONS = {
    "OAK": "LV",   # Raiders moved to Las Vegas in 2020
    "SD": "LAC",   # Chargers moved to Los Angeles in 2017
    "STL": "LA",   # Rams moved to Los Angeles in 2016
}

CURRENT_TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
})

KNOWN_CODES = CURRENT_TEAMS | frozenset(RELOCATIONS)


def franchise_of(code: str) -> str:
    """Map a source team code to a stable franchise id. Unknown codes raise."""
    if code not in KNOWN_CODES:
        raise KeyError(f"Unknown team code {code!r}; add it to teams.py after checking the source docs.")
    return RELOCATIONS.get(code, code)
