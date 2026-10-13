"""Registry of external data sources, with licensing recorded next to the URL.

Rule: a source is only added here after its documentation and license have
been checked by a human, and `verified_on` records when that happened.
Re-check periodically; licenses and URLs change.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    url: str
    license: str
    attribution: str
    docs_url: str
    verified_on: str  # ISO date the license/docs were last checked
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


NFLVERSE_SCHEDULES = Source(
    source_id="nflverse_schedules",
    name="nflverse schedules & results (games.csv)",
    url="https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
    license="CC-BY-4.0",
    attribution=(
        "NFL schedule and results data from the nflverse project "
        "(https://github.com/nflverse/nflverse-data), licensed CC BY 4.0."
    ),
    docs_url="https://github.com/nflverse/nfldata/blob/master/DATASETS.md#games",
    verified_on="2026-10-13",
    notes=(
        "One row per game, 1999-present, including scheduled (unplayed) games "
        "with blank scores. The file is updated in place, so we snapshot it. "
        "nflreadr's docs add: underlying NFL data belongs to its respective "
        "owners and is governed by their terms of use."
    ),
)

SOURCES = {s.source_id: s for s in [NFLVERSE_SCHEDULES]}
