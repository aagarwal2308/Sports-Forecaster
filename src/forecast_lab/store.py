"""Append-only prediction records in SQLite (built into Python; no server).

Two kinds of record, never mixed:
    backtest     made AFTER the games, replaying history. Useful for research,
                 but it is hindsight: the modeler already knew the results.
    prospective  made BEFORE kickoff. Refused unless the game is unplayed AND
                 the current time is earlier than its kickoff_utc. The
                 actual outcome is filled in later by `record_outcomes`.

The database itself refuses edits to a stored forecast (see the trigger),
so a pre-game forecast can never be quietly "improved" after the fact.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL,
    prediction_kind         TEXT NOT NULL CHECK (prediction_kind IN ('backtest', 'prospective')),
    game_id                 TEXT NOT NULL,
    season                  INTEGER NOT NULL,
    gameday                 TEXT NOT NULL,
    kickoff_utc             TEXT,
    predicted_at_utc        TEXT NOT NULL,
    data_snapshot_sha256    TEXT NOT NULL,
    model_version           TEXT NOT NULL,
    p_home_win              REAL NOT NULL CHECK (p_home_win BETWEEN 0 AND 1),
    p_away_win              REAL NOT NULL CHECK (p_away_win BETWEEN 0 AND 1),
    features_json           TEXT NOT NULL,
    warnings_json           TEXT NOT NULL,
    actual_outcome          TEXT,
    outcome_recorded_at_utc TEXT,
    UNIQUE (prediction_kind, game_id, model_version, data_snapshot_sha256)
);
DROP TRIGGER IF EXISTS predictions_are_append_only;
CREATE TRIGGER predictions_are_append_only
BEFORE UPDATE OF prediction_kind, game_id, kickoff_utc, predicted_at_utc, model_version,
                 p_home_win, p_away_win, features_json, data_snapshot_sha256 ON predictions
BEGIN
    SELECT RAISE(ABORT, 'stored predictions cannot be edited');
END;
"""


class ProspectiveForecastError(ValueError):
    """Tried to file a 'before the event' forecast for a game that already happened."""


@contextmanager
def connect(db_path: Path = DB_PATH):
    """Open the database (creating tables if needed), commit on success, always close."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        _migrate(con)
        con.executescript(SCHEMA)
        with con:  # transaction: commit on success, roll back on error
            yield con
    finally:
        con.close()


def _migrate(con: sqlite3.Connection) -> None:
    """Bring databases created by older versions up to date (adds columns only; never deletes)."""
    cols = {row[1] for row in con.execute("PRAGMA table_info(predictions)")}
    if cols and "kickoff_utc" not in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN kickoff_utc TEXT")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def save_predictions(preds: pd.DataFrame, kind: str, run_id: str, snapshot_sha256: str,
                     db_path: Path = DB_PATH, now: datetime | None = None) -> int:
    """Insert predictions. Returns how many new rows were stored (re-runs are ignored, never overwritten)."""
    now = now or _utc_now()
    kickoff = pd.to_datetime(preds["kickoff_utc"], utc=True)
    if kind == "prospective":
        late = preds[(preds["status"] != "scheduled") | kickoff.isna() | (kickoff <= pd.Timestamp(now))]
        if len(late):
            raise ProspectiveForecastError(
                f"{len(late)} game(s) already played, already kicked off, or with unknown kickoff "
                f"(e.g. {late['game_id'].iloc[0]}); file those as 'backtest', not 'prospective'."
            )
    elif kind != "backtest":
        raise ValueError(f"kind must be 'backtest' or 'prospective', got {kind!r}")

    stamp = now.isoformat(timespec="seconds")
    rows = [
        (run_id, kind, r.game_id, int(r.season), str(r.gameday), k, stamp, snapshot_sha256, r.model_version,
         float(r.p_home_win), float(r.p_away_win), r.features_json, r.warnings_json,
         None if kind == "prospective" or pd.isna(r.actual_outcome) else r.actual_outcome,
         None if kind == "prospective" or pd.isna(r.actual_outcome) else stamp)
        for r, k in zip(preds.itertuples(index=False), kickoff.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    ]
    with connect(db_path) as con:
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO predictions
               (run_id, prediction_kind, game_id, season, gameday, kickoff_utc, predicted_at_utc, data_snapshot_sha256,
                model_version, p_home_win, p_away_win, features_json, warnings_json,
                actual_outcome, outcome_recorded_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        return con.total_changes - before


def record_outcomes(games: pd.DataFrame, db_path: Path = DB_PATH, now: datetime | None = None) -> int:
    """Fill in actual results for prospective forecasts whose games have finished."""
    stamp = (now or _utc_now()).isoformat(timespec="seconds")
    finals = games[games["status"] == "final"][["game_id", "outcome"]]
    with connect(db_path) as con:
        before = con.total_changes
        con.executemany(
            """UPDATE predictions SET actual_outcome = ?, outcome_recorded_at_utc = ?
               WHERE game_id = ? AND prediction_kind = 'prospective' AND actual_outcome IS NULL""",
            [(o, stamp, g) for g, o in finals.itertuples(index=False)],
        )
        return con.total_changes - before


def load_predictions(db_path: Path = DB_PATH, kind: str | None = None) -> pd.DataFrame:
    with connect(db_path) as con:
        sql, params = "SELECT * FROM predictions", ()
        if kind:
            sql, params = sql + " WHERE prediction_kind = ?", (kind,)
        return pd.read_sql_query(sql + " ORDER BY gameday, game_id, model_version", con, params=params)
