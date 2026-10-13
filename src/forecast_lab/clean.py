"""Validate and standardize raw game records.

Two severities, applied consistently:
    error   -> the row is EXCLUDED from the clean table and logged in issues.
    warning -> the row is KEPT; the warning is logged and attached to the row
               (column `dq_warnings`) so downstream predictions can carry it.
Nothing is silently fixed or filled in.

Kickoff times in the source are US Eastern ("13:00"); we convert them to UTC
(`kickoff_utc`) so pre-game forecasts can be checked against the exact start.
If an unplayed game has no time, kickoff is taken as the START of its gameday
(Eastern), which is the conservative choice: forecasts must be filed earlier.

`as_of` is the date we judge "has this game happened yet?" against. It
defaults to the snapshot's download date (not today's date), so cleaning the
same snapshot always gives byte-identical output.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .config import CLEAN_DIR, RAW_DIR
from .ingest import load_snapshot, sha256_bytes
from .teams import KNOWN_CODES, franchise_of

REQUIRED_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday",
    "away_team", "away_score", "home_team", "home_score", "location",
]
# Extra source columns we carry through unchanged (useful later as features).
OPTIONAL_COLUMNS = ["gametime", "overtime", "div_game", "away_rest", "home_rest", "roof", "surface"]
# Betting-market columns in the source are deliberately NOT carried into clean
# data: this project is a non-gambling research simulator. They remain in raw.

GAME_TYPES = {"REG", "WC", "DIV", "CON", "SB"}
LOCATIONS = {"Home": False, "Neutral": True}  # value -> neutral_site

CLEAN_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "kickoff_utc",
    "home_team", "away_team", "home_franchise", "away_franchise",
    "home_score", "away_score", "home_margin", "outcome", "status",
    "neutral_site", *OPTIONAL_COLUMNS, "dq_warnings",
]


class SchemaError(ValueError):
    """The raw file is missing columns we depend on."""


@dataclass
class CleanResult:
    games: pd.DataFrame
    issues: pd.DataFrame


def read_raw_games(data: bytes) -> pd.DataFrame:
    """Parse raw CSV with every column as text and NO automatic NA guessing,
    so we decide explicitly what counts as missing ("")."""
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)


def _to_int(series: pd.Series) -> pd.Series:
    """Text -> nullable integer. Blank -> <NA>. Garbage/decimals -> <NA> (caller flags it)."""
    num = pd.to_numeric(series.replace("", None), errors="coerce")
    whole = num.notna() & (num % 1 == 0)
    return num.where(whole).astype("Int64")


def clean_games(raw: pd.DataFrame, as_of: date) -> CleanResult:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise SchemaError(f"Raw data is missing required columns: {missing_cols}")

    df = raw.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].apply(lambda s: s.str.strip())

    issues: list[dict] = []
    excluded = pd.Series(False, index=df.index)
    row_warnings: dict[int, list[str]] = {i: [] for i in df.index}

    def flag(mask: pd.Series, severity: str, check: str, message: str) -> None:
        nonlocal excluded
        mask = mask.fillna(False).astype(bool)
        for idx in df.index[mask]:
            issues.append({"game_id": df.at[idx, "game_id"], "severity": severity, "check": check, "message": message})
            if severity == "warning":
                row_warnings[idx].append(check)
        if severity == "error":
            excluded = excluded | mask

    # --- identifiers ------------------------------------------------------
    flag(df["game_id"] == "", "error", "missing_game_id", "game_id is blank")
    dup = df["game_id"].duplicated(keep=False) & (df["game_id"] != "")
    flag(dup, "error", "duplicate_game_id", "game_id appears more than once; all copies excluded (can't tell which is right)")

    for side in ("home", "away"):
        unknown = ~df[f"{side}_team"].isin(KNOWN_CODES)
        flag(unknown, "error", f"unknown_{side}_team", f"{side}_team is not a known team code")
    flag(df["home_team"] == df["away_team"], "error", "same_team", "home_team equals away_team")

    # --- dates, season, type, location ------------------------------------
    gameday_ts = pd.to_datetime(df["gameday"], format="%Y-%m-%d", errors="coerce")
    flag(gameday_ts.isna(), "error", "bad_gameday", "gameday is missing or not YYYY-MM-DD")
    season = _to_int(df["season"])
    week = _to_int(df["week"])
    flag(season.isna(), "error", "bad_season", "season is missing or not an integer")
    flag(week.isna(), "error", "bad_week", "week is missing or not an integer")
    flag(~df["game_type"].isin(GAME_TYPES), "error", "bad_game_type", f"game_type not in {sorted(GAME_TYPES)}")
    flag(~df["location"].isin(LOCATIONS), "error", "bad_location", f"location not in {sorted(LOCATIONS)}")

    # --- scores -------------------------------------------------------------
    home_score, away_score = _to_int(df["home_score"]), _to_int(df["away_score"])
    for side, parsed in (("home", home_score), ("away", away_score)):
        unparseable = (df[f"{side}_score"] != "") & parsed.isna()
        flag(unparseable, "error", f"bad_{side}_score", f"{side}_score is not a whole number")
        flag(parsed < 0, "error", f"negative_{side}_score", f"{side}_score is negative")

    has_home, has_away = df["home_score"] != "", df["away_score"] != ""
    flag(has_home != has_away, "error", "partial_score", "only one team's score is present")

    both = has_home & has_away
    neither = ~has_home & ~has_away
    in_past = gameday_ts < pd.Timestamp(as_of)    # NaT compares False
    in_future = gameday_ts > pd.Timestamp(as_of)
    flag(both & in_future, "error", "score_for_future_game", f"score present but gameday is after as_of={as_of}")
    flag(neither & in_past, "warning", "missing_result", f"game dated before as_of={as_of} has no score")

    # --- kickoff time (US Eastern in the source) -> UTC ----------------------
    time_ok = df["gametime"].str.fullmatch(r"\d{2}:\d{2}")
    flag((df["gametime"] != "") & ~time_ok, "warning", "bad_gametime",
         "gametime is not HH:MM; kickoff taken as start of gameday (US Eastern)")
    flag((df["gametime"] == "") & ~both, "warning", "kickoff_time_unknown",
         "unplayed game has no kickoff time; kickoff taken as start of gameday (US Eastern)")
    local = pd.to_datetime(df["gameday"] + " " + df["gametime"].where(time_ok, "00:00"),
                           format="%Y-%m-%d %H:%M", errors="coerce")
    kickoff_utc = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                   .dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    margin = home_score - away_score
    source_result = _to_int(raw["result"].str.strip()) if "result" in raw.columns else pd.Series(pd.NA, index=df.index, dtype="Int64")
    flag(both & source_result.notna() & (source_result != margin), "warning", "result_mismatch",
         "source 'result' column disagrees with home_score - away_score; scores used")

    # --- assemble clean table --------------------------------------------
    status = pd.Series("scheduled", index=df.index)
    status[both] = "final"
    status[neither & in_past] = "missing_result"

    outcome = pd.Series(pd.NA, index=df.index, dtype="object")
    outcome[both & (margin > 0)] = "home_win"
    outcome[both & (margin < 0)] = "away_win"
    outcome[both & (margin == 0)] = "tie"

    out = pd.DataFrame({
        "game_id": df["game_id"],
        "season": season,
        "game_type": df["game_type"],
        "week": week,
        "gameday": gameday_ts.dt.date,
        "kickoff_utc": kickoff_utc,
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_score": home_score,
        "away_score": away_score,
        "home_margin": margin,
        "outcome": outcome,
        "status": status,
        "neutral_site": df["location"].map(LOCATIONS),
        **{c: df[c] for c in OPTIONAL_COLUMNS},
        "dq_warnings": pd.Series({i: ";".join(w) for i, w in row_warnings.items()}),
    })[~excluded].copy()

    out["home_franchise"] = out["home_team"].map(franchise_of)
    out["away_franchise"] = out["away_team"].map(franchise_of)
    out["neutral_site"] = out["neutral_site"].astype(bool)
    out = out[CLEAN_COLUMNS].sort_values(["gameday", "game_id"]).reset_index(drop=True)

    issues_df = pd.DataFrame(issues, columns=["game_id", "severity", "check", "message"])
    issues_df = issues_df.sort_values(["severity", "check", "game_id"]).reset_index(drop=True)
    return CleanResult(out, issues_df)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Deterministic CSV encoding (same frame -> same bytes on every OS)."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def clean_snapshot(manifest: dict, raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR) -> dict:
    """Clean one raw snapshot and write games.csv, issues.csv and manifest.json."""
    data = load_snapshot(manifest, raw_dir)
    as_of = datetime.fromisoformat(manifest["retrieved_at_utc"]).date()
    result = clean_games(read_raw_games(data), as_of=as_of)

    # Versioned folder: a newer cleaner never overwrites what an older one produced.
    out_dir = Path(clean_dir) / manifest["source_id"] / manifest["sha256"][:12] / f"cleaner-{__version__}"
    out_dir.mkdir(parents=True, exist_ok=True)
    games_bytes = to_csv_bytes(result.games)
    (out_dir / "games.csv").write_bytes(games_bytes)
    (out_dir / "issues.csv").write_bytes(to_csv_bytes(result.issues))

    sev = result.issues["severity"].value_counts()
    clean_manifest = {
        "source_id": manifest["source_id"],
        "raw_sha256": manifest["sha256"],
        "raw_file": manifest["file"],
        "raw_retrieved_at_utc": manifest["retrieved_at_utc"],
        "attribution": manifest["attribution"],
        "as_of": as_of.isoformat(),
        "cleaner_version": __version__,
        "clean_sha256": sha256_bytes(games_bytes),
        "n_rows": len(result.games),
        "n_errors": int(sev.get("error", 0)),
        "n_warnings": int(sev.get("warning", 0)),
        "status_counts": {k: int(v) for k, v in result.games["status"].value_counts().sort_index().items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(clean_manifest, indent=2) + "\n")
    return clean_manifest


def load_clean_games(path: Path) -> pd.DataFrame:
    """Read a clean games.csv back with proper types."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # all text; "" = missing
    df["gameday"] = pd.to_datetime(df["gameday"], format="%Y-%m-%d").dt.date
    for c in ["season", "week", "home_score", "away_score", "home_margin"]:
        df[c] = _to_int(df[c])
    df["outcome"] = df["outcome"].replace("", pd.NA)
    df["neutral_site"] = df["neutral_site"].map({"True": True, "False": False}).astype(bool)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df
