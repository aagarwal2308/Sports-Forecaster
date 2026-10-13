"""Read-only data access for the dashboard.

Everything here only READS files and the prediction database. Nothing
downloads data, files forecasts, or edits records: the dashboard must
never be able to change the live, pre-registered test.

Kept separate from the Streamlit page so it can be unit-tested without a browser.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .baseline import model_version, predict_home_rate
from .calibration import reliability_table
from .clean import load_clean_games
from .config import TRAIN_SEASONS, VALIDATION_SEASONS
from .elo import EloParams
from .ingest import list_snapshots
from .scorecard import eligible_forecasts
from .sources import NFLVERSE_SCHEDULES
from .store import load_predictions

HISTORY_SEASONS = TRAIN_SEASONS + VALIDATION_SEASONS  # test seasons stay locked, so they are never shown
EASTERN = "America/New_York"


def current_versions() -> dict[str, str]:
    """Version string (with fingerprint) of each model as the code stands today."""
    home_rate_defaults = {k: v.default for k, v in inspect.signature(predict_home_rate).parameters.items() if k != "games"}
    return {
        "elo": model_version("elo", EloParams().to_dict()),
        "home_rate": model_version("home_rate", home_rate_defaults),
        "coin_flip": model_version("coin_flip"),
    }


def model_name(version: str) -> str:
    """'elo-1.0+208703b4' -> 'elo'"""
    return version.split("-")[0]


def load_clean(data_dir: Path) -> tuple[pd.DataFrame | None, dict | None, Path | None]:
    """Latest snapshot's clean games, its manifest, and its folder. (None, raw manifest, None) if not cleaned yet."""
    snaps = list_snapshots(NFLVERSE_SCHEDULES.source_id, Path(data_dir) / "raw")
    if not snaps:
        return None, None, None
    raw = snaps[-1]
    folder = Path(data_dir) / "clean" / raw["source_id"] / raw["sha256"][:12] / f"cleaner-{__version__}"
    if not (folder / "games.csv").exists():
        return None, raw, None
    return load_clean_games(folder / "games.csv"), json.loads((folder / "manifest.json").read_text()), folder


def _db_ready(db: Path) -> bool:
    return Path(db).exists()


def upcoming_forecasts(db: Path, games: pd.DataFrame, now: pd.Timestamp, version: str) -> pd.DataFrame:
    """The latest stored pre-kickoff forecast of `version` for each game that hasn't started yet."""
    if not _db_ready(db):
        return pd.DataFrame()
    df = load_predictions(db, "prospective")
    df = df[(df["model_version"] == version) & df["kickoff_utc"].notna()].copy()
    df["kickoff"] = pd.to_datetime(df["kickoff_utc"], utc=True, format="ISO8601")
    df["filed"] = pd.to_datetime(df["predicted_at_utc"], utc=True, format="ISO8601")
    df = df[df["kickoff"] > now].sort_values("filed").groupby("game_id", as_index=False).last()
    if df.empty:
        return pd.DataFrame()
    df = df.merge(games[["game_id", "away_team", "home_team", "week"]], on="game_id", how="left")
    df["warnings"] = df["warnings_json"].map(lambda w: ", ".join(json.loads(w)) or "none")
    out = pd.DataFrame({
        "kickoff (ET)": df["kickoff"].dt.tz_convert(EASTERN).dt.strftime("%a %b %d, %I:%M %p"),
        "week": df["week"],
        "away": df["away_team"], "home": df["home_team"],
        "P(home win)": df["p_home_win"], "P(away win)": df["p_away_win"],
        "filed (ET)": df["filed"].dt.tz_convert(EASTERN).dt.strftime("%a %b %d, %I:%M %p"),
        "data warnings": df["warnings"],
        "_kickoff": df["kickoff"],
    })
    return out.sort_values(["_kickoff", "home"]).drop(columns="_kickoff").reset_index(drop=True)


def finished_forecasts(db: Path, games: pd.DataFrame, versions: dict[str, str]) -> pd.DataFrame:
    """Scored live forecasts (latest pre-kickoff per game), one row per game and model."""
    if not _db_ready(db):
        return pd.DataFrame()
    latest, _ = eligible_forecasts(db)
    latest = latest[latest["model_version"].isin(versions.values()) & latest["actual_outcome"].isin(["home_win", "away_win"])]
    if latest.empty:
        return pd.DataFrame()
    d = latest.merge(games[["game_id", "away_team", "home_team", "home_score", "away_score"]], on="game_id", how="left")
    d["model"] = d["model_version"].map(model_name)
    d["y"] = (d["actual_outcome"] == "home_win").astype(float)
    d["brier"] = (d["p_home_win"] - d["y"]) ** 2
    d["favorite_won"] = np.where(d["p_home_win"] >= 0.5, d["y"] == 1, d["y"] == 0)
    return d.sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def running_brier(finished: pd.DataFrame) -> pd.DataFrame:
    """Cumulative Brier score after each finished game, per model (lower is better)."""
    if finished.empty:
        return pd.DataFrame()
    rows = []
    for model, d in finished.groupby("model"):
        d = d.sort_values(["kickoff", "game_id"])
        n = np.arange(1, len(d) + 1)
        rows.append(pd.DataFrame({"model": model, "games scored": n,
                                  "cumulative Brier": d["brier"].cumsum().to_numpy() / n,
                                  "through": d["kickoff"].dt.tz_convert(EASTERN).dt.strftime("%b %d").to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def _latest_backtests(db: Path, versions: dict[str, str]) -> pd.DataFrame:
    if not _db_ready(db):
        return pd.DataFrame()
    bt = load_predictions(db, "backtest")
    bt = bt[bt["model_version"].isin(versions.values())]
    # The same model may have been backtested on several snapshots; keep the latest per game.
    return bt.sort_values("predicted_at_utc").groupby(["game_id", "model_version"], as_index=False).last()


def history_by_season(db: Path, versions: dict[str, str]) -> pd.DataFrame:
    """Backtest Brier score per season for train + validation seasons only (test seasons stay locked)."""
    bt = _latest_backtests(db, versions)
    if bt.empty:
        return pd.DataFrame()
    bt = bt[bt["season"].isin(HISTORY_SEASONS) & bt["actual_outcome"].isin(["home_win", "away_win"])]
    y = (bt["actual_outcome"] == "home_win").astype(float)
    bt = bt.assign(brier=(bt["p_home_win"] - y) ** 2, model=bt["model_version"].map(model_name))
    out = bt.groupby(["season", "model"], as_index=False).agg(games=("brier", "size"), brier=("brier", "mean"))
    out["period"] = np.where(out["season"].isin(VALIDATION_SEASONS), "validation", "train")
    return out


def history_calibration(db: Path, versions: dict[str, str], model: str = "elo") -> pd.DataFrame:
    """Reliability table on validation seasons, from stored backtests."""
    bt = _latest_backtests(db, versions)
    if bt.empty:
        return pd.DataFrame()
    return reliability_table(bt[bt["model_version"] == versions[model]], VALIDATION_SEASONS)


def latest_simulation(data_dir: Path) -> tuple[pd.DataFrame | None, dict | None]:
    folder = Path(data_dir) / "simulations"
    metas = sorted(folder.glob("*.json")) if folder.exists() else []
    for meta_path in reversed(metas):
        csv = meta_path.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv), json.loads(meta_path.read_text())
    return None, None


def recent_runs(journal_dir: Path, limit: int = 25) -> pd.DataFrame:
    path = Path(journal_dir) / "experiments.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keep = ["logged_at_utc", "command", "run_id", "split", "raw_sha256", "code_version"]
    df = pd.DataFrame(rows)
    return df[[c for c in keep if c in df.columns]].tail(limit).iloc[::-1].reset_index(drop=True)
