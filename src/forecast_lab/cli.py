"""Command-line entry point:  forecast-lab <command>   (or: python -m forecast_lab <command>)

    ingest     download a fresh raw snapshot (kept only if the bytes changed)
    clean      validate + standardize a snapshot, show data-quality issues
    backtest   replay history with every model; score on a chosen split
    forecast   file pre-kickoff ESTIMATES (all models) for upcoming games; record finished ones
    scorecard  score live forecasts against what happened
    snapshots  list stored raw snapshots
    tune-elo   choose Elo settings using TRAIN seasons only
    simulate   Monte Carlo the rest of a season: win totals, division and playoff odds
    sim-backtest  check the simulator against past seasons
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import __version__, config
from .models import MODELS
from .clean import clean_snapshot, load_clean_games
from .elo import EloParams, tune_elo
from .ingest import find_snapshot, list_snapshots, snapshot_source
from .metrics import check_test_lock, evaluate, evaluate_by_season
from .scorecard import calibration_table, scorecard
from .simulate import backtest_simulator, simulate_season, summarize_backtest
from .sources import NFLVERSE_SCHEDULES
from .store import load_predictions, record_outcomes, save_predictions

DISCLAIMER = ("Forecasts are statistical ESTIMATES for research and education, not facts. "
              "This tool does not support or recommend betting.")
SPLITS = {"train": config.TRAIN_SEASONS, "validation": config.VALIDATION_SEASONS, "test": config.TEST_SEASONS}


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _journal(entry: dict, path: Path = config.JOURNAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "code_version": __version__, **entry}
    with path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _get_clean(sha_prefix: str | None) -> tuple[pd.DataFrame, dict]:
    """Load the clean table for a snapshot, cleaning it first if needed."""
    raw_manifest = find_snapshot(NFLVERSE_SCHEDULES.source_id, sha_prefix)
    clean_dir = config.CLEAN_DIR / raw_manifest["source_id"] / raw_manifest["sha256"][:12] / f"cleaner-{__version__}"
    manifest_path = clean_dir / "manifest.json"
    if manifest_path.exists():
        clean_manifest = json.loads(manifest_path.read_text())
    else:
        clean_manifest = clean_snapshot(raw_manifest)
    return load_clean_games(clean_dir / "games.csv"), clean_manifest


def cmd_ingest(args) -> None:
    manifest, is_new = snapshot_source(NFLVERSE_SCHEDULES)
    verb = "Stored NEW snapshot" if is_new else "Source unchanged; reusing snapshot"
    print(f"{verb}: {manifest['file']}  ({manifest['n_lines']:,} lines, sha256 {manifest['sha256'][:12]})")
    print(f"License: {manifest['license']}. {manifest['attribution']}")


def cmd_snapshots(args) -> None:
    for m in list_snapshots(NFLVERSE_SCHEDULES.source_id):
        print(f"{m['sha256'][:12]}  retrieved {m['retrieved_at_utc']}  {m['n_lines']:>6,} lines  {m['file']}")


def cmd_clean(args) -> None:
    raw_manifest = find_snapshot(NFLVERSE_SCHEDULES.source_id, args.snapshot)
    m = clean_snapshot(raw_manifest)
    print(f"Cleaned snapshot {m['raw_sha256'][:12]} (as_of {m['as_of']}) -> {m['n_rows']:,} games")
    print(f"  status counts: {m['status_counts']}")
    print(f"  issues: {m['n_errors']} errors (rows excluded), {m['n_warnings']} warnings (rows kept, flagged)")
    issues = pd.read_csv(Path(m["path"]) / "issues.csv")
    if len(issues):
        print("\nIssue summary:")
        print(issues.groupby(["severity", "check"]).size().to_string())
    print(f"\nclean sha256 {m['clean_sha256'][:12]}  ->  {m['path']}")


def _fmt(report: dict) -> dict:
    row = {k: report[k] for k in ("model_version", "n_games", "n_ties_excluded")}
    for k in ("accuracy", "brier", "log_loss"):
        lo, hi = report[f"{k}_ci95"]
        row[k] = f"{report[k]:.4f} [{lo:.4f}, {hi:.4f}]"
    return row


def cmd_backtest(args) -> None:
    games, cm = _get_clean(args.snapshot)
    seasons = SPLITS[args.split]
    if args.split == "test":
        print("!! Evaluating the LOCKED TEST SET. This is logged in the journal. Do not tune afterwards.")
    run_id = _new_run_id()
    reports, by_season = [], []
    for name, model in MODELS.items():
        preds = model(games)
        n_new = save_predictions(preds, "backtest", run_id, cm["raw_sha256"])
        report = evaluate(preds, seasons, unlock_test=args.unlock_test)
        reports.append(report)
        by_season.append(evaluate_by_season(preds, seasons, unlock_test=args.unlock_test).assign(model=name))
        print(f"{name}: {len(preds):,} walk-forward predictions ({n_new:,} newly stored)")

    print(f"\n== {args.split} seasons {min(seasons)}-{max(seasons)} | snapshot {cm['raw_sha256'][:12]} ==")
    print("metric [95% bootstrap interval]  (Brier/log loss: lower is better)")
    print(pd.DataFrame([_fmt(r) for r in reports]).to_string(index=False))
    print("\nBy season (Brier):")
    print(pd.concat(by_season).pivot(index="season", columns="model", values="brier").round(4).to_string())

    _journal({"run_id": run_id, "command": "backtest", "split": args.split, "unlocked_test": args.unlock_test,
              "raw_sha256": cm["raw_sha256"], "clean_sha256": cm["clean_sha256"], "reports": reports})
    print(f"\nLogged run {run_id} to {config.JOURNAL_PATH}")


def cmd_forecast(args) -> None:
    """File pre-kickoff forecasts from EVERY model, so baselines are scored on exactly the same games."""
    games, cm = _get_clean(args.snapshot)
    now = datetime.now(timezone.utc)
    n_scored = record_outcomes(games)
    if n_scored:
        print(f"Recorded actual outcomes for {n_scored} earlier prospective forecast(s).")

    as_of = datetime.fromisoformat(cm["raw_retrieved_at_utc"])
    if (now - as_of).total_seconds() > 24 * 3600:
        print(f"WARNING: snapshot is from {as_of:%Y-%m-%d %H:%M} UTC. Run `forecast-lab ingest` for fresh data.")

    run_id = _new_run_id()
    stored, shown = {}, None
    for name, model in MODELS.items():
        preds = model(games)
        upcoming = preds[(preds["status"] == "scheduled") & (preds["kickoff_utc"] > pd.Timestamp(now))]
        if upcoming.empty:
            print("No upcoming games (before kickoff) in this snapshot.")
            return
        upcoming = upcoming[upcoming["gameday"] <= upcoming["gameday"].min() + timedelta(days=args.days)]
        stored[upcoming["model_version"].iloc[0]] = save_predictions(upcoming, "prospective", run_id, cm["raw_sha256"], now=now)
        if name == args.show:
            shown = upcoming

    print(DISCLAIMER + "\n")
    show = shown.merge(games[["game_id", "home_team", "away_team"]], on="game_id")
    show["kickoff (ET)"] = show["kickoff_utc"].dt.tz_convert("America/New_York").dt.strftime("%a %b %d %H:%M")
    show = show[["kickoff (ET)", "away_team", "home_team", "p_home_win", "p_away_win", "warnings_json"]]
    print(f"Model {shown['model_version'].iloc[0]}:")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nStored new prospective forecasts (run {run_id}):")
    for version, n in stored.items():
        print(f"  {version:<22} {n:>3} new  (0 = already filed from this same data snapshot)")
    export = _export_run(run_id)
    if export:
        print(f"Exported this run's records to {export} (commit it to timestamp the forecasts publicly).")
    _journal({"run_id": run_id, "command": "forecast", "stored": stored, "n_games": len(shown),
              "raw_sha256": cm["raw_sha256"], "export": str(export.relative_to(config.PROJECT_ROOT)) if export else None})


def _export_run(run_id: str) -> Path | None:
    """Write the rows stored by one forecast run to forecasts/<season>/<run_id>.csv (tracked in git)."""
    rows = load_predictions(kind="prospective")
    rows = rows[rows["run_id"] == run_id].drop(columns=["actual_outcome", "outcome_recorded_at_utc"])
    if rows.empty:
        return None
    out = config.PROJECT_ROOT / "forecasts" / str(int(rows["season"].max())) / f"{run_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False, lineterminator="\n")
    return out


def cmd_scorecard(args) -> None:
    table, notes = scorecard()
    if table.empty:
        print("No live forecasts yet. Run `forecast-lab forecast` before games kick off.")
        return
    print("Live forecasts filed BEFORE kickoff, scored after the games (latest pre-kickoff forecast per game).")
    print("Brier / log loss: lower is better. [95% bootstrap interval]; intervals need ~10+ scored games.\n")
    show = table.copy()
    for m in ("accuracy", "brier", "log_loss"):
        if m in show:
            show[m] = [f"{v:.4f} [{lo:.3f}, {hi:.3f}]" if pd.notna(v) else "-" for v, (lo, hi) in
                       zip(show[m], show[f"{m}_ci95"].map(lambda x: x if isinstance(x, tuple) else (float("nan"),) * 2))]
    cols = [c for c in ["model_version", "forecast_games", "scored", "pending", "ties", "accuracy", "brier", "log_loss"] if c in show]
    print(show[cols].to_string(index=False))
    print("\nPre-registered expectation (journal/PREREGISTRATION.md): elo Brier about 0.22, beating both baselines over the season.")
    cal = calibration_table()
    if not cal.empty:
        print("\nCalibration (elo): favorite's forecast probability vs how often favorites actually won")
        print(cal.round(3).to_string())
    if notes["unverifiable_no_kickoff"]:
        print(f"\nExcluded {notes['unverifiable_no_kickoff']} early record(s) without a kickoff time (filed by v0.1).")


def cmd_records(args) -> None:
    df = load_predictions(kind=args.kind)
    print(df.groupby(["prediction_kind", "model_version"]).agg(
        n=("game_id", "size"), scored=("actual_outcome", "count")).to_string())


def cmd_tune_elo(args) -> None:
    games, cm = _get_clean(args.snapshot)
    seasons = [s for s in config.TRAIN_SEASONS if s >= 2002]  # 1999-2001 = rating warm-up
    grid = {"k": [10, 15, 20, 25, 30], "home_field": [30, 40, 48, 55, 65], "season_revert": [0.2, 1 / 3, 0.5]}
    print(f"Scoring {5 * 5 * 3} Elo settings on TRAIN seasons {seasons[0]}-{seasons[-1]} (validation/test untouched)...")
    table = tune_elo(games, seasons, grid)
    print(table.head(10).round(4).to_string(index=False))
    print(f"\nCurrent defaults: {EloParams().to_dict()}")
    _journal({"command": "tune-elo", "raw_sha256": cm["raw_sha256"], "seasons": f"{seasons[0]}-{seasons[-1]}",
              "top5": table.head(5).round(5).to_dict("records")})


def cmd_simulate(args) -> None:
    games, cm = _get_clean(args.snapshot)
    cutoff = datetime.fromisoformat(cm["as_of"]).date()  # snapshot date -> reproducible
    res = simulate_season(games, args.season, cutoff, model=args.model, n_sims=args.sims, seed=args.seed, hot=args.hot)
    m = res.meta
    print(DISCLAIMER)
    print(f"\n{args.season} season simulated {m['n_sims']:,} times from {m['cutoff']} "
          f"({m['games_known']} games played, {m['games_simulated']} simulated; model {m['model']}, seed {m['seed']})")
    print(f"Limitations: {m['limitations']}.")
    for w in m["warnings"]:
        print(f"WARNING: {w}")
    show = res.table.copy()
    show["record"] = show["wins"].astype(str) + "-" + show["losses"].astype(str) + show["ties"].map(lambda x: f"-{x}" if x else "")
    show["wins 10-90%"] = show["wins_p10"].map("{:g}".format) + "-" + show["wins_p90"].map("{:g}".format)
    for c in ("p_division", "p_playoffs", "p_top_seed"):
        show[c] = (100 * show[c]).map(lambda x: "<1%" if 0 < x < 1 else f"{x:.0f}%")
    cols = ["team", "record", "elo_now", "mean_wins", "wins 10-90%", "p_division", "p_playoffs", "p_top_seed"]
    for div, d in show.groupby("division", sort=True):
        print(f"\n{div}")
        print(d[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    run_id = _new_run_id()
    out_dir = config.DATA_DIR / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out_dir / f"{run_id}.csv", index=False)
    meta = {**m, "run_id": run_id, "raw_sha256": cm["raw_sha256"], "clean_sha256": cm["clean_sha256"]}
    (out_dir / f"{run_id}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    _journal({"command": "simulate", **meta})
    print(f"\nSaved {out_dir / (run_id + '.csv')}  (Monte Carlo noise on playoff odds <= "
          f"{100 * res.table['mc_se_playoffs'].max():.1f} pts)")


def cmd_sim_backtest(args) -> None:
    games, cm = _get_clean(args.snapshot)
    seasons = SPLITS[args.split]
    check_test_lock(seasons, args.unlock_test)
    print(f"Re-running the simulator after week {args.after_week} of seasons {min(seasons)}-{max(seasons)}, "
          f"using only information available at that point...")
    bt = backtest_simulator(games, seasons, args.after_week, n_sims=args.sims, unlock_test=args.unlock_test)
    summary = summarize_backtest(bt)
    print("\nplayoff_brier: lower is better. wins_mae: average miss on final wins.")
    print("actual_wins_inside_p10_p90 should be about 0.80 if the uncertainty is honest.")
    print(summary.round(4).to_string(index=False))
    elo = bt[bt["model"] == "elo"]
    bins = pd.cut(elo["p_playoffs"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    print("\nCalibration (elo): when the simulator said X%, how often did teams make it?")
    print(elo.groupby(bins, observed=True).agg(n=("made_playoffs", "size"), predicted=("p_playoffs", "mean"),
                                                 actual=("made_playoffs", "mean")).round(3).to_string())
    _journal({"command": "sim-backtest", "split": args.split, "after_week": args.after_week, "unlocked_test": args.unlock_test,
              "raw_sha256": cm["raw_sha256"], "summary": summary.to_dict("records")})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forecast-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="download a raw snapshot").set_defaults(func=cmd_ingest)
    sub.add_parser("snapshots", help="list raw snapshots").set_defaults(func=cmd_snapshots)
    c = sub.add_parser("clean", help="validate a snapshot")
    c.add_argument("--snapshot", help="sha256 prefix (default: latest)")
    c.set_defaults(func=cmd_clean)
    b = sub.add_parser("backtest", help="walk-forward baseline evaluation")
    b.add_argument("--snapshot", help="sha256 prefix (default: latest)")
    b.add_argument("--split", choices=list(SPLITS), default="validation")
    b.add_argument("--unlock-test", action="store_true", help="allow scoring the locked test seasons")
    b.set_defaults(func=cmd_backtest)
    f = sub.add_parser("forecast", help="pre-game estimates for upcoming games")
    f.add_argument("--snapshot", help="sha256 prefix (default: latest)")
    f.add_argument("--show", choices=list(MODELS), default="elo", help="which model's forecasts to print (all are filed)")
    f.add_argument("--days", type=int, default=7, help="forecast games within N days of the next game")
    f.set_defaults(func=cmd_forecast)
    sub.add_parser("scorecard", help="score live pre-kickoff forecasts").set_defaults(func=cmd_scorecard)
    r = sub.add_parser("records", help="summarize stored prediction records")
    r.add_argument("--kind", choices=["backtest", "prospective"])
    r.set_defaults(func=cmd_records)
    t = sub.add_parser("tune-elo", help="choose Elo settings on train seasons")
    t.add_argument("--snapshot", help="sha256 prefix (default: latest)")
    t.set_defaults(func=cmd_tune_elo)
    s = sub.add_parser("simulate", help="Monte Carlo season simulation")
    s.add_argument("--snapshot", help="sha256 prefix (default: latest)")
    s.add_argument("--season", type=int, default=2026)
    s.add_argument("--sims", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model", choices=["elo", "coin_flip"], default="elo")
    s.add_argument("--hot", action="store_true", help="update ratings inside each simulation (wider ranges)")
    s.set_defaults(func=cmd_simulate)
    sb = sub.add_parser("sim-backtest", help="check the simulator on past seasons")
    sb.add_argument("--snapshot", help="sha256 prefix (default: latest)")
    sb.add_argument("--split", choices=list(SPLITS), default="validation")
    sb.add_argument("--after-week", type=int, default=8)
    sb.add_argument("--sims", type=int, default=2000)
    sb.add_argument("--unlock-test", action="store_true")
    sb.set_defaults(func=cmd_sim_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "split", None) == "test" and not args.unlock_test:
        print("The test seasons are locked. Re-run with --unlock-test only for a final evaluation.", file=sys.stderr)
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
