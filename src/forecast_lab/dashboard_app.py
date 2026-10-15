"""The research dashboard (a Streamlit page). Start it with:  forecast-lab dashboard

READ-ONLY by design: it displays data, forecasts and results, but has no
buttons that download data or file forecasts. Filing happens only through
the scheduled `forecast-lab forecast` runs, which keeps the live test honest.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from forecast_lab import config
from forecast_lab import dashboard_data as dd
from forecast_lab.calibration import expected_calibration_error

DATA_DIR = Path(os.environ.get("FORECAST_LAB_DATA_DIR", config.DATA_DIR))
JOURNAL_DIR = Path(os.environ.get("FORECAST_LAB_JOURNAL_DIR", config.PROJECT_ROOT / "journal"))
DB = DATA_DIR / "lab.sqlite"

st.set_page_config(page_title="NFL Forecasting Lab", page_icon="🏈", layout="wide")

# Validated colorblind-safe palette (slots 1-3), with separate steps for dark mode.
DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
COLORS = ({"elo": "#3987e5", "home_rate": "#d95926", "coin_flip": "#199e70"} if DARK
          else {"elo": "#2a78d6", "home_rate": "#eb6834", "coin_flip": "#1baf7a"})
REFERENCE = "#8a8984"
MODEL_ORDER = ["elo", "home_rate", "coin_flip"]
color_scale = alt.Scale(domain=MODEL_ORDER, range=[COLORS[m] for m in MODEL_ORDER])


@st.cache_data(ttl=300, show_spinner="Loading data...")
def load_everything(data_dir: str, db: str, journal_dir: str):
    games, manifest, clean_folder = dd.load_clean(Path(data_dir))
    versions = dd.current_versions()
    return games, manifest, clean_folder, versions


st.title("🏈 NFL Forecasting Lab")
st.info("**Every probability on this page is a statistical estimate, not a fact.** "
        "This is a research and learning project. It does not give betting advice. "
        "The page is read-only: forecasts are filed only by the scheduled Tuesday/Thursday runs.")

games, manifest, clean_folder, versions = load_everything(str(DATA_DIR), str(DB), str(JOURNAL_DIR))
if games is None:
    if manifest is None:
        st.warning("No data yet. In Terminal, run `forecast-lab ingest`, then `forecast-lab clean`.")
    else:
        st.warning("The latest data download hasn't been cleaned yet. In Terminal, run `forecast-lab clean`.")
    st.stop()

st.caption(f"Data snapshot `{manifest['raw_sha256'][:12]}`, downloaded {manifest['raw_retrieved_at_utc']} (UTC). "
           f"{manifest['attribution']}")

now = pd.Timestamp.now(tz="UTC")
tab_week, tab_score, tab_history, tab_sim, tab_data, tab_journal = st.tabs(
    ["This week", "Live scorecard", "Model history", "Season simulator", "Data explorer", "Journal"])

# ---------------------------------------------------------------- This week
with tab_week:
    st.subheader("Upcoming games: Elo forecasts")
    st.write(f"Model **{versions['elo']}**, frozen for the 2026 live test. Each forecast was filed before kickoff; "
             "the bar shows the estimated chance the home team wins.")
    upcoming = dd.upcoming_forecasts(DB, games, now, versions["elo"])
    if upcoming.empty:
        st.info("No upcoming forecasts are stored. They appear after the next scheduled run "
                "(Tuesday/Thursday 9:00 AM), or after running `forecast-lab forecast`.")
    else:
        shown = upcoming.assign(**{c: (upcoming[c] * 100).round() for c in ["P(home win)", "P(away win)"]})
        st.dataframe(shown, hide_index=True, column_config={
            "P(home win)": st.column_config.ProgressColumn("P(home win)", format="%d%%", min_value=0, max_value=100,
                                                            color=COLORS["elo"]),
            "P(away win)": st.column_config.NumberColumn("P(away win)", format="%d%%"),
        })
        st.caption("A 60% estimate means: in many games like this, the home team would be expected to win about 6 in 10. "
                   "The other 4 in 10 are not 'upsets the model missed'. They are part of what 60% means.")

# ---------------------------------------------------------------- Live scorecard
with tab_score:
    st.subheader("How the live forecasts are doing")
    finished = dd.finished_forecasts(DB, games, versions)
    if finished.empty:
        st.info("No forecasted games have finished yet. Results appear here after each game ends "
                "and the next scheduled run records it.")
    else:
        summary = finished.groupby("model").agg(games=("brier", "size"), brier=("brier", "mean"),
                                                 favorite_won=("favorite_won", "mean")).reindex(MODEL_ORDER).dropna()
        cols = st.columns(len(summary))
        for col, (model, row) in zip(cols, summary.iterrows()):
            col.metric(f"{model} Brier score", f"{row['brier']:.3f}", help="Lower is better. Coin flip = 0.250.")
            col.caption(f"{int(row['games'])} games · favorite won {row['favorite_won']:.0%}")
        st.write("**Cumulative Brier score** (lower is better; the dashed line is a 50/50 coin flip). "
                 "Expect big swings early: a few games prove very little.")
        rb = dd.running_brier(finished)
        line = alt.Chart(rb).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=alt.X("games scored:Q", title="games scored"),
            y=alt.Y("cumulative Brier:Q", title="cumulative Brier score", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=color_scale, legend=alt.Legend(title="model", orient="top")),
            tooltip=["model", "games scored", alt.Tooltip("cumulative Brier:Q", format=".3f"), "through"])
        coin = alt.Chart(pd.DataFrame({"y": [0.25]})).mark_rule(strokeDash=[4, 3], color=REFERENCE).encode(y="y:Q")
        st.altair_chart(line + coin, width="stretch")
        with st.expander("Every scored game"):
            table = finished[finished["model"] == "elo"][["kickoff", "away_team", "home_team", "away_score",
                                                          "home_score", "p_home_win", "actual_outcome", "brier"]].copy()
            table["kickoff"] = table["kickoff"].dt.tz_convert(dd.EASTERN).dt.strftime("%b %d")
            st.dataframe(table, hide_index=True, column_config={
                "p_home_win": st.column_config.NumberColumn("Elo P(home win)", format="percent"),  # 0-1 scale
                "brier": st.column_config.NumberColumn("Brier", format="%.3f")})
    st.caption("Scoring rules were fixed before the season in journal/PREREGISTRATION.md. "
               "The verdict is judged at season end, not week to week.")

# ---------------------------------------------------------------- Model history
with tab_history:
    st.subheader("How each model did on past seasons")
    st.write("Each model replayed every season, using only games played before each forecast. "
             "Seasons 2021–2025 are the **locked test set** and are deliberately not shown.")
    hist = dd.history_by_season(DB, versions)
    if hist.empty:
        st.info("No backtests stored for the current model versions. Run `forecast-lab backtest`.")
    else:
        chart = alt.Chart(hist).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50, filled=True)).encode(
            x=alt.X("season:O", title="season"),
            y=alt.Y("brier:Q", title="Brier score (lower is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=color_scale, legend=alt.Legend(title="model", orient="top")),
            tooltip=["season", "model", "period", "games", alt.Tooltip("brier:Q", format=".4f")])
        st.altair_chart(chart, width="stretch")
        val = hist[hist["period"] == "validation"].groupby("model")["brier"].mean().reindex(MODEL_ORDER).dropna()
        st.write("Average on validation seasons 2016–2020: " +
                 " · ".join(f"**{m}** {b:.4f}" for m, b in val.items()))
        with st.expander("Table"):
            st.dataframe(hist.pivot(index="season", columns="model", values="brier").round(4))

        st.subheader("Calibration: does 70% mean 70%?")
        cal = dd.history_calibration(DB, versions)
        if not cal.empty:
            st.write(f"Elo on validation seasons (ECE {expected_calibration_error(cal):.3f}). Each dot groups games by "
                     "Elo's forecast; the dot's height is how often the home team really won. On the dashed diagonal = "
                     "perfectly calibrated. Bars are 95% intervals: wide where there are few games.")
            base = alt.Chart(cal)
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[4, 3], color=REFERENCE).encode(x="x:Q", y="y:Q")
            bars = base.mark_rule(strokeWidth=2, color=COLORS["elo"]).encode(
                x=alt.X("mean_forecast:Q", title="Elo forecast P(home win)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("ci_low:Q", title="share of games the home team won", scale=alt.Scale(domain=[0, 1])),
                y2="ci_high:Q")
            dots = base.mark_circle(size=90, color=COLORS["elo"], opacity=1).encode(
                x="mean_forecast:Q", y="observed:Q",
                tooltip=[alt.Tooltip("bin:N", title="forecast bin"), alt.Tooltip("n:Q", title="games"),
                         alt.Tooltip("mean_forecast:Q", title="average forecast", format=".1%"),
                         alt.Tooltip("observed:Q", title="home team won", format=".1%")])
            st.altair_chart(diag + bars + dots, width="stretch")

# ---------------------------------------------------------------- Season simulator
with tab_sim:
    st.subheader("Season simulator: playoff chances")
    sim, meta = dd.latest_simulation(DATA_DIR)
    if sim is None:
        st.info("No simulation saved yet. It runs every Tuesday, or run `forecast-lab simulate`.")
    else:
        st.write(f"The rest of the **{meta['season']}** season was played out **{meta['n_sims']:,} times** from "
                 f"{meta['cutoff']} ({meta['games_known']} games played, {meta['games_simulated']} simulated).")
        st.caption(f"Limitations: {meta['limitations']}.")
        conf = st.radio("Conference", ["AFC", "NFC"], horizontal=True)
        s = sim[sim["conference"] == conf].copy()
        s["record"] = s["wins"].astype(str) + "-" + s["losses"].astype(str) + s["ties"].map(lambda t: f"-{t}" if t else "")
        s["win range (10–90%)"] = s["wins_p10"].map("{:g}".format) + "–" + s["wins_p90"].map("{:g}".format)
        # "0%" would read as "impossible"; tiny-but-possible chances are shown as "<1%".
        s["playoff label"] = s["p_playoffs"].map(lambda x: "<1%" if 0 < x < 0.005 else f"{x:.0%}")
        bars = alt.Chart(s).mark_bar(color=COLORS["elo"], cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("team:N", sort="-x", title=None),
            x=alt.X("p_playoffs:Q", title="chance of making the playoffs", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=["team", "division", "record", alt.Tooltip("mean_wins:Q", title="average final wins", format=".1f"),
                     "win range (10–90%)", alt.Tooltip("p_division:Q", title="win division", format=".0%"),
                     alt.Tooltip("p_playoffs:Q", title="make playoffs", format=".0%"),
                     alt.Tooltip("p_top_seed:Q", title="#1 seed", format=".0%")])
        labels = bars.mark_text(align="left", dx=4, fontSize=11).encode(
            text="playoff label:N", color=alt.value("#52514e" if not DARK else "#c3c2b7"))
        st.altair_chart(bars + labels, width="stretch")
        with st.expander("Full table by division"):
            st.dataframe(s[["division", "team", "record", "elo_now", "mean_wins", "win range (10–90%)",
                            "p_division", "p_playoffs", "p_top_seed"]].sort_values(["division", "mean_wins"],
                                                                                   ascending=[True, False]),
                         hide_index=True, column_config={
                             "mean_wins": st.column_config.NumberColumn("avg wins", format="%.1f"),
                             "p_division": st.column_config.NumberColumn("division", format="percent"),
                             "p_playoffs": st.column_config.NumberColumn("playoffs", format="percent"),
                             "p_top_seed": st.column_config.NumberColumn("#1 seed", format="percent")})

# ---------------------------------------------------------------- Data explorer
with tab_data:
    st.subheader("Explore the game data")
    c1, c2 = st.columns(2)
    seasons = sorted(games["season"].dropna().unique().tolist(), reverse=True)
    season = c1.selectbox("Season", seasons)
    teams = ["All teams"] + sorted(set(games["home_team"]) | set(games["away_team"]))
    team = c2.selectbox("Team", teams)
    view = games[games["season"] == season]
    if team != "All teams":
        view = view[(view["home_team"] == team) | (view["away_team"] == team)]
    st.dataframe(view[["week", "game_type", "gameday", "away_team", "away_score", "home_team", "home_score",
                       "status", "neutral_site", "dq_warnings"]], hide_index=True)
    st.caption(f"{len(view)} games. Team codes are as the source recorded them (e.g. OAK before 2020); "
               "models use stable franchise ids (OAK → LV).")

    st.subheader("Data-quality checks")
    issues = pd.read_csv(clean_folder / "issues.csv") if (clean_folder / "issues.csv").exists() else pd.DataFrame()
    a, b, c = st.columns(3)
    a.metric("Games in clean data", f"{manifest['n_rows']:,}")
    b.metric("Errors (rows excluded)", manifest["n_errors"])
    c.metric("Warnings (rows flagged)", manifest["n_warnings"])
    if issues.empty:
        st.success("No data-quality issues found in this snapshot.")
    else:
        st.dataframe(issues, hide_index=True)

# ---------------------------------------------------------------- Journal
with tab_journal:
    st.subheader("Research journal")
    for name, label in [("PREREGISTRATION.md", "Pre-registration of the live test"), ("README.md", "Experiment notes")]:
        path = JOURNAL_DIR / name
        if path.exists():
            with st.expander(label, expanded=(name == "README.md")):
                st.markdown(path.read_text())
    runs = dd.recent_runs(JOURNAL_DIR)
    if not runs.empty:
        st.write("**Recent automatic runs**")
        st.dataframe(runs, hide_index=True)
