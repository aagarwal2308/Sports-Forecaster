"""Probability calibration: "when the model says 70%, does it happen ~70% of the time?"

Method: put every scored game into a bin by its forecast P(home win)
(0-10%, 10-20%, ...). In each bin compare the AVERAGE forecast with the
SHARE of games the home team actually won. A perfectly calibrated model
sits on the diagonal. Each observed share gets a 95% Wilson interval,
because bins with few games are noisy; a point is only clearly
miscalibrated if its interval misses the diagonal.

ECE (expected calibration error) = game-weighted average gap between
forecast and observed share. Smaller is better; it summarizes the plot
in one number but hides WHERE the miscalibration is, so read both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import check_test_lock, scoreable

# Reference palette slots 1-3 (validated colorblind-safe as a set), light surface.
MODEL_COLORS = {"elo": "#2a78d6", "home_rate": "#eb6834", "coin_flip": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def wilson_interval(successes: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% interval for a proportion that stays inside [0, 1] and behaves well for small n."""
    n = np.asarray(n, dtype=float)
    phat = np.divide(successes, n, out=np.zeros_like(n), where=n > 0)
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def reliability_table(preds: pd.DataFrame, seasons, n_bins: int = 10, unlock_test: bool = False) -> pd.DataFrame:
    seasons = list(seasons)
    check_test_lock(seasons, unlock_test)
    d = scoreable(preds[preds["season"].isin(seasons)])
    p = d["p_home_win"].to_numpy(dtype=float)
    y = (d["actual_outcome"] == "home_win").to_numpy(dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)  # bin index 0..n_bins-1
    rows = []
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        lo, hi = wilson_interval(np.array([y[m].sum()]), np.array([m.sum()]))
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_forecast": p[m].mean(), "observed": y[m].mean(),
                     "ci_low": float(lo[0]), "ci_high": float(hi[0])})
    t = pd.DataFrame(rows)
    t["diagonal_inside_ci"] = (t["ci_low"] <= t["mean_forecast"]) & (t["mean_forecast"] <= t["ci_high"])
    return t


def expected_calibration_error(table: pd.DataFrame) -> float:
    return float((table["n"] * (table["observed"] - table["mean_forecast"]).abs()).sum() / table["n"].sum())


def plot_reliability(tables: dict[str, pd.DataFrame], title: str, path: Path) -> Path:
    """One panel per model: reliability points with 95% intervals, game counts underneath."""
    import matplotlib
    matplotlib.use("Agg")  # draw to a file; no window needed
    import matplotlib.pyplot as plt

    names = list(tables)
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}, facecolor=SURFACE, squeeze=False)
    for col, name in enumerate(names):
        t, color = tables[name], MODEL_COLORS.get(name, "#2a78d6")
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)
        top.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label="perfect calibration")
        yerr = np.clip(np.vstack([t["observed"] - t["ci_low"], t["ci_high"] - t["observed"]]), 0, None)  # float noise
        top.errorbar(t["mean_forecast"], t["observed"], yerr=yerr, fmt="none", ecolor=color, elinewidth=1.5, capsize=0)
        top.plot(t["mean_forecast"], t["observed"], color=color, linewidth=2, marker="o", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        top.set_xlim(0, 1); top.set_ylim(0, 1)
        top.set_title(f"{name}   ECE {expected_calibration_error(t):.3f}", color=INK, fontsize=11, loc="left")
        top.set_ylabel("share of games home team won" if col == 0 else "", color=INK_2, fontsize=9)
        top.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
        bottom.bar(t["mean_forecast"], t["n"], width=0.06, color=color, edgecolor=SURFACE, linewidth=2)
        bottom.set_ylabel("games" if col == 0 else "", color=INK_2, fontsize=9)
        bottom.set_xlabel("forecast P(home win)", color=INK_2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Error bars: 95% Wilson intervals. Estimates for research and education, not betting advice.",
             color=INK_2, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
