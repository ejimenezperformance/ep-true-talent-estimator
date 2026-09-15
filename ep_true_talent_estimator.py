"""
EP True Talent Estimator — Empirical Bayes Shrinkage for Batting Average
Emerson Performance — Python / Bayesian statistics portfolio piece

WHAT THIS DOES
    Implements the classic Efron & Morris (1975) empirical Bayes shrinkage
    estimator — the original "Stein's paradox in baseball" study — updated
    with a modern validation design: fit a Beta prior on one season's batting
    averages, shrink each player's raw average toward the league mean by an
    amount inversely proportional to his sample size (AB), then test whether
    the SHRUNK estimate predicts that player's NEXT season better than his
    RAW average does.

    Same method popularized for a general audience in David Robinson's
    "Introduction to Empirical Bayes: Examples from Baseball Statistics."

METHOD (method-of-moments empirical Bayes, Beta-Binomial)
    1. mu     = AB-weighted league mean batting average in season Y
    2. var_total  = AB-weighted variance of observed averages in season Y
    3. var_binom  = average within-player binomial sampling variance,
                    mu*(1-mu)/AB, i.e. the noise we'd see even if every
                    player had the exact same true talent
    4. var_true   = var_total - var_binom   (the REAL spread in talent,
                    net of sampling noise — floored above zero)
    5. Solve the Beta distribution's variance formula for the two
       concentration parameters (alpha, beta), then shrink:
           shrunk_AVG = (alpha + H) / (alpha + beta + AB)
       A player with few AB gets pulled hard toward the league mean; a
       player with a full season of AB barely moves.

VALIDATION (CONFIRMED, not asserted)
    For every year Y in the sample, shrunk_AVG(Y) and raw_AVG(Y) are both
    compared against that same player's ACTUAL average in year Y+1. If
    shrinkage is doing real work, its predictions should be closer to next
    year's outcome than the raw averages are — that comparison is run and
    printed, not just claimed.

STATED LIMITATION
    Data is season-level totals only (no in-season split), so this validates
    against NEXT-SEASON performance rather than a held-out second half within
    the same season, as Efron & Morris's original design did. That is a
    reasonable modern substitute, not an equivalent design — it also mixes in
    genuine year-over-year talent change (aging, injury) that a same-season
    split would not.

DATA
    Sean Lahman's MLB database, distributed by the Chadwick Bureau /
    baseballdatabank (CC BY-SA 3.0 — attribution + share-alike required,
    not public domain). Downloaded automatically on first run.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- EP brand ---------------------------------------------------------
EP_NAVY = "#0B1B33"
EP_GOLD = "#D4A53A"
EP_OFF  = "#F5F3EC"
EP_GRAY = "#5B6472"

DATA_URL = "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/Batting.csv"
LOCAL_CSV = "Batting.csv"


def load_batting() -> pd.DataFrame:
    """Download Lahman's Batting.csv once, then reuse the local copy."""
    if not os.path.exists(LOCAL_CSV):
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_CSV)
    return pd.read_csv(LOCAL_CSV)


def season_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse stints (mid-season trades) into one AB/H total per player-year."""
    g = df.groupby(["playerID", "yearID"], as_index=False).agg(AB=("AB", "sum"), H=("H", "sum"))
    g = g[g.AB > 0]
    g["AVG"] = g.H / g.AB
    return g


def fit_beta_prior(p: np.ndarray, n: np.ndarray):
    """Method-of-moments Beta prior fit, accounting for unequal AB per player."""
    mu = np.average(p, weights=n)
    var_total = np.average((p - mu) ** 2, weights=n)
    var_binom = np.mean(mu * (1 - mu) / n)
    var_true = max(var_total - var_binom, 1e-7)
    nu = max(mu * (1 - mu) / var_true - 1, 1.0)
    alpha, beta = mu * nu, (1 - mu) * nu
    return alpha, beta, mu


def build_validation_set(season: pd.DataFrame, year_lo: int, year_hi: int, min_ab: int) -> pd.DataFrame:
    """For each year Y, fit a prior on Y, shrink, and attach the SAME player's Y+1 average as ground truth."""
    rows = []
    for y in range(year_lo, year_hi):
        cur = season[(season.yearID == y) & (season.AB >= min_ab)]
        nxt = season[(season.yearID == y + 1) & (season.AB >= min_ab)][["playerID", "AVG"]]
        nxt = nxt.rename(columns={"AVG": "AVG_next"})
        merged = cur.merge(nxt, on="playerID", how="inner")
        if len(merged) < 20:
            continue
        alpha, beta, mu = fit_beta_prior(cur.AVG.values, cur.AB.values)
        merged = merged.copy()
        merged["shrunk"] = (alpha + merged.H) / (alpha + beta + merged.AB)
        merged["year"] = y
        merged["league_mean"] = mu
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def main(year_lo=1990, year_hi=2019, min_ab=200):
    df = load_batting()
    season = season_totals(df)
    val = build_validation_set(season, year_lo, year_hi, min_ab)

    mse_raw = float(np.mean((val.AVG - val.AVG_next) ** 2))
    mse_shrunk = float(np.mean((val.shrunk - val.AVG_next) ** 2))
    mae_raw = float(np.mean(np.abs(val.AVG - val.AVG_next)))
    mae_shrunk = float(np.mean(np.abs(val.shrunk - val.AVG_next)))

    print(f"Player-season pairs : {len(val)}")
    print(f"Years                : {year_lo}-{year_hi} (predicting Y+1 from Y)")
    print(f"Min AB               : {min_ab}")
    print(f"MSE   raw={mse_raw:.6f}  shrunk={mse_shrunk:.6f}  "
          f"({(1 - mse_shrunk / mse_raw) * 100:.1f}% reduction)")
    print(f"MAE   raw={mae_raw:.6f}  shrunk={mae_shrunk:.6f}  "
          f"({(1 - mae_shrunk / mae_raw) * 100:.1f}% reduction)")

    make_chart(val, mse_raw, mse_shrunk)
    return val


def make_chart(val: pd.DataFrame, mse_raw: float, mse_shrunk: float):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    fig.patch.set_facecolor(EP_OFF)

    # Panel 1: raw vs. next-season actual
    ax = axes[0]
    ax.set_facecolor(EP_OFF)
    ax.scatter(val.AVG, val.AVG_next, s=8, alpha=0.35, color=EP_GRAY, linewidths=0)
    lims = [0.150, 0.400]
    ax.plot(lims, lims, color=EP_NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"Raw AVG → next season\nMSE = {mse_raw:.5f}", color=EP_NAVY, fontsize=11, fontweight="bold")
    ax.set_xlabel("Raw AVG, season Y", color=EP_GRAY)
    ax.set_ylabel("Actual AVG, season Y+1", color=EP_GRAY)

    # Panel 2: shrunk vs. next-season actual
    ax = axes[1]
    ax.set_facecolor(EP_OFF)
    ax.scatter(val.shrunk, val.AVG_next, s=8, alpha=0.35, color=EP_GOLD, linewidths=0)
    ax.plot(lims, lims, color=EP_NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"Shrunk AVG → next season\nMSE = {mse_shrunk:.5f}", color=EP_NAVY, fontsize=11, fontweight="bold")
    ax.set_xlabel("Empirical-Bayes shrunk AVG, season Y", color=EP_GRAY)
    ax.set_ylabel("")

    fig.suptitle("EP True Talent Estimator — Empirical Bayes Shrinkage vs. Raw Batting Average",
                  color=EP_NAVY, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("ep_true_talent_estimator.png", dpi=150, bbox_inches="tight", facecolor=EP_OFF)
    print("Saved chart to ep_true_talent_estimator.png")


if __name__ == "__main__":
    main()
