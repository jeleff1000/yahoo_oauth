"""
Playoff Odds Transformation

Calculates playoff probabilities, power ratings, and simulation-based predictions.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.

This module adds:
- Playoff probabilities (p_playoffs, p_bye, p_semis, p_final, p_champ)
- Expected outcomes (avg_seed, exp_final_wins, exp_final_pf)
- Power ratings (power_rating)
- Seed distributions (x1_seed through x10_seed)
- Win distributions (x0_win through x14_win)

Performance Improvements:
- Vectorized simulations using NumPy for 10-100x speedup
- Modular design for easier testing and maintenance
- Efficient kernel-based seed prediction using historical data
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys




# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect if we're in modules/ subdirectory or transformations/ directory
if _script_file.parent.name == 'modules':
    # We're in multi_league/transformations/modules/
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    # We're in multi_league/transformations/
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    # Fallback: assume we're somewhere in the tree, navigate up to find multi_league
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts directory
sys.path.insert(0, str(_scripts_dir))  # Allows: from multi_league.core.XXX
sys.path.insert(0, str(_multi_league_dir))  # Allows: from core.XXX

from core.league_context import LeagueContext
from modules.playoff_simulation import (
    build_team_models,
    ensure_params_for_future,
    compute_power_ratings,
    vectorized_regular_season_sim,
    vectorized_bracket_sim,
    normalize_power_rating,
    N_SIMS,
)
from modules.playoff_helpers import (
    wins_points_to_date,
    rank_and_seed,
    enforce_playoff_monotonicity,
    schedules_last_regular_week,
    future_regular_from_schedule,
    history_snapshots,
    empirical_kernel_seed_dist,
    normalize_seed_matrix_to_100,
    p_playoffs_from_seeds,
    p_bye_from_seeds,
)


# =========================================================
# Configuration Constants
# =========================================================
PLAYOFF_SLOTS = 6
BYE_SLOTS = 2
HALF_LIFE_WEEKS = 10
SHRINK_K = 6.0
SIGMA_FLOOR_MIN = 10

# Columns to add/update
TARGET_COLS = [
    "avg_seed", "p_playoffs", "p_bye", "exp_final_wins", "exp_final_pf",
    "p_semis", "p_final", "p_champ",
    "x1_seed", "x2_seed", "x3_seed", "x4_seed", "x5_seed",
    "x6_seed", "x7_seed", "x8_seed", "x9_seed", "x10_seed",
    "power_rating",
    "x0_win", "x1_win", "x2_win", "x3_win", "x4_win", "x5_win", "x6_win", "x7_win",
    "x8_win", "x9_win", "x10_win", "x11_win", "x12_win", "x13_win", "x14_win",
]


# =========================================================
# Core Calculation Functions
# =========================================================
def calc_regular_week_outputs(
    df_season: pd.DataFrame,
    df_sched: pd.DataFrame,
    season: int,
    week: int,
    history_df: pd.DataFrame,
    inflation_rate: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate playoff odds for a regular season week.

    Uses Monte Carlo simulation to estimate:
    1. Expected final record after simulating remaining games
    2. Playoff seed probabilities (blended simulation + historical kernel)
    3. Playoff advancement probabilities (using bracket simulation)
    4. Power ratings

    Args:
        df_season: Matchup data for this season
        df_sched: Schedule data (for future games)
        season: Season year
        week: Current week
        history_df: Historical snapshot data for kernel seed estimation
        inflation_rate: Optional inflation adjustment for power ratings

    Returns:
        odds: DataFrame with playoff odds columns
        seed_dist: DataFrame with seed probability distributions
        win_dist: DataFrame with win probability distributions
    """
    df_to_date = df_season[df_season["week"] <= week].copy()
    reg_to_date = df_to_date[df_to_date["is_playoffs"] == 0].copy()
    played_raw = reg_to_date.copy()

    # Get future regular season games
    future_canon = future_regular_from_schedule(df_sched, season, week)
    simulate_future_reg = not future_canon.empty

    # Build team models
    sigma_floor_dynamic = SIGMA_FLOOR_MIN if week >= 3 else max(SIGMA_FLOOR_MIN, 14)
    mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor = build_team_models(
        df_to_date, season, week, HALF_LIFE_WEEKS, SHRINK_K, sigma_floor_dynamic,
        boundary_penalty=0.05, prior_w_cap=2.0
    )
    ensure_params_for_future(mu_hat, sigma_hat, samples_by_team, future_canon, league_mu, sigma_floor)

    # Calculate power ratings
    power_s = compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4)
    if inflation_rate is not None and pd.notna(inflation_rate) and inflation_rate != 0:
        power_s = normalize_power_rating(power_s, inflation_rate)

    # Simulate remaining regular season + playoffs
    W_all, PF_all, mgr2idx, idx2mgr = vectorized_regular_season_sim(
        reg_to_date, future_canon, mu_hat, sigma_hat, season, week
    )

    # Simulate playoff bracket
    bracket_results = vectorized_bracket_sim(
        W_all, PF_all, mgr2idx, idx2mgr, mu_hat, sigma_hat, season, week
    )

    # Extract results
    seeds_idx = bracket_results["seeds_idx"]
    seed1 = bracket_results["seed1"]
    seed2 = bracket_results["seed2"]
    qf_winners = bracket_results["qf_winners"]
    sf_winners = bracket_results["sf_winners"]
    champions = bracket_results["champions"]

    n = len(idx2mgr)

    # Seed distribution from simulation
    seed_mat = np.zeros((n, n), dtype=float)
    for pos in range(n):
        teams_at_pos = seeds_idx[:, pos]
        seed_mat[:, pos] = np.bincount(teams_at_pos, minlength=n) / N_SIMS * 100.0
    seed_dist_sim = pd.DataFrame(seed_mat, index=idx2mgr, columns=list(range(1, n + 1)))

    # Blend with historical kernel prediction if simulating future
    if simulate_future_reg:
        hist_seed_dist = empirical_kernel_seed_dist(
            played_raw, week, history_df, n
        ).reindex(index=seed_dist_sim.index, columns=seed_dist_sim.columns, fill_value=0.0) * 100.0

        weeks_played = int(played_raw["week"].nunique())
        sim_w = 0.25 + 0.75 * min(1.0, max(0.0, (weeks_played - 1) / 4.0))
        blended_seed = sim_w * seed_dist_sim + (1 - sim_w) * hist_seed_dist
        blended_seed_norm = normalize_seed_matrix_to_100(blended_seed)
    else:
        blended_seed_norm = normalize_seed_matrix_to_100(seed_dist_sim)

    # Calculate probabilities
    def pct_counts(idx_arr, nteams):
        arr = np.asarray(idx_arr).reshape(-1).astype(np.int64, copy=False)
        if arr.size == 0:
            return np.zeros(nteams, dtype=float)
        c = np.bincount(arr, minlength=nteams).astype(float)
        return 100.0 * c / arr.size

    p_semis_sim = pct_counts(np.concatenate([seed1, seed2, qf_winners[:, 0], qf_winners[:, 1]]), n)
    p_final_sim = pct_counts(np.concatenate([sf_winners[:, 0], sf_winners[:, 1]]), n)
    p_champ_sim = pct_counts(champions, n)

    # Conditional probabilities
    # P(SF | bye), P(SF | no bye, won QF), P(Final | SF), etc.
    bye_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    bye_mat[np.arange(N_SIMS), seed1] = 1
    bye_mat[np.arange(N_SIMS), seed2] = 1
    no_bye_mat = 1 - bye_mat

    qfwin_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    qfwin_mat[np.arange(N_SIMS), qf_winners[:, 0]] = 1
    qfwin_mat[np.arange(N_SIMS), qf_winners[:, 1]] = 1

    sfwin_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    sfwin_mat[np.arange(N_SIMS), sf_winners[:, 0]] = 1
    sfwin_mat[np.arange(N_SIMS), sf_winners[:, 1]] = 1

    champ_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    champ_mat[np.arange(N_SIMS), champions] = 1

    # P(QF win | no bye)
    no_bye_counts = no_bye_mat.sum(axis=0).astype(float)
    qfwin_and_no_bye = (qfwin_mat & no_bye_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        qf_win_given_no_bye = np.where(no_bye_counts > 0,
                                       100.0 * qfwin_and_no_bye / no_bye_counts,
                                       0.0)

    # P(SF win | bye)
    bye_counts = bye_mat.sum(axis=0).astype(float)
    sfwin_and_bye = (sfwin_mat & bye_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_win_given_bye = np.where(bye_counts > 0,
                                    100.0 * sfwin_and_bye / bye_counts,
                                    0.0)

    # P(SF win | won QF)
    qfwin_counts = qfwin_mat.sum(axis=0).astype(float)
    sfwin_and_qf = (sfwin_mat & qfwin_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_win_given_nonbye = np.where(qfwin_counts > 0,
                                       100.0 * sfwin_and_qf / qfwin_counts,
                                       0.0)

    # P(Win Final | made Final)
    final_counts = sfwin_mat.sum(axis=0).astype(float)
    champ_and_final = (champ_mat & sfwin_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_final_given_final = np.where(final_counts > 0,
                                         100.0 * champ_and_final / final_counts,
                                         0.0)

    # Build odds DataFrame
    s_index = pd.Index(idx2mgr, name="manager")
    exp_final_wins = W_all.mean(axis=0)
    exp_final_pf = PF_all.mean(axis=0)
    avg_seed = seeds_idx.mean(axis=0) + 1  # Convert from 0-indexed

    odds = pd.DataFrame({
        "Exp_Final_Wins": exp_final_wins,
        "Exp_Final_PF": exp_final_pf,
        "Avg_Seed": avg_seed,
    }, index=s_index)

    odds["P_Playoffs"] = p_playoffs_from_seeds(blended_seed_norm, PLAYOFF_SLOTS)
    odds["P_Bye"] = p_bye_from_seeds(blended_seed_norm, BYE_SLOTS)

    # P(Semis) = P(Bye) + P(No Bye) * P(QF Win | No Bye)
    qf_cnd = pd.Series(qf_win_given_no_bye, index=s_index)
    odds["P_Semis"] = odds["P_Bye"] + (100.0 - odds["P_Bye"]) * (qf_cnd / 100.0)
    odds["P_Semis"] = np.minimum(100.0, np.maximum(odds["P_Semis"], odds["P_Bye"]))

    # P(Final) = P(Bye) * P(SF Win | Bye) + P(Semis, No Bye) * P(SF Win | Non-Bye)
    sf_bye = pd.Series(sf_win_given_bye, index=s_index)
    sf_nonbye = pd.Series(sf_win_given_nonbye, index=s_index)
    semis_nonbye = np.maximum(0.0, odds["P_Semis"] - odds["P_Bye"])
    odds["P_Final"] = (odds["P_Bye"] * sf_bye + semis_nonbye * sf_nonbye) / 100.0

    # P(Champ) = P(Final) * P(Win Final | Final)
    win_final = pd.Series(win_final_given_final, index=s_index)
    odds["P_Champ"] = odds["P_Final"] * (win_final / 100.0)

    # Enforce monotonicity
    odds["P_Final"] = np.minimum(odds["P_Semis"], odds["P_Final"])
    odds["P_Champ"] = np.minimum(odds["P_Final"], odds["P_Champ"])
    odds[["P_Semis", "P_Final", "P_Champ"]] = odds[["P_Semis", "P_Final", "P_Champ"]].clip(lower=0.0, upper=100.0)

    # Add power ratings
    if not power_s.empty:
        odds["Power_Rating"] = power_s.reindex(odds.index)

    # Win distribution
    W_int = np.rint(W_all).astype(int).clip(0, 14)
    win_prob_mat = np.zeros((n, 15), dtype=float)
    for k in range(15):
        mask = (W_int == k).astype(np.int32)
        counts_k = mask.sum(axis=0).astype(float)
        win_prob_mat[:, k] = 100.0 * counts_k / float(N_SIMS)

    win_cols = [f"x{k}_win" for k in range(15)]
    win_df = pd.DataFrame(win_prob_mat, index=idx2mgr, columns=win_cols)

    return odds, blended_seed_norm, win_df


def calc_playoff_week_outputs(
    df_season: pd.DataFrame,
    df_sched: pd.DataFrame,
    season: int,
    week: int,
    inflation_rate: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate playoff odds during playoff weeks.

    During playoffs, some games may have already been played. This function
    respects actual results and only simulates remaining games.

    Args:
        df_season: Matchup data for this season
        df_sched: Schedule data
        season: Season year
        week: Current week
        inflation_rate: Optional inflation adjustment

    Returns:
        odds: DataFrame with playoff odds
        seed_dist: DataFrame with seed distributions
        seeds: DataFrame with seeding table
    """
    from modules.playoff_simulation import simulate_game, get_rng

    df_to_date = df_season[df_season["week"] <= week].copy()

    # Get regular season standings
    reg = df_to_date[df_to_date["is_playoffs"] == 0].copy()
    try:
        wins_to_date, pts_to_date = wins_points_to_date(reg)
    except Exception as e:
        # Fallback: compute wins and points manually if helper fails.  A
        # common failure occurs when the helper assumes each match
        # consists of exactly two teams.  In practice, mis-matched
        # grouping sizes or incomplete data can cause a ValueError or
        # IndexError.  Rather than abort, derive simple win totals and
        # point sums from the matchup data.
        print(f"[WARN] wins_points_to_date failed: {e}. Falling back to manual computation.")
        if not reg.empty:
            # Wins: count number of games where manager_score > opponent_score
            win_series = reg.groupby("manager").apply(
                lambda df: (df["manager_score"] > df["opponent_score"]).sum()
            ).astype(float)
            # Points: sum of manager scores
            pt_series = reg.groupby("manager")["manager_score"].sum().astype(float)
            # Align indices to ensure same order
            wins_to_date = win_series.reindex(pt_series.index, fill_value=0.0)
            pts_to_date = pt_series
        else:
            wins_to_date = pd.Series([], dtype=float)
            pts_to_date = pd.Series([], dtype=float)
    seeds = rank_and_seed(wins_to_date, pts_to_date, PLAYOFF_SLOTS, BYE_SLOTS, played_raw=reg)
    top6 = seeds.loc[seeds["made_playoffs"], "manager"].tolist()

    if len(top6) != 6:
        # Not enough teams for playoffs
        return pd.DataFrame(), pd.DataFrame(), seeds

    # Build models
    last_reg_week = schedules_last_regular_week(df_sched, season)
    if last_reg_week is None:
        last_reg_week = reg["week"].max() if not reg.empty else week

    mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor = build_team_models(
        df_to_date, season, last_reg_week, HALF_LIFE_WEEKS, SHRINK_K, SIGMA_FLOOR_MIN,
        boundary_penalty=0.05, prior_w_cap=2.0
    )

    power_s = compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4)
    if inflation_rate is not None and pd.notna(inflation_rate) and inflation_rate != 0:
        power_s = normalize_power_rating(power_s, inflation_rate)

    # Determine bracket structure
    byes = top6[:BYE_SLOTS]
    qtrs = [(top6[3], top6[4]), (top6[2], top6[5])]  # (4 vs 5, 3 vs 6)

    # Check for actual results
    ROUND_COLS = {"qf": "quarterfinal", "sf": "semifinal", "fn": "championship"}

    def actual_winner_by_round(round_col, teamA, teamB):
        """Check if round has been played and return winner."""
        if (round_col not in df_to_date.columns) or (teamA is None) or (teamB is None):
            return None
        g = df_to_date[(df_to_date.get(round_col, 0) == 1) &
                       (df_to_date["manager"].isin([teamA, teamB])) &
                       (df_to_date["opponent"].isin([teamA, teamB]))]
        if "is_consolation" in df_to_date.columns:
            g = g[g["is_consolation"] == 0]
        if g.empty:
            return None

        weeks_a = set(g.loc[g["manager"] == teamA, "week"].dropna().astype(int).tolist())
        weeks_b = set(g.loc[g["manager"] == teamB, "week"].dropna().astype(int).tolist())
        shared_weeks = sorted([w for w in weeks_a.intersection(weeks_b) if w <= int(week)])
        if not shared_weeks:
            return None

        w_star = shared_weeks[-1]
        gw = g[g["week"] == w_star].copy()
        pts = gw.groupby("manager", as_index=True)["team_points"].mean()
        if set(pts.index) != {teamA, teamB}:
            return None
        if pd.isna(pts.get(teamA)) or pd.isna(pts.get(teamB)):
            return None
        if pts[teamA] > pts[teamB]:
            return teamA
        if pts[teamB] > pts[teamA]:
            return teamB
        return min(teamA, teamB)  # Deterministic tiebreaker

    qf_col = ROUND_COLS.get("qf", "quarterfinal")
    sf_col = ROUND_COLS.get("sf", "semifinal")
    fn_col = ROUND_COLS.get("fn", "championship")

    q1_actual = actual_winner_by_round(qf_col, qtrs[0][0], qtrs[0][1])
    q2_actual = actual_winner_by_round(qf_col, qtrs[1][0], qtrs[1][1])

    # Determine semifinal matchups
    def semi_opponents(q1_w, q2_w):
        winners = [w for w in [q1_w, q2_w] if w is not None]
        if len(winners) < 2:
            return None, None
        # No reseeding by default: 1 plays QF1 winner, 2 plays QF2 winner
        return (top6[0], q1_w), (top6[1], q2_w)

    s1_pair, s2_pair = semi_opponents(q1_actual, q2_actual)
    s1_actual = actual_winner_by_round(sf_col, *(s1_pair or (None, None))) if s1_pair else None
    s2_actual = actual_winner_by_round(sf_col, *(s2_pair or (None, None))) if s2_pair else None

    # Determine finalists
    finalists = None
    if (s1_actual is not None) and (s2_actual is not None):
        finalists = (s1_actual, s2_actual)

    champ_locked = None
    if finalists is not None:
        champ_locked = actual_winner_by_round(fn_col, finalists[0], finalists[1])

    # Run simulations
    rng = get_rng(season, week)
    sims, playoff_r2, playoff_r3, champions_list = [], [], [], []

    for _ in range(N_SIMS):
        q1_w = q1_actual if q1_actual else simulate_game(qtrs[0][0], qtrs[0][1], rng, mu_hat, sigma_hat, samples_by_team)
        q2_w = q2_actual if q2_actual else simulate_game(qtrs[1][0], qtrs[1][1], rng, mu_hat, sigma_hat, samples_by_team)

        semi1, semi2 = semi_opponents(q1_w, q2_w)
        if semi1 is None:
            continue

        s1_w = s1_actual if s1_actual else simulate_game(semi1[0], semi1[1], rng, mu_hat, sigma_hat, samples_by_team)
        s2_w = s2_actual if s2_actual else simulate_game(semi2[0], semi2[1], rng, mu_hat, sigma_hat, samples_by_team)

        if finalists is not None:
            s1_w, s2_w = finalists[0], finalists[1]

        champ = champ_locked if champ_locked else simulate_game(s1_w, s2_w, rng, mu_hat, sigma_hat, samples_by_team)

        playoff_r2.extend([semi1[0], semi2[0], q1_w, q2_w])
        playoff_r3.extend([s1_w, s2_w])
        champions_list.append(champ)
        sims.append(seeds)

    if not sims:
        return pd.DataFrame(), pd.DataFrame(), seeds

    # Aggregate results
    tall = pd.concat(sims, ignore_index=True)
    odds = (tall.groupby("manager")
            .agg(Exp_Final_Wins=("W", "mean"),
                 Exp_Final_PF=("PF", "mean")))
    odds["Avg_Seed"] = tall.groupby("manager")["seed"].mean()

    idx_mgrs = odds.index.tolist()
    odds["P_Semis"] = [playoff_r2.count(m) / N_SIMS * 100 for m in idx_mgrs]
    odds["P_Final"] = [playoff_r3.count(m) / N_SIMS * 100 for m in idx_mgrs]
    odds["P_Champ"] = [champions_list.count(m) / N_SIMS * 100 for m in idx_mgrs]

    # Lock probabilities for finished rounds
    if finalists is not None:
        finals_set = set(finalists)
        non_finalists = [m for m in idx_mgrs if m not in finals_set]
        if non_finalists:
            odds.loc[non_finalists, ["P_Final", "P_Champ"]] = 0.0
        odds.loc[list(finals_set), "P_Final"] = 100.0

    if champ_locked:
        odds["P_Champ"] = 0.0
        if champ_locked in odds.index:
            odds.at[champ_locked, "P_Champ"] = 100.0

    # Seed distribution (fixed to actual seeds)
    team_count = seeds.shape[0]
    seed_dist = (tall.pivot_table(index="manager", columns="seed", values="W",
                                  aggfunc="size", fill_value=0)
                 .div(len(sims)) * 100.0)
    all_cols = list(range(1, team_count + 1))
    seed_dist = seed_dist.reindex(columns=all_cols, fill_value=0.0)
    seed_dist_norm = normalize_seed_matrix_to_100(seed_dist)

    # Playoff/bye probabilities (fixed)
    top6_set = set(top6)
    odds["P_Playoffs"] = [100.0 if m in top6_set else 0.0 for m in idx_mgrs]
    odds["P_Bye"] = [100.0 if (m in top6_set and bool(seeds.loc[seeds['manager'] == m, 'bye'].iloc[0])) else 0.0
                     for m in idx_mgrs]

    # Power ratings
    if "Power_Rating" not in odds.columns:
        odds["Power_Rating"] = power_s.reindex(odds.index)

    return odds, seed_dist_norm, seeds


# =========================================================
# Main Transformation Function
# =========================================================
def calculate_playoff_odds(
    matchup_df: pd.DataFrame,
    schedule_df: Optional[pd.DataFrame] = None,
    current_week: Optional[int] = None,
    current_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate playoff odds for all managers across all weeks.

    This is the main entry point for the playoff odds transformation.

    Args:
        matchup_df: DataFrame with matchup data
        schedule_df: Optional schedule data for future games
        current_week: Current week number (for weekly updates)
        current_year: Current year (for weekly updates)

    Returns:
        DataFrame with playoff odds columns added
    """
    print("Calculating playoff odds...")

    df = matchup_df.copy()
    if schedule_df is None:
        schedule_df = pd.DataFrame()

    # Initialize all target columns
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Clean and coerce types
    df.replace("", np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    for col in ["week", "year", "is_playoffs", "is_consolation", "inflation_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Enforce playoff monotonicity
    df = enforce_playoff_monotonicity(df)
    if not schedule_df.empty:
        schedule_df = enforce_playoff_monotonicity(schedule_df)

    # Generate historical snapshots
    hist_df = history_snapshots(df, PLAYOFF_SLOTS)
    print(f"  Generated {len(hist_df)} historical snapshots")

    # Process each season
    seasons = sorted(df["year"].dropna().unique().astype(int))
    for season in seasons:
        print(f"\nProcessing season {season}...")
        df_season = df[df["year"] == season].copy()
        if df_season.empty:
            continue

        # Get inflation rate for this season
        if "inflation_rate" in df_season.columns:
            season_inflation = df_season["inflation_rate"].dropna()
            inflation_rate = season_inflation.iloc[0] if not season_inflation.empty else None
        else:
            inflation_rate = None

        # Split into regular and playoff weeks
        reg_weeks = sorted(df_season[df_season["is_playoffs"] == 0]["week"].dropna().unique().astype(int))
        po_weeks = sorted(df_season[(df_season["is_playoffs"] == 1) & (df_season["is_consolation"] == 0)][
                              "week"].dropna().unique().astype(int))

        # Process regular season weeks
        for w in reg_weeks:
            print(f"  Processing regular season week {w}...")
            mask_week = (df["year"] == season) & (df["week"] == w) & (df["is_consolation"] == 0)
            if not mask_week.any():
                continue

            odds_df, seed_df, win_df = calc_regular_week_outputs(
                df_season, schedule_df, season, w, hist_df, inflation_rate
            )

            # Write results to dataframe
            for idx in df[mask_week].index:
                m = df.at[idx, "manager"]
                if m not in odds_df.index:
                    continue

                def set_val(col, val):
                    if col in df.columns and pd.notna(val):
                        df.at[idx, col] = round(float(val), 2) if isinstance(val, (float, np.floating, int)) else val

                set_val("avg_seed", odds_df.at[m, "Avg_Seed"])
                set_val("exp_final_wins", odds_df.at[m, "Exp_Final_Wins"])
                set_val("exp_final_pf", odds_df.at[m, "Exp_Final_PF"])
                set_val("p_semis", odds_df.at[m, "P_Semis"])
                set_val("p_final", odds_df.at[m, "P_Final"])
                set_val("p_champ", odds_df.at[m, "P_Champ"])
                set_val("p_playoffs", odds_df.at[m, "P_Playoffs"])
                set_val("p_bye", odds_df.at[m, "P_Bye"])

                if "Power_Rating" in odds_df.columns:
                    set_val("power_rating", odds_df.at[m, "Power_Rating"])

                # Seed probabilities
                if not seed_df.empty and m in seed_df.index:
                    for k in range(1, 11):
                        if k in seed_df.columns:
                            set_val(f"x{k}_seed", seed_df.loc[m, k])

                # Win probabilities
                if (win_df is not None) and (m in win_df.index):
                    for k in range(0, 15):
                        col = f"x{k}_win"
                        if col in win_df.columns:
                            set_val(col, win_df.at[m, col])

        # Process playoff weeks
        for w in po_weeks:
            print(f"  Processing playoff week {w}...")
            mask_week = (
                (df["year"] == season)
                & (df["week"] == w)
                & (df["is_playoffs"] == 1)
                & (df["is_consolation"] == 0)
            )
            if not mask_week.any():
                continue

            odds_df, seed_df, seeds = calc_playoff_week_outputs(
                df_season, schedule_df, season, w, inflation_rate
            )

            for idx in df[mask_week].index:
                m = df.at[idx, "manager"]
                if m not in odds_df.index:
                    continue

                def set_val(col, val):
                    if col in df.columns and pd.notna(val):
                        df.at[idx, col] = round(float(val), 2) if isinstance(val, (float, np.floating, int)) else val

                set_val("avg_seed", odds_df.at[m, "Avg_Seed"])
                set_val("exp_final_wins", odds_df.at[m, "Exp_Final_Wins"])
                set_val("exp_final_pf", odds_df.at[m, "Exp_Final_PF"])
                set_val("p_semis", odds_df.at[m, "P_Semis"])
                set_val("p_final", odds_df.at[m, "P_Final"])
                set_val("p_champ", odds_df.at[m, "P_Champ"])
                set_val("p_playoffs", odds_df.at[m, "P_Playoffs"])
                set_val("p_bye", odds_df.at[m, "P_Bye"])

                if "Power_Rating" in odds_df.columns:
                    set_val("power_rating", odds_df.at[m, "Power_Rating"])

                if not seed_df.empty and m in seed_df.index:
                    for k in range(1, 11):
                        if k in seed_df.columns:
                            set_val(f"x{k}_seed", seed_df.loc[m, k])

    print(f"\nPlayoff odds calculation complete!")
    print(f"Updated {len(df)} records with {len(TARGET_COLS)} playoff odds columns")

    return df


# =========================================================
# CLI Interface
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate playoff odds using Monte Carlo simulation"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Path to league_context.json"
    )
    parser.add_argument(
        "--current-week",
        type=int,
        help="Current week number (for weekly updates)"
    )
    parser.add_argument(
        "--current-year",
        type=int,
        help="Current year (for weekly updates)"
    )

    args = parser.parse_args()

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"Loaded league context: {ctx.league_name}")

    # Load matchup data (use canonical file path)
    matchup_path = ctx.canonical_matchup_file
    print(f"Looking for matchup data at: {matchup_path}")
    print(f"File exists: {matchup_path.exists()}")
    if not matchup_path.exists():
        raise FileNotFoundError(f"Matchup data not found: {matchup_path}")

    matchup_df = pd.read_parquet(matchup_path)
    print(f"Loaded {len(matchup_df)} matchup records")

    # Load schedule data if available
    schedule_df = None
    schedule_path = ctx.data_directory / "schedule.parquet"
    if schedule_path.exists():
        schedule_df = pd.read_parquet(schedule_path)
        print(f"Loaded {len(schedule_df)} schedule records")

    # Calculate playoff odds
    enriched_df = calculate_playoff_odds(
        matchup_df,
        schedule_df=schedule_df,
        current_week=args.current_week,
        current_year=args.current_year
    )

    # Save results (back to canonical matchup file)
    output_path = ctx.canonical_matchup_file
    enriched_df.to_parquet(output_path, index=False)
    print(f"\nSaved enriched matchup data to: {output_path}")

    # Also save CSV
    csv_path = output_path.with_suffix(".csv")
    enriched_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
