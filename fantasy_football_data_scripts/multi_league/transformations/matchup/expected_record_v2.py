#!/usr/bin/env python3
"""
Expected Record Transformation

Calculates schedule-independent expected records by simulating random schedules.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.
"""

import sys
import argparse
import random
import numpy as np
from pathlib import Path
from functools import wraps
from typing import Optional
import pandas as pd


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

from core.data_normalization import normalize_numeric_columns, ensure_league_id
from core.league_context import LeagueContext
from modules.schedule_simulation import (
    run_simulations,
    current_regular_week,
    calculate_summary_stats,
    calculate_opponent_rank_percentile,
    lock_postseason_to_final_week,
)
from modules.bye_week_filler import fill_bye_weeks, validate_bye_week_coverage


def ensure_normalized(func):
    """Decorator to ensure data normalization"""
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Normalize input
        df = normalize_numeric_columns(df)
        
        # Run transformation
        result = func(df, *args, **kwargs)
        
        # Normalize output
        result = normalize_numeric_columns(result)
        
        # Ensure league_id present
        if 'league_id' in df.columns:
            league_id = df['league_id'].iloc[0] if len(df) > 0 else None
            if league_id is not None and pd.notna(league_id):
                result = ensure_league_id(result, league_id)
        
        return result
    
    return wrapper


# =========================================================
# Configuration Constants
# =========================================================
N_SIMS = 100000
RNG_SEED = None  # None for random, int for reproducibility

# Columns to add/update
SHUFFLE_SEED_COLS = [f"shuffle_{i}_seed" for i in range(1, 11)]
SHUFFLE_WIN_COLS = [f"shuffle_{w}_win" for w in range(0, 15)]
SHUFFLE_SUMMARY_COLS = [
    "shuffle_avg_wins",
    "shuffle_avg_seed",
    "shuffle_avg_playoffs",
    "shuffle_avg_bye",
    "wins_vs_shuffle_wins",
    "seed_vs_shuffle_seed",
]

OPP_SHUFFLE_SEED_COLS = [f"opp_shuffle_{i}_seed" for i in range(1, 11)]
OPP_SHUFFLE_WIN_COLS = [f"opp_shuffle_{w}_win" for w in range(0, 15)]
OPP_SHUFFLE_SUMMARY_COLS = [
    "opp_shuffle_avg_wins",
    "opp_shuffle_avg_seed",
    "opp_shuffle_avg_playoffs",
    "opp_shuffle_avg_bye",
    "wins_vs_opp_shuffle_wins",
    "seed_vs_opp_shuffle_seed",
]

OPP_RANK_COLS = ["opp_pts_week_rank", "opp_pts_week_pct"]


# =========================================================
# Main Transformation Function
# =========================================================

@ensure_normalized
def calculate_expected_records(
    matchup_df: pd.DataFrame,
    current_week: Optional[int] = None,
    current_year: Optional[int] = None,
    n_sims: int = N_SIMS,
    rng_seed: Optional[int] = RNG_SEED
) -> pd.DataFrame:
    """
    Calculate expected records using schedule simulations.

    Runs two types of simulations:
    1. Performance-based: Shuffle actual scores across random schedules
    2. Opponent difficulty: Shuffle opponent difficulty to measure schedule luck

    Args:
        matchup_df: DataFrame with matchup data
        current_week: Current week (optional, for partial season updates)
        current_year: Current year (optional)
        n_sims: Number of Monte Carlo simulations (default: 100,000)
        rng_seed: Random seed for reproducibility (None = random)

    Returns:
        DataFrame with expected record columns added
    """
    print(f"Calculating expected records ({n_sims:,} simulations)...")

    df = matchup_df.copy()
    rng = random.Random(rng_seed)

    # Initialize all target columns
    all_cols = (SHUFFLE_SEED_COLS + SHUFFLE_WIN_COLS + SHUFFLE_SUMMARY_COLS +
                OPP_SHUFFLE_SEED_COLS + OPP_SHUFFLE_WIN_COLS + OPP_SHUFFLE_SUMMARY_COLS +
                OPP_RANK_COLS)

    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Regular season mask
    regular_mask = (df['is_playoffs'] == 0) & (df['is_consolation'] == 0)

    # Clear all shuffle fields for non-regular rows
    df.loc[~regular_mask, all_cols] = np.nan

    # Process each season
    seasons = sorted(df[regular_mask]['year'].dropna().unique().astype(int))

    for year in seasons:
        print(f"\nProcessing season {year}...")
        df_season = df[df['year'] == year].copy()
        df_reg_season = df_season[df_season['is_playoffs'] == 0]

        if df_reg_season.empty:
            print(f"  Skipping - no regular season data")
            continue

        managers = tuple(sorted(df_reg_season['manager'].unique()))
        n_managers = len(managers)

        if n_managers % 2 != 0:
            print(f"  Skipping - odd team count ({n_managers})")
            continue

        last_wk = current_regular_week(df_reg_season)
        if last_wk == 0:
            print(f"  Skipping - no complete weeks")
            continue

        print(f"  {n_managers} managers, {last_wk} complete weeks")

        # Extract points data
        df_points = df_reg_season[['manager', 'week', 'team_points', 'opponent_points']].copy()

        # Process each week
        for wk in range(1, last_wk + 1):
            print(f"    Processing week {wk}...", end='', flush=True)

            n_weeks_cap = 2 * (n_managers - 1)
            n_weeks = min(wk, n_weeks_cap)

            df_to_week = df_points[df_points['week'] <= n_weeks]

            # ===========================
            # PERFORMANCE-BASED SIMULATION (team_points)
            # ===========================
            points_perf = {(r['manager'], int(r['week'])): float(r['team_points'])
                          for _, r in df_to_week.iterrows()}

            win_hists_perf, seed_hists_perf = run_simulations(
                points_perf, managers, n_weeks, n_sims, rng, mode="performance"
            )

            # Calculate denominators
            win_den = {m: max(1, sum(win_hists_perf[m])) for m in managers}
            seed_den = {m: max(1, sum(seed_hists_perf[m])) for m in managers}

            # Calculate expected wins/seeds/playoffs/byes DIRECTLY from counts (no rounding)
            expected_wins_perf = {}
            expected_seed_perf = {}
            expected_playoffs_perf = {}
            expected_bye_perf = {}
            for mgr in managers:
                # Expected wins = sum(wins * probability) = sum(wins * count) / total_count
                expected_wins_perf[mgr] = sum(wv * win_hists_perf[mgr][wv] for wv in range(len(win_hists_perf[mgr]))) / win_den[mgr]
                # Expected seed = sum(seed * probability) = sum(seed * count) / total_count
                expected_seed_perf[mgr] = sum((s + 1) * seed_hists_perf[mgr][s] for s in range(len(seed_hists_perf[mgr]))) / seed_den[mgr]
                # Playoff odds = probability of seeds 1-6 = sum(counts for seeds 1-6) / total
                expected_playoffs_perf[mgr] = 100.0 * sum(seed_hists_perf[mgr][s] for s in range(min(6, len(seed_hists_perf[mgr])))) / seed_den[mgr]
                # Bye odds = probability of seeds 1-2 = sum(counts for seeds 1-2) / total
                expected_bye_perf[mgr] = 100.0 * sum(seed_hists_perf[mgr][s] for s in range(min(2, len(seed_hists_perf[mgr])))) / seed_den[mgr]

            # Write performance-based results
            rows = df.index[(df['year'] == year) & (df['week'] == wk) & regular_mask]
            for idx in rows:
                mgr = df.at[idx, 'manager']
                if mgr not in managers:
                    continue

                # Win probabilities (still save for display, but don't use for expected value calculation)
                for wv in range(0, min(n_weeks, 14) + 1):
                    df.at[idx, f"shuffle_{wv}_win"] = round(
                        100.0 * win_hists_perf[mgr][wv] / win_den[mgr], 2
                    )

                # Seed probabilities
                for s in range(1, min(n_managers, 10) + 1):
                    df.at[idx, f"shuffle_{s}_seed"] = round(
                        100.0 * seed_hists_perf[mgr][s - 1] / seed_den[mgr], 2
                    )

                # Store expected values calculated directly from counts (no rounding errors)
                df.at[idx, "shuffle_avg_wins"] = round(expected_wins_perf[mgr], 2)
                df.at[idx, "shuffle_avg_seed"] = round(expected_seed_perf[mgr], 2)
                df.at[idx, "shuffle_avg_playoffs"] = round(expected_playoffs_perf[mgr], 2)
                df.at[idx, "shuffle_avg_bye"] = round(expected_bye_perf[mgr], 2)

            # ===========================
            # OPPONENT DIFFICULTY SIMULATION (opponent_points)
            # ===========================
            points_opp = {(r['manager'], int(r['week'])): float(r['opponent_points'])
                         for _, r in df_to_week.iterrows()
                         if pd.notna(r['opponent_points'])}

            if points_opp:  # Only if opponent_points available
                win_hists_opp, seed_hists_opp = run_simulations(
                    points_opp, managers, n_weeks, n_sims, rng, mode="opponent"
                )

                win_den_opp = {m: max(1, sum(win_hists_opp[m])) for m in managers}
                seed_den_opp = {m: max(1, sum(seed_hists_opp[m])) for m in managers}

                # Calculate expected wins/seeds/playoffs/byes DIRECTLY from counts (no rounding)
                expected_wins_opp = {}
                expected_seed_opp = {}
                expected_playoffs_opp = {}
                expected_bye_opp = {}
                for mgr in managers:
                    # Expected wins = sum(wins * probability) = sum(wins * count) / total_count
                    expected_wins_opp[mgr] = sum(wv * win_hists_opp[mgr][wv] for wv in range(len(win_hists_opp[mgr]))) / win_den_opp[mgr]
                    # Expected seed = sum(seed * probability) = sum(seed * count) / total_count
                    expected_seed_opp[mgr] = sum((s + 1) * seed_hists_opp[mgr][s] for s in range(len(seed_hists_opp[mgr]))) / seed_den_opp[mgr]
                    # Playoff odds = probability of seeds 1-6 = sum(counts for seeds 1-6) / total
                    expected_playoffs_opp[mgr] = 100.0 * sum(seed_hists_opp[mgr][s] for s in range(min(6, len(seed_hists_opp[mgr])))) / seed_den_opp[mgr]
                    # Bye odds = probability of seeds 1-2 = sum(counts for seeds 1-2) / total
                    expected_bye_opp[mgr] = 100.0 * sum(seed_hists_opp[mgr][s] for s in range(min(2, len(seed_hists_opp[mgr])))) / seed_den_opp[mgr]

                # Write opponent difficulty results
                for idx in rows:
                    mgr = df.at[idx, 'manager']
                    if mgr not in managers:
                        continue

                    # Win probabilities (ease-based, still save for display)
                    for wv in range(0, min(n_weeks, 14) + 1):
                        df.at[idx, f"opp_shuffle_{wv}_win"] = round(
                            100.0 * win_hists_opp[mgr][wv] / win_den_opp[mgr], 2
                        )

                    # Seed probabilities
                    for s in range(1, min(n_managers, 10) + 1):
                        df.at[idx, f"opp_shuffle_{s}_seed"] = round(
                            100.0 * seed_hists_opp[mgr][s - 1] / seed_den_opp[mgr], 2
                        )

                    # Store expected values calculated directly from counts (no rounding errors)
                    df.at[idx, "opp_shuffle_avg_wins"] = round(expected_wins_opp[mgr], 2)
                    df.at[idx, "opp_shuffle_avg_seed"] = round(expected_seed_opp[mgr], 2)
                    df.at[idx, "opp_shuffle_avg_playoffs"] = round(expected_playoffs_opp[mgr], 2)
                    df.at[idx, "opp_shuffle_avg_bye"] = round(expected_bye_opp[mgr], 2)

            print(" done")

    # ===========================
    # MARK FINAL REGULAR SEASON WEEK
    # ===========================
    # Mark which week is the last FULL regular season week for each year
    # This is the week where ALL managers played regular season (for end-of-season stats)
    print("\nMarking final regular season week for each year...")

    # Initialize column
    df['is_final_regular_week'] = 0

    for year in seasons:
        df_reg_year = df[(df['year'] == year) & (df['is_playoffs'] == 0) & (df['is_consolation'] == 0)]
        if df_reg_year.empty:
            continue

        # Find number of managers in this year
        all_managers = sorted(df_reg_year['manager'].unique())
        n_managers = len(all_managers)

        # Find last week where ALL managers played regular season
        last_full_week = None
        for week in sorted(df_reg_year['week'].unique(), reverse=True):
            week_data = df_reg_year[df_reg_year['week'] == week]
            if len(week_data['manager'].unique()) == n_managers:
                last_full_week = week
                break

        if last_full_week:
            # Mark this week for all managers in this year
            mask = (df['year'] == year) & (df['week'] == last_full_week) & (df['is_playoffs'] == 0)
            df.loc[mask, 'is_final_regular_week'] = 1
            print(f"  {year}: Final regular season week = {last_full_week} ({n_managers} managers)")

    # ===========================
    # COMPUTE SUMMARY STATISTICS
    # ===========================
    # NOTE: avg_wins, avg_seed, avg_playoffs, avg_bye are now calculated DIRECTLY
    # from histogram counts in the main loop above (no rounding errors).
    # The calculate_summary_stats function is no longer needed for these values.
    print("\nSummary statistics already calculated from raw counts (skipping recalculation)...")

    # Actual vs expected deltas (performance-based)
    if "wins_to_date" in df.columns and "shuffle_avg_wins" in df.columns:
        mask = regular_mask & df["wins_to_date"].notna() & df["shuffle_avg_wins"].notna()
        df.loc[mask, "wins_vs_shuffle_wins"] = (
            pd.to_numeric(df.loc[mask, "wins_to_date"], errors="coerce") -
            pd.to_numeric(df.loc[mask, "shuffle_avg_wins"], errors="coerce")
        ).round(2)

    if "playoff_seed_to_date" in df.columns and "shuffle_avg_seed" in df.columns:
        mask = regular_mask & df["playoff_seed_to_date"].notna() & df["shuffle_avg_seed"].notna()
        df.loc[mask, "seed_vs_shuffle_seed"] = (
            pd.to_numeric(df.loc[mask, "playoff_seed_to_date"], errors="coerce") -
            pd.to_numeric(df.loc[mask, "shuffle_avg_seed"], errors="coerce")
        ).round(2)

    # Actual vs expected deltas (opponent difficulty-based)
    if "wins_to_date" in df.columns and "opp_shuffle_avg_wins" in df.columns:
        mask = regular_mask & df["wins_to_date"].notna() & df["opp_shuffle_avg_wins"].notna()
        df.loc[mask, "wins_vs_opp_shuffle_wins"] = (
            pd.to_numeric(df.loc[mask, "wins_to_date"], errors="coerce") -
            pd.to_numeric(df.loc[mask, "opp_shuffle_avg_wins"], errors="coerce")
        ).round(2)

    if "playoff_seed_to_date" in df.columns and "opp_shuffle_avg_seed" in df.columns:
        mask = regular_mask & df["playoff_seed_to_date"].notna() & df["opp_shuffle_avg_seed"].notna()
        df.loc[mask, "seed_vs_opp_shuffle_seed"] = (
            pd.to_numeric(df.loc[mask, "playoff_seed_to_date"], errors="coerce") -
            pd.to_numeric(df.loc[mask, "opp_shuffle_avg_seed"], errors="coerce")
        ).round(2)

    # Simple opponent difficulty rank/percentile
    df = calculate_opponent_rank_percentile(df, regular_mask)

    # ===========================
    # LOCK POSTSEASON VALUES
    # ===========================
    print("Locking postseason values...")

    cols_to_lock = (SHUFFLE_SEED_COLS + SHUFFLE_WIN_COLS + SHUFFLE_SUMMARY_COLS +
                    OPP_SHUFFLE_SEED_COLS + OPP_SHUFFLE_WIN_COLS + OPP_SHUFFLE_SUMMARY_COLS +
                    OPP_RANK_COLS)

    df = lock_postseason_to_final_week(df, regular_mask, cols_to_lock)

    print(f"\nExpected records calculation complete!")
    print(f"Updated {len(df)} records with expected record simulations")

    # ===========================
    # FILL BYE WEEKS
    # ===========================
    # Add rows for teams with bye weeks (no opponent)
    # Weekly stats = 0, cumulative stats carried forward from previous week
    df = fill_bye_weeks(df)
    validate_bye_week_coverage(df)

    return df


# =========================================================
# CLI Interface
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate expected records using Monte Carlo schedule simulations"
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
    parser.add_argument(
        "--n-sims",
        type=int,
        default=N_SIMS,
        help=f"Number of simulations (default: {N_SIMS:,})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None/random)"
    )

    args = parser.parse_args()

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"Loaded league context: {ctx.league_name}")

    # Load matchup data
    matchup_path = ctx.canonical_matchup_file
    if not matchup_path.exists():
        raise FileNotFoundError(f"Matchup data not found: {matchup_path}")

    matchup_df = pd.read_parquet(matchup_path)
    print(f"Loaded {len(matchup_df)} matchup records")

    # Calculate expected records
    enriched_df = calculate_expected_records(
        matchup_df,
        current_week=args.current_week,
        current_year=args.current_year,
        n_sims=args.n_sims,
        rng_seed=args.seed
    )

    # Save results
    output_path = ctx.canonical_matchup_file
    enriched_df.to_parquet(output_path, index=False)
    print(f"\nSaved enriched matchup data to: {output_path}")

    # Also save CSV
    csv_path = output_path.with_suffix(".csv")
    enriched_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
