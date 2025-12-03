#!/usr/bin/env python3
"""
Enrich Schedule with Playoff Flags

After cumulative_stats_v2.py calculates playoff flags on matchup data,
this script merges those flags into schedule.parquet for downstream use
by playoff_odds_import.py.

This ensures schedule.parquet has:
- is_playoffs
- is_consolation
- playoff_round
- consolation_round
- Other relevant playoff metadata

Input:
- matchup.parquet (with playoff flags from cumulative_stats_v2.py)
- schedule.parquet (basic schedule data)

Output:
- schedule.parquet (enriched with playoff flags)

Usage:
    python enrich_schedule_with_playoff_flags.py --context /path/to/league_context.json
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()
_base_dir = _script_file.parent  # base/
_transformations_dir = _base_dir.parent  # transformations/
_multi_league_dir = _transformations_dir.parent  # multi_league/
_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts/
sys.path.insert(0, str(_scripts_dir))

from multi_league.core.league_context import LeagueContext


def enrich_schedule_with_playoff_flags(ctx: LeagueContext, verbose: bool = True) -> pd.DataFrame:
    """
    Merge playoff flags from matchup.parquet into schedule.parquet.

    Args:
        ctx: LeagueContext with paths to data files
        verbose: Print progress messages

    Returns:
        Enriched schedule DataFrame
    """
    def log(msg):
        if verbose:
            print(msg)

    # Load matchup data (source of playoff flags)
    matchup_path = ctx.canonical_matchup_file
    if not matchup_path.exists():
        raise FileNotFoundError(f"matchup.parquet not found: {matchup_path}")

    log(f"[LOAD] Reading matchup data from: {matchup_path}")
    matchup_df = pd.read_parquet(matchup_path)
    log(f"[LOAD] Loaded {len(matchup_df):,} matchup records")

    # Load schedule data (target for enrichment)
    schedule_path = Path(ctx.data_directory) / "schedule.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError(f"schedule.parquet not found: {schedule_path}")

    log(f"[LOAD] Reading schedule data from: {schedule_path}")
    schedule_df = pd.read_parquet(schedule_path)
    log(f"[LOAD] Loaded {len(schedule_df):,} schedule records")

    # Identify playoff flag columns to merge from matchup to schedule
    playoff_flag_cols = [
        'is_playoffs',
        'is_consolation',
        'playoff_round',
        'consolation_round',
        'playoff_week_index',
        'playoff_round_num',
        'quarterfinal',
        'semifinal',
        'championship',
        'placement_game',
        'consolation_semifinal',
        'consolation_final',
        'postseason',
    ]

    # Only merge columns that exist in matchup data
    cols_to_merge = [c for c in playoff_flag_cols if c in matchup_df.columns]

    if not cols_to_merge:
        log("[WARN] No playoff flag columns found in matchup.parquet - skipping enrichment")
        return schedule_df

    log(f"[MERGE] Merging {len(cols_to_merge)} playoff flag columns from matchup to schedule")
    log(f"[MERGE] Columns: {', '.join(cols_to_merge)}")

    # Create a mapping of (year, week, manager) -> playoff flags
    # We'll aggregate matchup data by year/week/manager to get unique playoff flags per manager-week
    join_keys = ['year', 'week', 'manager']

    # Ensure join keys exist in both dataframes
    missing_keys_matchup = [k for k in join_keys if k not in matchup_df.columns]
    missing_keys_schedule = [k for k in join_keys if k not in schedule_df.columns]

    if missing_keys_matchup:
        raise ValueError(f"matchup.parquet missing join keys: {missing_keys_matchup}")
    if missing_keys_schedule:
        raise ValueError(f"schedule.parquet missing join keys: {missing_keys_schedule}")

    # Extract playoff flags from matchup data
    # Group by join keys and take first value (should be same for all matchups in same week)
    playoff_flags = matchup_df[join_keys + cols_to_merge].groupby(join_keys, as_index=False).first()

    log(f"[MERGE] Extracted {len(playoff_flags):,} unique playoff flag records")

    # Merge playoff flags into schedule data
    # Left join to preserve all schedule records
    schedule_enriched = schedule_df.merge(
        playoff_flags,
        on=join_keys,
        how='left',
        suffixes=('', '_matchup')
    )

    # Handle any duplicate columns from merge (prefer matchup values)
    for col in cols_to_merge:
        matchup_col = f'{col}_matchup'
        if matchup_col in schedule_enriched.columns:
            # Use matchup value if available, otherwise keep schedule value
            if col in schedule_df.columns:
                schedule_enriched[col] = schedule_enriched[matchup_col].fillna(schedule_enriched[col])
            else:
                schedule_enriched[col] = schedule_enriched[matchup_col]
            schedule_enriched = schedule_enriched.drop(columns=[matchup_col])

    # Fill any remaining NaN values in playoff flags with defaults
    if 'is_playoffs' in schedule_enriched.columns:
        schedule_enriched['is_playoffs'] = schedule_enriched['is_playoffs'].fillna(0).astype(int)
    if 'is_consolation' in schedule_enriched.columns:
        schedule_enriched['is_consolation'] = schedule_enriched['is_consolation'].fillna(0).astype(int)

    # Count enriched records
    if 'is_playoffs' in schedule_enriched.columns:
        playoff_weeks = (schedule_enriched['is_playoffs'] == 1).sum()
        consolation_weeks = (schedule_enriched['is_consolation'] == 1).sum()
        log(f"[ENRICH] Playoff weeks: {playoff_weeks:,}")
        log(f"[ENRICH] Consolation weeks: {consolation_weeks:,}")

    # Save enriched schedule
    log(f"[WRITE] Writing enriched schedule to: {schedule_path}")
    schedule_enriched.to_parquet(schedule_path, index=False)

    # Also write CSV for inspection
    csv_path = schedule_path.parent / "schedule.csv"
    schedule_enriched.to_csv(csv_path, index=False)
    log(f"[WRITE] Wrote schedule CSV to: {csv_path}")

    log(f"[SUCCESS] Schedule enrichment complete ({len(schedule_enriched):,} records)")

    return schedule_enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich schedule.parquet with playoff flags from matchup.parquet")
    parser.add_argument("--context", required=True, help="Path to league_context.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)

    try:
        enrich_schedule_with_playoff_flags(ctx, verbose=not args.quiet)
        return 0
    except Exception as e:
        print(f"[ERROR] Schedule enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
