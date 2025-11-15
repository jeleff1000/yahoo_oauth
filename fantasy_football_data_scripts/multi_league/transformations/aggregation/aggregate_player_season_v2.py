"""
Aggregate Player to Season/Career (Multi-League V2)

Creates leaner aggregated player datasets for UI display.

This transformation removes weekly-specific columns and filters data to create:
- players_by_year.parquet - Season-aggregated player data
- Removes rolling/weekly columns not needed for season/career views
- Filters out zero-point rows without yahoo_player_id
- Optimized for season/career UI tabs

Usage:
    python aggregate_player_season_v2.py --context path/to/league_context.json
    python aggregate_player_season_v2.py --context path/to/league_context.json --dry-run
    python aggregate_player_season_v2.py --context path/to/league_context.json --overwrite
"""

import argparse
from pathlib import Path
from typing import Dict
import sys

import pandas as pd
import duckdb



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


# =========================================================
# Configuration
# =========================================================

# Columns to drop for season/career views
# These are weekly-specific or redundant for aggregated views
DROP_COLS = [
    # Original Yahoo/NFL columns (redundant after merge)
    "points_original",
    "points_dst_from_yahoo_settings",
    "fantasy_points_zero_ppr",
    "fantasy_points_ppr",
    "fantasy_points_half_ppr",

    # Key columns (not needed after processing)
    "player_key",
    "player_last_name_key",
    "position_key",
    "points_key",
    "year_key",
    "week_key",

    # Composite keys (can be reconstructed if needed)
    "manager_year",
    "opponent_year",
    "manager_week",
    "player_week",
    "opponent_week",

    # Weekly rolling columns (not meaningful for season/career)
    "rolling_point_total",
    "rolling_optimal_points",
    "rolling_3_avg",
    "rolling_5_avg",

    # Temporary/processing columns
    "dummy",
    "max_week",
    "team_1",
    "team_2",
    "bye",

    # Detailed kicking lists (too granular for season view)
    "fg_made_list",
    "fg_missed_list",

    # Weekly lineup positions
    "lineup_position",
    "optimal_lineup_position",
    "league_wide_optimal_position",

    # Weekly optimal points (use season aggregates instead)
    "optimal_points",
    "optimal_ppg",

    # Percentile columns (can be recalculated from ranks if needed)
    "manager_player_all_time_history_percentile",
    "manager_player_season_history_percentile",
    "manager_position_all_time_history_percentile",
    "manager_position_season_history_percentile",
    "player_personal_all_time_history_percentile",
    "player_personal_season_history_percentile",
    "position_all_time_history_percentile",
    "position_season_history_percentile",
    "player_all_time_history_percentile",
    "player_season_history_percentile",

    # Forward-looking keeper columns (not needed for historical views)
    "keeper_price",
    "kept_next_year",
    "avg_cost_next_year",
    "total_points_next_year",

    # Matchup-specific columns (not needed for player season aggregation)
    "matchup_name",
    "computed_points",

    # Miscellaneous
    "player_last_name",
]


# =========================================================
# Main Aggregation Function
# =========================================================

def aggregate_player_to_season(
    player_path: Path,
    output_path: Path,
    overwrite_input: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Create leaner player dataset for season/career UI views.

    Args:
        player_path: Path to player.parquet (weekly data)
        output_path: Path to players_by_year.parquet (season data)
        overwrite_input: If True, overwrite player.parquet instead of creating new file
        dry_run: If True, don't write changes

    Returns:
        Dict with statistics
    """
    print(f"Loading player data from: {player_path}")

    # Use DuckDB for fast parquet reading
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{player_path.as_posix()}')").fetchdf()
    con.close()

    before_rows = len(df)
    print(f"  Loaded {before_rows:,} player records")

    # Filter: remove ALL rows where points == 0 (or NaN)
    # For season/career views, zero-point weeks don't provide value and bloat file size
    if "points" in df.columns:
        print(f"\nFiltering zero-point rows...")
        p = pd.to_numeric(df["points"], errors="coerce").fillna(0)
        bad = (p == 0)
        df = df.loc[~bad].copy()
        filtered_count = bad.sum()
        print(f"  Filtered {filtered_count:,} zero-point rows")

    # Drop weekly-specific columns
    print(f"\nDropping weekly-specific columns...")
    existing_drop = [c for c in DROP_COLS if c in df.columns]

    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"  Dropped {len(existing_drop)} columns:")
        for i, col in enumerate(existing_drop):
            if i < 10:  # Show first 10
                print(f"    - {col}")
        if len(existing_drop) > 10:
            print(f"    ... and {len(existing_drop) - 10} more")
    else:
        print(f"  No columns to drop")

    # CRITICAL: Deduplicate rows to prevent duplicate player-week combinations
    print(f"\nDeduplicating rows...")
    before_dedup = len(df)

    # Identify duplicate key columns
    dedupe_keys = []
    key_candidates = ["league_id", "year", "week", "yahoo_player_id", "NFL_player_id", "manager", "player"]
    for key in key_candidates:
        if key in df.columns:
            dedupe_keys.append(key)

    if dedupe_keys:
        duplicates_found = df.duplicated(subset=dedupe_keys, keep=False).sum()
        if duplicates_found > 0:
            print(f"  WARNING: Found {duplicates_found:,} duplicate rows based on keys: {dedupe_keys}")
            df = df.drop_duplicates(subset=dedupe_keys, keep='last')
            after_dedup = len(df)
            print(f"  Removed {before_dedup - after_dedup:,} duplicate rows")
        else:
            print(f"  No duplicates found")
            after_dedup = before_dedup
    else:
        print(f"  WARNING: No dedupe keys found in data, skipping deduplication")
        after_dedup = before_dedup

    after_rows = len(df)

    # Calculate statistics
    stats = {
        "rows_before": before_rows,
        "rows_after": after_rows,
        "rows_dropped": before_rows - after_rows,
        "rows_deduped": before_dedup - after_dedup,
        "columns_dropped": len(existing_drop),
        "columns_remaining": len(df.columns),
    }

    print(f"\nAggregation Summary:")
    print(f"  Rows before:        {stats['rows_before']:,}")
    print(f"  Rows after:         {stats['rows_after']:,}")
    print(f"  Rows dropped:       {stats['rows_dropped']:,}")
    print(f"  Rows deduped:       {stats['rows_deduped']:,}")
    print(f"  Columns dropped:    {stats['columns_dropped']}")
    print(f"  Columns remaining:  {stats['columns_remaining']}")

    # CRITICAL: Fix cumulative_week to ensure it's Int64 and not string "nan"
    # This prevents "Could not convert string 'nan' to DECIMAL" errors in MotherDuck
    if "cumulative_week" in df.columns:
        if df["cumulative_week"].dtype == "object":
            print(f"\n[FIX] Converting cumulative_week from object to Int64")
            df["cumulative_week"] = df["cumulative_week"].replace("nan", pd.NA)
            df["cumulative_week"] = pd.to_numeric(df["cumulative_week"], errors="coerce").astype("Int64")

    # Save
    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)

        print("\nRemaining columns:")
        for i, col in enumerate(sorted(df.columns)):
            if i < 20:
                print(f"  {col}")
        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more")

        return stats

    if overwrite_input:
        df.to_parquet(player_path, index=False)
        print(f"\n[SAVED] Overwrote player data: {player_path}")
    else:
        df.to_parquet(output_path, index=False)

        # Also save CSV version
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)

        print(f"\n[SAVED] Season-aggregated player data: {output_path}")
        print(f"[SAVED] CSV version: {csv_path}")

    return stats


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate player data to season/career for UI display"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Path to league_context.json (if not provided, uses hardcoded paths)"
    )
    parser.add_argument(
        "--player",
        type=str,
        help="Override player data path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output path (default: players_by_year.parquet)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite player.parquet instead of creating new file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files"
    )

    args = parser.parse_args()

    # Determine file paths
    if args.context:
        ctx = LeagueContext.load(args.context)
        print(f"Loaded league context: {ctx.league_name}")

        player_path = Path(args.player) if args.player else ctx.canonical_player_file
        output_path = Path(args.output) if args.output else (ctx.data_directory / "players_by_year.parquet")
    else:
        # Fallback to hardcoded relative paths (V1 compatibility)
        THIS_FILE = Path(__file__).resolve()
        SCRIPT_DIR = THIS_FILE.parent.parent.parent
        ROOT_DIR = SCRIPT_DIR.parent
        DATA_DIR = ROOT_DIR / "fantasy_football_data"

        player_path = Path(args.player) if args.player else (DATA_DIR / "player.parquet")
        output_path = Path(args.output) if args.output else (DATA_DIR / "players_by_year.parquet")

    # Validate input exists
    if not player_path.exists():
        raise FileNotFoundError(f"Player data not found: {player_path}")

    # Run aggregation
    print("\n" + "="*60)
    print("PLAYER SEASON/CAREER AGGREGATION (V2)")
    print("="*60)

    stats = aggregate_player_to_season(
        player_path=player_path,
        output_path=output_path,
        overwrite_input=args.overwrite,
        dry_run=args.dry_run
    )

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
