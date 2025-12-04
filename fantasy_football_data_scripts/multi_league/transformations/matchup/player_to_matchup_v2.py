"""
Player to Matchup Import (Multi-League V2)

Aggregates player-level statistics to matchup level for manager performance context.

This transformation aggregates player data by manager_week and adds to matchup:
- optimal_points: Maximum possible points from optimal lineup
- bench_points: Total points left on bench
- lineup_efficiency: Percentage of optimal achieved (team_points / optimal * 100)
- optimal_ppg_season: Average optimal points per game (season)
- rolling_optimal_points: Cumulative optimal points (season to date)
- total_optimal_points: Total optimal points (full season)
- optimal_points_all_time: Cumulative optimal points (career)
- optimal_win/loss: W/L record if both managers played optimal lineups
- opponent_optimal_points: Opponent's optimal points this week
- total_player_points: Sum of all rostered player points (verification)
- players_rostered: Count of rostered players this week
- players_started: Count of started players this week

Join Key: manager_week (must align between player and matchup)

Usage:
    python player_to_matchup_v2.py --context path/to/league_context.json
    python player_to_matchup_v2.py --context path/to/league_context.json --dry-run
    python player_to_matchup_v2.py --context path/to/league_context.json --backup
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

import pandas as pd
import numpy as np



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

# Add transformations/common to path for shared utilities
_common_dir = _multi_league_dir / "transformations" / "common"
sys.path.insert(0, str(_common_dir))

from core.league_context import LeagueContext
from type_utils import ensure_canonical_types


# =========================================================
# Helper Functions
# =========================================================

def to_str(s: pd.Series) -> pd.Series:
    """Normalize to consistent string keys for joins."""
    return s.astype("string").fillna(pd.NA).str.strip()


def safe_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric; invalid parses become NaN."""
    return pd.to_numeric(s, errors="coerce")


def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


# =========================================================
# Main Import Function
# =========================================================

def aggregate_player_to_matchup(
    player_path: Path,
    matchup_path: Path,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Aggregate player stats to matchup level.

    Args:
        player_path: Path to player.parquet
        matchup_path: Path to matchup.parquet
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics about the aggregation
    """
    print(f"Loading player data from: {player_path}")
    player = pd.read_parquet(player_path)
    print(f"  Loaded {len(player):,} player records")

    print(f"\nLoading matchup data from: {matchup_path}")
    matchup = pd.read_parquet(matchup_path)
    print(f"  Loaded {len(matchup):,} matchup records")

    # Validate required columns (handle manager_week / manager_year_week alias)
    # Create manager_week from manager_year_week if needed (bidirectional alias)
    if "manager_week" not in player.columns:
        if "manager_year_week" in player.columns:
            player["manager_week"] = player["manager_year_week"]
            print("  Created manager_week from manager_year_week in player data")
        else:
            raise KeyError("player.parquet missing required column: manager_week (or manager_year_week)")

    if "manager_week" not in matchup.columns:
        if "manager_year_week" in matchup.columns:
            matchup["manager_week"] = matchup["manager_year_week"]
            print("  Created manager_week from manager_year_week in matchup data")
        else:
            raise KeyError("matchup.parquet missing required column: manager_week (or manager_year_week)")

    # Normalize join keys
    player["manager_week"] = to_str(player["manager_week"])
    matchup["manager_week"] = to_str(matchup["manager_week"])

    # Create opponent_week if it doesn't exist (needed for optimal win/loss calculation)
    if "opponent_week" not in matchup.columns:
        if "opponent" in matchup.columns and "cumulative_week" in matchup.columns:
            print("  Creating opponent_week column from opponent + cumulative_week")
            matchup["opponent_week"] = (
                matchup["opponent"].str.replace(" ", "", regex=False) +
                matchup["cumulative_week"].astype(str)
            )
        else:
            print("  [WARN] Cannot create opponent_week: missing opponent or cumulative_week columns")

    if "opponent_week" in matchup.columns:
        matchup["opponent_week"] = to_str(matchup["opponent_week"])

    # Normalize manager_year for season aggregations
    if "manager_year" not in matchup.columns:
        if "manager" in matchup.columns and "year" in matchup.columns:
            matchup["manager_year"] = (
                matchup["manager"].str.replace(" ", "", regex=False) +
                matchup["year"].astype(str)
            )

    print(f"\nAggregating player stats by manager_week...")

    # =========================================================
    # Aggregate Player Stats by Manager-Week
    # =========================================================

    # Check which columns are available
    has_optimal = "optimal_points" in player.columns
    has_bench = "bench_points" in player.columns
    has_fantasy = "fantasy_points" in player.columns
    has_rostered = "is_rostered" in player.columns
    has_started = "is_started" in player.columns

    agg_dict = {}

    if has_optimal:
        agg_dict["optimal_points"] = ("optimal_points", "first")  # Same for all players in week

    if has_bench:
        agg_dict["bench_points"] = ("bench_points", "first")  # Same for all players in week

    if has_fantasy:
        agg_dict["total_player_points"] = ("fantasy_points", "sum")

    if has_rostered:
        agg_dict["players_rostered"] = ("is_rostered", lambda x: (x == True).sum())

    if has_started:
        agg_dict["players_started"] = ("is_started", lambda x: (x == True).sum())

    if not agg_dict:
        print("[ERROR] No aggregatable columns found in player data!")
        print("  Expected at least one of: optimal_points, bench_points, fantasy_points")
        sys.exit(1)

    # Perform aggregation
    player_agg = (
        player[player["manager_week"].notna()]
        .groupby("manager_week", as_index=False, dropna=False)
        .agg(**agg_dict)
    )

    print(f"  Aggregated to {len(player_agg):,} unique manager_week combinations")

    # =========================================================
    # Merge Aggregated Stats into Matchup
    # =========================================================

    print(f"\nMerging aggregated stats into matchup...")

    before_counts = {col: matchup.get(col, pd.Series([np.nan]*len(matchup))).notna().sum()
                     for col in agg_dict.keys()}

    matchup = matchup.merge(
        player_agg,
        on="manager_week",
        how="left",
        suffixes=("_old", "")
    )

    # Resolve conflicts (prefer new values, fall back to old)
    for col in agg_dict.keys():
        old_col = f"{col}_old"
        if old_col in matchup.columns:
            matchup[col] = matchup[col].fillna(matchup[old_col])
            matchup.drop(columns=[old_col], inplace=True)

        # Ensure numeric
        if col not in ["players_rostered", "players_started"]:
            matchup[col] = safe_numeric(matchup[col])

    after_counts = {col: matchup[col].notna().sum() for col in agg_dict.keys()}

    for col in agg_dict.keys():
        print(f"  {col}: {before_counts[col]:,} -> {after_counts[col]:,} non-null")

    # =========================================================
    # Calculate Derived Metrics
    # =========================================================

    print(f"\nCalculating derived metrics...")

    # Lineup efficiency (actual / optimal * 100)
    if "optimal_points" in matchup.columns and "team_points" in matchup.columns:
        with np.errstate(invalid="ignore", divide="ignore"):
            matchup["lineup_efficiency"] = (
                matchup["team_points"] / matchup["optimal_points"] * 100.0
            ).round(2)
        print(f"  Added lineup_efficiency")

    # Optimal PPG (season average)
    if "optimal_points" in matchup.columns and "manager_year" in matchup.columns:
        season_sum = matchup.groupby("manager_year", dropna=False)["optimal_points"].transform("sum")
        season_cnt = matchup.groupby("manager_year", dropna=False)["optimal_points"].transform(
            lambda s: s.notna().sum()
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            matchup["optimal_ppg_season"] = (season_sum / season_cnt.replace(0, np.nan)).round(2)
        print(f"  Added optimal_ppg_season")

    # Rolling optimal points (cumulative within season)
    if "optimal_points" in matchup.columns and "manager_year" in matchup.columns and "week" in matchup.columns:
        matchup["week"] = safe_numeric(matchup["week"])
        matchup["_week_sort"] = matchup["week"].fillna(10**9)
        matchup["_opt_fill0"] = matchup["optimal_points"].fillna(0)

        matchup["rolling_optimal_points"] = (
            matchup.sort_values(["manager_year", "_week_sort"])
            .groupby("manager_year", dropna=False)["_opt_fill0"]
            .cumsum()
            .values
        )
        matchup.drop(columns=["_week_sort", "_opt_fill0"], inplace=True)
        print(f"  Added rolling_optimal_points")

    # Total optimal points (season total)
    if "optimal_points" in matchup.columns and "manager_year" in matchup.columns:
        matchup["total_optimal_points"] = (
            matchup.groupby("manager_year", dropna=False)["optimal_points"]
            .transform("sum")
            .fillna(0)
        )
        print(f"  Added total_optimal_points")

    # Optimal points all-time (career total)
    if "optimal_points" in matchup.columns and "manager" in matchup.columns:
        matchup["optimal_points_all_time"] = (
            matchup.groupby("manager", dropna=False)["optimal_points"]
            .transform("sum")
            .fillna(0)
        )
        print(f"  Added optimal_points_all_time")

    # Optimal win/loss (if both managers played optimal lineups)
    if "optimal_points" in matchup.columns and "opponent_week" in matchup.columns:
        print(f"\nCalculating optimal win/loss...")

        opp_optimal = matchup[["manager_week", "optimal_points"]].rename(columns={
            "manager_week": "opponent_week",
            "optimal_points": "opponent_optimal_points"
        })

        matchup = matchup.merge(opp_optimal, on="opponent_week", how="left")

        if "opponent_optimal_points" not in matchup.columns:
            matchup["opponent_optimal_points"] = np.nan

        matchup["optimal_win"] = np.where(
            matchup["optimal_points"] > matchup["opponent_optimal_points"], 1, 0
        )
        matchup["optimal_loss"] = np.where(
            matchup["optimal_points"] < matchup["opponent_optimal_points"], 1, 0
        )

        print(f"  Added optimal_win, optimal_loss, opponent_optimal_points")

    # =========================================================
    # Cleanup
    # =========================================================

    # Keep manager_week column for downstream transformations
    # Also ensure manager_year_week exists (bidirectional alias)
    if "manager_week" in matchup.columns and "manager_year_week" not in matchup.columns:
        matchup["manager_year_week"] = matchup["manager_week"]
    elif "manager_year_week" in matchup.columns and "manager_week" not in matchup.columns:
        matchup["manager_week"] = matchup["manager_year_week"]

    # Drop any duplicate or suffix columns
    cols_to_drop = []
    for c in matchup.columns:
        if any(c.endswith(suffix) for suffix in ['_from_player', '_old', '_x', '_y']):
            cols_to_drop.append(c)

    if cols_to_drop:
        print(f"\n[Cleanup] Dropping temporary columns: {cols_to_drop}")
        matchup.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Calculate statistics
    stats = {
        "total_matchup_rows": len(matchup),
        "rows_with_optimal": matchup.get("optimal_points", pd.Series([np.nan]*len(matchup))).notna().sum(),
        "rows_with_bench": matchup.get("bench_points", pd.Series([np.nan]*len(matchup))).notna().sum(),
    }

    # =========================================================
    # Save
    # =========================================================

    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)

        print("\nColumns added:")
        new_cols = ["optimal_points", "bench_points", "lineup_efficiency",
                    "optimal_ppg_season", "rolling_optimal_points",
                    "total_optimal_points", "optimal_points_all_time",
                    "optimal_win", "optimal_loss", "opponent_optimal_points",
                    "total_player_points", "players_rostered", "players_started"]
        for col in new_cols:
            if col in matchup.columns:
                print(f"  - {col}")

        print("\nSample data:")
        preview_cols = [c for c in [
            "manager", "year", "week", "team_points",
            "optimal_points", "lineup_efficiency",
            "optimal_win", "optimal_loss"
        ] if c in matchup.columns]
        print(matchup[preview_cols].head(10))

        return stats

    if make_backup and matchup_path.exists():
        bpath = backup_file(matchup_path)
        print(f"\n[Backup Created] {bpath}")

    # Ensure all join keys have correct types before saving
    matchup = ensure_canonical_types(matchup, verbose=False)

    # Write back
    matchup.to_parquet(matchup_path, index=False)
    print(f"\n[SAVED] Updated matchup data written to: {matchup_path}")

    return stats


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate player stats to matchup level for multi-league setup"
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
        "--matchup",
        type=str,
        help="Override matchup data path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create timestamped backup before saving"
    )

    args = parser.parse_args()

    # Determine file paths
    if args.context:
        ctx = LeagueContext.load(args.context)
        print(f"Loaded league context: {ctx.league_name}")

        player_path = Path(args.player) if args.player else ctx.canonical_player_file
        matchup_path = Path(args.matchup) if args.matchup else ctx.canonical_matchup_file
    else:
        # Fallback to hardcoded relative paths (V1 compatibility)
        if not args.player or not args.matchup:
            THIS_FILE = Path(__file__).resolve()
            SCRIPT_DIR = THIS_FILE.parent.parent.parent  # Up to fantasy_football_data_scripts
            ROOT_DIR = SCRIPT_DIR.parent  # Up to fantasy_football_data_downloads
            DATA_DIR = ROOT_DIR / "fantasy_football_data"

            player_path = Path(args.player) if args.player else (DATA_DIR / "player.parquet")
            matchup_path = Path(args.matchup) if args.matchup else (DATA_DIR / "matchup.parquet")
        else:
            player_path = Path(args.player)
            matchup_path = Path(args.matchup)

    # Validate paths
    if not player_path.exists():
        raise FileNotFoundError(f"Player data not found: {player_path}")
    if not matchup_path.exists():
        raise FileNotFoundError(f"Matchup data not found: {matchup_path}")

    # Run aggregation
    print("\n" + "="*60)
    print("PLAYER TO MATCHUP AGGREGATION (V2)")
    print("="*60)

    stats = aggregate_player_to_matchup(
        player_path=player_path,
        matchup_path=matchup_path,
        dry_run=args.dry_run,
        make_backup=args.backup
    )

    # Print summary
    print("\n" + "="*60)
    print("AGGREGATION SUMMARY")
    print("="*60)
    print(f"Total matchup rows:     {stats['total_matchup_rows']:,}")
    print(f"Rows with optimal:      {stats['rows_with_optimal']:,}")
    print(f"Rows with bench:        {stats['rows_with_bench']:,}")
    print("="*60)


if __name__ == "__main__":
    main()
