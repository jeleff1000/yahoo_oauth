"""
Draft to Player Import (Multi-League V2)

Adds draft context to player data for draft value analysis.

This transformation adds draft-related columns to player data:
- Draft position info (round, pick, overall_pick)
- Draft cost/value (cost, is_keeper)
- Season performance context for ROI analysis

Join Key: (yahoo_player_id, year) - Draft is season-level, player is weekly

Usage:
    python draft_to_player_v2.py --context path/to/league_context.json
    python draft_to_player_v2.py --context path/to/league_context.json --dry-run
    python draft_to_player_v2.py --context path/to/league_context.json --backup
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict
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

from multi_league.core.league_context import LeagueContext


# =========================================================
# Configuration
# =========================================================

# Columns to import from draft to player (focused set for analytics)
DRAFT_COLS_TO_IMPORT = [
    "round",
    "pick",
    "cost",  # Auction cost or draft value
    "is_keeper_status",
    "kept_next_year",  # Whether player was kept in following season
    # Note: overall_pick and draft_type don't exist in draft table, removed
]


# =========================================================
# Helper Functions
# =========================================================

def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest

def left_join_draft_player(left: pd.DataFrame, right: pd.DataFrame, import_cols: list[str]) -> pd.DataFrame:
    """Left-join draft-level data (right) onto player-level data (left).

    Strategy:
    1. Ensure both frames have a `player_year` key if possible (yahoo_player_id + '_' + year).
    2. Prefer joining on (yahoo_player_id, year) when that key exists in both
       frames and is cleanly 1:1 on the right (no nulls, no duplicates).
    3. If (yahoo_player_id, year) is not usable, fall back to joining on
       `player_year` when it exists and is unique on the right.
    4. As a last resort, attempt a best-effort merge on available common keys
       containing any of ['player_year','yahoo_player_id','year'].

    The function returns the left dataframe with requested import_cols merged in.
    """
    L = left.copy()
    R = right.copy()

    # Filter import_cols to those present in R to avoid KeyErrors
    available_imports = [c for c in import_cols if c in R.columns]
    if not available_imports:
        # Nothing to import
        return L

    # Build player_year if missing and yahoo_player_id+year available
    for df in (L, R):
        if 'player_year' not in df.columns and {'yahoo_player_id', 'year'}.issubset(df.columns):
            df['player_year'] = df['yahoo_player_id'].astype(str) + "_" + df['year'].astype(str)

    # Attempt 1: join on (league_id, yahoo_player_id, year) - multi-league safe
    keys1 = ['league_id', 'yahoo_player_id', 'year']
    if set(keys1).issubset(L.columns) and set(keys1).issubset(R.columns):
        # Ensure no nulls in join keys and uniqueness on right side
        left_has_nulls = L[keys1].isnull().any(axis=1).any()
        right_has_nulls = R[keys1].isnull().any(axis=1).any()
        right_has_dupes = R.duplicated(subset=keys1, keep=False).any()

        if not left_has_nulls and not right_has_nulls and not right_has_dupes:
            return L.merge(R[keys1 + available_imports], on=keys1, how='left', suffixes=('_old', ''))

    # Fallback: Try without league_id for backward compatibility
    keys1_fallback = ['yahoo_player_id', 'year']
    if set(keys1_fallback).issubset(L.columns) and set(keys1_fallback).issubset(R.columns):
        left_has_nulls = L[keys1_fallback].isnull().any(axis=1).any()
        right_has_nulls = R[keys1_fallback].isnull().any(axis=1).any()
        right_has_dupes = R.duplicated(subset=keys1_fallback, keep=False).any()

        if not left_has_nulls and not right_has_nulls and not right_has_dupes:
            return L.merge(R[keys1_fallback + available_imports], on=keys1_fallback, how='left', suffixes=('_old', ''))

    # Attempt 2: join on player_year
    keys2 = ['player_year']
    if set(keys2).issubset(L.columns) and set(keys2).issubset(R.columns):
        right_has_dupes_py = R.duplicated(subset=keys2, keep=False).any()
        if not right_has_dupes_py:
            return L.merge(R[keys2 + available_imports], on=keys2, how='left', suffixes=('_old', ''))

    # Last-resort best-effort: try joining on whatever common keys we have
    common_candidates = [k for k in ['player_year', 'yahoo_player_id', 'year'] if k in L.columns and k in R.columns]
    if common_candidates:
        on_cols = common_candidates
        return L.merge(R[on_cols + available_imports], on=on_cols, how='left', suffixes=('_old', ''))

    # If no common keys at all, return the left unchanged
    return L

# =========================================================
# Main Import Function
# =========================================================

def import_draft_to_player(
    player_path: Path,
    draft_path: Path,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Import draft columns into player data.

    Args:
        player_path: Path to player.parquet
        draft_path: Path to draft.parquet
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics about the import
    """
    print(f"Loading player data from: {player_path}")
    player = pd.read_parquet(player_path)
    print(f"  Loaded {len(player):,} player records")

    print(f"\nLoading draft data from: {draft_path}")
    draft = pd.read_parquet(draft_path)
    print(f"  Loaded {len(draft):,} draft records")

    # Validate required columns
    if "yahoo_player_id" not in player.columns:
        raise KeyError("player.parquet missing required column: yahoo_player_id")
    if "year" not in player.columns:
        raise KeyError("player.parquet missing required column: year")

    if "yahoo_player_id" not in draft.columns:
        raise KeyError("draft.parquet missing required column: yahoo_player_id")
    if "year" not in draft.columns:
        raise KeyError("draft.parquet missing required column: year")

    print(f"\nJoining on: (yahoo_player_id, year)")

    # Filter draft columns to only those that exist
    available_cols = [c for c in DRAFT_COLS_TO_IMPORT if c in draft.columns]
    missing_cols = [c for c in DRAFT_COLS_TO_IMPORT if c not in draft.columns]

    if missing_cols:
        print(f"\nNote: {len(missing_cols)} columns not found in draft (will be skipped):")
        for col in missing_cols:
            print(f"  - {col}")

    print(f"\nImporting {len(available_cols)} columns from draft to player")

    # Select needed columns from draft
    join_keys = ["yahoo_player_id", "year"]
    draft_subset = draft[join_keys + available_cols].copy()

    # CRITICAL: Ensure draft_subset has no duplicate keys before merge
    # Draft should be one row per player-year (season-level data)
    duplicates_in_draft = draft_subset.duplicated(subset=join_keys, keep=False).sum()
    if duplicates_in_draft > 0:
        print(f"\n  WARNING: Found {duplicates_in_draft} duplicate player-year combinations in draft data")
        print(f"  Removing duplicates (keeping first occurrence)...")
        draft_subset = draft_subset.drop_duplicates(subset=join_keys, keep='first')
        print(f"  Deduplicated to {len(draft_subset):,} unique player-year combinations")

    # Track before counts
    before_counts = {col: player.get(col, pd.Series([np.nan]*len(player))).notna().sum()
                     for col in available_cols}

    # Merge draft data into player using robust left-join helper that
    # prefers (yahoo_player_id, year) when clean, and falls back to player_year.
    player = left_join_draft_player(player, draft_subset, available_cols)

    # Resolve conflicts (prefer new draft values, fall back to old)
    for col in available_cols:
        old_col = f"{col}_old"
        if old_col in player.columns:
            player[col] = player[col].fillna(player[old_col])
            player.drop(columns=[old_col], inplace=True)

    after_counts = {col: player[col].notna().sum() for col in available_cols}

    # Calculate statistics
    stats = {
        "total_player_rows": len(player),
        "unique_player_years": player[['yahoo_player_id', 'year']].drop_duplicates().shape[0],
    }

    for col in available_cols:
        stats[f"{col}_added"] = after_counts[col] - before_counts[col]
        stats[f"{col}_total"] = after_counts[col]

    # Calculate match rate
    drafted_players = player[player['round'].notna()] if 'round' in player.columns else pd.DataFrame()
    match_rate = (len(drafted_players) / len(player) * 100) if len(player) > 0 else 0

    stats["players_with_draft_data"] = len(drafted_players)
    stats["match_rate_pct"] = round(match_rate, 2)

    # Print summary
    print(f"\nImport Summary:")
    for col in available_cols:
        before = before_counts[col]
        after = after_counts[col]
        added = after - before
        print(f"  {col:>20}: {before:,} -> {after:,} (+{added:,})")

    print(f"\nOverall match rate: {stats['match_rate_pct']}% of player rows have draft data")

    # Save (unless dry-run)
    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)
        return stats

    if make_backup and player_path.exists():
        bpath = backup_file(player_path)
        print(f"\n[Backup Created] {bpath}")

    # Ensure draft columns have proper data types before writing to parquet
    for col in available_cols:
        if col in player.columns:
            # Convert is_keeper_status and similar columns to Int64 (nullable integer)
            if col in ['is_keeper_status', 'kept_next_year', 'round', 'pick']:
                try:
                    player[col] = pd.to_numeric(player[col], errors='coerce').astype('Int64')
                except Exception as e:
                    print(f"  Warning: Could not convert {col} to Int64: {e}")
            # Convert cost to float
            elif col in ['cost']:
                try:
                    player[col] = pd.to_numeric(player[col], errors='coerce')
                except Exception as e:
                    print(f"  Warning: Could not convert {col} to numeric: {e}")

    # Write back to player.parquet
    player.to_parquet(player_path, index=False)
    print(f"\n[SAVED] Updated player data written to: {player_path}")

    return stats


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Import draft context into player data for multi-league setup"
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
        "--draft",
        type=str,
        help="Override draft data path"
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
        draft_path = Path(args.draft) if args.draft else ctx.canonical_draft_file
    else:
        # Fallback to hardcoded relative paths (V1 compatibility)
        if not args.player or not args.draft:
            THIS_FILE = Path(__file__).resolve()
            SCRIPT_DIR = THIS_FILE.parent.parent.parent  # Up to fantasy_football_data_scripts
            ROOT_DIR = SCRIPT_DIR.parent  # Up to fantasy_football_data_downloads
            DATA_DIR = ROOT_DIR / "fantasy_football_data"

            player_path = Path(args.player) if args.player else (DATA_DIR / "player.parquet")
            draft_path = Path(args.draft) if args.draft else (DATA_DIR / "draft.parquet")
        else:
            player_path = Path(args.player)
            draft_path = Path(args.draft)

    # Validate paths
    if not player_path.exists():
        raise FileNotFoundError(f"Player data not found: {player_path}")
    if not draft_path.exists():
        raise FileNotFoundError(f"Draft data not found: {draft_path}")

    # Run import
    print("\n" + "="*60)
    print("DRAFT TO PLAYER IMPORT (V2)")
    print("="*60)

    stats = import_draft_to_player(
        player_path=player_path,
        draft_path=draft_path,
        dry_run=args.dry_run,
        make_backup=args.backup
    )

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total player rows:        {stats['total_player_rows']:,}")
    print(f"Unique player-years:      {stats['unique_player_years']:,}")
    print(f"Rows with draft data:     {stats['players_with_draft_data']:,} ({stats['match_rate_pct']}%)")
    print("="*60)


if __name__ == "__main__":
    main()
