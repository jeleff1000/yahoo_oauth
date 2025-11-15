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

"""
Matchup to Player Import (Multi-League V2)

Joins matchup data to player data to add game context for each player performance.

This transformation:
- Imports matchup outcome columns (win, loss, playoffs, etc.)
- Imports scoring context (team_points, opponent_points, margins)
- Imports projection metrics (above/below projections, spread performance)
- Imports league comparison metrics (weekly rank, teams beat, etc.)
- Imports cumulative records (wins to date, streaks, etc.)
- Imports playoff odds (p_playoffs, expected wins, etc.)
- Imports expected records (shuffle metrics, schedule luck)

NOTE: cumulative_week and manager_week are NOT imported from matchup.
      They are calculated from player table's own year/week columns in initial_import_v2.py.

Join Key: (manager, year, week) or manager_week

Usage:
    python matchup_to_player_v2.py --context path/to/league_context.json
    python matchup_to_player_v2.py --context path/to/league_context.json --dry-run
    python matchup_to_player_v2.py --context path/to/league_context.json --backup
"""

from functools import wraps
import sys
from pathlib import Path


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

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys

import pandas as pd
import numpy as np

from core.league_context import LeagueContext


# =========================================================
# Column Configuration
# =========================================================

# Focused columns for player-level analysis
# Goal: Add game context without bloating file size

# Core matchup outcome columns (essential for player context)
# NOTE: cumulative_week is NOT imported from matchup - it's calculated from player's own year/week columns
CORE_OUTCOME_COLS = [
    "win", "loss", "team_points", "opponent", "opponent_points", "opponent_year", "margin",
    "is_playoffs", "is_consolation"
]

# Playoff achievement columns (useful for "clutch" player analysis)
PLAYOFF_ACHIEVEMENT_COLS = [
    "team_made_playoffs", "quarterfinal", "semifinal", "champion"
]

# League comparison columns (helps contextualize player performance)
LEAGUE_COMPARISON_COLS = [
    "weekly_rank", "teams_beat_this_week",
    "above_league_median",
    "manager_all_time_win_pct",
    "manager_all_time_gp",
    "manager_all_time_wins",
    "manager_all_time_losses"
]

# Combine all import columns (focused set)
ALL_IMPORT_COLS = (
    CORE_OUTCOME_COLS +
    PLAYOFF_ACHIEVEMENT_COLS +
    LEAGUE_COMPARISON_COLS
)


# =========================================================
# Helper Functions
# =========================================================

@ensure_normalized
def ensure_join_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure join keys exist in DataFrame.

    Required keys: manager, year, week
    Optional convenience key: manager_week (will be created if missing)
    """
    required = ["manager", "year", "week"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required join keys in {name}: {missing}. "
            f"Expected columns: {required}"
        )

    # Create manager_week if not present (for convenience)
    if "manager_week" not in df.columns:
        df["manager_week"] = df.apply(
            lambda r: (
                re.sub(r"\s+", "", str(r.get('manager', ''))) + str(int(r['cumulative_week']))
                if pd.notna(r.get("manager")) and pd.notna(r.get("cumulative_week"))
                else pd.NA
            ) if "cumulative_week" in df.columns else (
                re.sub(r"\s+", "", str(r.get('manager', ''))) +
                str(int(r['year'])) + str(int(r['week']))
                if pd.notna(r.get("manager")) and pd.notna(r.get("year")) and pd.notna(r.get("week"))
                else pd.NA
            ),
            axis=1
        ).astype("string")

    return df


def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


# =========================================================
# Main Import Function
# =========================================================

def import_matchup_to_player(
    player_path: Path,
    matchup_path: Path,
    import_cols: List[str] = ALL_IMPORT_COLS,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Import matchup columns into player data.

    Args:
        player_path: Path to player.parquet
        matchup_path: Path to matchup.parquet
        import_cols: List of columns to import from matchup
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics about the import
    """
    print(f"Loading player data from: {player_path}")
    player = pd.read_parquet(player_path)
    print(f"  Loaded {len(player):,} player records")

    print(f"\nLoading matchup data from: {matchup_path}")
    matchup = pd.read_parquet(matchup_path)
    print(f"  Loaded {len(matchup):,} matchup records")

    # Ensure join keys exist
    player = ensure_join_keys(player, "player")
    matchup = ensure_join_keys(matchup, "matchup")

    # Filter import_cols to only those that exist in matchup
    available_cols = [c for c in import_cols if c in matchup.columns]
    missing_cols = sorted(set(import_cols) - set(available_cols))

    if missing_cols:
        print(f"\nNote: {len(missing_cols)} columns not found in matchup (will be skipped):")
        for col in missing_cols[:10]:  # Show first 10
            print(f"  - {col}")
        if len(missing_cols) > 10:
            print(f"  ... and {len(missing_cols) - 10} more")

    print(f"\nImporting {len(available_cols)} columns from matchup to player")

    # Select needed columns from matchup
    join_keys = ["manager", "year", "week"]
    matchup_subset = matchup[join_keys + available_cols].copy()

    # Merge on (manager, year, week)
    print(f"\nJoining on: {join_keys}")
    merged = player.merge(
        matchup_subset,
        on=join_keys,
        how="left",
        suffixes=("", "_from_matchup")
    )

    # Track update statistics
    update_stats: Dict[str, int] = {}

    # Process each import column
    for col in available_cols:
        src_col = f"{col}_from_matchup"

        if src_col not in merged.columns:
            # Column already existed in player, no suffix added
            continue

        # Count how many rows will be updated
        src = merged[src_col]
        if col in player.columns:
            before = merged[col]
            will_update_mask = src.notna() & (before.isna() | (src != before))
        else:
            before = pd.Series([np.nan] * len(merged))
            will_update_mask = src.notna()

        update_stats[col] = int(will_update_mask.sum())

        # Prefer matchup value when available, otherwise keep existing
        merged[col] = src.combine_first(merged.get(col, pd.Series([np.nan]*len(merged))))

        # Drop the helper column
        merged.drop(columns=[src_col], inplace=True)

    # Fix is_playoffs/is_consolation logic
    if "is_consolation" in merged.columns and "is_playoffs" in merged.columns:
        consolation_mask = merged["is_consolation"].fillna(0).astype(int) == 1
        if consolation_mask.any():
            merged.loc[consolation_mask, "is_playoffs"] = 0
            print(f"\n[Fix Applied] Set is_playoffs=0 for {consolation_mask.sum():,} consolation game rows")

    # NOTE: cumulative_week and manager_week are NOT recreated here
    # They are calculated from player table's own year/week columns in initial_import_v2.py
    # This script only imports win/loss/opponent/points data from matchup table

    # Create matchup_name, team_1, team_2 for H2H UI
    if "manager" in merged.columns and "opponent" in merged.columns:
        print(f"\n[Fix Applied] Creating matchup_name, team_1, team_2 columns...")

        def create_matchup_cols(row):
            """Create matchup columns with alphabetical ordering."""
            manager = str(row.get('manager', ''))
            opponent = str(row.get('opponent', ''))

            if not manager or manager == 'nan' or not opponent or opponent == 'nan':
                return pd.Series({
                    'matchup_name': pd.NA,
                    'team_1': pd.NA,
                    'team_2': pd.NA
                })

            # Sort alphabetically for consistent naming
            teams = sorted([manager, opponent])
            matchup_name = '__vs__'.join(teams)

            return pd.Series({
                'matchup_name': matchup_name,
                'team_1': teams[0],
                'team_2': teams[1]
            })

        matchup_cols = merged.apply(create_matchup_cols, axis=1)
        merged['matchup_name'] = matchup_cols['matchup_name'].astype("string")
        merged['team_1'] = matchup_cols['team_1'].astype("string")
        merged['team_2'] = matchup_cols['team_2'].astype("string")

        # Count how many created
        created_count = merged['matchup_name'].notna().sum()
        print(f"  Created matchup columns for {created_count:,} player rows")

    # Calculate match rate
    matched_rows = merged[available_cols[0]].notna().sum() if available_cols else 0
    match_rate = (matched_rows / len(merged) * 100) if len(merged) > 0 else 0

    result = {
        "total_player_rows": len(merged),
        "matched_rows": matched_rows,
        "match_rate_pct": round(match_rate, 2),
        **update_stats
    }

    # Save (unless dry-run)
    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)
        return result

    if make_backup and player_path.exists():
        bpath = backup_file(player_path)
        print(f"\n[Backup Created] {bpath}")

    # Write back to player.parquet
    merged.to_parquet(player_path, index=False)
    print(f"\n[SAVED] Updated player data written to: {player_path}")

    return result


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Import matchup columns into player data for multi-league setup"
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

    # Run import
    print("\n" + "="*60)
    print("MATCHUP TO PLAYER IMPORT (V2)")
    print("="*60)

    stats = import_matchup_to_player(
        player_path=player_path,
        matchup_path=matchup_path,
        import_cols=ALL_IMPORT_COLS,
        dry_run=args.dry_run,
        make_backup=args.backup
    )

    # Print summary
    print("\n" + "="*60)
    print("IMPORT SUMMARY")
    print("="*60)
    print(f"Total player rows:    {stats['total_player_rows']:,}")
    print(f"Rows with matchup:    {stats['matched_rows']:,} ({stats['match_rate_pct']}%)")

    # Show top columns by update count
    col_updates = {k: v for k, v in stats.items()
                   if k not in ['total_player_rows', 'matched_rows', 'match_rate_pct']}

    if col_updates:
        print(f"\nTop 20 columns by rows updated:")
        sorted_cols = sorted(col_updates.items(), key=lambda x: x[1], reverse=True)[:20]
        for col, count in sorted_cols:
            print(f"  {col:>30}: {count:,}")

        if len(col_updates) > 20:
            print(f"\n  ... and {len(col_updates) - 20} more columns")

    print("="*60)


if __name__ == "__main__":
    main()
