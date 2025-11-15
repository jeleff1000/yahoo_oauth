"""
Draft Enrichment Transformation (Multi-League V2)

Adds draft value metrics and forward-looking keeper columns to draft data.

This transformation adds:
1. Draft Value Metrics (refactored from draft_data_v2.py):
   - pick_savings: Picks saved vs ADP (positive = better value)
   - cost_savings: Auction dollars saved vs average cost (positive = better value)
   - savings: Unified value metric (cost_savings for auction, pick_savings for snake)
   - cost_bucket: Position-based value tier (1 = picks 1-3 at position, 2 = picks 4-6, etc.)

2. Forward-Looking Keeper Columns:
   - kept_next_year: Whether this player was kept in the following season

Logic:
- kept_next_year = 1 if is_keeper_status for same yahoo_player_id in year+1 is 1
- kept_next_year = 0 otherwise
- Value metrics use draft_type from fetcher or heuristic (25%+ non-zero cost = auction)

Usage:
    python draft_enrichment_v2.py --context path/to/league_context.json
    python draft_enrichment_v2.py --context path/to/league_context.json --dry-run
    python draft_enrichment_v2.py --context path/to/league_context.json --backup
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

from core.league_context import LeagueContext


def backup_file(file_path: Path) -> Path:
    """Create a timestamped backup of a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_name(f"{file_path.stem}_backup_{timestamp}{file_path.suffix}")

    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
        df.to_parquet(backup_path, index=False)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        df.to_csv(backup_path, index=False)
    else:
        import shutil
        shutil.copy2(file_path, backup_path)

    return backup_path


def assign_cost_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign cost buckets based on position and year.

    Cost buckets group players into tiers within their position/year:
    - Bucket 1: Picks 1-3 at position
    - Bucket 2: Picks 4-6 at position
    - Bucket 3: Picks 7-9 at position
    - etc.

    Args:
        df: DataFrame with draft data

    Returns:
        DataFrame with cost_bucket column added
    """
    df['cost_bucket'] = pd.NA
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

    mask = (
        df['cost'].notna()
        & (df['cost'] > 0)
        & df['yahoo_position'].notna()
        & (df['yahoo_position'] != "")
    )

    if not mask.any():
        return df

    df_to_bucket = df[mask].copy()
    df_to_bucket = df_to_bucket.sort_values(['yahoo_position', 'year', 'cost'], ascending=[True, True, True])
    df_to_bucket['cost_bucket'] = (df_to_bucket.groupby(['yahoo_position', 'year'], dropna=False).cumcount() // 3) + 1
    df.loc[mask, 'cost_bucket'] = df_to_bucket['cost_bucket']

    return df


def calculate_draft_value(draft: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate draft value metrics: pick_savings, cost_savings, savings, cost_bucket.

    This function handles the transformation logic that was previously in draft_data_v2.py.

    Logic:
    - pick_savings: avg_pick (or preseason_avg_pick) - actual pick
      - Positive = drafted later than expected (better value)
      - Fallback: estimate from avg_round if avg_pick missing
    - cost_savings: avg_cost (or preseason_avg_cost) - actual cost
      - Positive = cost less than expected (better value)
      - Keeper adjustment: uses is_keeper_cost instead of cost
    - savings: unified metric (cost_savings for auction, pick_savings for snake)
    - cost_bucket: position-based value tier (1-3 = tier 1, 4-6 = tier 2, etc.)

    Args:
        draft: Draft DataFrame with raw Yahoo API data

    Returns:
        Draft DataFrame with value metrics added
    """
    print("  Calculating draft value metrics (pick_savings, cost_savings, savings, cost_bucket)...")

    # Ensure required columns exist
    required_cols = ['year', 'pick', 'cost', 'yahoo_position']
    missing = [c for c in required_cols if c not in draft.columns]
    if missing:
        print(f"  Warning: Missing required columns for value calculation: {missing}")
        draft['pick_savings'] = pd.NA
        draft['cost_savings'] = pd.NA
        draft['savings'] = pd.NA
        draft['cost_bucket'] = pd.NA
        return draft

    # Normalize numerics
    pick        = pd.to_numeric(draft.get("pick"), errors="coerce")
    avg_pick    = pd.to_numeric(draft.get("avg_pick"), errors="coerce")
    avg_round   = pd.to_numeric(draft.get("avg_round"), errors="coerce")
    cost        = pd.to_numeric(draft.get("cost"), errors="coerce")
    avg_cost    = pd.to_numeric(draft.get("avg_cost"), errors="coerce")
    pre_avg_pk  = pd.to_numeric(draft.get("preseason_avg_pick"), errors="coerce")
    pre_avg_cs  = pd.to_numeric(draft.get("preseason_avg_cost"), errors="coerce")
    keeper_cost = pd.to_numeric(draft.get("is_keeper_cost"), errors="coerce")

    # --- PICK SAVINGS (snake draft value) ---
    pick_filled = avg_pick.fillna(pre_avg_pk)

    # Fallback: approximate from avg_round if team count available
    teams_per_year = draft.groupby("year")["team_key"].nunique().rename("teams")
    draft = draft.merge(teams_per_year, on="year", how="left")
    teams = pd.to_numeric(draft["teams"], errors="coerce")
    use_round_mask = pick_filled.isna() & avg_round.notna() & teams.notna() & (teams > 0)
    pick_filled.loc[use_round_mask] = ((avg_round[use_round_mask] - 1) * teams[use_round_mask]) + (teams[use_round_mask] / 2)
    pick_savings = (pick_filled - pick)

    # --- COST SAVINGS (auction draft value) ---
    cost_filled = avg_cost.fillna(pre_avg_cs)
    cost_savings = (cost_filled - cost)

    # Keeper override: for keepers, compare to keeper cost (not draft cost)
    kc_mask = keeper_cost.notna() & cost_filled.notna()
    cost_savings.loc[kc_mask] = cost_filled[kc_mask] - keeper_cost[kc_mask]

    # Store individual metrics
    draft["pick_savings"] = pick_savings
    draft["cost_savings"] = cost_savings

    # --- UNIFIED SAVINGS (based on draft_type) ---
    # Determine draft type from draft_type column or heuristic
    if 'draft_type' in draft.columns:
        # Use draft_type from fetcher (preferred method)
        is_auction_series = (draft['draft_type'].str.lower() == 'auction')
        draft["savings"] = np.where(is_auction_series, cost_savings, pick_savings)
    else:
        # Fallback heuristic: if 25%+ of picks have non-zero cost, treat as auction
        nonzero_cost = cost.fillna(0).gt(0).sum()
        is_auction = nonzero_cost >= max(1, int(len(draft) * 0.25))

        if is_auction:
            draft["savings"] = cost_savings
            print("  Detected auction draft (using cost_savings for unified 'savings' metric)")
        else:
            draft["savings"] = pick_savings
            print("  Detected snake draft (using pick_savings for unified 'savings' metric)")

    # --- COST BUCKETS (position-based value tiers) ---
    draft = assign_cost_buckets(draft)

    # Drop temporary 'teams' column if we added it
    if 'teams' in draft.columns:
        draft = draft.drop(columns=['teams'])

    value_count = draft['savings'].notna().sum()
    print(f"  Calculated value metrics for {value_count:,} draft picks")

    return draft


def add_kept_next_year(draft: pd.DataFrame) -> pd.DataFrame:
    """
    Add kept_next_year column to draft data.

    Logic:
    - kept_next_year = 1 if same yahoo_player_id has is_keeper_status=1 in year+1
    - kept_next_year = 0 otherwise

    Args:
        draft: Draft DataFrame with yahoo_player_id, year, is_keeper_status columns

    Returns:
        Draft DataFrame with kept_next_year column added
    """
    print("  Adding kept_next_year column...")

    # Ensure required columns exist
    required_cols = ['yahoo_player_id', 'year', 'is_keeper_status']
    missing = [c for c in required_cols if c not in draft.columns]
    if missing:
        print(f"  Warning: Missing required columns: {missing}. Setting kept_next_year=0 for all rows.")
        draft['kept_next_year'] = 0
        return draft

    # Create a lookup: player_year_next -> is_keeper_status
    # This tells us which players were keepers in each year
    draft_copy = draft.copy()
    draft_copy['yahoo_player_id'] = draft_copy['yahoo_player_id'].astype(str)
    draft_copy['year'] = draft_copy['year'].astype(int)

    # Create keeper lookup for next year
    # CRITICAL: Use drop_duplicates to prevent duplicate joins
    keeper_lookup = (
        draft_copy[draft_copy['is_keeper_status'] == 1]
        [['yahoo_player_id', 'year']]
        .drop_duplicates()  # Ensure unique (yahoo_player_id, year) combinations
    )

    # For each draft row, check if player was kept in year+1
    draft_copy['year_next'] = draft_copy['year'] + 1
    draft_copy['player_year_next'] = (
        draft_copy['yahoo_player_id'].astype(str) + '_' + draft_copy['year_next'].astype(str)
    )

    # Check if we have any keepers to look up
    if len(keeper_lookup) > 0:
        keeper_lookup['kept_in_this_year'] = 1

        # Create keeper next year lookup
        keeper_lookup['player_year'] = (
            keeper_lookup['yahoo_player_id'].astype(str) + '_' + keeper_lookup['year'].astype(str)
        )
        keeper_next = keeper_lookup[['player_year', 'kept_in_this_year']].rename(
            columns={'player_year': 'player_year_next', 'kept_in_this_year': 'kept_next_year'}
        ).drop_duplicates(subset=['player_year_next'])  # CRITICAL: Ensure unique player_year_next

        # Merge to get kept_next_year (now safe - keeper_next has unique keys)
        draft_copy = draft_copy.merge(
            keeper_next,
            on='player_year_next',
            how='left',
            validate='many_to_one'  # Ensure right side has unique keys
        )

        # Fill NaN with 0 and convert to int
        draft_copy['kept_next_year'] = draft_copy['kept_next_year'].fillna(0).astype(int)
    else:
        # No keepers found - all players were not kept next year
        print("  No keepers found in draft data - setting kept_next_year to 0 for all players")
        draft_copy['kept_next_year'] = 0

    # Drop temporary columns
    draft_copy = draft_copy.drop(columns=['year_next', 'player_year_next'])

    kept_count = (draft_copy['kept_next_year'] == 1).sum()
    print(f"  Found {kept_count:,} players who were kept in the following season")

    return draft_copy


def transform_draft(
    context_path: str,
    dry_run: bool = False,
    make_backup: bool = False
) -> Dict:
    """
    Transform draft data by adding forward-looking keeper columns.

    Args:
        context_path: Path to league_context.json
        dry_run: If True, show changes without writing files
        make_backup: If True, create backup before overwriting

    Returns:
        Dict with statistics about the transformation
    """
    print("\n" + "="*60)
    print("DRAFT ENRICHMENT TRANSFORMATION")
    print("="*60)

    # Load context
    ctx = LeagueContext.load(context_path)
    draft_path = Path(ctx.data_directory) / "draft.parquet"

    if not draft_path.exists():
        print(f"\n[ERROR] Draft file not found: {draft_path}")
        print("Run initial data fetch first to create draft.parquet")
        return {"error": "Draft file not found"}

    # Load draft data
    print(f"\nLoading draft data from {draft_path}...")
    draft = pd.read_parquet(draft_path)
    print(f"  Loaded {len(draft):,} draft records")
    print(f"  Years: {sorted(draft['year'].unique())}")

    # Add kept_next_year column
    print("\nAdding forward-looking keeper columns...")
    draft_enriched = add_kept_next_year(draft)

    # Calculate draft value metrics (refactored from draft_data_v2.py)
    print("\nCalculating draft value metrics...")
    draft_enriched = calculate_draft_value(draft_enriched)

    # Statistics
    stats = {
        "total_draft_records": len(draft_enriched),
        "records_with_kept_next_year": (draft_enriched['kept_next_year'] == 1).sum(),
        "records_with_value_metrics": draft_enriched['savings'].notna().sum(),
        "years_processed": sorted(draft_enriched['year'].unique()),
    }

    print("\n" + "="*60)
    print("TRANSFORMATION STATISTICS")
    print("="*60)
    print(f"Total draft records: {stats['total_draft_records']:,}")
    print(f"Players kept in following season: {stats['records_with_kept_next_year']:,}")
    print(f"Picks with value metrics: {stats['records_with_value_metrics']:,}")
    print(f"Years: {stats['years_processed']}")

    # Show sample
    if stats['records_with_kept_next_year'] > 0:
        print("\nSample of players kept in following season:")
        sample_cols = ['player', 'year', 'manager', 'cost', 'is_keeper_status', 'kept_next_year']
        sample_cols = [c for c in sample_cols if c in draft_enriched.columns]
        sample = draft_enriched[draft_enriched['kept_next_year'] == 1][sample_cols].head(10)
        print(sample.to_string())

    # Save or show dry-run results
    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)
        return stats

    # Backup if requested
    if make_backup and draft_path.exists():
        bpath = backup_file(draft_path)
        print(f"\n[Backup Created] {bpath}")

    # Write enriched draft data
    print(f"\nWriting enriched draft data to {draft_path}...")
    draft_enriched.to_parquet(draft_path, index=False)
    print("  Wrote draft.parquet")

    # Also write CSV for inspection
    csv_path = draft_path.with_suffix('.csv')
    draft_enriched.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path.name}")

    print("\n" + "="*60)
    print("DRAFT ENRICHMENT COMPLETE")
    print("="*60)

    return stats


def main():
    """Main entry point for draft enrichment transformation."""
    parser = argparse.ArgumentParser(
        description="Add draft value metrics and forward-looking keeper columns to draft data"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Path to league_context.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before overwriting"
    )

    args = parser.parse_args()

    try:
        stats = transform_draft(
            context_path=args.context,
            dry_run=args.dry_run,
            make_backup=args.backup
        )

        if "error" in stats:
            sys.exit(1)

        print("\nTransformation completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
