"""
Keeper Economics Transformation (Multi-League V2)

Calculates keeper prices and adds keeper economics to player data.

This transformation combines draft and transaction data to calculate:
- Keeper price (based on draft cost + FAAB bids)
- Whether player was kept next year
- Next year cost and points (ROI analysis)

Join Key: (yahoo_player_id, year) - Season-level aggregation

Formula: keeper_price = max(draft_cost * 1.5 + 7.5, faab_bid / 2, 1)

Usage:
    python keeper_economics_v2.py --context path/to/league_context.json
    python keeper_economics_v2.py --context path/to/league_context.json --dry-run
    python keeper_economics_v2.py --context path/to/league_context.json --backup
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

from multi_league.core.league_context import LeagueContext


# =========================================================
# Helper Functions
# =========================================================

def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


def increment_year(yahoo_player_id: str, year: int) -> str:
    """Create next year's player_year key."""
    return f"{yahoo_player_id}_{year + 1}"


def calculate_keeper_price(cost: float, faab_bid: float, is_keeper: bool) -> int:
    """
    Calculate keeper price using league formula.

    Formula:
    - If player was a keeper: base = cost * 1.5 + 7.5
    - Otherwise: base = cost
    - Final = max(base, faab_bid / 2, 1)

    Args:
        cost: Draft cost (auction value or positional value)
        faab_bid: Maximum FAAB bid spent on this player
        is_keeper: Whether player was kept (not freshly drafted)

    Returns:
        Keeper price (integer, minimum 1)
    """
    cost = float(cost) if not pd.isna(cost) else 0.0
    faab_bid = float(faab_bid) if not pd.isna(faab_bid) else 0.0
    is_keeper = bool(is_keeper) if not pd.isna(is_keeper) else False

    # Base calculation
    if is_keeper:
        base_price = cost * 1.5 + 7.5
    else:
        base_price = cost

    # Consider FAAB (half of highest bid)
    half_faab = faab_bid / 2.0

    # Take maximum, minimum of 1
    final_price = max(base_price, half_faab, 1.0)

    return int(np.floor(final_price + 0.5))


# =========================================================
# Main Transformation Function
# =========================================================

def calculate_keeper_economics(
    player_path: Path,
    draft_path: Path,
    transactions_path: Path,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Calculate keeper economics and add to player data.

    Args:
        player_path: Path to player.parquet
        draft_path: Path to draft.parquet
        transactions_path: Path to transactions.parquet
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics
    """
    print(f"Loading player data from: {player_path}")
    player = pd.read_parquet(player_path)
    print(f"  Loaded {len(player):,} player records")

    # Check if player already has cost and max_faab_bid columns
    has_cost = 'cost' in player.columns
    has_faab = 'max_faab_bid' in player.columns
    has_keeper_status = 'is_keeper_status' in player.columns

    if has_cost and has_faab and has_keeper_status:
        print(f"\n[FAST PATH] Player data already has cost, max_faab_bid, and is_keeper_status")
        print(f"  Calculating keeper_price directly from existing columns...")

        # Calculate keeper_price directly from existing columns
        player['cost'] = pd.to_numeric(player['cost'], errors='coerce').fillna(0.0)
        player['max_faab_bid'] = pd.to_numeric(player['max_faab_bid'], errors='coerce').fillna(0.0)
        player['is_keeper_status'] = pd.to_numeric(player['is_keeper_status'], errors='coerce').fillna(0).astype(int)

        # Apply keeper price formula to each row
        def calc_keeper_price_row(row):
            cost = float(row['cost']) if not pd.isna(row['cost']) else 0.0
            faab = float(row['max_faab_bid']) if not pd.isna(row['max_faab_bid']) else 0.0
            is_keeper = bool(row['is_keeper_status']) if not pd.isna(row['is_keeper_status']) else False

            # Base calculation
            if is_keeper:
                base_price = cost * 1.5 + 7.5
            else:
                base_price = cost

            # Consider FAAB (half of highest bid)
            half_faab = faab / 2.0

            # Take maximum, minimum of 1
            final_price = max(base_price, half_faab, 1.0)
            return int(np.floor(final_price + 0.5))

        player['keeper_price'] = player.apply(calc_keeper_price_row, axis=1)

        rows_with_keeper_price = (player['keeper_price'] > 0).sum()
        print(f"  Calculated keeper_price for {rows_with_keeper_price:,} player rows")

        # Calculate statistics
        stats = {
            "total_player_rows": len(player),
            "players_with_keeper_data": player['keeper_price'].notna().sum(),
            "players_kept_next_year": 0,  # Not calculated in fast path
        }

        # Save (unless dry-run)
        if dry_run:
            print("\n" + "="*60)
            print("[DRY RUN] No files were written.")
            print("="*60)

            print("\nSample keeper prices:")
            sample_cols = ['player', 'year', 'cost', 'is_keeper_status', 'max_faab_bid', 'keeper_price']
            sample_cols = [c for c in sample_cols if c in player.columns]
            print(player[player['keeper_price'] > 0][sample_cols].head(20).to_string())

            return stats

        if make_backup and player_path.exists():
            bpath = backup_file(player_path)
            print(f"\n[Backup Created] {bpath}")

        # Write back to player.parquet
        player.to_parquet(player_path, index=False)
        print(f"\n[SAVED] Updated player data written to: {player_path}")

        return stats

    # SLOW PATH: Load draft and transactions to build keeper economics from scratch
    print(f"\n[SLOW PATH] Building keeper economics from draft and transactions...")
    print(f"\nLoading draft data from: {draft_path}")
    draft = pd.read_parquet(draft_path)
    print(f"  Loaded {len(draft):,} draft records")

    print(f"\nLoading transaction data from: {transactions_path}")
    transactions = pd.read_parquet(transactions_path)
    print(f"  Loaded {len(transactions):,} transaction records")

    # Validate required columns
    if 'yahoo_player_id' not in draft.columns or 'year' not in draft.columns:
        raise KeyError("draft.parquet missing yahoo_player_id or year")
    if 'yahoo_player_id' not in player.columns or 'year' not in player.columns:
        raise KeyError("player.parquet missing yahoo_player_id or year")

    print(f"\nCalculating keeper prices...")

    # =========================================================
    # Step 1: Get draft cost and keeper status per player-year
    # =========================================================

    # Extract cost column (try multiple names)
    cost_col = None
    for col in ['cost', 'draft_cost', 'auction_cost']:
        if col in draft.columns:
            cost_col = col
            break

    if cost_col is None:
        print("[WARNING] No cost column found in draft, using 0")
        draft['cost'] = 0.0
        cost_col = 'cost'

    draft_base = draft[['yahoo_player_id', 'year', cost_col, 'is_keeper_status']].copy()
    draft_base.rename(columns={cost_col: 'cost'}, inplace=True)
    draft_base['cost'] = pd.to_numeric(draft_base['cost'], errors='coerce').fillna(0.0)
    draft_base['is_keeper_status'] = pd.to_numeric(draft_base['is_keeper_status'], errors='coerce').fillna(0).astype(int)

    # =========================================================
    # Step 2: Get maximum FAAB bid per player-year from transactions
    # =========================================================

    faab_col = 'faab_bid' if 'faab_bid' in transactions.columns else None

    if faab_col:
        trans_faab = transactions[
            transactions['yahoo_player_id'].notna() &
            transactions['year'].notna() &
            transactions[faab_col].notna()
        ].copy()

        trans_faab_agg = (
            trans_faab.groupby(['yahoo_player_id', 'year'], as_index=False)[faab_col]
            .max()
            .rename(columns={faab_col: 'max_faab_bid'})
        )
    else:
        print("[WARNING] No faab_bid column in transactions, using 0")
        trans_faab_agg = pd.DataFrame(columns=['yahoo_player_id', 'year', 'max_faab_bid'])

    # =========================================================
    # Step 3: Merge draft + FAAB and calculate keeper price
    # =========================================================

    keeper_base = draft_base.merge(
        trans_faab_agg,
        on=['yahoo_player_id', 'year'],
        how='left'
    )
    keeper_base['max_faab_bid'] = keeper_base['max_faab_bid'].fillna(0.0)

    # Calculate keeper price
    keeper_base['keeper_price'] = keeper_base.apply(
        lambda r: calculate_keeper_price(r['cost'], r['max_faab_bid'], r['is_keeper_status']),
        axis=1
    )

    print(f"  Calculated keeper prices for {len(keeper_base):,} player-years")

    # =========================================================
    # Step 4: Look up next year's keeper status and performance
    # =========================================================

    # Create next year lookup key
    keeper_base['player_year_next'] = keeper_base.apply(
        lambda r: increment_year(r['yahoo_player_id'], r['year']),
        axis=1
    )

    # Get next year's keeper status from draft
    draft_next = draft[['yahoo_player_id', 'year', 'is_keeper_status']].copy()
    draft_next['yahoo_player_id'] = draft_next['yahoo_player_id'].astype(str)
    draft_next['year'] = draft_next['year'].astype(int)
    draft_next['player_year_next'] = draft_next.apply(
        lambda r: f"{r['yahoo_player_id']}_{r['year']}",
        axis=1
    )
    draft_next = draft_next[['player_year_next', 'is_keeper_status']].rename(
        columns={'is_keeper_status': 'kept_next_year'}
    )

    keeper_base = keeper_base.merge(draft_next, on='player_year_next', how='left')
    keeper_base['kept_next_year'] = pd.to_numeric(keeper_base['kept_next_year'], errors='coerce').fillna(0).astype(int)

    # Get next year's total points from player data
    # Aggregate player to season level first
    # Use NFL_player_id to only count games where player actually played (not BN/IR weeks)
    if 'fantasy_points' in player.columns and 'NFL_player_id' in player.columns:
        # Filter to only rows where player actually played (has NFL_player_id)
        # This excludes bench/IR weeks where yahoo tracks the player but they didn't play
        player_season = (
            player[player['NFL_player_id'].notna()]
            .groupby(['yahoo_player_id', 'year'], as_index=False)
            .agg({
                'fantasy_points': ['sum', 'mean', 'count']
            })
        )

        # Flatten column names
        player_season.columns = ['yahoo_player_id', 'year', 'season_total_points', 'avg_points_per_game', 'games_played']

        player_season['yahoo_player_id'] = player_season['yahoo_player_id'].astype(str)
        player_season['year'] = player_season['year'].astype(int)
        player_season['player_year'] = player_season.apply(
            lambda r: f"{r['yahoo_player_id']}_{r['year']}",
            axis=1
        )

        player_season_next = player_season[['player_year', 'season_total_points', 'avg_points_per_game', 'games_played']].rename(
            columns={
                'season_total_points': 'total_points_next_year',
                'avg_points_per_game': 'avg_points_next_year',
                'games_played': 'games_played_next_year'
            }
        )

        keeper_base = keeper_base.merge(
            player_season_next.rename(columns={'player_year': 'player_year_next'}),
            on='player_year_next',
            how='left'
        )
        keeper_base['total_points_next_year'] = keeper_base['total_points_next_year'].fillna(0.0)
        keeper_base['avg_points_next_year'] = keeper_base['avg_points_next_year'].fillna(0.0)
        keeper_base['games_played_next_year'] = keeper_base['games_played_next_year'].fillna(0).astype(int)
    else:
        keeper_base['total_points_next_year'] = 0.0
        keeper_base['avg_points_next_year'] = 0.0
        keeper_base['games_played_next_year'] = 0

    # =========================================================
    # Step 5: Merge keeper economics into player data
    # =========================================================

    print(f"\nMerging keeper economics into player data...")

    keeper_cols = [
        'yahoo_player_id',
        'year',
        'cost',
        'is_keeper_status',
        'max_faab_bid',
        'keeper_price',
        'kept_next_year',
        'total_points_next_year',
        'avg_points_next_year',
        'games_played_next_year'
    ]

    keeper_export = keeper_base[keeper_cols].copy()

    # CRITICAL: Ensure keeper_export has no duplicate keys before merge
    # Draft should be one row per player-year (season-level data)
    duplicates_in_keeper = keeper_export.duplicated(subset=['yahoo_player_id', 'year'], keep=False).sum()
    if duplicates_in_keeper > 0:
        print(f"\n  WARNING: Found {duplicates_in_keeper} duplicate player-year combinations in keeper data")
        print(f"  Removing duplicates (keeping first occurrence)...")
        keeper_export = keeper_export.drop_duplicates(subset=['yahoo_player_id', 'year'], keep='first')
        print(f"  Deduplicated to {len(keeper_export):,} unique player-year combinations")

    # Drop existing keeper columns from player to avoid conflicts
    cols_to_drop = [c for c in keeper_cols if c in player.columns and c not in ['yahoo_player_id', 'year']]
    if cols_to_drop:
        print(f"  Dropping existing columns: {cols_to_drop}")
        player.drop(columns=cols_to_drop, inplace=True)

    # Normalize yahoo_player_id and year types to match before merge
    if 'yahoo_player_id' in player.columns:
        player['yahoo_player_id'] = player['yahoo_player_id'].astype(str)
    if 'yahoo_player_id' in keeper_export.columns:
        keeper_export['yahoo_player_id'] = keeper_export['yahoo_player_id'].astype(str)
    if 'year' in player.columns:
        player['year'] = pd.to_numeric(player['year'], errors='coerce').astype('Int64')
    if 'year' in keeper_export.columns:
        keeper_export['year'] = pd.to_numeric(keeper_export['year'], errors='coerce').astype('Int64')

    # Debug: Check how many player rows should match
    player_years = set(zip(player['yahoo_player_id'].astype(str), player['year'].astype(str)))
    keeper_years = set(zip(keeper_export['yahoo_player_id'].astype(str), keeper_export['year'].astype(str)))
    matching_years = player_years.intersection(keeper_years)
    print(f"  Debug: {len(player_years)} unique player-years in player data")
    print(f"  Debug: {len(keeper_years)} unique player-years in keeper data")
    print(f"  Debug: {len(matching_years)} player-years that will match in merge")

    # Merge
    before_len = len(player)
    player = player.merge(
        keeper_export,
        on=['yahoo_player_id', 'year'],
        how='left',
        validate='many_to_one'  # Player data is weekly (many), keeper is season (one)
    )
    after_len = len(player)

    if after_len != before_len:
        print(f"  WARNING: Row count changed from {before_len:,} to {after_len:,} after merge!")
        print(f"  This indicates duplicate keys in keeper_export or data issue")

    # Check how many rows got keeper_price
    rows_with_keeper_price = player['keeper_price'].notna().sum()
    print(f"  Result: {rows_with_keeper_price:,} player rows now have keeper_price")

    print(f"  Merged keeper economics to {len(player):,} player records")

    # Calculate statistics
    stats = {
        "total_player_rows": len(player),
        "players_with_keeper_data": player['keeper_price'].notna().sum(),
        "players_kept_next_year": player['kept_next_year'].sum() if 'kept_next_year' in player.columns else 0,
    }

    # Save (unless dry-run)
    if dry_run:
        print("\n" + "="*60)
        print("[DRY RUN] No files were written.")
        print("="*60)

        print("\nSample keeper economics:")
        sample_cols = ['player', 'year', 'cost', 'is_keeper_status', 'keeper_price',
                       'kept_next_year', 'total_points_next_year', 'avg_points_next_year', 'games_played_next_year']
        sample_cols = [c for c in sample_cols if c in player.columns]
        print(player[player['keeper_price'].notna()][sample_cols].head(10).to_string())

        return stats

    if make_backup and player_path.exists():
        bpath = backup_file(player_path)
        print(f"\n[Backup Created] {bpath}")

    # Write back to player.parquet
    player.to_parquet(player_path, index=False)
    print(f"\n[SAVED] Updated player data written to: {player_path}")

    return stats


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate keeper economics for multi-league setup"
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
        "--transactions",
        type=str,
        help="Override transactions data path"
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
        transactions_path = Path(args.transactions) if args.transactions else ctx.canonical_transaction_file
    else:
        # Fallback to hardcoded relative paths (V1 compatibility)
        THIS_FILE = Path(__file__).resolve()
        SCRIPT_DIR = THIS_FILE.parent.parent.parent
        ROOT_DIR = SCRIPT_DIR.parent
        DATA_DIR = ROOT_DIR / "fantasy_football_data"

        player_path = Path(args.player) if args.player else (DATA_DIR / "player.parquet")
        draft_path = Path(args.draft) if args.draft else (DATA_DIR / "draft.parquet")
        transactions_path = Path(args.transactions) if args.transactions else (DATA_DIR / "transactions.parquet")

    # Validate paths
    if not player_path.exists():
        raise FileNotFoundError(f"Player data not found: {player_path}")
    if not draft_path.exists():
        raise FileNotFoundError(f"Draft data not found: {draft_path}")
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transaction data not found: {transactions_path}")

    # Run calculation
    print("\n" + "="*60)
    print("KEEPER ECONOMICS CALCULATION (V2)")
    print("="*60)

    stats = calculate_keeper_economics(
        player_path=player_path,
        draft_path=draft_path,
        transactions_path=transactions_path,
        dry_run=args.dry_run,
        make_backup=args.backup
    )

    # Print final summary
    print("\n" + "="*60)
    print("CALCULATION SUMMARY")
    print("="*60)
    print(f"Total player rows:        {stats['total_player_rows']:,}")
    print(f"With keeper data:         {stats['players_with_keeper_data']:,}")
    print(f"Kept next year:           {stats['players_kept_next_year']:,}")
    print("="*60)


if __name__ == "__main__":
    main()
