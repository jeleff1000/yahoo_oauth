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
Transactions to Player Import (Multi-League V2)

Adds transaction context (FAAB bids) to player data for keeper value analysis.

This transformation adds transaction-related columns to player data:
- max_faab_bid_to_date: Highest FAAB bid spent on this player up to this week
- first_acquisition_week: Week when player was first acquired (if any)
- total_acquisitions: Total number of times player was acquired this season

Join Key: (yahoo_player_id, year, cumulative_week) - Cumulative by week

Usage:
    python transactions_to_player_v2.py --context path/to/league_context.json
    python transactions_to_player_v2.py --context path/to/league_context.json --dry-run
    python transactions_to_player_v2.py --context path/to/league_context.json --backup
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
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from core.league_context import LeagueContext


# =========================================================
# Configuration
# =========================================================

# Columns to import from transactions to player
TRANSACTION_COLS_TO_IMPORT = [
    "max_faab_bid_to_date",      # Highest FAAB bid up to this week
    "first_acquisition_week",     # Week player was first acquired
    "total_acquisitions",         # Total times acquired this season
    "faab_bid",                   # FAAB bid for this specific transaction (if any)
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


@ensure_normalized
def calculate_cumulative_faab(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative FAAB metrics for each player-year-week.

    For each player, tracks the maximum FAAB bid spent up to each week.
    This is the key metric for keeper price calculations.

    Args:
        transactions_df: Transaction data with columns:
            - yahoo_player_id
            - year
            - cumulative_week (YYYYWW format)
            - faab_bid
            - transaction_type

    Returns:
        DataFrame with cumulative FAAB metrics per player-year-week
    """
    # Filter to only 'add' transactions with FAAB bids
    if 'transaction_type' in transactions_df.columns:
        trans = transactions_df[
            (transactions_df['transaction_type'].str.lower().isin(['add', 'trade'])) &
            (transactions_df['faab_bid'].notna()) &
            (transactions_df['faab_bid'] > 0)
        ].copy()
    else:
        trans = transactions_df[
            (transactions_df['faab_bid'].notna()) &
            (transactions_df['faab_bid'] > 0)
        ].copy()

    if trans.empty:
        return pd.DataFrame(columns=[
            'yahoo_player_id', 'year', 'cumulative_week',
            'max_faab_bid_to_date', 'first_acquisition_week', 'total_acquisitions'
        ])

    # Sort by player, year, week
    trans = trans.sort_values(['yahoo_player_id', 'year', 'cumulative_week'])

    # Group by player-year (with league_id for multi-league isolation)
    group_keys = ['yahoo_player_id', 'year']
    if 'league_id' in trans.columns:
        group_keys = ['league_id'] + group_keys
        trans = trans.sort_values(['league_id', 'yahoo_player_id', 'year', 'cumulative_week'])

    results = []

    for group_vals, group in trans.groupby(group_keys):
        group = group.sort_values('cumulative_week')

        # Unpack group_vals tuple - order matches group_keys
        # group_keys can be ['league_id', 'yahoo_player_id', 'year'] or ['yahoo_player_id', 'year']
        if len(group_keys) == 3:  # Has league_id
            league_id_val, player_id, year = group_vals
        else:  # Only yahoo_player_id and year
            player_id, year = group_vals
            league_id_val = None

        # Get all unique weeks this player had transactions
        all_weeks = group['cumulative_week'].unique()

        # For each week, calculate cumulative max FAAB
        for week in all_weeks:
            # All transactions up to and including this week
            up_to_week = group[group['cumulative_week'] <= week]

            max_faab = up_to_week['faab_bid'].max()
            first_week = up_to_week['cumulative_week'].min()
            num_acquisitions = len(up_to_week)

            result_dict = {
                'yahoo_player_id': player_id,
                'year': year,
                'cumulative_week': week,
                'max_faab_bid_to_date': max_faab,
                'first_acquisition_week': first_week,
                'total_acquisitions': num_acquisitions
            }

            # Add league_id if it was in group_keys
            if league_id_val is not None:
                result_dict['league_id'] = league_id_val

            results.append(result_dict)

    result_df = pd.DataFrame(results)

    # Convert types
    if not result_df.empty:
        result_df['max_faab_bid_to_date'] = pd.to_numeric(result_df['max_faab_bid_to_date'], errors='coerce')
        result_df['first_acquisition_week'] = pd.to_numeric(result_df['first_acquisition_week'], errors='coerce').astype('Int64')
        result_df['total_acquisitions'] = pd.to_numeric(result_df['total_acquisitions'], errors='coerce').astype('Int64')

    return result_df


@ensure_normalized
def merge_transactions_to_player(
    player_df: pd.DataFrame,
    cumulative_faab_df: pd.DataFrame,
    transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge cumulative FAAB data and individual transaction data into player DataFrame.

    Strategy:
    1. Merge cumulative FAAB metrics (forward-filled through season)
    2. Merge individual transaction faab_bid (only for the exact week of transaction)

    Args:
        player_df: Player data (weekly granularity)
        cumulative_faab_df: Cumulative FAAB data from calculate_cumulative_faab()
        transactions_df: Original transactions data with faab_bid

    Returns:
        Player DataFrame with FAAB columns added
    """
    if cumulative_faab_df.empty:
        # No transactions - add empty columns
        for col in TRANSACTION_COLS_TO_IMPORT:
            if col not in player_df.columns:
                if col in ['max_faab_bid_to_date', 'faab_bid']:
                    player_df[col] = 0.0
                else:
                    player_df[col] = pd.NA
        return player_df

    # Merge on exact week match first (with multi-league isolation)
    # Try with league_id first, fallback without for backward compatibility
    merge_keys = ['yahoo_player_id', 'year', 'cumulative_week']
    if 'league_id' in player_df.columns and 'league_id' in cumulative_faab_df.columns:
        merge_keys = ['league_id'] + merge_keys

    # CRITICAL: Normalize data types before merge to prevent type mismatch errors
    # yahoo_player_id can be int64 in one dataframe but object/string in another
    # cumulative_week can be float64 (201401.0) in player but Int64 (201401) in transactions
    # Convert all merge keys: numeric -> int -> string for consistent merging
    print("  Normalizing data types for merge keys...")
    for key in merge_keys:
        for df in [player_df, cumulative_faab_df, transactions_df]:
            if key in df.columns:
                # Convert to numeric first (handles strings, floats, ints)
                df[key] = pd.to_numeric(df[key], errors='coerce')
                # Convert to Int64 (pandas nullable int - handles NA)
                df[key] = df[key].astype('Int64')
                # Finally convert to string for merge (avoids type mismatches)
                df[key] = df[key].astype(str)

    player_with_trans = player_df.merge(
        cumulative_faab_df,
        on=merge_keys,
        how='left',
        suffixes=('', '_trans')
    )

    # Also merge individual transaction faab_bid (only for exact week match)
    if 'faab_bid' in transactions_df.columns:
        trans_faab = transactions_df[merge_keys + ['faab_bid']].copy()
        # Keep only the highest FAAB bid if multiple transactions in same week
        trans_faab = trans_faab.groupby(merge_keys, as_index=False)['faab_bid'].max()

        # Drop old faab_bid columns from previous runs to prevent duplicate column errors
        player_with_trans = player_with_trans.drop(columns=['faab_bid', 'faab_bid_old'], errors='ignore')

        player_with_trans = player_with_trans.merge(
            trans_faab,
            on=merge_keys,
            how='left',
            suffixes=('_old', '')
        )

        # Fill NaN faab_bid with 0
        if 'faab_bid' in player_with_trans.columns:
            player_with_trans['faab_bid'] = player_with_trans['faab_bid'].fillna(0.0)

    # Forward-fill FAAB values within each player-year
    # If player was acquired in week 5 for $20, weeks 6+ should also show $20
    player_with_trans = player_with_trans.sort_values(['yahoo_player_id', 'year', 'cumulative_week'])

    for col in ['max_faab_bid_to_date', 'first_acquisition_week', 'total_acquisitions']:
        if col in player_with_trans.columns:
            player_with_trans[col] = (
                player_with_trans.groupby(['yahoo_player_id', 'year'])[col]
                .ffill()
            )

    # Fill remaining NaN with 0 for max_faab_bid_to_date
    if 'max_faab_bid_to_date' in player_with_trans.columns:
        player_with_trans['max_faab_bid_to_date'] = (
            player_with_trans['max_faab_bid_to_date'].fillna(0.0)
        )

    return player_with_trans


# =========================================================
# Main Import Function
# =========================================================

def import_transactions_to_player(
    player_path: Path,
    transactions_path: Path,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Import transaction FAAB data into player data.

    Args:
        player_path: Path to player.parquet
        transactions_path: Path to transactions.parquet
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics about the import
    """
    print(f"Loading player data from: {player_path}")
    player_original = pd.read_parquet(player_path)
    print(f"  Loaded {len(player_original):,} player records")

    # Create filtered copy for enrichment (excludes historical NFL data)
    # We'll merge enrichment back into original to preserve all rows
    if 'yahoo_player_id' in player_original.columns:
        player_for_enrichment = player_original[
            (player_original['yahoo_player_id'].notna()) &
            (player_original['yahoo_player_id'] != 'None') &
            (player_original['yahoo_player_id'] != '')
        ].copy()
        filtered_count = len(player_original) - len(player_for_enrichment)
        if filtered_count > 0:
            print(f"  Working with {len(player_for_enrichment):,} records for enrichment ({filtered_count:,} historical rows preserved)")
    else:
        player_for_enrichment = player_original.copy()

    player = player_for_enrichment  # Use filtered version for enrichment logic

    print(f"\nLoading transaction data from: {transactions_path}")
    transactions = pd.read_parquet(transactions_path)
    print(f"  Loaded {len(transactions):,} transaction records")

    # Filter out any invalid transactions (shouldn't exist after v2 regeneration)
    if 'yahoo_player_id' in transactions.columns:
        initial_trans = len(transactions)
        transactions = transactions[
            (transactions['yahoo_player_id'].notna()) &
            (transactions['yahoo_player_id'] != 'None') &
            (transactions['yahoo_player_id'] != '')
        ].copy()
        filtered_trans = initial_trans - len(transactions)
        if filtered_trans > 0:
            print(f"  Filtered out {filtered_trans:,} invalid transaction rows")
            print(f"  Working with {len(transactions):,} transaction records for enrichment")

    # Validate required columns
    required_player_cols = ['yahoo_player_id', 'year', 'cumulative_week']
    for col in required_player_cols:
        if col not in player.columns:
            raise KeyError(f"player.parquet missing required column: {col}")

    required_trans_cols = ['yahoo_player_id', 'year', 'cumulative_week']
    for col in required_trans_cols:
        if col not in transactions.columns:
            raise KeyError(f"transactions.parquet missing required column: {col}")

    # Check for faab_bid column
    if 'faab_bid' not in transactions.columns:
        print("[WARNING] No faab_bid column in transactions, will add zero values")
        transactions['faab_bid'] = 0.0

    # Calculate cumulative FAAB metrics
    print("\nCalculating cumulative FAAB metrics...")
    cumulative_faab = calculate_cumulative_faab(transactions)
    print(f"  Calculated FAAB data for {len(cumulative_faab):,} player-year-week records")

    if not cumulative_faab.empty:
        print(f"  Players with FAAB acquisitions: {cumulative_faab['yahoo_player_id'].nunique():,}")
        print(f"  Max FAAB bid in data: ${cumulative_faab['max_faab_bid_to_date'].max():.2f}")
        print(f"  Avg FAAB bid: ${cumulative_faab['max_faab_bid_to_date'].mean():.2f}")

    # Merge into player data
    print("\nMerging transaction data into player data...")
    player_before_cols = set(player.columns)
    player_enriched = merge_transactions_to_player(player, cumulative_faab, transactions)
    new_cols = set(player_enriched.columns) - player_before_cols

    print(f"  Added {len(new_cols)} new columns: {sorted(new_cols)}")

    # Count how many player records got FAAB data
    if 'max_faab_bid_to_date' in player_enriched.columns:
        records_with_faab = (player_enriched['max_faab_bid_to_date'] > 0).sum()
        print(f"  Player-week records with cumulative FAAB > 0: {records_with_faab:,} ({records_with_faab/len(player_enriched)*100:.1f}%)")

    if 'faab_bid' in player_enriched.columns:
        records_with_transaction = (player_enriched['faab_bid'] > 0).sum()
        print(f"  Player-week records with transaction FAAB > 0: {records_with_transaction:,} ({records_with_transaction/len(player_enriched)*100:.1f}%)")

    # Merge enriched columns back into original player data to preserve historical rows
    print(f"\nMerging enrichment back into original data to preserve historical rows...")
    if len(player_original) > len(player_enriched):
        # Identify rows that were filtered out (historical data)
        enriched_keys = set(zip(player_enriched['yahoo_player_id'], player_enriched['year'], player_enriched['cumulative_week']))
        original_keys = set(zip(player_original['yahoo_player_id'], player_original['year'], player_original['cumulative_week']))
        historical_keys = original_keys - enriched_keys

        print(f"  Enriched rows: {len(player_enriched):,}")
        print(f"  Historical rows to preserve: {len(historical_keys):,}")

        # Add new columns to original data (initialize with NA/0)
        for col in new_cols:
            if col not in player_original.columns:
                if col in ['max_faab_bid_to_date', 'faab_bid']:
                    player_original[col] = 0.0
                else:
                    player_original[col] = pd.NA

        # Update enriched rows (left join keeps all original rows, updates where match exists)
        merge_keys = ['yahoo_player_id', 'year', 'cumulative_week']
        if 'league_id' in player_original.columns and 'league_id' in player_enriched.columns:
            merge_keys = ['league_id'] + merge_keys

        # CRITICAL: Normalize merge key data types in original to match enriched (both converted to string earlier)
        print(f"  Normalizing merge key data types for final merge...")

        # Drop new columns first to get base dataframe
        player_base = player_original.drop(columns=list(new_cols), errors='ignore').copy()

        # Now normalize merge keys in BOTH dataframes (numeric -> Int64 -> string)
        for key in merge_keys:
            if key in player_base.columns:
                # Same normalization as enrichment: numeric -> int -> string
                player_base[key] = pd.to_numeric(player_base[key], errors='coerce').astype('Int64').astype(str)

        # Select only new columns from enriched data
        enriched_subset = player_enriched[merge_keys + list(new_cols)].copy()

        # CRITICAL: Also normalize enriched_subset merge keys to ensure type consistency
        for key in merge_keys:
            if key in enriched_subset.columns:
                enriched_subset[key] = pd.to_numeric(enriched_subset[key], errors='coerce').astype('Int64').astype(str)

        # Merge: keep all original rows, update with enriched values where they exist
        player_updated = player_base.merge(
            enriched_subset,
            on=merge_keys,
            how='left'
        )

        print(f"  Final player data: {len(player_updated):,} rows (all historical data preserved)")
    else:
        # No historical rows to preserve
        player_updated = player_enriched

    # Statistics
    stats = {
        'player_records_before': len(player_original),
        'player_records_after': len(player_updated),
        'transaction_records': len(transactions),
        'cumulative_faab_records': len(cumulative_faab),
        'new_columns_added': len(new_cols),
    }

    # Write updated player data
    if not dry_run:
        if make_backup:
            backup_path = backup_file(player_path)
            print(f"\n[BACKUP] Created backup: {backup_path}")

        player_updated.to_parquet(player_path, index=False, engine='pyarrow')
        print(f"\n[SAVE] Wrote updated player data to: {player_path}")
        print(f"        Total rows: {len(player_updated):,} (includes {len(player_updated) - len(player_enriched):,} historical rows)")
    else:
        print("\n[DRY-RUN] Skipping write (use without --dry-run to save changes)")

    return stats


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Add transaction FAAB data to player.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using league context
  python transactions_to_player_v2.py --context leagues/kmffl/league_context.json

  # Dry run (preview changes)
  python transactions_to_player_v2.py --context leagues/kmffl/league_context.json --dry-run

  # With backup
  python transactions_to_player_v2.py --context leagues/kmffl/league_context.json --backup
        """
    )

    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Path to league_context.json"
    )
    parser.add_argument(
        "--player",
        type=str,
        help="Override player data path"
    )
    parser.add_argument(
        "--transactions",
        type=str,
        help="Override transactions data path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before writing"
    )

    args = parser.parse_args()

    # Load context
    ctx = LeagueContext.load(args.context)
    print(f"Loaded context for league: {ctx.league_name} ({ctx.league_id})")

    # Determine paths
    player_path = Path(args.player) if args.player else ctx.canonical_player_file
    transactions_path = Path(args.transactions) if args.transactions else ctx.canonical_transaction_file

    # Validate paths
    if not player_path.exists():
        raise FileNotFoundError(f"Player data not found: {player_path}")
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transaction data not found: {transactions_path}")

    # Run import
    print(f"\n{'='*80}")
    print("TRANSACTIONS -> PLAYER IMPORT")
    print(f"{'='*80}\n")

    stats = import_transactions_to_player(
        player_path=player_path,
        transactions_path=transactions_path,
        dry_run=args.dry_run,
        make_backup=args.backup,
    )

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
