"""
Fix Unknown Manager Names in Transactions

This transformation fixes transactions where manager = "Unknown" by finding
the most recent "add" transaction for that player_key and using that manager.

This typically happens in add/drop combos where the API doesn't properly
identify the manager for one side of the transaction.

Usage:
    python fix_unknown_managers.py --context path/to/league_context.json
    python fix_unknown_managers.py --context path/to/league_context.json --dry-run
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()
_transformations_dir = _script_file.parent.parent
_multi_league_dir = _transformations_dir.parent
sys.path.insert(0, str(_multi_league_dir.parent))
sys.path.insert(0, str(_multi_league_dir))

from core.league_context import LeagueContext


def fix_unknown_managers(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix transactions where manager = "Unknown" by backfilling from most recent add.

    For each transaction with manager="Unknown", finds the most recent "add"
    transaction for that player_key and uses that manager.

    Args:
        transactions_df: DataFrame with transaction data

    Returns:
        DataFrame with Unknown managers fixed
    """
    if transactions_df.empty:
        return transactions_df

    df = transactions_df.copy()

    # Find rows with Unknown manager
    unknown_mask = df['manager'] == 'Unknown'
    unknown_count = unknown_mask.sum()

    if unknown_count == 0:
        print("  No Unknown managers found")
        return df

    print(f"  Found {unknown_count:,} transactions with Unknown manager")

    # Sort by timestamp to process chronologically
    df['_timestamp_numeric'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.sort_values(['player_key', '_timestamp_numeric'])

    # For each player_key, find the most recent "add" transaction with a known manager
    fixes_made = 0

    for player_key in df[unknown_mask]['player_key'].unique():
        if pd.isna(player_key) or player_key == '':
            continue

        # Get all transactions for this player
        player_txns = df[df['player_key'] == player_key].copy()

        # Find all "add" transactions with known managers
        known_adds = player_txns[
            (player_txns['transaction_type'] == 'add') &
            (player_txns['manager'] != 'Unknown') &
            (player_txns['manager'].notna())
        ]

        if known_adds.empty:
            # No known adds for this player - can't fix
            continue

        # Get all Unknown transactions for this player
        unknown_txns = player_txns[player_txns['manager'] == 'Unknown']

        for idx in unknown_txns.index:
            txn_timestamp = df.at[idx, '_timestamp_numeric']

            # Find the most recent add BEFORE this transaction
            prior_adds = known_adds[known_adds['_timestamp_numeric'] <= txn_timestamp]

            if not prior_adds.empty:
                # Use manager from most recent prior add
                most_recent_add = prior_adds.sort_values('_timestamp_numeric').iloc[-1]
                manager_to_use = most_recent_add['manager']
            else:
                # No prior adds - use the earliest add after this transaction
                future_adds = known_adds[known_adds['_timestamp_numeric'] > txn_timestamp]
                if not future_adds.empty:
                    earliest_add = future_adds.sort_values('_timestamp_numeric').iloc[0]
                    manager_to_use = earliest_add['manager']
                else:
                    continue

            df.at[idx, 'manager'] = manager_to_use
            fixes_made += 1

    # Drop temporary column
    df = df.drop(columns=['_timestamp_numeric'])

    print(f"  Fixed {fixes_made:,} Unknown managers")

    return df


def main(args):
    """Main entry point for fixing unknown managers."""
    print("\n" + "="*80)
    print("FIX UNKNOWN MANAGERS IN TRANSACTIONS")
    print("="*80 + "\n")

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] Data directory: {ctx.data_directory}")

    # Check for transactions file
    transactions_file = ctx.canonical_transaction_file
    if not transactions_file.exists():
        raise FileNotFoundError(f"[ERROR] Transactions file not found: {transactions_file}")

    # Dry run check
    if args.dry_run:
        print("\n[DRY RUN] No changes will be made")
        print(f"   Would read from: {transactions_file}")
        print(f"   Would write to: {transactions_file}")
        return

    # Load transactions
    print(f"\n[Loading] Transactions data from {transactions_file.name}...")
    transactions_df = pd.read_parquet(transactions_file)
    print(f"   Loaded {len(transactions_df):,} transactions")

    # Fix Unknown managers
    print("\n[Fixing] Unknown managers...")
    transactions_df = fix_unknown_managers(transactions_df)

    # Save updated transactions file (both parquet and CSV)
    print(f"\n[Saving] Writing updated transactions data to {transactions_file}...")
    transactions_df.to_parquet(transactions_file, index=False)
    print(f"   [OK] Saved {len(transactions_df):,} rows to parquet")

    # Also save CSV version
    csv_file = transactions_file.with_suffix('.csv')
    transactions_df.to_csv(csv_file, index=False)
    print(f"   [OK] Saved {len(transactions_df):,} rows to CSV: {csv_file.name}")

    print("\n" + "="*80)
    print("[SUCCESS] UNKNOWN MANAGER FIX COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fix Unknown managers in transactions'
    )
    parser.add_argument(
        '--context',
        required=True,
        help='Path to league_context.json'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()
    main(args)
