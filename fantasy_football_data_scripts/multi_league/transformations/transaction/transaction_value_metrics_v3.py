"""
Transaction Value Metrics V3 (Multi-League)

SPAR-based transaction value analysis for pickups, drops, and trades.

This transformation adds:
1. Rest-of-Season SPAR Metrics:
   - replacement_ppg_ros: Window-based replacement baseline (Week W+1 → 17)
   - fa_spar_ros: Rest-of-Season Points Above Replacement
   - fa_ppg_ros: Rest-of-season PPG
   - fa_pgvor_ros: Rest-of-season per-game VOR
   - waiver_cost_norm: Normalized waiver cost (FAAB $ or Priority-equivalent $)
   - fa_roi: Return on Investment (fa_spar_ros / waiver_cost_norm)

Key Improvements over V2:
- Uses dynamic window-based replacement levels (not full season)
- Transaction value = what you GET after the transaction (not sunk cost)
- Comparable metrics across FAAB and Priority waiver systems
- SPAR-based ROI instead of raw points/cost

Metrics by Transaction Type:
- Add/Pickup: Positive fa_spar_ros = value acquired
- Drop: Negative fa_spar_ros = opportunity cost (value lost)
- Trade: Sum of fa_spar_ros for acquired minus dropped players

Usage:
    python transaction_value_metrics_v3.py --context path/to/league_context.json
    python transaction_value_metrics_v3.py --context path/to/league_context.json --dry-run
    python transaction_value_metrics_v3.py --context path/to/league_context.json --backup
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect directory structure
if _script_file.parent.name == 'modules':
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transaction_enrichment':
    _transformations_dir = _script_file.parent.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))

from core.league_context import LeagueContext
from multi_league.transformations.transaction.modules.transaction_spar_calculator import (
    calculate_all_transaction_metrics
)


def backup_file(file_path: Path) -> Path:
    """Create a timestamped backup of a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_name(f"{file_path.stem}_backup_{timestamp}{file_path.suffix}")

    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
        df.to_parquet(backup_path, index=False)
    else:
        import shutil
        shutil.copy2(file_path, backup_path)

    print(f"[OK] Backup created: {backup_path}")
    return backup_path


def main(args):
    """Main entry point for transaction value metrics calculation."""
    print("\n" + "="*80)
    print("TRANSACTION VALUE METRICS V3 - SPAR-BASED ANALYSIS")
    print("="*80 + "\n")

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] Data directory: {ctx.data_directory}")

    # Check for required files
    transactions_file = ctx.canonical_transaction_file
    if not transactions_file.exists():
        raise FileNotFoundError(f"[ERROR] Transactions file not found: {transactions_file}")

    # Find replacement levels file (WEEKLY, not season - we need window-based)
    transformations_dir = Path(ctx.data_directory) / 'transformations'
    replacement_file = transformations_dir / 'replacement_levels.parquet'

    if not replacement_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Replacement levels file not found: {replacement_file}\n"
            f"   Make sure replacement_level_v2.py has run first!"
        )

    # Find league settings
    league_settings_dir = Path(ctx.data_directory) / 'league_settings'
    settings_files = sorted(league_settings_dir.glob('league_settings_*.json'))
    if not settings_files:
        raise FileNotFoundError(f"[ERROR] No league settings files found in {league_settings_dir}")

    league_settings_path = settings_files[-1]
    print(f"[Settings] Using league settings: {league_settings_path.name}")

    # Backup if requested
    if args.backup and transactions_file.exists():
        backup_file(transactions_file)

    # Dry run check
    if args.dry_run:
        print("\n[DRY RUN] - No changes will be made")
        print(f"   Would read from: {transactions_file}")
        print(f"   Would read replacement levels from: {replacement_file}")
        print(f"   Would write to: {transactions_file}")
        return

    # Load data
    print(f"\n[Loading] Transactions data from {transactions_file.name}...")
    transactions_df = pd.read_parquet(transactions_file)
    print(f"   Loaded {len(transactions_df):,} transactions")

    print(f"\n[Loading] Weekly replacement levels from {replacement_file.name}...")
    replacement_df = pd.read_parquet(replacement_file)
    print(f"   Loaded {len(replacement_df):,} weekly replacement baselines")

    # Validate required columns
    required_cols = ['year', 'week', 'yahoo_player_id']
    missing_cols = [c for c in required_cols if c not in transactions_df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing required columns: {missing_cols}")

    # Handle both 'type' and 'transaction_type' column names
    if 'transaction_type' in transactions_df.columns and 'type' not in transactions_df.columns:
        transactions_df['type'] = transactions_df['transaction_type']
    elif 'type' not in transactions_df.columns:
        raise ValueError(f"[ERROR] Missing transaction type column (expected 'type' or 'transaction_type')")

    # Check that performance metrics exist (from player_to_transactions)
    if 'total_points_rest_of_season' not in transactions_df.columns:
        print("[WARNING] total_points_rest_of_season not found - SPAR will be 0")
        print("   Make sure player_to_transactions_v2.py has run before this script!")

    # Calculate SPAR metrics
    print("\n[Calculating] rest-of-season SPAR metrics...")
    print("   (Calculating window-based replacement for each transaction)")
    transactions_df = calculate_all_transaction_metrics(
        transactions_df, replacement_df, league_settings_path
    )
    print("   [OK] Added: replacement_ppg_ros, fa_spar_ros, fa_ppg_ros, fa_pgvor_ros, waiver_cost_norm, fa_roi")
    print("   [OK] Added: spar_per_faab, net_spar_ros, spar_efficiency, position_spar_percentile")

    # Display sample by transaction type
    print("\n[Stats] Sample transaction value metrics:")
    display_cols = [
        'year', 'week', 'type', 'player_name',
        'total_points_rest_of_season', 'weeks_rest_of_season',
        'replacement_ppg_ros', 'fa_spar_ros', 'fa_pgvor_ros',
        'waiver_cost_norm', 'fa_roi'
    ]
    available_cols = [c for c in display_cols if c in transactions_df.columns]

    # Show sample by transaction type
    for trans_type in ['add', 'drop', 'trade']:
        type_df = transactions_df[transactions_df['type'] == trans_type]
        if not type_df.empty:
            print(f"\n{trans_type.upper()} transactions (first 5):")
            print(type_df[available_cols].head(5).to_string(index=False))

    # Show top ROI pickups (add/trade only)
    add_trade_df = transactions_df[transactions_df['type'].isin(['add', 'trade'])]
    if 'fa_roi' in add_trade_df.columns and add_trade_df['fa_roi'].notna().any():
        print("\n[Top 10] ROI Pickups:")
        top_roi = add_trade_df.nlargest(10, 'fa_roi')[available_cols]
        print(top_roi.to_string(index=False))

    # Save updated transactions file (both parquet and CSV)
    print(f"\n[Saving] Writing updated transactions data to {transactions_file}...")
    transactions_df.to_parquet(transactions_file, index=False)
    print(f"   [OK] Saved {len(transactions_df):,} rows to parquet")

    # Also save CSV version
    csv_file = transactions_file.with_suffix('.csv')
    transactions_df.to_csv(csv_file, index=False)
    print(f"   [OK] Saved {len(transactions_df):,} rows to CSV: {csv_file.name}")

    print("\n" + "="*80)
    print("[SUCCESS] TRANSACTION VALUE METRICS CALCULATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate SPAR-based transaction value metrics'
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
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before modifying files'
    )

    args = parser.parse_args()
    main(args)
