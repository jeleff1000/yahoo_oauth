"""
Normalize Data Types in Canonical Files

This FINAL transformation ensures all join keys have consistent data types
across player.parquet, draft.parquet, and transactions.parquet.

This must run LAST in the pipeline after all other transformations.

Normalizations:
- yahoo_player_id → Int64 (nullable integer)
- year → Int64
- week → Int64
- cumulative_week → Int64
- league_id → string (no change)

Usage:
    python normalize_canonical_types.py --context path/to/league_context.json
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


def normalize_dataframe_types(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    Normalize join key data types to ensure consistent merging.

    Args:
        df: DataFrame to normalize
        file_name: Name of file (for logging)

    Returns:
        DataFrame with normalized types
    """
    print(f"\n[{file_name}] Normalizing data types...")

    # Define columns that should be Int64
    int64_cols = ['yahoo_player_id', 'year', 'week', 'cumulative_week',
                  'nfl_player_id', 'season', 'transaction_sequence']

    changes = []

    for col in int64_cols:
        if col not in df.columns:
            continue

        original_dtype = str(df[col].dtype)

        # Convert to Int64 (nullable integer type)
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            new_dtype = str(df[col].dtype)

            if original_dtype != new_dtype:
                changes.append(f"    {col}: {original_dtype} -> {new_dtype}")
        except Exception as e:
            print(f"    [WARNING] Could not convert {col}: {e}")

    if changes:
        print("\n".join(changes))
        print(f"  [OK] Normalized {len(changes)} columns")
    else:
        print(f"  [OK] All types already correct")

    return df


def main(args):
    """Main entry point."""
    print("\n" + "="*80)
    print("NORMALIZE CANONICAL FILE DATA TYPES")
    print("="*80 + "\n")

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] Data directory: {ctx.data_directory}")

    # Process each canonical file
    files = {
        'player': ctx.canonical_player_file,
        'draft': ctx.canonical_draft_file,
        'transactions': ctx.canonical_transaction_file
    }

    for name, file_path in files.items():
        if not file_path.exists():
            print(f"\n[SKIP] {name}.parquet not found: {file_path}")
            continue

        print(f"\nProcessing {name}.parquet ({file_path.name})...")
        print(f"  Loading... ", end='')
        df = pd.read_parquet(file_path)
        print(f"{len(df):,} rows")

        # Normalize types
        df = normalize_dataframe_types(df, name)

        # Save back
        if not args.dry_run:
            print(f"  Saving... ", end='')
            df.to_parquet(file_path, index=False)
            print(f"[OK]")
        else:
            print(f"  [DRY RUN] Would save to {file_path}")

    print("\n" + "="*80)
    print("[SUCCESS] TYPE NORMALIZATION COMPLETE")
    print("="*80)
    print("\nAll join keys now have consistent Int64 types.")
    print("Re-run the diagnostic to verify: python diagnose_join_keys.py --context ...")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalize data types in canonical parquet files'
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
