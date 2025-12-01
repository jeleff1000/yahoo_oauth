"""
Resolve Hidden Manager Names Using GUID

This transformation runs early in the pipeline to unify manager names across
years using the persistent Yahoo manager GUID.

When a manager has their profile set to private (--hidden--), their name
falls back to their team name, which can change from year to year. This
makes it difficult to track the same manager across seasons.

The GUID is a persistent identifier that stays the same across all years
for the same Yahoo user. This transformation:

1. Groups all records by manager_guid
2. For each guid, finds the most recent year's manager name
3. Updates all records for that guid to use that consistent name

This ensures the same person always shows with the same name, even if
their team name changes across seasons.

Usage:
    python resolve_hidden_managers.py --context path/to/league_context.json
    python resolve_hidden_managers.py --context path/to/league_context.json --dry-run
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


def build_guid_to_name_mapping(dataframes: dict) -> dict:
    """
    Build a mapping from manager_guid to the most recent year's manager name.

    Args:
        dataframes: Dict of table_name -> DataFrame with manager and manager_guid columns

    Returns:
        Dict mapping guid -> most recent manager name
    """
    # Collect all (guid, year, manager) tuples
    all_records = []

    for table_name, df in dataframes.items():
        if df is None or df.empty:
            continue
        if 'manager_guid' not in df.columns or 'manager' not in df.columns:
            continue
        if 'year' not in df.columns:
            continue

        # Get unique (guid, year, manager) combinations
        subset = df[['manager_guid', 'year', 'manager']].dropna(subset=['manager_guid'])
        subset = subset.drop_duplicates()

        for _, row in subset.iterrows():
            guid = row['manager_guid']
            year = row['year']
            manager = row['manager']
            if guid and pd.notna(guid) and manager and pd.notna(manager):
                all_records.append({
                    'guid': str(guid),
                    'year': int(year),
                    'manager': str(manager)
                })

    if not all_records:
        return {}

    # Convert to DataFrame for easier processing
    records_df = pd.DataFrame(all_records)

    # For each guid, get the manager name from the most recent year
    guid_to_name = {}
    for guid in records_df['guid'].unique():
        guid_records = records_df[records_df['guid'] == guid]
        # Get the most recent year for this guid
        most_recent = guid_records.loc[guid_records['year'].idxmax()]
        guid_to_name[guid] = most_recent['manager']

    return guid_to_name


def update_manager_names(df: pd.DataFrame, guid_to_name: dict, table_name: str) -> tuple:
    """
    Update manager names in a DataFrame using the guid mapping.

    Args:
        df: DataFrame with manager and manager_guid columns
        guid_to_name: Dict mapping guid -> unified manager name
        table_name: Name of the table (for logging)

    Returns:
        Tuple of (updated_df, num_updates)
    """
    if df is None or df.empty:
        return df, 0

    if 'manager_guid' not in df.columns or 'manager' not in df.columns:
        return df, 0

    df = df.copy()
    updates = 0

    for idx, row in df.iterrows():
        guid = row.get('manager_guid')
        if guid and pd.notna(guid):
            guid_str = str(guid)
            if guid_str in guid_to_name:
                new_name = guid_to_name[guid_str]
                old_name = row['manager']
                if old_name != new_name:
                    df.at[idx, 'manager'] = new_name
                    updates += 1

    return df, updates


def resolve_hidden_managers(ctx: LeagueContext, dry_run: bool = False) -> dict:
    """
    Resolve hidden manager names across all tables using GUID.

    Args:
        ctx: LeagueContext with paths to data files
        dry_run: If True, only show what would be done without making changes

    Returns:
        Dict with statistics about updates made
    """
    stats = {
        'guids_found': 0,
        'tables_updated': {},
        'total_updates': 0
    }

    # Define tables to process
    tables = {
        'player': ctx.canonical_player_file,
        'matchup': ctx.canonical_matchup_file,
        'draft': ctx.canonical_draft_file,
        'transactions': ctx.canonical_transaction_file,
        'schedule': Path(ctx.data_directory) / "schedule.parquet",
    }

    # Load all tables
    dataframes = {}
    for table_name, file_path in tables.items():
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                dataframes[table_name] = df
                print(f"  Loaded {table_name}: {len(df):,} rows")
                if 'manager_guid' in df.columns:
                    non_null = df['manager_guid'].notna().sum()
                    print(f"    - manager_guid: {non_null:,} non-null values")
            except Exception as e:
                print(f"  WARNING: Could not load {table_name}: {e}")
                dataframes[table_name] = None
        else:
            print(f"  {table_name}: not found (skipping)")
            dataframes[table_name] = None

    # Build guid -> name mapping
    print("\n[Building] GUID to name mapping...")
    guid_to_name = build_guid_to_name_mapping(dataframes)
    stats['guids_found'] = len(guid_to_name)
    print(f"  Found {len(guid_to_name)} unique GUIDs with manager names")

    if not guid_to_name:
        print("  No GUIDs found - nothing to do")
        return stats

    # Show the mapping
    print("\n[Mapping] GUID -> Manager Name (most recent year):")
    for guid, name in sorted(guid_to_name.items(), key=lambda x: x[1]):
        print(f"  {guid[:12]}... -> {name}")

    if dry_run:
        print("\n[DRY RUN] Would update the following tables:")
        for table_name, df in dataframes.items():
            if df is not None and 'manager_guid' in df.columns:
                _, potential_updates = update_manager_names(df.copy(), guid_to_name, table_name)
                print(f"  {table_name}: {potential_updates:,} potential updates")
        return stats

    # Update each table
    print("\n[Updating] Manager names in each table...")
    for table_name, df in dataframes.items():
        if df is None:
            continue

        updated_df, num_updates = update_manager_names(df, guid_to_name, table_name)
        stats['tables_updated'][table_name] = num_updates
        stats['total_updates'] += num_updates

        if num_updates > 0:
            # Save updated table
            file_path = tables[table_name]
            updated_df.to_parquet(file_path, index=False)
            print(f"  {table_name}: Updated {num_updates:,} rows -> {file_path.name}")

            # Also save CSV version
            csv_path = file_path.with_suffix('.csv')
            updated_df.to_csv(csv_path, index=False)
        else:
            print(f"  {table_name}: No updates needed")

    return stats


def main(args):
    """Main entry point."""
    print("\n" + "="*80)
    print("RESOLVE HIDDEN MANAGERS BY GUID")
    print("="*80 + "\n")

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] Data directory: {ctx.data_directory}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made")

    print("\n[Loading] Data files...")
    stats = resolve_hidden_managers(ctx, dry_run=args.dry_run)

    print("\n" + "="*80)
    print("[SUMMARY]")
    print(f"  GUIDs found: {stats['guids_found']}")
    print(f"  Total updates: {stats['total_updates']}")
    for table_name, count in stats.get('tables_updated', {}).items():
        print(f"    - {table_name}: {count:,}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resolve hidden manager names using GUID'
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
