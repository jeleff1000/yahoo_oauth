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


def is_valid_guid(guid) -> bool:
    """
    Check if a GUID is valid (not a placeholder for hidden managers).

    Invalid GUIDs include:
    - None, NaN, empty string
    - '--' (Yahoo's placeholder for hidden managers)
    - Any other placeholder patterns

    When a GUID is invalid, we should NOT unify records because they may be
    different people who both have hidden profiles.
    """
    if guid is None or pd.isna(guid):
        return False
    guid_str = str(guid).strip()
    if not guid_str or guid_str in ('--', 'None', 'nan', '<NA>'):
        return False
    return True


def build_guid_to_name_mapping(dataframes: dict) -> dict:
    """
    Build a mapping from manager_guid to the most recent year's manager name.

    Args:
        dataframes: Dict of table_name -> DataFrame with manager and manager_guid columns

    Returns:
        Dict mapping guid -> most recent manager name

    Note: GUIDs that are '--' or other placeholders are SKIPPED because they
    represent hidden managers who may be different people.
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
            # Skip invalid GUIDs (like '--') - these are different hidden managers
            if not is_valid_guid(guid):
                continue
            if manager and pd.notna(manager):
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

    Updates:
    - manager column (using guid)
    - opponent column (by name matching, since opponent doesn't have guid)
    - manager_year and manager_week composite keys

    For hidden managers (GUID = '--'), uses team_name as manager name to keep
    them as separate individuals.

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
    hidden_updates = 0  # Track hidden manager fixes

    # Step 0: Fix hidden managers - use team_name as manager name
    # This keeps different hidden managers as separate people
    if 'team_name' in df.columns:
        hidden_mask = df['manager_guid'] == '--'
        for idx in df.index[hidden_mask]:
            team_name = df.at[idx, 'team_name']
            old_name = df.at[idx, 'manager']
            if team_name and pd.notna(team_name) and str(team_name).strip():
                new_name = str(team_name).strip()
                if old_name != new_name:
                    df.at[idx, 'manager'] = new_name
                    hidden_updates += 1

                    # Recalculate composite keys
                    new_name_no_spaces = new_name.replace(" ", "")

                    if 'manager_year' in df.columns and 'year' in df.columns:
                        year_val = df.at[idx, 'year']
                        if pd.notna(year_val):
                            df.at[idx, 'manager_year'] = f"{new_name_no_spaces}{int(year_val)}"

                    if 'manager_week' in df.columns and 'cumulative_week' in df.columns:
                        cw = df.at[idx, 'cumulative_week']
                        if pd.notna(cw):
                            df.at[idx, 'manager_week'] = f"{new_name_no_spaces}{int(cw)}"

        if hidden_updates > 0:
            print(f"    Fixed {hidden_updates} hidden manager names using team_name")

            # Build mapping of hidden manager old names to new names for opponent updates
            # Multiple hidden managers may share the same old name, so we need per-row matching
            # Create lookup: (year, week, old_manager_name, team_points) -> new_name
            hidden_lookup = {}
            for idx in df.index[hidden_mask]:
                team_name = df.at[idx, 'team_name']
                if team_name and pd.notna(team_name) and str(team_name).strip():
                    # Use combination of identifying fields to uniquely identify
                    year = df.at[idx, 'year'] if 'year' in df.columns else None
                    week = df.at[idx, 'week'] if 'week' in df.columns else None
                    pts = df.at[idx, 'team_points'] if 'team_points' in df.columns else None
                    hidden_lookup[(year, week, pts)] = str(team_name).strip()

            # Update opponent names for hidden managers
            if 'opponent' in df.columns and 'opponent_points' in df.columns:
                opponent_updates = 0
                for idx in df.index:
                    opp = df.at[idx, 'opponent']
                    if opp == 'I Like Ceeeereal':  # Only fix the conflated name
                        year = df.at[idx, 'year'] if 'year' in df.columns else None
                        week = df.at[idx, 'week'] if 'week' in df.columns else None
                        opp_pts = df.at[idx, 'opponent_points']
                        # Opponent's team_points = our opponent_points
                        if (year, week, opp_pts) in hidden_lookup:
                            df.at[idx, 'opponent'] = hidden_lookup[(year, week, opp_pts)]
                            opponent_updates += 1
                if opponent_updates > 0:
                    print(f"    Fixed {opponent_updates} opponent references for hidden managers")

    # Step 0b: Normalize opponent casing to match manager names
    # Yahoo API sometimes returns opponents with different casing than manager names
    # This breaks SQL joins (m1.opponent = m2.manager)
    if 'opponent' in df.columns and 'manager' in df.columns:
        # Build lowercase -> actual manager name mapping
        manager_names = df['manager'].dropna().unique()
        manager_case_map = {str(m).lower(): str(m) for m in manager_names}

        # Fix opponent casing to match manager names
        case_fixes = 0
        for idx in df.index:
            opp = df.at[idx, 'opponent']
            if opp and pd.notna(opp):
                opp_lower = str(opp).lower()
                if opp_lower in manager_case_map:
                    correct_name = manager_case_map[opp_lower]
                    if str(opp) != correct_name:
                        df.at[idx, 'opponent'] = correct_name
                        case_fixes += 1
        if case_fixes > 0:
            print(f"    Fixed {case_fixes} opponent name casing issues")

    # Step 1: Build old_name -> new_name mapping from guid
    old_to_new = {}
    for idx, row in df.iterrows():
        guid = row.get('manager_guid')
        # Skip invalid GUIDs - don't unify hidden managers
        if not is_valid_guid(guid):
            continue
        guid_str = str(guid)
        if guid_str in guid_to_name:
            old_name = row['manager']
            new_name = guid_to_name[guid_str]
            if old_name != new_name and old_name not in old_to_new:
                old_to_new[old_name] = new_name

    # Step 2: Update manager column using guid (most accurate)
    updates = hidden_updates  # Start with hidden manager fixes
    for idx, row in df.iterrows():
        guid = row.get('manager_guid')
        # Skip invalid GUIDs - don't unify hidden managers
        if not is_valid_guid(guid):
            continue
        guid_str = str(guid)
        if guid_str in guid_to_name:
            new_name = guid_to_name[guid_str]
            old_name = row['manager']
            if old_name != new_name:
                df.at[idx, 'manager'] = new_name
                updates += 1

                # Recalculate composite keys
                new_name_no_spaces = new_name.replace(" ", "")

                if 'manager_year' in df.columns and 'year' in df.columns:
                    year_val = row.get('year')
                    if pd.notna(year_val):
                        df.at[idx, 'manager_year'] = f"{new_name_no_spaces}{int(year_val)}"

                if 'manager_week' in df.columns and 'cumulative_week' in df.columns:
                    cw = row.get('cumulative_week')
                    if pd.notna(cw):
                        df.at[idx, 'manager_week'] = f"{new_name_no_spaces}{int(cw)}"

    # Step 3: Update opponent column using old_to_new mapping
    if 'opponent' in df.columns:
        opponent_updates = 0
        for old_name, new_name in old_to_new.items():
            mask = df['opponent'] == old_name
            opponent_updates += mask.sum()
            df.loc[mask, 'opponent'] = new_name
        if opponent_updates > 0:
            print(f"    Also updated {opponent_updates} opponent references")

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
