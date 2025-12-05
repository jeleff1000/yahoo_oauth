"""
Discover Franchises Transformation

Runs early in the transformation pipeline to establish franchise_id for each team.
Must run BEFORE cumulative_stats.py so career stats are grouped correctly.

This transformation:
1. Loads existing franchise_config.json (preserves manual edits)
2. Scans matchup + draft data for all team instances
3. Creates/updates franchises with disambiguation when needed
4. Adds franchise_id and franchise_name columns to matchup.parquet
5. Saves updated franchise_config.json

Usage:
    python discover_franchises.py --context path/to/league_context.json
    python discover_franchises.py --context path/to/league_context.json --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()
_transformations_dir = _script_file.parent.parent
_multi_league_dir = _transformations_dir.parent
sys.path.insert(0, str(_multi_league_dir.parent))
sys.path.insert(0, str(_multi_league_dir))

from core.league_context import LeagueContext
from core.franchise_registry import FranchiseRegistry


def discover_franchises(ctx: LeagueContext, dry_run: bool = False) -> dict:
    """
    Discover and apply franchise identifiers to league data.

    Args:
        ctx: LeagueContext with paths to data files
        dry_run: If True, show what would be done without saving

    Returns:
        Dict with statistics about franchises discovered
    """
    stats = {
        'franchises_total': 0,
        'franchises_needing_disambiguation': 0,
        'tables_updated': {},
        'orphan_teams': []
    }

    data_dir = Path(ctx.data_directory)
    config_path = data_dir / "franchise_config.json"

    # Load data files
    print("\n[Loading] Data files...")

    matchup_file = ctx.canonical_matchup_file
    draft_file = ctx.canonical_draft_file

    matchup_df = None
    draft_df = None

    if matchup_file.exists():
        matchup_df = pd.read_parquet(matchup_file)
        print(f"  Loaded matchup: {len(matchup_df):,} rows")
    else:
        print(f"  [WARN] Matchup file not found: {matchup_file}")
        return stats

    if draft_file.exists():
        draft_df = pd.read_parquet(draft_file)
        print(f"  Loaded draft: {len(draft_df):,} rows")
    else:
        print(f"  [INFO] Draft file not found (optional): {draft_file}")

    # Discover franchises
    print("\n[Discovering] Franchises...")

    registry = FranchiseRegistry.from_data(
        matchup_df=matchup_df,
        draft_df=draft_df,
        existing_config=config_path if config_path.exists() else None
    )

    stats['franchises_total'] = len(registry.franchises)

    # Count franchises needing disambiguation
    for franchise in registry.franchises.values():
        if ' - ' in franchise.franchise_name:
            stats['franchises_needing_disambiguation'] += 1

    # Show summary
    print("\n[Summary] Franchise Registry:")
    summary_df = registry.get_summary_df()
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            print(f"  {row['franchise_name']}")
            print(f"    ID: {row['franchise_id']}")
            print(f"    Years: {row['first_year']}-{row['last_year']} ({row['years_active']} seasons)")
            print(f"    Record: {row['career_wins']}W-{row['career_losses']}L")
            if len(row['team_names']) > 1:
                print(f"    Team names: {row['team_names']}")
            print()

    if dry_run:
        print("\n[DRY RUN] Would update the following:")
        print(f"  - Add franchise_id/franchise_name to matchup.parquet")
        print(f"  - Save franchise_config.json with {len(registry.franchises)} franchises")
        return stats

    # Apply franchise columns to matchup data
    print("\n[Applying] Franchise columns to matchup data...")

    matchup_df = registry.apply_franchise_columns(matchup_df)

    # Verify columns were added
    if 'franchise_id' in matchup_df.columns:
        non_null = matchup_df['franchise_id'].notna().sum()
        print(f"  franchise_id: {non_null:,}/{len(matchup_df):,} rows mapped")
        stats['tables_updated']['matchup'] = non_null
    else:
        print("  [WARN] franchise_id column not added!")

    # Save updated matchup
    print(f"\n[Saving] Updated matchup to {matchup_file.name}...")
    matchup_df.to_parquet(matchup_file, index=False)

    # Also save CSV version
    csv_path = matchup_file.with_suffix('.csv')
    try:
        matchup_df.to_csv(csv_path, index=False)
    except Exception:
        pass

    # Save franchise config
    print(f"\n[Saving] Franchise config to {config_path.name}...")
    registry.save(config_path)

    # Apply to other tables if they exist
    print("\n[Propagating] Franchise columns to other tables...")
    other_tables = {
        'draft': ctx.canonical_draft_file,
        'transactions': ctx.canonical_transaction_file,
        'schedule': data_dir / "schedule.parquet",
        'player': ctx.canonical_player_file,
    }

    for table_name, file_path in other_tables.items():
        if not file_path.exists():
            print(f"  [SKIP] {table_name}: file not found at {file_path}")
            stats['tables_updated'][table_name] = 'not_found'
            continue

        try:
            df = pd.read_parquet(file_path)
            total_rows = len(df)

            if 'manager_guid' not in df.columns:
                print(f"  [SKIP] {table_name}: no manager_guid column ({total_rows:,} rows)")
                stats['tables_updated'][table_name] = 'no_guid'
                continue

            # Check how many rows have valid manager_guid
            guid_count = df['manager_guid'].notna().sum()
            print(f"  [APPLY] {table_name}: {total_rows:,} rows, {guid_count:,} with manager_guid")

            df = registry.apply_franchise_columns(df)
            df.to_parquet(file_path, index=False)

            non_null = df['franchise_id'].notna().sum() if 'franchise_id' in df.columns else 0
            stats['tables_updated'][table_name] = non_null
            print(f"  [OK] {table_name}: {non_null:,}/{total_rows:,} rows mapped to franchise_id")

            # CSV backup
            try:
                df.to_csv(file_path.with_suffix('.csv'), index=False)
            except Exception:
                pass
        except Exception as e:
            print(f"  [WARN] Could not update {table_name}: {e}")
            stats['tables_updated'][table_name] = f'error: {e}'

    # Check for orphan teams
    orphans = registry.get_orphan_teams()
    if orphans:
        stats['orphan_teams'] = orphans
        print(f"\n[WARN] {len(orphans)} orphan teams need manual linking:")
        for orphan in orphans:
            print(f"  - {orphan}")

    return stats


def main(args):
    """Main entry point."""
    print("\n" + "=" * 80)
    print("DISCOVER FRANCHISES")
    print("=" * 80)

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"\n[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] {ctx.data_directory}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made")

    stats = discover_franchises(ctx, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"  Total franchises: {stats['franchises_total']}")
    print(f"  Needing disambiguation: {stats['franchises_needing_disambiguation']}")
    print(f"\n  Tables with franchise_id:")
    for table, result in stats['tables_updated'].items():
        if isinstance(result, int):
            print(f"    - {table}: {result:,} rows mapped")
        else:
            print(f"    - {table}: {result}")
    if stats['orphan_teams']:
        print(f"\n  Orphan teams: {len(stats['orphan_teams'])} (need manual linking)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discover and apply franchise identifiers'
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
