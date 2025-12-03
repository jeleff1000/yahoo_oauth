"""
Replacement Level Calculation (Multi-League V2)

Calculates WEEK-BY-WEEK position-specific replacement levels for SPAR.

This transformation computes:
1. Weekly Replacement Levels (PERCENTILE-BASED HYBRID APPROACH):
   - Filter players to fantasy_points > 0
   - Rank by fantasy_points within position
   - Replacement = avg of ranks N and N+1, where N varies week-to-week

   For LEAGUE WEEKS (has Yahoo roster data):
   - N = actual count of players who were BOTH:
     * Rostered in Yahoo (have yahoo_player_id)
     * Actually played (have NFL_player_id from NFLverse)
   - Excludes players rostered but didn't play (injured, bye, etc.)
   - Example: Week 5 has 52 RBs rostered & played → 52nd RB is replacement

   For NON-LEAGUE WEEKS (NFL playoffs, pre-league history):
   - Calculate percentile from league weeks (rostered_and_played / total_who_played)
   - Apply this percentile to the actual player pool that week
   - Example: If RB percentile is 55% and 60 RBs played in playoffs:
     → Replacement is 56th percentile = 33rd RB (60 × 0.55)
   - This adapts to varying player pool sizes (fewer teams play in playoffs)

   - Handles flex positions (distributed among RB/WR/TE based on actual roster data)

2. Season Replacement Levels:
   - Average of weekly replacement levels across weeks 1-17
   - Used for draft value metrics (full season baseline)

3. Window Replacement Levels (optional):
   - Average of weekly replacement levels for rest-of-season windows
   - Used for transaction value metrics (Week W → 17)

Output:
- Creates replacement_levels.parquet with:
  * Weekly: year, week, position, replacement_ppg, n_pos, roster_count_source
  * Season: year, position, replacement_ppg_season, n_pos

Key Features:
- Week-by-week replacement (52nd RB one week, 55th another)
- Adapts to non-league weeks (percentile-based for NFL playoffs, pre-league history)
- Based on actual contributing roster spots (rostered AND played)

Usage:
    python replacement_level_v2.py --context path/to/league_context.json
    python replacement_level_v2.py --context path/to/league_context.json --dry-run
    python replacement_level_v2.py --context path/to/league_context.json --backup
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

# Auto-detect if we're in modules/ subdirectory or transformations/ directory
if _script_file.parent.name == 'modules':
    _modules_dir = _script_file.parent
    _transformations_dir = _modules_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'player_enrichment':
    # We're in multi_league/transformations/player_enrichment/
    _transformations_dir = _script_file.parent.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    # Fallback: navigate up to find multi_league
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent  # fantasy_football_data_scripts directory
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))

from core.league_context import LeagueContext
from multi_league.transformations.player.modules.replacement_calculator_dynamic import (
    calculate_all_replacements
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
    """Main entry point for replacement level calculation."""
    print("\n" + "="*80)
    print("REPLACEMENT LEVEL CALCULATION V2")
    print("="*80 + "\n")

    # Load league context
    ctx = LeagueContext.load(args.context)
    print(f"[League] {ctx.league_name} ({ctx.league_id})")
    print(f"[Dir] Data directory: {ctx.data_directory}")

    # Check for required files
    player_file = ctx.canonical_player_file
    if not player_file.exists():
        raise FileNotFoundError(f"[ERROR] Player file not found: {player_file}")

    # Find league settings (use most recent year)
    league_settings_dir = Path(ctx.data_directory) / 'league_settings'
    if not league_settings_dir.exists():
        raise FileNotFoundError(f"[ERROR] League settings directory not found: {league_settings_dir}")

    # Get the most recent settings file
    settings_files = sorted(league_settings_dir.glob('league_settings_*.json'))
    if not settings_files:
        raise FileNotFoundError(f"[ERROR] No league settings files found in {league_settings_dir}")

    league_settings_path = settings_files[-1]  # Most recent by filename
    print(f"[Settings] Using league settings: {league_settings_path.name}")

    # Define output path
    output_dir = Path(ctx.data_directory) / 'transformations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'replacement_levels.parquet'

    # Backup existing file if requested
    if args.backup and output_file.exists():
        backup_file(output_file)

    # Dry run check
    if args.dry_run:
        print("\n[DRY RUN] - No changes will be made")
        print(f"   Would read from: {player_file}")
        print(f"   Would write to: {output_file}")
        return

    # Load player data
    print(f"\n[Loading] Player data from {player_file.name}...")
    player_df = pd.read_parquet(player_file)
    print(f"   Loaded {len(player_df):,} rows")

    # Validate required columns
    required_cols = ['year', 'week', 'position', 'fantasy_points']
    missing_cols = [c for c in required_cols if c not in player_df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing required columns: {missing_cols}")

    # Check that fantasy_points has been calculated
    if player_df['fantasy_points'].isna().all():
        raise ValueError(
            "[ERROR] fantasy_points column is all NaN. "
            "Make sure player_stats_v2.py has run first!"
        )

    # Calculate replacement levels
    print("\n[Calculating] replacement levels...")
    print("   (This may take a few minutes for large datasets)")

    weekly_replacement, season_replacement = calculate_all_replacements(
        player_df, league_settings_path
    )

    print(f"   [OK] Calculated weekly replacement for {len(weekly_replacement):,} year/week/position combinations")
    print(f"   [OK] Calculated season replacement for {len(season_replacement):,} year/position combinations")

    # Combine weekly and season data
    # Weekly data already has year, week, position, replacement_ppg, n_pos
    # Season data has year, position, replacement_ppg_season, n_pos

    # Merge season data into weekly for convenience
    combined_df = weekly_replacement.merge(
        season_replacement[['year', 'position', 'replacement_ppg_season']],
        on=['year', 'position'],
        how='left'
    )

    # Display sample
    print("\n[Stats] Sample replacement levels:")
    print("\nWeekly (first 10 rows):")
    print(combined_df.head(10).to_string(index=False))

    print("\nSeason averages by position (Year 2024):")
    if 2024 in combined_df['year'].values:
        season_sample = combined_df[
            (combined_df['year'] == 2024) &
            (combined_df['week'] == 1)
        ][['position', 'replacement_ppg_season', 'n_pos']].drop_duplicates()
        print(season_sample.to_string(index=False))

    # Save to parquet
    print(f"\n[Saving] Writing replacement levels to {output_file}...")
    combined_df.to_parquet(output_file, index=False)
    print(f"   [OK] Saved {len(combined_df):,} rows")

    # Also save a season-only file for easy joining
    season_file = output_dir / 'replacement_levels_season.parquet'
    season_replacement.to_parquet(season_file, index=False)
    print(f"   [OK] Saved season averages to {season_file.name} ({len(season_replacement):,} rows)")

    # =========================================================================
    # NEW: Add SPAR columns directly to player.parquet
    # This solves the circular dependency issue where player_stats_v2.py runs
    # before replacement_levels exist, so SPAR was never calculated
    # =========================================================================
    print("\n[SPAR] Adding SPAR columns to player.parquet...")

    # Drop any existing SPAR-related columns from player_df
    # (player_stats_v2.py adds placeholder columns when replacement_levels.parquet doesn't exist)
    spar_cols_to_drop = [
        'replacement_ppg', 'replacement_ppg_season',
        'player_spar', 'manager_spar', 'spar_season'
    ]
    existing_spar_cols = [c for c in spar_cols_to_drop if c in player_df.columns]
    if existing_spar_cols:
        print(f"   Dropping existing SPAR placeholder columns: {existing_spar_cols}")
        player_df = player_df.drop(columns=existing_spar_cols)

    # Merge weekly replacement levels to player data
    player_with_spar = player_df.merge(
        combined_df[['year', 'week', 'position', 'replacement_ppg', 'replacement_ppg_season']],
        on=['year', 'week', 'position'],
        how='left'
    )

    # Calculate SPAR metrics (matching player_stats_v2.py naming conventions)

    # player_spar: Weekly points above replacement (all games - talent metric)
    player_with_spar['player_spar'] = (
        player_with_spar['fantasy_points'] - player_with_spar['replacement_ppg']
    ).round(2)

    # manager_spar: Weekly points above replacement (started games only - usage metric)
    if 'is_started' in player_with_spar.columns:
        player_with_spar['manager_spar'] = np.where(
            player_with_spar['is_started'] == 1,
            player_with_spar['player_spar'],
            0.0
        )
    else:
        # Fallback: same as player_spar if is_started not available
        player_with_spar['manager_spar'] = player_with_spar['player_spar']

    # spar_season: Season-level SPAR (for draft value comparison)
    player_with_spar['spar_season'] = (
        player_with_spar['fantasy_points'] - player_with_spar['replacement_ppg_season']
    ).round(2)

    # Count how many players got SPAR values
    spar_count = player_with_spar['player_spar'].notna().sum()
    total_count = len(player_with_spar)
    print(f"   [OK] Calculated SPAR for {spar_count:,}/{total_count:,} player rows ({100*spar_count/total_count:.1f}%)")

    # Save updated player.parquet
    player_with_spar.to_parquet(player_file, index=False)
    print(f"   [OK] Updated {player_file.name} with SPAR columns")

    # Also update CSV if it exists
    csv_file = player_file.with_suffix('.csv')
    if csv_file.exists():
        player_with_spar.to_csv(csv_file, index=False)
        print(f"   [OK] Updated {csv_file.name}")

    print("\n" + "="*80)
    print("[SUCCESS] REPLACEMENT LEVEL CALCULATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate replacement levels for SPAR metrics'
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
