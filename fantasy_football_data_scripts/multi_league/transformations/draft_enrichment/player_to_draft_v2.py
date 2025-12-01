#!/usr/bin/env python3
"""
Player to Draft Import (Multi-League V2)

Enriches draft data with season performance stats for ROI analysis.

This transformation adds comprehensive performance context to each draft pick:
- Season performance metrics (total points, PPG, consistency)
- Position and overall rankings
- Value vs expectations (price rank vs finish, pick rank vs finish)
- ROI metrics (points per dollar, value over replacement)
"""

import sys
from pathlib import Path
from functools import wraps
from typing import Optional
from datetime import datetime
import argparse
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

from core.data_normalization import normalize_numeric_columns, ensure_league_id
from core.league_context import LeagueContext
# Add transformations/modules to path for shared utilities
_modules_dir = _multi_league_dir / "transformations" / "modules"
sys.path.insert(0, str(_modules_dir))

from type_utils import safe_merge, ensure_canonical_types


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


# =========================================================
# Helper Functions
# =========================================================

def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


def calculate_season_performance_metrics(
    yahoo_player_id: str,
    year: int,
    player_df: pd.DataFrame,
    league_id: Optional[str] = None
) -> dict:
    """
    Calculate season aggregated performance metrics for a drafted player.

    Args:
        yahoo_player_id: Yahoo player ID
        year: Season year
        player_df: Full player DataFrame with weekly stats
        league_id: League ID for multi-league isolation (optional for backward compatibility)

    Returns:
        Dict of season performance metrics
    """
    # Filter to this player and season (with multi-league isolation)
    filter_mask = (player_df['yahoo_player_id'] == yahoo_player_id) & (player_df['year'] == year)
    if league_id and 'league_id' in player_df.columns:
        filter_mask = filter_mask & (player_df['league_id'] == league_id)

    player_season = player_df[filter_mask].copy()

    if player_season.empty:
        return {}

    metrics = {}

    # Get position (most common)
    if 'position' in player_season.columns:
        position_mode = player_season['position'].mode()
        metrics['position'] = position_mode.iloc[0] if not position_mode.empty else None
    else:
        metrics['position'] = None

    # Get NFL team (most common)
    if 'nfl_team' in player_season.columns:
        team_mode = player_season['nfl_team'].mode()
        metrics['nfl_team'] = team_mode.iloc[0] if not team_mode.empty else None
    else:
        metrics['nfl_team'] = None

    # Performance metrics - ALL weeks (player talent metric)
    points_col = 'fantasy_points' if 'fantasy_points' in player_season.columns else 'points'
    if points_col in player_season.columns:
        points = player_season[points_col].fillna(0)

        metrics['total_fantasy_points'] = points.sum()
        metrics['games_played'] = len(player_season)
        metrics['games_with_points'] = (points > 0).sum()
        metrics['season_ppg'] = points.mean() if len(points) > 0 else 0.0

        # Performance metrics - STARTED weeks only (manager usage metric)
        # Filter to started weeks (fantasy_position not BN/IR)
        if 'fantasy_position' in player_season.columns:
            started_filter = (
                player_season['fantasy_position'].notna() &
                ~player_season['fantasy_position'].isin(['BN', 'IR', 'IR+'])
            )
            started_weeks = player_season[started_filter].copy()

            if not started_weeks.empty:
                started_points = started_weeks[points_col].fillna(0)
                metrics['total_fantasy_points_started'] = started_points.sum()
                metrics['games_started'] = len(started_weeks)
                metrics['season_ppg_started'] = started_points.mean() if len(started_points) > 0 else 0.0
            else:
                # Player was never started
                metrics['total_fantasy_points_started'] = 0.0
                metrics['games_started'] = 0
                metrics['season_ppg_started'] = 0.0
        else:
            # Fallback: if no fantasy_position column, assume all weeks are started
            metrics['total_fantasy_points_started'] = metrics['total_fantasy_points']
            metrics['games_started'] = metrics['games_played']
            metrics['season_ppg_started'] = metrics['season_ppg']

        # Best/worst weeks (overall, not started-specific)
        if len(points) > 0:
            metrics['best_week_points'] = points.max()
            points_nonzero = points[points > 0]
            metrics['worst_week_points'] = points_nonzero.min() if len(points_nonzero) > 0 else 0.0
        else:
            metrics['best_week_points'] = 0.0
            metrics['worst_week_points'] = 0.0

        # Consistency (coefficient of variation - overall)
        if metrics['season_ppg'] > 0:
            metrics['consistency_score'] = (points.std() / metrics['season_ppg']) * 100
        else:
            metrics['consistency_score'] = None
    else:
        metrics['total_fantasy_points'] = 0.0
        metrics['games_played'] = 0
        metrics['games_with_points'] = 0
        metrics['season_ppg'] = 0.0
        metrics['total_fantasy_points_started'] = 0.0
        metrics['games_started'] = 0
        metrics['season_ppg_started'] = 0.0
        metrics['best_week_points'] = 0.0
        metrics['worst_week_points'] = 0.0
        metrics['consistency_score'] = None

    return metrics


@ensure_normalized
def calculate_ranking_metrics(
    draft_df: pd.DataFrame,
    player_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate position and overall rankings for all drafted players.

    This includes:
    - Season position rank (where they finished within their position)
    - Season overall rank (where they finished overall)
    - Price rank within position (where they were drafted by cost)
    - Pick rank within position (where they were drafted by pick number)
    - Value deltas (price_rank - finish_rank, pick_rank - finish_rank)

    Args:
        draft_df: Draft DataFrame with performance metrics already joined
        player_df: Full player DataFrame (not used, kept for consistency)

    Returns:
        Draft DataFrame with ranking columns added
    """
    df = draft_df.copy()

    # Ensure required columns exist
    if 'total_fantasy_points' not in df.columns:
        print("  Warning: total_fantasy_points not found, skipping ranking metrics")
        return df

    # Process each year-position group
    for year in df['year'].unique():
        year_mask = df['year'] == year
        year_df = df[year_mask].copy()

        # Overall ranking for this year (by total_fantasy_points)
        # Handle NaN values
        year_df['season_overall_rank'] = year_df['total_fantasy_points'].rank(
            method='min',
            ascending=False,
            na_option='keep'
        ).astype('Int64')

        # Update main df with overall ranks
        df.loc[year_mask, 'season_overall_rank'] = year_df['season_overall_rank']

        # Position-level rankings
        for position in year_df['position'].dropna().unique():
            pos_mask = (year_df['position'] == position)
            pos_df = year_df[pos_mask].copy()

            if len(pos_df) == 0:
                continue

            # Finish rank (by total_fantasy_points within position)
            pos_df['season_position_rank'] = pos_df['total_fantasy_points'].rank(
                method='min',
                ascending=False,
                na_option='keep'
            ).astype('Int64')

            # Total players at position
            pos_df['total_position_players'] = len(pos_df)

            # Price rank within position (by cost, lower cost = higher rank number)
            if 'cost' in pos_df.columns and pos_df['cost'].notna().any():
                pos_df['price_rank_within_position'] = pos_df['cost'].rank(
                    method='min',
                    ascending=False
                ).astype('Int64')
            else:
                pos_df['price_rank_within_position'] = pd.NA

            # Pick rank within position (by pick number, earlier pick = lower rank number)
            if 'pick' in pos_df.columns and pos_df['pick'].notna().any():
                pos_df['pick_rank_within_position'] = pos_df['pick'].rank(
                    method='min',
                    ascending=True
                ).astype('Int64')
            else:
                pos_df['pick_rank_within_position'] = pd.NA

            # Value deltas (positive = outperformed expectations)
            # Price rank vs finish: drafted 6th by cost, finished 3rd = +3 (good)
            if 'price_rank_within_position' in pos_df.columns:
                pos_df['price_rank_vs_finish_rank'] = pd.to_numeric(
                    pos_df['price_rank_within_position'] - pos_df['season_position_rank'],
                    errors='coerce'
                ).astype('Int64')

            # Pick rank vs finish: drafted 6th overall at position, finished 3rd = +3 (good)
            if 'pick_rank_within_position' in pos_df.columns:
                pos_df['pick_rank_vs_finish_rank'] = pd.to_numeric(
                    pos_df['pick_rank_within_position'] - pos_df['season_position_rank'],
                    errors='coerce'
                ).astype('Int64')

            # Update main df with position-level metrics
            year_pos_mask = year_mask & (df['position'] == position)
            for col in [
                'season_position_rank', 'total_position_players',
                'price_rank_within_position', 'pick_rank_within_position',
                'price_rank_vs_finish_rank', 'pick_rank_vs_finish_rank'
            ]:
                if col in pos_df.columns:
                    df.loc[year_pos_mask, col] = pos_df[col].values

    return df


@ensure_normalized
def calculate_roi_metrics(draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ROI and value metrics.

    Args:
        draft_df: Draft DataFrame with performance and ranking metrics

    Returns:
        Draft DataFrame with ROI columns added
    """
    df = draft_df.copy()

    # Points per dollar (auction leagues)
    if 'cost' in df.columns:
        df['points_per_dollar'] = df['total_fantasy_points'] / df['cost']
        df['points_per_dollar'] = df['points_per_dollar'].replace([np.inf, -np.inf], np.nan)

    # Points per pick (all leagues)
    if 'pick' in df.columns:
        df['points_per_pick'] = df['total_fantasy_points'] / df['pick']
        df['points_per_pick'] = df['points_per_pick'].replace([np.inf, -np.inf], np.nan)

    # Value over replacement (points above positional average)
    for year in df['year'].unique():
        year_mask = df['year'] == year
        year_df = df[year_mask]

        for position in year_df['position'].dropna().unique():
            pos_mask = year_mask & (df['position'] == position)
            pos_avg = df.loc[pos_mask, 'total_fantasy_points'].mean()
            df.loc[pos_mask, 'value_over_replacement'] = (
                df.loc[pos_mask, 'total_fantasy_points'] - pos_avg
            )

    # Value over ADP (if avg_pick exists from Yahoo)
    if 'avg_pick' in df.columns and 'pick' in df.columns:
        # Rough heuristic: compare actual vs expected performance
        # Players drafted earlier than ADP who overperform are great value
        # Convert avg_pick to numeric first (it may be stored as string with '-' for missing values)
        avg_pick_numeric = pd.to_numeric(df['avg_pick'], errors='coerce')
        pick_numeric = pd.to_numeric(df['pick'], errors='coerce')
        df['draft_position_delta'] = avg_pick_numeric - pick_numeric  # positive = drafted earlier than expected

    return df


@ensure_normalized
def enrich_draft_with_player_stats(
    draft_df: pd.DataFrame,
    player_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Main function to enrich draft data with player performance stats.

    Args:
        draft_df: Draft DataFrame
        player_df: Player DataFrame with weekly stats

    Returns:
        Enriched draft DataFrame
    """
    print(f"Enriching {len(draft_df)} draft picks with player stats...")

    # Required columns check
    if 'yahoo_player_id' not in draft_df.columns:
        print("ERROR: Draft data missing yahoo_player_id column")
        return draft_df

    if 'yahoo_player_id' not in player_df.columns:
        print("ERROR: Player data missing yahoo_player_id column")
        return draft_df

    # Clean up any existing enrichment columns from previous runs
    enrichment_cols = [
        'position', 'nfl_team', 'nfl_team_x', 'nfl_team_y', 'headshot_url',
        'total_fantasy_points', 'games_played', 'games_with_points',
        'season_ppg', 'season_std', 'best_game', 'worst_game',
        'weeks_rostered', 'weeks_started',
        'season_overall_rank', 'season_position_rank', 'total_position_players',
        'price_rank_within_position', 'pick_rank_within_position',
        'price_rank_vs_finish_rank', 'pick_rank_vs_finish_rank',
        'points_per_dollar', 'points_per_pick',
        'value_over_replacement', 'draft_position_delta'
    ]
    cols_to_drop = [c for c in enrichment_cols if c in draft_df.columns]
    if cols_to_drop:
        print(f"  Removing {len(cols_to_drop)} existing enrichment columns from previous run")
        draft_df = draft_df.drop(columns=cols_to_drop)

    # CRITICAL: Determine if league_id should be used as a join key
    # If player_df has ALL NA league_ids, we should NOT use it as a join key
    # because it will cause dtype mismatches and failed joins
    use_league_id_for_join = False

    if 'league_id' in draft_df.columns and 'league_id' in player_df.columns:
        # Convert player league_id to string dtype (not Int64)
        if player_df['league_id'].dtype != 'string':
            player_df['league_id'] = player_df['league_id'].astype('string')

        # Convert draft league_id to string dtype to match (CRITICAL for merge!)
        if draft_df['league_id'].dtype != 'string':
            draft_df['league_id'] = draft_df['league_id'].astype('string')

        # Get unique league_ids from draft (the leagues we need to match)
        draft_leagues = set(draft_df['league_id'].dropna().unique()) - {'<NA>'}

        # Get unique non-NA league_ids from player data
        player_leagues = set(player_df['league_id'].dropna().unique()) - {'<NA>'}

        print(f"  Draft leagues: {draft_leagues}")
        print(f"  Player leagues with data: {player_leagues}")

        # Check if player data covers ALL draft leagues
        if player_leagues and draft_leagues.issubset(player_leagues):
            # Player data covers all draft leagues - safe to use league_id for join
            use_league_id_for_join = True
            print(f"  [INFO] Player data covers all draft leagues - using league_id for join")
        elif player_leagues:
            # Player data has SOME league_ids but not all - DON'T use league_id for join
            # This would cause 0 matches for leagues not in player data
            missing = draft_leagues - player_leagues
            print(f"  [WARNING] Player data missing leagues: {missing}")
            print(f"  [INFO] Using yahoo_player_id + year ONLY for join (ignoring league_id)")
            use_league_id_for_join = False
        else:
            # Player data has ALL NA league_ids - don't use for join
            print(f"  [INFO] Player data has all-NA league_ids - NOT using league_id for join")
            use_league_id_for_join = False

    # OPTIMIZED: Pre-aggregate player stats by (yahoo_player_id, year, [league_id])
    print("  Aggregating player stats by season...")

    # Determine points column
    points_col = 'fantasy_points' if 'fantasy_points' in player_df.columns else 'points'

    # Group by keys (only include league_id if it has meaningful values)
    group_keys = ['yahoo_player_id', 'year']
    if use_league_id_for_join:
        group_keys.append('league_id')

    # Aggregate stats
    agg_dict = {}

    # Position (most common)
    if 'position' in player_df.columns:
        agg_dict['position'] = ('position', lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # NFL team (most common)
    if 'nfl_team' in player_df.columns:
        agg_dict['nfl_team'] = ('nfl_team', lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # Headshot URL (first non-null value - should be constant per player)
    if 'headshot_url' in player_df.columns:
        agg_dict['headshot_url'] = ('headshot_url', lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None)

    # Performance metrics
    if points_col in player_df.columns:
        # Count unique weeks if week column exists, otherwise count rows with points
        if 'week' in player_df.columns:
            # Count unique weeks where player had a record (games actually played)
            games_played_func = ('week', 'nunique')
            games_with_points_func = (points_col, lambda x: (x > 0).sum())
        else:
            # Fallback: count rows (for non-weekly data)
            games_played_func = (points_col, lambda x: x.notna().sum())
            games_with_points_func = (points_col, lambda x: (x > 0).sum())

        agg_dict.update({
            'total_fantasy_points': (points_col, lambda x: x.fillna(0).sum()),
            'games_played': games_played_func,
            'games_with_points': games_with_points_func,
            'season_ppg': (points_col, lambda x: x.mean() if len(x) > 0 else 0.0),
            'season_std': (points_col, lambda x: x.std() if len(x) > 1 else 0.0),
            'best_game': (points_col, lambda x: x.max() if len(x) > 0 else 0.0),
            'worst_game': (points_col, lambda x: x.min() if len(x) > 0 else 0.0),
        })

    # Roster metrics
    if 'is_rostered' in player_df.columns:
        agg_dict['weeks_rostered'] = ('is_rostered', lambda x: x.sum())
    if 'is_started' in player_df.columns:
        agg_dict['weeks_started'] = ('is_started', lambda x: x.sum())

    # SPAR metrics (season totals from weekly SPAR)
    # JOIN from player table instead of recalculating
    if 'player_spar' in player_df.columns:
        agg_dict['player_spar'] = ('player_spar', lambda x: x.fillna(0).sum())
    if 'manager_spar' in player_df.columns:
        agg_dict['manager_spar'] = ('manager_spar', lambda x: x.fillna(0).sum())
    if 'replacement_ppg' in player_df.columns:
        # Replacement PPG: use season average (first non-null value)
        agg_dict['replacement_ppg'] = ('replacement_ppg', lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 0.0)

    # Aggregate
    metrics_df = player_df.groupby(group_keys, as_index=False).agg(**agg_dict)

    print(f"  Aggregated to {len(metrics_df):,} unique player-year combinations")

    # CRITICAL: Ensure metrics_df has no duplicate keys before merge
    # This prevents duplicate draft rows after the join
    duplicates_before = metrics_df.duplicated(subset=group_keys, keep=False).sum()
    if duplicates_before > 0:
        print(f"  WARNING: Found {duplicates_before} duplicate keys in metrics_df, removing...")
        metrics_df = metrics_df.drop_duplicates(subset=group_keys, keep='first')
        print(f"  Deduplicated to {len(metrics_df):,} unique player-year combinations")

    # CRITICAL: Normalize data types before merge to prevent type mismatch errors
    # yahoo_player_id can be int64 in draft data but object/string in player data (or vice versa)
    # Convert both to string via Int64 intermediate to avoid .0 suffix from floats
    print("  Normalizing data types for merge keys...")
    for key in group_keys:
        # Skip league_id (already string)
        if key == 'league_id':
            continue
        if key in draft_df.columns:
            draft_df[key] = pd.to_numeric(draft_df[key], errors='coerce').astype('Int64')
        if key in metrics_df.columns:
            metrics_df[key] = pd.to_numeric(metrics_df[key], errors='coerce').astype('Int64')

    # CRITICAL: Drop columns from draft_df that will come from metrics_df to avoid _x/_y suffixes
    # This makes the pipeline idempotent (can run multiple times without creating duplicates)
    metrics_cols = set(metrics_df.columns) - set(group_keys)
    existing_overlap = [c for c in metrics_cols if c in draft_df.columns]
    if existing_overlap:
        print(f"  Dropping {len(existing_overlap)} existing columns to avoid duplicates: {existing_overlap[:5]}{'...' if len(existing_overlap) > 5 else ''}")
        draft_df = draft_df.drop(columns=existing_overlap)

    # Join performance metrics to draft data
    print(f"  Joining performance metrics to draft data on keys {group_keys}...")
    enriched_df = draft_df.merge(
        metrics_df,
        on=group_keys,
        how='left',
        validate='many_to_one'  # Ensure metrics_df has unique keys (one row per player-year)
    )
    # Check match rate by seeing if total_fantasy_points was populated
    matched = enriched_df['total_fantasy_points'].notna().sum() if 'total_fantasy_points' in enriched_df.columns else 0
    print(f"  [JOIN] {matched:,}/{len(enriched_df):,} draft picks matched with player stats ({100*matched/len(enriched_df) if len(enriched_df) > 0 else 0:.1f}%)")

    # Calculate ranking metrics
    print("Calculating ranking metrics...")
    enriched_df = calculate_ranking_metrics(enriched_df, player_df)

    # Calculate ROI metrics
    print("Calculating ROI metrics...")
    enriched_df = calculate_roi_metrics(enriched_df)

    return enriched_df


# =========================================================
# Main Transformation
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enrich draft data with player season performance stats",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--context', type=Path, required=True,
                        help='Path to league_context.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without saving')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before overwriting')

    args = parser.parse_args()

    # Load league context
    try:
        ctx = LeagueContext.load(args.context)
        print(f"Loaded league context: {ctx.league_name} ({ctx.league_id})")
    except Exception as e:
        print(f"ERROR: Failed to load league context: {e}")
        return 1

    # Use canonical draft file (created by normalize_draft_parquet)
    draft_file = ctx.canonical_draft_file

    if not draft_file.exists():
        print(f"ERROR: Draft file not found: {draft_file}")
        print(f"  Make sure normalize_draft_parquet has run first")
        return 1

    print(f"Using draft data: {draft_file}")

    # Locate player data
    player_file = ctx.canonical_player_file

    if not player_file.exists():
        print(f"ERROR: Player file not found: {player_file}")
        return 1

    # Load data
    print(f"\nLoading draft data from: {draft_file}")
    draft_df = pd.read_parquet(draft_file)
    print(f"  Loaded {len(draft_df):,} draft picks")

    print(f"\nLoading player data from: {player_file}")
    player_df = pd.read_parquet(player_file)
    print(f"  Loaded {len(player_df):,} player records")

    # Enrich draft data
    print("\n" + "="*60)
    print("ENRICHING DRAFT DATA WITH PLAYER STATS")
    print("="*60)

    enriched_df = enrich_draft_with_player_stats(draft_df, player_df)

    # Report new columns
    new_cols = [c for c in enriched_df.columns if c not in draft_df.columns]
    print(f"\nAdded {len(new_cols)} new columns:")
    for col in new_cols:
        non_null = enriched_df[col].notna().sum()
        pct = (non_null / len(enriched_df)) * 100
        print(f"  - {col}: {non_null:,} non-null ({pct:.1f}%)")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    if 'total_fantasy_points' in enriched_df.columns:
        print(f"\nTotal Fantasy Points:")
        print(f"  Mean: {enriched_df['total_fantasy_points'].mean():.2f}")
        print(f"  Median: {enriched_df['total_fantasy_points'].median():.2f}")
        print(f"  Max: {enriched_df['total_fantasy_points'].max():.2f}")

    if 'points_per_dollar' in enriched_df.columns:
        ppd_valid = enriched_df['points_per_dollar'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ppd_valid) > 0:
            print(f"\nPoints Per Dollar:")
            print(f"  Mean: {ppd_valid.mean():.2f}")
            print(f"  Median: {ppd_valid.median():.2f}")
            print(f"  Top 5:")
            top5 = enriched_df.nlargest(5, 'points_per_dollar')[['player', 'cost', 'total_fantasy_points', 'points_per_dollar']]
            print(top5.to_string(index=False))

    if 'price_rank_vs_finish_rank' in enriched_df.columns:
        pvf_valid = enriched_df['price_rank_vs_finish_rank'].dropna()
        if len(pvf_valid) > 0:
            print(f"\nPrice Rank vs Finish Rank (positive = outperformed):")
            print(f"  Mean: {pvf_valid.mean():.2f}")
            print(f"  Top 5 Overperformers:")
            top5 = enriched_df.nlargest(5, 'price_rank_vs_finish_rank')[
                ['player', 'position', 'price_rank_within_position', 'season_position_rank', 'price_rank_vs_finish_rank']
            ]
            print(top5.to_string(index=False))

    # Save or preview
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - No files modified")
        print("="*60)
        return 0

    # Backup if requested
    if args.backup:
        backup_path = backup_file(draft_file)
        print(f"\nCreated backup: {backup_path}")

    # Save enriched data to source location
    print(f"\nSaving enriched draft data to: {draft_file}")
    enriched_df.to_parquet(draft_file, index=False, engine='pyarrow')
    print(f"[SAVED] {len(enriched_df):,} rows to draft_data directory")

    # IMPORTANT: Also save to canonical location (used by the app)
    canonical_draft_file = ctx.canonical_draft_file
    print(f"\nSaving to canonical location: {canonical_draft_file}")
    enriched_df.to_parquet(canonical_draft_file, index=False, engine='pyarrow')
    print(f"[SAVED] Canonical draft.parquet")

    # Also save CSV versions in both locations
    csv_file = draft_file.with_suffix('.csv')
    print(f"\nSaving CSV version to: {csv_file}")
    enriched_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"[SAVED] CSV to draft_data directory")

    canonical_csv_file = canonical_draft_file.with_suffix('.csv')
    enriched_df.to_csv(canonical_csv_file, index=False, encoding='utf-8-sig')
    print(f"[SAVED] CSV to canonical location")

    print("\n" + "="*60)
    print("DRAFT ENRICHMENT COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
