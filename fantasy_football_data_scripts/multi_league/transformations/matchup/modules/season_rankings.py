
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
            if league_id:
                result = ensure_league_id(result, league_id)
        
        return result
    
    return wrapper

"""
Season Rankings Module

Calculates season and all-time rankings.

MIXED:
- SET-AND-FORGET: Season rankings (after championship)
- RECALCULATE WEEKLY: All-time rankings
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
import pandas as pd
import numpy as np
@ensure_normalized
def calculate_season_rankings(
    df: pd.DataFrame,
    championship_complete: bool = False
) -> pd.DataFrame:
    """
    Calculate season-level rankings and final records.
    SET-AND-FORGET (only if championship_complete=True):
    - final_wins: Total wins for the season
    - final_losses: Total losses for the season
    - final_regular_wins: Regular season wins only
    - final_regular_losses: Regular season losses only
    - season_mean: Average points per game (season)
    - season_median: Median points per game (season)
    RECALCULATE WEEKLY:
    - manager_season_ranking: Ranks this week's points against all other weeks
                              for this manager in this season (1 = best week)
    Args:
        df: DataFrame with season data
        championship_complete: If True, calculate SET-AND-FORGET columns
    Returns:
        DataFrame with season ranking columns added
    """
    df = df.copy()
    # We always compute season aggregates so fields are never blank.
    # If championship_complete is False, treat these as provisional.
    print(
        "  Calculating season rankings "
        + ("(finalized: championship complete)" if championship_complete else "(provisional: championship not complete)")
    )

    # Use franchise_id for grouping if available (handles multi-team managers correctly)
    # Falls back to manager name for backwards compatibility
    group_col = 'franchise_id' if 'franchise_id' in df.columns and df['franchise_id'].notna().any() else 'manager'

    # Calculate final season records per franchise/manager
    season_stats = df.groupby(['year', group_col]).agg({
        'win': 'sum',
        'loss': 'sum',
        'team_points': ['mean', 'median', 'sum']
    }).reset_index()
    season_stats.columns = ['year', group_col, 'total_wins', 'total_losses',
                            'season_mean', 'season_median', 'season_total_points']
    # Regular season only (exclude playoffs and consolation)
    regular_mask = (df['is_playoffs'].fillna(0) != 1) & (df['is_consolation'].fillna(0) != 1)
    regular_stats = df[regular_mask].groupby(['year', group_col]).agg({
        'win': 'sum',
        'loss': 'sum'
    }).reset_index()
    regular_stats.columns = ['year', group_col, 'final_regular_wins', 'final_regular_losses']

    # Drop existing season stats columns to avoid merge conflicts
    existing_cols = ['total_wins', 'total_losses', 'season_mean', 'season_median', 'season_total_points',
                     'final_regular_wins', 'final_regular_losses', 'final_wins', 'final_losses']
    df = df.drop(columns=[c for c in existing_cols if c in df.columns], errors='ignore')

    # Merge back to main dataframe
    df = df.merge(season_stats, on=['year', group_col], how='left')
    df = df.merge(regular_stats, on=['year', group_col], how='left')
    # Rename for final columns
    df['final_wins'] = df['total_wins'].astype("Int64")
    df['final_losses'] = df['total_losses'].astype("Int64")
    df = df.drop(columns=['total_wins', 'total_losses', 'season_total_points'], errors='ignore')
    # manager_season_ranking: Rank each week's points within that franchise's season
    # 1 = best week for that franchise in that season
    # This is RECALCULATE WEEKLY - ranks current week against all prior weeks
    pts_col = next((c for c in ['team_points', 'points_for', 'pf'] if c in df.columns), None)
    if pts_col is not None:
        df[pts_col] = pd.to_numeric(df[pts_col], errors='coerce')
        df['manager_season_ranking'] = (
            df.groupby([group_col, 'year'])[pts_col]
              .rank(method='dense', ascending=False)
              .astype("Int64")
        )
    else:
        df['manager_season_ranking'] = pd.NA
    # Playoff outcome columns should always exist, default 0
    for col in ['champion', 'semifinal', 'quarterfinal', 'sacko']:
        if col not in df.columns:
            df[col] = 0
    if championship_complete:
        print(f"  Finalized season stats for {df['year'].nunique()} seasons (locked)")
        print(f"  NOTE: Set playoff outcomes via: df.loc[condition, 'champion'] = 1")
    else:
        print(f"  Provisional season stats computed for {df['year'].nunique()} seasons (championship not complete)")
    return df
@ensure_normalized
def calculate_alltime_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all-time rankings and percentiles.
    RECALCULATE WEEKLY columns:
    - manager_all_time_ranking: All-time rank across all managers (by total wins)
    - manager_all_time_ranking_percentile: Percentile rank
    - league_all_time_ranking: League-wide all-time rank
    - league_all_time_ranking_percentile: League percentile
    Args:
        df: DataFrame with historical data
    Returns:
        DataFrame with all-time ranking columns added
    """
    df = df.copy()

    # Use franchise_id for grouping if available (handles multi-team managers correctly)
    # Falls back to manager name for backwards compatibility
    group_col = 'franchise_id' if 'franchise_id' in df.columns and df['franchise_id'].notna().any() else 'manager'

    # Calculate all-time totals per franchise/manager
    alltime_stats = df.groupby(group_col).agg({
        'win': 'sum',
        'team_points': 'sum'
    }).reset_index()
    alltime_stats.columns = [group_col, 'alltime_wins', 'alltime_points']
    # Rank franchises/managers by all-time wins
    alltime_stats['manager_all_time_ranking'] = alltime_stats['alltime_wins'].rank(
        method='min',
        ascending=False
    ).astype(int)
    # Calculate percentile
    total_franchises = len(alltime_stats)
    alltime_stats['manager_all_time_ranking_percentile'] = (
        (1 - (alltime_stats['manager_all_time_ranking'] - 1) / total_franchises) * 100
    )
    # For league-wide, use same logic (could be different if multiple leagues)
    alltime_stats['league_all_time_ranking'] = alltime_stats['manager_all_time_ranking']
    alltime_stats['league_all_time_ranking_percentile'] = alltime_stats['manager_all_time_ranking_percentile']
    # Merge back
    df = df.merge(
        alltime_stats[[group_col, 'manager_all_time_ranking', 'manager_all_time_ranking_percentile',
                      'league_all_time_ranking', 'league_all_time_ranking_percentile']],
        on=group_col,
        how='left'
    )
    # Convert to proper types
    df['manager_all_time_ranking'] = df['manager_all_time_ranking'].astype("Int64")
    df['league_all_time_ranking'] = df['league_all_time_ranking'].astype("Int64")
    print(f"  All-time rankings calculated for {total_franchises} franchises/managers")
    return df