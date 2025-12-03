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
Weekly Metrics Module

Calculates league-relative weekly performance metrics.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.
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
def calculate_weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekly league-relative metrics.
    RECALCULATE WEEKLY columns:
    - teams_beat_this_week: How many league teams would you have beaten?
    - above_league_median: Did you score above league median?
    - weekly_rank: Your rank this week within the league
    - league_weekly_mean: League average points this week
    - league_weekly_median: League median points this week
    Args:
        df: DataFrame with team_points, year, week
    Returns:
        DataFrame with weekly metrics added
    """
    df = df.copy()
    # Ensure required columns exist
    for col in ['year', 'week', 'team_points']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Convert team_points to float - PRESERVE NULLS (don't fill with 0.0)
    # Null scores occur in special cases like KMFFL 2013 playoffs where scores were lost to history
    df['team_points'] = pd.to_numeric(df['team_points'], errors='coerce')
    # Calculate league-wide metrics per week (skipna=True to exclude null scores)
    df['league_weekly_mean'] = df.groupby(['year', 'week'])['team_points'].transform(lambda x: x.mean(skipna=True))
    df['league_weekly_median'] = df.groupby(['year', 'week'])['team_points'].transform(lambda x: x.median(skipna=True))
    # Above/below league median - handle nulls properly
    # For rows with null team_points, result should be null (can't compare)
    df['above_league_median'] = (
        df['team_points'] > df['league_weekly_median']
    ).where(df['team_points'].notna(), pd.NA).astype("Int64")
    df['below_league_median'] = (
        df['team_points'] < df['league_weekly_median']
    ).where(df['team_points'].notna(), pd.NA).astype("Int64")
    # Calculate opponent's season median for comparison
    # Join opponent's median scores to compare manager vs their opponent's typical performance
    if 'opponent_week' in df.columns:
        # Calculate each manager's season median up to current week
        manager_medians = df.groupby(['manager', 'year'], group_keys=False).apply(
            lambda g: g.sort_values('week').assign(
                manager_cumulative_median=lambda x: x['team_points'].expanding().median()
            ),
            include_groups=False
        )[['manager_week', 'manager_cumulative_median']].drop_duplicates()
        # Rename to match opponent_week for joining
        opponent_medians = manager_medians.rename(columns={
            'manager_week': 'opponent_week',
            'manager_cumulative_median': 'opponent_median'
        })
        # Drop opponent_median if it already exists (from previous run)
        if 'opponent_median' in df.columns:
            df = df.drop(columns=['opponent_median'])
        # Merge opponent's median
        df = df.merge(opponent_medians, on='opponent_week', how='left')
        # Above/below opponent median - only if merge was successful
        if 'opponent_median' in df.columns:
            df['above_opponent_median'] = (
                df['team_points'] > df['opponent_median']
            ).astype("Int64")
            df['below_opponent_median'] = (
                df['team_points'] < df['opponent_median']
            ).astype("Int64")
        else:
            print(f"  [WARN] opponent_median column not created after merge (no matching opponent_week keys)")
            df['above_opponent_median'] = pd.NA
            df['below_opponent_median'] = pd.NA
        print(f"  Added opponent median comparison columns")
    else:
        print(f"  [WARN] Cannot create opponent median columns: missing opponent_week")
        df['above_opponent_median'] = pd.NA
        df['below_opponent_median'] = pd.NA
    # Teams beat this week (how many teams in the league would you have beaten?)
    def count_teams_beat(group):
        # For each team's score, count how many other scores it beats
        # Handle null scores: if score is null, teams_beat should be 0 (can't compare)
        scores = group['team_points'].values
        teams_beat = []
        for score in scores:
            # If score is null/NaN, can't beat anyone
            if pd.isna(score):
                teams_beat.append(0)
            else:
                # Count non-null scores less than this score
                beaten = np.nansum(scores < score)
                teams_beat.append(beaten)
        return pd.Series(teams_beat, index=group.index)
    df['teams_beat_this_week'] = df.groupby(['year', 'week'], group_keys=False).apply(
        count_teams_beat,
        include_groups=False
    ).astype("Int64")
    # Weekly rank (1 = highest score that week)
    # na_option='keep' preserves null ranks for null scores
    df['weekly_rank'] = df.groupby(['year', 'week'])['team_points'].rank(
        method='min',
        ascending=False,
        na_option='keep'
    ).astype("Int64")

    # Last place week (1 = lowest score that week)
    # Manager has last_place_week=1 if their weekly_rank equals the max rank for that week
    max_rank_per_week = df.groupby(['year', 'week'])['weekly_rank'].transform('max')
    df['last_place_week'] = (df['weekly_rank'] == max_rank_per_week).astype("Int64")
    # Handle null weekly_rank (null scores can't be last place)
    df.loc[df['weekly_rank'].isna(), 'last_place_week'] = pd.NA

    print(f"  League mean range: {df['league_weekly_mean'].min():.1f} - {df['league_weekly_mean'].max():.1f}")
    print(f"  Last place weeks: {df['last_place_week'].sum()}")
    print(f"  Max teams beaten in a week: {df['teams_beat_this_week'].max()}")
    # Ensure weekly columns exist and are null-safe for downstream merges
    for col, dtype in [("weekly_rank","Int64"),("teams_beat_this_week","Int64"),("above_league_median","Int64")]:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)
    return df