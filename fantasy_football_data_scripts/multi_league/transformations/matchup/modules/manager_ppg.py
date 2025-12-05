"""
Manager PPG Module

Calculates manager/team points-per-game metrics.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week
as new games are added to the season.

NOTE: This is different from ppg_calculator.py, which calculates PLAYER stats.
This module calculates MANAGER/TEAM weekly averages.
"""

from functools import wraps
import sys
from pathlib import Path
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


@ensure_normalized
def calculate_manager_ppg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate manager weekly mean and median PPG.

    RECALCULATE WEEKLY: These are running season-to-date averages,
    so they change every week as new games are added.

    Args:
        df: DataFrame with manager, year, week, team_points columns

    Returns:
        DataFrame with weekly_mean and weekly_median columns added
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ['manager', 'year', 'week', 'team_points']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert team_points to float
    df['team_points'] = pd.to_numeric(df['team_points'], errors='coerce')

    # Use franchise_id for grouping if available (handles multi-team managers correctly)
    # Falls back to manager name for backwards compatibility
    group_col = 'franchise_id' if 'franchise_id' in df.columns and df['franchise_id'].notna().any() else 'manager'

    # Calculate weekly mean/median per franchise for each week in the season
    # This is the franchise's season-to-date average up to and including current week
    weekly_stats = df.groupby([group_col, 'year', 'week'])['team_points'].first()

    # For each row, calculate the mean/median of all weeks up to current week
    df['weekly_mean'] = df.apply(
        lambda r: weekly_stats.loc[(r[group_col], r['year'], slice(None))].loc[:r['week']].mean()
        if (r[group_col], r['year'], r['week']) in weekly_stats.index else np.nan,
        axis=1
    )

    df['weekly_median'] = df.apply(
        lambda r: weekly_stats.loc[(r[group_col], r['year'], slice(None))].loc[:r['week']].median()
        if (r[group_col], r['year'], r['week']) in weekly_stats.index else np.nan,
        axis=1
    )

    print(f"  Calculated franchise PPG metrics (weekly_mean, weekly_median)")
    print(f"  Mean range: {df['weekly_mean'].min():.1f} - {df['weekly_mean'].max():.1f}")

    return df
