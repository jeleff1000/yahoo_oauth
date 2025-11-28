"""
Matchup Rankings Module

Calculates manager-specific rankings and percentiles for matchup data.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.

Rankings calculated:
- manager_all_time_ranking: Rank of this week's performance vs all weeks in manager's history (1 = best ever)
- manager_all_time_percentile: Percentile rank (higher is better)
- manager_season_ranking: Rank of this week vs all weeks in this season (1 = best this season)
"""

from functools import wraps
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directories to path for imports
_script_file = Path(__file__).resolve()
_modules_dir = _script_file.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))

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
            if league_id is not None and pd.notna(league_id):
                result = ensure_league_id(result, league_id)

        return result

    return wrapper


@ensure_normalized
def calculate_manager_all_time_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate manager all-time rankings and percentiles.

    RECALCULATE WEEKLY: Ranks each week's performance against all weeks in manager's history.

    Adds:
    - manager_all_time_ranking: Rank within manager's all-time performances (1 = best ever)
    - manager_all_time_percentile: Percentile rank (0-100, higher is better)

    Args:
        df: DataFrame with matchup data

    Returns:
        DataFrame with ranking columns added
    """
    df = df.copy()

    # Find points column
    pts_col = next((c for c in ['team_points', 'points_for', 'pf'] if c in df.columns), None)

    if pts_col is None:
        print("  [WARN] No points column found, setting manager_all_time_ranking to NA")
        df['manager_all_time_ranking'] = pd.NA
        df['manager_all_time_percentile'] = pd.NA
        return df

    # Ensure points are numeric
    df[pts_col] = pd.to_numeric(df[pts_col], errors='coerce')

    # Rank within each manager's entire history (1 = best ever performance)
    # Use method='min' so ties get the same best rank
    ranks = (
        df.groupby('manager')[pts_col]
        .rank(method='min', ascending=False)
        .astype('Int64')
    )
    df['manager_all_time_ranking'] = ranks

    # Calculate percentile
    # Number of non-null historical records per manager
    counts = df.groupby('manager')[pts_col].transform('count').astype(float)

    # Percentile: higher is better
    # Formula: (count - rank) / (count - 1) * 100
    # Special case: if only one record, set percentile to 100.0
    pct = ((counts - ranks.astype(float)) / (counts - 1.0)).where(counts > 1, 1.0) * 100.0
    df['manager_all_time_percentile'] = pct.round(2)

    print(f"  Added manager_all_time_ranking and manager_all_time_percentile")

    return df


@ensure_normalized
def calculate_manager_season_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate manager season rankings.

    RECALCULATE WEEKLY: Ranks each week's performance against all weeks in that season.
    Uses FULL SEASON data (look-ahead bias) - all weeks are ranked against each other.

    Adds:
    - manager_season_ranking: Rank within manager-season (1 = best week this season)

    Args:
        df: DataFrame with matchup data

    Returns:
        DataFrame with ranking column added
    """
    df = df.copy()

    # Find points column
    pts_col = next((c for c in ['team_points', 'points_for', 'pf'] if c in df.columns), None)

    if pts_col is None:
        print("  [WARN] No points column found, setting manager_season_ranking to NA")
        df['manager_season_ranking'] = pd.NA
        return df

    # Ensure points are numeric
    df[pts_col] = pd.to_numeric(df[pts_col], errors='coerce')

    # Rank within each manager's season (1 = best week this season)
    # Use method='dense' for consistent ranking even with ties
    # FULL SEASON: Ranks all weeks in the season against each other
    df['manager_season_ranking'] = (
        df.groupby(['manager', 'year'])[pts_col]
        .rank(method='dense', ascending=False)
        .astype('Int64')
    )

    print(f"  Added manager_season_ranking (FULL SEASON - look-ahead)")

    return df


@ensure_normalized
def calculate_all_matchup_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all manager-specific rankings and percentiles.

    Convenience function that calculates:
    - manager_all_time_ranking + percentile
    - manager_season_ranking

    Args:
        df: DataFrame with matchup data

    Returns:
        DataFrame with all ranking columns added
    """
    df = df.copy()

    print("  Calculating manager all-time ranking...")
    df = calculate_manager_all_time_ranking(df)

    print("  Calculating manager season ranking...")
    df = calculate_manager_season_ranking(df)

    return df
