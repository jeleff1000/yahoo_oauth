"""
All-Play Extended Module

Extends weekly_metrics.py with opponent-specific all-play metrics.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.

NOTE: weekly_metrics.py already calculates teams_beat_this_week.
This module adds opponent_teams_beat_this_week.
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
def calculate_opponent_all_play(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate opponent's all-play record this week.

    RECALCULATE WEEKLY column:
    - opponent_teams_beat_this_week: How many teams opponent would have beaten

    Args:
        df: DataFrame with year, week, opponent_points columns

    Returns:
        DataFrame with opponent_teams_beat_this_week column added
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ['year', 'week', 'opponent_points']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert opponent_points to float
    df['opponent_points'] = pd.to_numeric(df['opponent_points'], errors='coerce')

    # Initialize column
    df['opponent_teams_beat_this_week'] = 0

    # For each week, calculate how many teams each opponent would have beaten
    for (y, w), g in df.groupby(['year', 'week'], sort=False):
        # Get all scores this week (unique by manager)
        if 'manager' in df.columns:
            scores = g.groupby('manager')['team_points'].first().to_dict()
        else:
            scores = g['team_points'].to_dict()

        for idx, row in g.iterrows():
            opp_score = row['opponent_points']

            # Count how many teams in the league this opponent's score would beat
            if pd.notna(opp_score):
                opp_teams_beat = sum(1 for s in scores.values() if pd.notna(s) and opp_score > s)
            else:
                opp_teams_beat = 0

            df.at[idx, 'opponent_teams_beat_this_week'] = opp_teams_beat

    print(f"  Calculated opponent all-play metrics")
    print(f"  Max opponent teams beaten: {df['opponent_teams_beat_this_week'].max()}")

    return df
