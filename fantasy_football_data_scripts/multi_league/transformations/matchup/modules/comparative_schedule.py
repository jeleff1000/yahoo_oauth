"""
Comparative Schedule Module

Calculates schedule strength comparisons by simulating what would happen
if you played each manager's schedule.

RECALCULATE WEEKLY: All columns in this module must be recalculated every week.
"""

from functools import wraps
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

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


def _mgr_token(name: str) -> str:
    """Convert manager name to token for column names."""
    s = str(name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"


@ensure_normalized
def calculate_comparative_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate W/L vs each manager's schedule.

    This answers: "If I played Manager X's schedule, how would I do?"
    For each manager's schedule, we compare your points against their opponents' points.

    RECALCULATE WEEKLY columns:
    - w_vs_{manager}_sched: Would you have won this week if you played this manager's opponent?
    - l_vs_{manager}_sched: Would you have lost this week if you played this manager's opponent?

    Args:
        df: DataFrame with manager, year, week, team_points, opponent_points

    Returns:
        DataFrame with w_vs_{manager}_sched, l_vs_{manager}_sched columns added (0 or 1 per week)
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ['manager', 'year', 'week', 'team_points', 'opponent_points']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert to numeric
    df['team_points'] = pd.to_numeric(df['team_points'], errors='coerce')
    df['opponent_points'] = pd.to_numeric(df['opponent_points'], errors='coerce')

    # Get all managers
    mgr_all = sorted(df['manager'].dropna().unique())
    mgr_tokens = {m: _mgr_token(m) for m in mgr_all}

    # Create a map of (manager, year, week) -> opponent_points
    # This tells us who each manager played each week
    opp_points_map = df.set_index(['manager', 'year', 'week'])['opponent_points'].to_dict()

    # For each manager, create w_vs_{manager}_sched and l_vs_{manager}_sched columns
    for m in mgr_all:
        tok = mgr_tokens[m]

        # For each row in df, get this manager's opponent's points for that (year, week)
        # Then compare row's team_points against that opponent's points
        aligned_sched = np.array([
            opp_points_map.get((m, y, w), np.nan)
            for y, w in zip(df['year'], df['week'])
        ])

        # Win/loss indicators (0 or 1 per week, not cumulative)
        df[f"w_vs_{tok}_sched"] = (df['team_points'].to_numpy() > aligned_sched).astype(int)
        df[f"l_vs_{tok}_sched"] = (df['team_points'].to_numpy() < aligned_sched).astype(int)

    print(f"  Created {len(mgr_all) * 2} comparative schedule columns (w_vs_X_sched, l_vs_X_sched)")

    return df
