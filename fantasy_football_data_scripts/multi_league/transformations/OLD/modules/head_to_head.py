
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
Head-to-Head Records Module

Calculates manager vs manager historical records.

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
import re
POINTS_COL_CANDIDATES_SELF = ["team_points", "points_for", "pf"]
ID_YEAR = ["year", "week"]
def _pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None
def _mgr_token(name: str) -> str:
    """Convert manager name to token for column names."""
    s = str(name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"
@ensure_normalized
def calculate_head_to_head_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate head-to-head records between managers as weekly one-vs-all indicators.
    Uses vectorized operations for optimal performance.
    For each row (manager, year, week) we create columns:
    - w_vs_{opponent}: 1 iff this manager's points > opponent's points in the SAME week
    - l_vs_{opponent}: 1 iff this manager's points < opponent's points in the SAME week
    Self-comparison is forced to 0. Missing point comparisons become pd.NA.
    Args:
        df: DataFrame with at least the 'manager' column and a points column
    Returns:
        DataFrame with head-to-head weekly indicator columns added
    """
    df = df.copy()
    if 'manager' not in df.columns:
        raise ValueError("Missing required column: manager")
    pts_col = _pick_col(df, POINTS_COL_CANDIDATES_SELF)
    if pts_col is None:
        raise ValueError("Could not find team points column (team_points/points_for/pf)")
    # Ensure numeric points
    df[pts_col] = pd.to_numeric(df[pts_col], errors='coerce')
    # Ensure year/week exist for grouping; if missing, create sentinel values
    for c in ID_YEAR:
        if c not in df.columns:
            df[c] = -1
    # Unique managers and tokens
    managers = sorted(pd.unique(df['manager'].dropna()))
    tokens = {m: _mgr_token(m) for m in managers}
    # Create pivot table with all manager scores by week
    # This reshapes data so each row is a (year, week) and columns are managers
    pivot = df.pivot_table(
        index=['year', 'week'],
        columns='manager',
        values=pts_col,
        aggfunc='first'
    )
    # Reindex to ensure all managers are present
    pivot = pivot.reindex(columns=managers)
    # Initialize all H2H columns in the original dataframe
    for m in managers:
        tok = tokens[m]
        df[f'w_vs_{tok}'] = pd.NA
        df[f'l_vs_{tok}'] = pd.NA
    # Create a mapping from (year, week, manager) to original df index
    df['_temp_yw'] = list(zip(df['year'], df['week']))
    index_map = df.set_index(['year', 'week', 'manager']).index
    # Vectorized comparison for each manager
    for manager in managers:
        tok = tokens[manager]
        # Get this manager's scores as a column vector
        manager_scores = pivot[manager].values
        # Broadcast comparison against all opponents (vectorized)
        # Shape: (n_weeks, n_managers)
        wins = (manager_scores[:, np.newaxis] > pivot.values).astype(int)
        losses = (manager_scores[:, np.newaxis] < pivot.values).astype(int)
        # Handle NaN comparisons - set to pd.NA
        manager_has_score = pd.notna(manager_scores[:, np.newaxis])
        opponents_have_scores = pd.notna(pivot.values)
        valid_comparisons = manager_has_score & opponents_have_scores
        wins = np.where(valid_comparisons, wins, pd.NA)
        losses = np.where(valid_comparisons, losses, pd.NA)
        # Map results back to original dataframe
        for opp_idx, opponent in enumerate(managers):
            opp_tok = tokens[opponent]
            win_col = f'w_vs_{opp_tok}'
            loss_col = f'l_vs_{opp_tok}'
            if manager == opponent:
                # Self-comparison is always 0
                mask = df['manager'] == manager
                df.loc[mask, win_col] = 0
                df.loc[mask, loss_col] = 0
            else:
                # Map the vectorized results back to the original rows
                for week_idx, (year_week) in enumerate(pivot.index):
                    year, week = year_week
                    mask = (df['year'] == year) & (df['week'] == week) & (df['manager'] == manager)
                    if mask.any():
                        df.loc[mask, win_col] = wins[week_idx, opp_idx]
                        df.loc[mask, loss_col] = losses[week_idx, opp_idx]
    # Clean up temporary column
    df.drop(columns=['_temp_yw'], inplace=True)
    # Cast indicator columns to Int64
    for m in managers:
        tok = tokens[m]
        wcol = f'w_vs_{tok}'
        lcol = f'l_vs_{tok}'
        try:
            df[wcol] = df[wcol].astype('Int64')
        except Exception:
            pass
        try:
            df[lcol] = df[lcol].astype('Int64')
        except Exception:
            pass
    print(f"  Created {len(managers) * 2} weekly H2H indicator columns (vectorized)")
    return df