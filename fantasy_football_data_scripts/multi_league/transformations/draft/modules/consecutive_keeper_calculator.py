"""
Consecutive Keeper Years Calculator

Calculates how many consecutive years each player has been kept based on:
- yahoo_player_id: Unique player identifier
- is_keeper_status: Whether player was kept that year (1=kept, 0=drafted fresh)
- year: Season year

Output columns:
- consecutive_years_kept: Number of consecutive years this player has been kept (0 if not a keeper)
- first_kept_year: First year in the current keeper streak
- keeper_streak_id: Unique identifier for each keeper streak (resets if player is re-drafted)

Usage:
    from modules.consecutive_keeper_calculator import calculate_consecutive_keeper_years

    df = calculate_consecutive_keeper_years(draft_df)
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_consecutive_keeper_years(
    df: pd.DataFrame,
    player_id_col: str = 'yahoo_player_id',
    keeper_col: str = 'is_keeper_status',
    year_col: str = 'year',
    manager_col: Optional[str] = 'manager',
) -> pd.DataFrame:
    """
    Calculate consecutive years kept for each player.

    Args:
        df: DataFrame with draft data
        player_id_col: Column name for player ID
        keeper_col: Column name for keeper status (1=kept, 0=drafted)
        year_col: Column name for year
        manager_col: Optional column for manager (tracks keeper streaks per manager)

    Returns:
        DataFrame with added columns:
        - consecutive_years_kept: 0 if not keeper, 1+ if keeper (increments each consecutive year)
        - first_kept_year: First year of current keeper streak (null if not keeper)
        - keeper_streak_id: Unique ID for each keeper streak
    """
    if df.empty:
        df['consecutive_years_kept'] = 0
        df['first_kept_year'] = pd.NA
        df['keeper_streak_id'] = pd.NA
        return df

    # Make a copy to avoid modifying original
    result = df.copy()

    # Ensure required columns exist
    if player_id_col not in result.columns:
        print(f"[WARNING] Column '{player_id_col}' not found, cannot calculate keeper years")
        result['consecutive_years_kept'] = 0
        result['first_kept_year'] = pd.NA
        result['keeper_streak_id'] = pd.NA
        return result

    if keeper_col not in result.columns:
        print(f"[WARNING] Column '{keeper_col}' not found, assuming no keepers")
        result['consecutive_years_kept'] = 0
        result['first_kept_year'] = pd.NA
        result['keeper_streak_id'] = pd.NA
        return result

    # Initialize output columns
    result['consecutive_years_kept'] = 0
    result['first_kept_year'] = pd.NA
    result['keeper_streak_id'] = pd.NA

    # Sort by player and year for proper sequential processing
    result = result.sort_values([player_id_col, year_col]).reset_index(drop=True)

    # Track keeper history per player (and optionally per manager)
    # Structure: {(player_id, manager): {'streak_count': N, 'first_year': Y, 'streak_id': S}}
    keeper_history = {}
    streak_counter = 0

    for idx, row in result.iterrows():
        player_id = row.get(player_id_col)
        year = row.get(year_col)
        is_keeper = bool(row.get(keeper_col, 0))
        manager = row.get(manager_col) if manager_col and manager_col in result.columns else None

        if pd.isna(player_id) or pd.isna(year):
            continue

        # Create key for tracking (include manager if tracking per-manager streaks)
        if manager_col and manager is not None and not pd.isna(manager):
            key = (str(player_id), str(manager))
        else:
            key = (str(player_id), None)

        year = int(year)

        if key not in keeper_history:
            keeper_history[key] = {
                'last_year': None,
                'streak_count': 0,
                'first_year': None,
                'streak_id': None
            }

        history = keeper_history[key]

        if is_keeper:
            # Check if this is a continuation of a streak
            if history['last_year'] is not None and history['last_year'] == year - 1:
                # Consecutive keeper - increment streak
                history['streak_count'] += 1
            else:
                # New keeper streak (first time kept or gap in keeping)
                streak_counter += 1
                history['streak_count'] = 1
                history['first_year'] = year
                history['streak_id'] = streak_counter

            history['last_year'] = year

            result.at[idx, 'consecutive_years_kept'] = history['streak_count']
            result.at[idx, 'first_kept_year'] = history['first_year']
            result.at[idx, 'keeper_streak_id'] = history['streak_id']
        else:
            # Not a keeper this year - reset streak
            history['streak_count'] = 0
            history['first_year'] = None
            history['streak_id'] = None
            history['last_year'] = year  # Still track that player was on roster

            result.at[idx, 'consecutive_years_kept'] = 0
            result.at[idx, 'first_kept_year'] = pd.NA
            result.at[idx, 'keeper_streak_id'] = pd.NA

    # Convert types
    result['consecutive_years_kept'] = result['consecutive_years_kept'].astype(int)

    return result


def get_keeper_streak_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of all keeper streaks.

    Returns DataFrame with columns:
    - keeper_streak_id
    - yahoo_player_id
    - player (name)
    - manager
    - first_kept_year
    - last_kept_year
    - total_years_kept
    """
    if 'keeper_streak_id' not in df.columns or df['keeper_streak_id'].isna().all():
        return pd.DataFrame()

    keepers = df[df['consecutive_years_kept'] > 0].copy()

    if keepers.empty:
        return pd.DataFrame()

    summary = keepers.groupby('keeper_streak_id').agg({
        'yahoo_player_id': 'first',
        'player': 'first',
        'manager': 'first',
        'first_kept_year': 'first',
        'year': ['min', 'max', 'count'],
        'consecutive_years_kept': 'max'
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'keeper_streak_id', 'yahoo_player_id', 'player', 'manager',
        'first_kept_year', 'min_year', 'last_kept_year', 'years_in_data',
        'max_consecutive_years'
    ]

    return summary.sort_values('max_consecutive_years', ascending=False)
