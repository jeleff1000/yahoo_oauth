"""
DataFrame utility functions for cleaning and transforming data.

This module centralizes common DataFrame operations that were previously
duplicated across many files.
"""

import pandas as pd
from typing import List, Dict, Optional


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe by removing duplicate columns and ensuring proper types.

    This function should be used EVERYWHERE instead of inline cleanup code.

    Args:
        df: DataFrame to clean

    Returns:
        Cleaned DataFrame

    Example:
        ```python
        from streamlit_ui.shared.dataframe_utils import clean_dataframe

        df = load_data()
        df = clean_dataframe(df)  # Remove duplicates, fix types
        ```
    """
    if df is None or df.empty:
        return df

    # Remove duplicate columns (keep first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Ensure all column names are strings
    df.columns = [str(col) for col in df.columns]

    return df


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Ensure specified columns are numeric types.

    Args:
        df: DataFrame to process
        columns: List of column names to convert to numeric

    Returns:
        DataFrame with specified columns as numeric

    Example:
        ```python
        df = ensure_numeric(df, ['points', 'yards', 'touchdowns'])
        ```
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def apply_common_renames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard column renames used across all views.

    This ensures consistent column naming throughout the app.

    Args:
        df: DataFrame to rename columns in

    Returns:
        DataFrame with renamed columns

    Example:
        ```python
        df = apply_common_renames(df)
        # 'player' -> 'Player', 'points' -> 'Points', etc.
        ```
    """
    # Common renames used across all views
    renames = {
        # Core identification
        'player': 'Player',
        'nfl_team': 'Team',
        'opponent_nfl_team': 'Vs',
        'manager': 'Manager',
        'opponent': 'Opponent',

        # Time dimensions
        'week': 'Week',
        'year': 'Year',

        # Positions
        'nfl_position': 'Position',
        'fantasy_position': 'Roster Slot',
        'lineup_position': 'Lineup Slot',

        # Points and scoring
        'points': 'Points',
        'fantasy_points': 'Fantasy Points',
        'spar': 'SPAR',
        'player_spar': 'Player SPAR',
        'manager_spar': 'Manager SPAR',

        # Passing stats
        'passing_yards': 'Pass Yds',
        'passing_tds': 'Pass TD',
        'passing_interceptions': 'Pass INT',
        'completions': 'Comp',
        'attempts': 'Pass Att',
        'passing_epa': 'Pass EPA',
        'passing_cpoe': 'CPOE',
        'pacr': 'PACR',

        # Rushing stats
        'rushing_yards': 'Rush Yds',
        'rushing_tds': 'Rush TD',
        'carries': 'Att',
        'rushing_epa': 'Rush EPA',

        # Receiving stats
        'receptions': 'Rec',
        'receiving_yards': 'Rec Yds',
        'receiving_tds': 'Rec TD',
        'targets': 'Tgt',
        'target_share': 'Tgt %',
        'receiving_epa': 'Rec EPA',
        'wopr': 'WOPR',
        'racr': 'RACR',

        # Kicker stats
        'fg_made': 'FGM',
        'fg_att': 'FGA',
        'fg_pct': 'FG%',
        'pat_made': 'XPM',
        'pat_att': 'XPA',

        # Defense stats
        'def_sacks': 'Sacks',
        'def_interceptions': 'INT',
        'pts_allow': 'PA',
        'def_tds': 'TD',

        # Matchup context
        'matchup_name': 'Matchup',
        'team_points': 'My Team',
        'opponent_points': 'Opp Team',
        'win': 'Won',
        'loss': 'Lost',
        'is_playoffs': 'Playoffs',

        # Flags
        'started': 'Started',
        'optimal_player': 'Optimal',
    }

    # Only rename columns that exist in the dataframe
    renames_to_apply = {k: v for k, v in renames.items() if k in df.columns}

    return df.rename(columns=renames_to_apply)


def format_numeric_columns(df: pd.DataFrame, round_decimals: int = 2) -> pd.DataFrame:
    """
    Format numeric columns for display.

    Args:
        df: DataFrame to format
        round_decimals: Number of decimal places to round to

    Returns:
        DataFrame with formatted numeric columns
    """
    numeric_columns = df.select_dtypes(include=['float64', 'float32']).columns

    for col in numeric_columns:
        df[col] = df[col].round(round_decimals)

    return df


def get_stat_columns_by_position(position: str) -> List[str]:
    """
    Get relevant stat columns for a given position.

    Args:
        position: Position code (QB, RB, WR, TE, K, DEF)

    Returns:
        List of relevant column names for that position

    Example:
        ```python
        qb_cols = get_stat_columns_by_position('QB')
        # Returns: ['Pass Yds', 'Pass TD', 'Pass INT', ...]
        ```
    """
    position_stats = {
        'QB': [
            'Pass Yds', 'Pass TD', 'Pass INT', 'Comp', 'Pass Att',
            'Rush Yds', 'Rush TD', 'Pass EPA', 'CPOE', 'PACR'
        ],
        'RB': [
            'Rush Yds', 'Rush TD', 'Att', 'Rec', 'Rec Yds', 'Rec TD',
            'Tgt', 'Rush EPA', 'Rec EPA', 'Tgt %'
        ],
        'WR': [
            'Rec', 'Rec Yds', 'Rec TD', 'Tgt', 'Tgt %',
            'Rec EPA', 'WOPR', 'RACR', 'Rush Yds', 'Rush TD'
        ],
        'TE': [
            'Rec', 'Rec Yds', 'Rec TD', 'Tgt', 'Tgt %',
            'Rec EPA', 'Rush Yds', 'Rush TD'
        ],
        'K': [
            'FGM', 'FGA', 'FG%', 'FG 0-19', 'FG 20-29', 'FG 30-39',
            'FG 40-49', 'FG 50+', 'XPM', 'XPA'
        ],
        'DEF': [
            'Sacks', 'INT', 'PA', 'TD', 'FF', 'Fum Rec',
            'Total Tkl', 'TFL', 'PD'
        ],
    }

    return position_stats.get(position, [])


def create_display_dataframe(
    df: pd.DataFrame,
    position: Optional[str] = None,
    include_core: bool = True
) -> pd.DataFrame:
    """
    Create a display-ready dataframe with appropriate columns.

    This is a helper function that combines cleaning, renaming, and column selection.

    Args:
        df: Raw dataframe from database
        position: Optional position filter for position-specific columns
        include_core: Whether to include core columns (Player, Team, etc.)

    Returns:
        Cleaned and formatted dataframe ready for display

    Example:
        ```python
        df = load_data()
        display_df = create_display_dataframe(df, position='QB')
        st.dataframe(display_df)
        ```
    """
    # Clean the dataframe
    df = clean_dataframe(df)

    # Apply common renames
    df = apply_common_renames(df)

    # Core columns to always include
    core_cols = ['Player', 'Team', 'Week', 'Year', 'Manager', 'Points']

    # Build column list
    display_cols = []

    if include_core:
        display_cols.extend([col for col in core_cols if col in df.columns])

    # Add position-specific stats
    if position:
        stat_cols = get_stat_columns_by_position(position)
        display_cols.extend([col for col in stat_cols if col in df.columns])

    # If no columns selected, show all
    if not display_cols:
        display_cols = df.columns.tolist()

    # Filter to only display columns
    result_df = df[[col for col in display_cols if col in df.columns]].copy()

    # Format numeric columns
    result_df = format_numeric_columns(result_df)

    return result_df
