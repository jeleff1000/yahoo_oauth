"""
SPAR (Season/Started Points Above Replacement) Calculator

Calculates SPAR metrics distinguishing player talent from manager usage:

1. Player SPAR - Player's true production (all weeks played, regardless of roster status)
   - Measures player talent/value independent of being rostered/started
   - Includes unrostered performances (e.g., Stafford's 26.9 on waivers)

2. Manager SPAR - Manager's realized value (only weeks they started the player)
   - Measures fantasy impact for a specific manager
   - Excludes benched/unrostered weeks

3. Roster SPAR - Value while on a specific roster (for trades/cuts analysis)

Uses weekly-varying replacement levels (QB18 one week, QB16 another).
"""

import pandas as pd
import polars as pl
from typing import Optional


def add_weekly_spar(
    player_df: pd.DataFrame,
    weekly_replacement_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add weekly SPAR metrics to player DataFrame.

    Calculates TWO metrics:
    1. player_weekly_spar: All weeks the player played (measures player talent)
    2. manager_weekly_spar: Only weeks the player was started by a manager (measures fantasy impact)

    Args:
        player_df: Player DataFrame with fantasy_points
        weekly_replacement_df: Weekly replacement levels (year, week, position, replacement_ppg)

    Returns:
        DataFrame with player_weekly_spar and manager_weekly_spar columns added

    Note: Replacement baseline includes all performances (even benched/unrostered players).
          This ensures we measure true replacement level, not filtered replacement level.
    """
    df = player_df.copy()

    # Join with weekly replacement levels
    df = df.merge(
        weekly_replacement_df[['year', 'week', 'position', 'replacement_ppg']],
        on=['year', 'week', 'position'],
        how='left'
    )

    # Fill missing replacement values with 0
    df['replacement_ppg'] = df['replacement_ppg'].fillna(0)

    # Player SPAR: All weeks the player played (regardless of roster status)
    # This measures the player's true talent/production
    df['player_weekly_spar'] = df['fantasy_points'] - df['replacement_ppg']

    # Manager SPAR: Only weeks the player was started (not BN/IR/Unrostered)
    # This measures the fantasy impact for managers who started them
    if 'fantasy_position' in df.columns:
        # Only count weeks where player was started (not BN/IR)
        started_mask = (
            df['fantasy_position'].notna() &
            ~df['fantasy_position'].isin(['BN', 'IR', 'IR+'])
        )
        df['manager_weekly_spar'] = 0.0
        df.loc[started_mask, 'manager_weekly_spar'] = (
            df.loc[started_mask, 'fantasy_points'] - df.loc[started_mask, 'replacement_ppg']
        )
    else:
        # Fallback: If no fantasy_position column, assume all weeks are "started"
        df['manager_weekly_spar'] = df['player_weekly_spar']

    return df


def calculate_season_spar(
    player_df: pd.DataFrame,
    group_cols: list = None
) -> pd.DataFrame:
    """
    Calculate season-level SPAR aggregations with dual player/manager metrics.

    Args:
        player_df: Player DataFrame with player_weekly_spar and manager_weekly_spar calculated
        group_cols: Columns to group by (default: ['player_id', 'year'])

    Returns:
        DataFrame with season SPAR metrics:

        Player SPAR (talent-based, all weeks played):
        - player_spar: Total player SPAR for the season (all weeks with points)
        - player_points: Total fantasy points (all weeks)
        - weeks_played: Number of weeks player scored points

        Manager SPAR (usage-based, only started weeks):
        - manager_spar: Total manager SPAR for the season (only started weeks)
        - manager_points: Total fantasy points from started weeks
        - weeks_started: Number of weeks player was started

        Rate statistics:
        - player_spar_per_week: player_spar / weeks_played
        - manager_spar_per_week: manager_spar / weeks_started
        - weeks_rostered: Total weeks on a roster (started + benched)
    """
    if group_cols is None:
        # Use player_id if available, otherwise yahoo_player_id
        id_col = 'player_id' if 'player_id' in player_df.columns else 'yahoo_player_id'
        group_cols = [id_col, 'year']

    df = player_df.copy()

    # Filter to weeks where player actually played (has fantasy_points > 0 or not null)
    played_filter = df['fantasy_points'].notna() & (df['fantasy_points'] != 0)
    df_played = df[played_filter].copy()

    # Calculate player SPAR aggregations (all weeks played)
    player_stats = df_played.groupby(group_cols).agg({
        'player_weekly_spar': 'sum',
        'fantasy_points': 'sum',
        'week': 'count'
    }).reset_index()

    player_stats.rename(columns={
        'player_weekly_spar': 'player_spar',
        'fantasy_points': 'player_points',
        'week': 'weeks_played'
    }, inplace=True)

    # Calculate manager SPAR aggregations (only started weeks)
    # Filter to started weeks (fantasy_position not BN/IR/Unrostered)
    if 'fantasy_position' in df.columns:
        started_filter = (
            df['fantasy_position'].notna() &
            ~df['fantasy_position'].isin(['BN', 'IR', 'IR+'])
        )
        df_started = df[started_filter].copy()
    else:
        # Fallback: if no fantasy_position, assume all weeks are started
        df_started = df.copy()

    manager_stats = df_started.groupby(group_cols).agg({
        'manager_weekly_spar': 'sum',
        'fantasy_points': 'sum',
        'week': 'count'
    }).reset_index()

    manager_stats.rename(columns={
        'manager_weekly_spar': 'manager_spar',
        'fantasy_points': 'manager_points',
        'week': 'weeks_started'
    }, inplace=True)

    # Merge player and manager stats
    season_stats = player_stats.merge(
        manager_stats,
        on=group_cols,
        how='outer'  # Outer join to include unrostered players (have player_spar but no manager_spar)
    )

    # Fill NaN values for unrostered players
    season_stats['manager_spar'] = season_stats['manager_spar'].fillna(0)
    season_stats['manager_points'] = season_stats['manager_points'].fillna(0)
    season_stats['weeks_started'] = season_stats['weeks_started'].fillna(0)

    # Calculate weeks_rostered (started + benched)
    if 'fantasy_position' in df.columns:
        rostered_weeks = df[
            df['fantasy_position'].notna()
        ].groupby(group_cols).size().reset_index(name='weeks_rostered')

        season_stats = season_stats.merge(rostered_weeks, on=group_cols, how='left')
        season_stats['weeks_rostered'] = season_stats['weeks_rostered'].fillna(season_stats['weeks_started'])
    else:
        season_stats['weeks_rostered'] = season_stats['weeks_started']

    # Calculate rate statistics
    # Player rate: per week played (talent efficiency)
    season_stats['player_spar_per_week'] = (
        season_stats['player_spar'] / season_stats['weeks_played'].clip(lower=1)
    )

    # Manager rate: per week started (usage efficiency)
    season_stats['manager_spar_per_week'] = (
        season_stats['manager_spar'] / season_stats['weeks_started'].clip(lower=1)
    )

    return season_stats


def calculate_roster_spar(
    player_df: pd.DataFrame,
    manager_col: str = 'manager'
) -> pd.DataFrame:
    """
    Calculate SPAR while on each manager's roster with dual player/manager metrics.

    Handles trades/cuts by grouping by player + manager + year.

    Args:
        player_df: Player DataFrame with player_weekly_spar and manager_weekly_spar
        manager_col: Column name for manager/team

    Returns:
        DataFrame with roster SPAR metrics:

        Player metrics (talent while on roster):
        - roster_player_spar: Total player SPAR while on this roster
        - roster_player_points: Total points (all weeks on roster)

        Manager metrics (value realized while on roster):
        - roster_manager_spar: Total manager SPAR while on this roster (started only)
        - roster_manager_points: Total points from started weeks

        Context:
        - weeks_started_with_manager: Weeks started by this manager
        - weeks_rostered_with_manager: Weeks on roster (started + benched)
        - first_week_with_manager: First week with this manager
        - last_week_with_manager: Last week with this manager
    """
    df = player_df.copy()

    # Use player_id if available, otherwise yahoo_player_id
    id_col = 'player_id' if 'player_id' in df.columns else 'yahoo_player_id'

    # Filter to rostered players only (exclude Unrostered)
    rostered = df[df[manager_col] != 'Unrostered'].copy()

    if rostered.empty:
        return pd.DataFrame()

    # Group by player + manager + year
    group_cols = [id_col, manager_col, 'year']

    # Calculate player SPAR (all weeks on roster, including benched)
    roster_stats_player = rostered.groupby(group_cols).agg({
        'player_weekly_spar': 'sum',
        'fantasy_points': 'sum',
        'week': ['min', 'max']
    }).reset_index()

    roster_stats_player.columns = [
        id_col, manager_col, 'year',
        'roster_player_spar', 'roster_player_points',
        'first_week_with_manager', 'last_week_with_manager'
    ]

    # Calculate manager SPAR (only started weeks)
    started_filter = (
        rostered['fantasy_position'].notna() &
        ~rostered['fantasy_position'].isin(['BN', 'IR', 'IR+'])
    ) if 'fantasy_position' in rostered.columns else pd.Series([True] * len(rostered))

    rostered_started = rostered[started_filter].copy()

    roster_stats_manager = rostered_started.groupby(group_cols).agg({
        'manager_weekly_spar': 'sum',
        'fantasy_points': 'sum',
        'week': 'count'
    }).reset_index()

    roster_stats_manager.columns = [
        id_col, manager_col, 'year',
        'roster_manager_spar', 'roster_manager_points',
        'weeks_started_with_manager'
    ]

    # Merge player and manager stats
    roster_stats = roster_stats_player.merge(
        roster_stats_manager,
        on=group_cols,
        how='left'
    )

    # Fill NaN values for players who were rostered but never started
    roster_stats['roster_manager_spar'] = roster_stats['roster_manager_spar'].fillna(0)
    roster_stats['roster_manager_points'] = roster_stats['roster_manager_points'].fillna(0)
    roster_stats['weeks_started_with_manager'] = roster_stats['weeks_started_with_manager'].fillna(0)

    # Calculate weeks_rostered_with_manager (started + benched)
    if 'fantasy_position' in rostered.columns:
        rostered_weeks = rostered[
            rostered['fantasy_position'].notna()
        ].groupby(group_cols).size().reset_index(name='weeks_rostered_with_manager')

        roster_stats = roster_stats.merge(rostered_weeks, on=group_cols, how='left')
        roster_stats['weeks_rostered_with_manager'] = roster_stats['weeks_rostered_with_manager'].fillna(
            roster_stats['weeks_started_with_manager']
        )
    else:
        roster_stats['weeks_rostered_with_manager'] = roster_stats['weeks_started_with_manager']

    return roster_stats


def calculate_all_spar_metrics(
    player_df,
    weekly_replacement_df: pd.DataFrame,
    manager_col: str = 'manager'
):
    """
    Calculate all SPAR metrics and add to player DataFrame.

    This is the main entry point - calculates dual player/manager SPAR metrics:
    1. Weekly SPAR (player_weekly_spar, manager_weekly_spar)
    2. Season SPAR (player_spar, manager_spar)
    3. Roster SPAR (roster_player_spar, roster_manager_spar)

    Args:
        player_df: Player DataFrame (pandas or polars)
        weekly_replacement_df: Weekly replacement levels
        manager_col: Column name for manager/team

    Returns:
        DataFrame with all SPAR columns added:

        Weekly:
        - player_weekly_spar: SPAR for all weeks played
        - manager_weekly_spar: SPAR for started weeks only

        Season:
        - player_spar: Total player SPAR for season
        - manager_spar: Total manager SPAR for season
        - player_points, manager_points
        - weeks_played, weeks_started, weeks_rostered

        Roster (if rostered):
        - roster_player_spar: SPAR while on roster
        - roster_manager_spar: SPAR while started by manager
        - weeks_started_with_manager, weeks_rostered_with_manager
    """
    # Convert polars to pandas if needed
    is_polars = False
    try:
        if isinstance(player_df, pl.DataFrame):
            is_polars = True
            df = player_df.to_pandas()
        else:
            df = player_df.copy()
    except:
        df = player_df.copy()

    # Step 1: Add weekly SPAR (both player and manager)
    df = add_weekly_spar(df, weekly_replacement_df)

    # Step 2: Calculate season SPAR and join back
    id_col = 'player_id' if 'player_id' in df.columns else 'yahoo_player_id'
    season_spar_df = calculate_season_spar(df, group_cols=[id_col, 'year'])

    df = df.merge(
        season_spar_df,
        on=[id_col, 'year'],
        how='left',
        suffixes=('', '_season_agg')
    )

    # Step 3: Calculate roster SPAR and join back
    roster_spar_df = calculate_roster_spar(df, manager_col=manager_col)

    if not roster_spar_df.empty:
        df = df.merge(
            roster_spar_df,
            on=[id_col, manager_col, 'year'],
            how='left',
            suffixes=('', '_roster_agg')
        )

    # Clean up duplicate columns from merges
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert back to polars if needed
    if is_polars:
        return pl.from_pandas(df)

    return df


def calculate_ros_spar(
    player_df: pd.DataFrame,
    weekly_replacement_df: pd.DataFrame,
    start_week: int,
    end_week: int = 17,
    year: int = None
) -> pd.DataFrame:
    """
    Calculate rest-of-season SPAR from a specific week forward.

    Used for transaction analysis (waiver pickups, trades).

    Args:
        player_df: Player DataFrame
        weekly_replacement_df: Weekly replacement levels
        start_week: First week of ROS window (transaction week + 1)
        end_week: Last week of ROS window (default: 17)
        year: Optional year filter

    Returns:
        DataFrame with ROS SPAR metrics:
        - ros_spar: Rest-of-season SPAR
        - ros_weeks: Number of weeks in ROS window
        - ros_points: Total points in ROS window
    """
    df = player_df.copy()

    # Filter to ROS window
    ros_filter = (df['week'] >= start_week) & (df['week'] <= end_week)
    if year is not None:
        ros_filter &= (df['year'] == year)

    ros_df = df[ros_filter].copy()

    # Add weekly SPAR for ROS window
    ros_df = add_weekly_spar(ros_df, weekly_replacement_df, started_only=True)

    # Aggregate ROS metrics
    id_col = 'player_id' if 'player_id' in ros_df.columns else 'yahoo_player_id'

    ros_stats = ros_df.groupby(id_col).agg({
        'weekly_spar': 'sum',
        'fantasy_points': 'sum',
        'week': 'count'
    }).reset_index()

    ros_stats.rename(columns={
        'weekly_spar': 'ros_spar',
        'fantasy_points': 'ros_points',
        'week': 'ros_weeks'
    }, inplace=True)

    return ros_stats
