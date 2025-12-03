#!/usr/bin/env python3
"""
Recap player data loader for Homepage tab - Recaps section.

Optimization: Loads only the columns needed for the 2-week player slice
instead of all 270+ columns from players_by_year table.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
from md.core import run_query, get_current_league_db

# Columns needed for player recaps
RECAPS_PLAYER_COLUMNS = [
    # Identity
    "player",
    "manager",
    "headshot_url",

    # Time
    "year",
    "week",
    "cumulative_week",

    # Position
    "nfl_position",
    "fantasy_position",
    "yahoo_position",
    "optimal_position",

    # Points
    "points",

    # Lineup status
    "started",
    "optimal_player",

    # Team & opponent info
    "nfl_team",
    "opponent",
    "opponent_nfl_team",

    # Rankings & Percentiles (for exceptional performance detection)
    "position_week_rank",
    "position_week_pct",
    "position_season_rank",
    "position_season_pct",
    "player_personal_week_rank",
    "player_personal_week_pct",
    "all_players_week_rank",
    "all_players_week_pct",
    "all_players_alltime_rank",
    "all_players_alltime_pct",
]


@st.cache_data(show_spinner=True, ttl=120)
def load_player_two_week_slice(year: int, week: int) -> pd.DataFrame:
    """
    Load only the 2-week player slice needed for recaps with optimized column selection.

    This replaces the old load_player_two_week_slice which loaded ALL columns.

    Args:
        year: The target year
        week: The target week

    Returns:
        DataFrame with player data for current week and previous week (by cumulative_week)

    Columns loaded (27):
        - Identity (3): player, manager, headshot_url
        - Time (3): year, week, cumulative_week
        - Position (4): nfl_position, fantasy_position, yahoo_position, optimal_position
        - Points (1): points
        - Lineup (2): started, optimal_player
        - Team/Opponent (3): nfl_team, opponent, opponent_nfl_team
        - Rankings/Percentiles (9): position_week_rank, position_week_pct, position_season_rank,
                                     position_season_pct, player_personal_week_rank, player_personal_week_pct,
                                     all_players_week_rank, all_players_week_pct, all_players_alltime_rank,
                                     all_players_alltime_pct

    Optimization:
        - Loads 27 columns instead of 270+ (90% reduction)
        - Only loads 2 cumulative weeks of data (not entire table)
    """
    try:
        db = get_current_league_db()
        players_table = f"{db}.public.players_by_year"

        # Build column list
        cols_str = ", ".join(RECAPS_PLAYER_COLUMNS)

        # First, get cumulative week values
        cum_query = f"""
            WITH current_cum AS (
                SELECT DISTINCT cumulative_week
                FROM {players_table}
                WHERE year = {int(year)} AND week = {int(week)}
                LIMIT 1
            ),
            prev_cum AS (
                SELECT MAX(cumulative_week) AS prev_cumulative_week
                FROM {players_table}
                WHERE cumulative_week < (SELECT cumulative_week FROM current_cum)
            )
            SELECT
                (SELECT cumulative_week FROM current_cum) AS current_cum,
                (SELECT prev_cumulative_week FROM prev_cum) AS prev_cum
        """
        cum = run_query(cum_query)

        # If no cumulative week found, just get current week
        if cum.empty or cum["current_cum"].isna().all():
            query = f"""
                SELECT {cols_str}
                FROM {players_table}
                WHERE year = {int(year)} AND week = {int(week)}
                ORDER BY points DESC NULLS LAST
            """
            return run_query(query)

        cur_cum = float(cum.iloc[0]["current_cum"])
        prev_cum = cum.iloc[0]["prev_cum"]

        # If no previous week, just get current
        if prev_cum is None or pd.isna(prev_cum):
            query = f"""
                SELECT {cols_str}
                FROM {players_table}
                WHERE year = {int(year)} AND week = {int(week)}
                ORDER BY points DESC NULLS LAST
            """
            return run_query(query)

        # Get both weeks
        query = f"""
            SELECT {cols_str}
            FROM {players_table}
            WHERE cumulative_week IN ({float(prev_cum)}, {cur_cum})
            ORDER BY cumulative_week DESC, points DESC NULLS LAST
        """
        return run_query(query)

    except Exception as e:
        st.error(f"Failed to load player two-week slice: {e}")
        return pd.DataFrame()
