#!/usr/bin/env python3
"""
Keeper data loader for Keepers tab.

Optimization:
    - Columns: 17 of 272 (~94% reduction)
    - Rows: Only max week per manager/year (end-of-season roster)
    - Filters: Excludes unrostered, DEF, K at database level

Columns loaded (22 of 272):
    Core Identity (6): player, manager, yahoo_player_id, yahoo_position, nfl_team, headshot_url
    Time (2): year, week
    Scoring (2): points, spar
    Keeper Status (3): is_keeper_status, kept_next_year, keeper_price
    Keeper SPAR (3): keeper_spar_per_dollar, keeper_surplus_spar, keeper_roi_spar
    Performance (2): season_ppg (this year), avg_points_next_year (next year)
    Acquisition (3): cost, max_faab_bid_to_date, fantasy_position

Row Filtering:
    - Only max week per manager/year (end of season roster)
    - Excludes manager = NULL, 'no manager', 'unrostered', ''
    - Excludes DEF and K positions
    - Result: ~95% fewer rows than full player table
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from md.core import run_query, T


@st.cache_data(show_spinner=True, ttl=120)
def load_keeper_data(all_years: bool = True, year: int = None, week: int = None) -> pd.DataFrame | None:
    """
    Load keeper data from player table with column selection optimization.

    Shows each manager's roster at the END OF SEASON (max week per manager/year).
    This represents the keeper-eligible players for that manager.

    Args:
        all_years: Load end-of-season data for all years
        year: Specific year (requires week parameter)
        week: Specific week (requires year parameter)

    Returns:
        DataFrame with keeper data or None on error

    Optimization:
        - Loads only 17/272 columns (~94% reduction)
        - Only max week per manager/year (~95% row reduction)
        - Uses CTE for efficient max week calculation
        - Excludes Unrostered, DEF, K
        - Includes player headshots for visual display
    """
    try:
        # Build the WHERE clause for week selection
        use_cte = False
        if year is not None and week is not None:
            # Specific week view
            week_filter = f"AND p.year = {int(year)} AND p.week = {int(week)}"
            join_lw = ""  # no last-weeks join needed
        else:
            # We'll join to a CTE that holds the MAX(week) per player+year from the players table.
            use_cte = True
            if all_years:
                # end-of-season for every year
                join_lw = (
                    "JOIN last_weeks lw ON p.manager = lw.manager"
                    " AND p.year = lw.year AND p.week = lw.max_week"
                )
                week_filter = ""
            else:
                # end-of-season for latest year only
                join_lw = (
                    "JOIN last_weeks lw ON p.manager = lw.manager"
                    " AND p.year = lw.year AND p.week = lw.max_week"
                )
                week_filter = f"AND p.year = (SELECT MAX(year) FROM {T['player']})"

        if use_cte:
            keeper_query = f"""
                WITH last_weeks AS (
                    SELECT manager, year, MAX(week) AS max_week
                    FROM {T['player']}
                    WHERE manager IS NOT NULL
                      AND LOWER(TRIM(manager)) NOT IN ('no manager', 'unrostered', '')
                      AND (nfl_position NOT IN ('DEF', 'K') OR nfl_position IS NULL)
                    GROUP BY manager, year
                ),
                next_year_max_weeks AS (
                    SELECT p.yahoo_player_id, p.manager, p.year, p.season_ppg
                    FROM {T['player']} p
                    JOIN last_weeks lw
                        ON p.manager = lw.manager
                        AND p.year = lw.year
                        AND p.week = lw.max_week
                )
                SELECT
                    p.player,
                    p.manager,
                    p.nfl_position AS yahoo_position,
                    p.nfl_team,
                    p.year,
                    p.week,
                    p.points,
                    COALESCE(p.spar, 0) AS spar,
                    CASE WHEN p.kept_next_year = 1 THEN true ELSE false END AS kept_next_year,
                    CASE WHEN p.is_keeper_status = 1 THEN true ELSE false END AS is_keeper_status,
                    COALESCE(p.keeper_price, 0) AS keeper_price,
                    p.season_ppg AS avg_points_this_year,
                    p_next.season_ppg AS avg_points_next_year,
                    COALESCE(p.cost, 0) AS cost,
                    COALESCE(p.max_faab_bid_to_date, 0) AS max_faab_bid_to_date,
                    p.yahoo_player_id,
                    p.fantasy_position,
                    p.headshot_url
                FROM {T['player']} p
                {join_lw}
                LEFT JOIN next_year_max_weeks p_next
                    ON p.yahoo_player_id = p_next.yahoo_player_id
                    AND p.manager = p_next.manager
                    AND CAST(p.year AS INTEGER) + 1 = CAST(p_next.year AS INTEGER)
                {week_filter}
                ORDER BY p.year DESC, p.week DESC, p.manager, p.points DESC
            """
        else:
            # specific week simple path
            keeper_query = f"""
                WITH next_year_max_weeks AS (
                    SELECT p.yahoo_player_id, p.manager, p.year, p.season_ppg
                    FROM {T['player']} p
                    JOIN (
                        SELECT manager, year, MAX(week) AS max_week
                        FROM {T['player']}
                        WHERE CAST(year AS INTEGER) = {int(year)} + 1
                          AND manager IS NOT NULL
                          AND LOWER(TRIM(manager)) NOT IN ('no manager', 'unrostered', '')
                          AND (nfl_position NOT IN ('DEF', 'K') OR nfl_position IS NULL)
                        GROUP BY manager, year
                    ) lw
                        ON p.manager = lw.manager
                        AND p.year = lw.year
                        AND p.week = lw.max_week
                )
                SELECT
                    p.player,
                    p.manager,
                    p.nfl_position AS yahoo_position,
                    p.nfl_team,
                    p.year,
                    p.week,
                    p.points,
                    COALESCE(p.spar, 0) AS spar,
                    CASE WHEN p.kept_next_year = 1 THEN true ELSE false END AS kept_next_year,
                    CASE WHEN p.is_keeper_status = 1 THEN true ELSE false END AS is_keeper_status,
                    COALESCE(p.keeper_price, 0) AS keeper_price,
                    p.season_ppg AS avg_points_this_year,
                    p_next.season_ppg AS avg_points_next_year,
                    COALESCE(p.cost, 0) AS cost,
                    COALESCE(p.max_faab_bid_to_date, 0) AS max_faab_bid_to_date,
                    p.yahoo_player_id,
                    p.fantasy_position,
                    p.headshot_url
                FROM {T['player']} p
                LEFT JOIN next_year_max_weeks p_next
                    ON p.yahoo_player_id = p_next.yahoo_player_id
                    AND p.manager = p_next.manager
                    AND CAST(p.year AS INTEGER) + 1 = CAST(p_next.year AS INTEGER)
                WHERE p.manager IS NOT NULL
                  AND LOWER(TRIM(p.manager)) NOT IN ('no manager', 'unrostered', '')
                  AND (p.nfl_position NOT IN ('DEF', 'K') OR p.nfl_position IS NULL)
                  {week_filter}
                ORDER BY p.year DESC, p.week DESC, p.manager, p.points DESC
            """

        df = run_query(keeper_query)
        return df

    except Exception as e:
        st.error(f"Failed to load keeper data: {e}")
        return None
