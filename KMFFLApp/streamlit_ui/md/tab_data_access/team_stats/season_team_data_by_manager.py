#!/usr/bin/env python3
"""
Season team stats data loader - aggregates by MANAGER ONLY (all positions combined).
"""
from __future__ import annotations
import streamlit as st
from md.data_access import run_query, T


@st.cache_data(show_spinner=True, ttl=600)
def load_season_team_data_by_manager(
    year: int | None = None,
    include_regular_season: bool = True,
    include_playoffs: bool = True,
    include_consolation: bool = False
):
    """
    Load season team stats aggregated by manager only (all positions combined).
    Filters out Unrostered players and BN/IR positions.

    Args:
        year: Optional year filter
        include_playoffs: Whether to include playoff weeks (default: True)
        include_consolation: Whether to include consolation weeks (default: False)

    Returns:
        DataFrame with season team stats grouped by manager only
    """
    try:
        where = [
            "manager IS NOT NULL",
            "manager <> ''",
            "manager <> 'Unrostered'",
            "fantasy_position IS NOT NULL",
            "fantasy_position NOT IN ('BN', 'IR')"
        ]

        if year is not None:
            where.append(f"year = {int(year)}")

        # Apply week type filters (regular season, playoffs, consolation)
        # Each checkbox controls whether that type of game is included
        week_type_conditions = []
        if include_regular_season:
            week_type_conditions.append("((is_playoffs IS NULL OR is_playoffs = 0) AND (is_consolation IS NULL OR is_consolation = 0))")
        if include_playoffs:
            week_type_conditions.append("is_playoffs = 1")
        if include_consolation:
            week_type_conditions.append("is_consolation = 1")

        # If no week types selected, return no data
        if week_type_conditions:
            where.append(f"({' OR '.join(week_type_conditions)})")
        else:
            where.append("1 = 0")  # No data if nothing selected

        where_sql = "WHERE " + " AND ".join(where)

        # Query aggregating by manager, year only
        sql = f"""
            SELECT
                manager,
                year,
                MAX(is_playoffs) as is_playoffs,
                MAX(is_consolation) as is_consolation,
                SUM(CAST(points AS DOUBLE)) as points,
                COUNT(*) as games_played,
                SUM(CAST(points AS DOUBLE)) / NULLIF(COUNT(*), 0) as season_ppg,
                SUM(CAST(spar AS DOUBLE)) as spar,
                SUM(CAST(player_spar AS DOUBLE)) as player_spar,
                SUM(CAST(manager_spar AS DOUBLE)) as manager_spar,
                SUM(CAST(passing_yards AS DOUBLE)) as passing_yards,
                SUM(CAST(passing_tds AS DOUBLE)) as passing_tds,
                SUM(CAST(passing_interceptions AS DOUBLE)) as passing_interceptions,
                SUM(CAST(completions AS DOUBLE)) as completions,
                SUM(CAST(attempts AS DOUBLE)) as attempts,
                SUM(CAST(passing_air_yards AS DOUBLE)) as passing_air_yards,
                SUM(CAST(passing_yards_after_catch AS DOUBLE)) as passing_yards_after_catch,
                SUM(CAST(passing_first_downs AS DOUBLE)) as passing_first_downs,
                SUM(CAST(passing_epa AS DOUBLE)) as passing_epa,
                AVG(CAST(passing_cpoe AS DOUBLE)) as passing_cpoe,
                AVG(CAST(pacr AS DOUBLE)) as pacr,
                SUM(CAST(carries AS DOUBLE)) as rush_att,
                SUM(CAST(rushing_yards AS DOUBLE)) as rushing_yards,
                SUM(CAST(rushing_tds AS DOUBLE)) as rushing_tds,
                SUM(CAST(rushing_fumbles AS DOUBLE)) as rushing_fumbles,
                SUM(CAST(rushing_fumbles_lost AS DOUBLE)) as fum_lost,
                SUM(CAST(rushing_first_downs AS DOUBLE)) as rushing_first_downs,
                SUM(CAST(rushing_epa AS DOUBLE)) as rushing_epa,
                SUM(CAST(passing_2pt_conversions AS DOUBLE)) as passing_2pt_conversions,
                SUM(CAST(rushing_2pt_conversions AS DOUBLE)) as rushing_2pt_conversions,
                SUM(CAST(receptions AS DOUBLE)) as receptions,
                SUM(CAST(targets AS DOUBLE)) as targets,
                SUM(CAST(receiving_yards AS DOUBLE)) as receiving_yards,
                SUM(CAST(receiving_tds AS DOUBLE)) as receiving_tds,
                SUM(CAST(receiving_fumbles AS DOUBLE)) as receiving_fumbles,
                SUM(CAST(receiving_fumbles_lost AS DOUBLE)) as receiving_fumbles_lost,
                SUM(CAST(receiving_first_downs AS DOUBLE)) as receiving_first_downs,
                SUM(CAST(receiving_epa AS DOUBLE)) as receiving_epa,
                SUM(CAST(receiving_2pt_conversions AS DOUBLE)) as receiving_2pt_conversions,
                AVG(CAST(target_share AS DOUBLE)) as target_share,
                AVG(CAST(wopr AS DOUBLE)) as wopr,
                AVG(CAST(racr AS DOUBLE)) as racr,
                SUM(CAST(receiving_air_yards AS DOUBLE)) as receiving_air_yards,
                SUM(CAST(receiving_yards_after_catch AS DOUBLE)) as receiving_yards_after_catch,
                AVG(CAST(air_yards_share AS DOUBLE)) as air_yards_share,
                SUM(CAST(fg_made AS DOUBLE)) as fg_made,
                SUM(CAST(fg_att AS DOUBLE)) as fg_att,
                AVG(CAST(fg_pct AS DOUBLE)) as fg_pct,
                MAX(CAST(fg_long AS DOUBLE)) as fg_long,
                SUM(CAST(fg_made_0_19 AS DOUBLE)) as fg_made_0_19,
                SUM(CAST(fg_made_20_29 AS DOUBLE)) as fg_made_20_29,
                SUM(CAST(fg_made_30_39 AS DOUBLE)) as fg_made_30_39,
                SUM(CAST(fg_made_40_49 AS DOUBLE)) as fg_made_40_49,
                SUM(CAST(fg_made_50_59 AS DOUBLE)) as fg_made_50_59,
                SUM(CAST(fg_missed AS DOUBLE)) as fg_miss,
                SUM(CAST(pat_made AS DOUBLE)) as pat_made,
                SUM(CAST(pat_att AS DOUBLE)) as pat_att,
                SUM(CAST(pat_missed AS DOUBLE)) as pat_missed,
                SUM(CAST(def_sacks AS DOUBLE)) as def_sacks,
                SUM(CAST(def_sack_yards AS DOUBLE)) as def_sack_yards,
                SUM(CAST(def_qb_hits AS DOUBLE)) as def_qb_hits,
                SUM(CAST(def_interceptions AS DOUBLE)) as def_interceptions,
                SUM(CAST(def_interception_yards AS DOUBLE)) as def_interception_yards,
                SUM(CAST(def_pass_defended AS DOUBLE)) as def_pass_defended,
                SUM(CAST(def_tackles_solo AS DOUBLE)) as def_tackles_solo,
                SUM(CAST(def_tackle_assists AS DOUBLE)) as def_tackle_assists,
                SUM(CAST(def_tackles_with_assist AS DOUBLE)) as def_tackles_with_assist,
                SUM(CAST(def_tackles_for_loss AS DOUBLE)) as def_tackles_for_loss,
                SUM(CAST(def_tackles_for_loss_yards AS DOUBLE)) as def_tackles_for_loss_yards,
                SUM(CAST(def_fumbles AS DOUBLE)) as def_fumbles,
                SUM(CAST(def_fumbles_forced AS DOUBLE)) as def_fumbles_forced,
                SUM(CAST(def_safeties AS DOUBLE)) as def_safeties,
                SUM(CAST(def_tds AS DOUBLE)) as def_td,
                SUM(CAST(pts_allow AS DOUBLE)) as pts_allow,
                SUM(CAST(dst_points_allowed AS DOUBLE)) as dst_points_allowed,
                SUM(CAST(points_allowed AS DOUBLE)) as points_allowed,
                SUM(CAST(passing_yds_allowed AS DOUBLE)) as pass_yds_allowed,
                SUM(CAST(rushing_yds_allowed AS DOUBLE)) as rushing_yds_allowed,
                SUM(CAST(total_yds_allowed AS DOUBLE)) as total_yds_allowed,
                SUM(CAST(fumble_recovery_opp AS DOUBLE)) as fumble_recovery_opp,
                SUM(CAST(fumble_recovery_tds AS DOUBLE)) as fumble_recovery_tds,
                SUM(CAST(three_out AS DOUBLE)) as three_out
            FROM {T['player']}
            {where_sql}
            GROUP BY manager, year
            ORDER BY year DESC, manager
        """

        df = run_query(sql)
        return df

    except Exception as e:
        st.error(f"Failed to load season team data by manager: {e}")
        return None
