#!/usr/bin/env python3
"""
Season player data loader for Player Stats tab.

Optimization: Loads only ~108 out of 237 columns from players_by_year table.
This reduces data transfer by ~54% and improves query performance.

Based on comprehensive analysis of ALL season player subprocess files:
- season_player_basic_stats.py
- season_player_advanced_stats.py
- season_player_matchup_stats.py

IMPORTANT: Column names verified against actual players_by_year table schema.
"""
from __future__ import annotations
from typing import Sequence
import streamlit as st
from md.core import run_query, T, sql_quote, sql_in_list, sql_upper, sql_upper_in_list, sql_manager_norm
import pandas as pd

# Columns needed for season player stats (~116 out of 237 columns = 51% reduction)
# Source columns from players_by_year table (before aggregation)
SEASON_PLAYER_SOURCE_COLUMNS = [
    # === Identifiers (10) ===
    "player",
    "yahoo_player_id",
    "NFL_player_id",
    "nfl_team",
    "opponent_nfl_team",
    "manager",
    "opponent",
    "headshot_url",

    # === Time/Period (2) ===
    "year",
    "week",

    # === Positions (2) ===
    "nfl_position",
    "fantasy_position",

    # === Core Stats (5) ===
    "points",
    "season_ppg",
    "is_started",
    "win",
    "season_type",

    # === SPAR Metrics (3) ===
    "spar",             # Legacy SPAR
    "player_spar",      # Total SPAR produced
    "manager_spar",     # SPAR while on manager's roster

    # === Matchup Context (11) ===
    "team_points",
    "opponent_points",
    "is_playoffs",
    "is_consolation",
    "playoff_round",
    "consolation_round",
    "championship",
    "champion",
    "sacko",
    "optimal_player",
    "league_wide_optimal_player",
    "loss",
    "position_season_rank",

    # === Passing Stats (11) ===
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "attempts",
    "completions",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_cpoe",
    "pacr",

    # === Rushing Stats (8) ===
    "rushing_yards",
    "carries",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",

    # === Receiving Stats (14) ===
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "targets",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_2pt_conversions",
    "target_share",
    "wopr",
    "racr",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "air_yards_share",

    # === Kicking Stats (13) ===
    "fg_made",
    "fg_att",
    "fg_pct",
    "fg_long",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_missed",
    "pat_made",
    "pat_att",
    "pat_missed",

    # === Defense Stats (21) ===
    "def_sacks",
    "def_sack_yards",
    "def_qb_hits",
    "def_interceptions",
    "def_interception_yards",
    "def_pass_defended",
    "def_tackles_solo",
    "def_tackle_assists",
    "def_tackles_with_assist",
    "def_tackles_for_loss",
    "def_tackles_for_loss_yards",
    "def_fumbles",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "pts_allow",
    "dst_points_allowed",
    "points_allowed",
    "passing_yds_allowed",
    "rushing_yds_allowed",
    "total_yds_allowed",
    "fumble_recovery_opp",
    "fumble_recovery_tds",
    "special_teams_tds",
    "three_out",
    "fourth_down_stop",
]


@st.cache_data(show_spinner=True, ttl=120)
def load_season_player_data(
    position: str | None = None,
    player_query: str | None = None,
    manager_query: str | None = None,
    manager: Sequence[str] | None = None,
    nfl_team: Sequence[str] | None = None,
    opponent: Sequence[str] | None = None,
    opponent_nfl_team: Sequence[str] | None = None,
    year: Sequence[int] | None = None,
    rostered_only: bool = False,
    started_only: bool = False,
    exclude_postseason: bool = False,
    sort_column: str = "points",
    sort_direction: str = "DESC",
) -> pd.DataFrame:
    """
    Load season player data with ONLY the columns needed.

    Season view: Aggregates by (year, player_key) from players_by_year table.

    Args:
        position: Position filter (QB, RB, WR, TE, K, DEF)
        player_query: Player name search
        manager_query: Manager name search
        manager: List of managers to filter by
        nfl_team: List of NFL teams
        opponent: List of opponent managers
        opponent_nfl_team: List of opponent NFL teams
        year: List of years
        rostered_only: Only rostered players
        started_only: Only started players
        exclude_postseason: Exclude postseason games
        sort_column: Column to sort by
        sort_direction: ASC or DESC

    Returns:
        DataFrame with season aggregated player data

    Columns loaded (~116 out of 237 = 51% reduction):
        Identifiers (10): player, IDs, teams, manager, opponent, headshot
        Time (2): year, week
        Positions (2): nfl_position, fantasy_position
        Core (5): points, season_ppg, started, win, season_type, SPAR
        Matchup (11): team_points, opponent_points, playoffs, consolation, playoff_round, consolation_round, championship, champion, sacko, optimal, loss
        Passing (11): yards, TDs, INTs, attempts, completions, air yards, EPA, CPOE, PACR
        Rushing (8): yards, TDs, attempts, fumbles, first downs, EPA, 2PT
        Receiving (14): yards, TDs, targets, fumbles, EPA, target share, WOPR, RACR, air yards
        Kicking (13): FG stats by distance, XP stats
        Defense (21): sacks, tackles, INTs, fumbles, TFL, points/yards allowed
    """
    try:
        sort_column = (sort_column or "points")
        sort_direction = (sort_direction or "DESC").upper()
        if sort_direction not in ("ASC", "DESC"):
            sort_direction = "DESC"

        team_norm = sql_upper("nfl_team")
        opp_team_norm = sql_upper("opponent_nfl_team")
        m_norm = sql_manager_norm("manager")

        where: list[str] = []
        if position and position != "All":
            where.append("nfl_position = " + sql_quote(position))
        if player_query:
            esc = player_query.replace("'", "''")
            where.append("LOWER(player) LIKE LOWER('%%" + esc + "%%')")
        if manager_query:
            esc = manager_query.strip().lower().replace("'", "''")
            where.append("LOWER(" + m_norm + ") LIKE '%%" + esc + "%%'")
        if manager:
            where.append("TRIM(manager) IN (" + sql_in_list(manager) + ")")
        if nfl_team:
            where.append(f"{team_norm} IN ({sql_upper_in_list(nfl_team)})")
        if opponent:
            where.append("opponent IN (" + sql_in_list(opponent) + ")")
        if opponent_nfl_team:
            where.append(f"{opp_team_norm} IN ({sql_upper_in_list(opponent_nfl_team)})")
        if year:
            nums = ", ".join(str(int(y)) for y in year if str(y).strip())
            if nums:
                where.append("year IN (" + nums + ")")

        if rostered_only:
            where.append("manager IS NOT NULL AND manager <> '' AND manager <> 'Unrostered'")
        if started_only:
            where.append("started = 1")
        if exclude_postseason:
            where.append("(season_type IS NULL OR season_type = 'REG')")

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        # Build column list for SELECT (optimized!)
        cols_str = ", ".join(SEASON_PLAYER_SOURCE_COLUMNS)

        # Optimized query - only select needed columns instead of SELECT *
        # IMPORTANT: Use NFL_player_id as primary key to avoid incorrect merging
        # The old bridging logic caused issues with players sharing partial names
        # (e.g., "St. Brown" brothers, or "Bay" in team names)
        q = f"""
            WITH raw AS (
              SELECT {cols_str}
              FROM {T['players_by_year']}
              {where_sql}
            ),
            rows AS (
              SELECT
                year,
                TRIM(player) AS player,
                nfl_position,
                -- Use NFL_player_id as primary key, fallback to yahoo_player_id, then name
                -- This prevents incorrect merging of different players
                COALESCE(
                  NULLIF(TRIM(CAST(NFL_player_id AS VARCHAR)), ''),
                  NULLIF(TRIM(CAST(yahoo_player_id AS VARCHAR)), ''),
                  LOWER(TRIM(player))
                ) AS player_key,
                UPPER(NULLIF(TRIM(nfl_team), '')) AS nfl_team_norm,
                UPPER(NULLIF(TRIM(opponent_nfl_team), '')) AS opp_team_norm,
                CASE
                  WHEN manager IS NULL OR TRIM(manager) = '' OR LOWER(TRIM(manager)) = 'no manager'
                    THEN NULL
                  ELSE TRIM(manager)
                END AS manager_norm,
                opponent,
                headshot_url,
                points,
                season_ppg,
                -- Cast is_started to ensure it's treated as numeric, then convert to 1/0
                CASE WHEN CAST(is_started AS INTEGER) = 1 THEN 1 ELSE 0 END AS started_i,
                COALESCE(win, 0) AS win_i,
                COALESCE(CASE WHEN win = 0 THEN 1 ELSE 0 END, 0) AS loss_i,
                COALESCE(team_points, 0) AS team_pts_i,
                COALESCE(opponent_points, 0) AS opp_pts_i,
                COALESCE(is_playoffs, 0) AS is_playoffs_i,
                COALESCE(is_consolation, 0) AS is_consolation_i,
                COALESCE(championship, 0) AS championship_i,
                COALESCE(champion, 0) AS champion_i,
                COALESCE(sacko, 0) AS sacko_i,
                COALESCE(optimal_player, 0) AS optimal_player_i,
                COALESCE(league_wide_optimal_player, 0) AS league_wide_optimal_player_i,
                position_season_rank,
                playoff_round,
                consolation_round,
                -- All stat columns (with CAST to DOUBLE for aggregation)
                CAST(passing_yards AS DOUBLE) AS passing_yards,
                CAST(passing_tds AS DOUBLE) AS passing_tds,
                CAST(passing_interceptions AS DOUBLE) AS passing_interceptions,
                CAST(attempts AS DOUBLE) AS attempts,
                CAST(completions AS DOUBLE) AS completions,
                CAST(passing_air_yards AS DOUBLE) AS passing_air_yards,
                CAST(passing_yards_after_catch AS DOUBLE) AS passing_yards_after_catch,
                CAST(passing_first_downs AS DOUBLE) AS passing_first_downs,
                CAST(passing_epa AS DOUBLE) AS passing_epa,
                CAST(passing_cpoe AS DOUBLE) AS passing_cpoe,
                CAST(pacr AS DOUBLE) AS pacr,
                CAST(rushing_yards AS DOUBLE) AS rushing_yards,
                CAST(carries AS DOUBLE) AS carries,
                CAST(rushing_tds AS DOUBLE) AS rushing_tds,
                CAST(rushing_fumbles AS DOUBLE) AS rushing_fumbles,
                CAST(rushing_fumbles_lost AS DOUBLE) AS rushing_fumbles_lost,
                CAST(rushing_first_downs AS DOUBLE) AS rushing_first_downs,
                CAST(rushing_epa AS DOUBLE) AS rushing_epa,
                CAST(rushing_2pt_conversions AS DOUBLE) AS rushing_2pt_conversions,
                CAST(receptions AS DOUBLE) AS receptions,
                CAST(receiving_yards AS DOUBLE) AS receiving_yards,
                CAST(receiving_tds AS DOUBLE) AS receiving_tds,
                CAST(targets AS DOUBLE) AS targets,
                CAST(receiving_fumbles AS DOUBLE) AS receiving_fumbles,
                CAST(receiving_fumbles_lost AS DOUBLE) AS receiving_fumbles_lost,
                CAST(receiving_first_downs AS DOUBLE) AS receiving_first_downs,
                CAST(receiving_epa AS DOUBLE) AS receiving_epa,
                CAST(receiving_2pt_conversions AS DOUBLE) AS receiving_2pt_conversions,
                CAST(target_share AS DOUBLE) AS target_share,
                CAST(wopr AS DOUBLE) AS wopr,
                CAST(racr AS DOUBLE) AS racr,
                CAST(receiving_air_yards AS DOUBLE) AS receiving_air_yards,
                CAST(receiving_yards_after_catch AS DOUBLE) AS receiving_yards_after_catch,
                CAST(air_yards_share AS DOUBLE) AS air_yards_share,
                CAST(fg_made AS DOUBLE) AS fg_made,
                CAST(fg_att AS DOUBLE) AS fg_att,
                CAST(fg_pct AS DOUBLE) AS fg_pct,
                CAST(fg_long AS DOUBLE) AS fg_long,
                CAST(fg_made_0_19 AS DOUBLE) AS fg_made_0_19,
                CAST(fg_made_20_29 AS DOUBLE) AS fg_made_20_29,
                CAST(fg_made_30_39 AS DOUBLE) AS fg_made_30_39,
                CAST(fg_made_40_49 AS DOUBLE) AS fg_made_40_49,
                CAST(fg_made_50_59 AS DOUBLE) AS fg_made_50_59,
                CAST(fg_missed AS DOUBLE) AS fg_missed,
                CAST(pat_made AS DOUBLE) AS pat_made,
                CAST(pat_att AS DOUBLE) AS pat_att,
                CAST(pat_missed AS DOUBLE) AS pat_missed,
                CAST(def_sacks AS DOUBLE) AS def_sacks,
                CAST(def_sack_yards AS DOUBLE) AS def_sack_yards,
                CAST(def_qb_hits AS DOUBLE) AS def_qb_hits,
                CAST(def_interceptions AS DOUBLE) AS def_interceptions,
                CAST(def_interception_yards AS DOUBLE) AS def_interception_yards,
                CAST(def_pass_defended AS DOUBLE) AS def_pass_defended,
                CAST(def_tackles_solo AS DOUBLE) AS def_tackles_solo,
                CAST(def_tackle_assists AS DOUBLE) AS def_tackle_assists,
                CAST(def_tackles_with_assist AS DOUBLE) AS def_tackles_with_assist,
                CAST(def_tackles_for_loss AS DOUBLE) AS def_tackles_for_loss,
                CAST(def_tackles_for_loss_yards AS DOUBLE) AS def_tackles_for_loss_yards,
                CAST(def_fumbles AS DOUBLE) AS def_fumbles,
                CAST(def_fumbles_forced AS DOUBLE) AS def_fumbles_forced,
                CAST(def_safeties AS DOUBLE) AS def_safeties,
                CAST(def_tds AS DOUBLE) AS def_tds,
                CAST(pts_allow AS DOUBLE) AS pts_allow,
                CAST(dst_points_allowed AS DOUBLE) AS dst_points_allowed,
                CAST(points_allowed AS DOUBLE) AS points_allowed,
                CAST(passing_yds_allowed AS DOUBLE) AS passing_yds_allowed,
                CAST(rushing_yds_allowed AS DOUBLE) AS rushing_yds_allowed,
                CAST(total_yds_allowed AS DOUBLE) AS total_yds_allowed,
                CAST(fumble_recovery_opp AS DOUBLE) AS fumble_recovery_opp,
                CAST(fumble_recovery_tds AS DOUBLE) AS fumble_recovery_tds,
                CAST(special_teams_tds AS DOUBLE) AS special_teams_tds,
                CAST(three_out AS DOUBLE) AS three_out,
                CAST(fourth_down_stop AS DOUBLE) AS fourth_down_stop
              FROM raw
            )
            SELECT
                player_key,
                year,
                MAX(player) AS player,
                MAX(nfl_position) AS nfl_position,
                ANY_VALUE(nfl_team_norm) AS nfl_team,
                MAX(headshot_url) AS headshot_url,
                STRING_AGG(DISTINCT manager_norm, ', ')
                    FILTER (WHERE manager_norm IS NOT NULL) AS manager,
                STRING_AGG(DISTINCT opponent, ', ' ORDER BY opponent) AS opponent,
                STRING_AGG(DISTINCT opp_team_norm, ', ' ORDER BY opp_team_norm) AS opponent_nfl_team,
                SUM(points) AS points,
                MAX(season_ppg) AS season_ppg,
                SUM(started_i) AS games_started,
                COUNT(*) AS fantasy_games,
                SUM(win_i) AS win,
                SUM(loss_i) AS loss,
                SUM(CASE WHEN is_playoffs_i = 1 AND win_i = 1 THEN 1 ELSE 0 END) AS playoff_wins,
                SUM(CASE WHEN is_playoffs_i = 1 AND loss_i = 1 THEN 1 ELSE 0 END) AS playoff_losses,
                SUM(champion_i) AS championships,
                SUM(team_pts_i) AS team_points,
                SUM(opp_pts_i) AS opponent_points,
                MAX(is_playoffs_i) AS is_playoffs,
                MAX(is_consolation_i) AS is_consolation,
                MAX(championship_i) AS championship,
                MAX(champion_i) AS champion,
                MAX(sacko_i) AS sacko,
                SUM(optimal_player_i) AS optimal_player,
                SUM(league_wide_optimal_player_i) AS league_wide_optimal_player,
                MIN(position_season_rank) AS position_season_rank,
                -- Passing stats
                SUM(passing_yards) AS passing_yards,
                SUM(passing_tds) AS passing_tds,
                SUM(passing_interceptions) AS passing_interceptions,
                SUM(attempts) AS attempts,
                SUM(completions) AS completions,
                SUM(passing_air_yards) AS passing_air_yards,
                SUM(passing_yards_after_catch) AS passing_yards_after_catch,
                SUM(passing_first_downs) AS passing_first_downs,
                SUM(passing_epa) AS passing_epa,
                AVG(passing_cpoe) AS passing_cpoe,
                AVG(pacr) AS pacr,
                -- Rushing stats
                SUM(rushing_yards) AS rushing_yards,
                SUM(carries) AS carries,
                SUM(rushing_tds) AS rushing_tds,
                SUM(rushing_fumbles) AS rushing_fumbles,
                SUM(rushing_fumbles_lost) AS rushing_fumbles_lost,
                SUM(rushing_first_downs) AS rushing_first_downs,
                SUM(rushing_epa) AS rushing_epa,
                SUM(rushing_2pt_conversions) AS rushing_2pt_conversions,
                -- Receiving stats
                SUM(receptions) AS receptions,
                SUM(receiving_yards) AS receiving_yards,
                SUM(receiving_tds) AS receiving_tds,
                SUM(targets) AS targets,
                SUM(receiving_fumbles) AS receiving_fumbles,
                SUM(receiving_fumbles_lost) AS receiving_fumbles_lost,
                SUM(receiving_first_downs) AS receiving_first_downs,
                SUM(receiving_epa) AS receiving_epa,
                SUM(receiving_2pt_conversions) AS receiving_2pt_conversions,
                AVG(target_share) AS target_share,
                AVG(wopr) AS wopr,
                AVG(racr) AS racr,
                SUM(receiving_air_yards) AS receiving_air_yards,
                SUM(receiving_yards_after_catch) AS receiving_yards_after_catch,
                AVG(air_yards_share) AS air_yards_share,
                -- Kicking stats
                SUM(fg_made) AS fg_made,
                SUM(fg_att) AS fg_att,
                AVG(fg_pct) AS fg_pct,
                MAX(fg_long) AS fg_long,
                SUM(fg_made_0_19) AS fg_made_0_19,
                SUM(fg_made_20_29) AS fg_made_20_29,
                SUM(fg_made_30_39) AS fg_made_30_39,
                SUM(fg_made_40_49) AS fg_made_40_49,
                SUM(fg_made_50_59) AS fg_made_50_59,
                SUM(fg_missed) AS fg_missed,
                SUM(pat_made) AS pat_made,
                SUM(pat_att) AS pat_att,
                SUM(pat_missed) AS pat_missed,
                -- Defense stats
                SUM(def_sacks) AS def_sacks,
                SUM(def_sack_yards) AS def_sack_yards,
                SUM(def_qb_hits) AS def_qb_hits,
                SUM(def_interceptions) AS def_interceptions,
                SUM(def_interception_yards) AS def_interception_yards,
                SUM(def_pass_defended) AS def_pass_defended,
                SUM(def_tackles_solo) AS def_tackles_solo,
                SUM(def_tackle_assists) AS def_tackle_assists,
                SUM(def_tackles_with_assist) AS def_tackles_with_assist,
                SUM(def_tackles_for_loss) AS def_tackles_for_loss,
                SUM(def_tackles_for_loss_yards) AS def_tackles_for_loss_yards,
                SUM(def_fumbles) AS def_fumbles,
                SUM(def_fumbles_forced) AS def_fumbles_forced,
                SUM(def_safeties) AS def_safeties,
                SUM(def_tds) AS def_tds,
                SUM(pts_allow) AS pts_allow,
                SUM(dst_points_allowed) AS dst_points_allowed,
                SUM(points_allowed) AS points_allowed,
                SUM(passing_yds_allowed) AS passing_yds_allowed,
                SUM(rushing_yds_allowed) AS rushing_yds_allowed,
                SUM(total_yds_allowed) AS total_yds_allowed,
                SUM(fumble_recovery_opp) AS fumble_recovery_opp,
                SUM(fumble_recovery_tds) AS fumble_recovery_tds,
                SUM(special_teams_tds) AS special_teams_tds,
                SUM(three_out) AS three_out,
                SUM(fourth_down_stop) AS fourth_down_stop
            FROM rows
            GROUP BY player_key, year
            ORDER BY {sort_column} {sort_direction} NULLS LAST, player
        """

        df = run_query(q)
        if df is None or df.empty:
            return pd.DataFrame()

        # CRITICAL: Remove any duplicate columns to prevent React error #185
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Add compatibility aliases
        if "nfl_position" in df.columns and "position" not in df.columns:
            df["position"] = df["nfl_position"]
        if "points" in df.columns and "total_points" not in df.columns:
            df["total_points"] = df["points"]

        df.attrs["total_count"] = len(df)
        df.attrs["offset"] = 0
        df.attrs["limit"] = len(df)
        return df

    except Exception as e:
        st.error(f"Failed to load season player data: {e}")
        return pd.DataFrame()
