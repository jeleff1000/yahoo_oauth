#!/usr/bin/env python3
"""
Career player data loader for Player Stats tab.

Optimization: Loads only ~93 out of 237 columns from the players_by_year table.
This reduces data transfer by ~61% and memory usage significantly.

Comprehensively analyzed ALL career player subprocesses:
- career_player_basic_stats.py
- career_player_advanced_stats.py
- career_player_matchup_stats.py

IMPORTANT: All column names verified against actual players_by_year table schema.
Only includes columns that actually exist in the database.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from md.core import run_query, T, sql_in_list

# Columns needed for career player stats (~113 out of 237 columns = 52% reduction)
# All columns verified to exist in actual players_by_year table
CAREER_PLAYER_SOURCE_COLUMNS = [
    # === Identifiers (8) ===
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

    # === Core Stats (3) ===
    # Note: total_points, ppg_all_time, games_started, games_played, fantasy_games are computed in aggregation
    "points",
    "season_ppg",
    "is_started",

    # === SPAR Metrics (3) ===
    "spar",             # Legacy SPAR
    "player_spar",      # Total SPAR produced
    "manager_spar",     # SPAR while on manager's roster

    # === Matchup Context (12) ===
    "win",
    "loss",
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

    # === Passing Stats (11) ===
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_cpoe",
    "pacr",

    # === Rushing Stats (8) ===
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",

    # === Receiving Stats (14) ===
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_2pt_conversions",
    "target_share",
    "wopr",
    "racr",
    "air_yards_share",

    # === Kicker Stats (13) ===
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

    # === Defense/IDP Stats (14) ===
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

    # === DST Stats (9) ===
    "pts_allow",
    "dst_points_allowed",
    "points_allowed",
    "passing_yds_allowed",
    "rushing_yds_allowed",
    "total_yds_allowed",
    "fumble_recovery_opp",
    "fumble_recovery_tds",
    "special_teams_tds",

    # === Other (2) ===
    "season_type",
    "passing_2pt_conversions",
]


@st.cache_data(show_spinner=True, ttl=120)
def load_career_player_data(
    position=None,
    player_query="",
    manager_query="",
    manager=None,
    nfl_team=None,
    opponent=None,
    opponent_nfl_team=None,
    year=None,
    rostered_only=False,
    started_only=False,
    exclude_postseason=False,
    sort_column="points",
    sort_direction="DESC"
) -> pd.DataFrame:
    """
    Load career player data with ONLY the columns needed.

    Career stats aggregate across ALL years for each player (GROUP BY player_key only).

    Args:
        position: Position filter (filters on nfl_position)
        player_query: Player name search
        manager_query: Manager name search
        manager: List of managers
        nfl_team: List of NFL teams
        opponent: List of opponent managers
        opponent_nfl_team: List of opponent NFL teams
        year: List of years to include (before aggregation)
        rostered_only: Only show rostered players
        started_only: Only show started players
        exclude_postseason: Exclude postseason games
        sort_column: Column to sort by
        sort_direction: ASC or DESC

    Returns:
        DataFrame with career player data aggregated across all years

    Columns loaded (~113 out of 237 = 52% reduction):
        Identifiers (8): player, IDs, teams, manager info
        Time (2): year, week (for pre-aggregation filtering)
        Positions (2): nfl_position, fantasy_position (used to compute fantasy_games)
        Core (3): points, season_ppg, is_started
        Matchup (12): wins, losses, team points, playoffs, consolation, playoff_round, consolation_round, championship, champion, sacko, most_advanced_round, optimal_player
        Passing (11): attempts, completions, yards, TDs, INTs, air yards, EPA, CPOE, PACR
        Rushing (8): carries, yards, TDs, fumbles, fumbles_lost, first downs, EPA, 2PT conversions
        Receiving (14): receptions, targets, yards, TDs, fumbles, fumbles_lost, EPA, target share, WOPR, RACR, air yards

        Note: fum_lost computed as rushing_fumbles_lost + receiving_fumbles_lost
        Kicking (13): FG stats by distance, XP stats, percentages (fg_missed → output as fg_miss)
        Defense/IDP (13): sacks, sack_yards, tackles, INTs, fumbles, TFL, pass defended, safeties, TDs
        DST (9): points allowed, yards allowed, fumble recoveries, special teams TDs
        Other (2): season_type, passing_2pt_conversions
    """
    try:
        # Build WHERE clause for filtering BEFORE aggregation
        where = []

        # Position filter
        if position and position != "All":
            if position in ["QB", "RB", "WR", "TE", "K", "DEF"]:
                where.append(f"nfl_position = '{position}'")

        # Year filter (pre-aggregation)
        if year and isinstance(year, list) and year:
            year_list = ", ".join([str(int(y)) for y in year if str(y).strip()])
            if year_list:
                where.append(f"year IN ({year_list})")

        # Player search
        if player_query and player_query.strip():
            esc = player_query.strip().replace("'", "''")
            where.append(f"LOWER(player) LIKE LOWER('%{esc}%')")

        # Rostered only
        if rostered_only:
            where.append("manager IS NOT NULL AND manager <> '' AND manager <> 'Unrostered'")

        # Started only
        if started_only:
            where.append("CAST(is_started AS INTEGER) = 1")

        # Exclude postseason
        if exclude_postseason:
            where.append("(season_type IS NULL OR season_type = 'REG')")

        # Manager filter
        if manager and isinstance(manager, list) and manager:
            where.append(f"manager IN ({sql_in_list(manager)})")

        # Manager search
        if manager_query and manager_query.strip():
            esc = manager_query.strip().replace("'", "''")
            where.append(f"LOWER(manager) LIKE LOWER('%{esc}%')")

        # NFL team filter
        if nfl_team and isinstance(nfl_team, list) and nfl_team:
            where.append(f"nfl_team IN ({sql_in_list(nfl_team)})")

        # Opponent manager filter
        if opponent and isinstance(opponent, list) and opponent:
            where.append(f"opponent IN ({sql_in_list(opponent)})")

        # Opponent NFL team filter
        if opponent_nfl_team and isinstance(opponent_nfl_team, list) and opponent_nfl_team:
            where.append(f"opponent_nfl_team IN ({sql_in_list(opponent_nfl_team)})")

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        # Build column list for SELECT clause
        cols_str = ", ".join(CAREER_PLAYER_SOURCE_COLUMNS)

        # Career aggregation: GROUP BY player_key only (no year)
        # IMPORTANT: Use NFL_player_id as primary key to avoid incorrect merging
        # The old bridging logic caused issues with players sharing partial names
        # (e.g., "St. Brown" brothers, or "Bay" in team names)
        sql = f"""
            WITH raw AS (
              SELECT {cols_str}
              FROM {T['players_by_year']}
              {where_sql}
            ),
            rows AS (
              SELECT
                TRIM(player) AS player,
                yahoo_player_id,
                NFL_player_id,
                -- Use NFL_player_id as primary key, fallback to yahoo_player_id, then name
                -- This prevents incorrect merging of different players
                COALESCE(
                  NULLIF(TRIM(CAST(NFL_player_id AS VARCHAR)), ''),
                  NULLIF(TRIM(CAST(yahoo_player_id AS VARCHAR)), ''),
                  LOWER(TRIM(player))
                ) AS player_key,
                nfl_team,
                nfl_position,
                fantasy_position,
                headshot_url,
                manager,
                opponent,
                opponent_nfl_team,
                year,
                week,
                points,
                season_ppg,
                is_started,
                win,
                loss,
                team_points,
                opponent_points,
                is_playoffs,
                is_consolation,
                playoff_round,
                consolation_round,
                championship,
                champion,
                sacko,
                optimal_player,
                league_wide_optimal_player,
                season_type,
                completions,
                attempts,
                passing_yards,
                passing_tds,
                passing_interceptions,
                passing_air_yards,
                passing_yards_after_catch,
                passing_first_downs,
                passing_epa,
                passing_cpoe,
                pacr,
                passing_2pt_conversions,
                carries,
                rushing_yards,
                rushing_tds,
                rushing_fumbles,
                rushing_fumbles_lost,
                rushing_first_downs,
                rushing_epa,
                rushing_2pt_conversions,
                receptions,
                targets,
                receiving_yards,
                receiving_tds,
                receiving_fumbles,
                receiving_fumbles_lost,
                receiving_first_downs,
                receiving_epa,
                receiving_air_yards,
                receiving_yards_after_catch,
                receiving_2pt_conversions,
                target_share,
                wopr,
                racr,
                air_yards_share,
                fg_made,
                fg_att,
                fg_pct,
                fg_long,
                fg_made_0_19,
                fg_made_20_29,
                fg_made_30_39,
                fg_made_40_49,
                fg_made_50_59,
                fg_missed,
                pat_made,
                pat_att,
                pat_missed,
                def_sacks,
                def_sack_yards,
                def_qb_hits,
                def_interceptions,
                def_interception_yards,
                def_pass_defended,
                def_tackles_solo,
                def_tackle_assists,
                def_tackles_with_assist,
                def_tackles_for_loss,
                def_tackles_for_loss_yards,
                def_fumbles,
                def_fumbles_forced,
                def_safeties,
                def_tds,
                pts_allow,
                dst_points_allowed,
                points_allowed,
                passing_yds_allowed,
                rushing_yds_allowed,
                total_yds_allowed,
                fumble_recovery_opp,
                fumble_recovery_tds,
                special_teams_tds
              FROM raw
            )
            SELECT
                player_key,
                MAX(player) AS player,
                MAX(yahoo_player_id) AS yahoo_player_id,
                MAX(NFL_player_id) AS NFL_player_id,
                ANY_VALUE(nfl_team) AS nfl_team,
                MAX(nfl_position) AS nfl_position,
                MAX(fantasy_position) AS fantasy_position,
                MAX(headshot_url) AS headshot_url,
                STRING_AGG(DISTINCT manager, ', ') FILTER (WHERE manager IS NOT NULL AND manager <> '' AND manager <> 'Unrostered') AS manager,

                -- Career aggregates (SUM across all years)
                SUM(points) AS points,
                SUM(points) AS total_points,
                AVG(season_ppg) AS ppg_all_time,
                SUM(CASE WHEN CAST(is_started AS INTEGER) = 1 THEN 1 ELSE 0 END) AS games_started,
                COUNT(CASE WHEN fantasy_position IS NOT NULL
                           AND TRIM(fantasy_position) <> ''
                           AND UPPER(TRIM(fantasy_position)) NOT IN ('BN', 'IR')
                      THEN 1 END) AS fantasy_games,
                COUNT(CASE WHEN fantasy_position IS NOT NULL
                           AND TRIM(fantasy_position) <> ''
                           AND UPPER(TRIM(fantasy_position)) NOT IN ('BN', 'IR')
                      THEN 1 END) AS games_played,
                -- Roster status counts
                COUNT(CASE WHEN manager IS NOT NULL AND manager <> '' AND manager <> 'Unrostered' THEN 1 END) AS weeks_rostered,
                COUNT(CASE WHEN UPPER(TRIM(fantasy_position)) = 'BN' THEN 1 END) AS weeks_on_bench,
                COUNT(CASE WHEN UPPER(TRIM(fantasy_position)) = 'IR' THEN 1 END) AS weeks_on_ir,
                SUM(win) AS win,
                SUM(loss) AS loss,
                SUM(CASE WHEN is_playoffs = 1 AND win = 1 THEN 1 ELSE 0 END) AS playoff_wins,
                SUM(CASE WHEN is_playoffs = 1 AND loss = 1 THEN 1 ELSE 0 END) AS playoff_losses,
                SUM(CASE WHEN champion = 1 THEN 1 ELSE 0 END) AS championships,
                SUM(team_points) AS team_points,
                SUM(opponent_points) AS opponent_points,
                SUM(CASE WHEN is_playoffs = 1 THEN 1 ELSE 0 END) AS is_playoffs,
                SUM(optimal_player) AS optimal_player,
                SUM(league_wide_optimal_player) AS league_wide_optimal_player,

                -- Passing stats
                SUM(completions) AS completions,
                SUM(attempts) AS attempts,
                SUM(passing_yards) AS pass_yds,
                SUM(passing_tds) AS pass_td,
                SUM(passing_interceptions) AS passing_interceptions,
                SUM(passing_air_yards) AS passing_air_yards,
                SUM(passing_yards_after_catch) AS passing_yards_after_catch,
                SUM(passing_first_downs) AS passing_first_downs,
                SUM(passing_epa) AS passing_epa,
                AVG(passing_cpoe) AS passing_cpoe,
                AVG(pacr) AS pacr,
                SUM(passing_2pt_conversions) AS passing_2pt_conversions,

                -- Rushing stats
                SUM(carries) AS rush_att,
                SUM(rushing_yards) AS rush_yds,
                SUM(rushing_tds) AS rush_td,
                SUM(rushing_fumbles) AS rushing_fumbles,
                SUM(rushing_fumbles_lost) AS rushing_fumbles_lost,
                SUM(rushing_first_downs) AS rushing_first_downs,
                SUM(rushing_epa) AS rushing_epa,
                SUM(rushing_2pt_conversions) AS rushing_2pt_conversions,

                -- Receiving stats
                SUM(receptions) AS rec,
                SUM(targets) AS targets,
                SUM(receiving_yards) AS rec_yds,
                SUM(receiving_tds) AS rec_td,
                SUM(receiving_fumbles) AS receiving_fumbles,
                SUM(receiving_fumbles_lost) AS receiving_fumbles_lost,
                SUM(receiving_first_downs) AS receiving_first_downs,
                SUM(receiving_epa) AS receiving_epa,
                SUM(receiving_air_yards) AS receiving_air_yards,
                SUM(receiving_yards_after_catch) AS receiving_yards_after_catch,
                SUM(receiving_2pt_conversions) AS receiving_2pt_conversions,
                AVG(target_share) AS target_share,
                AVG(wopr) AS wopr,
                AVG(racr) AS racr,
                AVG(air_yards_share) AS air_yards_share,

                -- Total fumbles lost (computed from rushing + receiving)
                SUM(rushing_fumbles_lost) + SUM(receiving_fumbles_lost) AS fum_lost,

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
                SUM(fg_missed) AS fg_miss,
                SUM(pat_made) AS pat_made,
                SUM(pat_att) AS pat_att,
                SUM(pat_missed) AS pat_missed,

                -- Defense/IDP stats
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

                -- DST stats
                SUM(pts_allow) AS pts_allow,
                SUM(dst_points_allowed) AS dst_points_allowed,
                SUM(points_allowed) AS points_allowed,
                SUM(passing_yds_allowed) AS pass_yds_allowed,
                SUM(rushing_yds_allowed) AS rushing_yds_allowed,
                SUM(total_yds_allowed) AS total_yds_allowed,
                SUM(fumble_recovery_opp) AS fumble_recovery_opp,
                SUM(fumble_recovery_tds) AS fumble_recovery_tds,
                SUM(special_teams_tds) AS special_teams_tds

              FROM rows
              GROUP BY player_key
              ORDER BY {sort_column} {sort_direction} NULLS LAST, player
        """

        result = run_query(sql)

        # CRITICAL: Remove any duplicate columns to prevent React error #185
        if result is not None and not result.empty:
            if result.columns.duplicated().any():
                result = result.loc[:, ~result.columns.duplicated(keep='first')]

        return result

    except Exception as e:
        st.error(f"Failed to load career player data: {e}")
        return pd.DataFrame()
