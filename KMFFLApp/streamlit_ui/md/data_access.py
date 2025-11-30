#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Dict
import pandas as pd
import streamlit as st
from .motherduck_connection import MotherDuckConnection

# ---------------------------------------
# Dynamic Database Configuration
# ---------------------------------------

def get_current_league_db() -> str:
    """
    Get the current league database name from session state or environment.
    Falls back to 'kmffl' for backwards compatibility.
    """
    # Check session state first (set by the landing page)
    if "selected_league_db" in st.session_state:
        return st.session_state.selected_league_db

    # Check environment variable (set by main.py before importing)
    env_db = os.environ.get("SELECTED_LEAGUE_DB")
    if env_db:
        return env_db

    # Default fallback for backwards compatibility
    return "kmffl"


def get_table_dict() -> Dict[str, str]:
    """
    Get the table name dictionary with the current league database prefix.
    This is regenerated each time to ensure it uses the current database.
    Tables are in the 'public' schema within each database.
    """
    db = get_current_league_db()
    return {
        "matchup": f"{db}.public.matchup",
        "player": f"{db}.public.player",
        "player_season": f"{db}.public.players_by_year",
        "players_by_year": f"{db}.public.players_by_year",
        "draft": f"{db}.public.draft",
        "injury": f"{db}.public.injury",
        "schedule": f"{db}.public.schedule",
        "transactions": f"{db}.public.transactions",
    }


# For backwards compatibility, T is now a property-like accessor
# Code can still use T['matchup'] but it will dynamically get the right database
class _TableDict:
    """Dynamic table dictionary that returns prefixed table names based on current league."""

    def __getitem__(self, key: str) -> str:
        return get_table_dict()[key]

    def get(self, key: str, default: str = None) -> str:
        return get_table_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in get_table_dict()

    def keys(self):
        return get_table_dict().keys()

    def values(self):
        return get_table_dict().values()

    def items(self):
        return get_table_dict().items()


# Global T dictionary - now dynamically resolves to current league database
T = _TableDict()


# ---------------------------------------
# Connection + Query Helpers
# ---------------------------------------

@st.cache_resource
def get_motherduck_connection():
    try:
        return st.connection("motherduck", type=MotherDuckConnection)
    except Exception as e:
        st.error(f"Failed to create MotherDuck connection: {e}")
        raise

@st.cache_data(ttl=600, show_spinner=True)
def _execute_query(sql: str, retry_count: int = 0):
    """Cached query execution - uses retry_count to bust cache on retry."""
    conn = get_motherduck_connection()
    return conn.query(sql, ttl=0)  # No caching in connection layer

def run_query(sql: str, ttl: int = 600):
    """Execute query and return DataFrame with error handling and retry logic for catalog changes."""
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Use attempt number as cache buster - each retry gets fresh execution
            result = _execute_query(sql, retry_count=attempt)
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a catalog error that requires cache clearing
            if "catalog" in error_msg or "remote catalog has changed" in error_msg:
                if attempt < max_retries - 1:
                    # Clear both resource and data caches for catalog errors
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    time.sleep(0.5)
                    continue
            
            # For other errors or final attempt
            if attempt < max_retries - 1:
                st.cache_data.clear()
                time.sleep(0.3)
                continue
            else:
                # Final failure after all retries
                st.error(f"Query failed after {max_retries} attempts: {sql[:100]}... Error: {e}")
                raise

# ---------------------------------------
# SQL helpers (safe quoting + normalization)
# ---------------------------------------

def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"

# Accept Sequence[str] to match callers (they may pass Sequence)
def sql_in_list(values: Sequence[str]) -> str:
    return ", ".join(sql_quote(v) for v in values)

def sql_upper(col: str) -> str:
    return f"UPPER(NULLIF(TRIM({col}), ''))"

# Accept Sequence[str] for the upper-case in-list helper as well
def sql_upper_in_list(values: Sequence[str]) -> str:
    return ", ".join(f"UPPER({sql_quote(v)})" for v in values)

def sql_manager_norm(col: str = "manager") -> str:
    """
    Normalize manager for filters/aggregation:
      - NULL for NULL/blank/'No manager'
      - TRIM(col) for real names
    """
    return (
        f"CASE WHEN {col} IS NULL OR TRIM({col}) = '' "
        f"OR LOWER(TRIM({col})) = 'no manager' THEN NULL "
        f"ELSE TRIM({col}) END"
    )

# ---------------------------------------
# Table names (now dynamically configured - see _TableDict class above)
# ---------------------------------------

# Note: T is now defined above as _TableDict() which dynamically resolves
# table names based on the current league database selection.

# Placeholder used in SQL templates where callers may inject ORDER/BY clauses.
# Kept as empty string to avoid f-string NameError when templates are defined.
SORT_MARKER = ""

# ---------------------------------------
# Basic lists / helpers
# ---------------------------------------

def latest_season_and_week() -> Tuple[int, int]:
    sql = f"""
      SELECT year, week
      FROM {T['matchup']}
      QUALIFY ROW_NUMBER() OVER (ORDER BY cumulative_week DESC) = 1
    """
    df = run_query(sql)
    if df.empty:
        return (0, 0)
    return int(df.loc[0, "year"]), int(df.loc[0, "week"])

def list_seasons() -> Sequence[int]:
    df = run_query(f"SELECT DISTINCT year FROM {T['matchup']} ORDER BY year DESC")
    return df["year"].tolist()

def list_weeks(year: int) -> Sequence[int]:
    df = run_query(f"SELECT DISTINCT week FROM {T['matchup']} WHERE year = {int(year)} ORDER BY week")
    return df["week"].tolist()

def list_managers(year: Optional[int] = None) -> Sequence[str]:
    if year is None:
        df = run_query(f"SELECT DISTINCT manager FROM {T['matchup']} ORDER BY manager")
    else:
        df = run_query(f"SELECT DISTINCT manager FROM {T['matchup']} WHERE year = {int(year)} ORDER BY manager")
    return df["manager"].tolist()

def list_player_seasons() -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT year
        FROM {T['player']}
        WHERE year IS NOT NULL
        ORDER BY year
    """)
    return [] if df.empty else df["year"].astype(int).tolist()

def list_player_weeks(year: int) -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT week
        FROM {T['player']}
        WHERE year = {int(year)} AND week IS NOT NULL
        ORDER BY week
    """)
    return [] if df.empty else df["week"].astype(int).tolist()

def list_player_positions() -> Sequence[str]:
    """Used by the Season overview to populate the Position select."""
    df = run_query(f"""
        SELECT DISTINCT nfl_position
        FROM {T['players_by_year']}
        WHERE nfl_position IS NOT NULL
        ORDER BY nfl_position
    """)
    return [] if df.empty else df["nfl_position"].astype(str).tolist()


# ---------------------------------------
# Roster Structure Detection
# ---------------------------------------

# Position patterns for classification
STARTER_POSITIONS = ["QB", "RB", "WR", "TE", "W/R/T", "K", "DEF"]
BENCH_POSITIONS = ["BN", "IR", "IL"]


@st.cache_data(ttl=3600, show_spinner=False)
def detect_roster_structure() -> Optional[Dict]:
    """
    Detect roster configuration from the player table's lineup_position column.

    Queries ONE manager's roster from the most recent week to count position slots.
    Uses same approach as head_to_head.py for consistency.

    Returns:
        Dict with:
        - starter_count: total starter slots
        - bench_count: total bench/IR slots
        - position_counts: dict of position -> count (e.g., {'QB': 1, 'RB': 2, 'WR': 3, ...})
        - total_roster: total roster size
        - flex_positions: list of flex position names (e.g., ['W/R/T'])
    """
    try:
        # Query ONE manager's lineup positions to get roster structure
        sql = f"""
            SELECT DISTINCT lineup_position
            FROM {T['player']}
            WHERE year = (SELECT MAX(year) FROM {T['player']})
              AND week = 1
              AND lineup_position IS NOT NULL
              AND lineup_position != ''
              AND manager = (
                  SELECT manager FROM {T['player']}
                  WHERE year = (SELECT MAX(year) FROM {T['player']})
                    AND week = 1
                    AND manager IS NOT NULL
                    AND manager != ''
                  LIMIT 1
              )
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        # Parse lineup positions (QB1, RB1, RB2, WR1, WR2, WR3, W/R/T1, BN1, BN2, IR1, etc.)
        position_counts = {}
        starter_count = 0
        bench_count = 0
        flex_positions = []

        for _, row in df.iterrows():
            pos = str(row['lineup_position']).upper().strip()

            # Extract base position (remove trailing numbers)
            import re
            base_pos = re.sub(r'\d+$', '', pos)

            # Count this position
            position_counts[base_pos] = position_counts.get(base_pos, 0) + 1

            # Classify as starter or bench
            is_bench = any(pos.startswith(bp) for bp in BENCH_POSITIONS)
            if is_bench:
                bench_count += 1
            else:
                starter_count += 1
                # Track flex positions
                if '/' in base_pos and base_pos not in flex_positions:
                    flex_positions.append(base_pos)

        return {
            'starter_count': starter_count,
            'bench_count': bench_count,
            'position_counts': position_counts,
            'total_roster': starter_count + bench_count,
            'flex_positions': flex_positions
        }

    except Exception:
        return None


# ---------------------------------------
# Players - Weekly (raw)
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_player_week(year: int, week: int):
    """
    OPTIMIZED: Now uses tab_data_access.players.weekly_player_data
    which loads H2H-specific columns (16 out of 270 columns = 94% reduction)
    """
    from .tab_data_access.players import load_h2h_week_data
    return load_h2h_week_data(year, week)

@st.cache_data(show_spinner=True, ttl=600)
def load_player_two_week_slice(year: int, week: int):
    cum_query = f"""
        WITH current_cum AS (
            SELECT DISTINCT cumulative_week
            FROM {get_current_league_db()}.public.players_by_year
            WHERE year = {int(year)} AND week = {int(week)}
            LIMIT 1
        ),
        prev_cum AS (
            SELECT MAX(cumulative_week) AS prev_cumulative_week
            FROM {get_current_league_db()}.public.players_by_year
            WHERE cumulative_week < (SELECT cumulative_week FROM current_cum)
        )
        SELECT 
            (SELECT cumulative_week FROM current_cum) AS current_cum,
            (SELECT prev_cumulative_week FROM prev_cum) AS prev_cum
    """
    cum = run_query(cum_query)

    if cum.empty or cum["current_cum"].isna().all():
        return run_query(f"""
            SELECT *
            FROM {get_current_league_db()}.public.players_by_year
            WHERE year = {int(year)} AND week = {int(week)}
            ORDER BY points DESC NULLS LAST
        """)

    cur_cum = float(cum.iloc[0]["current_cum"])
    prev_cum = cum.iloc[0]["prev_cum"]

    if prev_cum is None or pd.isna(prev_cum):
        return run_query(f"""
            SELECT *
            FROM {get_current_league_db()}.public.players_by_year
            WHERE year = {int(year)} AND week = {int(week)}
            ORDER BY points DESC NULLS LAST
        """)

    return run_query(f"""
        SELECT *
        FROM {get_current_league_db()}.public.players_by_year
        WHERE cumulative_week IN ({float(prev_cum)}, {cur_cum})
        ORDER BY cumulative_week DESC, points DESC NULLS LAST
    """)

@st.cache_data(show_spinner=True, ttl=600)
def list_optimal_seasons() -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT year
        FROM {get_current_league_db()}.public.players_by_year
        WHERE COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY year
    """)
    return [] if df.empty else df["year"].astype(int).tolist()

@st.cache_data(show_spinner=True, ttl=600)
def list_optimal_weeks(year: int) -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT week
        FROM {get_current_league_db()}.public.players_by_year
        WHERE year = {int(year)}
          AND COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY week
    """)
    return [] if df.empty else df["week"].astype(int).tolist()

@st.cache_data(show_spinner=True, ttl=600)
def load_optimal_week(year: int, week: int):
    """
    OPTIMIZED: Now uses tab_data_access.players.weekly_player_data
    which loads H2H-specific columns (16 out of 270 columns = 94% reduction)
    """
    from .tab_data_access.players import load_h2h_optimal_week_data
    return load_h2h_optimal_week_data(year, week)

# ---------------------------------------
# Homepage
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_homepage_data():
    try:
        summary = {}
        summary["matchup_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['matchup']}").iloc[0]["count"]
        summary["player_count"] = run_query(f"SELECT COUNT(*) AS count FROM {get_current_league_db()}.public.players_by_year").iloc[0]["count"]
        summary["draft_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['draft']}").iloc[0]["count"]
        summary["transactions_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['transactions']}").iloc[0]["count"]
        summary["injuries_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['injury']}").iloc[0]["count"]

        latest_data = run_query(f"""
            WITH latest AS (
                SELECT year, week
                FROM {T['matchup']}
                ORDER BY year DESC, week DESC
                LIMIT 1
            )
            SELECT m.year, m.week, COUNT(*) AS games
            FROM {T['matchup']} m
            JOIN latest l ON m.year = l.year AND m.week = l.week
            GROUP BY m.year, m.week
        """)

        if not latest_data.empty:
            summary["latest_year"] = int(latest_data.iloc[0]["year"])
            summary["latest_week"] = int(latest_data.iloc[0]["week"])
            summary["latest_games"] = int(latest_data.iloc[0]["games"])
        return summary
    except Exception as e:
        st.error(f"Failed to load homepage data: {e}")
        return {"error": str(e)}

# ---------------------------------------
# Managers
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_managers_data():
    try:
        matchup_summary = run_query(f"""
            SELECT 
                year, manager, 
                COUNT(*) AS games_played,
                SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) AS losses,
                AVG(team_points) AS avg_points,
                SUM(team_points) AS total_points
            FROM {T['matchup']} 
            GROUP BY year, manager 
            ORDER BY year DESC, wins DESC
        """)

        h2h_summary = run_query(f"""
            SELECT 
                m1.manager, m2.manager AS opponent,
                COUNT(*) AS games_played,
                SUM(CASE WHEN m1.team_points > m2.team_points THEN 1 ELSE 0 END) AS wins,
                AVG(m1.team_points - m2.team_points) AS avg_margin
            FROM {T['matchup']} m1
            JOIN {T['matchup']} m2
              ON m1.year = m2.year AND m1.week = m2.week AND m1.opponent = m2.manager
            GROUP BY m1.manager, m2.manager
            ORDER BY m1.manager, wins DESC
        """)

        # Load all matchups for display
        all_matchups = run_query(f"SELECT * FROM {T['matchup']} ORDER BY year DESC, week DESC")

        return {
            "summary": matchup_summary,
            "h2h": h2h_summary,
            "recent": all_matchups,
        }
    except Exception as e:
        st.error(f"Failed to load managers data: {e}")
        return {"error": str(e)}

# ---------------------------------------
# Players - Weekly (raw list/paginated helpers for other pages)
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_players_weekly_data(
    year: int | None = None,
    week: int | None = None,
    limit: int = 100,
    offset: int = 0,
    sort_column: str = "points",
    sort_direction: str = "DESC"
):
    """
    OPTIMIZED: Now uses tab_data_access.players.weekly_player_data
    which loads 116 out of 270 columns (57% reduction)
    """
    from .tab_data_access.players import load_weekly_player_data
    return load_weekly_player_data(year, week, limit, offset, sort_column, sort_direction)

@st.cache_data(show_spinner=True, ttl=600)
def load_filtered_weekly_data(
    filters: dict,
    limit: int = 500,
    offset: int = 0,
    sort_column: str = "points",
    sort_direction: str = "DESC"
):
    """
    OPTIMIZED: Now uses tab_data_access.players.weekly_player_data
    which loads 116 out of 270 columns (57% reduction)
    """
    from .tab_data_access.players import load_filtered_weekly_player_data
    return load_filtered_weekly_player_data(filters, limit, offset, sort_column, sort_direction)

# ================================================================
# SEASON OVERVIEW (group by bridged player_key + year)
# ================================================================
def get_season_query(
    where_sql: str,
    sort_column: str,
    sort_direction: str,
) -> str:
    return f"""
        WITH raw AS (
          SELECT *
          FROM {get_current_league_db()}.public.players_by_year
          {where_sql}
        ),
        base AS (
          SELECT
            year,
            TRIM(player) AS player,
            nfl_position,
            NULLIF(TRIM(LOWER(CAST(yahoo_player_id AS VARCHAR))), '') AS yid_norm,
            NULLIF(TRIM(LOWER(CAST(NFL_player_id AS VARCHAR))), '') AS nid_norm,
            UPPER(NULLIF(TRIM(nfl_team), '')) AS nfl_team_norm,
            UPPER(NULLIF(TRIM(opponent_nfl_team), '')) AS opp_team_norm,
            CASE
              WHEN manager IS NULL OR TRIM(manager) = '' OR LOWER(TRIM(manager)) = 'no manager'
                THEN NULL
              ELSE TRIM(manager)
            END AS manager_norm,
            opponent,
            points,
            season_ppg,
            CASE WHEN started = 1 THEN 1 ELSE 0 END AS started_i,
            COALESCE(win, 0) AS win_i,
            COALESCE(CASE WHEN win = 0 THEN 1 ELSE 0 END, 0) AS loss_i,
            COALESCE(team_points, 0) AS team_pts_i,
            COALESCE(opponent_points, 0) AS opp_pts_i,
            COALESCE(is_playoffs, 0) AS is_playoffs_i,
            -- ADD ALL STAT COLUMNS HERE WITH NUMERIC CASTS:
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
        ),
        bridged AS (
          SELECT
            *,
            CASE
              WHEN nid_norm IS NOT NULL
                THEN MIN(yid_norm) OVER (PARTITION BY nid_norm)
              ELSE NULL
            END AS y_from_n,
            CASE
              WHEN yid_norm IS NOT NULL
                THEN MIN(nid_norm) OVER (PARTITION BY yid_norm)
              ELSE NULL
            END AS n_from_y
          FROM base
        ),
        rows AS (
          SELECT
            year,
            player, nfl_position, nfl_team_norm, opp_team_norm, manager_norm, opponent,
            points, season_ppg, started_i, win_i, loss_i, team_pts_i, opp_pts_i, is_playoffs_i,
            yid_norm,
            COALESCE(
              yid_norm,
              y_from_n,
              nid_norm,
              n_from_y,
              LOWER(TRIM(player))
            ) AS player_key,
            -- All stat columns are now available from base CTE
            passing_yards, passing_tds, passing_interceptions, attempts, completions,
            passing_air_yards, passing_yards_after_catch, passing_first_downs,
            passing_epa, passing_cpoe, pacr,
            rushing_yards, carries, rushing_tds, rushing_fumbles, rushing_fumbles_lost,
            rushing_first_downs, rushing_epa, rushing_2pt_conversions,
            receptions, receiving_yards, receiving_tds, targets, receiving_fumbles, receiving_fumbles_lost,
            receiving_first_downs, receiving_epa, receiving_2pt_conversions,
            target_share, wopr, racr, receiving_air_yards, receiving_yards_after_catch,
            air_yards_share,
            fg_made, fg_att, fg_pct, fg_long, fg_made_0_19, fg_made_20_29,
            fg_made_30_39, fg_made_40_49, fg_made_50_59, fg_missed,
            pat_made, pat_att, pat_missed,
            def_sacks, def_sack_yards, def_qb_hits, def_interceptions, 
            def_interception_yards, def_pass_defended,
            def_tackles_solo, def_tackle_assists, def_tackles_with_assist,
            def_tackles_for_loss, def_tackles_for_loss_yards,
            def_fumbles, def_fumbles_forced, def_safeties, def_tds,
            pts_allow, dst_points_allowed, points_allowed, passing_yds_allowed,
            rushing_yds_allowed, total_yds_allowed,
            fumble_recovery_opp, fumble_recovery_tds, special_teams_tds,
            three_out, fourth_down_stop
          FROM bridged
        )
        SELECT
            player_key,
            year,
            MAX(player)              AS player,
            MAX(nfl_position)        AS nfl_position,
            ANY_VALUE(nfl_team_norm) AS nfl_team,
            STRING_AGG(DISTINCT manager_norm, ', ')
                FILTER (WHERE manager_norm IS NOT NULL) AS manager,
            STRING_AGG(DISTINCT opponent, ', ' ORDER BY opponent) AS opponent,
            STRING_AGG(DISTINCT opp_team_norm, ', ' ORDER BY opp_team_norm) AS opponent_nfl_team,
            SUM(points)              AS points,
            MAX(season_ppg)          AS season_ppg,
            SUM(started_i)           AS games_started,
            COUNT(yid_norm)          AS fantasy_games,
            SUM(win_i)               AS win,
            SUM(team_pts_i)          AS team_points,
            SUM(opp_pts_i)           AS opponent_points,
            MAX(is_playoffs_i)       AS is_playoffs,
            -- Passing stats
            SUM(passing_yards)            AS passing_yards,
            SUM(passing_tds)             AS passing_tds,
            SUM(passing_interceptions) AS passing_interceptions,
            SUM(attempts)            AS attempts,
            SUM(completions)         AS completions,
            SUM(passing_air_yards)   AS passing_air_yards,
            SUM(passing_yards_after_catch) AS passing_yards_after_catch,
            SUM(passing_first_downs) AS passing_first_downs,
            SUM(passing_epa)         AS passing_epa,
            AVG(passing_cpoe)        AS passing_cpoe,
            AVG(pacr)                AS pacr,
            -- Rushing stats
            SUM(rushing_yards)            AS rushing_yards,
            SUM(carries)             AS carries,
            SUM(rushing_tds)             AS rushing_tds,
            SUM(rushing_fumbles)     AS rushing_fumbles,
            SUM(rushing_fumbles_lost) AS rushing_fumbles_lost,
            SUM(rushing_first_downs) AS rushing_first_downs,
            SUM(rushing_epa)         AS rushing_epa,
            SUM(rushing_2pt_conversions) AS rushing_2pt_conversions,
            -- Receiving stats
            SUM(receptions)                 AS receptions,
            SUM(receiving_yards)             AS receiving_yards,
            SUM(receiving_tds)              AS receiving_tds,
            SUM(targets)             AS targets,
            SUM(receiving_fumbles)   AS receiving_fumbles,
            SUM(receiving_fumbles_lost) AS receiving_fumbles_lost,
            SUM(receiving_first_downs) AS receiving_first_downs,
            SUM(receiving_epa)       AS receiving_epa,
            SUM(receiving_2pt_conversions) AS receiving_2pt_conversions,
            AVG(target_share)        AS target_share,
            AVG(wopr)                AS wopr,
            AVG(racr)                AS racr,
            SUM(receiving_air_yards) AS receiving_air_yards,
            SUM(receiving_yards_after_catch) AS receiving_yards_after_catch,
            AVG(air_yards_share)     AS air_yards_share,
            -- Kicking stats
            SUM(fg_made)             AS fg_made,
            SUM(fg_att)              AS fg_att,
            AVG(fg_pct)              AS fg_pct,
            MAX(fg_long)             AS fg_long,
            SUM(fg_made_0_19)        AS fg_made_0_19,
            SUM(fg_made_20_29)       AS fg_made_20_29,
            SUM(fg_made_30_39)       AS fg_made_30_39,
            SUM(fg_made_40_49)       AS fg_made_40_49,
            SUM(fg_made_50_59)       AS fg_made_50_59,
            SUM(fg_missed)           AS fg_missed,
            SUM(pat_made)            AS pat_made,
            SUM(pat_att)             AS pat_att,
            SUM(pat_missed)          AS pat_missed,
            -- Defense stats
            SUM(def_sacks)           AS def_sacks,
            SUM(def_sack_yards)      AS def_sack_yards,
            SUM(def_qb_hits)         AS def_qb_hits,
            SUM(def_interceptions)   AS def_interceptions,
            SUM(def_interception_yards) AS def_interception_yards,
            SUM(def_pass_defended)   AS def_pass_defended,
            SUM(def_tackles_solo)    AS def_tackles_solo,
            SUM(def_tackle_assists)  AS def_tackle_assists,
            SUM(def_tackles_with_assist) AS def_tackles_with_assist,
            SUM(def_tackles_for_loss) AS def_tackles_for_loss,
            SUM(def_tackles_for_loss_yards) AS def_tackles_for_loss_yards,
            SUM(def_fumbles)         AS def_fumbles,
            SUM(def_fumbles_forced)  AS def_fumbles_forced,
            SUM(def_safeties)        AS def_safeties,
            SUM(def_tds)             AS def_tds,
            SUM(pts_allow)           AS pts_allow,
            SUM(dst_points_allowed)  AS dst_points_allowed,
            SUM(points_allowed)      AS points_allowed,
            SUM(passing_yds_allowed) AS passing_yds_allowed,
            SUM(rushing_yds_allowed) AS rushing_yds_allowed,
            SUM(total_yds_allowed)   AS total_yds_allowed,
            SUM(fumble_recovery_opp) AS fumble_recovery_opp,
            SUM(fumble_recovery_tds) AS fumble_recovery_tds,
            SUM(special_teams_tds)   AS special_teams_tds,
            SUM(three_out)       AS three_out,
            SUM(fourth_down_stop)    AS fourth_down_stop
        FROM rows
        GROUP BY year, player_key
        ORDER BY {sort_column} {sort_direction} NULLS LAST, player
    """

@st.cache_data(show_spinner=True, ttl=600)
def load_players_season_data(
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
    Season view: grouped by (year, player_key).

    DEPRECATED: Import from md.tab_data_access.players instead:
        from md.tab_data_access.players import load_season_player_data

    This function kept for backwards compatibility with legacy code.
    """
    try:
        # Import here to avoid circular dependency
        from .tab_data_access.players import load_season_player_data

        df = load_season_player_data(
            position=position,
            player_query=player_query,
            manager_query=manager_query,
            manager=manager,
            nfl_team=nfl_team,
            opponent=opponent,
            opponent_nfl_team=opponent_nfl_team,
            year=year,
            rostered_only=rostered_only,
            started_only=started_only,
            exclude_postseason=exclude_postseason,
            sort_column=sort_column,
            sort_direction=sort_direction,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Add compatibility columns for legacy code
        if "nfl_position" in df.columns and "position" not in df.columns:
            df["position"] = df["nfl_position"]
        if "points" in df.columns and "total_points" not in df.columns:
            df["total_points"] = df["points"]

        # Set attributes for pagination (even though we return all rows)
        df.attrs["total_count"] = len(df)
        df.attrs["offset"] = 0
        df.attrs["limit"] = len(df)

        return df

    except Exception as e:
        st.error(f"Failed to load season player data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True, ttl=600)
def load_players_career_data(
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
    **_,
) -> pd.DataFrame:
    """
    Career aggregates using {get_current_league_db()}.public.players_by_year.
    IMPORTANT: Career groups across ALL seasons by player_key ONLY (no year).

    DEPRECATED: Import from md.tab_data_access.players instead:
        from md.tab_data_access.players import load_career_player_data

    This function kept for backwards compatibility with legacy code.
    """
    try:
        # Import here to avoid circular dependency
        from .tab_data_access.players import load_career_player_data

        df = load_career_player_data(
            position=position,
            player_query=player_query,
            manager_query=manager_query,
            manager=manager,
            nfl_team=nfl_team,
            opponent=opponent,
            opponent_nfl_team=opponent_nfl_team,
            year=year,
            rostered_only=rostered_only,
            started_only=started_only,
            exclude_postseason=exclude_postseason,
            sort_column=sort_column,
            sort_direction=sort_direction,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Add compatibility columns for legacy code
        if "nfl_position" in df.columns and "position" not in df.columns:
            df["position"] = df["nfl_position"]
        if "points" in df.columns and "total_points" not in df.columns:
            df["total_points"] = df["points"]

        # Set attributes for pagination (even though we return all rows)
        df.attrs["total_count"] = len(df)
        df.attrs["offset"] = 0
        df.attrs["limit"] = len(df)

        return df

    except Exception as e:
        st.error(f"Failed to load career player data: {e}")
        return pd.DataFrame()

# ---------------------------------------
# Draft / Transactions / Simulations / Graphs / Keepers / Team Names
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_draft_data(all_years: bool = True):
    try:
        # Check if draft table has been enriched with player stats (schema check only, no data)
        draft_cols = run_query(f"SELECT * FROM {T['draft']} WHERE 1=0").columns.tolist()
        has_total_points = 'total_fantasy_points' in draft_cols
        has_season_ppg = 'season_ppg' in draft_cols

        # Build query based on available columns
        points_agg = "AVG(total_fantasy_points) AS avg_points_scored" if has_total_points else "NULL AS avg_points_scored"
        ppg_agg = "AVG(season_ppg) AS avg_ppg" if has_season_ppg else "NULL AS avg_ppg"

        draft_summary = run_query(f"""
            SELECT
                year,
                COUNT(*) AS total_picks,
                COUNT(DISTINCT manager) AS managers,
                AVG(cost) AS avg_cost,
                MAX(cost) AS max_cost,
                COUNT(DISTINCT player) AS unique_players,
                {points_agg},
                {ppg_agg}
            FROM {T['draft']}
            WHERE cost > 0
            GROUP BY year
            ORDER BY year DESC
        """)
        latest_year = run_query(f"SELECT MAX(year) AS year FROM {T['draft']}").iloc[0]["year"]
        if all_years:
            picks = run_query(f"""
                SELECT *
                FROM {T['draft']}
                ORDER BY year DESC, round, pick
            """)
        else:
            picks = run_query(f"""
                SELECT *
                FROM {T['draft']}
                WHERE year = {int(latest_year)}
                ORDER BY round, pick
            """)
        return {
            "summary": draft_summary,
            "Draft History": picks,
            "latest_year": latest_year,
            "recent_picks": picks[picks["year"] == latest_year].copy()
        }
    except Exception as e:
        st.error(f"Failed to load draft data: {e}")
        return {"error": str(e)}

@st.cache_data(show_spinner=True, ttl=600)
def load_transactions_data(limit: Optional[int] = 1000):
    try:
        # Pull full transaction table (all source columns)
        recent_sql = f"SELECT * FROM {T['transactions']} ORDER BY year DESC, week DESC"
        if limit is not None:
            recent_sql += f" LIMIT {int(limit)}"
        df = run_query(recent_sql)

        # If empty, return empty structures
        if df is None:
            df = pd.DataFrame()
        else:
            t = df.copy()

            # Create aliases for enrichment columns to maintain backward compatibility
            # The enrichment script creates these columns with specific names
            if 'transaction_quality_score' in t.columns and 'transaction_score' not in t.columns:
                t['transaction_score'] = t['transaction_quality_score']
            if 'nfl_team' in t.columns and 'nfl_team_at_transaction' not in t.columns:
                t['nfl_team_at_transaction'] = t['nfl_team']
            if 'weeks_after' in t.columns and 'weeks_after_transaction' not in t.columns:
                t['weeks_after_transaction'] = t['weeks_after']
            if 'weeks_before' in t.columns and 'weeks_before_transaction' not in t.columns:
                t['weeks_before_transaction'] = t['weeks_before']
            if 'total_points_after_4wks' in t.columns and 'total_points_after' not in t.columns:
                t['total_points_after'] = t['total_points_after_4wks']
            if 'position_rank_after_transaction' in t.columns and 'avg_position_rank_after' not in t.columns:
                t['avg_position_rank_after'] = t['position_rank_after_transaction']

            # Ensure player_name column exists (used by UI)
            if 'player_name' not in t.columns and 'player' in t.columns:
                t['player_name'] = t['player']
            elif 'player' not in t.columns and 'player_name' in t.columns:
                t['player'] = t['player_name']

            # Canonical list of transaction columns (actual enriched columns + aliases)
            canonical = [
                "transaction_id",
                "manager",
                "player",
                "player_name",
                "position",
                "transaction_type",
                "faab_bid",
                "week",
                "year",
                "cumulative_week",
                # Enrichment columns (from player_to_transactions_v2.py)
                "nfl_team",
                "nfl_team_at_transaction",
                "points_at_transaction",
                "ppg_before_transaction",
                "weeks_before",
                "weeks_before_transaction",
                "ppg_after_transaction",
                "total_points_after_4wks",
                "total_points_after",
                "weeks_after",
                "weeks_after_transaction",
                "total_points_rest_of_season",
                "ppg_rest_of_season",
                "weeks_rest_of_season",
                "position_rank_at_transaction",
                "position_rank_before_transaction",
                "position_rank_after_transaction",
                "avg_position_rank_after",
                "position_total_players",
                "points_per_faab_dollar",
                "transaction_quality_score",
                "transaction_score",
                # Original transaction columns
                "week_start",
                "week_end",
                "source_type",
                "destination",
                "status",
                "human_readable_timestamp",
                "manager_week",
                "manager_year",
                "player_week",
                "player_year",
                "yahoo_player_id",
                "player_key",
                "timestamp",
                "transaction_datetime",
                "transaction_sequence",
                "league_id",
            ]

            # Add missing canonical columns as NA to stabilize schema
            for c in canonical:
                if c not in t.columns:
                    t[c] = pd.NA

            # Prefer canonical `transaction_id`. If only `action_id` exists, create transaction_id from it
            if "transaction_id" not in t.columns or t["transaction_id"].isna().all():
                if "action_id" in t.columns:
                    t["transaction_id"] = t["action_id"].astype("string")
                else:
                    t["transaction_id"] = pd.NA

            # Normalize player to plain Python str
            if "player" in t.columns:
                t["player"] = t["player"].astype("string").fillna("").astype("str")

            # Coerce numeric-ish columns
            for c in ("week", "year", "cumulative_week", "manager_week", "manager_year", "player_week", "player_year"):
                if c in t.columns:
                    t[c] = pd.to_numeric(t[c], errors="coerce").astype("Int64")
            for ncol in ("faab_bid", "points_at_transaction", "total_points_after", "transaction_score", "points_per_faab_dollar"):
                if ncol in t.columns:
                    t[ncol] = pd.to_numeric(t[ncol], errors="coerce")

            # Ensure manager has a sensible default
            if "manager" in t.columns:
                t["manager"] = t["manager"].fillna("Unknown")

            # Keep `action_id` if present but ensure `transaction_id` is the primary id column
            # (we already created/ensured `transaction_id` above)

            df = t

        transaction_summary = run_query(f"""
            SELECT 
                manager, year,
                COUNT(*) AS total_transactions,
                SUM(CASE WHEN LOWER(transaction_type) = 'add' THEN 1 ELSE 0 END) AS adds,
                SUM(CASE WHEN LOWER(transaction_type) = 'drop' THEN 1 ELSE 0 END) AS drops,
                SUM(CASE WHEN LOWER(transaction_type) = 'trade' THEN 1 ELSE 0 END) AS trades
            FROM {T['transactions']} 
            GROUP BY manager, year 
            ORDER BY year DESC, total_transactions DESC
        """)

        # Load related data with reasonable limits for performance
        player_data = run_query(f"SELECT * FROM {get_current_league_db()}.public.players_by_year ORDER BY year DESC, week DESC LIMIT 10000")
        injury_data = run_query(f"SELECT * FROM {T['injury']} ORDER BY year DESC, week DESC LIMIT 10000")
        # Limit draft data to recent years (last ~5 seasons assuming 10 teams x 18 rounds = 180 picks/year)
        draft_data = run_query(f"SELECT * FROM {T['draft']} ORDER BY year DESC, round, pick LIMIT 1000")

        return {
            "transactions": df,
            "summary": transaction_summary,
            "player_data": player_data,
            "injury_data": injury_data,
            "draft_data": draft_data
        }
    except Exception as e:
        st.error(f"Failed to load transactions data: {e}")
        return {"error": str(e)}

@st.cache_data(show_spinner=True, ttl=600)
def load_simulations_data(include_all_years: bool = True, max_rows: int = None):
    """
    Load simulation data with optional row limit for performance.

    Args:
        include_all_years: If True, load all years; if False, current year only
        max_rows: Maximum matchup rows to load (default None for unlimited)
    """
    try:
        # Get current year once (used in both branches)
        current_year = run_query(f"SELECT MAX(year) AS year FROM {T['matchup']}").iloc[0]["year"]

        if include_all_years:
            limit_clause = f" LIMIT {max_rows}" if max_rows else ""
            matchups = run_query(f"SELECT * FROM {T['matchup']} ORDER BY year DESC, week DESC{limit_clause}")
            player_averages = run_query(f"""
                SELECT
                    player, manager, nfl_position AS position, year,
                    AVG(points) AS avg_points, STDDEV(points) AS std_points,
                    COUNT(*) AS games_played
                FROM {get_current_league_db()}.public.players_by_year
                GROUP BY player, manager, nfl_position, year
                HAVING COUNT(*) >= 3
                ORDER BY year DESC, avg_points DESC
            """)
        else:
            matchups = run_query(f"SELECT * FROM {T['matchup']} WHERE year = {int(current_year)} ORDER BY week DESC")
            player_averages = run_query(f"""
                SELECT 
                    player, manager, nfl_position AS position,
                    AVG(points) AS avg_points, STDDEV(points) AS std_points,
                    COUNT(*) AS games_played
                FROM {get_current_league_db()}.public.players_by_year
                WHERE year = {int(current_year)}
                GROUP BY player, manager, nfl_position
                HAVING COUNT(*) >= 3
            """)
        return {"matchups": matchups, "player_averages": player_averages, "current_year": current_year}
    except Exception as e:
        st.error(f"Failed to load simulations data: {e}")
        return {"error": str(e)}

@st.cache_data(show_spinner=True, ttl=600)
def load_graphs_data():
    try:
        season_totals = run_query(f"""
            SELECT 
                year, manager,
                SUM(team_points) AS total_points,
                AVG(team_points) AS avg_points,
                COUNT(*) AS games_played,
                SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) AS wins
            FROM {T['matchup']} 
            GROUP BY year, manager
            ORDER BY year, total_points DESC
        """)
        weekly_trends = run_query(f"""
            SELECT 
                year, week, manager, team_points,
                AVG(team_points) OVER (
                    PARTITION BY manager, year 
                    ORDER BY week 
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS rolling_avg
            FROM {T['matchup']} 
            ORDER BY year, week, manager
        """)
        return {"season_totals": season_totals, "weekly_trends": weekly_trends}
    except Exception as e:
        st.error(f"Failed to load graphs data: {e}")
        return {"error": str(e)}

@st.cache_data(show_spinner=True, ttl=600)
def load_keepers_data(all_years: bool = True, year: int = None, week: int = None):
    """
    Load keeper data from the player table.

    - Uses max_week (end of season) to determine who owned/kept the player
    - is_keeper_status: Indicates keeper eligibility (1 = is a keeper)
    - keeper_price: Cost to keep for next year
    - avg_points_this_year = season_ppg for current year
    - avg_points_next_year = season_ppg for player in year+1

    Note: Manager shown is the END OF SEASON owner (max_week), as they're the one
    who actually makes the keeper decision and owns the player going into next year.
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
                    "JOIN last_weeks lw ON p.yahoo_player_id = lw.yahoo_player_id"
                    " AND p.year = lw.year AND p.week = lw.max_week"
                )
                week_filter = ""
            else:
                # end-of-season for latest year only
                join_lw = (
                    "JOIN last_weeks lw ON p.yahoo_player_id = lw.yahoo_player_id"
                    " AND p.year = lw.year AND p.week = lw.max_week"
                )
                week_filter = f"AND p.year = (SELECT MAX(year) FROM {T['player']})"

        if use_cte:
            keeper_query = f"""
                WITH last_weeks AS (
                    SELECT yahoo_player_id, year, MAX(week) AS max_week
                    FROM {T['player']}
                    GROUP BY yahoo_player_id, year
                )
                SELECT
                    p.player,
                    p.manager,
                    p.nfl_position AS yahoo_position,
                    p.nfl_team,
                    p.year,
                    p.week,
                    p.points,
                    CASE WHEN p.kept_next_year = 1 THEN true ELSE false END AS kept_next_year,
                    CASE WHEN p.is_keeper_status = 1 THEN true ELSE false END AS is_keeper_status,
                    COALESCE(p.keeper_price, 0) AS keeper_price,
                    COALESCE(p.season_ppg, 0) AS avg_points_this_year,
                    COALESCE(p_next.season_ppg, 0) AS avg_points_next_year,
                    COALESCE(p.cost, 0) AS cost,
                    COALESCE(p.faab_bid, 0) AS faab_bid,
                    p.yahoo_player_id,
                    p.fantasy_position
                FROM {T['player']} p
                {join_lw}
                LEFT JOIN {T['player']} p_next 
                    ON p.yahoo_player_id = p_next.yahoo_player_id
                    AND p.year + 1 = p_next.year
                LEFT JOIN last_weeks lw_next
                    ON p_next.yahoo_player_id = lw_next.yahoo_player_id
                    AND p_next.year = lw_next.year
                    AND p_next.week = lw_next.max_week
                WHERE p.manager IS NOT NULL 
                  AND LOWER(TRIM(p.manager)) != 'no manager'
                  AND (p.nfl_position NOT IN ('DEF', 'K') OR p.nfl_position IS NULL)
                  {week_filter}
                ORDER BY p.year DESC, p.week DESC, p.manager, p.points DESC
            """
        else:
            # specific week simple path
            keeper_query = f"""
                SELECT
                    p.player,
                    p.manager,
                    p.nfl_position AS yahoo_position,
                    p.nfl_team,
                    p.year,
                    p.week,
                    p.points,
                    CASE WHEN p.kept_next_year = 1 THEN true ELSE false END AS kept_next_year,
                    CASE WHEN p.is_keeper_status = 1 THEN true ELSE false END AS is_keeper_status,
                    COALESCE(p.keeper_price, 0) AS keeper_price,
                    COALESCE(p.season_ppg, 0) AS avg_points_this_year,
                    COALESCE(p_next.season_ppg, 0) AS avg_points_next_year,
                    COALESCE(p.cost, 0) AS cost,
                    COALESCE(p.faab_bid, 0) AS faab_bid,
                    p.yahoo_player_id,
                    p.fantasy_position
                FROM {T['player']} p
                LEFT JOIN {T['player']} p_next 
                    ON p.yahoo_player_id = p_next.yahoo_player_id
                    AND p.year + 1 = p_next.year
                    AND p_next.week = (
                        SELECT MAX(week) FROM {T['player']} WHERE yahoo_player_id = p_next.yahoo_player_id AND year = p_next.year
                    )
                WHERE p.manager IS NOT NULL 
                  AND LOWER(TRIM(p.manager)) != 'no manager'
                  AND (p.nfl_position NOT IN ('DEF', 'K') OR p.nfl_position IS NULL)
                  {week_filter}
                ORDER BY p.year DESC, p.week DESC, p.manager, p.points DESC
            """

        keeper_raw_data = run_query(keeper_query)

        return keeper_raw_data
    except Exception as e:
        st.error(f"Failed to load keepers data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_data(show_spinner=True, ttl=600)
def load_team_names_data():
    """
    DEPRECATED: Use md.tab_data_access.team_names.load_optimized_team_names_data() instead.

    This function loads from the matchup table with team_name column.
    """
    try:
        # Load from matchup table with team_name column
        q_matchup = f"""
            SELECT DISTINCT manager, year, team_name, division_id, league_id
            FROM {T['matchup']}
            WHERE manager IS NOT NULL
              AND LOWER(TRIM(manager)) NOT IN ('no manager', 'unrostered', '')
            ORDER BY year DESC, manager
        """
        team_names = run_query(q_matchup)

        return team_names
    except Exception as e:
        st.error(f"Failed to load team names data: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_draft_optimizer_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Optimized loader for draft optimizer - aggregates at DB level.
    Returns position/cost_bucket aggregates only, not individual draft picks.
    """
    try:
        q = f"""
            SELECT
                primary_position,
                cost_bucket,
                AVG(cost) AS cost,
                MEDIAN(season_ppg) AS PPG
            FROM {T['draft']}
            WHERE year BETWEEN {int(start_year)} AND {int(end_year)}
              AND cost > 0
              AND cost_bucket > 0
              AND season_ppg > 0
              AND (is_keeper_status IS NULL OR is_keeper_status != 1)
            GROUP BY primary_position, cost_bucket
            HAVING AVG(cost) > 0 AND MEDIAN(season_ppg) > 0
            ORDER BY primary_position, cost_bucket
        """
        df = run_query(q)
        if df.empty:
            return pd.DataFrame(columns=['primary_position', 'cost_bucket', 'cost', 'PPG'])

        # Clean dtypes
        df['cost'] = df['cost'].round(2).astype(float)
        df['PPG'] = df['PPG'].round(2).astype(float)
        df['cost_bucket'] = df['cost_bucket'].astype(int)

        return df
    except Exception as e:
        st.error(f"Failed to load draft optimizer data: {e}")
        return pd.DataFrame(columns=['primary_position', 'cost_bucket', 'cost', 'PPG'])
