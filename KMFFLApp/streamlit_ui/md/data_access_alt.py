#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, List
import pandas as pd
import streamlit as st
from .motherduck_connection import MotherDuckConnection

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

def run_query(sql: str, ttl: int = 600):
    """Execute query and return DataFrame with error handling."""
    try:
        conn = get_motherduck_connection()
        return conn.query(sql, ttl=ttl)
    except Exception as e:
        st.error(f"Query failed: {sql[:100]}... Error: {e}")
        st.cache_resource.clear()
        conn = get_motherduck_connection()
        return conn.query(sql, ttl=ttl)

# ---------------------------------------
# SQL helpers (safe quoting + normalization)
# ---------------------------------------

def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"

def sql_in_list(values: list) -> str:
    return ", ".join(sql_quote(v) for v in values)

def sql_upper(col: str) -> str:
    return f"UPPER(NULLIF(TRIM({col}), ''))"

def sql_upper_in_list(values: list[str]) -> str:
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
# Table names
# ---------------------------------------

T: Dict[str, str] = {
    "matchup": "kmffl.matchup",
    "player": "kmffl.player",
    "player_season": "kmffl.players_by_year",  # not used by season agg; kept for reference
    "draft": "kmffl.draft",
    "injury": "kmffl.injury",
    "schedule": "kmffl.schedule",
    "transactions": "kmffl.transactions",
}

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
    df = run_query("""
        SELECT DISTINCT nfl_position
        FROM kmffl.player
        WHERE nfl_position IS NOT NULL
        ORDER BY nfl_position
    """)
    return [] if df.empty else df["nfl_position"].astype(str).tolist()

# ---------------------------------------
# Players - Weekly (raw)
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_player_week(year: int, week: int):
    q = f"""
        SELECT *
        FROM {T['player']}
        WHERE year = {int(year)} AND week = {int(week)}
        ORDER BY points DESC NULLS LAST
    """
    return run_query(q)

@st.cache_data(show_spinner=True, ttl=600)
def load_player_two_week_slice(year: int, week: int):
    cum_query = f"""
        WITH current_cum AS (
            SELECT DISTINCT cumulative_week
            FROM {T['player']}
            WHERE year = {int(year)} AND week = {int(week)}
            LIMIT 1
        ),
        prev_cum AS (
            SELECT MAX(cumulative_week) AS prev_cumulative_week
            FROM {T['player']}
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
            FROM {T['player']}
            WHERE year = {int(year)} AND week = {int(week)}
            ORDER BY points DESC NULLS LAST
        """)

    cur_cum = float(cum.iloc[0]["current_cum"])
    prev_cum = cum.iloc[0]["prev_cum"]

    if prev_cum is None or pd.isna(prev_cum):
        return run_query(f"""
            SELECT *
            FROM {T['player']}
            WHERE year = {int(year)} AND week = {int(week)}
            ORDER BY points DESC NULLS LAST
        """)

    return run_query(f"""
        SELECT *
        FROM {T['player']}
        WHERE cumulative_week IN ({float(prev_cum)}, {cur_cum})
        ORDER BY cumulative_week DESC, points DESC NULLS LAST
    """)

@st.cache_data(show_spinner=True, ttl=600)
def list_optimal_seasons() -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT year
        FROM {T['player']}
        WHERE COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY year
    """)
    return [] if df.empty else df["year"].astype(int).tolist()

@st.cache_data(show_spinner=True, ttl=600)
def list_optimal_weeks(year: int) -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT week
        FROM {T['player']}
        WHERE year = {int(year)}
          AND COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY week
    """)
    return [] if df.empty else df["week"].astype(int).tolist()

@st.cache_data(show_spinner=True, ttl=600)
def load_optimal_week(year: int, week: int):
    q = f"""
        SELECT *
        FROM {T['player']}
        WHERE year = {int(year)}
          AND week = {int(week)}
          AND COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY CASE position
                   WHEN 'QB' THEN 0
                   WHEN 'RB' THEN 1
                   WHEN 'WR' THEN 2
                   WHEN 'TE' THEN 3
                   WHEN 'W/R/T' THEN 4
                   WHEN 'K' THEN 5
                   WHEN 'DEF' THEN 6
                   ELSE 999
                 END,
                 points DESC
    """
    return run_query(q)

# ---------------------------------------
# Homepage
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_homepage_data():
    try:
        summary = {}
        summary["matchup_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['matchup']} WHERE league_id = '{LEAGUE_ID}'").iloc[0]["count"]
        summary["player_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['player']} WHERE (league_id = '{LEAGUE_ID}' OR league_id IS NULL)").iloc[0]["count"]
        summary["draft_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['draft']} WHERE league_id = '{LEAGUE_ID}'").iloc[0]["count"]
        summary["transactions_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['transactions']} WHERE league_id = '{LEAGUE_ID}'").iloc[0]["count"]
        summary["injuries_count"] = run_query(f"SELECT COUNT(*) AS count FROM {T['injury']} WHERE league_id = '{LEAGUE_ID}'").iloc[0]["count"]

        latest_data = run_query(f"""
            SELECT year, week, COUNT(*) AS games
            FROM {T['matchup']}
            WHERE league_id = '{LEAGUE_ID}'
            GROUP BY year, week
            ORDER BY cumulative_week DESC
            LIMIT 1
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
            WHERE league_id = '{LEAGUE_ID}'
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
              ON m1.league_id = m2.league_id
              AND m1.year = m2.year
              AND m1.week = m2.week
              AND m1.opponent = m2.manager
            WHERE m1.league_id = '{LEAGUE_ID}'
            GROUP BY m1.manager, m2.manager
            ORDER BY m1.manager, wins DESC
        """)

        all_matchups = run_query(f"SELECT * FROM {T['matchup']} WHERE league_id = '{LEAGUE_ID}' ORDER BY cumulative_week DESC")

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
    try:
        where = []
        if year is not None:
            where.append(f"year = {int(year)}")
        if week is not None:
            where.append(f"week = {int(week)}")
        where_sql = "WHERE " + " AND ".join(where) if where else ""

        total_count = run_query(f"SELECT COUNT(*) AS total FROM {T['player']} {where_sql}").iloc[0]["total"]

        numeric_cols = {"points","pass_yds","pass_td","rush_yds","rush_td","rec_yds","rec_td","rec"}
        if sort_column in numeric_cols:
            order_by = f"ORDER BY {sort_column} {sort_direction} NULLS LAST"
        else:
            order_by = f"ORDER BY {sort_column} {sort_direction}"

        if sort_column != "points":
            order_by += ", points DESC NULLS LAST"
        order_by += ", player"

        sql = f"""
            SELECT *
            FROM {T['player']}
            {where_sql}
            {order_by}
            LIMIT {int(limit)} OFFSET {int(offset)}
        """

        df = run_query(sql)
        df.attrs["total_count"] = int(total_count)
        df.attrs["offset"] = offset
        df.attrs["limit"] = limit
        return df

    except Exception as e:
        st.error(f"Failed to load weekly player data: {e}")
        return None

@st.cache_data(show_spinner=True, ttl=600)
def load_filtered_weekly_data(
    filters: dict,
    limit: int = 500,
    offset: int = 0,
    sort_column: str = "points",
    sort_direction: str = "DESC"
):
    try:
        where = []

        q = (filters.get("player_query") or "").strip()
        if q:
            esc = q.replace("'", "''")
            where.append("LOWER(player) LIKE LOWER('%" + esc + "%')")

        if filters.get("rostered_only"):
            where.append("manager IS NOT NULL AND manager <> ''")
        if filters.get("started_only"):
            where.append("started = 1")

        for col in ["manager", "nfl_position", "fantasy_position", "nfl_team", "opponent_nfl_team"]:
            vals = filters.get(col) or []
            if vals:
                where.append(f"{col} IN ({sql_in_list(vals)})")

        if filters.get("opp_manager"):
            vals = filters.get("opp_manager")
            if vals:
                where.append("opponent IN (" + sql_in_list(vals) + ")")

        for col in ["week", "year"]:
            vals = filters.get(col) or []
            if vals:
                nums = ", ".join([str(int(v)) for v in vals if str(v).strip()])
                if nums:
                    where.append(f"{col} IN ({nums})")

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        numeric_cols = {"points","pass_yds","pass_td","rush_yds","rush_td","rec_yds","rec_td","rec"}
        if sort_column in numeric_cols:
            order_by = f"ORDER BY {sort_column} {sort_direction} NULLS LAST"
        else:
            order_by = f"ORDER BY {sort_column} {sort_direction}"

        if sort_column != "points":
            order_by += ", points DESC NULLS LAST"
        order_by += ", year DESC, week DESC, player"

        count_sql = f"SELECT COUNT(*) AS total FROM {T['player']} {where_sql};"
        data_sql = f"""
            SELECT *
            FROM {T['player']}
            {where_sql}
            {order_by}
            LIMIT {int(limit)} OFFSET {int(offset)};
        """

        total = run_query(count_sql).iloc[0]["total"]
        df = run_query(data_sql)
        df.attrs["total_count"] = int(total)
        df.attrs["offset"] = offset
        df.attrs["limit"] = limit
        return df

    except Exception as e:
        st.error(f"Failed to load filtered weekly data: {e}")
        return None

# ================================================================
# SEASON OVERVIEW (group by bridged player_key + year)
# ================================================================
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
    sort_column: str = "points",
    sort_direction: str = "DESC",
) -> pd.DataFrame:
    """
    2-layer bridge: rows merge if either yahoo_player_id OR NFL_player_id matches.
    player_key = COALESCE(yid, y_from_n, nid, n_from_y, LOWER(TRIM(player)))

    Changes:
    - fantasy_games counts only rows where yid_norm IS NOT NULL
    - Carry yid_norm through CTE for COUNT(yid_norm)
    - rostered_only: filters to only weeks where manager IS NOT NULL
    - started_only: filters to only weeks where started = 1
    """
    try:
        sort_column    = (sort_column or "points")
        sort_direction = (sort_direction or "DESC").upper()
        if sort_direction not in ("ASC", "DESC"):
            sort_direction = "DESC"

        team_norm      = sql_upper("nfl_team")
        opp_team_norm  = sql_upper("opponent_nfl_team")
        manager_norm   = sql_manager_norm("manager")

        where: List[str] = []
        if position and position != "All":
            where.append("nfl_position = " + sql_quote(position))
        if player_query:
            esc = player_query.replace("'", "''")
            where.append("LOWER(player) LIKE LOWER('%%" + esc + "%%')")
        if manager_query:
            esc = manager_query.strip().lower().replace("'", "''")
            where.append("LOWER(TRIM(manager)) LIKE '%%" + esc + "%%'")
        if manager:
            # Use simple TRIM(manager) for WHERE clause, not the complex manager_norm
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

        # Add rostered_only and started_only filters at SQL level
        if rostered_only:
            where.append("manager IS NOT NULL AND manager <> ''")
        if started_only:
            where.append("started = 1")

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        q = f"""
            WITH base AS (
              SELECT
                year,                                  -- include in Season; omit in Career
                TRIM(player)                                        AS player,
                nfl_position,
                -- normalize IDs (lower + trim + cast to text) and blank->NULL
                NULLIF(TRIM(LOWER(CAST(yahoo_player_id AS VARCHAR))), '') AS yid_norm,
                NULLIF(TRIM(LOWER(CAST(NFL_player_id   AS VARCHAR))), '') AS nid_norm,

                -- normalized teams/managers
                UPPER(NULLIF(TRIM(nfl_team), ''))            AS nfl_team_norm,
                UPPER(NULLIF(TRIM(opponent_nfl_team), ''))   AS opp_team_norm,
                CASE
                  WHEN manager IS NULL OR TRIM(manager) = '' OR LOWER(TRIM(manager)) = 'no manager'
                    THEN NULL
                  ELSE TRIM(manager)
                END AS manager_norm,

                opponent,

                -- metrics
                points,
                season_ppg,
                CASE WHEN started = 1 THEN 1 ELSE 0 END AS started_i,
                COALESCE(win, 0)                           AS win_i,
                COALESCE(CASE WHEN win = 0 THEN 1 ELSE 0 END, 0) AS loss_i,
                COALESCE(team_points, 0)                   AS team_pts_i,
                COALESCE(opponent_points, 0)      AS opp_pts_i,
                COALESCE(is_playoffs, 0)          AS is_playoffs_i
              FROM kmffl.player
              {where_sql}
            ),
            bridged AS (
              SELECT
                *,
                -- Only bridge y_from_n when nid_norm is present
                CASE
                  WHEN nid_norm IS NOT NULL
                    THEN MIN(yid_norm) OVER (PARTITION BY nid_norm)
                  ELSE NULL
                END AS y_from_n,
                -- Only bridge n_from_y when yid_norm is present
                CASE
                  WHEN yid_norm IS NOT NULL
                    THEN MIN(nid_norm) OVER (PARTITION BY yid_norm)
                  ELSE NULL
                END AS n_from_y
              FROM base
            ),
            rows AS (
              SELECT
                year,                                 -- include in Season; omit in Career
                player, nfl_position, nfl_team_norm, opp_team_norm, manager_norm,
                opponent,
                points, season_ppg, started_i, win_i, team_pts_i, opp_pts_i, is_playoffs_i,

                -- carry yid_norm for fantasy_games counting
                yid_norm,

                -- final robust key: match if EITHER ID matches; fall back to name
                COALESCE(
                  yid_norm,
                  y_from_n,
                  nid_norm,
                  n_from_y,
                  LOWER(TRIM(player))
                ) AS player_key
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
                COUNT(yid_norm)          AS fantasy_games,   -- only rows with Yahoo ID
                SUM(win_i)               AS win,
                SUM(team_pts_i)          AS team_points,
                SUM(opp_pts_i)           AS opponent_points,
                MAX(is_playoffs_i)       AS is_playoffs
            FROM rows
            GROUP BY year, player_key
            ORDER BY year DESC, {sort_column} {sort_direction} NULLS LAST, player;
        """
        df = run_query(q)
        if df is None or df.empty:
            return pd.DataFrame()
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
# ================================================================
# CAREER OVERVIEW (group across years by bridged player_key)
# ================================================================
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
    sort_column: str = "points",
    sort_direction: str = "DESC",
    **_,
) -> pd.DataFrame:
    """
    Career aggregates using the same 2-layer bridge as Season, but grouped across years.

    Changes:
    - fantasy_games counts only rows where yid_norm IS NOT NULL
    - ppg_all_time is sourced (no recompute): MAX(ppg_all_time)
    - rostered_only: filters to only weeks where manager IS NOT NULL
    - started_only: filters to only weeks where started = 1
    """
    try:
        sort_column    = (sort_column or "points")
        sort_direction = (sort_direction or "DESC").upper()
        if sort_direction not in ("ASC", "DESC"):
            sort_direction = "DESC"

        team_norm     = sql_upper("nfl_team")
        opp_team_norm = sql_upper("opponent_nfl_team")
        m_norm        = sql_manager_norm("manager")

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
            where.append(m_norm + " IN (" + sql_in_list(manager) + ")")
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

        # Add rostered_only and started_only filters at SQL level
        if rostered_only:
            where.append("manager IS NOT NULL AND manager <> ''")
        if started_only:
            where.append("started = 1")

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        q = f"""
            WITH base AS (
              SELECT
                -- keep year only for optional filtering; career will drop it in final group
                year,
                TRIM(player) AS player,
                nfl_position,
                NULLIF(TRIM(LOWER(CAST(yahoo_player_id AS VARCHAR))), '') AS yid_norm,
                NULLIF(TRIM(LOWER(CAST(NFL_player_id   AS VARCHAR))), '') AS nid_norm,

                UPPER(NULLIF(TRIM(nfl_team), ''))          AS nfl_team_norm,
                UPPER(NULLIF(TRIM(opponent_nfl_team), '')) AS opp_team_norm,
                {m_norm} AS manager_norm,
                opponent,

                -- metrics used downstream (all per-row source fields)
                COALESCE(points, 0)                        AS points_i,
                CASE WHEN started = 1 THEN 1 ELSE 0 END   AS started_i,
                COALESCE(win, 0)                           AS win_i,
                COALESCE(CASE WHEN win = 0 THEN 1 ELSE 0 END, 0) AS loss_i,
                COALESCE(team_points, 0)                   AS team_pts_i,
                COALESCE(opponent_points, 0)               AS opp_pts_i,
                COALESCE(is_playoffs, 0)                   AS is_playoffs_i,
                ppg_all_time                                AS ppg_all_time_i
              FROM {T['player']}
              {where_sql}
            ),
            bridged AS (
              SELECT
                *,
                CASE WHEN nid_norm IS NOT NULL
                       THEN MIN(yid_norm) OVER (PARTITION BY nid_norm)
                     ELSE NULL END AS y_from_n,
                CASE WHEN yid_norm IS NOT NULL
                       THEN MIN(nid_norm) OVER (PARTITION BY yid_norm)
                     ELSE NULL END AS n_from_y
              FROM base
            ),
            rows AS (
              SELECT
                player, nfl_position, nfl_team_norm, opp_team_norm, manager_norm, opponent,
                points_i, started_i, win_i, loss_i, team_pts_i, opp_pts_i, is_playoffs_i,
                ppg_all_time_i,

                -- carry yid_norm for fantasy_games counting
                yid_norm,

                COALESCE(yid_norm, y_from_n, nid_norm, n_from_y, LOWER(TRIM(player))) AS player_key
              FROM bridged
            )
            SELECT
                player_key,
                MAX(player)                     AS player,
                MAX(nfl_position)               AS nfl_position,
                ANY_VALUE(nfl_team_norm)        AS nfl_team,
                STRING_AGG(DISTINCT manager_norm, ', ')
                    FILTER (WHERE manager_norm IS NOT NULL) AS manager,
                STRING_AGG(DISTINCT opponent, ', ' ORDER BY opponent) AS opponent,
                STRING_AGG(DISTINCT opp_team_norm, ', ' ORDER BY opp_team_norm) AS opponent_nfl_team,

                -- career aggregates
                SUM(points_i)                   AS points,
                SUM(started_i)                  AS games_started,
                COUNT(yid_norm)                 AS fantasy_games,  -- only rows with Yahoo ID

                SUM(win_i)                      AS win,
                SUM(loss_i)                     AS loss,
                SUM(team_pts_i)                 AS team_points,
                SUM(opp_pts_i)                  AS opponent_points,
                MAX(is_playoffs_i)              AS is_playoffs,

                -- sourced (do not recompute): choose a stable representative
                MAX(ppg_all_time_i)             AS ppg_all_time
            FROM rows
            GROUP BY player_key
            ORDER BY {sort_column} {sort_direction} NULLS LAST, player;
        """
        df = run_query(q)
        if df is None or df.empty:
            return pd.DataFrame()

        # aliases for UI parity
        if "nfl_position" in df.columns and "position" not in df.columns:
            df["position"] = df["nfl_position"]
        if "points" in df.columns and "total_points" not in df.columns:
            df["total_points"] = df["points"]

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
        draft_summary = run_query(f"""
            SELECT 
                year, 
                COUNT(*) AS total_picks,
                COUNT(DISTINCT manager) AS managers,
                AVG(cost) AS avg_cost,
                MAX(cost) AS max_cost,
                COUNT(DISTINCT player) AS unique_players,
                AVG(points) AS avg_points_scored,
                AVG(season_ppg) AS avg_ppg
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
        recent_sql = f"SELECT * FROM {T['transactions']} ORDER BY year DESC, week DESC"
        if limit is not None:
            recent_sql += f" LIMIT {int(limit)}"
        recent_transactions = run_query(recent_sql)

        transaction_summary = run_query(f"""
            SELECT 
                manager, year,
                COUNT(*) AS total_transactions,
                SUM(CASE WHEN transaction_type = 'Add' THEN 1 ELSE 0 END) AS adds,
                SUM(CASE WHEN transaction_type = 'Drop' THEN 1 ELSE 0 END) AS drops,
                SUM(CASE WHEN transaction_type = 'Trade' THEN 1 ELSE 0 END) AS trades
            FROM {T['transactions']} 
            GROUP BY manager, year 
            ORDER BY year DESC, total_transactions DESC
        """)

        player_data = run_query(f"SELECT * FROM {T['player']} ORDER BY year DESC, week DESC LIMIT 10000")
        injury_data = run_query(f"SELECT * FROM {T['injury']} ORDER BY year DESC, week DESC LIMIT 10000")
        draft_data = run_query(f"SELECT * FROM {T['draft']} ORDER BY year DESC, round, pick")

        return {
            "transactions": recent_transactions,
            "summary": transaction_summary,
            "player_data": player_data,
            "injury_data": injury_data,
            "draft_data": draft_data
        }
    except Exception as e:
        st.error(f"Failed to load transactions data: {e}")
        return {"error": str(e)}

@st.cache_data(show_spinner=True, ttl=600)
def load_simulations_data(include_all_years: bool = True):
    try:
        if include_all_years:
            matchups = run_query(f"SELECT * FROM {T['matchup']} ORDER BY year DESC, week DESC")
            player_averages = run_query(f"""
                SELECT 
                    player, manager, nfl_position AS position, year,
                    AVG(points) AS avg_points, STDDEV(points) AS std_points,
                    COUNT(*) AS games_played
                FROM {T['player']}
                GROUP BY player, manager, nfl_position, year
                HAVING COUNT(*) >= 3
                ORDER BY year DESC, avg_points DESC
            """)
            current_year = run_query(f"SELECT MAX(year) AS year FROM {T['matchup']}").iloc[0]["year"]
        else:
            current_year = run_query(f"SELECT MAX(year) AS year FROM {T['matchup']}").iloc[0]["year"]
            matchups = run_query(f"SELECT * FROM {T['matchup']} WHERE year = {int(current_year)} ORDER BY week DESC")
            player_averages = run_query(f"""
                SELECT 
                    player, manager, nfl_position AS position,
                    AVG(points) AS avg_points, STDDEV(points) AS std_points,
                    COUNT(*) AS games_played
                FROM {T['player']}
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
        if year is not None and week is not None:
            # Specific week view
            week_filter = f"AND p.year = {int(year)} AND p.week = {int(week)}"
        elif all_years:
            # Max week for all years (end of season - who actually owns/keeps the player)
            week_filter = "AND p.max_week = 1"
        else:
            # Max week for latest year only
            week_filter = f"""
                AND p.max_week = 1 
                AND p.year = (SELECT MAX(year) FROM {T['player']})
            """

        keeper_raw_data = run_query(f"""
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
                AND p_next.max_week = 1
            WHERE p.manager IS NOT NULL 
              AND LOWER(TRIM(p.manager)) != 'no manager'
              AND (p.nfl_position NOT IN ('DEF', 'K') OR p.nfl_position IS NULL)
              {week_filter}
            ORDER BY p.year DESC, p.week DESC, p.manager, p.points DESC
        """)

        return keeper_raw_data
    except Exception as e:
        st.error(f"Failed to load keepers data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_data(show_spinner=True, ttl=600)
def load_team_names_data():
    try:
        team_names = run_query(f"""
            SELECT DISTINCT manager, year, COUNT(*) AS games
            FROM {T['matchup']} 
            GROUP BY manager, year
            ORDER BY year DESC, manager
        """)
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
