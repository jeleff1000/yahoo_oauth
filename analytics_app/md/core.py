#!/usr/bin/env python3
"""
Core data access primitives for MotherDuck.

This module contains the foundational database utilities:
- Database configuration (get_current_league_db, get_table_dict, T)
- Connection management (get_motherduck_connection, run_query)
- SQL helpers (sql_quote, sql_in_list, sql_upper, sql_upper_in_list, sql_manager_norm)
- Common queries (list_seasons, list_weeks, list_managers, etc.)
- Utilities (detect_roster_structure)

Tab-specific data loaders are in md.tab_data_access/.
"""
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
        "schedule": f"{db}.public.schedule",
        "transactions": f"{db}.public.transactions",
        "league_settings": f"{db}.public.league_settings",
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

@st.cache_data(ttl=120, show_spinner=True)
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
# Optimal Week Helpers
# ---------------------------------------

@st.cache_data(show_spinner=True, ttl=120)
def list_optimal_seasons() -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT year
        FROM {get_current_league_db()}.public.players_by_year
        WHERE COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY year
    """)
    return [] if df.empty else df["year"].astype(int).tolist()

@st.cache_data(show_spinner=True, ttl=120)
def list_optimal_weeks(year: int) -> list[int]:
    df = run_query(f"""
        SELECT DISTINCT week
        FROM {get_current_league_db()}.public.players_by_year
        WHERE year = {int(year)}
          AND COALESCE(league_wide_optimal_player, 0) = 1
        ORDER BY week
    """)
    return [] if df.empty else df["week"].astype(int).tolist()


# ---------------------------------------
# Roster Structure Detection
# ---------------------------------------

# Position patterns for classification
STARTER_POSITIONS = ["QB", "RB", "WR", "TE", "W/R/T", "K", "DEF"]
BENCH_POSITIONS = ["BN", "IR", "IL"]


@st.cache_data(ttl=3600, show_spinner=False)
def detect_roster_structure() -> Optional[Dict]:
    """
    Detect roster configuration from the league_settings table.

    Parses the roster_positions array from the settings_json column
    for the most recent year. This is the canonical source of truth.

    Falls back to inferring from player data if league_settings unavailable.

    Returns:
        Dict with:
        - starter_count: total starter slots
        - bench_count: total bench/IR slots
        - position_counts: dict of position -> count (e.g., {'QB': 1, 'RB': 2, 'WR': 3, ...})
        - total_roster: total roster size
        - flex_positions: list of flex position names (e.g., ['W/R/T'])
    """
    import json

    try:
        # Try league_settings table first (canonical source)
        sql = f"""
            SELECT settings_json
            FROM {T['league_settings']}
            WHERE year = (SELECT MAX(year) FROM {T['league_settings']})
            LIMIT 1
        """
        df = run_query(sql)

        if df is not None and not df.empty and 'settings_json' in df.columns:
            settings_str = df.iloc[0]['settings_json']
            if settings_str:
                settings = json.loads(settings_str) if isinstance(settings_str, str) else settings_str
                roster_positions = settings.get('roster_positions', [])

                if roster_positions:
                    position_counts = {}
                    starter_count = 0
                    bench_count = 0
                    flex_positions = []

                    for slot in roster_positions:
                        pos = slot.get('position', '').upper()
                        count = int(slot.get('count', 0))
                        if pos and count > 0:
                            position_counts[pos] = count

                            is_bench = pos in BENCH_POSITIONS
                            if is_bench:
                                bench_count += count
                            else:
                                starter_count += count
                                if '/' in pos:
                                    flex_positions.append(pos)

                    if position_counts:
                        return {
                            'starter_count': starter_count,
                            'bench_count': bench_count,
                            'position_counts': position_counts,
                            'total_roster': starter_count + bench_count,
                            'flex_positions': flex_positions
                        }

    except Exception:
        pass  # Fall through to player-based detection

    # Fallback: infer from player data
    try:
        sql = f"""
            SELECT fantasy_position, COUNT(*) as slot_count
            FROM {T['player']}
            WHERE year = (SELECT MAX(year) FROM {T['player']})
              AND week = 1
              AND fantasy_position IS NOT NULL
              AND fantasy_position != ''
              AND manager = (
                  SELECT manager FROM {T['player']}
                  WHERE year = (SELECT MAX(year) FROM {T['player']})
                    AND week = 1
                    AND manager IS NOT NULL
                    AND manager != ''
                  LIMIT 1
              )
            GROUP BY fantasy_position
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        position_counts = {}
        starter_count = 0
        bench_count = 0
        flex_positions = []

        for _, row in df.iterrows():
            pos = str(row['fantasy_position']).upper().strip()
            count = int(row['slot_count'])
            position_counts[pos] = count

            is_bench = pos in BENCH_POSITIONS
            if is_bench:
                bench_count += count
            else:
                starter_count += count
                if '/' in pos and pos not in flex_positions:
                    flex_positions.append(pos)

        return {
            'starter_count': starter_count,
            'bench_count': bench_count,
            'position_counts': position_counts,
            'total_roster': starter_count + bench_count,
            'flex_positions': flex_positions
        }

    except Exception:
        return None