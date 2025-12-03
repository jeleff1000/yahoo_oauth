#!/usr/bin/env python3
"""
Summary statistics loader for Homepage overview.

Handles missing tables gracefully - not all leagues have all tables.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.core import run_query, T, get_current_league_db


def _safe_count(table_name: str) -> int:
    """Safely count rows in a table, returning 0 if table doesn't exist."""
    try:
        result = run_query(f"SELECT COUNT(*) AS cnt FROM {table_name}")
        return int(result.iloc[0]["cnt"]) if not result.empty else 0
    except Exception:
        return 0


@st.cache_data(show_spinner=True, ttl=120)
def load_homepage_summary_stats() -> Dict[str, Any]:
    """
    Load summary statistics for homepage overview.

    Returns:
        Dict with summary statistics or "error" key on failure.

    Stats loaded:
        - matchup_count: Total matchups in database
        - player_count: Total player records
        - draft_count: Total draft picks
        - transactions_count: Total transactions
        - latest_year: Most recent season
        - latest_week: Most recent week
        - latest_games: Number of games in latest week
    """
    try:
        db = get_current_league_db()

        # Core stats - these tables should always exist
        matchup_count = _safe_count(T["matchup"])
        player_count = _safe_count(f"{db}.public.players_by_year")
        draft_count = _safe_count(T["draft"])
        transactions_count = _safe_count(T["transactions"])

        # Get latest week info
        try:
            latest_query = f"""
                SELECT year, week, COUNT(*) AS games
                FROM {T['matchup']}
                GROUP BY year, week
                ORDER BY year DESC, week DESC
                LIMIT 1
            """
            latest_result = run_query(latest_query)
            if not latest_result.empty:
                latest_year = int(latest_result.iloc[0]["year"])
                latest_week = int(latest_result.iloc[0]["week"])
                latest_games = int(latest_result.iloc[0]["games"])
            else:
                latest_year = 0
                latest_week = 0
                latest_games = 0
        except Exception:
            latest_year = 0
            latest_week = 0
            latest_games = 0

        return {
            "matchup_count": matchup_count,
            "player_count": player_count,
            "draft_count": draft_count,
            "transactions_count": transactions_count,
            "latest_year": latest_year,
            "latest_week": latest_week,
            "latest_games": latest_games,
        }

    except Exception as e:
        st.error(f"Failed to load homepage summary stats: {e}")
        return {"error": str(e)}
