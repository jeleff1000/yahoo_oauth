#!/usr/bin/env python3
"""
Summary statistics loader for Homepage overview.

Optimization: Combines 5 separate COUNT queries into 1 combined query,
reducing database round-trips and improving performance by ~5x.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.data_access import run_query, T, get_current_league_db


@st.cache_data(show_spinner=True, ttl=600)
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
        - injuries_count: Total injury records
        - latest_year: Most recent season
        - latest_week: Most recent week
        - latest_games: Number of games in latest week
    """
    try:
        db = get_current_league_db()
        # Combined query using CTEs - single database round-trip!
        summary_query = f"""
            WITH matchup_stats AS (
                SELECT COUNT(*) AS matchup_count
                FROM {T['matchup']}
            ),
            player_stats AS (
                SELECT COUNT(*) AS player_count
                FROM {db}.public.players_by_year
            ),
            draft_stats AS (
                SELECT COUNT(*) AS draft_count
                FROM {T['draft']}
            ),
            transaction_stats AS (
                SELECT COUNT(*) AS transactions_count
                FROM {T['transactions']}
            ),
            injury_stats AS (
                SELECT COUNT(*) AS injuries_count
                FROM {T['injury']}
            ),
            latest_week_info AS (
                SELECT year, week
                FROM {T['matchup']}
                ORDER BY year DESC, week DESC
                LIMIT 1
            ),
            latest_data AS (
                SELECT
                    m.year,
                    m.week,
                    COUNT(*) AS games
                FROM {T['matchup']} m
                INNER JOIN latest_week_info l
                    ON m.year = l.year AND m.week = l.week
                GROUP BY m.year, m.week
            )
            SELECT
                m.matchup_count,
                p.player_count,
                d.draft_count,
                t.transactions_count,
                i.injuries_count,
                l.year AS latest_year,
                l.week AS latest_week,
                l.games AS latest_games
            FROM matchup_stats m
            CROSS JOIN player_stats p
            CROSS JOIN draft_stats d
            CROSS JOIN transaction_stats t
            CROSS JOIN injury_stats i
            CROSS JOIN latest_data l
        """

        result = run_query(summary_query)

        if result.empty:
            return {"error": "No data"}

        # Extract values from result row
        row = result.iloc[0]
        return {
            "matchup_count": int(row["matchup_count"]),
            "player_count": int(row["player_count"]),
            "draft_count": int(row["draft_count"]),
            "transactions_count": int(row["transactions_count"]),
            "injuries_count": int(row["injuries_count"]),
            "latest_year": int(row["latest_year"]),
            "latest_week": int(row["latest_week"]),
            "latest_games": int(row["latest_games"]),
        }

    except Exception as e:
        st.error(f"Failed to load homepage summary stats: {e}")
        return {"error": str(e)}
