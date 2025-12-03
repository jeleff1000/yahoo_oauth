#!/usr/bin/env python3
"""
Summary data loaders for Managers tab.

Provides aggregated matchup and head-to-head statistics.
These are already optimized (using aggregations, not SELECT *).
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.core import run_query, T


@st.cache_data(show_spinner=True, ttl=120)
def load_managers_summary_data() -> Dict[str, Any]:
    """
    Load summary statistics for managers tab.

    Returns:
        Dict with "summary" and "h2h" keys or "error" key on failure.

    Data loaded:
        - summary: Year-by-year manager statistics
        - h2h: Head-to-head records between managers
    """
    try:
        # Matchup summary by year and manager (already optimized - uses aggregation)
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

        # Head-to-head summary (already optimized - uses aggregation)
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

        return {
            "summary": matchup_summary,
            "h2h": h2h_summary,
        }

    except Exception as e:
        st.error(f"Failed to load managers summary data: {e}")
        return {"error": str(e)}
