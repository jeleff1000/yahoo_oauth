#!/usr/bin/env python3
"""
Matchup data loader for Homepage tab.

Optimization: Loads only the 36 columns needed by homepage components
instead of all 276 columns from the matchup table.

This reduces data transfer by ~87% and memory usage significantly.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.data_access import run_query, T

# Core columns needed across all homepage components (36 out of 276!)
HOMEPAGE_MATCHUP_COLUMNS = [
    # Time & Identity
    "year",
    "week",
    "manager",
    "team_name",
    "opponent",
    "opponent_team",

    # Scoring
    "team_points",
    "opponent_points",
    "margin",
    "total_matchup_score",

    # Projections
    "team_projected_points",
    "opponent_projected_points",
    "proj_score_error",
    "abs_proj_score_error",
    "above_proj_score",
    "below_proj_score",

    # Expected/Spread Stats
    "expected_spread",
    "expected_odds",
    "win_vs_spread",
    "lose_vs_spread",
    "underdog_wins",
    "favorite_losses",

    # League Context
    "league_weekly_mean",
    "league_weekly_median",

    # Results
    "win",
    "loss",

    # Playoff/Season Info
    "is_playoffs",
    "is_consolation",
    "playoff_round",
    "consolation_round",
    "champion",
    "sacko",
    "final_playoff_seed",
    "cumulative_week",
]


@st.cache_data(show_spinner=True, ttl=600)
def load_homepage_matchup_data() -> Dict[str, Any]:
    """
    Load matchup data for homepage with ONLY the columns needed.

    Returns:
        Dict with "Matchup Data" key containing DataFrame, or "error" key on failure.

    Columns loaded (36):
        - Time & Identity (6): year, week, manager, team_name, opponent, opponent_team
        - Scoring (4): team_points, opponent_points, margin, total_matchup_score
        - Projections (6): team_projected_points, opponent_projected_points, proj_score_error,
                          abs_proj_score_error, above_proj_score, below_proj_score
        - Expected/Spread (6): expected_spread, expected_odds, win_vs_spread, lose_vs_spread,
                              underdog_wins, favorite_losses
        - League Context (2): league_weekly_mean, league_weekly_median
        - Results (2): win, loss
        - Playoff/Season (10): is_playoffs, is_consolation, playoff_round, consolation_round,
                              champion, sacko, final_playoff_seed, cumulative_week
    """
    try:
        # Build column list for SELECT clause
        cols_str = ", ".join(HOMEPAGE_MATCHUP_COLUMNS)

        # Query: all rows, but only needed columns
        query = f"""
            SELECT {cols_str}
            FROM {T['matchup']}
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)

        # Add column aliases for backward compatibility with existing tabs
        if "team_projected_points" in df.columns and "manager_proj_score" not in df.columns:
            df["manager_proj_score"] = df["team_projected_points"]
        if "opponent_projected_points" in df.columns and "opponent_proj_score" not in df.columns:
            df["opponent_proj_score"] = df["opponent_projected_points"]
        if "team_name" in df.columns and "manager_team" not in df.columns:
            df["manager_team"] = df["team_name"]

        return {"Matchup Data": df}

    except Exception as e:
        st.error(f"Failed to load homepage matchup data: {e}")
        return {"error": str(e)}
