#!/usr/bin/env python3
"""
Recap matchup data loader for Homepage tab - Recaps section.

Optimization: Loads additional columns needed for recaps beyond the base homepage matchup data.
Recaps need many contextual columns for narrative generation.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.core import run_query, T

# Columns needed for recaps (in addition to base homepage columns, this is a superset)
RECAPS_MATCHUP_COLUMNS = [
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
    "teams_beat_this_week",
    "above_league_median",
    # Results
    "win",
    "loss",
    "close_margin",
    # Playoff/Season Info
    "is_playoffs",
    "is_consolation",
    "playoff_round",
    "consolation_round",
    "champion",
    "sacko",
    "final_playoff_seed",
    "cumulative_week",
    # Cumulative/Season stats for recaps
    "wins_to_date",
    "losses_to_date",
    "optimal_points",
    "grade",
    "gpa",
    "winning_streak",
    "losing_streak",
    "playoff_seed_to_date",
    "p_playoffs",
    "p_bye",
    "proj_wins",
    # Additional useful columns
    "cumulative_wins",
    "cumulative_losses",
    "final_wins",
    "final_losses",
    "win_streak",
    "loss_streak",
    # Playoff odds and projections
    "avg_seed",
    "p_champ",
    # Shuffled schedule metrics
    "shuffle_avg_wins",
    "shuffle_avg_playoffs",
    "wins_vs_shuffle_wins",
    # Manager rankings and percentiles (for exceptional performance detection)
    "manager_all_time_ranking",
    "manager_all_time_percentile",
    "manager_season_ranking",
]


@st.cache_data(show_spinner=True, ttl=120)
def load_recaps_matchup_data() -> Dict[str, Any]:
    """
    Load matchup data for recaps with ONLY the columns needed.

    Returns:
        Dict with "Matchup Data" key containing DataFrame, or "error" key on failure.

    Columns loaded (74):
        - Time & Identity (6): year, week, manager, team_name, opponent, opponent_team
        - Scoring (4): team_points, opponent_points, margin, total_matchup_score
        - Projections (6): team_projected_points, opponent_projected_points, proj_score_error,
                          abs_proj_score_error, above_proj_score, below_proj_score
        - Expected/Spread (6): expected_spread, expected_odds, win_vs_spread, lose_vs_spread,
                              underdog_wins, favorite_losses
        - League Context (4): league_weekly_mean, league_weekly_median,
                             teams_beat_this_week, above_league_median
        - Results (3): win, loss, close_margin
        - Playoff/Season (8): is_playoffs, is_consolation, playoff_round, consolation_round,
                              champion, sacko, final_playoff_seed, cumulative_week
        - Cumulative Stats (17): wins_to_date, losses_to_date, optimal_points, grade, gpa,
                                winning_streak, losing_streak, playoff_seed_to_date, p_playoffs, p_bye,
                                proj_wins, cumulative_wins, cumulative_losses, final_wins, final_losses,
                                win_streak, loss_streak
        - Playoff Odds (2): avg_seed, p_champ
        - Shuffled Schedule Metrics (3): shuffle_avg_wins, shuffle_avg_playoffs, wins_vs_shuffle_wins
        - Rankings/Percentiles (3): manager_all_time_ranking, manager_all_time_percentile,
                                    manager_season_ranking
    """
    try:
        # Build column list for SELECT clause
        cols_str = ", ".join(RECAPS_MATCHUP_COLUMNS)

        # Query: all rows, but only needed columns
        query = f"""
            SELECT {cols_str}
            FROM {T['matchup']}
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)

        # Add column aliases for backward compatibility with existing tabs
        if (
            "team_projected_points" in df.columns
            and "manager_proj_score" not in df.columns
        ):
            df["manager_proj_score"] = df["team_projected_points"]
        if (
            "opponent_projected_points" in df.columns
            and "opponent_proj_score" not in df.columns
        ):
            df["opponent_proj_score"] = df["opponent_projected_points"]
        if "team_name" in df.columns and "manager_team" not in df.columns:
            df["manager_team"] = df["team_name"]

        return {"Matchup Data": df}

    except Exception as e:
        st.error(f"Failed to load recaps matchup data: {e}")
        return {"error": str(e)}
