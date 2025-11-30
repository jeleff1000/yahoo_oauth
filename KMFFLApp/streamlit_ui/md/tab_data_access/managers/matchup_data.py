#!/usr/bin/env python3
"""
Matchup data loader for Managers tab.

Optimization: Loads only the 22-25 columns needed by managers components
instead of all 276 columns from the matchup table.

This reduces data transfer by ~91% and memory usage significantly.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.data_access import run_query, T

# Columns needed for managers/matchups tab (~86 out of 275)
# Comprehensively analyzed ALL sub-sub tabs in: Weekly, Seasons, Career, Visualize
# This list covers ALL columns referenced across every component
# Updated: Added weekly_rank, playoff_round, lineup_efficiency, bench_points, win_probability
MANAGERS_MATCHUP_COLUMNS = [
    # === Time dimensions (3) ===
    "year",
    "week",
    "cumulative_week",

    # === Team identifiers (4) ===
    "manager",
    "team_name",
    "opponent",
    "manager_year_week",

    # === Scoring (Yahoo API column names) (4) ===
    "team_points",
    "opponent_points",
    "team_projected_points",
    "opponent_projected_points",

    # === Results (3) ===
    "win",
    "loss",
    "margin",

    # === Game type flags (2) ===
    "is_playoffs",
    "is_consolation",

    # === Optimal/efficiency metrics (6) ===
    "optimal_points",
    "optimal_win",
    "optimal_loss",
    "lineup_efficiency",  # Lineup efficiency percentage (actual/optimal)
    "bench_points",       # Points left on bench
    "win_probability",    # Pre-game win probability

    # === Spread/projection metrics (10) ===
    "expected_spread",
    "expected_odds",
    "win_vs_spread",
    "above_proj_score",
    "below_proj_score",
    "proj_wins",
    "proj_losses",
    "proj_score_error",
    "abs_proj_score_error",
    "underdog_wins",
    "favorite_losses",

    # === Competition metrics (5) ===
    "teams_beat_this_week",  # Keep for backward compatibility
    "weekly_rank",           # NEW: Weekly league ranking (1-8)
    "opponent_teams_beat_this_week",
    "total_matchup_score",
    "close_margin",

    # === League comparison (4) ===
    "league_weekly_mean",
    "league_weekly_median",
    "above_league_median",
    "below_league_median",

    # === Opponent comparison (3) ===
    "opponent_median",
    "above_opponent_median",
    "below_opponent_median",

    # === Season stats (6) ===
    "manager_season_mean",
    "manager_season_median",
    "personal_season_mean",
    "personal_season_median",
    "season_mean",
    "season_median",

    # === Streaks (2) ===
    "winning_streak",
    "losing_streak",

    # === Performance metrics (3) ===
    "gpa",
    "grade",
    "power_rating",

    # === Playoff details (6) ===
    "playoff_round",      # Playoff round name (Quarterfinal, Semifinal, Championship)
    "consolation_round",  # Consolation round name (5th Place Game, 7th Place Game, etc.)
    "quarterfinal",       # Keep for backward compatibility
    "semifinal",          # Keep for backward compatibility
    "championship",       # Keep for backward compatibility
    "final_playoff_seed",

    # === Season outcomes (2) ===
    "champion",
    "sacko",

    # === Simulation/Prediction Columns (for Team Ratings subtab) (13) ===
    # Playoff probabilities
    "avg_seed",
    "p_playoffs",
    "p_bye",
    "p_semis",
    "p_final",
    "p_champ",

    # Expected values
    "exp_final_wins",
    "exp_final_pf",

    # Schedule shuffle simulations (what if we shuffled all schedules)
    "shuffle_1_seed",
    "shuffle_avg_wins",
    "wins_vs_shuffle_wins",
    "shuffle_avg_playoffs",
    "shuffle_avg_bye",
    "shuffle_avg_seed",
    "seed_vs_shuffle_seed",
]


@st.cache_data(show_spinner=True, ttl=600)
def load_managers_matchup_data() -> Dict[str, Any]:
    """
    Load matchup data for managers tab with ONLY the columns needed.

    COMPREHENSIVE ANALYSIS: Checked ALL sub-sub tabs in Weekly, Seasons, Career, Visualize

    Returns:
        Dict with "recent" key containing DataFrame, or "error" key on failure.

    Columns loaded (~80 out of 275):
        Time (3): year, week, cumulative_week
        Teams (4): manager, team_name, opponent, manager_year_week
        Scoring (4): team_points, opponent_points, team_projected_points, opponent_projected_points
        Results (3): win, loss, margin
        Game Type (2): is_playoffs, is_consolation
        Efficiency (6): optimal_points, optimal_win, optimal_loss, lineup_efficiency,
                       bench_points, win_probability
        Spread Metrics (11): expected_spread, expected_odds, win_vs_spread, above_proj_score,
                            below_proj_score, proj_wins, proj_losses, proj_score_error,
                            abs_proj_score_error, underdog_wins, favorite_losses
        Competition (5): teams_beat_this_week, weekly_rank, opponent_teams_beat_this_week,
                        total_matchup_score, close_margin
        League Comparison (4): league_weekly_mean, league_weekly_median,
                              above_league_median, below_league_median
        Opponent Comparison (3): opponent_median, above_opponent_median, below_opponent_median
        Season Stats (6): manager_season_mean, manager_season_median, personal_season_mean,
                         personal_season_median, weekly_mean, weekly_median
        Streaks (2): winning_streak, losing_streak
        Performance (3): gpa, grade, power_rating
        Playoff Details (6): playoff_round, consolation_round, quarterfinal, semifinal,
                            championship, final_playoff_seed
        Outcomes (2): champion, sacko
        Simulations (13): avg_seed, p_playoffs, p_bye, p_semis, p_final, p_champ,
                         exp_final_wins, exp_final_pf, shuffle_1_seed, shuffle_avg_wins,
                         wins_vs_shuffle_wins, shuffle_avg_playoffs, shuffle_avg_bye,
                         shuffle_avg_seed, seed_vs_shuffle_seed

    Uses Yahoo API column names:
        - team_points, opponent_points
        - team_projected_points, opponent_projected_points
    """
    try:
        # Build column list for SELECT clause
        cols_str = ", ".join(MANAGERS_MATCHUP_COLUMNS)

        # Query: all rows, but only needed columns
        query = f"""
            SELECT {cols_str}
            FROM {T['matchup']}
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)

        # Add column aliases for backward compatibility with existing tabs
        # Some tabs still use the old renamed column names
        if "team_projected_points" in df.columns and "manager_proj_score" not in df.columns:
            df["manager_proj_score"] = df["team_projected_points"]
        if "opponent_projected_points" in df.columns and "opponent_proj_score" not in df.columns:
            df["opponent_proj_score"] = df["opponent_projected_points"]
        if "team_name" in df.columns and "manager_team" not in df.columns:
            df["manager_team"] = df["team_name"]

        # Return with "recent" key for backward compatibility
        return {"recent": df}

    except Exception as e:
        st.error(f"Failed to load managers matchup data: {e}")
        return {"error": str(e)}
