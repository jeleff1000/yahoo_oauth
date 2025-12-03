#!/usr/bin/env python3
"""
Simulation matchup data loader.

Note: Simulations use dynamically generated shuffle_* columns for various
simulation scenarios. Currently loads all columns to ensure compatibility.

Column Categories:
==================

Core Columns:
    - manager, opponent, year, week, team_points
    - is_playoffs, is_consolation
    - wins_to_date, losses_to_date, points_for_to_date

Schedule Shuffle Columns:
    - shuffle_avg_wins, wins_vs_shuffle_wins
    - opp_shuffle_avg_wins
    - Dynamic: shuffle_*_win, shuffle_*_seed

Playoff Odds Columns:
    - p_playoffs, p_bye, p_champ, p_semis, p_final
    - avg_seed, exp_final_wins, exp_final_pf
    - x1_seed through x10_seed (seed probabilities)
    - power_rating

Playoff Scenario Columns (NEW):
    - playoff_magic_number, bye_magic_number, first_seed_magic_number
    - elimination_number
    - clinched_playoffs, clinched_bye, clinched_first_seed
    - eliminated_from_playoffs, eliminated_from_bye
    - p_playoffs_change, p_champ_change, p_bye_change
    - is_critical_matchup, is_dramatic_win, is_dramatic_loss, drama_score
    - team_mu, team_sigma, win_probability_vs_avg
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.data_access import run_query, T


@st.cache_data(show_spinner=True, ttl=120)
def load_simulation_matchup_data() -> Dict[str, Any]:
    """
    Load matchup data for simulations.

    Currently uses SELECT * because simulations rely on many dynamically
    generated shuffle_* columns (e.g., shuffle_1_win, shuffle_1_seed, etc.)
    and other simulation-specific columns.

    Returns:
        Dict with "matchups" key or "error" key on failure
    """
    try:
        query = f"""
            SELECT *
            FROM {T['matchup']}
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)
        return {"matchups": df}

    except Exception as e:
        st.error(f"Failed to load simulation matchup data: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=True, ttl=120)
def load_playoff_machine_data() -> Dict[str, Any]:
    """
    Load data specifically for the Playoff Machine.

    Optimized query with only columns needed for scenario calculations:
    - Core identifiers and standings
    - Team power ratings (for simulation)
    - Remaining schedule info

    Returns:
        Dict with "matchups" key or "error" key on failure
    """
    try:
        query = f"""
            SELECT
                manager, opponent, year, week, team_points,
                is_playoffs, is_consolation, is_final_regular_week,
                wins_to_date, losses_to_date, points_scored_to_date,
                team_mu, team_sigma, win_probability_vs_avg,
                playoff_magic_number, bye_magic_number,
                clinched_playoffs, clinched_bye,
                eliminated_from_playoffs, eliminated_from_bye,
                p_playoffs, p_bye, p_champ
            FROM {T['matchup']}
            WHERE is_consolation = 0
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)
        return {"matchups": df}

    except Exception as e:
        st.error(f"Failed to load playoff machine data: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=True, ttl=120)
def load_critical_matchups_data() -> Dict[str, Any]:
    """
    Load data for critical matchups / year-in-review analysis.

    Returns matchups with significant playoff odds swings.

    Returns:
        Dict with "critical_matchups" key or "error" key on failure
    """
    try:
        query = f"""
            SELECT
                manager, opponent, year, week,
                team_points, win, loss,
                p_playoffs, p_playoffs_change, p_playoffs_prev,
                p_champ, p_champ_change,
                is_critical_matchup, is_dramatic_win, is_dramatic_loss,
                drama_score,
                clinched_playoffs, eliminated_from_playoffs
            FROM {T['matchup']}
            WHERE is_playoffs = 0
              AND is_consolation = 0
              AND (is_critical_matchup = 1 OR drama_score > 10)
            ORDER BY drama_score DESC, year DESC, week DESC
        """

        df = run_query(query)
        return {"critical_matchups": df}

    except Exception as e:
        st.error(f"Failed to load critical matchups data: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=True, ttl=120)
def load_playoff_machine_schedule() -> Dict[str, Any]:
    """
    Load schedule data for remaining games in Playoff Machine.

    Fetches future matchups (games with 0 or null points) from schedule table.

    Returns:
        Dict with "schedule" key or "error" key on failure
    """
    try:
        query = f"""
            SELECT
                manager, opponent, year, week,
                team_points, is_playoffs, is_consolation
            FROM {T['schedule']}
            WHERE (team_points = 0 OR team_points IS NULL)
              AND is_playoffs = 0
              AND is_consolation = 0
            ORDER BY year DESC, week ASC
        """

        df = run_query(query)
        return {"schedule": df}

    except Exception as e:
        st.error(f"Failed to load playoff machine schedule: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=True, ttl=120)
def load_clinch_scenarios_data() -> Dict[str, Any]:
    """
    Load data for clinch/elimination scenarios.

    Returns current magic numbers and clinch status for all teams.

    Returns:
        Dict with "clinch_data" key or "error" key on failure
    """
    try:
        query = f"""
            SELECT
                manager, year, week,
                wins_to_date, losses_to_date,
                playoff_magic_number, bye_magic_number, first_seed_magic_number,
                elimination_number,
                clinched_playoffs, clinched_bye, clinched_first_seed,
                eliminated_from_playoffs, eliminated_from_bye,
                p_playoffs, p_bye
            FROM {T['matchup']}
            WHERE is_playoffs = 0
              AND is_consolation = 0
            ORDER BY year DESC, week DESC, playoff_magic_number ASC
        """

        df = run_query(query)
        return {"clinch_data": df}

    except Exception as e:
        st.error(f"Failed to load clinch scenarios data: {e}")
        return {"error": str(e)}
