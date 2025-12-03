#!/usr/bin/env python3
"""
Transaction summary statistics loader.

Optimization: Single combined query instead of multiple separate queries.
Includes pre-computed engagement metric aggregations for fast UI rendering.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.core import run_query, T


@st.cache_data(show_spinner=True, ttl=120)
def load_transaction_summary() -> Dict[str, Any]:
    """
    Load transaction summary statistics by year with engagement metrics.

    Combines aggregations into a single query for efficiency.
    Includes grade distributions, result categories, and FAAB tier breakdowns.

    Returns:
        Dict with "summary" DataFrame containing:
            - year: Year
            - total_transactions: Total number of transactions
            - adds, drops, trades: Count by type
            - total_faab_spent: Total FAAB spent that year
            - avg_net_spar: Average NET SPAR
            - avg_transaction_score, total_transaction_score: Weighted score metrics
            - grade_a_count, grade_b_count, etc.: Grade distribution
            - elite_pickups, great_pickups, etc.: Result category counts
    """
    try:
        query = f"""
            SELECT
                year,
                COUNT(*) AS total_transactions,
                SUM(CASE WHEN LOWER(transaction_type) = 'add' THEN 1 ELSE 0 END) AS adds,
                SUM(CASE WHEN LOWER(transaction_type) = 'drop' THEN 1 ELSE 0 END) AS drops,
                SUM(CASE WHEN LOWER(transaction_type) = 'trade' THEN 1 ELSE 0 END) AS trades,

                -- FAAB metrics
                COALESCE(SUM(faab_bid), 0) AS total_faab_spent,
                COALESCE(AVG(CASE WHEN faab_bid > 0 THEN faab_bid END), 0) AS avg_faab_per_pickup,

                -- SPAR metrics
                COALESCE(AVG(net_manager_spar_ros), 0) AS avg_net_spar,
                COALESCE(SUM(net_manager_spar_ros), 0) AS total_net_spar,

                -- Transaction Score (weighted metric)
                COALESCE(AVG(transaction_score), 0) AS avg_transaction_score,
                COALESCE(SUM(transaction_score), 0) AS total_transaction_score,

                -- Grade distribution
                SUM(CASE WHEN transaction_grade = 'A' THEN 1 ELSE 0 END) AS grade_a_count,
                SUM(CASE WHEN transaction_grade = 'B' THEN 1 ELSE 0 END) AS grade_b_count,
                SUM(CASE WHEN transaction_grade = 'C' THEN 1 ELSE 0 END) AS grade_c_count,
                SUM(CASE WHEN transaction_grade = 'D' THEN 1 ELSE 0 END) AS grade_d_count,
                SUM(CASE WHEN transaction_grade = 'F' THEN 1 ELSE 0 END) AS grade_f_count,

                -- Result categories (pickups)
                SUM(CASE WHEN transaction_result = 'Elite Pickup' THEN 1 ELSE 0 END) AS elite_pickups,
                SUM(CASE WHEN transaction_result = 'Great Pickup' THEN 1 ELSE 0 END) AS great_pickups,
                SUM(CASE WHEN transaction_result = 'Good Pickup' THEN 1 ELSE 0 END) AS good_pickups,

                -- Drop regret distribution
                SUM(CASE WHEN drop_regret_tier = 'Disaster' THEN 1 ELSE 0 END) AS disaster_drops,
                SUM(CASE WHEN drop_regret_tier = 'Major Regret' THEN 1 ELSE 0 END) AS major_regret_drops,
                SUM(CASE WHEN drop_regret_tier = 'No Regret' THEN 1 ELSE 0 END) AS no_regret_drops,

                -- FAAB value tier distribution (adds only)
                SUM(CASE WHEN faab_value_tier = 'Steal' THEN 1 ELSE 0 END) AS steal_count,
                SUM(CASE WHEN faab_value_tier = 'Great Value' THEN 1 ELSE 0 END) AS great_value_count,
                SUM(CASE WHEN faab_value_tier = 'Overpay' THEN 1 ELSE 0 END) AS overpay_count,

                -- Timing distribution
                SUM(CASE WHEN timing_category = 'Early Season' THEN 1 ELSE 0 END) AS early_season_count,
                SUM(CASE WHEN timing_category = 'Mid Season' THEN 1 ELSE 0 END) AS mid_season_count,
                SUM(CASE WHEN timing_category = 'Late Season' THEN 1 ELSE 0 END) AS late_season_count,
                SUM(CASE WHEN timing_category = 'Playoffs' THEN 1 ELSE 0 END) AS playoffs_count

            FROM {T['transactions']}
            GROUP BY year
            ORDER BY year DESC
        """

        summary_df = run_query(query)
        return {"summary": summary_df}

    except Exception as e:
        st.error(f"Failed to load transaction summary: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=True, ttl=120)
def load_manager_transaction_summary() -> Dict[str, Any]:
    """
    Load transaction summary statistics by manager (career-level).

    Pre-aggregated data for manager leaderboards and report cards.

    Returns:
        Dict with "manager_summary" DataFrame
    """
    try:
        query = f"""
            SELECT
                manager,
                COUNT(*) AS total_transactions,
                SUM(CASE WHEN LOWER(transaction_type) = 'add' THEN 1 ELSE 0 END) AS total_adds,
                SUM(CASE WHEN LOWER(transaction_type) = 'drop' THEN 1 ELSE 0 END) AS total_drops,
                SUM(CASE WHEN LOWER(transaction_type) = 'trade' THEN 1 ELSE 0 END) AS total_trades,

                -- FAAB
                COALESCE(SUM(faab_bid), 0) AS career_faab_spent,
                COALESCE(AVG(CASE WHEN faab_bid > 0 THEN faab_bid END), 0) AS avg_faab_per_pickup,

                -- SPAR
                COALESCE(SUM(net_manager_spar_ros), 0) AS career_net_spar,
                COALESCE(AVG(net_manager_spar_ros), 0) AS avg_net_spar,

                -- Transaction Score (weighted metric)
                COALESCE(SUM(transaction_score), 0) AS career_total_score,
                COALESCE(AVG(transaction_score), 0) AS avg_transaction_score,

                -- Efficiency
                COALESCE(AVG(spar_efficiency), 0) AS avg_spar_efficiency,

                -- Grade distribution
                SUM(CASE WHEN transaction_grade = 'A' THEN 1 ELSE 0 END) AS grade_a_count,
                SUM(CASE WHEN transaction_grade = 'B' THEN 1 ELSE 0 END) AS grade_b_count,
                SUM(CASE WHEN transaction_grade = 'C' THEN 1 ELSE 0 END) AS grade_c_count,
                SUM(CASE WHEN transaction_grade = 'D' THEN 1 ELSE 0 END) AS grade_d_count,
                SUM(CASE WHEN transaction_grade = 'F' THEN 1 ELSE 0 END) AS grade_f_count,

                -- Win rate (positive NET SPAR)
                ROUND(100.0 * SUM(CASE WHEN net_manager_spar_ros > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS win_rate,

                -- Elite performance
                SUM(CASE WHEN transaction_result LIKE '%Elite%' THEN 1 ELSE 0 END) AS elite_moves,
                SUM(CASE WHEN drop_regret_tier = 'Disaster' THEN 1 ELSE 0 END) AS disaster_drops

            FROM {T['transactions']}
            WHERE manager IS NOT NULL
            GROUP BY manager
            ORDER BY career_total_score DESC
        """

        summary_df = run_query(query)
        return {"manager_summary": summary_df}

    except Exception as e:
        st.error(f"Failed to load manager transaction summary: {e}")
        return {"error": str(e)}
