#!/usr/bin/env python3
"""
Combined data loader for Transactions tab.

Main entry point for loading all data needed by the transactions tab.
Optimized to use pre-computed engagement metrics from source data.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from .transaction_data import load_transaction_data
from .summary_data import load_transaction_summary, load_manager_transaction_summary
from md.data_access import run_query, T, get_current_league_db


@st.cache_data(show_spinner=True, ttl=120)
def load_optimized_transactions_data() -> Dict[str, Any]:
    """
    Load all data for transactions tab in one optimized call.

    Optimization Summary:
        - Loads ALL transactions (no row limit on transactions table)
        - Loads essential columns from transactions including dual SPAR metrics
        - Pre-computed engagement metrics eliminate UI calculations:
            * transaction_grade (A-F)
            * transaction_result (human-readable)
            * faab_value_tier (Steal → Overpay)
            * drop_regret_score/tier
            * timing_category, pickup_type, result_emoji
        - Pre-aggregated summaries by year and manager
        - Uses generic table mapping for scalability
        - Includes related player and draft data with limits

    Returns:
        Dict with keys:
            - "transactions": Full transaction DataFrame with engagement metrics
            - "summary": Summary statistics by year (with grade distributions)
            - "manager_summary": Pre-aggregated manager stats for leaderboards
            - "player_data": Player stats (limited to 10,000 recent rows)
            - "draft_data": Draft data (limited to 1,000 recent rows)
            - "error": Error message (if any)
    """
    try:
        # Load transaction data (all rows, optimized columns including engagement metrics)
        transaction_result = load_transaction_data()
        if "error" in transaction_result:
            return {"error": transaction_result["error"]}

        # Load summary statistics (with grade/tier distributions)
        summary_result = load_transaction_summary()
        if "error" in summary_result:
            return {"error": summary_result["error"]}

        # Load manager-level summary (pre-aggregated for leaderboards)
        manager_summary_result = load_manager_transaction_summary()
        if "error" in manager_summary_result:
            return {"error": manager_summary_result["error"]}

        # Load related data needed by transaction viewer (with reasonable limits)
        # Note: Using LIMIT here is acceptable as these are supporting datasets
        db = get_current_league_db()
        player_data = run_query(f"SELECT * FROM {db}.public.players_by_year ORDER BY year DESC, week DESC LIMIT 10000")
        draft_data = run_query(f"SELECT * FROM {T['draft']} ORDER BY year DESC, round, pick LIMIT 1000")

        # Combine results
        return {
            "transactions": transaction_result["transactions"],
            "summary": summary_result["summary"],
            "manager_summary": manager_summary_result["manager_summary"],
            "player_data": player_data,
            "draft_data": draft_data,
        }

    except Exception as e:
        st.error(f"Failed to load transactions data: {e}")
        return {"error": str(e)}
