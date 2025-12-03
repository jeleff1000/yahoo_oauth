#!/usr/bin/env python3
"""
Transaction data loader with optimized column selection.

Optimization: Loads only required columns from transactions table (no row limits).
- All rows included (no LIMIT clause)
- Only essential columns selected (17 of 41 columns = ~59% reduction)
- Uses generic table mapping for scalability
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
import pandas as pd
from md.core import run_query, T


# Essential columns for transaction display and analysis
# Based on usage analysis from all transaction display files
TRANSACTION_COLUMNS = [
    # Core identifiers
    "transaction_id",
    "manager",
    "player_name",
    "yahoo_player_id",  # For joining with player table

    # Time dimensions
    "year",
    "week",

    # Transaction details
    "transaction_type",
    "faab_bid",

    # Player info
    "position",

    # Performance metrics (before transaction)
    "ppg_before_transaction",
    "weeks_before",
    "position_rank_before_transaction",

    # Performance metrics (at transaction)
    "points_at_transaction",
    "position_rank_at_transaction",

    # Performance metrics (after transaction)
    "ppg_after_transaction",
    "weeks_after",
    "total_points_rest_of_season",  # Legacy column (kept for compatibility)
    "ppg_rest_of_season",           # Legacy column
    "weeks_rest_of_season",         # Legacy column
    "position_rank_after_transaction",

    # Analysis metrics
    "points_per_faab_dollar",
    "transaction_quality_score",

    # NEW SPAR METRICS (dual managed/total)
    # Individual player ROS metrics - Managed (on YOUR roster)
    "manager_spar_ros_managed",       # ROS SPAR while on YOUR roster
    "total_points_ros_managed",       # Fantasy points while on your roster
    "ppg_ros_managed",                # PPG while rostered
    "weeks_ros_managed",              # Number of weeks on your roster
    "replacement_ppg_ros_managed",    # Replacement baseline for managed window
    # "manager_spar_per_game_managed",  # Per-game SPAR managed - TODO: Add to pipeline

    # Individual player ROS metrics - Total (regardless of roster)
    "player_spar_ros_total",          # Total ROS SPAR (opportunity cost)
    "total_points_ros_total",         # Total fantasy points ROS
    "ppg_ros_total",                  # PPG for all ROS
    "weeks_ros_total",                # Total ROS weeks
    "replacement_ppg_ros_total",      # Replacement baseline for total window
    # "player_spar_per_game_total",     # Per-game SPAR total - TODO: Add to pipeline

    # NET transaction metrics (grouped by transaction_id + manager)
    "net_manager_spar_ros",           # Net SPAR actually captured (adds - drops)
    "net_player_spar_ros",            # Net opportunity cost (total)
    "spar_efficiency",                # SPAR per FAAB dollar
    "net_spar_rank",                  # Rank by net managed SPAR

    # Legacy SPAR columns (backward compatibility - may be deprecated)
    "replacement_ppg_ros",
    "fa_spar_ros",
    "fa_ppg_ros",
    "fa_pgvor_ros",
    "waiver_cost_norm",
    "fa_roi",
    "spar_per_faab",
    "net_spar_ros",
    "position_spar_percentile",
    "value_vs_avg_pickup",
    "spar_per_faab_rank",

    # NEW: Engagement metrics (pre-computed for UI)
    "transaction_grade",        # A-F grade based on NET SPAR percentile
    "transaction_result",       # Human-readable result ("Elite Pickup", "Big Regret", etc.)
    "faab_value_tier",          # FAAB efficiency tier ("Steal", "Great Value", "Fair", "Overpay")
    "drop_regret_score",        # For drops, SPAR player produced after being dropped
    "drop_regret_tier",         # Category for drop regret ("No Regret" → "Disaster")
    "timing_category",          # Season timing ("Early Season", "Mid Season", etc.)
    "pickup_type",              # Source type ("Waiver Claim", "Free Agent", "Trade")
    "result_emoji",             # Quick visual indicator emoji
    "net_spar_percentile",      # Percentile for grade calculation

    # NEW: Weighted Transaction Score (timing, playoffs, FAAB efficiency, regret)
    "transaction_score",        # Final weighted score for ranking transactions
    "score_row_spar",           # Base SPAR used (managed for adds, regret for drops)
    "score_drop_regret",        # Drop regret component
    "score_faab_bonus",         # FAAB efficiency bonus component
    "score_hold_bonus",         # Hold duration bonus component
    "score_timing_mult",        # Timing multiplier applied
    "score_playoff_mult",       # Playoff multiplier applied

    # Player info (enriched from player table in pipeline)
    "headshot_url",             # Player headshot image URL
]


@st.cache_data(show_spinner=True, ttl=120)
def load_transaction_data() -> Dict[str, Any]:
    """
    Load transaction data with optimized column selection.

    Loads ALL transactions (no row limit) with essential columns including new dual SPAR metrics.

    Columns loaded:
        - Core: transaction_id, manager, player_name
        - Time: year, week
        - Details: transaction_type, faab_bid, position
        - Before: ppg_before_transaction, weeks_before, position_rank_before_transaction
        - At: points_at_transaction, position_rank_at_transaction
        - After: ppg_after_transaction, weeks_after, total_points_rest_of_season (legacy),
                 ppg_rest_of_season (legacy), weeks_rest_of_season (legacy), position_rank_after_transaction
        - Analysis: points_per_faab_dollar, transaction_quality_score

        - NEW SPAR Metrics (Dual Managed/Total):
            * Managed (on YOUR roster): manager_spar_ros_managed, total_points_ros_managed,
                                       ppg_ros_managed, weeks_ros_managed, replacement_ppg_ros_managed,
                                       manager_spar_per_game_managed
            * Total (opportunity cost): player_spar_ros_total, total_points_ros_total,
                                       ppg_ros_total, weeks_ros_total, replacement_ppg_ros_total,
                                       player_spar_per_game_total
            * NET metrics: net_manager_spar_ros, net_player_spar_ros, spar_efficiency, net_spar_rank

        - Legacy SPAR (for backward compatibility): fa_spar_ros, fa_ppg_ros, fa_pgvor_ros,
                                                    replacement_ppg_ros, waiver_cost_norm, fa_roi,
                                                    spar_per_faab, net_spar_ros, position_spar_percentile,
                                                    value_vs_avg_pickup, spar_per_faab_rank

        - NEW Engagement Metrics (pre-computed for UI):
            * transaction_grade: A-F grade based on NET SPAR percentile
            * transaction_result: Human-readable result ("Elite Pickup", "Big Regret", etc.)
            * faab_value_tier: FAAB efficiency tier ("Steal", "Great Value", "Fair", "Overpay")
            * drop_regret_score: For drops, SPAR player produced after being dropped
            * drop_regret_tier: Category ("No Regret" → "Disaster")
            * timing_category: Season timing ("Early Season", "Mid Season", etc.)
            * pickup_type: Source type ("Waiver Claim", "Free Agent", "Trade")
            * result_emoji: Quick visual indicator emoji

        - NEW Weighted Transaction Score:
            * transaction_score: Final weighted score combining timing, playoffs, FAAB efficiency, regret
            * score_row_spar: Base SPAR (managed for adds, regret for drops)
            * score_drop_regret: Drop regret component
            * score_faab_bonus: FAAB efficiency bonus
            * score_hold_bonus: Hold duration bonus
            * score_timing_mult: Timing multiplier (early season = higher)
            * score_playoff_mult: Playoff multiplier (last 3 weeks = 1.25x)

    Returns:
        Dict with "transactions" DataFrame or "error" key on failure
    """
    try:
        cols_str = ", ".join(TRANSACTION_COLUMNS)

        query = f"""
            SELECT {cols_str}
            FROM {T['transactions']}
            ORDER BY year DESC, week DESC
        """

        df = run_query(query)

        return {"transactions": df}

    except Exception as e:
        st.error(f"Failed to load transaction data: {e}")
        return {"error": str(e)}
