#!/usr/bin/env python3
"""
Draft table data loader with column selection.

Optimization: Loads only the columns actually used by draft tab components,
rather than SELECT * which would load all columns.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from md.data_access import run_query, T

# Only the columns actually used by draft tab components
DRAFT_COLUMNS = [
    # Core identifiers
    "player",
    "yahoo_position",
    "position",
    "nfl_team",
    "manager",
    "year",

    # Draft position & cost
    "cost",
    "pick",
    "round",
    "cost_bucket",

    # Keeper status & economics
    "is_keeper_status",
    "is_keeper_cost",
    "kept_next_year",
    "keeper_price",         # Next year's keeper price (from context rules)
    "keeper_year",          # How many consecutive years kept

    # Performance metrics
    "total_fantasy_points",
    "season_ppg",
    "games_played",
    "games_missed",         # Games missed (injury tracking)
    "weeks_rostered",
    "weeks_started",

    # Dual SPAR metrics (player vs manager)
    "player_spar",      # Total SPAR (opportunity cost - what player produced)
    "manager_spar",     # Managed SPAR (actual value - SPAR while rostered)
    "spar",            # Legacy SPAR column

    # Dual PPG metrics
    "player_ppg",       # Player's total PPG
    "manager_ppg",      # PPG while on manager's roster

    # Dual PGVOR metrics
    "player_pgvor",     # Total PGVOR
    "manager_pgvor",    # Managed PGVOR
    "pgvor",           # Legacy PGVOR column

    # Replacement level
    "replacement_ppg",

    # Value metrics
    "draft_roi",
    "spar_per_dollar",
    "spar_per_pick",
    "spar_per_round",
    "points_per_dollar",
    "points_per_pick",

    # Rankings
    "season_overall_rank",
    "season_position_rank",
    "price_rank_within_position",
    "pick_rank_within_position",

    # Efficiency metrics
    "value_over_replacement",
    "draft_position_delta",
    "pick_savings",
    "cost_savings",

    # === NEW: Draft Grades & Value Tiers ===
    "spar_percentile",      # SPAR percentile within position-year (0-100)
    "draft_grade",          # A/B/C/D/F letter grade
    "value_tier",           # Steal/Good Value/Fair/Overpay/Bust

    # === NEW: Draft Flags & Tiers ===
    "round_percentile",     # Percentile of round within year (0-100)
    "draft_tier",           # Early (1-5) / Mid (6-10) / Late (11-16)
    "is_breakout",          # Late-round pick with top finish
    "is_bust",              # Early-round pick with bottom finish
    "is_injured",           # Missed 50%+ of games
    "is_injury_bust",       # Bust due to injury
    "is_performance_bust",  # Bust due to poor performance (not injury)

    # === NEW: Manager Draft Grades ===
    "pick_quality_score",           # How good was this pick vs peers (0-100)
    "pick_weight",                  # Weight for this pick (early picks weighted more)
    "manager_draft_score",          # Manager's weighted avg score for year
    "manager_draft_percentile",     # Grading percentile (all-time)
    "manager_draft_percentile_year",    # Percentile within that year
    "manager_draft_percentile_alltime", # Percentile across ALL years
    "manager_draft_grade",          # A/B/C/D/F manager grade
    "manager_total_spar",           # Sum of SPAR for all picks
    "manager_avg_spar",             # Average SPAR per pick
    "manager_hit_rate",             # % of picks with positive SPAR
    "manager_picks_count",          # Total picks made

    # === NEW: Starter Designation ===
    "position_draft_rank",          # Rank within position for this manager-year (QB1=1, QB2=2)
    "position_draft_label",         # Label like "QB1", "RB2", "WR3"
    "starter_slots_available",      # How many starter slots exist for this position
    "drafted_as_starter",           # 1 if drafted to be a starter, 0 if backup
    "drafted_as_backup",            # 1 if drafted to be a backup, 0 if starter

    # === NEW: Position Tiers ===
    "position_tier",                # Dynamic tier number (varies by position-year)
    "position_tier_label",          # Descriptive label ("Elite RB", "Starter WR", etc.)
    "position_tier_count",          # How many tiers exist for this position-year
    "position_percentile",          # Exact percentile rank (0-100) within position-year

    # === NEW: Bench Insurance Metrics ===
    "bench_insurance_discount",     # Position-specific bench value discount (0.0-1.0)
    "bench_spar",                   # Pre-computed bench SPAR = max(0, manager_spar) * discount
    "position_failure_rate",        # Starter failure rate for this position
    "position_activation_rate",     # Bench activation rate for this position

    # === NEW: Rank-Based Bench Values (data-driven) ===
    "bench_value_by_rank",          # Median SPAR for this position draft rank (QB2, RB3, etc.)
    "slot_median_spar",             # Historical median SPAR for this draft slot
]


def get_available_columns() -> list:
    """
    Query the draft table to get available columns, then intersect with desired columns.
    This ensures we only request columns that actually exist.
    """
    try:
        # Get schema info
        schema_query = f"DESCRIBE {T['draft']}"
        schema_df = run_query(schema_query)

        # Get column names from schema
        if 'column_name' in schema_df.columns:
            available = set(schema_df['column_name'].tolist())
        elif len(schema_df.columns) > 0:
            # First column usually has names
            available = set(schema_df.iloc[:, 0].tolist())
        else:
            # Fallback: use all desired columns
            return DRAFT_COLUMNS

        # Return intersection of desired and available
        return [c for c in DRAFT_COLUMNS if c in available]

    except Exception:
        # If schema query fails, return core columns that should always exist
        core_cols = [
            "player", "yahoo_position", "position", "nfl_team", "manager", "year",
            "cost", "pick", "round", "cost_bucket",
            "is_keeper_status", "is_keeper_cost", "kept_next_year",
            "total_fantasy_points", "season_ppg", "games_played",
            "weeks_rostered", "weeks_started",
            "player_spar", "manager_spar", "spar",
            "player_ppg", "manager_ppg",
            "player_pgvor", "manager_pgvor", "pgvor",
            "replacement_ppg",
            "draft_roi", "spar_per_dollar", "spar_per_pick", "spar_per_round",
            "points_per_dollar", "points_per_pick",
            "season_overall_rank", "season_position_rank",
            "price_rank_within_position", "pick_rank_within_position",
            "value_over_replacement", "draft_position_delta",
            "pick_savings", "cost_savings",
        ]
        return core_cols


@st.cache_data(show_spinner=True, ttl=120)
def load_draft_data() -> Dict[str, Any]:
    """
    Load draft data with optimized column selection.

    Dynamically detects available columns to avoid errors when new columns
    haven't been added yet by the enrichment pipeline.

    Key columns loaded:
        Core identifiers: player, yahoo_position, position, nfl_team, manager, year
        Draft info: cost, pick, round, cost_bucket
        Keeper tracking: is_keeper_status, is_keeper_cost, kept_next_year
        Performance: total_fantasy_points, season_ppg, games_played, weeks_rostered, weeks_started

        Dual SPAR metrics (player vs manager):
            - player_spar: Total SPAR (opportunity cost - what player produced)
            - manager_spar: Managed SPAR (actual value - SPAR while rostered)
            - player_ppg / manager_ppg: PPG metrics
            - player_pgvor / manager_pgvor: PGVOR metrics

        Value metrics: draft_roi, spar_per_dollar, spar_per_pick, points_per_dollar, etc.
        Rankings: season_overall_rank, season_position_rank, price/pick ranks
        Efficiency: value_over_replacement, draft_position_delta, savings metrics

        NEW (if available from enrichment pipeline):
            - Draft grades: draft_grade, spar_percentile, value_tier
            - Draft flags: is_breakout, is_bust, draft_tier, etc.
            - Manager grades: manager_draft_grade, manager_draft_score, etc.
            - Keeper economics: keeper_price, keeper_year

    Returns:
        Dict with "Draft History" key containing DataFrame, or "error" key on failure
    """
    try:
        # Get columns that actually exist in the table
        available_cols = get_available_columns()
        cols_str = ", ".join(available_cols)

        query = f"""
            SELECT {cols_str}
            FROM {T['draft']}
            ORDER BY year DESC, round, pick
        """

        df = run_query(query)

        # Rename total_fantasy_points to points for compatibility with existing code
        if 'total_fantasy_points' in df.columns:
            df = df.rename(columns={'total_fantasy_points': 'points'})

        return {"Draft History": df}

    except Exception as e:
        st.error(f"Failed to load draft data: {e}")
        return {"error": str(e)}
