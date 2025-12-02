#!/usr/bin/env python3
"""
Combined data loader for Draft tab.

Main entry point for loading all data needed by the draft tab.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import re
import streamlit as st
from .draft_data import load_draft_data
from md.data_access import run_query, T


@st.cache_data(show_spinner=True, ttl=600)
def load_optimized_draft_data() -> Dict[str, Any]:
    """
    Load all data for draft tab in one optimized call.

    Returns:
        Dict with "Draft History" key containing DataFrame, or "error" key on failure
    """
    try:
        # Load draft data with column selection
        draft_data = load_draft_data()

        # Check for errors
        if "error" in draft_data:
            return {"error": draft_data["error"]}

        # Return the data
        return draft_data

    except Exception as e:
        st.error(f"Failed to load draft tab data: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=False, ttl=3600)
def load_roster_config_for_optimizer() -> Optional[Dict[str, Any]]:
    """
    Load roster configuration for the draft optimizer.

    Queries player table for lineup_position values from ONE manager's roster
    to determine roster structure. Very lightweight - only fetches position slots.

    Returns:
        Dict with position counts (qb, rb, wr, te, flex, def, k, bench) and budget,
        or None if detection fails
    """
    try:
        # Query ONE manager's lineup positions from the most recent week
        # DISTINCT gives us unique slot names (QB1, RB1, RB2, WR1, WR2, WR3, W/R/T1, BN1, etc.)
        sql = f"""
            SELECT DISTINCT lineup_position
            FROM {T['player']}
            WHERE year = (SELECT MAX(year) FROM {T['player']})
              AND week = 1
              AND lineup_position IS NOT NULL
              AND lineup_position != ''
              AND manager = (
                  SELECT manager FROM {T['player']}
                  WHERE year = (SELECT MAX(year) FROM {T['player']})
                    AND week = 1
                    AND manager IS NOT NULL
                    AND manager != ''
                  LIMIT 1
              )
        """
        df = run_query(sql)

        if df is None or df.empty:
            return None

        # Parse lineup positions using regex to extract base position
        # QB1 -> QB, RB2 -> RB, W/R/T1 -> W/R/T, BN3 -> BN, etc.
        position_counts = {}

        for _, row in df.iterrows():
            pos = str(row['lineup_position']).upper().strip()
            # Extract base position (remove trailing numbers)
            base_pos = re.sub(r'\d+$', '', pos)
            position_counts[base_pos] = position_counts.get(base_pos, 0) + 1

        # Convert to optimizer format
        # Known bench/IR positions
        bench_positions = ['BN', 'IR', 'IL']

        # Count bench slots
        bench_count = sum(position_counts.get(bp, 0) for bp in bench_positions)

        # Count flex positions (anything with / that's not bench)
        flex_count = 0
        for pos, count in position_counts.items():
            if '/' in pos and pos not in bench_positions:
                flex_count += count

        config = {
            "qb": position_counts.get('QB', 0),
            "rb": position_counts.get('RB', 0),
            "wr": position_counts.get('WR', 0),
            "te": position_counts.get('TE', 0),
            "flex": flex_count,
            "def": position_counts.get('DEF', 0),
            "k": position_counts.get('K', 0),
            "bench": bench_count,
            "budget": 200,
            "_position_counts": position_counts,  # For debugging
        }

        # Validate: should have at least 5 starter positions
        total_starters = sum(config[k] for k in ['qb', 'rb', 'wr', 'te', 'flex', 'def', 'k'])
        if total_starters < 5:
            return None

        return config

    except Exception as e:
        st.warning(f"Roster config detection failed: {e}")
        return None
