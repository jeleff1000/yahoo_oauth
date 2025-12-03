#!/usr/bin/env python3
"""
Combined data loader for Simulations tab.

Main entry point for loading all data needed by the simulations tab.
"""
from __future__ import annotations
from typing import Dict, Any
import streamlit as st
from .matchup_data import load_simulation_matchup_data


@st.cache_data(show_spinner=True, ttl=120)
def load_optimized_simulations_data() -> Dict[str, Any]:
    """
    Load all data for simulations tab in one optimized call.

    Note: Currently loads all matchup columns (SELECT *) because simulations
    depend on many dynamically generated shuffle_* columns. Future optimization
    could identify and exclude unused columns.

    Returns:
        Dict with keys:
            - "matchups": Full matchup DataFrame (all columns)
            - "error": Error message (if any)
    """
    try:
        # Load matchup data (currently all columns for simulation compatibility)
        matchup_result = load_simulation_matchup_data()
        if "error" in matchup_result:
            return {"error": matchup_result["error"]}

        return {
            "matchups": matchup_result["matchups"],
        }

    except Exception as e:
        st.error(f"Failed to load simulations data: {e}")
        return {"error": str(e)}
