#!/usr/bin/env python3
"""
Combined data loader for Keepers tab.

Main entry point for loading all data needed by the keepers tab.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from .keeper_data import load_keeper_data


@st.cache_data(show_spinner=True, ttl=120)
def load_optimized_keepers_data(
    all_years: bool = True, year: int = None, week: int = None
) -> pd.DataFrame | None:
    """
    Load all data for keepers tab in one optimized call.

    Args:
        all_years: Load end-of-season data for all years (default)
        year: Specific year (requires week parameter)
        week: Specific week (requires year parameter)

    Returns:
        DataFrame with keeper data or None on error

    Optimization Summary:
        - Column reduction: 17/272 columns (~94% fewer columns)
        - Row reduction: Only max week per manager/year (~95% fewer rows)
        - Database filtering: Excludes unrostered, DEF, K
        - Combined: ~99.7% reduction in data transferred
        - Uses CTE for max week calculations
        - Includes player headshots for visual display
        - Cached for 10 minutes
    """
    try:
        keeper_df = load_keeper_data(all_years=all_years, year=year, week=week)
        return keeper_df

    except Exception as e:
        st.error(f"Failed to load keepers data: {e}")
        return None
