#!/usr/bin/env python3
"""
Team name data loader for Team Names tab.

Optimization:
    - Columns: 5 of 276 (~98% reduction)
    - Rows: Distinct manager/year combinations only
    - Database-level DISTINCT reduces duplicates

Columns loaded (5 of 276):
    Core Identity (2): manager, team_name
    Context (2): year, division_id
    League (1): league_id

Row Filtering:
    - DISTINCT on manager, year, team_name, division_id, league_id
    - Removes weekly duplicates (since team names don't change mid-season)
    - Result: ~95% fewer rows than full matchup table
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from md.data_access import run_query, T


@st.cache_data(show_spinner=True, ttl=120)
def load_team_name_data() -> pd.DataFrame | None:
    """
    Load team name data from matchup table with column selection optimization.

    Returns unique combinations of manager, team name, division, and year.
    Since team names don't change during a season, we use DISTINCT to eliminate
    weekly duplicates.

    Returns:
        DataFrame with team name data or None on error

    Optimization:
        - Loads only 5/276 columns (~98% reduction)
        - DISTINCT reduces rows by ~95% (eliminates weekly duplicates)
        - Ordered by year DESC, manager ASC for easy viewing
    """
    try:
        team_name_query = f"""
            SELECT DISTINCT
                manager,
                team_name,
                year,
                division_id,
                league_id
            FROM {T['matchup']}
            WHERE manager IS NOT NULL
              AND LOWER(TRIM(manager)) NOT IN ('no manager', 'unrostered', '')
            ORDER BY year DESC, manager ASC
        """

        df = run_query(team_name_query)
        return df

    except Exception as e:
        st.error(f"Failed to load team name data: {e}")
        return None
