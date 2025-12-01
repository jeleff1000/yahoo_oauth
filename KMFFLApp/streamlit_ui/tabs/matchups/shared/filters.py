"""
Shared filtering logic and UI components for matchups tab.
Centralizes filter operations to reduce code duplication.
"""
import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any
from datetime import datetime


@st.cache_data(show_spinner=False)
def filter_matchup_data(
    _cache_key: Any,
    df: pd.DataFrame,
    selected_managers: List[str],
    selected_opponents: List[str],
    selected_years: List[int],
    selected_weeks: Optional[List[int]] = None,
    selected_positions: Optional[List[str]] = None,
    regular_season: bool = True,
    playoffs: bool = True,
    consolation: bool = False,
    result_filter: str = "All",
) -> pd.DataFrame:
    """
    Filter matchup data based on user selections with loading indicator.

    Args:
        _cache_key: Cache invalidation key (pass None to force refresh)
        df: Source DataFrame to filter
        selected_managers: List of manager names to include
        selected_opponents: List of opponent names to include
        selected_years: List of years to include
        selected_weeks: Optional list of weeks to include (for weekly views)
        selected_positions: Optional list of positions to include (for career views)
        regular_season: Include regular season games
        playoffs: Include playoff games
        consolation: Include consolation games
        result_filter: Filter by result ("All", "Wins Only", "Losses Only")

    Returns:
        Filtered DataFrame
    """
    # Start with manager and opponent filters
    filtered_df = df[
        df['manager'].isin(selected_managers) &
        df['opponent'].isin(selected_opponents)
    ]

    # Apply result filter
    if result_filter == "Wins Only" and 'win' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['win'] == 1]
    elif result_filter == "Losses Only" and 'win' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['win'] == 0]

    # Apply game type filters if any are selected
    if regular_season or playoffs or consolation:
        conditions = []
        if regular_season:
            conditions.append(
                (filtered_df['is_playoffs'] == 0) &
                (filtered_df['is_consolation'] == 0)
            )
        if playoffs:
            conditions.append(filtered_df['is_playoffs'] == 1)
        if consolation:
            conditions.append(filtered_df['is_consolation'] == 1)

        # Combine conditions with OR logic
        if conditions:
            filtered_df = filtered_df[pd.concat(conditions, axis=1).any(axis=1)]

    # Apply year filter
    if selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]

    # Apply week filter (for weekly views)
    if selected_weeks is not None and len(selected_weeks) > 0:
        filtered_df = filtered_df[filtered_df['week'].isin(selected_weeks)]

    # Apply position filter (for career views with position data)
    if selected_positions and "position" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['position'].isin(selected_positions)]

    return filtered_df


def render_filter_ui(
    df: pd.DataFrame,
    prefix: str = "",
    show_weeks: bool = False,
    show_positions: bool = False,
    data_last_updated: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Render consistent filter UI for matchup data views.

    Args:
        df: DataFrame containing the data to filter
        prefix: Unique prefix for widget keys to avoid conflicts
        show_weeks: Whether to show week filter
        show_positions: Whether to show position filter
        data_last_updated: Optional timestamp of when data was last loaded

    Returns:
        Dictionary with filter selections
    """
    # Compact filter expander - collapsed by default to reduce clutter
    with st.expander("🔎 Filters", expanded=False):
        # Row 1: Main entity filters (2 columns on mobile, 4 on desktop)
        col1, col2 = st.columns(2)

        with col1:
            managers = sorted(df['manager'].unique().tolist())
            selected_managers = st.multiselect(
                "Manager(s)",
                managers,
                default=[],
                key=f"{prefix}_managers",
                placeholder="All managers"
            )
            if not selected_managers:
                selected_managers = managers

        with col2:
            opponents = sorted(df['opponent'].unique().tolist())
            selected_opponents = st.multiselect(
                "Opponent(s)",
                opponents,
                default=[],
                key=f"{prefix}_opponents",
                placeholder="All opponents"
            )
            if not selected_opponents:
                selected_opponents = opponents

        # Row 2: Time filters
        time_cols = st.columns(2 if show_weeks else 1)

        with time_cols[0]:
            years = sorted(df['year'].astype(int).unique().tolist())
            selected_years = st.multiselect(
                "Year(s)",
                years,
                default=[],
                key=f"{prefix}_years",
                placeholder="All years"
            )
            if not selected_years:
                selected_years = years

        selected_weeks = []
        if show_weeks:
            with time_cols[1]:
                weeks = sorted(df['week'].astype(int).unique().tolist())
                selected_weeks = st.multiselect(
                    "Week(s)",
                    weeks,
                    default=[],
                    key=f"{prefix}_weeks",
                    placeholder="All weeks"
                )
                if not selected_weeks:
                    selected_weeks = weeks

        # Optional position filter
        selected_positions = []
        if show_positions and "position" in df.columns:
            positions = sorted(df['position'].dropna().unique().tolist())
            selected_positions = st.multiselect(
                "Position(s)",
                positions,
                default=[],
                key=f"{prefix}_positions",
                placeholder="All positions"
            )
            if not selected_positions:
                selected_positions = positions

        # Row 3: Game type toggles (inline, compact)
        st.markdown(
            '<p style="margin: 0.5rem 0 0.25rem 0; font-size: 0.85rem; opacity: 0.7;">Game Types</p>',
            unsafe_allow_html=True
        )
        toggle_cols = st.columns(4)

        with toggle_cols[0]:
            regular_season = st.checkbox("Regular", value=True, key=f"{prefix}_regular_season")
        with toggle_cols[1]:
            playoffs = st.checkbox("Playoffs", value=True, key=f"{prefix}_playoffs")
        with toggle_cols[2]:
            consolation = st.checkbox("Consolation", value=False, key=f"{prefix}_consolation")
        with toggle_cols[3]:
            result_filter = st.selectbox(
                "Result",
                ["All", "Wins", "Losses"],
                key=f"{prefix}_result_filter",
                label_visibility="collapsed"
            )
            # Map back to original values
            if result_filter == "Wins":
                result_filter = "Wins Only"
            elif result_filter == "Losses":
                result_filter = "Losses Only"

    # Return all filter selections
    result = {
        'managers': selected_managers,
        'opponents': selected_opponents,
        'years': selected_years,
        'regular_season': regular_season,
        'playoffs': playoffs,
        'consolation': consolation,
        'result_filter': result_filter,
    }

    if show_weeks:
        result['weeks'] = selected_weeks

    if show_positions:
        result['positions'] = selected_positions

    return result


def apply_filters_with_loading(
    df: pd.DataFrame,
    filters: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply filters to dataframe.

    Args:
        df: Source DataFrame
        filters: Dictionary of filter selections from render_filter_ui()

    Returns:
        Filtered DataFrame
    """
    filtered_df = filter_matchup_data(
        _cache_key=None,
        df=df,
        selected_managers=filters['managers'],
        selected_opponents=filters['opponents'],
        selected_years=filters['years'],
        selected_weeks=filters.get('weeks'),
        selected_positions=filters.get('positions'),
        regular_season=filters['regular_season'],
        playoffs=filters['playoffs'],
        consolation=filters['consolation'],
        result_filter=filters.get('result_filter', 'All'),
    )

    # Only show count if filters are active (not showing all data)
    total_rows = len(df)
    filtered_rows = len(filtered_df)

    if filtered_rows < total_rows:
        st.caption(f"Showing {filtered_rows:,} of {total_rows:,} matchups")

    return filtered_df
