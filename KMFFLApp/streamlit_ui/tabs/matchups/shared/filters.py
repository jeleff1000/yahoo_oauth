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
        Dictionary with filter selections:
        {
            'managers': List[str],
            'opponents': List[str],
            'years': List[int],
            'weeks': List[int] (if show_weeks=True),
            'positions': List[str] (if show_positions=True),
            'regular_season': bool,
            'playoffs': bool,
            'consolation': bool
        }
    """
    # Display data freshness info if provided
    if data_last_updated:
        time_ago = (datetime.now() - data_last_updated).total_seconds()
        if time_ago < 60:
            freshness = f"{int(time_ago)}s ago"
        elif time_ago < 3600:
            freshness = f"{int(time_ago / 60)}m ago"
        else:
            freshness = f"{int(time_ago / 3600)}h ago"

        st.caption(f"🕐 Data loaded: {freshness}")

    # Info box with tip
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
                padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;
                border-left: 4px solid #2196F3;">
    <p style="margin: 0; font-size: 0.9rem;">
    💡 <strong>Tip:</strong> Use filters below to customize your view. Leave filters empty to view all data.
    Filters maintain your tab position during calculations.
    </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔎 Filter Options", expanded=True):
        st.caption("Filter data by managers, opponents, years, and game types")

        # === MAIN FILTERS ===
        # Determine number of columns based on what's shown
        num_cols = 3 + (1 if show_weeks else 0) + (1 if show_positions else 0)
        cols = st.columns([1] * num_cols)

        col_idx = 0

        # Manager filter
        with cols[col_idx]:
            managers = sorted(df['manager'].unique().tolist())
            selected_managers = st.multiselect(
                "Manager(s)",
                managers,
                default=[],
                key=f"{prefix}_managers",
                help="Filter by manager. Leave empty for all."
            )
            if not selected_managers:
                selected_managers = managers
        col_idx += 1

        # Opponent filter
        with cols[col_idx]:
            opponents = sorted(df['opponent'].unique().tolist())
            selected_opponents = st.multiselect(
                "Opponent(s)",
                opponents,
                default=[],
                key=f"{prefix}_opponents",
                help="Filter by opponent. Leave empty for all."
            )
            if not selected_opponents:
                selected_opponents = opponents
        col_idx += 1

        # Year filter
        with cols[col_idx]:
            years = sorted(df['year'].astype(int).unique().tolist())
            selected_years = st.multiselect(
                "Year(s)",
                years,
                default=[],
                key=f"{prefix}_years",
                help="Filter by year. Leave empty for all."
            )
            if not selected_years:
                selected_years = years
        col_idx += 1

        # Optional week filter
        selected_weeks = []
        if show_weeks:
            with cols[col_idx]:
                weeks = sorted(df['week'].astype(int).unique().tolist())
                selected_weeks = st.multiselect(
                    "Week(s)",
                    weeks,
                    default=[],
                    key=f"{prefix}_weeks",
                    help="Filter by week. Leave empty for all."
                )
                if not selected_weeks:
                    selected_weeks = weeks
            col_idx += 1

        # Optional position filter
        selected_positions = []
        if show_positions and "position" in df.columns:
            with cols[col_idx]:
                positions = sorted(df['position'].dropna().unique().tolist())
                selected_positions = st.multiselect(
                    "Position(s)",
                    positions,
                    default=[],
                    key=f"{prefix}_positions",
                    help="Filter by position. Leave empty for all."
                )
                if not selected_positions:
                    selected_positions = positions

        # === QUICK FILTERS ROW ===
        st.markdown("---")
        st.markdown("**Quick Filters:**")
        quick_filter_cols = st.columns([1, 1, 1, 1])

        # Game type checkboxes
        with quick_filter_cols[0]:
            regular_season = st.checkbox(
                "Regular Season",
                value=True,
                key=f"{prefix}_regular_season",
                help="Include regular season games"
            )
        with quick_filter_cols[1]:
            playoffs = st.checkbox(
                "Playoffs",
                value=True,
                key=f"{prefix}_playoffs",
                help="Include playoff games"
            )
        with quick_filter_cols[2]:
            consolation = st.checkbox(
                "Consolation",
                value=False,
                key=f"{prefix}_consolation",
                help="Include consolation bracket games"
            )
        with quick_filter_cols[3]:
            # Result filter (optional, can be used in weekly view)
            result_filter = st.selectbox(
                "Results",
                ["All", "Wins Only", "Losses Only"],
                key=f"{prefix}_result_filter",
                help="Filter by game outcome"
            )

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
    Apply filters to dataframe with loading indicator.

    Args:
        df: Source DataFrame
        filters: Dictionary of filter selections from render_filter_ui()

    Returns:
        Filtered DataFrame
    """
    with st.spinner("🔄 Applying filters..."):
        filtered_df = filter_matchup_data(
            _cache_key=None,  # Can be enhanced to track filter changes
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

    # Show result count
    total_rows = len(df)
    filtered_rows = len(filtered_df)

    if filtered_rows < total_rows:
        st.info(f"📊 Showing {filtered_rows:,} of {total_rows:,} matchups ({filtered_rows/total_rows*100:.1f}%)")
    else:
        st.info(f"📊 Showing all {total_rows:,} matchups")

    return filtered_df
