"""
Optimized Weekly Player Stats Overview with better performance and UX.
This replaces the pagination approach with load-more and better filtering.
"""

import streamlit as st
import pandas as pd
from ..shared.modern_styles import apply_modern_styles

from .weekly_player_subprocesses.weekly_player_basic_stats import get_basic_stats
from .weekly_player_subprocesses.weekly_player_advanced_stats import get_advanced_stats
from .weekly_player_subprocesses.weekly_player_matchup_stats import (
    CombinedMatchupStatsViewer,
)
from .base.table_display import EnhancedTableDisplay
from .base.smart_filters import SmartFilterPanel
from .weekly_player_subprocesses.head_to_head import H2HViewer
from md.core import (
    list_optimal_weeks,
    list_player_seasons,
    list_player_weeks,
)
from md.tab_data_access.players import (
    load_filtered_weekly_data,
    load_optimal_week,
    load_player_week,
    load_players_weekly_data,
)


class OptimizedWeeklyPlayerViewer:
    """
    Optimized weekly player data viewer with:
    - Load more instead of pagination
    - Better filter UX with quick filters
    - Column selection to reduce table width
    - Export functionality
    - Efficient data loading
    """

    def __init__(self, player_data: pd.DataFrame):
        self.initial_data = player_data.copy()
        self.player_data = player_data.copy()

        # Normalize types
        for col in ["year", "week"]:
            if col in self.player_data.columns:
                self.player_data[col] = pd.to_numeric(
                    self.player_data[col], errors="coerce"
                )

        # Enhanced display managers for each tab
        self.display_basic = EnhancedTableDisplay("weekly_basic")
        self.display_advanced = EnhancedTableDisplay("weekly_advanced")
        self.display_matchup = EnhancedTableDisplay("weekly_matchup")

    def get_sort_columns_for_viewer(
        self, viewer_type: str, active_position: str = None
    ) -> list:
        """Get relevant sort columns based on viewer type and active position."""
        # Common columns always available
        common_cols = ["points", "player", "week", "year"]

        if viewer_type == "basic":
            # Basic stats - simple columns only
            base_cols = common_cols.copy()
            base_cols.extend(["manager", "nfl_team"])

            if active_position == "QB":
                base_cols.extend(
                    [
                        "pass_yds",
                        "pass_td",
                        "passing_interceptions",
                        "rush_yds",
                        "rush_td",
                    ]
                )
            elif active_position == "RB":
                base_cols.extend(["rush_yds", "rush_td", "rec", "rec_yds", "rec_td"])
            elif active_position in ["WR", "TE"]:
                base_cols.extend(["rec", "rec_yds", "rec_td", "targets"])
            elif active_position == "K":
                base_cols.extend(["fg_made", "fg_att"])
            elif active_position == "DEF":
                base_cols.extend(["def_sacks", "def_interceptions", "pts_allow"])
            else:
                # Mixed - show common stats
                base_cols.extend(["pass_yds", "rush_yds", "rec_yds", "rec"])

            return base_cols

        elif viewer_type == "advanced":
            # Advanced stats - include advanced metrics
            base_cols = common_cols.copy()
            base_cols.extend(["manager", "nfl_team"])

            if active_position == "QB":
                base_cols.extend(
                    [
                        "pass_yds",
                        "pass_td",
                        "passing_interceptions",
                        "passing_air_yards",
                        "passing_epa",
                        "passing_cpoe",
                        "rush_yds",
                        "rush_td",
                        "rushing_epa",
                    ]
                )
            elif active_position == "RB":
                base_cols.extend(
                    [
                        "rush_yds",
                        "rush_td",
                        "rushing_epa",
                        "rec",
                        "rec_yds",
                        "rec_td",
                        "receiving_epa",
                        "target_share",
                        "wopr",
                    ]
                )
            elif active_position in ["WR", "TE"]:
                base_cols.extend(
                    [
                        "rec",
                        "rec_yds",
                        "rec_td",
                        "targets",
                        "receiving_epa",
                        "target_share",
                        "wopr",
                        "racr",
                        "receiving_air_yards",
                        "air_yards_share",
                    ]
                )
            elif active_position == "K":
                base_cols.extend(["fg_made", "fg_att", "fg_pct"])
            elif active_position == "DEF":
                base_cols.extend(
                    ["def_sacks", "def_interceptions", "pts_allow", "def_td"]
                )
            else:
                # Mixed - show all common advanced stats
                base_cols.extend(
                    [
                        "pass_yds",
                        "passing_epa",
                        "rush_yds",
                        "rushing_epa",
                        "rec_yds",
                        "rec",
                        "receiving_epa",
                    ]
                )

            return base_cols

        elif viewer_type == "matchup":
            # Matchup stats - NO numerical stats, only matchup context columns
            # This is about fantasy matchup performance, not sorting by yards/TDs
            matchup_cols = [
                "points",
                "player",
                "week",
                "year",
                "manager",
                "opponent",
                "team_points",
                "opponent_points",
                "nfl_team",
            ]
            return matchup_cols

        return common_cols

    @st.fragment
    def display_sort_options(self, tab_name: str, active_position: str = None):
        """Display sort options for any tab."""
        sort_col_key = f"{tab_name}_sort_col"
        sort_dir_key = f"{tab_name}_sort_dir"

        # Get default sort column and direction - default to points DESC
        sort_col = st.session_state.get(sort_col_key, "points")
        sort_dir = st.session_state.get(sort_dir_key, "DESC")

        # Ensure defaults are set in session state if not present
        if sort_col_key not in st.session_state:
            st.session_state[sort_col_key] = "points"
        if sort_dir_key not in st.session_state:
            st.session_state[sort_dir_key] = "DESC"

        # Get relevant sort columns for this viewer
        available_sort_cols = self.get_sort_columns_for_viewer(
            tab_name, active_position
        )

        # Sorting controls in expander
        with st.expander("⬆️⬇️ Sort Options", expanded=False):
            st.info(
                "ℹ️ **Sort affects data loading**: The sort order determines which records are fetched from the database. "
                "For example, sorting by 'Rush Yds DESC' will load the top rushing performances, not just sort visible rows. "
                "Use filters + sort together to find specific stats (e.g., filter by position, then sort by yards)."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                # Make sure current sort_col is in available options
                if sort_col not in available_sort_cols:
                    sort_col = "points"

                sort_col = st.selectbox(
                    "Sort by",
                    options=available_sort_cols,
                    index=(
                        available_sort_cols.index(sort_col)
                        if sort_col in available_sort_cols
                        else 0
                    ),
                    key=f"{tab_name}_sort_select",
                )
            with col2:
                sort_dir = st.radio(
                    "Direction",
                    options=["DESC", "ASC"],
                    index=0 if sort_dir == "DESC" else 1,
                    key=f"{tab_name}_sort_direction",
                    horizontal=True,
                )
            with col3:
                if st.button("Reset to Points ⬇️", key=f"{tab_name}_sort_reset"):
                    st.session_state[sort_col_key] = "points"
                    st.session_state[sort_dir_key] = "DESC"
                    st.rerun()

            # Show example of what current sort does
            if sort_col and sort_dir:
                direction_text = (
                    "highest to lowest" if sort_dir == "DESC" else "lowest to highest"
                )
                st.caption(
                    f"📊 Currently loading: **{sort_col}** from {direction_text}"
                )

        # Store sort preferences
        st.session_state[sort_col_key] = sort_col
        st.session_state[sort_dir_key] = sort_dir

        return sort_col, sort_dir

    def get_unique_values(self, column, filters=None):
        """Get unique values for a column with optional filtering."""
        filtered_data = self.player_data if not filters else self.apply_filters(filters)
        if column not in filtered_data.columns:
            return []
        series = filtered_data[column].dropna()
        if column in ["year", "week"]:
            try:
                return sorted(series.astype(float).astype(int).unique().tolist())
            except Exception:
                return sorted(series.astype(str).unique().tolist())
        else:
            return sorted(series.astype(str).unique().tolist())

    def apply_filters(self, filters):
        """Apply filters to data."""
        df = self.player_data
        for column, values in (filters or {}).items():
            if values and column in df.columns:
                if column in ["year", "week"]:
                    df = df[
                        df[column]
                        .astype("Int64")
                        .isin(pd.Series(values, dtype="Int64"))
                    ]
                else:
                    df = df[df[column].astype(str).isin([str(v) for v in values])]
        return df

    def has_active_filters(self, filters):
        """Check if any filters are active."""
        return any(
            [
                filters.get("player_query"),
                filters.get("manager"),
                filters.get("opp_manager"),
                filters.get("position"),
                filters.get("nfl_position"),
                filters.get("fantasy_position"),
                filters.get("nfl_team"),
                filters.get("opponent_nfl_team"),
                filters.get("week"),
                filters.get("year"),
                filters.get("rostered_only"),
                filters.get("started_only"),
            ]
        )

    @st.fragment
    def display(self):
        apply_modern_styles()

        # Compact CSS for player stats page
        st.markdown(
            """
        <style>
        /* View mode label - caption size, low contrast */
        .view-mode-label {
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.45;
            margin-bottom: 0.75rem;
            font-weight: 500;
        }
        /* Tab container styling */
        .player-stats-view .stTabs {
            margin-top: 0 !important;
        }
        .player-stats-view .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
            background: rgba(128, 128, 128, 0.08);
            padding: 0.35rem;
            border-radius: 8px;
        }
        .player-stats-view .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 0.4rem 0.75rem !important;
            font-size: 0.85rem !important;
        }
        .player-stats-view .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--primary-color, #ff4b4b) !important;
            color: white !important;
        }
        /* Compact metrics row */
        .player-stats-view [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        .player-stats-view [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
        }
        /* Tighter table cell padding */
        .player-stats-view [data-testid="stDataFrame"] td {
            padding: 0.2rem 0.4rem !important;
        }
        </style>
        <div class="player-stats-view">
        """,
            unsafe_allow_html=True,
        )

        # View mode label above tabs - subtle caption
        st.markdown('<p class="view-mode-label">View Mode</p>', unsafe_allow_html=True)

        # Create tabs
        tabs = st.tabs(
            ["Basic Stats", "Advanced Stats", "Matchup Stats", "Head-to-Head"]
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ==================== BASIC STATS TAB ====================
        with tabs[0]:
            self._display_basic_stats_tab()

        # ==================== ADVANCED STATS TAB ====================
        with tabs[1]:
            self._display_advanced_stats_tab()

        # ==================== MATCHUP STATS TAB ====================
        with tabs[2]:
            self._display_matchup_stats_tab()

        # ==================== HEAD TO HEAD TAB ====================
        with tabs[3]:
            self._display_h2h_tab()

    @st.fragment
    def _display_basic_stats_tab(self):
        """Display Basic Stats tab with optimized layout."""
        # Collapsible filters (matching matchups style)
        with st.expander("🔎 Filters", expanded=False):
            filter_panel = SmartFilterPanel("basic", self)
            filters, active_position = filter_panel.display_filters()

        # Data display (full width)
        sort_col = "points"
        sort_dir = "DESC"

        if self.has_active_filters(filters):
            with st.spinner("Loading filtered data..."):
                limit = st.session_state.get("weekly_basic_displayed_rows", 5000)
                filtered_data = load_filtered_weekly_data(
                    filters, limit=limit, sort_column=sort_col, sort_direction=sort_dir
                )
                if filtered_data is not None:
                    if (
                        filters.get("rostered_only")
                        and "rostered" in filtered_data.columns
                    ):
                        filtered_data = filtered_data[filtered_data["rostered"]]
                    if (
                        filters.get("started_only")
                        and "started" in filtered_data.columns
                    ):
                        filtered_data = filtered_data[filtered_data["started"]]
                    basic_stats_df = get_basic_stats(
                        filtered_data, active_position or "All"
                    )
                    total_count = getattr(filtered_data, "attrs", {}).get(
                        "total_count", len(filtered_data)
                    )
                    self.display_basic.display_table_with_load_more(
                        basic_stats_df, total_available=total_count, height=600
                    )
                    self.display_basic.display_quick_export(
                        basic_stats_df, "weekly_basic_stats"
                    )
        else:
            limit = st.session_state.get("weekly_basic_displayed_rows", 5000)
            with st.spinner("Loading player data..."):
                sorted_data = load_players_weekly_data(
                    year=None,
                    week=None,
                    limit=limit,
                    offset=0,
                    sort_column=sort_col,
                    sort_direction=sort_dir,
                )
                if sorted_data is not None:
                    basic_stats_df = get_basic_stats(
                        sorted_data, active_position or "All"
                    )
                    total_count = getattr(sorted_data, "attrs", {}).get(
                        "total_count", len(sorted_data)
                    )
                    self.display_basic.display_table_with_load_more(
                        basic_stats_df, total_available=total_count, height=600
                    )
                    self.display_basic.display_quick_export(
                        basic_stats_df, "weekly_basic_stats"
                    )

    @st.fragment
    def _display_advanced_stats_tab(self):
        """Display Advanced Stats tab with optimized layout."""
        # Collapsible filters
        with st.expander("🔎 Filters", expanded=False):
            filter_panel = SmartFilterPanel("advanced", self)
            filters, active_position = filter_panel.display_filters()

        sort_col, sort_dir = "points", "DESC"

        if self.has_active_filters(filters):
            with st.spinner("Loading filtered data..."):
                limit = st.session_state.get("weekly_advanced_displayed_rows", 5000)
                filtered_data = load_filtered_weekly_data(
                    filters, limit=limit, sort_column=sort_col, sort_direction=sort_dir
                )
                if filtered_data is not None:
                    advanced_stats_df = get_advanced_stats(
                        filtered_data, active_position or "All"
                    )
                    total_count = getattr(filtered_data, "attrs", {}).get(
                        "total_count", len(filtered_data)
                    )
                    self.display_advanced.display_table_with_load_more(
                        advanced_stats_df, total_available=total_count, height=600
                    )
                    self.display_advanced.display_quick_export(
                        advanced_stats_df, "weekly_advanced_stats"
                    )
                else:
                    st.warning(
                        "No data returned from query. Please try adjusting your filters."
                    )
        else:
            limit = st.session_state.get("weekly_advanced_displayed_rows", 5000)
            with st.spinner("Loading player data..."):
                sorted_data = load_players_weekly_data(
                    year=None,
                    week=None,
                    limit=limit,
                    offset=0,
                    sort_column=sort_col,
                    sort_direction=sort_dir,
                )
                if sorted_data is not None:
                    advanced_stats_df = get_advanced_stats(
                        sorted_data, active_position or "All"
                    )
                    total_count = getattr(sorted_data, "attrs", {}).get(
                        "total_count", len(sorted_data)
                    )
                    self.display_advanced.display_table_with_load_more(
                        advanced_stats_df, total_available=total_count, height=600
                    )
                    self.display_advanced.display_quick_export(
                        advanced_stats_df, "weekly_advanced_stats"
                    )

    @st.fragment
    def _display_matchup_stats_tab(self):
        """Display Matchup Stats tab with optimized layout."""
        # Collapsible filters
        with st.expander("🔎 Filters", expanded=False):
            filter_panel = SmartFilterPanel("matchup", self)
            filters, active_position = filter_panel.display_filters()
            filters["rostered_only"] = True

        sort_col, sort_dir = "year", "DESC"
        with st.spinner("Loading matchup data..."):
            filtered_data = load_filtered_weekly_data(
                filters, limit=100000, sort_column=sort_col, sort_direction=sort_dir
            )
            if filtered_data is not None and not filtered_data.empty:
                st.markdown(f"**{len(filtered_data):,} matchup rows**")
                viewer = CombinedMatchupStatsViewer(filtered_data)
                viewer.display(prefix="matchup_stats")
            else:
                st.warning("No matchup data available with current filters.")

    @st.fragment
    def _display_h2h_tab(self):
        """Display Head-to-Head tab with compact selectors."""
        # Compact CSS for H2H selectors
        st.markdown(
            """
        <style>
        .h2h-selectors .stSelectbox {
            margin-bottom: 0 !important;
        }
        .h2h-selectors .stSelectbox > div > div {
            padding: 0.25rem 0.5rem !important;
            min-height: 2rem !important;
        }
        .h2h-selectors label {
            font-size: 0.75rem !important;
            margin-bottom: 0 !important;
        }
        </style>
        <div class="h2h-selectors">
        """,
            unsafe_allow_html=True,
        )

        # All selectors in one row
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            seasons = list_player_seasons()
            default_season_idx = len(seasons) - 1 if seasons else 0
            selected_season = st.selectbox(
                "Season",
                seasons,
                index=default_season_idx,
                key="h2h_season",
                label_visibility="collapsed",
            )
        with col2:
            weeks = list_player_weeks(selected_season) if selected_season else []
            default_week_idx = len(weeks) - 1 if weeks else 0
            selected_week = st.selectbox(
                "Week",
                weeks,
                index=default_week_idx,
                key="h2h_week",
                label_visibility="collapsed",
            )
        with col3:
            # Pre-load matchup options
            if selected_season and selected_week:
                player_week_data = load_player_week(selected_season, selected_week)
                if player_week_data is not None and not player_week_data.empty:
                    viewer = H2HViewer(player_week_data)
                    matchup_names = viewer.get_matchup_names()
                    optimal_weeks = list_optimal_weeks(selected_season)
                    if selected_week in optimal_weeks:
                        matchup_options = ["League Optimal"] + matchup_names
                    else:
                        matchup_options = matchup_names
                else:
                    matchup_options = []
            else:
                matchup_options = []

            selected_matchup = (
                st.selectbox(
                    "Matchup",
                    matchup_options,
                    key="h2h_selected_matchup",
                    label_visibility="collapsed",
                )
                if matchup_options
                else None
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Display the matchup immediately (no extra spacing)
        if selected_season and selected_week and selected_matchup:
            if "player_week_data" not in dir() or player_week_data is None:
                player_week_data = load_player_week(selected_season, selected_week)

            if player_week_data is not None and not player_week_data.empty:
                viewer = H2HViewer(player_week_data)

                if selected_matchup == "League Optimal":
                    optimal_data = load_optimal_week(selected_season, selected_week)
                    if optimal_data is not None and not optimal_data.empty:
                        viewer_optimal = H2HViewer(optimal_data)
                        viewer_optimal.display_league_optimal(prefix="h2h_optimal")
                else:
                    viewer.display(prefix="h2h_matchup", matchup_name=selected_matchup)
            else:
                st.warning("No player data available for this week/season.")
        elif not matchup_options:
            st.info("Select a season and week to view matchups.")
