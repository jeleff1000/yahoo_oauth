#!/usr/bin/env python3
"""Season Team Stats Overview - shows aggregated stats by manager and/or position."""
import streamlit as st
import pandas as pd

from .season_team_subprocesses.season_team_basic_stats import get_basic_stats as get_basic_stats_by_pos
from .season_team_subprocesses.season_team_advanced_stats import get_advanced_stats as get_advanced_stats_by_pos
from .season_team_subprocesses.season_team_basic_stats_by_manager import get_basic_stats as get_basic_stats_by_mgr
from .team_stats_visualizations import TeamStatsVisualizer

from .shared.theme import apply_theme_styles, render_empty_state
from ..shared.modern_styles import apply_modern_styles
from .shared.constants import TAB_LABELS
from .shared.column_config import get_column_config_for_position


class SeasonTeamViewer:
    """Season team data viewer - shows stats aggregated by manager and/or position."""

    def __init__(self, team_data_by_position: pd.DataFrame, team_data_by_manager: pd.DataFrame, team_data_by_lineup_position: pd.DataFrame = None):
        self.team_data_by_position = team_data_by_position.copy() if team_data_by_position is not None else pd.DataFrame()
        self.team_data_by_manager = team_data_by_manager.copy() if team_data_by_manager is not None else pd.DataFrame()
        self.team_data_by_lineup_position = team_data_by_lineup_position.copy() if team_data_by_lineup_position is not None else pd.DataFrame()

    def display(self):
        """Display season team stats with clean, mobile-friendly layout."""
        apply_theme_styles()
        apply_modern_styles()

        # Collapsible filters expander (like matchups)
        filters = self._render_filter_ui()

        # View tabs for different stat types - By Manager first
        view_tabs = st.tabs(["By Manager", "By Position", "By Lineup Slot", "📈 Visualizations"])

        with view_tabs[0]:
            self._display_by_manager(filters)

        with view_tabs[1]:
            self._display_by_position(filters)

        with view_tabs[2]:
            self._display_by_lineup_position(filters)

        with view_tabs[3]:
            self._display_visualizations(filters)

    def _render_filter_ui(self) -> dict:
        """Render compact collapsible filter UI matching matchups style."""
        # Compact filter styling
        st.markdown("""
        <style>
        [data-testid="stExpander"] .stMultiSelect,
        [data-testid="stExpander"] .stCheckbox,
        [data-testid="stExpander"] .stSelectbox {
            margin-bottom: 0.25rem !important;
        }
        [data-testid="stExpander"] [data-testid="column"] {
            padding: 0 0.25rem !important;
        }
        [data-testid="stExpander"] .stMarkdown p {
            margin-bottom: 0.25rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.expander("🔎 Filters", expanded=False):
            col1, col2 = st.columns(2)

            # Position filter
            with col1:
                positions = ["All", "QB", "RB", "WR", "TE", "K", "DEF", "W/R/T"]
                selected_position = st.selectbox(
                    "Position",
                    positions,
                    index=0,
                    key="season_team_position"
                )

            # Manager filter
            with col2:
                if not self.team_data_by_position.empty and 'manager' in self.team_data_by_position.columns:
                    managers = sorted(self.team_data_by_position['manager'].dropna().unique())
                    selected_managers = st.multiselect(
                        "Manager(s)",
                        managers,
                        default=[],
                        key="season_team_managers",
                        placeholder="All managers"
                    )
                    if not selected_managers:
                        selected_managers = managers
                else:
                    selected_managers = []

            # Year filter
            if not self.team_data_by_position.empty and 'year' in self.team_data_by_position.columns:
                years = sorted(self.team_data_by_position['year'].dropna().unique())
                selected_years = st.multiselect(
                    "Year(s)",
                    years,
                    default=[],
                    key="season_team_years",
                    placeholder="All years"
                )
                if not selected_years:
                    selected_years = years
            else:
                selected_years = []

            # Game type toggles (inline, compact)
            st.markdown(
                '<p style="margin: 0.5rem 0 0.25rem 0; font-size: 0.85rem; opacity: 0.7;">Game Types</p>',
                unsafe_allow_html=True
            )
            toggle_cols = st.columns(3)

            with toggle_cols[0]:
                regular_season = st.checkbox(
                    "Regular",
                    value=st.session_state.get("season_team_include_regular_season", True),
                    key="season_team_include_regular_season"
                )
            with toggle_cols[1]:
                playoffs = st.checkbox(
                    "Playoffs",
                    value=st.session_state.get("season_team_include_playoffs", True),
                    key="season_team_include_playoffs"
                )
            with toggle_cols[2]:
                consolation = st.checkbox(
                    "Consolation",
                    value=st.session_state.get("season_team_include_consolation", False),
                    key="season_team_include_consolation"
                )

        return {
            'position': selected_position,
            'managers': selected_managers,
            'years': selected_years,
            'regular_season': regular_season,
            'playoffs': playoffs,
            'consolation': consolation
        }

    def _apply_filters(self, data: pd.DataFrame, filters: dict, filter_position: bool = True) -> pd.DataFrame:
        """Apply filters to dataframe."""
        if data.empty:
            return data

        filtered = data.copy()

        # Position filter
        if filter_position and filters['position'] != "All" and 'fantasy_position' in filtered.columns:
            filtered = filtered[filtered['fantasy_position'] == filters['position']]

        # Year filter
        if filters['years'] and 'year' in filtered.columns:
            filtered = filtered[filtered['year'].isin(filters['years'])]

        # Manager filter
        if filters['managers'] and 'manager' in filtered.columns:
            filtered = filtered[filtered['manager'].isin(filters['managers'])]

        return filtered

    def _display_by_position(self, filters: dict):
        """Display stats grouped by manager and position."""
        if self.team_data_by_position.empty:
            render_empty_state(
                title="No Data Available",
                message="Season team data has not been loaded yet.",
                icon="📭"
            )
            return

        filtered_data = self._apply_filters(self.team_data_by_position, filters)

        if filtered_data.empty:
            st.warning("No data matches the selected filters.")
            return

        # Stats tabs
        stat_tabs = st.tabs([TAB_LABELS['basic_stats'], TAB_LABELS['advanced_stats']])

        with stat_tabs[0]:
            df = get_basic_stats_by_pos(filtered_data, filters['position'])
            self._render_data_table(df, "season_basic_pos", filters['position'])

        with stat_tabs[1]:
            df = get_advanced_stats_by_pos(filtered_data, filters['position'])
            self._render_data_table(df, "season_advanced_pos", filters['position'])

    def _display_by_lineup_position(self, filters: dict):
        """Display stats grouped by manager and lineup position."""
        if self.team_data_by_lineup_position.empty:
            render_empty_state(
                title="No Data Available",
                message="Season lineup position data has not been loaded yet.",
                icon="📭"
            )
            return

        # Apply filters (without position filter)
        filtered_data = self._apply_filters(self.team_data_by_lineup_position, filters, filter_position=False)

        # Lineup position selector
        if 'lineup_position' in filtered_data.columns:
            lineup_positions = ["All"] + sorted(filtered_data['lineup_position'].dropna().unique().tolist())
            selected_lp = st.selectbox(
                "Lineup Position",
                lineup_positions,
                index=0,
                key="season_team_lp_select"
            )
            if selected_lp != "All":
                filtered_data = filtered_data[filtered_data['lineup_position'] == selected_lp]

        if filtered_data.empty:
            st.warning("No data matches the selected filters.")
            return

        df = self._get_basic_stats_by_lineup_pos(filtered_data)
        self._render_data_table(df, "season_lineup_pos", "All")

    def _display_by_manager(self, filters: dict):
        """Display stats grouped by manager only."""
        if self.team_data_by_manager.empty:
            render_empty_state(
                title="No Data Available",
                message="Season manager data has not been loaded yet.",
                icon="📭"
            )
            return

        filtered_data = self._apply_filters(self.team_data_by_manager, filters, filter_position=False)

        if filtered_data.empty:
            st.warning("No data matches the selected filters.")
            return

        df = get_basic_stats_by_mgr(filtered_data)
        self._render_data_table(df, "season_manager", "All")

    def _display_visualizations(self, filters: dict):
        """Display visualizations."""
        filtered_pos_data = self._apply_filters(self.team_data_by_position, filters)
        filtered_mgr_data = self._apply_filters(self.team_data_by_manager, filters, filter_position=False)

        if filtered_pos_data.empty and filtered_mgr_data.empty:
            st.info("No data available for visualizations with the selected filters.")
            return

        visualizer = TeamStatsVisualizer(filtered_pos_data, filtered_mgr_data)
        visualizer.display_season_visualizations()

    def _get_basic_stats_by_lineup_pos(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get basic stats formatted for lineup position view."""
        if data.empty:
            return pd.DataFrame()

        display_cols = [col for col in ['manager', 'year', 'lineup_position', 'points', 'manager_spar', 'games_played', 'season_ppg']
                       if col in data.columns]

        if not display_cols:
            return pd.DataFrame()

        result = data[display_cols].copy()

        # Round numeric columns
        for col in ['points', 'manager_spar', 'season_ppg']:
            if col in result.columns:
                result[col] = result[col].round(2)

        # Rename columns
        rename_map = {
            'manager': 'Manager', 'year': 'Year',
            'lineup_position': 'Lineup Position',
            'points': 'Points', 'manager_spar': 'Manager SPAR',
            'games_played': 'Games', 'season_ppg': 'PPG'
        }
        result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

        # Sort
        sort_cols = [c for c in ['Year', 'Points'] if c in result.columns]
        if sort_cols:
            result = result.sort_values(sort_cols, ascending=[False, False][:len(sort_cols)])

        return result

    def _render_data_table(self, df: pd.DataFrame, prefix: str, position: str):
        """Render a data table with count and export button."""
        if df.empty:
            render_empty_state(
                title="No Data Found",
                message=f"No stats available for the selected filters",
                icon="🔍"
            )
            return

        # Get column config
        column_config = get_column_config_for_position(position)

        # Display table
        table_height = min(max(300, len(df) * 35), 600)
        st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=table_height
        )

        # Export button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{prefix}.csv",
            mime="text/csv",
            key=f"{prefix}_download"
        )
