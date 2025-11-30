#!/usr/bin/env python3
"""Weekly Team Stats Overview - shows aggregated stats by manager and/or position."""
import streamlit as st
import pandas as pd

from .weekly_team_subprocesses.weekly_team_basic_stats import get_basic_stats as get_basic_stats_by_pos
from .weekly_team_subprocesses.weekly_team_advanced_stats import get_advanced_stats as get_advanced_stats_by_pos
from .weekly_team_subprocesses.weekly_team_basic_stats_by_manager import get_basic_stats as get_basic_stats_by_mgr
from .team_stats_visualizations import TeamStatsVisualizer

# Import new shared components
from .shared.theme import (
    apply_theme_styles,
    render_gradient_header,
    render_info_box,
    render_empty_state,
    render_metric_grid,
    render_filter_count
)
from .shared.modern_styles import apply_modern_styles
from .shared.filters import render_simple_position_filter
from .shared.constants import TAB_LABELS
from .shared.column_config import get_column_config_for_position
from .shared.table_formatting import create_summary_stats


class WeeklyTeamViewer:
    """Weekly team data viewer - shows stats aggregated by manager and/or position."""

    def __init__(self, team_data_by_position: pd.DataFrame, team_data_by_manager: pd.DataFrame, team_data_by_lineup_position: pd.DataFrame = None):
        self.team_data_by_position = team_data_by_position.copy() if team_data_by_position is not None else pd.DataFrame()
        self.team_data_by_manager = team_data_by_manager.copy() if team_data_by_manager is not None else pd.DataFrame()
        self.team_data_by_lineup_position = team_data_by_lineup_position.copy() if team_data_by_lineup_position is not None else pd.DataFrame()

    def display(self):
        """Display weekly team stats."""
        # Apply styling
        apply_theme_styles()
        apply_modern_styles()

        # Gradient header
        render_gradient_header(
            title="Weekly Team Stats",
            subtitle="Analyze weekly performance by position and manager",
            icon="📊"
        )

        # Grouping selector
        grouping_options = ["By Manager & Position", "By Manager & Lineup Position", "By Manager Only"]
        selected_grouping = st.radio(
            "Group By",
            grouping_options,
            horizontal=True,
            key="weekly_team_grouping",
            label_visibility="visible"
        )

        if selected_grouping == "By Manager & Position":
            self.display_by_position()
        elif selected_grouping == "By Manager & Lineup Position":
            self.display_by_lineup_position()
        else:
            self.display_by_manager()

    def display_by_position(self):
        """Display stats grouped by manager and position with player_stats-style layout."""
        if self.team_data_by_position.empty:
            render_empty_state(
                title="No Team Data Available",
                message="Weekly team data has not been loaded yet.",
                icon="📭"
            )
            return

        # Use player_stats style: filters on left (1), data on right (3)
        filter_col, data_col = st.columns([1, 3])

        with filter_col:
            st.markdown("### 🎛️ Filters")

            render_info_box(
                "Filter by position to see manager performance for that position group",
                icon="ℹ️"
            )

            # Position filter
            selected_position = render_simple_position_filter(
                prefix="weekly_team",
                default="All"
            )

            st.markdown("---")

            # Week type filters (Regular Season, Playoffs, Consolation)
            st.markdown("**🏆 Week Type Filters:**")
            st.checkbox(
                "Regular Season",
                value=st.session_state.get("weekly_team_include_regular_season", True),
                key="weekly_team_include_regular_season",
                help="Include regular season weeks"
            )
            playoff_col1, playoff_col2 = st.columns(2)
            with playoff_col1:
                st.checkbox(
                    "Playoffs",
                    value=st.session_state.get("weekly_team_include_playoffs", True),
                    key="weekly_team_include_playoffs",
                    help="Include weeks from the fantasy playoffs"
                )
            with playoff_col2:
                st.checkbox(
                    "Consolation",
                    value=st.session_state.get("weekly_team_include_consolation", False),
                    key="weekly_team_include_consolation",
                    help="Include weeks from the consolation bracket"
                )

            st.markdown("---")

            # Year filter (defaults to empty = all years)
            st.markdown("**📅 Year Range:**")
            if 'year' in self.team_data_by_position.columns:
                years = sorted(self.team_data_by_position['year'].dropna().unique())
                if years:
                    selected_years = st.multiselect(
                        "Filter by Year",
                        options=years,
                        default=[],  # Empty means all years
                        key="weekly_team_years",
                        label_visibility="collapsed",
                        help="Leave empty to show all years"
                    )
                else:
                    selected_years = []
            else:
                selected_years = []

            st.markdown("---")

            # Manager filter
            st.markdown("**👤 Manager Filter:**")
            if 'manager' in self.team_data_by_position.columns:
                managers = sorted(self.team_data_by_position['manager'].dropna().unique())
                selected_managers = st.multiselect(
                    "Select Manager(s)",
                    options=managers,
                    default=[],
                    key="weekly_team_managers",
                    help="Leave empty to show all managers",
                    label_visibility="collapsed"
                )
            else:
                selected_managers = []

            st.markdown("---")

            # Active filters summary
            st.markdown("**🎯 Active Filters:**")
            active_filters = []
            if selected_position != "All":
                active_filters.append(f"Position: {selected_position}")
            if not st.session_state.get("weekly_team_include_regular_season", True):
                active_filters.append("Regular Season: Excluded")
            if not st.session_state.get("weekly_team_include_playoffs", True):
                active_filters.append("Playoffs: Excluded")
            if st.session_state.get("weekly_team_include_consolation", False):
                active_filters.append("Consolation: Included")
            if selected_years:
                if len(selected_years) <= 3:
                    active_filters.append(f"Years: {', '.join(map(str, selected_years))}")
                else:
                    active_filters.append(f"Years: {min(selected_years)}-{max(selected_years)}")
            if selected_managers:
                if len(selected_managers) <= 2:
                    active_filters.append(f"Managers: {', '.join(selected_managers)}")
                else:
                    active_filters.append(f"Managers: {len(selected_managers)} selected")

            if active_filters:
                for filter_text in active_filters:
                    st.caption(f"🏷️ {filter_text}")
            else:
                st.caption("No filters active")

            # Clear all filters button
            if active_filters:
                if st.button("🔄 Clear All Filters", key="weekly_team_clear_all", use_container_width=True):
                    # Clear session state for filters
                    for key in list(st.session_state.keys()):
                        if key.startswith("weekly_team"):
                            del st.session_state[key]
                    st.rerun()

        with data_col:
            # Apply filters to data
            filtered_data = self.team_data_by_position.copy()

            # Position filter
            if selected_position != "All":
                filtered_data = filtered_data[filtered_data["fantasy_position"] == selected_position]

            # Year filter
            if selected_years:
                filtered_data = filtered_data[filtered_data['year'].isin(selected_years)]

            # Manager filter
            if selected_managers:
                filtered_data = filtered_data[filtered_data['manager'].isin(selected_managers)]

            # Show quick metrics for filtered data
            self._display_quick_metrics_for_data(filtered_data, selected_position)

            # Stat type tabs
            stat_tabs = st.tabs([
                TAB_LABELS['basic_stats'],
                TAB_LABELS['advanced_stats'],
                TAB_LABELS['visualizations']
            ])

            with stat_tabs[0]:
                df = get_basic_stats_by_pos(filtered_data, selected_position)
                if df.empty:
                    render_empty_state(
                        title="No Data Found",
                        message=f"No basic stats available for {selected_position}",
                        icon="🔍"
                    )
                else:
                    # Show data count
                    render_filter_count(len(df), len(filtered_data))

                    # Get intelligent column config for this position
                    column_config = get_column_config_for_position(selected_position)

                    # Display summary stats for numeric columns
                    with st.expander("📊 Summary Statistics", expanded=False):
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        key_stats = [c for c in numeric_cols if c in ['Points', 'SPAR', 'Pass Yds', 'Rush Yds', 'Rec Yds']]
                        if key_stats:
                            summary_df = create_summary_stats(df, key_stats)
                            st.dataframe(summary_df, use_container_width=True)

                    # Display table with dynamic height and enhanced column config
                    table_height = min(max(300, len(df) * 35), 700)
                    st.dataframe(
                        df,
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,
                        height=table_height
                    )

                    # Export button
                    self._render_export_button(df, "weekly_basic_stats")

            with stat_tabs[1]:
                df = get_advanced_stats_by_pos(filtered_data, selected_position)
                if df.empty:
                    render_empty_state(
                        title="No Data Found",
                        message=f"No advanced stats available for {selected_position}",
                        icon="🔍"
                    )
                else:
                    # Show data count
                    render_filter_count(len(df), len(filtered_data))

                    # Get intelligent column config for this position
                    column_config = get_column_config_for_position(selected_position)

                    # Display table with dynamic height and enhanced column config
                    table_height = min(max(300, len(df) * 35), 700)
                    st.dataframe(
                        df,
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,
                        height=table_height
                    )

                    # Export button
                    self._render_export_button(df, "weekly_advanced_stats")

            with stat_tabs[2]:
                visualizer = TeamStatsVisualizer(filtered_data, self.team_data_by_manager)
                visualizer.display_weekly_visualizations()

    def display_by_lineup_position(self):
        """Display stats grouped by manager and lineup position (WR1, WR2, RB1, etc.)."""
        if self.team_data_by_lineup_position.empty:
            render_empty_state(
                title="No Team Data Available",
                message="Weekly team data by lineup position has not been loaded yet.",
                icon="📭"
            )
            return

        # Use player_stats style: filters on left (1), data on right (3)
        filter_col, data_col = st.columns([1, 3])

        with filter_col:
            st.markdown("### 🎛️ Filters")

            render_info_box(
                "Filter by lineup position (WR1, WR2, RB1, etc.) to see manager performance",
                icon="ℹ️"
            )

            # Lineup Position filter
            st.markdown("**📍 Lineup Position:**")
            if 'lineup_position' in self.team_data_by_lineup_position.columns:
                lineup_positions = sorted(self.team_data_by_lineup_position['lineup_position'].dropna().unique())
                lineup_options = ["All"] + list(lineup_positions)
                selected_lineup_pos = st.selectbox(
                    "Select Lineup Position",
                    options=lineup_options,
                    index=0,
                    key="weekly_team_lineup_position",
                    label_visibility="collapsed"
                )
            else:
                selected_lineup_pos = "All"

            st.markdown("---")

            # Week type filters (Regular Season, Playoffs, Consolation)
            # Use the same keys as the position view so data loader gets updated values
            st.markdown("**🏆 Week Type Filters:**")
            st.checkbox(
                "Regular Season",
                value=st.session_state.get("weekly_team_include_regular_season", True),
                key="weekly_team_include_regular_season",
                help="Include regular season weeks"
            )
            playoff_col1, playoff_col2 = st.columns(2)
            with playoff_col1:
                st.checkbox(
                    "Playoffs",
                    value=st.session_state.get("weekly_team_include_playoffs", True),
                    key="weekly_team_include_playoffs",
                    help="Include weeks from the fantasy playoffs"
                )
            with playoff_col2:
                st.checkbox(
                    "Consolation",
                    value=st.session_state.get("weekly_team_include_consolation", False),
                    key="weekly_team_include_consolation",
                    help="Include weeks from the consolation bracket"
                )

            st.markdown("---")

            # Year filter
            st.markdown("**📅 Year Range:**")
            if 'year' in self.team_data_by_lineup_position.columns:
                years = sorted(self.team_data_by_lineup_position['year'].dropna().unique())
                if years:
                    selected_years = st.multiselect(
                        "Filter by Year",
                        options=years,
                        default=[],
                        key="weekly_team_lp_years",
                        label_visibility="collapsed",
                        help="Leave empty to show all years"
                    )
                else:
                    selected_years = []
            else:
                selected_years = []

            st.markdown("---")

            # Manager filter
            st.markdown("**👤 Manager Filter:**")
            if 'manager' in self.team_data_by_lineup_position.columns:
                managers = sorted(self.team_data_by_lineup_position['manager'].dropna().unique())
                selected_managers = st.multiselect(
                    "Select Manager(s)",
                    options=managers,
                    default=[],
                    key="weekly_team_lp_managers",
                    help="Leave empty to show all managers",
                    label_visibility="collapsed"
                )
            else:
                selected_managers = []

            st.markdown("---")

            # Active filters summary
            st.markdown("**🎯 Active Filters:**")
            active_filters = []
            if selected_lineup_pos != "All":
                active_filters.append(f"Lineup Pos: {selected_lineup_pos}")
            if not st.session_state.get("weekly_team_include_regular_season", True):
                active_filters.append("Regular Season: Excluded")
            if not st.session_state.get("weekly_team_include_playoffs", True):
                active_filters.append("Playoffs: Excluded")
            if st.session_state.get("weekly_team_include_consolation", False):
                active_filters.append("Consolation: Included")
            if selected_years:
                if len(selected_years) <= 3:
                    active_filters.append(f"Years: {', '.join(map(str, selected_years))}")
                else:
                    active_filters.append(f"Years: {min(selected_years)}-{max(selected_years)}")
            if selected_managers:
                if len(selected_managers) <= 2:
                    active_filters.append(f"Managers: {', '.join(selected_managers)}")
                else:
                    active_filters.append(f"Managers: {len(selected_managers)} selected")

            if active_filters:
                for filter_text in active_filters:
                    st.caption(f"🏷️ {filter_text}")
            else:
                st.caption("No filters active")

            # Clear all filters button
            if active_filters:
                if st.button("🔄 Clear All Filters", key="weekly_team_lp_clear_all", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        if key.startswith("weekly_team"):
                            del st.session_state[key]
                    st.rerun()

        with data_col:
            # Apply filters to data
            filtered_data = self.team_data_by_lineup_position.copy()

            # Lineup position filter
            if selected_lineup_pos != "All":
                filtered_data = filtered_data[filtered_data["lineup_position"] == selected_lineup_pos]

            # Year filter
            if selected_years:
                filtered_data = filtered_data[filtered_data['year'].isin(selected_years)]

            # Manager filter
            if selected_managers:
                filtered_data = filtered_data[filtered_data['manager'].isin(selected_managers)]

            # Show quick metrics for filtered data
            self._display_quick_metrics_for_lineup_data(filtered_data, selected_lineup_pos)

            # Stat type tabs
            stat_tabs = st.tabs([
                TAB_LABELS['basic_stats'],
                TAB_LABELS['visualizations']
            ])

            with stat_tabs[0]:
                df = self._get_basic_stats_by_lineup_pos(filtered_data, selected_lineup_pos)
                if df.empty:
                    render_empty_state(
                        title="No Data Found",
                        message=f"No basic stats available for {selected_lineup_pos}",
                        icon="🔍"
                    )
                else:
                    # Show data count
                    render_filter_count(len(df), len(filtered_data))

                    # Get column config
                    column_config = get_column_config_for_position("All")

                    # Display table with dynamic height
                    table_height = min(max(300, len(df) * 35), 700)
                    st.dataframe(
                        df,
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,
                        height=table_height
                    )

                    # Export button
                    self._render_export_button(df, "weekly_lineup_pos_stats")

            with stat_tabs[1]:
                visualizer = TeamStatsVisualizer(filtered_data, self.team_data_by_manager)
                visualizer.display_weekly_visualizations()

    def _get_basic_stats_by_lineup_pos(self, data: pd.DataFrame, lineup_position: str) -> pd.DataFrame:
        """Get basic stats formatted for lineup position view."""
        if data.empty:
            return pd.DataFrame()

        # Select columns to display (weekly granularity - no aggregation)
        display_cols = []
        for col in ['manager', 'year', 'week', 'lineup_position', 'player', 'points', 'manager_spar']:
            if col in data.columns:
                display_cols.append(col)

        if not display_cols:
            return pd.DataFrame()

        result = data[display_cols].copy()

        # Round numeric columns for cleaner display
        if 'points' in result.columns:
            result['points'] = result['points'].round(2)
        if 'manager_spar' in result.columns:
            result['manager_spar'] = result['manager_spar'].round(2)

        # Rename columns for display
        rename_map = {
            'manager': 'Manager',
            'year': 'Year',
            'week': 'Week',
            'lineup_position': 'Lineup Position',
            'player': 'Player',
            'points': 'Points',
            'manager_spar': 'Manager SPAR'
        }
        result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

        # Sort by year desc, week desc, then manager
        sort_cols = []
        if 'Year' in result.columns:
            sort_cols.append('Year')
        if 'Week' in result.columns:
            sort_cols.append('Week')
        if 'Manager' in result.columns:
            sort_cols.append('Manager')

        if sort_cols:
            result = result.sort_values(sort_cols, ascending=[False, False, True] if len(sort_cols) == 3 else [False] * len(sort_cols))

        return result

    def _display_quick_metrics_for_lineup_data(self, data: pd.DataFrame, lineup_position: str):
        """Display quick summary metrics for lineup position data."""
        try:
            if data.empty:
                return

            # Calculate metrics
            total_points = data["points"].sum() if "points" in data.columns else 0
            avg_points = data["points"].mean() if "points" in data.columns else 0
            total_records = len(data)
            unique_managers = data["manager"].nunique() if "manager" in data.columns else 0

            # Display metrics in grid
            metrics = [
                {"label": "Total Points", "value": f"{total_points:,.1f}"},
                {"label": "Avg Points", "value": f"{avg_points:.1f}"},
                {"label": "Total Records", "value": f"{total_records:,}"},
                {"label": "Managers", "value": f"{unique_managers}"},
            ]

            render_metric_grid(metrics, columns=4)

        except Exception:
            pass

    def display_by_manager(self):
        """Display stats grouped by manager only."""
        if self.team_data_by_manager.empty:
            render_empty_state(
                title="No Team Data Available",
                message="Weekly team data has not been loaded yet.",
                icon="📭"
            )
            return

        # Use player_stats style: filters on left (1), data on right (3)
        filter_col, data_col = st.columns([1, 3])

        with filter_col:
            st.markdown("### 🎛️ Filters")

            render_info_box(
                "Aggregated stats for each manager across all positions",
                icon="ℹ️"
            )

            st.markdown("---")

            # Week type filters (Regular Season, Playoffs, Consolation) - shared with position view
            # Use the same keys as the position view so data loader gets updated values
            st.markdown("**🏆 Week Type Filters:**")
            st.checkbox(
                "Regular Season",
                value=st.session_state.get("weekly_team_include_regular_season", True),
                key="weekly_team_include_regular_season",
                help="Include regular season weeks"
            )
            playoff_col1, playoff_col2 = st.columns(2)
            with playoff_col1:
                st.checkbox(
                    "Playoffs",
                    value=st.session_state.get("weekly_team_include_playoffs", True),
                    key="weekly_team_include_playoffs",
                    help="Include weeks from the fantasy playoffs"
                )
            with playoff_col2:
                st.checkbox(
                    "Consolation",
                    value=st.session_state.get("weekly_team_include_consolation", False),
                    key="weekly_team_include_consolation",
                    help="Include weeks from the consolation bracket"
                )

            st.markdown("---")

            # Year filter
            st.markdown("**📅 Year Range:**")
            if 'year' in self.team_data_by_manager.columns:
                years = sorted(self.team_data_by_manager['year'].dropna().unique())
                if years:
                    selected_years = st.multiselect(
                        "Filter by Year",
                        options=years,
                        default=[],
                        key="weekly_team_mgr_years",
                        label_visibility="collapsed",
                        help="Leave empty to show all years"
                    )
                else:
                    selected_years = []
            else:
                selected_years = []

            st.markdown("---")

            # Manager filter
            st.markdown("**👤 Manager Filter:**")
            if 'manager' in self.team_data_by_manager.columns:
                managers = sorted(self.team_data_by_manager['manager'].dropna().unique())
                selected_managers = st.multiselect(
                    "Select Manager(s)",
                    options=managers,
                    default=[],
                    key="weekly_team_mgr_managers",
                    help="Leave empty to show all managers",
                    label_visibility="collapsed"
                )
            else:
                selected_managers = []

            st.markdown("---")

            # Active filters summary
            st.markdown("**🎯 Active Filters:**")
            active_filters = []
            if not st.session_state.get("weekly_team_include_regular_season", True):
                active_filters.append("Regular Season: Excluded")
            if not st.session_state.get("weekly_team_include_playoffs", True):
                active_filters.append("Playoffs: Excluded")
            if st.session_state.get("weekly_team_include_consolation", False):
                active_filters.append("Consolation: Included")
            if selected_years:
                if len(selected_years) <= 3:
                    active_filters.append(f"Years: {', '.join(map(str, selected_years))}")
                else:
                    active_filters.append(f"Years: {min(selected_years)}-{max(selected_years)}")
            if selected_managers:
                if len(selected_managers) <= 2:
                    active_filters.append(f"Managers: {', '.join(selected_managers)}")
                else:
                    active_filters.append(f"Managers: {len(selected_managers)} selected")

            if active_filters:
                for filter_text in active_filters:
                    st.caption(f"🏷️ {filter_text}")
            else:
                st.caption("No filters active")

            # Clear all filters button
            if active_filters:
                if st.button("🔄 Clear All Filters", key="weekly_team_mgr_clear_all", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        if key.startswith("weekly_team"):
                            del st.session_state[key]
                    st.rerun()

        with data_col:
            # Apply filters to data
            filtered_data = self.team_data_by_manager.copy()

            # Year filter
            if selected_years:
                filtered_data = filtered_data[filtered_data['year'].isin(selected_years)]

            # Manager filter
            if selected_managers:
                filtered_data = filtered_data[filtered_data['manager'].isin(selected_managers)]

            # Stat type tabs (no advanced stats for manager-only view)
            stat_tabs = st.tabs([
                TAB_LABELS['basic_stats'],
                TAB_LABELS['visualizations']
            ])

            with stat_tabs[0]:
                df = get_basic_stats_by_mgr(filtered_data)
                if df.empty:
                    render_empty_state(
                        title="No Data Found",
                        message="No basic stats available",
                        icon="🔍"
                    )
                else:
                    # Show data count
                    st.markdown(f"**Showing {len(df)} managers**")

                    # Get column config (use 'All' since this is aggregated across positions)
                    column_config = get_column_config_for_position("All")

                    # Display table with dynamic height and enhanced column config
                    table_height = min(max(300, len(df) * 35), 700)
                    st.dataframe(
                        df,
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,
                        height=table_height
                    )

                    # Export button
                    self._render_export_button(df, "weekly_manager_stats")

            with stat_tabs[1]:
                visualizer = TeamStatsVisualizer(filtered_data, filtered_data)
                visualizer.display_weekly_visualizations()

    def _display_quick_metrics(self, position: str):
        """Display quick summary metrics for the selected position."""
        try:
            # Filter data by position if not 'All'
            if position != "All":
                data = self.team_data_by_position[
                    self.team_data_by_position["fantasy_position"] == position
                ]
            else:
                data = self.team_data_by_position

            if data.empty:
                return

            # Calculate metrics
            total_points = data["points"].sum() if "points" in data.columns else 0
            avg_points = data["points"].mean() if "points" in data.columns else 0
            total_records = len(data)
            unique_managers = data["manager"].nunique() if "manager" in data.columns else 0

            # Display metrics in grid
            metrics = [
                {"label": "Total Points", "value": f"{total_points:,.1f}"},
                {"label": "Avg Points", "value": f"{avg_points:.1f}"},
                {"label": "Total Records", "value": f"{total_records:,}"},
                {"label": "Managers", "value": f"{unique_managers}"},
            ]

            render_metric_grid(metrics, columns=4)

        except Exception as e:
            # Silently fail if metrics can't be calculated
            pass

    def _display_quick_metrics_for_data(self, data: pd.DataFrame, position: str):
        """Display quick summary metrics for already-filtered data."""
        try:
            if data.empty:
                return

            # Calculate metrics from the filtered data
            total_points = data["points"].sum() if "points" in data.columns else 0
            avg_points = data["points"].mean() if "points" in data.columns else 0
            total_records = len(data)
            unique_managers = data["manager"].nunique() if "manager" in data.columns else 0

            # Display metrics in grid
            metrics = [
                {"label": "Total Points", "value": f"{total_points:,.1f}"},
                {"label": "Avg Points", "value": f"{avg_points:.1f}"},
                {"label": "Total Records", "value": f"{total_records:,}"},
                {"label": "Managers", "value": f"{unique_managers}"},
            ]

            render_metric_grid(metrics, columns=4)

        except Exception as e:
            # Silently fail if metrics can't be calculated
            pass

    def _render_export_button(self, df: pd.DataFrame, filename_prefix: str):
        """Render CSV export button for a dataframe."""
        if df.empty:
            return

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            key=f"{filename_prefix}_download"
        )
