"""
Reusable filter components for team stats.

Provides:
- Smart filter panel with quick and advanced filters
- Active filter display with chips
- Filter application and validation
- Position, year, week, manager filtering
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from .constants import POSITION_OPTIONS
from .theme import render_filter_count


class TeamStatsFilterPanel:
    """
    Reusable filter panel for team stats views.

    Provides both quick filters (always visible) and advanced filters
    (collapsible) with support for:
    - Position filtering
    - Year range selection
    - Week range selection
    - Manager selection
    - Active filter display
    """

    def __init__(self, prefix: str, available_data: pd.DataFrame):
        """
        Initialize filter panel.

        Args:
            prefix: Unique prefix for widget keys
            available_data: DataFrame to extract filter options from
        """
        self.prefix = prefix
        self.data = available_data

        # Extract available options from data
        self.available_years = self._get_available_years()
        self.available_managers = self._get_available_managers()
        self.available_weeks = self._get_available_weeks()
        self.available_positions = self._get_available_positions()

    def _get_available_years(self) -> List[int]:
        """Get list of available years from data."""
        if 'year' in self.data.columns:
            years = self.data['year'].dropna().unique()
            return sorted([int(y) for y in years if pd.notna(y)])
        return []

    def _get_available_managers(self) -> List[str]:
        """Get list of available managers from data."""
        if 'manager' in self.data.columns:
            managers = self.data['manager'].dropna().unique()
            return sorted([str(m) for m in managers if pd.notna(m)])
        return []

    def _get_available_weeks(self) -> List[int]:
        """Get list of available weeks from data."""
        if 'week' in self.data.columns:
            weeks = self.data['week'].dropna().unique()
            return sorted([int(w) for w in weeks if pd.notna(w)])
        return []

    def _get_available_positions(self) -> List[str]:
        """Get list of available positions from data."""
        if 'fantasy_position' in self.data.columns:
            positions = self.data['fantasy_position'].dropna().unique()
            available = sorted([str(p) for p in positions if pd.notna(p)])
            return ['All'] + available
        elif 'Position' in self.data.columns:
            positions = self.data['Position'].dropna().unique()
            available = sorted([str(p) for p in positions if pd.notna(p)])
            return ['All'] + available
        return POSITION_OPTIONS

    def display_quick_filters(self) -> Dict[str, Any]:
        """
        Display quick filters (always visible).

        Returns:
            Dictionary of filter selections
        """
        filters = {}

        st.markdown("### 🎛️ Quick Filters")

        # Position filter
        if self.available_positions:
            filters['position'] = st.selectbox(
                "Position",
                options=self.available_positions,
                index=0,
                key=f"{self.prefix}_position"
            )

        # Year range (for weekly/season views)
        if self.available_years:
            col1, col2 = st.columns(2)
            with col1:
                filters['year_start'] = st.selectbox(
                    "From Year",
                    options=self.available_years,
                    index=0,
                    key=f"{self.prefix}_year_start"
                )
            with col2:
                filters['year_end'] = st.selectbox(
                    "To Year",
                    options=self.available_years,
                    index=len(self.available_years) - 1,
                    key=f"{self.prefix}_year_end"
                )

        # Manager filter (multiselect)
        if self.available_managers:
            filters['managers'] = st.multiselect(
                "Managers",
                options=self.available_managers,
                default=[],
                key=f"{self.prefix}_managers",
                help="Leave empty to show all managers"
            )

        return filters

    def display_advanced_filters(self) -> Dict[str, Any]:
        """
        Display advanced filters (collapsible).

        Returns:
            Dictionary of advanced filter selections
        """
        filters = {}

        with st.expander("⚙️ Advanced Filters", expanded=False):

            # Week range (for weekly view)
            if self.available_weeks:
                filters['week_range'] = st.slider(
                    "Week Range",
                    min_value=min(self.available_weeks),
                    max_value=max(self.available_weeks),
                    value=(min(self.available_weeks), max(self.available_weeks)),
                    key=f"{self.prefix}_week_range"
                )

            # Minimum points threshold
            filters['min_points'] = st.number_input(
                "Minimum Points",
                min_value=0.0,
                max_value=200.0,
                value=0.0,
                step=5.0,
                key=f"{self.prefix}_min_points",
                help="Filter out low-scoring performances"
            )

            # Include playoffs checkbox
            filters['include_playoffs'] = st.checkbox(
                "Include Playoffs",
                value=True,
                key=f"{self.prefix}_include_playoffs"
            )

            # Include bench checkbox (if applicable)
            filters['include_bench'] = st.checkbox(
                "Include Bench Players",
                value=False,
                key=f"{self.prefix}_include_bench",
                help="Include players who were benched"
            )

        return filters

    def apply_filters(
        self,
        df: pd.DataFrame,
        quick_filters: Dict[str, Any],
        advanced_filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply filters to dataframe.

        Args:
            df: Input dataframe
            quick_filters: Quick filter selections
            advanced_filters: Optional advanced filter selections

        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()

        # Position filter
        if 'position' in quick_filters and quick_filters['position'] != 'All':
            pos_col = 'fantasy_position' if 'fantasy_position' in filtered_df.columns else 'Position'
            if pos_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[pos_col] == quick_filters['position']]

        # Year range filter
        if 'year_start' in quick_filters and 'year_end' in quick_filters:
            year_col = 'year' if 'year' in filtered_df.columns else 'Year'
            if year_col in filtered_df.columns:
                filtered_df[year_col] = pd.to_numeric(filtered_df[year_col], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df[year_col] >= quick_filters['year_start']) &
                    (filtered_df[year_col] <= quick_filters['year_end'])
                ]

        # Manager filter
        if 'managers' in quick_filters and quick_filters['managers']:
            manager_col = 'manager' if 'manager' in filtered_df.columns else 'Manager'
            if manager_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[manager_col].isin(quick_filters['managers'])]

        # Apply advanced filters if provided
        if advanced_filters:

            # Week range filter
            if 'week_range' in advanced_filters:
                week_col = 'week' if 'week' in filtered_df.columns else 'Week'
                if week_col in filtered_df.columns:
                    filtered_df[week_col] = pd.to_numeric(filtered_df[week_col], errors='coerce')
                    filtered_df = filtered_df[
                        (filtered_df[week_col] >= advanced_filters['week_range'][0]) &
                        (filtered_df[week_col] <= advanced_filters['week_range'][1])
                    ]

            # Minimum points filter
            if 'min_points' in advanced_filters and advanced_filters['min_points'] > 0:
                points_col = 'points' if 'points' in filtered_df.columns else 'Points'
                if points_col in filtered_df.columns:
                    filtered_df[points_col] = pd.to_numeric(filtered_df[points_col], errors='coerce')
                    filtered_df = filtered_df[filtered_df[points_col] >= advanced_filters['min_points']]

            # Playoff filter
            if not advanced_filters.get('include_playoffs', True):
                if 'game_type' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['game_type'] != 'playoff']
                elif 'week' in filtered_df.columns:
                    # Assume weeks > 14 are playoffs (adjust as needed)
                    filtered_df = filtered_df[pd.to_numeric(filtered_df['week'], errors='coerce') <= 14]

            # Bench filter
            if not advanced_filters.get('include_bench', False):
                if 'roster_position' in filtered_df.columns:
                    filtered_df = filtered_df[~filtered_df['roster_position'].isin(['BN', 'IR'])]

        return filtered_df

    def display_active_filters(
        self,
        quick_filters: Dict[str, Any],
        advanced_filters: Optional[Dict[str, Any]] = None
    ):
        """
        Display active filters as chips with clear option.

        Args:
            quick_filters: Quick filter selections
            advanced_filters: Optional advanced filter selections
        """
        active_filters = []

        # Collect active filters
        if 'position' in quick_filters and quick_filters['position'] != 'All':
            active_filters.append(f"Position: {quick_filters['position']}")

        if 'year_start' in quick_filters and 'year_end' in quick_filters:
            if quick_filters['year_start'] == quick_filters['year_end']:
                active_filters.append(f"Year: {quick_filters['year_start']}")
            else:
                active_filters.append(f"Years: {quick_filters['year_start']}-{quick_filters['year_end']}")

        if 'managers' in quick_filters and quick_filters['managers']:
            manager_str = ", ".join(quick_filters['managers'][:2])
            if len(quick_filters['managers']) > 2:
                manager_str += f" +{len(quick_filters['managers']) - 2} more"
            active_filters.append(f"Managers: {manager_str}")

        if advanced_filters:
            if 'week_range' in advanced_filters:
                week_start, week_end = advanced_filters['week_range']
                if week_start != min(self.available_weeks) or week_end != max(self.available_weeks):
                    active_filters.append(f"Weeks: {week_start}-{week_end}")

            if 'min_points' in advanced_filters and advanced_filters['min_points'] > 0:
                active_filters.append(f"Min Points: {advanced_filters['min_points']:.0f}")

            if not advanced_filters.get('include_playoffs', True):
                active_filters.append("Regular Season Only")

            if not advanced_filters.get('include_bench', False):
                active_filters.append("Starters Only")

        # Display active filters
        if active_filters:
            st.markdown("**Active Filters:**")
            filter_html = " ".join([
                f'<span style="background: #3b82f6; color: white; padding: 0.25rem 0.75rem; '
                f'border-radius: 1rem; font-size: 0.85rem; margin: 0.25rem; display: inline-block;">'
                f'{f}</span>'
                for f in active_filters
            ])
            st.markdown(filter_html, unsafe_allow_html=True)

            if st.button("🗑️ Clear All Filters", key=f"{self.prefix}_clear_filters"):
                st.rerun()
        else:
            st.info("No active filters. Showing all data.")


def render_simple_position_filter(prefix: str, default: str = "All") -> str:
    """
    Render a simple position dropdown filter.

    Args:
        prefix: Unique prefix for widget key
        default: Default position selection

    Returns:
        Selected position
    """
    return st.selectbox(
        "Filter by Position",
        options=POSITION_OPTIONS,
        index=POSITION_OPTIONS.index(default) if default in POSITION_OPTIONS else 0,
        key=f"{prefix}_simple_position"
    )


def render_year_selector(
    prefix: str,
    available_years: List[int],
    allow_multiple: bool = False
) -> Any:
    """
    Render a year selector (single or multi-select).

    Args:
        prefix: Unique prefix for widget key
        available_years: List of available years
        allow_multiple: If True, use multiselect; else use selectbox

    Returns:
        Selected year(s)
    """
    if not available_years:
        st.warning("No years available in data")
        return [] if allow_multiple else None

    if allow_multiple:
        return st.multiselect(
            "Select Year(s)",
            options=available_years,
            default=[max(available_years)],
            key=f"{prefix}_year_multi"
        )
    else:
        return st.selectbox(
            "Select Year",
            options=available_years,
            index=len(available_years) - 1,  # Default to most recent
            key=f"{prefix}_year_single"
        )


def render_week_slider(
    prefix: str,
    min_week: int = 1,
    max_week: int = 17
) -> Tuple[int, int]:
    """
    Render a week range slider.

    Args:
        prefix: Unique prefix for widget key
        min_week: Minimum week value
        max_week: Maximum week value

    Returns:
        Tuple of (start_week, end_week)
    """
    return st.slider(
        "Week Range",
        min_value=min_week,
        max_value=max_week,
        value=(min_week, max_week),
        key=f"{prefix}_week_slider"
    )


def render_manager_selector(
    prefix: str,
    available_managers: List[str],
    allow_multiple: bool = True,
    default_all: bool = False
) -> List[str]:
    """
    Render a manager selector.

    Args:
        prefix: Unique prefix for widget key
        available_managers: List of available managers
        allow_multiple: If True, use multiselect; else use selectbox
        default_all: If True, default to all managers

    Returns:
        List of selected managers
    """
    if not available_managers:
        st.warning("No managers available in data")
        return []

    if allow_multiple:
        default = available_managers if default_all else []
        selected = st.multiselect(
            "Select Manager(s)",
            options=available_managers,
            default=default,
            key=f"{prefix}_manager_multi",
            help="Leave empty to show all managers"
        )
        return selected if selected else available_managers
    else:
        selected = st.selectbox(
            "Select Manager",
            options=available_managers,
            key=f"{prefix}_manager_single"
        )
        return [selected] if selected else []


def has_active_filters(filters: Dict[str, Any]) -> bool:
    """
    Check if any filters are active (non-default).

    Args:
        filters: Dictionary of filter selections

    Returns:
        True if any filter is active
    """
    # Position filter
    if filters.get('position', 'All') != 'All':
        return True

    # Manager filter
    if filters.get('managers', []):
        return True

    # Min points filter
    if filters.get('min_points', 0) > 0:
        return True

    # Playoff filter
    if not filters.get('include_playoffs', True):
        return True

    # Bench filter
    if not filters.get('include_bench', False):
        return True

    return False
