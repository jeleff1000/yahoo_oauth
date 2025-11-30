"""
Enhanced filter UI components with better UX and performance.
"""
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional


class QuickFilters:
    """
    Quick filter chips that provide instant visual feedback without full page reloads.
    Uses session state efficiently to minimize reruns.
    """

    def __init__(self, key_prefix: str):
        self.key_prefix = key_prefix
        self.state_key = f"{key_prefix}_quick_filters"

        # Initialize quick filter state
        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = {
                'active_position': None,
                'active_year': None,
                'active_week': None,
            }

    def display_position_chips(self, available_positions: List[str] = None):
        """
        Display position filter as a dropdown selector.
        Cleaner UI and works better on mobile.
        """
        if available_positions is None:
            available_positions = ["All", "QB", "RB", "WR", "TE", "K", "DEF"]

        st.markdown("**🏈 Position Filter:**")

        # Use selectbox - let it manage its own state entirely
        selected = st.selectbox(
            "Position",
            options=available_positions,
            index=0,  # Default to "All"
            key=f"{self.key_prefix}_pos_select",
            label_visibility="collapsed"
        )

        # Return None if "All" is selected, otherwise return the position
        return None if selected == 'All' else selected

    def display_year_slider(self, min_year: int = 1999, max_year: int = None):
        """
        Year range selector with number input boxes (plus/minus buttons).
        Defaults to ALL years instead of last 3.
        No fragment decorator to avoid nesting issues.
        """
        if max_year is None:
            max_year = datetime.now().year

        st.markdown("**📅 Year Range:**")

        col1, col2 = st.columns(2)

        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=min_year,
                max_value=max_year,
                value=min_year,
                step=1,
                key=f"{self.key_prefix}_start_year_widget"
            )

        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=int(start_year),
                max_value=max_year,
                value=max_year,
                step=1,
                key=f"{self.key_prefix}_end_year_widget"
            )

        return list(range(int(start_year), int(end_year) + 1))

    def display_week_selector(self):
        """
        Week selector with preset options (All, Regular Season, Playoffs).
        No fragment decorator to avoid nesting issues.
        """
        st.markdown("**📆 Weeks:**")

        col1, col2, col3, col4 = st.columns(4)

        preset = None
        with col1:
            if st.button("All Weeks", key=f"{self.key_prefix}_weeks_all", use_container_width=True):
                preset = "all"
        with col2:
            if st.button("Regular (1-14)", key=f"{self.key_prefix}_weeks_regular", use_container_width=True):
                preset = "regular"
        with col3:
            if st.button("Playoffs (15+)", key=f"{self.key_prefix}_weeks_playoffs", use_container_width=True):
                preset = "playoffs"
        with col4:
            if st.button("Custom", key=f"{self.key_prefix}_weeks_custom", use_container_width=True):
                preset = "custom"

        if preset == "all":
            return []  # Empty list means no filter
        elif preset == "regular":
            return list(range(1, 15))
        elif preset == "playoffs":
            return list(range(15, 22))
        elif preset == "custom":
            # Show multiselect for custom selection
            return st.multiselect(
                "Select specific weeks",
                list(range(1, 22)),
                key=f"{self.key_prefix}_weeks_multiselect"
            )

        return []  # Default to all

    def display_search_bar(self, placeholder: str = "Search players..."):
        """
        Enhanced search bar with clear button and search tips.
        No fragment decorator to avoid nesting issues.
        """
        search_key = f"{self.key_prefix}_search"

        col1, col2 = st.columns([4, 1])

        with col1:
            search_value = st.text_input(
                "Player Search",
                placeholder=placeholder,
                key=search_key,
                label_visibility="collapsed"
            )

        with col2:
            if st.button("❌ Clear", key=f"{self.key_prefix}_clear_search"):
                # Delete the key from session state (can't modify after widget creation)
                if search_key in st.session_state:
                    del st.session_state[search_key]
                st.rerun()

        # Show search tips if empty
        if not search_value:
            st.caption("💡 Tip: Search by player name (e.g., 'Mahomes')")

        return search_value

    def display_advanced_filters_toggle(self):
        """
        Toggle for advanced filters - keeps UI clean by default.
        No fragment decorator to avoid nesting issues.
        """
        return st.toggle(
            "🔧 Show Advanced Filters",
            key=f"{self.key_prefix}_show_advanced",
            help="Show additional filtering options"
        )


class SmartFilterPanel:
    """
    Combined filter panel with smart defaults and better organization.
    """

    def __init__(self, key_prefix: str, data_loader):
        self.key_prefix = key_prefix
        self.data_loader = data_loader
        self.quick_filters = QuickFilters(key_prefix)

    def display_filters(self, show_advanced: bool = False) -> tuple[Dict[str, Any], str]:
        """
        Display organized filter panel with quick filters prominently.
        Returns tuple of (filter dictionary, active_position string).

        The active_position helps with position-specific column selection.

        Note: No @st.fragment decorator to avoid nesting issues with parent fragments.
        """
        filters = {}
        active_position = None

        # === QUICK FILTERS (Always Visible) ===
        st.markdown("### ⚡ Quick Filters")

        # Player search
        player_query = self.quick_filters.display_search_bar()
        if player_query:
            filters['player_query'] = player_query

        st.markdown("---")

        # === MANAGER SEARCH (Prominent, always visible) ===
        st.markdown("**👤 Manager Filter:**")
        manager_values = st.multiselect(
            "Select Manager(s)",
            options=self._get_managers(),
            key=f"{self.key_prefix}_manager_quick",
            help="Filter by fantasy team manager"
        )
        if manager_values:
            filters['manager'] = manager_values

        st.markdown("---")

        # Position chips - capture the active position
        active_position = self.quick_filters.display_position_chips()
        if active_position and active_position != "All":
            filters['nfl_position'] = [active_position]  # Filter on 'nfl_position' column

        st.markdown("---")

        # Year range slider
        year_range = self.quick_filters.display_year_slider()
        if year_range:
            filters['year'] = year_range

        st.markdown("---")

        # === ROSTER STATUS FILTERS (Prominent, always visible) ===
        st.markdown("**👥 Roster Status:**")
        col1, col2 = st.columns(2)
        with col1:
            rostered_only = st.checkbox(
                "✅ Rostered Players Only",
                value=False,
                key=f"{self.key_prefix}_rostered_quick",
                help="Show only players that were on a roster"
            )
            if rostered_only:
                filters['rostered_only'] = True

        with col2:
            started_only = st.checkbox(
                "🎯 Started Players Only",
                value=False,
                key=f"{self.key_prefix}_started_quick",
                help="Show only players that were in starting lineup (not benched)"
            )
            if started_only:
                filters['started_only'] = True

        # Postseason filter
        include_postseason = st.checkbox(
            "🏆 Include NFL Postseason",
            value=True,
            key=f"{self.key_prefix}_include_postseason",
            help="Include games from NFL playoffs (POST season_type)"
        )
        if not include_postseason:
            filters['exclude_postseason'] = True

        st.markdown("---")

        # === ADVANCED FILTERS (Collapsible) ===
        if show_advanced or self.quick_filters.display_advanced_filters_toggle():
            st.markdown("### 🔧 Advanced Filters")

            with st.expander("Teams & Positions", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    nfl_position_values = st.multiselect(
                        "NFL Position",
                        options=self._get_nfl_positions(),
                        key=f"{self.key_prefix}_nfl_pos_adv"
                    )
                    if nfl_position_values:
                        filters['nfl_position'] = nfl_position_values

                    nfl_team_values = st.multiselect(
                        "NFL Team",
                        options=self._get_nfl_teams(),
                        key=f"{self.key_prefix}_nfl_team_adv"
                    )
                    if nfl_team_values:
                        filters['nfl_team'] = nfl_team_values

                with col2:
                    fantasy_position_values = st.multiselect(
                        "Fantasy Position (Multi)",
                        options=["QB", "RB", "WR", "TE", "W/R/T", "K", "DEF", "BN"],
                        key=f"{self.key_prefix}_fantasy_pos_multi_adv"
                    )
                    if fantasy_position_values:
                        filters['fantasy_position'] = fantasy_position_values
                        # If multiple positions selected, no single active position
                        if len(fantasy_position_values) > 1:
                            active_position = None

                    opponent_nfl_team_values = st.multiselect(
                        "Opponent NFL Team",
                        options=self._get_nfl_teams(),
                        key=f"{self.key_prefix}_opp_nfl_team_adv"
                    )
                    if opponent_nfl_team_values:
                        filters['opponent_nfl_team'] = opponent_nfl_team_values

            with st.expander("Week & Status", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    week_values = st.multiselect(
                        "Specific Weeks",
                        options=list(range(1, 22)),
                        key=f"{self.key_prefix}_week_adv"
                    )
                    if week_values:
                        filters['week'] = week_values

        # === FILTER SUMMARY ===
        self._display_active_filters_summary(filters)

        return filters, active_position

    def _display_active_filters_summary(self, filters: Dict[str, Any]):
        """
        Display a summary of active filters with quick clear buttons.
        Removed @st.fragment to avoid React key conflicts.
        """
        if not filters or all(not v for v in filters.values()):
            return

        st.markdown("---")
        st.markdown("**🎯 Active Filters:**")

        # Create chips for each active filter
        chips = []
        if filters.get('player_query'):
            chips.append(f"Player: '{filters['player_query']}'")
        if filters.get('nfl_position'):
            chips.append(f"Position: {', '.join(filters['nfl_position'])}")
        if filters.get('fantasy_position'):
            chips.append(f"Fantasy Position: {', '.join(filters['fantasy_position'])}")
        if filters.get('year'):
            years = filters['year']
            if len(years) > 3:
                chips.append(f"Years: {min(years)}-{max(years)}")
            else:
                chips.append(f"Years: {', '.join(map(str, years))}")
        if filters.get('manager'):
            chips.append(f"Manager: {', '.join(filters['manager'][:2])}...")
        if filters.get('rostered_only'):
            chips.append("Rostered Only")
        if filters.get('started_only'):
            chips.append("Started Only")

        if chips:
            # Display chips as text instead of dynamic columns to avoid React key issues
            chips_text = " | ".join([f"🏷️ {chip}" for chip in chips])
            st.caption(chips_text)

        # Clear button on its own line
        if st.button("🔄 Clear All Filters", key=f"{self.key_prefix}_clear_all", use_container_width=True):
            # Clear session state for this filter panel
            for key in list(st.session_state.keys()):
                if key.startswith(self.key_prefix):
                    del st.session_state[key]
            st.rerun()

    def _get_managers(self) -> List[str]:
        """Get list of managers from data loader."""
        try:
            # Import here to avoid circular dependency
            from md.data_access import list_managers
            # Get all managers across all years for the filter
            managers = list_managers(year=None)
            return sorted(managers) if managers else []
        except Exception as e:
            st.warning(f"Could not load managers: {e}")
            return []

    def _get_nfl_positions(self) -> List[str]:
        """Get NFL positions in standard order."""
        return ["QB", "RB", "WR", "TE", "K", "DEF", "C", "CB", "DB", "DE", "DL",
                "DT", "FB", "FS", "G", "ILB", "LB", "LS", "MLB", "NT", "OL", "OLB",
                "OT", "P", "S", "SAF"]

    def _get_nfl_teams(self) -> List[str]:
        """Get NFL teams."""
        return ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL",
                "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV",
                "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",
                "TB", "TEN", "WAS"]
