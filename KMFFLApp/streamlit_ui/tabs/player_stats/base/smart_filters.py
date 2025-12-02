"""
Compact filter UI components optimized for mobile and desktop.
"""
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any


class SmartFilterPanel:
    """Compact filter panel with smart defaults."""

    def __init__(self, key_prefix: str, data_loader=None):
        self.key_prefix = key_prefix
        self.data_loader = data_loader

    def display_filters(self, show_advanced: bool = False) -> tuple[Dict[str, Any], str]:
        """Display compact filter panel. Returns (filters dict, active_position)."""
        filters = {}
        active_position = None

        # Compact CSS
        st.markdown("""
        <style>
        [data-testid="stExpander"] .stTextInput,
        [data-testid="stExpander"] .stMultiSelect,
        [data-testid="stExpander"] .stSelectbox,
        [data-testid="stExpander"] .stNumberInput {
            margin-bottom: 0.25rem !important;
        }
        [data-testid="stExpander"] [data-testid="column"] {
            padding: 0 0.25rem !important;
        }
        [data-testid="stExpander"] label {
            font-size: 0.85rem !important;
            margin-bottom: 0.1rem !important;
        }
        [data-testid="stExpander"] .stCheckbox {
            margin-bottom: 0 !important;
        }
        [data-testid="stExpander"] .stCheckbox label span {
            font-size: 0.8rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Row 1: Search + Position
        c1, c2 = st.columns([2, 1])
        with c1:
            player_query = st.text_input(
                "Player",
                placeholder="Search player...",
                key=f"{self.key_prefix}_player",
                label_visibility="collapsed"
            )
            if player_query:
                filters['player_query'] = player_query

        with c2:
            positions = ["All", "QB", "RB", "WR", "TE", "K", "DEF"]
            pos = st.selectbox("Position", positions, key=f"{self.key_prefix}_pos", label_visibility="collapsed")
            if pos != "All":
                active_position = pos
                filters['nfl_position'] = [pos]

        # Row 2: Manager + Year range
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            managers = self._get_managers()
            mgr = st.multiselect("Manager", managers, key=f"{self.key_prefix}_mgr", placeholder="All", label_visibility="collapsed")
            if mgr:
                filters['manager'] = mgr

        current_year = datetime.now().year
        with c2:
            start_yr = st.number_input("From", min_value=1999, max_value=current_year, value=1999, key=f"{self.key_prefix}_yr1", label_visibility="collapsed")
        with c3:
            end_yr = st.number_input("To", min_value=int(start_yr), max_value=current_year, value=current_year, key=f"{self.key_prefix}_yr2", label_visibility="collapsed")

        if start_yr or end_yr:
            filters['year'] = list(range(int(start_yr), int(end_yr) + 1))

        # Row 3: Toggles (compact inline)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.checkbox("Rostered", key=f"{self.key_prefix}_rost"):
                filters['rostered_only'] = True
        with c2:
            if st.checkbox("Started", key=f"{self.key_prefix}_start"):
                filters['started_only'] = True
        with c3:
            if not st.checkbox("Postseason", value=True, key=f"{self.key_prefix}_post"):
                filters['exclude_postseason'] = True

        # Row 4: Advanced (optional, collapsed by default)
        with st.expander("More filters", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                nfl_teams = self._get_nfl_teams()
                team = st.multiselect("NFL Team", nfl_teams, key=f"{self.key_prefix}_team", placeholder="All")
                if team:
                    filters['nfl_team'] = team

            with c2:
                opp_team = st.multiselect("Opponent", nfl_teams, key=f"{self.key_prefix}_opp", placeholder="All")
                if opp_team:
                    filters['opponent_nfl_team'] = opp_team

            weeks = st.multiselect("Weeks", list(range(1, 22)), key=f"{self.key_prefix}_wk", placeholder="All weeks")
            if weeks:
                filters['week'] = weeks

        return filters, active_position

    def _get_managers(self) -> List[str]:
        """Get list of managers."""
        try:
            from md.data_access import list_managers
            managers = list_managers(year=None)
            return sorted(managers) if managers else []
        except:
            return []

    def _get_nfl_teams(self) -> List[str]:
        """Get NFL teams."""
        return ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL",
                "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV",
                "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",
                "TB", "TEN", "WAS"]


# Keep QuickFilters for backwards compatibility but simplified
class QuickFilters:
    """Simplified quick filters (kept for backwards compatibility)."""

    def __init__(self, key_prefix: str):
        self.key_prefix = key_prefix

    def display_position_chips(self, available_positions: List[str] = None):
        if available_positions is None:
            available_positions = ["All", "QB", "RB", "WR", "TE", "K", "DEF"]
        selected = st.selectbox("Position", options=available_positions, index=0, key=f"{self.key_prefix}_pos_select", label_visibility="collapsed")
        return None if selected == 'All' else selected

    def display_year_slider(self, min_year: int = 1999, max_year: int = None):
        if max_year is None:
            max_year = datetime.now().year
        c1, c2 = st.columns(2)
        with c1:
            start = st.number_input("From", min_value=min_year, max_value=max_year, value=min_year, key=f"{self.key_prefix}_start_yr")
        with c2:
            end = st.number_input("To", min_value=int(start), max_value=max_year, value=max_year, key=f"{self.key_prefix}_end_yr")
        return list(range(int(start), int(end) + 1))

    def display_search_bar(self, placeholder: str = "Search..."):
        return st.text_input("Search", placeholder=placeholder, key=f"{self.key_prefix}_search", label_visibility="collapsed")

    def display_advanced_filters_toggle(self):
        return st.toggle("More", key=f"{self.key_prefix}_adv")
