"""
Optimized Career Player Stats Overview matching Weekly tab UI/UX.
"""
import streamlit as st
from ..shared.modern_styles import apply_modern_styles

from md.tab_data_access.players import load_career_player_data as load_players_career_data
from .career_player_subprocesses.career_player_basic_stats import get_basic_stats
from .career_player_subprocesses.career_player_advanced_stats import get_advanced_stats
from .career_player_subprocesses.career_player_matchup_stats import CombinedMatchupStatsViewer
from .base.table_display import EnhancedTableDisplay
from .base.smart_filters import SmartFilterPanel


class OptimizedCareerPlayerViewer:
    """
    Optimized career player data viewer matching Weekly tab UI/UX with:
    - Smart filter panel with search bars
    - Enhanced table display with load more
    - Column selection
    - Same layout as Weekly tab
    """

    def __init__(self):
        # Enhanced display managers for each tab
        self.display_basic = EnhancedTableDisplay("career_basic")
        self.display_advanced = EnhancedTableDisplay("career_advanced")
        self.display_matchup = EnhancedTableDisplay("career_matchup")

    def get_sort_columns_for_viewer(self, viewer_type: str, active_position: str = None) -> list:
        """Get relevant sort columns based on viewer type and active position."""
        # Keep career-specific common columns but mirror weekly per-position columns
        common_cols = ["points", "player", "ppg_all_time", "games_started", "fantasy_games"]

        if viewer_type == "basic":
            base_cols = common_cols.copy()
            base_cols.extend(["manager", "nfl_team"])

            if active_position == "QB":
                base_cols.extend(["pass_yds", "pass_td", "passing_interceptions", "rush_yds", "rush_td"])
            elif active_position == "RB":
                base_cols.extend(["rush_yds", "rush_td", "rec", "rec_yds", "rec_td"])
            elif active_position in ["WR", "TE"]:
                base_cols.extend(["rec", "rec_yds", "rec_td", "targets"])
            elif active_position == "K":
                base_cols.extend(["fg_made", "fg_att"])
            elif active_position == "DEF":
                base_cols.extend(["def_sacks", "def_interceptions", "pts_allow"])
            else:
                base_cols.extend(["pass_yds", "rush_yds", "rec_yds", "rec"])

            return base_cols

        elif viewer_type == "advanced":
            base_cols = common_cols.copy()
            base_cols.extend(["manager", "nfl_team"])

            if active_position == "QB":
                base_cols.extend(["pass_yds", "pass_td", "passing_interceptions",
                                 "passing_air_yards", "passing_epa", "passing_cpoe",
                                 "rush_yds", "rush_td", "rushing_epa"])
            elif active_position == "RB":
                base_cols.extend(["rush_yds", "rush_td", "rushing_epa",
                                 "rec", "rec_yds", "rec_td", "receiving_epa",
                                 "target_share", "wopr"])
            elif active_position in ["WR", "TE"]:
                base_cols.extend(["rec", "rec_yds", "rec_td", "targets", "receiving_epa",
                                 "target_share", "wopr", "racr", "receiving_air_yards",
                                 "air_yards_share"])
            elif active_position == "K":
                base_cols.extend(["fg_made", "fg_att", "fg_pct"])
            elif active_position == "DEF":
                base_cols.extend(["def_sacks", "def_interceptions", "pts_allow", "def_td"])
            else:
                base_cols.extend(["pass_yds", "passing_epa", "rush_yds", "rushing_epa",
                                 "rec_yds", "rec", "receiving_epa"])

            return base_cols

        elif viewer_type == "matchup":
            matchup_cols = ["points", "player", "manager", "opponent",
                           "team_points", "opponent_points", "nfl_team"]
            return matchup_cols

        return common_cols

    @st.fragment
    def display_sort_options(self, viewer_type: str, active_position: str = None):
        """Display compact sort options."""
        available_cols = self.get_sort_columns_for_viewer(viewer_type, active_position)

        sort_col_key = f"career_{viewer_type}_sort_col"
        sort_dir_key = f"career_{viewer_type}_sort_dir"

        # Initialize defaults - points only for career stats
        if sort_col_key not in st.session_state:
            st.session_state[sort_col_key] = "points"
        if sort_dir_key not in st.session_state:
            st.session_state[sort_dir_key] = "DESC"

        col1, col2 = st.columns([3, 1])

        with col1:
            default_sort_col = st.session_state.get(sort_col_key, "points")
            sort_col = st.selectbox(
                "Sort by",
                available_cols,
                index=available_cols.index(default_sort_col) if default_sort_col in available_cols else 0,
                key=f"career_{viewer_type}_sort_select"
            )

        with col2:
            default_sort_dir = st.session_state.get(sort_dir_key, "DESC")
            sort_dir = st.selectbox(
                "Order",
                ["DESC", "ASC"],
                index=0 if default_sort_dir == "DESC" else 1,
                key=f"career_{viewer_type}_dir_select"
            )

        # Only update session state if values changed (prevent infinite loop)
        if st.session_state.get(sort_col_key) != sort_col:
            st.session_state[sort_col_key] = sort_col
        if st.session_state.get(sort_dir_key) != sort_dir:
            st.session_state[sort_dir_key] = sort_dir

        return sort_col, sort_dir

    def has_active_filters(self, filters):
        """Check if any filters are active."""
        return any([
            filters.get('player_query'),
            filters.get('manager_query'),
            filters.get('manager'),
            filters.get('position'),
            filters.get('nfl_position'),
            filters.get('nfl_team'),
            filters.get('opponent'),
            filters.get('opponent_nfl_team'),
            filters.get('year'),
            filters.get('rostered_only'),
            filters.get('started_only'),
        ])

    @st.fragment
    def display(self):
        apply_modern_styles()

        # Compact CSS for player stats page
        st.markdown("""
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
        """, unsafe_allow_html=True)

        # View mode label above tabs - subtle caption
        st.markdown('<p class="view-mode-label">View Mode</p>', unsafe_allow_html=True)

        # Create tabs
        tabs = st.tabs(["Basic Stats", "Advanced Stats", "Matchup Stats", "Optimal Lineup"])

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

        # ==================== OPTIMAL LINEUP TAB ====================
        with tabs[3]:
            from .career_player_subprocesses.optimal_lineup_career import OptimalLineupCareerViewer
            OptimalLineupCareerViewer().display()

    @st.fragment
    def _display_basic_stats_tab(self):
        """Display Basic Stats tab with optimized layout."""
        # Collapsible filters
        with st.expander("🔎 Filters", expanded=False):
            filter_panel = SmartFilterPanel("career_basic", self)
            filters, active_position = filter_panel.display_filters()

        sort_col, sort_dir = "points", "DESC"
        with st.spinner("Loading career data..."):
            position_filter = filters.get('position') or filters.get('nfl_position')
            if isinstance(position_filter, list) and len(position_filter) == 1:
                position_filter = position_filter[0]
            elif isinstance(position_filter, list) and len(position_filter) > 1:
                position_filter = None

            career_data = load_players_career_data(
                position=position_filter,
                player_query=filters.get('player_query'),
                manager_query=filters.get('manager_query'),
                manager=filters.get('manager'),
                nfl_team=filters.get('nfl_team'),
                opponent=filters.get('opponent'),
                opponent_nfl_team=filters.get('opponent_nfl_team'),
                year=filters.get('year'),
                rostered_only=filters.get('rostered_only', False),
                started_only=filters.get('started_only', False),
                sort_column=sort_col,
                sort_direction=sort_dir
            )

            if career_data is not None and not career_data.empty:
                basic_stats_df = get_basic_stats(career_data, active_position or "All")
                self.display_basic.display_table_with_load_more(basic_stats_df, total_available=len(basic_stats_df))
            else:
                st.warning("No data available for the selected filters.")

    @st.fragment
    def _display_advanced_stats_tab(self):
        """Display Advanced Stats tab."""
        # Collapsible filters
        with st.expander("🔎 Filters", expanded=False):
            filter_panel = SmartFilterPanel("career_advanced", self)
            filters, active_position = filter_panel.display_filters()

        sort_col, sort_dir = "points", "DESC"
        with st.spinner("Loading advanced career data..."):
            position_filter = filters.get('position') or filters.get('nfl_position')
            if isinstance(position_filter, list) and len(position_filter) == 1:
                position_filter = position_filter[0]
            elif isinstance(position_filter, list) and len(position_filter) > 1:
                position_filter = None

            career_data = load_players_career_data(
                position=position_filter,
                player_query=filters.get('player_query'),
                manager_query=filters.get('manager_query'),
                manager=filters.get('manager'),
                nfl_team=filters.get('nfl_team'),
                opponent=filters.get('opponent'),
                opponent_nfl_team=filters.get('opponent_nfl_team'),
                year=filters.get('year'),
                rostered_only=filters.get('rostered_only', False),
                started_only=filters.get('started_only', False),
                exclude_postseason=filters.get('exclude_postseason', False),
                sort_column=sort_col,
                sort_direction=sort_dir
            )

            if career_data is not None and not career_data.empty:
                advanced_stats_df = get_advanced_stats(career_data, active_position or "All")
                self.display_advanced.display_table_with_load_more(advanced_stats_df, total_available=len(advanced_stats_df))
            else:
                st.warning("No data available for the selected filters.")

    @st.fragment
    def _display_matchup_stats_tab(self):
        """Display Matchup Stats tab."""
        # Collapsible filters
        with st.expander("🔎 Filters", expanded=False):
            filter_panel = SmartFilterPanel("career_matchup", self)
            filters, active_position = filter_panel.display_filters()

        sort_col, sort_dir = "points", "DESC"
        with st.spinner("Loading matchup career data..."):
            position_filter = filters.get('position') or filters.get('nfl_position')
            if isinstance(position_filter, list) and len(position_filter) == 1:
                position_filter = position_filter[0]
            elif isinstance(position_filter, list) and len(position_filter) > 1:
                position_filter = None

            career_data = load_players_career_data(
                position=position_filter,
                player_query=filters.get('player_query'),
                manager_query=filters.get('manager_query'),
                manager=filters.get('manager'),
                nfl_team=filters.get('nfl_team'),
                opponent=filters.get('opponent'),
                opponent_nfl_team=filters.get('opponent_nfl_team'),
                year=filters.get('year'),
                rostered_only=True,
                started_only=filters.get('started_only', False),
                sort_column=sort_col,
                sort_direction=sort_dir
            )

            if career_data is not None and not career_data.empty:
                st.markdown(f"**{len(career_data):,} career matchup rows**")
                matchup_viewer = CombinedMatchupStatsViewer(career_data)
                matchup_viewer.display(prefix="career_matchup")
            else:
                st.warning("No data available for the selected filters.")
