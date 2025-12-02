#!/usr/bin/env python3
"""
Optimized KMFFL Analytics App
Performance improvements:
- Smart caching with session state
- Lazy loading of tabs
- Progressive data loading
- Reusable UI components
- Performance monitoring
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the streamlit_ui directory is in the Python path
streamlit_ui_dir = Path(__file__).parent.resolve()
if str(streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(streamlit_ui_dir))

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from utils.performance import PerformanceMonitor, DataLoader, perf_monitor
from utils.ui_components import render_header, render_empty_state, render_loading_skeleton
from md.data_cache import cached_data_loader, invalidate_tab_cache, render_cache_stats

# Performance monitor
monitor = PerformanceMonitor()


def _safe_boot() -> bool:
    """Cheap health check for MD connectivity."""
    try:
        from md.data_access import run_query
        run_query("SELECT 1")
        return True
    except Exception as e:
        st.error(f"MotherDuck unavailable: {e}")
        st.info("Check Streamlit Secrets (MOTHERDUCK_TOKEN, MD_ATTACH_URL) and your shared DB path.")
        return False


def _init_session_defaults():
    """Initialize session state with smart defaults - no MD queries on bootup"""
    # Only query latest_season_and_week if not already cached
    if "year" not in st.session_state or "week" not in st.session_state:
        from md.data_access import latest_season_and_week
        try:
            y, w = latest_season_and_week()
            st.session_state["year"] = int(y) if y else 0
            st.session_state["week"] = int(w) if w else 0
        except Exception:
            st.session_state["year"] = 0
            st.session_state["week"] = 0

    # Tab tracking for lazy loading
    st.session_state.setdefault("active_main_tab", 0)
    st.session_state.setdefault("active_players_subtab", 0)
    st.session_state.setdefault("active_draft_subtab", 0)
    st.session_state.setdefault("active_extras_subtab", 0)

    # Performance settings
    st.session_state.setdefault("show_performance_stats", False)
    st.session_state.setdefault("enable_progressive_loading", True)


# ============= Lazy Tab Loaders =============

@cached_data_loader(ttl=300, spinner_text="Loading homepage...")
def load_homepage_tab():
    """
    Optimized homepage data loader.

    KEY OPTIMIZATIONS:
    1. Only loads 17 columns from matchup table instead of all 276 (85% reduction!)
    2. Combines 5 separate summary queries into 1 combined query
    3. Removes redundant player_two_week_slice load (loaded on-demand by recaps tab)
    4. Removes massive load_simulations_data(include_all_years=True) call

    Result: ~90% reduction in data transfer and memory usage for homepage!
    """
    with monitor.time_operation("load_homepage"):
        from md.tab_data_access.homepage import load_optimized_homepage_data

        # Single optimized call that loads only needed columns
        data = load_optimized_homepage_data()

        if "error" in data:
            st.error(f"Failed to load homepage data: {data['error']}")
            return {"summary": {}, "Matchup Data": None}

        return data


@cached_data_loader(ttl=300, spinner_text="Loading managers data...")
def load_managers_tab():
    """
    Optimized managers data loader.

    KEY OPTIMIZATIONS:
    1. Only loads ~60 columns from matchup table instead of all 276 (78% reduction!)
    2. Uses ACTUAL column names (manager_proj_score NOT team_projected_points)
    3. Keeps summary queries optimized (already using aggregations)

    Result: ~78% reduction in data transfer and memory usage for managers tab!
    """
    with monitor.time_operation("load_managers"):
        from md.tab_data_access.managers import load_optimized_managers_data

        # Single optimized call that loads only needed columns
        data = load_optimized_managers_data()

        if "error" in data:
            st.error(f"Failed to load managers data: {data['error']}")
            return {"recent": None, "summary": None, "h2h": None}

        return data


@cached_data_loader(ttl=300, spinner_text="Loading draft data...")
def load_draft_tab():
    """Lazy load draft data with optimized column selection"""
    with monitor.time_operation("load_draft"):
        from md.tab_data_access.draft import load_optimized_draft_data
        return load_optimized_draft_data()


@cached_data_loader(ttl=300, spinner_text="Loading transactions...")
def load_transactions_tab():
    """Lazy load transactions data with optimized column selection"""
    with monitor.time_operation("load_transactions"):
        from md.tab_data_access.transactions import load_optimized_transactions_data
        return load_optimized_transactions_data()


@cached_data_loader(ttl=300, spinner_text="Loading simulations...")
def load_simulations_tab():
    """Lazy load simulations data with optimized data access"""
    with monitor.time_operation("load_simulations"):
        from md.tab_data_access.simulations import load_optimized_simulations_data
        return load_optimized_simulations_data()


# ============= Tab Renderers =============

@st.fragment
def render_home_tab():
    """Render homepage with lazy loading"""
    data = load_homepage_tab()
    from tabs.homepage.homepage_overview import display_homepage_overview
    display_homepage_overview(data)


@st.fragment
def render_managers_tab():
    """Render managers tab with lazy loading"""
    data = load_managers_tab()
    if "error" not in data:
        from tabs.matchups.matchup_overview import display_matchup_overview
        display_matchup_overview(data)
    else:
        st.error(f"Failed to load managers data: {data['error']}")


@st.fragment
def render_team_stats_tab():
    """Render team stats tab"""
    from tabs.team_stats.team_stats_overview import display_team_stats_overview
    display_team_stats_overview()


@st.fragment
def render_players_tab():
    """Render players tab - subtab controlled via hamburger menu"""
    subtab_idx = st.session_state.get("subtab_Players", 0)

    # Render only the active subtab (lazy loading)
    if subtab_idx == 0:
        render_players_weekly()
    elif subtab_idx == 1:
        render_players_season()
    elif subtab_idx == 2:
        render_players_career()
    elif subtab_idx == 3:
        render_players_visualize()


@st.fragment
def render_players_weekly():
    """Weekly players view"""
    from md.data_access import load_players_weekly_data
    from tabs.player_stats.weekly_player_stats_optimized import OptimizedWeeklyPlayerViewer

    st.session_state.setdefault("weekly_offset", 0)
    st.session_state.setdefault("weekly_limit", 100)

    @st.cache_data(ttl=300, show_spinner=False)
    def get_weekly_data(offset, limit):
        return load_players_weekly_data(year=None, week=None, limit=limit, offset=offset)

    with st.spinner("Loading weekly player data..."):
        weekly_data = get_weekly_data(
            st.session_state.weekly_offset,
            st.session_state.weekly_limit
        )
        if weekly_data is not None and not weekly_data.empty:
            OptimizedWeeklyPlayerViewer(weekly_data).display()
        else:
            render_empty_state("No weekly player data available")


@st.fragment
def render_players_season():
    """Season players view"""
    from tabs.player_stats.season_player_stats_optimized import OptimizedSeasonPlayerViewer

    st.session_state.setdefault("_current_season_position", "All")

    with st.spinner("Loading season player data..."):
        OptimizedSeasonPlayerViewer().display()


@st.fragment
def render_players_career():
    """Career players view"""
    from tabs.player_stats.career_player_stats_optimized import OptimizedCareerPlayerViewer

    with st.spinner("Loading career player data..."):
        OptimizedCareerPlayerViewer().display()


@st.fragment
def render_players_visualize():
    """Player visualization graphs with native tabs"""
    tab_names = ["Player Cards", "Player Analysis", "SPAR Efficiency", "League Trends"]
    tabs = st.tabs(tab_names)

    try:
        # Tab 1: Player Cards
        with tabs[0]:
            graph_tabs = st.tabs(["Player Card", "Weekly Heatmap", "Scoring Trends"])
            with graph_tabs[0]:
                from tabs.player_stats.graphs.player_graphs.player_card import display_player_card
                display_player_card(prefix="players_card")
            with graph_tabs[1]:
                from tabs.player_stats.graphs.player_graphs.weekly_heatmap import display_weekly_performance_heatmap
                display_weekly_performance_heatmap(prefix="players_heatmap")
            with graph_tabs[2]:
                from tabs.player_stats.graphs.player_graphs.player_scoring_graph import display_player_scoring_graphs
                display_player_scoring_graphs(prefix="players_player_scoring")

        # Tab 2: Player Analysis
        with tabs[1]:
            graph_tabs = st.tabs(["Consistency", "Boom/Bust", "Radar Comparison"])
            with graph_tabs[0]:
                from tabs.player_stats.graphs.player_graphs.player_consistency import display_player_consistency_graph
                display_player_consistency_graph(prefix="players_consistency")
            with graph_tabs[1]:
                from tabs.player_stats.graphs.player_graphs.boom_bust_distribution import display_boom_bust_distribution
                display_boom_bust_distribution(prefix="players_boom_bust")
            with graph_tabs[2]:
                from tabs.player_stats.graphs.player_graphs.player_radar_comparison import display_player_radar_comparison
                display_player_radar_comparison(prefix="players_radar")

        # Tab 3: SPAR Efficiency
        with tabs[2]:
            graph_tabs = st.tabs(["Capture Rate", "Per Week", "Waterfall", "Scatter", "Cumulative", "vs PPG"])
            with graph_tabs[0]:
                from tabs.player_stats.graphs.spar_graphs.manager_capture_rate import display_manager_spar_capture_rate
                display_manager_spar_capture_rate(prefix="spar_capture")
            with graph_tabs[1]:
                from tabs.player_stats.graphs.spar_graphs.spar_per_week import display_spar_per_week_played
                display_spar_per_week_played(prefix="spar_week")
            with graph_tabs[2]:
                from tabs.player_stats.graphs.spar_graphs.weekly_spar_waterfall import display_weekly_spar_waterfall
                display_weekly_spar_waterfall(prefix="spar_waterfall")
            with graph_tabs[3]:
                from tabs.player_stats.graphs.spar_graphs.spar_consistency_scatter import display_spar_consistency_scatter
                display_spar_consistency_scatter(prefix="spar_consistency")
            with graph_tabs[4]:
                from tabs.player_stats.graphs.spar_graphs.cumulative_spar import display_cumulative_spar_over_season
                display_cumulative_spar_over_season(prefix="spar_cumulative")
            with graph_tabs[5]:
                from tabs.player_stats.graphs.spar_graphs.spar_vs_ppg_efficiency import display_spar_vs_ppg_efficiency
                display_spar_vs_ppg_efficiency(prefix="spar_ppg")

        # Tab 4: League Trends
        with tabs[3]:
            graph_tabs = st.tabs(["Position Groups", "SPAR Distribution", "Leaderboard"])
            with graph_tabs[0]:
                from tabs.player_stats.graphs.league_graphs.position_group_scoring import display_position_group_scoring_graphs
                display_position_group_scoring_graphs(prefix="league_pos_group")
            with graph_tabs[1]:
                from tabs.player_stats.graphs.league_graphs.position_spar_boxplot import display_position_spar_boxplot
                display_position_spar_boxplot(prefix="league_spar_box")
            with graph_tabs[2]:
                from tabs.player_stats.graphs.league_graphs.manager_spar_leaderboard import display_manager_spar_leaderboard
                display_manager_spar_leaderboard(prefix="league_manager_board")

    except Exception as e:
        st.warning(f"Player graphs unavailable: {e}")


@st.fragment
def render_draft_tab():
    """Render draft tab with integrated visualizations"""
    # Load and display draft data with integrated graphs
    data = load_draft_tab()
    if "error" not in data:
        from tabs.draft_data.draft_data_overview import display_draft_data_overview
        display_draft_data_overview(data)
    else:
        st.error(f"Failed to load draft data: {data['error']}")


@st.fragment
def render_transactions_tab():
    """Render transactions tab"""
    data = load_transactions_tab()

    if "error" not in data:
        from tabs.transactions.transactions_adds_drops_trades_overview import AllTransactionsViewer
        AllTransactionsViewer(
            data["transactions"],
            data["player_data"],
            data["injury_data"],
            data["draft_data"]
        ).display()
    else:
        st.error(f"Failed to load transactions data: {data['error']}")


@st.fragment
def render_simulations_tab():
    """Render simulations tab"""
    data = load_simulations_tab()

    if "error" not in data:
        from tabs.simulations.simulation_home import display_simulations_viewer
        display_simulations_viewer(data["matchups"])
    else:
        st.error(f"Failed to load simulations data: {data['error']}")


@st.fragment
def render_extras_tab():
    """Render extras tab - subtab controlled via hamburger menu"""
    subtab_idx = st.session_state.get("subtab_Extras", 0)

    if subtab_idx == 0:
        from md.tab_data_access.keepers import load_optimized_keepers_data
        from tabs.keepers.keepers_home import KeeperDataViewer

        with st.spinner("Loading keepers data..."):
            keepers_data = load_optimized_keepers_data()
            if keepers_data is not None:
                KeeperDataViewer(keepers_data).display()
            else:
                render_empty_state("Failed to load keepers data")

    elif subtab_idx == 1:
        from md.tab_data_access.team_names import load_optimized_team_names_data
        from tabs.team_names.team_names import display_team_names

        with st.spinner("Loading team names..."):
            team_names_data = load_optimized_team_names_data()
            if team_names_data is not None:
                display_team_names(team_names_data)
            else:
                render_empty_state("Failed to load team names data")


def main():
    """Main application entry point"""
    # Only set page config if not already set (when running standalone)
    # When embedded from main.py, page config is already set
    try:
        st.set_page_config(
            page_title="KMFFL Analytics",
            layout="wide",
            page_icon="🏈",
            initial_sidebar_state="collapsed"
        )
    except st.errors.StreamlitAPIException:
        # Page config already set by parent (main.py)
        pass

    # Apply modern styles
    from tabs.shared.modern_styles import apply_modern_styles
    apply_modern_styles()

    # Check connectivity
    if not _safe_boot():
        st.stop()

    # Initialize session state
    _init_session_defaults()

    # Navigation structure: main tabs and their subtabs
    tab_names = ["Home", "Managers", "Team Stats", "Players", "Draft", "Transactions", "Simulations", "Extras"]
    tab_icons = ["🏠", "⚔️", "📊", "👤", "🎯", "💼", "🔮", "⭐"]

    # Subtabs for each main section (None = no subtabs)
    subtabs = {
        "Home": ["Overview", "Hall of Fame", "Standings", "Schedules", "Head-to-Head", "Recaps"],
        "Managers": ["Weekly", "Seasons", "Career", "Visualize"],
        "Team Stats": ["Weekly", "Seasons", "Career"],
        "Players": ["Weekly", "Season", "Career", "Visualize"],
        "Draft": None,
        "Transactions": None,
        "Simulations": None,
        "Extras": ["Keeper", "Team Names"],
    }

    # Get current state
    current_idx = st.session_state.get("active_main_tab", 0)
    selected_tab = tab_names[current_idx]
    current_subtab_idx = st.session_state.get(f"subtab_{selected_tab}", 0)

    # Clean hamburger menu - main sections only, subtabs shown as horizontal tabs in content
    st.markdown("""
    <style>
    /* Wider popover */
    [data-testid="stPopoverBody"] {
        min-width: 200px !important;
    }

    /* Primary button (current section) - bright glow */
    .stPopover button[kind="primary"],
    [data-testid="stPopoverBody"] button[kind="primary"] {
        background: linear-gradient(135deg, #818CF8 0%, #A78BFA 100%) !important;
        border: none !important;
        box-shadow: 0 0 12px rgba(129, 140, 248, 0.4) !important;
        font-weight: 600 !important;
    }
    .stPopover button[kind="primary"]:hover,
    [data-testid="stPopoverBody"] button[kind="primary"]:hover {
        box-shadow: 0 0 18px rgba(129, 140, 248, 0.6) !important;
    }

    /* Secondary buttons (other sections) */
    .stPopover button:not([kind="primary"]),
    [data-testid="stPopoverBody"] button:not([kind="primary"]) {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.8) !important;
    }
    .stPopover button:not([kind="primary"]):hover,
    [data-testid="stPopoverBody"] button:not([kind="primary"]):hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(129, 140, 248, 0.4) !important;
    }

    @media (max-width: 768px) {
        [data-testid="stPopoverBody"] {
            min-width: 180px !important;
        }
    }

    /* Horizontal subtab buttons - compact tab style (not bulky pills) */
    .stColumns button {
        padding: 0.4rem 0.75rem !important;
        min-height: unset !important;
        height: auto !important;
        font-size: 0.85rem !important;
        border-radius: 6px 6px 0 0 !important;
        margin-bottom: -1px !important;
    }
    .stColumns button[kind="primary"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid #818CF8 !important;
        color: #A78BFA !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .stColumns button[kind="secondary"],
    .stColumns button:not([kind="primary"]) {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: rgba(255, 255, 255, 0.5) !important;
        font-weight: 500 !important;
    }
    .stColumns button[kind="secondary"]:hover,
    .stColumns button:not([kind="primary"]):hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-bottom: 2px solid rgba(129, 140, 248, 0.4) !important;
        color: rgba(255, 255, 255, 0.85) !important;
    }
    /* Underline container for subtabs */
    .stColumns {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 0.75rem !important;
    }

    /* Mobile: smaller text */
    @media (max-width: 768px) {
        .stColumns button {
            font-size: 0.75rem !important;
            padding: 0.35rem 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Simple hamburger menu - main sections only
    with st.popover("☰ Menu"):
        for i, (name, icon) in enumerate(zip(tab_names, tab_icons)):
            is_current = (i == current_idx)

            if is_current:
                st.button(f"{icon} {name}", key=f"current_{name}", use_container_width=True, type="primary", disabled=True)
            else:
                if st.button(f"{icon} {name}", key=f"main_{i}", use_container_width=True):
                    st.session_state["active_main_tab"] = i
                    st.session_state[f"subtab_{name}"] = 0
                    st.rerun()

    # Get subtabs for current section
    section_subtabs = subtabs.get(selected_tab)

    # Show horizontal subtab navigation for sections that have subtabs
    if section_subtabs:
        # Create pill-style buttons for subtab navigation
        cols = st.columns(len(section_subtabs))
        for idx, (col, subtab_name) in enumerate(zip(cols, section_subtabs)):
            with col:
                is_active = (idx == current_subtab_idx)
                btn_type = "primary" if is_active else "secondary"
                if st.button(subtab_name, key=f"subtab_btn_{selected_tab}_{idx}", use_container_width=True, type=btn_type):
                    if not is_active:
                        st.session_state[f"subtab_{selected_tab}"] = idx
                        st.rerun()

        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Render the active section using existing render functions
    if selected_tab == "Home":
        render_home_tab()
    elif selected_tab == "Managers":
        render_managers_tab()
    elif selected_tab == "Team Stats":
        render_team_stats_tab()
    elif selected_tab == "Players":
        render_players_tab()
    elif selected_tab == "Draft":
        render_draft_tab()
    elif selected_tab == "Transactions":
        render_transactions_tab()
    elif selected_tab == "Simulations":
        render_simulations_tab()
    elif selected_tab == "Extras":
        render_extras_tab()


if __name__ == "__main__":
    main()
