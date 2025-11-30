#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

from md.data_access import (
    latest_season_and_week,
    load_homepage_data,
    load_players_weekly_data,
    load_players_career_data,     # career loader (no min_games / no limits)
    load_draft_data,
    load_transactions_data,
    load_simulations_data,        # used to fetch the raw matchup DF
    load_keepers_data,
    load_team_names_data,
    load_player_two_week_slice,   # two-week slice for homepage recaps
    load_managers_data,           # Managers tab
)

# Tabs
from tabs.simulations.simulation_home import display_simulations_viewer
from tabs.keepers.keepers_home import KeeperDataViewer
from tabs.player_stats.weekly_player_stats_optimized import OptimizedWeeklyPlayerViewer
from tabs.player_stats.season_player_stats_optimized import OptimizedSeasonPlayerViewer
from tabs.player_stats.career_player_stats_optimized import OptimizedCareerPlayerViewer
from tabs.draft_data.draft_data_overview import display_draft_data_overview
from tabs.transactions.transactions_adds_drops_trades_overview import AllTransactionsViewer
from tabs.team_names.team_names import display_team_names
from tabs.homepage.homepage_overview import display_homepage_overview
from tabs.matchups.matchup_overview import display_matchup_overview


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


def _defaults():
    try:
        y, w = latest_season_and_week()
        y = int(y) if y else 0
        w = int(w) if w else 0
    except Exception:
        y, w = 0, 0
    st.session_state.setdefault("year", y)
    st.session_state.setdefault("week", w)

    # ============= Main Tab Tracking =============
    st.session_state.setdefault("active_main_tab", 0)

    # ============= Home Tab Subtabs (6 subtabs) =============
    st.session_state.setdefault("active_home_subtab", 0)
    # Home subtabs: Overview, Champions, Season Standings, Schedules, Head-to-Head, Team Recaps

    # ============= Managers Tab Subtabs (3 subtabs) =============
    st.session_state.setdefault("active_managers_subtab", 0)
    # Managers subtabs: Weekly, Seasons, Career

    # Each Manager subtab has further nested tabs
    st.session_state.setdefault("active_managers_weekly_subtab", 0)
    st.session_state.setdefault("active_managers_season_subtab", 0)
    st.session_state.setdefault("active_managers_career_subtab", 0)

    # ============= Players Tab Subtabs (3 subtabs) =============
    st.session_state.setdefault("active_players_subtab", 0)
    # Players subtabs: Weekly, Season, Career

    # Each Player subtab has further nested tabs
    st.session_state.setdefault("active_players_weekly_stats_subtab", 0)  # Basic, Advanced, Matchup, H2H
    st.session_state.setdefault("active_players_season_stats_subtab", 0)  # Basic, Advanced, Matchup
    st.session_state.setdefault("active_players_career_stats_subtab", 0)  # Basic, Advanced, Matchup

    # ============= Draft Tab Subtabs (6 subtabs) =============
    st.session_state.setdefault("active_draft_subtab", 0)
    # Draft subtabs: Draft Summary, Scoring Outcomes, Career Draft Stats, Draft Optimizer, Draft Preferences, Average Draft Prices

    # Draft Preferences has 2 sub-subtabs
    st.session_state.setdefault("active_draft_preferences_subtab", 0)  # Draft Tables, Cost Over Time Graph

    # ============= Transactions Tab Subtabs =============
    st.session_state.setdefault("active_transactions_subtab", 0)
    # Transactions main tabs: Adds, Drops, Trades, All Transactions

    # Nested transaction tabs
    st.session_state.setdefault("active_transactions_adds_subtab", 0)
    st.session_state.setdefault("active_transactions_drops_subtab", 0)
    st.session_state.setdefault("active_transactions_trades_subtab", 0)
    st.session_state.setdefault("active_transactions_combo_subtab", 0)  # Weekly, Season, Career

    # ============= Simulations Tab (HEAVY COMPUTATIONAL) =============
    st.session_state.setdefault("active_simulations_main_tab", 0)  # Season Projections, Alternative Scenarios

    # Season Projections subtab (Predictive)
    st.session_state.setdefault("active_simulations_predictive_type", "")

    # Alternative Scenarios has 3 sub-subtabs
    st.session_state.setdefault("active_simulations_whatif_subtab", 0)  # Schedule Sims, Opponent Strength, Score Adjustments

    # Schedule Simulations has selections
    st.session_state.setdefault("active_simulations_schedule_type", "")

    # For heavy computational tabs like shuffled wins
    st.session_state.setdefault("active_simulations_shuffle_view", 0)  # Year, Manager, All
    st.session_state.setdefault("active_simulations_opponent_view", 0)  # Year, Manager, All

    # Opponent Strength has selections
    st.session_state.setdefault("active_simulations_strength_type", "")

    # ============= Extras Tab Subtabs (now 2 subtabs) =============
    st.session_state.setdefault("active_extras_subtab", 0)
    # Extras subtabs: Keeper, Team Names

    # ============= Graphs Sub-Subtabs (3 sub-subtabs) =============
    st.session_state.setdefault("active_graphs_subtab", 0)
    # Graphs sub-subtabs: Manager Graphs, Draft Graphs, Playoff Odds

    # ============= Manager Graphs Sub-Sub-Subtabs (5 tabs) =============
    st.session_state.setdefault("active_manager_graphs_subtab", 0)
    # Manager Graphs tabs: Weekly Scoring, All-Time Scoring, Total Wins, Win %, Power Rating

    # ============= Draft Graphs Sub-Sub-Subtabs (1 tab) =============
    st.session_state.setdefault("active_draft_graphs_subtab", 0)
    # Draft Graphs tabs: Draft Preferences

    # ============= Playoff Odds Sub-Sub-Subtabs (2 tabs) =============
    st.session_state.setdefault("active_playoff_odds_subtab", 0)
    # Playoff Odds tabs: Weekly, Cumulative Week

    # ============= Injury Data (if used) =============
    st.session_state.setdefault("active_injury_subtab", 0)

    # ============= Hall of Fame (if used) =============
    st.session_state.setdefault("active_hall_of_fame_subtab", 0)
    st.session_state.setdefault("active_top_teams_subtab", 0)  # Top Seasons, Top Weeks

    # ============= Data Cache Flags =============
    st.session_state.setdefault("_data_loaded", {})

    # ============= Cached Data Storage =============
    st.session_state.setdefault("_cached_homepage_data", None)
    st.session_state.setdefault("_cached_managers_data", None)
    st.session_state.setdefault("_cached_draft_data", None)
    st.session_state.setdefault("_cached_transactions_data", None)
    st.session_state.setdefault("_cached_simulations_data", None)
    st.session_state.setdefault("_cached_keepers_data", None)
    st.session_state.setdefault("_cached_team_names_data", None)
    st.session_state.setdefault("_cached_weekly_player_data", None)
    st.session_state.setdefault("_cached_season_player_data", None)
    st.session_state.setdefault("_cached_career_player_data", None)
    st.session_state.setdefault("_cached_graphs_data", None)

    # ============= Cached Data for Heavy Computations =============
    st.session_state.setdefault("_cached_draft_optimizer_data", None)
    st.session_state.setdefault("_cached_simulation_results", {})


# ------------------------- Data reload helpers -------------------------

@st.cache_data(ttl=300, show_spinner=False)
def reload_weekly_data():
    offset = st.session_state.get("weekly_offset", 0)
    limit  = st.session_state.get("weekly_limit", 100)
    return load_players_weekly_data(
        year=None,
        week=None,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=300, show_spinner=False)
def reload_career_data():
    """
    Career page now has no row limits or legacy params.
    Keep a simple call here in case the Career viewer expects a DF.
    """
    # No min_games, no limit/offset — data_access aggregates across years with all pre-agg filters.
    return load_players_career_data()


# ------------------------------ Main ----------------------------------

def main():
    st.set_page_config(page_title="KMFFL Analytics", layout="wide", page_icon="🏈")

    # Apply modern styles
    from tabs.shared.modern_styles import apply_modern_styles
    apply_modern_styles()

    if not _safe_boot():
        st.stop()

    _defaults()

    # Create tabs with on_change callback to track active tab
    tab_names = ["Home", "Managers", "Players", "Draft", "Transactions", "Simulations", "Extras"]

    # Use a container to manage tab state
    tab_container = st.container()

    with tab_container:
        # Create radio button group (hidden via CSS) to track tab state
        # This is a workaround since st.tabs doesn't directly support session state
        col1, col2 = st.columns([6, 1])
        with col2:
            st.write("")  # Spacer

        tabs = st.tabs(tab_names)

    # ------------------------- Home -----------------------------------
    with tabs[0]:
        with st.spinner("Loading homepage data..."):
            summary_data = load_homepage_data()

            sims_payload = load_simulations_data(include_all_years=True)
            matchup_df = None if ("error" in sims_payload) else sims_payload.get("matchups")

            # Two-week Player slice for the latest week (for H2H/recaps)
            two_week_players = None
            try:
                y, w = latest_season_and_week()
                if y and w:
                    two_week_players = load_player_two_week_slice(y, w)
            except Exception as e:
                st.warning(f"Two-week player slice unavailable: {e}")

            display_homepage_overview({
                "summary": summary_data,
                "Matchup Data": matchup_df,
                "Player Data": two_week_players,
            })

    # ------------------------- Managers --------------------------------
    with tabs[1]:
        with st.spinner("Loading managers data..."):
            managers_data = load_managers_data()
            if "error" not in managers_data:
                display_matchup_overview(managers_data)
            else:
                st.error(f"Failed to load managers data: {managers_data['error']}")

    # ------------------------- Players ---------------------------------
    with tabs[2]:
        sub = st.tabs(["Weekly", "Season", "Career", "Visualize"])  # added Visualize subtab

        # -------- Weekly --------
        with sub[0]:
            if "weekly_offset" not in st.session_state:
                st.session_state.weekly_offset = 0
            if "weekly_limit" not in st.session_state:
                st.session_state.weekly_limit = 100

            with st.spinner("Loading weekly player data..."):
                weekly_data = reload_weekly_data()
                if weekly_data is not None and not weekly_data.empty:
                    OptimizedWeeklyPlayerViewer(weekly_data).display()
                else:
                    st.error("Failed to load weekly player data")

        # -------- Season --------
        with sub[1]:
            # The Season viewer self-loads (search bars push filters down pre-aggregation),
            # so we don't pass any preloaded DF here.
            if "_current_season_position" not in st.session_state:
                st.session_state["_current_season_position"] = "All"

            with st.spinner("Loading season player data..."):
                OptimizedSeasonPlayerViewer().display()

        # -------- Career --------
        with sub[2]:
            # If your Career viewer also self-loads, you can swap to: StreamlitCareerPlayerDataViewer(None).display()
            # For backward compatibility we pass a DF (no limits/legacy args in loader).
            with st.spinner("Loading career player data..."):
                career_data = reload_career_data()
                if career_data is not None and not career_data.empty:
                    OptimizedCareerPlayerViewer().display()
                else:
                    st.error("Failed to load career player data")

        # -------- Visualize (Player Graphs) --------
        with sub[3]:
            st.markdown("*Visual analytics for players*")
            try:
                # Defensive imports so page doesn't break if graph modules fail
                from tabs.graphs.player_graphs.player_scoring_graph import display_player_scoring_graphs
                from tabs.graphs.player_graphs.position_group_scoring import display_position_group_scoring_graphs
                from tabs.graphs.player_graphs.player_consistency import display_player_consistency_graph

                # Provide subtabs for player graphs
                player_graph_tabs = st.tabs(["📈 Scoring Trends", "📊 Position Groups", "🎯 Consistency"])

                with player_graph_tabs[0]:
                    display_player_scoring_graphs(prefix="players_player_scoring")
                with player_graph_tabs[1]:
                    display_position_group_scoring_graphs(prefix="players_pos_group")
                with player_graph_tabs[2]:
                    display_player_consistency_graph(prefix="players_consistency")

            except Exception as e:
                st.warning(f"Player graphs unavailable: {e}")

    # ------------------------- Draft -----------------------------------
    with tabs[3]:
        with st.spinner("Loading draft data..."):
            draft_data = load_draft_data()
            if "error" not in draft_data:
                # Create subtabs for Draft Overview and Draft Graphs
                draft_main_tabs = st.tabs(["Overview", "Graphs"])

                with draft_main_tabs[0]:
                    display_draft_data_overview(draft_data)

                with draft_main_tabs[1]:
                    st.markdown("*Draft visualizations*")
                    try:
                        # Defensive imports
                        from tabs.graphs.draft_graphs.draft_spending_trends import display_draft_spending_trends
                        from tabs.graphs.draft_graphs.draft_round_efficiency import display_draft_round_efficiency
                        from tabs.graphs.draft_graphs.draft_market_trends import display_draft_market_trends
                        from tabs.graphs.draft_graphs.draft_keeper_analysis import display_draft_keeper_analysis

                        draft_graph_tabs = st.tabs(["💸 Spending Trends", "🔁 Round Efficiency", "📈 Market Trends", "🔒 Keeper Analysis"])

                        with draft_graph_tabs[0]:
                            display_draft_spending_trends(prefix="draft_spending")
                        with draft_graph_tabs[1]:
                            display_draft_round_efficiency(prefix="draft_round_eff")
                        with draft_graph_tabs[2]:
                            display_draft_market_trends(prefix="draft_market")
                        with draft_graph_tabs[3]:
                            display_draft_keeper_analysis(prefix="draft_keeper")

                    except Exception as e:
                        st.warning(f"Draft graphs unavailable: {e}")

            else:
                st.error(f"Failed to load draft data: {draft_data['error']}")

    # ------------------------- Transactions ----------------------------
    with tabs[4]:
        with st.spinner("Loading transactions data..."):
            transactions_data = load_transactions_data(limit=1000)
            if "error" not in transactions_data:
                AllTransactionsViewer(
                    transactions_data["transactions"],
                    transactions_data["player_data"],
                    transactions_data["injury_data"],
                    transactions_data["draft_data"]
                ).display()
            else:
                st.error(f"Failed to load transactions data: {transactions_data['error']}")

    # ------------------------- Simulations -----------------------------
    with tabs[5]:
        with st.spinner("Loading simulations data... This may take a moment."):
            simulations_data = load_simulations_data()
            if "error" not in simulations_data:
                display_simulations_viewer(simulations_data["matchups"])
            else:
                st.error(f"Failed to load simulations data: {simulations_data['error']}")

    # ------------------------- Extras ----------------------------------
    with tabs[6]:
        sub = st.tabs(["Keeper", "Team Names"])  # removed Graphs subtab

        with sub[0]:
            with st.spinner("Loading keepers data..."):
                keepers_data = load_keepers_data()
                if keepers_data is not None:
                    KeeperDataViewer(keepers_data).display()
                else:
                    st.error("Failed to load keepers data")

        with sub[1]:
            with st.spinner("Loading team names..."):
                team_names_data = load_team_names_data()
                if team_names_data is not None:
                    display_team_names(team_names_data)
                else:
                    st.error("Failed to load team names data")


if __name__ == "__main__":
    main()
