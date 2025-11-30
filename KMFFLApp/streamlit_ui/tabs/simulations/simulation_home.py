from __future__ import annotations

import streamlit as st
import duckdb
import pandas as pd
from ..shared.modern_styles import apply_modern_styles

# ---- WHAT-IF viewers ----
from .what_if.shuffle_schedules.shuffled_win_total_viewer import GaviStatViewer
from .what_if.strength_of_schedule.opponent_shuffle_win_total import OpponentGaviStatViewer
from .what_if.strength_of_schedule.everyones_schedule_viewer import EveryonesScheduleViewer
from .what_if.shuffle_schedules.vs_one_opponent_viewer import VsOneOpponentViewer
from .what_if.tweak_scoring.tweak_scoring_viewer import TweakScoringViewer
from .what_if.shuffle_schedules.expected_record_viewer import display_expected_record
from .what_if.shuffle_schedules.expected_seed_viewer import display_expected_seed
from .what_if.strength_of_schedule.sos_expected_record_viewer import display_opp_expected_record
from .what_if.strength_of_schedule.sos_expected_seed_viewer import display_opp_expected_seed

# ---- PREDICTIVE viewers ----
from .predictive.playoff_odds import PlayoffOddsViewer as PlayoffOddsSnapshot
from .predictive.playoff_odds_graph import PlayoffOddsViewer as PlayoffOddsGraph
from .predictive.playoff_machine import display_playoff_machine
from .predictive.predictive_record import display_predicted_record
from .predictive.predictive_seed import display_predicted_seed

# Additional playoff visualization modules
from .predictive.yearly_playoff_graph import PlayoffOddsCumulativeViewer
from .predictive.playoff_simulation_enhanced import display_playoff_simulation_dashboard


class SimulationDataViewer:
    def __init__(self, matchup_data_df: pd.DataFrame | None):
        self.matchup_data_df = matchup_data_df

        # Safe DuckDB session
        self.con = duckdb.connect()
        if self.matchup_data_df is not None:
            self.con.register("matchup_data", self.matchup_data_df)

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).fetchdf()

    @st.fragment
    def display(self):
        apply_modern_styles()

        if self.matchup_data_df is None or self.matchup_data_df.empty:
            st.info("📊 No matchup data available. Please ensure data is loaded.")
            return

        # Compact header - matching modern style
        st.header("🎲 Season Simulations")
        st.caption("Monte Carlo simulations (100K iterations) and ML predictions for playoff odds, schedule luck, and season outcomes.")

        # Two main category tabs for clear separation
        main_tabs = st.tabs(["📊 Predictive Analytics", "🔀 What-If Scenarios"])

        # ==================== PREDICTIVE ANALYTICS ====================
        with main_tabs[0]:
            pred_tabs = st.tabs([
                "🏆 Playoff Dashboard",
                "🎰 Playoff Machine",
                "📊 Final Records",
                "🎯 Playoff Seeds",
                "📅 Weekly Odds",
                "📈 Multi-Year Trends"
            ])

            with pred_tabs[0]:
                try:
                    display_playoff_simulation_dashboard(prefix="playoff_sim")
                except Exception as e:
                    st.warning(f"Playoff dashboard unavailable: {e}")

            with pred_tabs[1]:
                try:
                    display_playoff_machine(self.matchup_data_df)
                except Exception as e:
                    st.warning(f"Playoff machine unavailable: {e}")

            with pred_tabs[2]:
                try:
                    display_predicted_record(self.matchup_data_df)
                except Exception as e:
                    st.warning(f"Predicted records unavailable: {e}")

            with pred_tabs[3]:
                try:
                    display_predicted_seed(self.matchup_data_df)
                except Exception as e:
                    st.warning(f"Predicted seeds unavailable: {e}")

            with pred_tabs[4]:
                try:
                    PlayoffOddsGraph(self.matchup_data_df).display()
                except Exception as e:
                    st.warning(f"Weekly odds unavailable: {e}")

            with pred_tabs[5]:
                try:
                    PlayoffOddsCumulativeViewer(self.matchup_data_df).display()
                except Exception as e:
                    st.warning(f"Multi-year trends unavailable: {e}")

        # ==================== WHAT-IF SCENARIOS ====================
        with main_tabs[1]:
            whatif_tabs = st.tabs([
                "🔀 Schedule Shuffles",
                "💪 Strength of Schedule",
                "⚖️ Score Sensitivity"
            ])

            # ---- Schedule Shuffles (4 sub-tabs) ----
            with whatif_tabs[0]:
                st.caption("What if you kept your scores but faced randomized opponents?")
                schedule_subtabs = st.tabs([
                    "📊 Win Distribution",
                    "🤝 Head-to-Head",
                    "📈 Expected Records",
                    "🎯 Expected Seeding"
                ])

                with schedule_subtabs[0]:
                    try:
                        GaviStatViewer(self.matchup_data_df).display()
                    except Exception as e:
                        st.warning(f"Win distribution unavailable: {e}")

                with schedule_subtabs[1]:
                    try:
                        VsOneOpponentViewer(self.matchup_data_df).display()
                    except Exception as e:
                        st.warning(f"Head-to-head analysis unavailable: {e}")

                with schedule_subtabs[2]:
                    try:
                        display_expected_record(self.matchup_data_df)
                    except Exception as e:
                        st.warning(f"Expected records unavailable: {e}")

                with schedule_subtabs[3]:
                    try:
                        display_expected_seed(self.matchup_data_df)
                    except Exception as e:
                        st.warning(f"Expected seeding unavailable: {e}")

            # ---- Strength of Schedule (4 sub-tabs) ----
            with whatif_tabs[1]:
                st.caption("What if everyone faced your opponents? Measures schedule difficulty.")
                opponent_subtabs = st.tabs([
                    "📊 Win Distribution",
                    "📅 Everyone's Schedule",
                    "📈 Expected Records",
                    "🎯 Expected Seeding"
                ])

                with opponent_subtabs[0]:
                    try:
                        OpponentGaviStatViewer(self.matchup_data_df).display()
                    except Exception as e:
                        st.warning(f"Opponent analysis unavailable: {e}")

                with opponent_subtabs[1]:
                    try:
                        EveryonesScheduleViewer(self.matchup_data_df).display()
                    except Exception as e:
                        st.warning(f"Schedule comparison unavailable: {e}")

                with opponent_subtabs[2]:
                    try:
                        display_opp_expected_record(self.matchup_data_df)
                    except Exception as e:
                        st.warning(f"Record vs difficulty unavailable: {e}")

                with opponent_subtabs[3]:
                    try:
                        display_opp_expected_seed(self.matchup_data_df)
                    except Exception as e:
                        st.warning(f"Seeding vs difficulty unavailable: {e}")

            # ---- Score Sensitivity (direct view) ----
            with whatif_tabs[2]:
                try:
                    TweakScoringViewer(self.matchup_data_df).display()
                except Exception as e:
                    st.warning(f"Score sensitivity analysis unavailable: {e}")


@st.fragment
def display_simulations_viewer(matchup_data_df: pd.DataFrame | None, player_data_df=None):
    """Main entry point for simulations viewer"""
    SimulationDataViewer(matchup_data_df).display()