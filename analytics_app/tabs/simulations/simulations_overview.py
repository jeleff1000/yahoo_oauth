from __future__ import annotations

import streamlit as st
import duckdb
import pandas as pd
from datetime import datetime
from ..shared.modern_styles import apply_modern_styles
from .shared.simulation_styles import (
    apply_simulation_styles,
    start_simulation_container,
    end_simulation_container,
)
from .shared.unified_header import (
    render_context_card,
    get_today_display,
)

# ---- WHAT-IF viewers ----
from .what_if.shuffle_schedules.shuffled_win_total_viewer import GaviStatViewer
from .what_if.strength_of_schedule.opponent_shuffle_win_total import (
    OpponentGaviStatViewer,
)
from .what_if.strength_of_schedule.everyones_schedule_viewer import (
    EveryonesScheduleViewer,
)
from .what_if.shuffle_schedules.vs_one_opponent_viewer import VsOneOpponentViewer
from .what_if.tweak_scoring.tweak_scoring_viewer import TweakScoringViewer
from .what_if.shuffle_schedules.expected_record_viewer import display_expected_record
from .what_if.shuffle_schedules.expected_seed_viewer import display_expected_seed
from .what_if.strength_of_schedule.sos_expected_record_viewer import (
    display_opp_expected_record,
)
from .what_if.strength_of_schedule.sos_expected_seed_viewer import (
    display_opp_expected_seed,
)

# ---- PREDICTIVE viewers ----
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
        apply_simulation_styles()

        if self.matchup_data_df is None or self.matchup_data_df.empty:
            st.info("No matchup data available. Please ensure data is loaded.")
            return

        start_simulation_container()

        # ==================== LAYER 1: UNIFIED HEADER ====================
        # Get available years and weeks from data
        years = sorted(
            self.matchup_data_df["year"].astype(int).unique(), reverse=True
        )
        max_year = years[0] if years else datetime.now().year

        # Session state for year/week selection
        if "sim_selected_year" not in st.session_state:
            st.session_state["sim_selected_year"] = max_year
        if "sim_selected_week" not in st.session_state:
            year_data = self.matchup_data_df[
                self.matchup_data_df["year"] == max_year
            ]
            st.session_state["sim_selected_week"] = int(year_data["week"].max())

        selected_year = st.session_state["sim_selected_year"]
        weeks_for_year = sorted(
            self.matchup_data_df[self.matchup_data_df["year"] == selected_year][
                "week"
            ]
            .astype(int)
            .unique()
        )
        selected_week = st.session_state["sim_selected_week"]

        # Header row: Title | Mode Toggle | Year/Week/Date
        header_cols = st.columns([2, 3, 3])

        with header_cols[0]:
            st.markdown("**Playoff Simulation**")

        with header_cols[1]:
            # Segmented control for Predictive/What-If
            main_tab_names = ["Predictive", "What-If"]
            current_main_idx = st.session_state.get("subtab_Simulations", 0)

            seg_cols = st.columns(2)
            for idx, (col, name) in enumerate(zip(seg_cols, main_tab_names)):
                with col:
                    is_active = idx == current_main_idx
                    if st.button(
                        name,
                        key=f"sim_main_{idx}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if not is_active:
                            st.session_state["subtab_Simulations"] = idx
                            st.rerun()

        with header_cols[2]:
            # Year, Week, Today's Date
            meta_cols = st.columns([1.2, 1, 1.5])
            with meta_cols[0]:
                new_year = st.selectbox(
                    "Season",
                    years,
                    index=years.index(selected_year) if selected_year in years else 0,
                    key="sim_year_select",
                    label_visibility="collapsed",
                )
                if new_year != selected_year:
                    st.session_state["sim_selected_year"] = new_year
                    # Reset week to max for new year
                    year_data = self.matchup_data_df[
                        self.matchup_data_df["year"] == new_year
                    ]
                    st.session_state["sim_selected_week"] = int(year_data["week"].max())
                    st.rerun()

            with meta_cols[1]:
                new_week = st.selectbox(
                    "Week",
                    weeks_for_year,
                    index=(
                        weeks_for_year.index(selected_week)
                        if selected_week in weeks_for_year
                        else len(weeks_for_year) - 1
                    ),
                    key="sim_week_select",
                    label_visibility="collapsed",
                )
                if new_week != selected_week:
                    st.session_state["sim_selected_week"] = new_week
                    st.rerun()

            with meta_cols[2]:
                st.caption(f"Today: {get_today_display()}")

        # ==================== LAYER 2: SUB-NAVIGATION ====================
        # ==================== PREDICTIVE ANALYTICS ====================
        if current_main_idx == 0:
            pred_tabs = st.tabs(
                [
                    "Playoff Dashboard",
                    "Playoff Machine",
                    "Final Records",
                    "Playoff Seeds",
                    "Weekly Odds",
                    "Multi-Year Trends",
                ]
            )

            # Context card pinned under tabs
            render_context_card(
                season=selected_year,
                week=selected_week,
                num_simulations=10000,
                help_text="Based on current standings and power ratings",
            )

            with pred_tabs[0]:
                try:
                    display_playoff_simulation_dashboard(
                        prefix="playoff_sim",
                        year=selected_year,
                        week=selected_week,
                    )
                except Exception as e:
                    st.warning(f"Playoff dashboard unavailable: {e}")

            with pred_tabs[1]:
                try:
                    display_playoff_machine(
                        self.matchup_data_df,
                        year=selected_year,
                        week=selected_week,
                    )
                except Exception as e:
                    st.warning(f"Playoff machine unavailable: {e}")

            with pred_tabs[2]:
                try:
                    display_predicted_record(
                        self.matchup_data_df,
                        year=selected_year,
                        week=selected_week,
                    )
                except Exception as e:
                    st.warning(f"Predicted records unavailable: {e}")

            with pred_tabs[3]:
                try:
                    display_predicted_seed(
                        self.matchup_data_df,
                        year=selected_year,
                        week=selected_week,
                    )
                except Exception as e:
                    st.warning(f"Predicted seeds unavailable: {e}")

            with pred_tabs[4]:
                try:
                    PlayoffOddsGraph(self.matchup_data_df).display(
                        year=selected_year,
                        week=selected_week,
                    )
                except Exception as e:
                    st.warning(f"Weekly odds unavailable: {e}")

            with pred_tabs[5]:
                try:
                    PlayoffOddsCumulativeViewer(self.matchup_data_df).display(
                        year=selected_year,
                        week=selected_week,
                    )
                except Exception as e:
                    st.warning(f"Multi-year trends unavailable: {e}")

        # ==================== WHAT-IF SCENARIOS ====================
        elif current_main_idx == 1:
            whatif_tabs = st.tabs(
                ["Schedule Shuffles", "Strength of Schedule", "Score Sensitivity"]
            )

            # Context card pinned under tabs (shared year/week)
            render_context_card(
                season=selected_year,
                week=selected_week,
                num_simulations=10000,
                help_text="What-if analysis based on historical data",
            )

            # ---- Schedule Shuffles (4 sub-tabs) ----
            with whatif_tabs[0]:
                st.caption(
                    "What if you kept your scores but faced randomized opponents?"
                )
                schedule_subtabs = st.tabs(
                    [
                        "Win Distribution",
                        "Head-to-Head",
                        "Expected Records",
                        "Expected Seeding",
                    ]
                )

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
                st.caption(
                    "What if everyone faced your opponents? Measures schedule difficulty."
                )
                opponent_subtabs = st.tabs(
                    [
                        "Win Distribution",
                        "Everyone's Schedule",
                        "Expected Records",
                        "Expected Seeding",
                    ]
                )

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

        end_simulation_container()


@st.fragment
def display_simulations_overview(
    matchup_data_df: pd.DataFrame | None, player_data_df=None
):
    """Main entry point for simulations tab."""
    SimulationDataViewer(matchup_data_df).display()


# Backward compatibility alias
display_simulations_viewer = display_simulations_overview
