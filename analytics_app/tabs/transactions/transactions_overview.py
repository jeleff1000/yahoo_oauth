"""
Transactions Tab - Main Entry Point

Structure:
- Weekly (transaction-by-transaction): Add/Drop, Trades, Drop Regrets
- Season (season aggregates): Add/Drop, Trades, Report Card
- Career (all-time aggregates): Add/Drop, Trades, Report Card
"""

import streamlit as st
from ..shared.modern_styles import apply_modern_styles

# Weekly views
from .weekly_add_drop import display_weekly_add_drop
from .trade_by_trade_summary_data import display_trade_by_trade_summary_data
from .drop_regret_analysis import display_drop_regret_analysis

# Season views
from .season_add_drop import display_season_add_drop
from .season_trade_data import display_season_trade_data
from .transaction_report_card import display_transaction_report_card

# Career views
from .career_add_drop import display_career_add_drop
from .career_trade_data import display_career_trade_data


class AllTransactionsViewer:
    def __init__(self, transaction_df, player_df, draft_history_df):
        self.transaction_df = transaction_df
        self.player_df = player_df
        self.draft_history_df = draft_history_df

    @st.fragment
    def display(self):
        apply_modern_styles()

        # Top-level navigation buttons
        main_tab_names = ["Weekly", "Season", "Career"]
        current_main_idx = st.session_state.get("subtab_Transactions", 0)

        cols = st.columns(len(main_tab_names))
        for idx, (col, name) in enumerate(zip(cols, main_tab_names)):
            with col:
                is_active = idx == current_main_idx
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    name,
                    key=f"trans_main_{idx}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    if not is_active:
                        st.session_state["subtab_Transactions"] = idx
                        st.rerun()

        # ==================== WEEKLY ====================
        if current_main_idx == 0:
            weekly_subtabs = st.tabs(["Add/Drop", "Trades", "Drop Regrets"])

            with weekly_subtabs[0]:
                add_drop_keys = {
                    "year_search": "year_search_weekly_add_drop",
                    "added_player_search": "added_player_search_weekly",
                    "nickname_search": "nickname_search_weekly",
                    "dropped_player_search": "dropped_player_search_weekly",
                    "added_position_search": "added_position_search_weekly",
                    "dropped_position_search": "dropped_position_search_weekly",
                }
                display_weekly_add_drop(
                    self.transaction_df, self.player_df, add_drop_keys
                )

            with weekly_subtabs[1]:
                display_trade_by_trade_summary_data(
                    self.transaction_df, self.player_df, self.draft_history_df
                )

            with weekly_subtabs[2]:
                display_drop_regret_analysis(self.transaction_df)

        # ==================== SEASON ====================
        elif current_main_idx == 1:
            season_subtabs = st.tabs(["Add/Drop", "Trades", "Report Card"])

            with season_subtabs[0]:
                display_season_add_drop(self.transaction_df, self.player_df)

            with season_subtabs[1]:
                display_season_trade_data(
                    self.transaction_df, self.player_df, self.draft_history_df
                )

            with season_subtabs[2]:
                display_transaction_report_card(self.transaction_df, self.player_df)

        # ==================== CAREER ====================
        elif current_main_idx == 2:
            career_subtabs = st.tabs(["Add/Drop", "Trades", "Report Card"])

            with career_subtabs[0]:
                display_career_add_drop(self.transaction_df, self.player_df)

            with career_subtabs[1]:
                display_career_trade_data(
                    self.transaction_df, self.player_df, self.draft_history_df
                )

            with career_subtabs[2]:
                # Career report card - pass flag to show all-time view
                display_transaction_report_card(
                    self.transaction_df, self.player_df, career_view=True
                )


@st.fragment
def display_transactions_overview(transaction_df, player_df, draft_history_df):
    """Main entry point for transactions tab."""
    AllTransactionsViewer(transaction_df, player_df, draft_history_df).display()
