"""
Injury Data Overview - Main entry point for injury stats tab.
"""

import streamlit as st
import pandas as pd
from ..shared.modern_styles import apply_modern_styles
from .weekly_injury_stats import WeeklyInjuryStatsViewer
from .season_injury_stats import SeasonInjuryStatsViewer
from .career_injury_stats import CareerInjuryStatsViewer


@st.fragment
def display_injury_overview(df_dict):
    injury_data = df_dict.get("Injury Data")
    player_data = df_dict.get("Player Data")
    required_columns = {"player", "week", "year"}

    if injury_data is not None and player_data is not None:
        if "full_name" in injury_data.columns:
            injury_data = injury_data.rename(columns={"full_name": "player"})
        missing_injury = required_columns - set(injury_data.columns)
        missing_player = required_columns - set(player_data.columns)
        if not missing_injury and not missing_player:
            for col in ["year", "week"]:
                if col in injury_data.columns:
                    injury_data[col] = pd.to_numeric(injury_data[col], errors="coerce")
                if col in player_data.columns:
                    player_data[col] = pd.to_numeric(player_data[col], errors="coerce")
            injury_data["player"] = injury_data["player"].astype(str)
            player_data["player"] = player_data["player"].astype(str)
            merged_data = pd.merge(
                injury_data, player_data, on=["player", "week", "year"], how="inner"
            )

            # Remove duplicate position columns and standardize to nfl_position
            if (
                "position_x" in merged_data.columns
                and "position_y" in merged_data.columns
            ):
                merged_data = merged_data.drop(columns=["position_x"])
                merged_data = merged_data.rename(columns={"position_y": "nfl_position"})
            elif "position_y" in merged_data.columns:
                merged_data = merged_data.rename(columns={"position_y": "nfl_position"})
            elif "position" in merged_data.columns:
                merged_data = merged_data.rename(columns={"position": "nfl_position"})

            # Remove any accidental duplicate nfl_position columns
            merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

            injury_stats_viewer = InjuryStatsViewer()
            injury_stats_viewer.display(merged_data)
        else:
            st.error(
                f"Missing columns: "
                f"Injury Data: {missing_injury if missing_injury else 'None'}, "
                f"Player Data: {missing_player if missing_player else 'None'}"
            )
    else:
        st.error("Injury data or Player data not found.")


class InjuryStatsViewer:
    """Viewer for injury statistics with Weekly/Season/Career breakdown."""

    def __init__(self):
        self.weekly_viewer = WeeklyInjuryStatsViewer()
        self.season_viewer = SeasonInjuryStatsViewer()
        self.career_viewer = CareerInjuryStatsViewer()

    @st.fragment
    def display(self, merged_data):
        apply_modern_styles()

        # Top-level navigation buttons (consistent with other tabs)
        main_tab_names = ["Weekly", "Season", "Career"]
        current_main_idx = st.session_state.get("subtab_Injury", 0)

        cols = st.columns(len(main_tab_names))
        for idx, (col, name) in enumerate(zip(cols, main_tab_names)):
            with col:
                is_active = idx == current_main_idx
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    name,
                    key=f"injury_main_{idx}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    if not is_active:
                        st.session_state["subtab_Injury"] = idx
                        st.rerun()

        # Render only the active subtab (lazy loading)
        if current_main_idx == 0:
            self.weekly_viewer.display(merged_data)
        elif current_main_idx == 1:
            self.season_viewer.display(merged_data)
        elif current_main_idx == 2:
            self.career_viewer.display(merged_data)
