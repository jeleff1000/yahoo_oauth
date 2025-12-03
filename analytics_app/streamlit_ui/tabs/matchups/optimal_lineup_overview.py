import streamlit as st
from .weekly.weekly_optimal_lineups import display_weekly_optimal_lineup
from .season.season_optimal_lineups import display_season_optimal_lineup
from .all_time.career_optimal_lineups import display_alltime_optimal_lineup

@st.fragment
def display_optimal_lineup(player_df, matchup_data):
    sub_tab_names = ["Weekly", "Season", "Career"]
    sub_tabs = st.tabs(sub_tab_names)

    for i, sub_tab_name in enumerate(sub_tab_names):
        with sub_tabs[i]:
            st.subheader(sub_tab_name)
            if sub_tab_name == "Weekly":
                display_weekly_optimal_lineup(matchup_data)
            elif sub_tab_name == "Season":
                display_season_optimal_lineup(matchup_data)
            elif sub_tab_name == "Career":
                display_alltime_optimal_lineup(matchup_data)
