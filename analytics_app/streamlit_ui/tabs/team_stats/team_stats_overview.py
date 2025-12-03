#!/usr/bin/env python3
"""Team Stats Overview - Main entry point for team stats tab."""
import streamlit as st

from .weekly_team_stats import WeeklyTeamViewer
from .season_team_stats import SeasonTeamViewer
from .career_team_stats import CareerTeamViewer
from .team_stats_visualizations import TeamStatsVisualizer
from md.tab_data_access.team_stats import (
    load_weekly_team_data,
    load_season_team_data,
    load_career_team_data,
    load_weekly_team_data_by_manager,
    load_season_team_data_by_manager,
    load_career_team_data_by_manager,
    load_weekly_team_data_by_lineup_position,
    load_season_team_data_by_lineup_position,
    load_career_team_data_by_lineup_position,
)
from ..shared.modern_styles import apply_modern_styles
from .shared.theme import apply_theme_styles


@st.fragment
def display_team_stats_overview():
    """Main team stats display with Weekly, Season, and Career subtabs."""
    apply_modern_styles()
    apply_theme_styles()

    # Get subtab from session state (controlled by hamburger menu)
    subtab_idx = st.session_state.get("subtab_Team Stats", 0)
    sub_tab_names = ["Weekly", "Seasons", "Career", "Visualize"]
    sub_tab_name = sub_tab_names[subtab_idx] if subtab_idx < len(sub_tab_names) else "Weekly"

    # Render only the active subtab (lazy loading)
    if sub_tab_name == "Weekly":
        render_weekly_team()
    elif sub_tab_name == "Seasons":
        render_season_team()
    elif sub_tab_name == "Career":
        render_career_team()
    elif sub_tab_name == "Visualize":
        render_visualizations()


@st.fragment
def render_weekly_team():
    """Render weekly team stats."""
    # Get filter values from session state
    include_regular_season = st.session_state.get("weekly_team_include_regular_season", True)
    include_playoffs = st.session_state.get("weekly_team_include_playoffs", True)
    include_consolation = st.session_state.get("weekly_team_include_consolation", False)

    with st.spinner("Loading weekly team data..."):
        team_data_by_pos = load_weekly_team_data(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        team_data_by_mgr = load_weekly_team_data_by_manager(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        team_data_by_lineup_pos = load_weekly_team_data_by_lineup_position(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        if (team_data_by_pos is not None and not team_data_by_pos.empty) or \
           (team_data_by_mgr is not None and not team_data_by_mgr.empty) or \
           (team_data_by_lineup_pos is not None and not team_data_by_lineup_pos.empty):
            WeeklyTeamViewer(team_data_by_pos, team_data_by_mgr, team_data_by_lineup_pos).display()
        else:
            st.info("No weekly team data available")


@st.fragment
def render_season_team():
    """Render season team stats."""
    # Get filter values from session state
    include_regular_season = st.session_state.get("season_team_include_regular_season", True)
    include_playoffs = st.session_state.get("season_team_include_playoffs", True)
    include_consolation = st.session_state.get("season_team_include_consolation", False)

    with st.spinner("Loading season team data..."):
        team_data_by_pos = load_season_team_data(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        team_data_by_mgr = load_season_team_data_by_manager(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        team_data_by_lineup_pos = load_season_team_data_by_lineup_position(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        if (team_data_by_pos is not None and not team_data_by_pos.empty) or \
           (team_data_by_mgr is not None and not team_data_by_mgr.empty) or \
           (team_data_by_lineup_pos is not None and not team_data_by_lineup_pos.empty):
            SeasonTeamViewer(team_data_by_pos, team_data_by_mgr, team_data_by_lineup_pos).display()
        else:
            st.info("No season team data available")


@st.fragment
def render_career_team():
    """Render career team stats."""
    # Get filter values from session state
    include_regular_season = st.session_state.get("career_team_include_regular_season", True)
    include_playoffs = st.session_state.get("career_team_include_playoffs", True)
    include_consolation = st.session_state.get("career_team_include_consolation", False)

    with st.spinner("Loading career team data..."):
        team_data_by_pos = load_career_team_data(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        team_data_by_mgr = load_career_team_data_by_manager(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        team_data_by_lineup_pos = load_career_team_data_by_lineup_position(
            include_regular_season=include_regular_season,
            include_playoffs=include_playoffs,
            include_consolation=include_consolation
        )
        if (team_data_by_pos is not None and not team_data_by_pos.empty) or \
           (team_data_by_mgr is not None and not team_data_by_mgr.empty) or \
           (team_data_by_lineup_pos is not None and not team_data_by_lineup_pos.empty):
            CareerTeamViewer(team_data_by_pos, team_data_by_mgr, team_data_by_lineup_pos).display()
        else:
            st.info("No career team data available")


@st.fragment
def render_visualizations():
    """Render team stats visualizations with tabs for different time periods."""
    st.markdown('*Visual analytics for team stats - explore scoring patterns, position performance, and manager comparisons*')

    # Visualization type tabs
    viz_tabs = st.tabs(["Weekly", "Season", "Career"])

    with viz_tabs[0]:
        # Load weekly data for visualizations
        include_regular_season = st.session_state.get("weekly_team_include_regular_season", True)
        include_playoffs = st.session_state.get("weekly_team_include_playoffs", True)
        include_consolation = st.session_state.get("weekly_team_include_consolation", False)

        with st.spinner("Loading weekly data..."):
            team_data_by_pos = load_weekly_team_data(
                include_regular_season=include_regular_season,
                include_playoffs=include_playoffs,
                include_consolation=include_consolation
            )
            team_data_by_mgr = load_weekly_team_data_by_manager(
                include_regular_season=include_regular_season,
                include_playoffs=include_playoffs,
                include_consolation=include_consolation
            )

            if (team_data_by_pos is not None and not team_data_by_pos.empty) or \
               (team_data_by_mgr is not None and not team_data_by_mgr.empty):
                visualizer = TeamStatsVisualizer(team_data_by_pos, team_data_by_mgr)
                visualizer.display_weekly_visualizations()
            else:
                st.info("No weekly data available for visualizations")

    with viz_tabs[1]:
        # Load season data for visualizations
        include_regular_season = st.session_state.get("season_team_include_regular_season", True)
        include_playoffs = st.session_state.get("season_team_include_playoffs", True)
        include_consolation = st.session_state.get("season_team_include_consolation", False)

        with st.spinner("Loading season data..."):
            team_data_by_pos = load_season_team_data(
                include_regular_season=include_regular_season,
                include_playoffs=include_playoffs,
                include_consolation=include_consolation
            )
            team_data_by_mgr = load_season_team_data_by_manager(
                include_regular_season=include_regular_season,
                include_playoffs=include_playoffs,
                include_consolation=include_consolation
            )

            if (team_data_by_pos is not None and not team_data_by_pos.empty) or \
               (team_data_by_mgr is not None and not team_data_by_mgr.empty):
                visualizer = TeamStatsVisualizer(team_data_by_pos, team_data_by_mgr)
                visualizer.display_season_visualizations()
            else:
                st.info("No season data available for visualizations")

    with viz_tabs[2]:
        # Load career data for visualizations
        include_regular_season = st.session_state.get("career_team_include_regular_season", True)
        include_playoffs = st.session_state.get("career_team_include_playoffs", True)
        include_consolation = st.session_state.get("career_team_include_consolation", False)

        with st.spinner("Loading career data..."):
            team_data_by_pos = load_career_team_data(
                include_regular_season=include_regular_season,
                include_playoffs=include_playoffs,
                include_consolation=include_consolation
            )
            team_data_by_mgr = load_career_team_data_by_manager(
                include_regular_season=include_regular_season,
                include_playoffs=include_playoffs,
                include_consolation=include_consolation
            )

            if (team_data_by_pos is not None and not team_data_by_pos.empty) or \
               (team_data_by_mgr is not None and not team_data_by_mgr.empty):
                visualizer = TeamStatsVisualizer(team_data_by_pos, team_data_by_mgr)
                visualizer.display_career_visualizations()
            else:
                st.info("No career data available for visualizations")
