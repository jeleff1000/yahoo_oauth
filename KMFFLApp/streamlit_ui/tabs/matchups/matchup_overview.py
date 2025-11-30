# streamlit_ui/tabs/matchup_data_and_simulations/matchups/matchup_overview.py

from .weekly.weekly_matchup_overview import WeeklyMatchupDataViewer
from .season.season_matchup_overview import SeasonMatchupOverviewViewer
from .all_time.career_matchup_overview import CareerMatchupOverviewViewer

def display_matchup_overview(df_dict, prefix=""):
    import streamlit as st
    import logging
    from ..shared.modern_styles import apply_modern_styles
    from .shared.theme import apply_theme_styles

    # Configure logger for this module
    logger = logging.getLogger(__name__)

    apply_modern_styles()
    apply_theme_styles()

    st.markdown("""
    <div class="hero-section">
    <h2>Manager Matchups</h2>
    <p style="margin: 0.5rem 0 0 0;">Analyze head-to-head performance and matchup history.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get subtab from session state (controlled by hamburger menu)
    subtab_idx = st.session_state.get("subtab_Managers", 0)
    sub_tab_names = ["Weekly", "Seasons", "Career", "Visualize"]
    sub_tab_name = sub_tab_names[subtab_idx] if subtab_idx < len(sub_tab_names) else "Weekly"

    # Use "recent" instead of "matchup" based on load_managers_data()
    matchup_data = df_dict.get("recent")

    if matchup_data is None:
        st.error(f"Matchup Data not found.")
        return

    # Render only the active subtab (lazy loading)
    if sub_tab_name == "Weekly":
        matchup_data_viewer = WeeklyMatchupDataViewer(matchup_data)
        matchup_data_viewer.display(prefix=f"{prefix}_weekly_matchup_data")

    elif sub_tab_name == "Seasons":
        matchup_data_viewer = SeasonMatchupOverviewViewer(matchup_data)
        matchup_data_viewer.display(prefix=f"{prefix}_season_matchup_data")

    elif sub_tab_name == "Career":
        career_matchup_viewer = CareerMatchupOverviewViewer(matchup_data)
        career_matchup_viewer.display(prefix=f"{prefix}_career_matchup_data")

    elif sub_tab_name == "Visualize":
        # Render manager graphs inline with improved error handling
        st.markdown('*Visual analytics for managers - explore scoring patterns, performance, and efficiency*')

        # Import graph functions with specific error handling
        graph_imports = {}

        try:
            from .graphs.scoring_trends import display_scoring_trends
            graph_imports['scoring_trends'] = display_scoring_trends
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Scoring trends graph unavailable: {e}")

        try:
            from .graphs.win_percentage_graph import display_win_percentage_graph
            graph_imports['win_percentage'] = display_win_percentage_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Win percentage graph unavailable: {e}")

        try:
            from .graphs.power_rating import display_power_rating_graph
            graph_imports['power_rating'] = display_power_rating_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Power rating graph unavailable: {e}")

        try:
            from .graphs.scoring_distribution import display_scoring_distribution_graph
            graph_imports['scoring_distribution'] = display_scoring_distribution_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Scoring distribution graph unavailable: {e}")

        try:
            from .graphs.margin_of_victory import display_margin_of_victory_graph
            graph_imports['margin_of_victory'] = display_margin_of_victory_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Margin of victory graph unavailable: {e}")

        try:
            from .graphs.optimal_lineup_efficiency import display_optimal_lineup_efficiency_graph
            graph_imports['lineup_efficiency'] = display_optimal_lineup_efficiency_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Lineup efficiency graph unavailable: {e}")

        try:
            from .graphs.playoff_vs_regular import display_playoff_vs_regular_graph
            graph_imports['playoff_vs_regular'] = display_playoff_vs_regular_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Playoff vs regular graph unavailable: {e}")

        try:
            from .graphs.strength_of_schedule import display_strength_of_schedule_graph
            graph_imports['strength_of_schedule'] = display_strength_of_schedule_graph
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Strength of schedule graph unavailable: {e}")

        # Only show graph tabs if at least one graph is available
        if graph_imports:
            # Provide subtabs for all available graphs
            graph_tabs = st.tabs([
                "Scoring Trends",
                "Win Percentage",
                "Power Rating",
                "Score Distribution",
                "Margin of Victory",
                "Lineup Efficiency",
                "Playoff Performance",
                "Strength of Schedule"
            ])

            # Many graph functions accept a df_dict or load their own data
            df_wrapper = {"matchups": matchup_data, "recent": matchup_data}

            # Scoring Trends (replaces weekly + all-time scoring)
            with graph_tabs[0]:
                if 'scoring_trends' in graph_imports:
                    try:
                        graph_imports['scoring_trends'](df_wrapper, prefix=f"{prefix}_scoring_trends")
                    except Exception as e:
                        logger.error(f"Error displaying scoring trends graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Scoring Trends graph is not available")

            # Win Percentage (now includes win/loss records tab)
            with graph_tabs[1]:
                if 'win_percentage' in graph_imports:
                    try:
                        graph_imports['win_percentage'](df_wrapper, prefix=f"{prefix}_win_pct")
                    except Exception as e:
                        logger.error(f"Error displaying win percentage graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Win Percentage graph is not available")

            # Power Rating
            with graph_tabs[2]:
                if 'power_rating' in graph_imports:
                    try:
                        graph_imports['power_rating'](df_wrapper, prefix=f"{prefix}_power_rating")
                    except Exception as e:
                        logger.error(f"Error displaying power rating graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Power Rating graph is not available")

            # Score Distribution
            with graph_tabs[3]:
                if 'scoring_distribution' in graph_imports:
                    try:
                        graph_imports['scoring_distribution'](prefix=f"{prefix}_score_dist")
                    except Exception as e:
                        logger.error(f"Error displaying scoring distribution graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Score Distribution graph is not available")

            # Margin of Victory
            with graph_tabs[4]:
                if 'margin_of_victory' in graph_imports:
                    try:
                        graph_imports['margin_of_victory'](df_wrapper, prefix=f"{prefix}_margin")
                    except Exception as e:
                        logger.error(f"Error displaying margin of victory graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Margin of Victory graph is not available")

            # Lineup Efficiency
            with graph_tabs[5]:
                if 'lineup_efficiency' in graph_imports:
                    try:
                        graph_imports['lineup_efficiency'](df_wrapper, prefix=f"{prefix}_efficiency")
                    except Exception as e:
                        logger.error(f"Error displaying lineup efficiency graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Lineup Efficiency graph is not available")

            # Playoff vs Regular
            with graph_tabs[6]:
                if 'playoff_vs_regular' in graph_imports:
                    try:
                        graph_imports['playoff_vs_regular'](df_wrapper, prefix=f"{prefix}_playoff")
                    except Exception as e:
                        logger.error(f"Error displaying playoff performance graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Playoff Performance graph is not available")

            # Strength of Schedule
            with graph_tabs[7]:
                if 'strength_of_schedule' in graph_imports:
                    try:
                        graph_imports['strength_of_schedule'](df_wrapper, prefix=f"{prefix}_sos")
                    except Exception as e:
                        logger.error(f"Error displaying strength of schedule graph: {e}", exc_info=True)
                        st.error(f"Error displaying graph: {str(e)}")
                else:
                    st.warning("Strength of Schedule graph is not available")
        else:
            st.warning("Manager graphs are currently unavailable. Some graph modules could not be loaded.")
