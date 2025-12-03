import streamlit as st
import random
from datetime import datetime
from ...shared.modern_styles import apply_modern_styles
from ..shared.filters import render_filter_ui, apply_filters_with_loading
from ..shared.fun_facts import get_career_fun_facts
from ..shared.config import VIEW_DESCRIPTIONS, MATCHUP_TAB_NAMES
from ..shared.theme import apply_theme_styles, render_info_box
from .career_head_to_head_overview import CareerHeadToHeadViewer
from .career_team_ratings import CareerTeamRatingsViewer
from .career_optimal_lineups import display_alltime_optimal_lineup


class CareerMatchupOverviewViewer:
    def __init__(self, matchup_df):
        self.matchup_df = matchup_df
        self.data_loaded_at = datetime.now()

    @st.fragment
    def display(self, prefix=""):
        """Display career/all-time matchup data with shared components."""
        # Apply styles
        apply_modern_styles()
        apply_theme_styles()

        if self.matchup_df is None or self.matchup_df.empty:
            st.info("No data available")
            return

        # Generate fun facts
        fun_facts = get_career_fun_facts(self.matchup_df)

        # Render filter UI (with positions for career view)
        filters = render_filter_ui(
            df=self.matchup_df, prefix=prefix, show_weeks=False, show_positions=True
        )

        # Apply filters with loading indicator
        filtered_df = apply_filters_with_loading(self.matchup_df, filters)

        # Check if we have data after filtering
        if filtered_df.empty:
            st.warning(
                "⚠️ No matchups found with the selected filters. Try adjusting your filter criteria."
            )
            return

        # Create tabs
        tabs = st.tabs(MATCHUP_TAB_NAMES)

        # --- Data Tabs ---
        # Career/all-time uses lazy imports for stats viewers
        tab_keys = [
            "matchup_stats",
            "advanced_stats",
            "projected_stats",
            "team_ratings",
            "optimal_lineups",
            "head_to_head",
        ]

        for tab_idx, tab_key in enumerate(tab_keys, start=0):
            with tabs[tab_idx]:
                try:
                    if tab_key == "matchup_stats":
                        from .career_matchup_stats import CareerMatchupStatsViewer

                        viewer = CareerMatchupStatsViewer(filtered_df)
                        viewer.display(prefix=f"{prefix}_{tab_key}")

                    elif tab_key == "advanced_stats":
                        from .career_advanced_stats import (
                            SeasonAdvancedStatsViewer as CareerAdvancedStatsViewer,
                        )

                        viewer = CareerAdvancedStatsViewer(filtered_df)
                        viewer.display(prefix=f"{prefix}_{tab_key}")

                    elif tab_key == "projected_stats":
                        from .career_projected_stats import (
                            SeasonProjectedStatsViewer as CareerProjectedStatsViewer,
                        )

                        viewer = CareerProjectedStatsViewer(filtered_df)
                        viewer.display(prefix=f"{prefix}_{tab_key}")

                    elif tab_key == "optimal_lineups":
                        display_alltime_optimal_lineup(filtered_df)

                    elif tab_key == "team_ratings":
                        viewer = CareerTeamRatingsViewer(filtered_df)
                        viewer.display(prefix=f"{prefix}_{tab_key}")

                    elif tab_key == "head_to_head":
                        head_to_head_viewer = CareerHeadToHeadViewer(filtered_df)
                        head_to_head_viewer.display()

                except ImportError as e:
                    st.error(f"❌ Failed to load {tab_key} component: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error displaying {tab_key}: {str(e)}")

        # --- About Tab (Last) ---
        with tabs[6]:
            view_desc = VIEW_DESCRIPTIONS["career"]
            st.title(view_desc["title"])
            st.markdown(f"### {view_desc['subtitle']}")
            st.markdown("**What can you do here?**")

            # Display features
            for tab_name, description in view_desc["features"].items():
                st.markdown(f"- **{tab_name}:** {description}")

            st.markdown("---")

            # Display a random fun fact
            st.info(f"💡 **Fun Fact:** {random.choice(fun_facts)}")

            # Success message
            render_info_box(
                "<strong>Tip:</strong> Use the filter options above to customize your view, then explore the tabs!",
                icon="👉",
            )
