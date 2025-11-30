import pandas as pd
import streamlit as st
import random
from datetime import datetime
from ...shared.modern_styles import apply_modern_styles
from ..shared.filters import render_filter_ui, apply_filters_with_loading
from ..shared.fun_facts import get_weekly_fun_facts
from ..shared.config import VIEW_DESCRIPTIONS, MATCHUP_TAB_NAMES
from ..shared.theme import apply_theme_styles, render_info_box
from .weekly_advanced_stats import WeeklyAdvancedStatsViewer
from .weekly_matchup_stats import WeeklyMatchupStatsViewer
from .weekly_projected_stats import WeeklyProjectedStatsViewer
from .weekly_optimal_lineups import display_weekly_optimal_lineup
from .weekly_team_ratings import WeeklyTeamRatingsViewer
from .weekly_head_to_head import WeeklyHeadToHeadViewer


class WeeklyMatchupDataViewer:
    def __init__(self, matchup_df):
        self.matchup_df = matchup_df
        self.data_loaded_at = datetime.now()

    @st.fragment
    def display(self, prefix=""):
        """Display weekly matchup data with shared components."""
        # Apply styles
        apply_modern_styles()
        apply_theme_styles()

        if self.matchup_df is None or self.matchup_df.empty:
            st.info("No data available")
            return

        # Generate fun facts
        fun_facts = get_weekly_fun_facts(self.matchup_df)

        # Render filter UI with data freshness
        filters = render_filter_ui(
            df=self.matchup_df,
            prefix=prefix,
            show_weeks=True,
            show_positions=False,
            data_last_updated=self.data_loaded_at
        )

        # Apply filters with loading indicator
        filtered_df = apply_filters_with_loading(self.matchup_df, filters)

        # Check if we have data after filtering
        if filtered_df.empty:
            st.warning("⚠️ No matchups found with the selected filters. Try adjusting your filter criteria.")
            return

        # Create tabs
        tabs = st.tabs(MATCHUP_TAB_NAMES)

        # --- Data Tabs ---
        tab_viewers = {
            0: ('matchup_stats', WeeklyMatchupStatsViewer),
            1: ('advanced_stats', WeeklyAdvancedStatsViewer),
            2: ('projected_stats', WeeklyProjectedStatsViewer),
            3: ('team_ratings', WeeklyTeamRatingsViewer),
            4: ('optimal_lineups', None),  # Special handling
            5: ('head_to_head', WeeklyHeadToHeadViewer),
        }

        for tab_idx, (tab_key, viewer_class) in tab_viewers.items():
            with tabs[tab_idx]:
                try:
                    if tab_key == 'optimal_lineups':
                        # Special case: function instead of class
                        display_weekly_optimal_lineup(filtered_df)
                    elif viewer_class:
                        viewer = viewer_class(filtered_df)
                        viewer.display(prefix=f"{prefix}_{tab_key}")
                except Exception as e:
                    st.error(f"❌ Error displaying {tab_key}: {str(e)}")

        # --- About Tab (Last) ---
        with tabs[6]:
            view_desc = VIEW_DESCRIPTIONS['weekly']
            st.title(view_desc['title'])
            st.markdown(f"### {view_desc['subtitle']}")
            st.markdown("**What can you do here?**")

            # Display features
            for tab_name, description in view_desc['features'].items():
                st.markdown(f"- **{tab_name}:** {description}")

            st.markdown("---")

            # Display a random fun fact
            st.info(f"💡 **Fun Fact:** {random.choice(fun_facts)}")

            # Success message
            render_info_box(
                "<strong>Tip:</strong> Use the filter options above to customize your view, then explore the tabs!",
                icon="👉"
            )
