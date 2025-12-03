"""
OPTIMIZED Schedules Viewer - Uses schedule table directly from MotherDuck

Key optimizations:
- Queries MotherDuck directly instead of passing large dataframes
- Uses persistent DuckDB connection
- Efficient SQL aggregations
- No intermediate pandas operations
- Theme-aware styling (light/dark mode support)
- Mobile responsive layouts
- Dynamic database selection (supports multiple leagues)

Source columns in schedule table:
is_playoffs, is_consolation, manager, team_name, manager_week, manager_year,
opponent, opponent_week, opponent_year, week, year, team_points, opponent_points,
win, loss
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure analytics_app directory is in path for imports
_app_dir = Path(__file__).parent.parent.parent.resolve()
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from md.core import T, run_query
from ..shared.modern_styles import apply_modern_styles
from shared.themes import detect_theme


class SchedulesViewer:
    def __init__(self):
        """No need to pass data - queries MotherDuck directly"""
        self.table = T["schedule"]

    def _get_regular_season_data(self, manager: str, year: str):
        """Get regular season schedule"""
        query = f"""
            SELECT
                week AS "Wk",
                CASE
                    WHEN win = 1 THEN 'W'
                    WHEN loss = 1 THEN 'L'
                    ELSE '-'
                END AS "Result",
                opponent AS "Opponent",
                ROUND(CAST(team_points AS DOUBLE), 2) AS "PF",
                ROUND(CAST(opponent_points AS DOUBLE), 2) AS "PA",
                ROUND(CAST(team_points AS DOUBLE) - CAST(opponent_points AS DOUBLE), 2) AS "Margin"
            FROM {self.table}
            WHERE manager = '{manager.replace("'", "''")}'
              AND CAST(year AS VARCHAR) = '{year}'
              AND COALESCE(is_playoffs, 0) = 0
              AND COALESCE(is_consolation, 0) = 0
            ORDER BY week NULLS LAST
        """
        return run_query(query)

    def _get_postseason_data(self, manager: str, year: str):
        """Get postseason schedule (playoffs + consolation) with round info"""
        query = f"""
            SELECT
                week AS "Wk",
                CASE
                    WHEN COALESCE(is_playoffs, 0) = 1 THEN
                        CASE
                            WHEN playoff_round = 'quarterfinal' THEN 'Quarterfinal'
                            WHEN playoff_round = 'semifinal' THEN 'Semifinal'
                            WHEN playoff_round = 'championship' THEN 'Championship'
                            ELSE COALESCE(playoff_round, 'Playoffs')
                        END
                    WHEN COALESCE(is_consolation, 0) = 1 THEN
                        CASE
                            WHEN consolation_round = 'consolation_semifinal' THEN 'Consolation Semi'
                            WHEN consolation_round = 'third_place_game' THEN '3rd Place'
                            WHEN consolation_round = 'fifth_place_game' THEN '5th Place'
                            WHEN consolation_round = 'seventh_place_game' THEN '7th Place'
                            WHEN consolation_round = 'ninth_place_game' THEN '9th Place'
                            ELSE COALESCE(consolation_round, 'Consolation')
                        END
                    ELSE 'Postseason'
                END AS "Round",
                CASE
                    WHEN win = 1 THEN 'W'
                    WHEN loss = 1 THEN 'L'
                    ELSE '-'
                END AS "Result",
                opponent AS "Opponent",
                ROUND(CAST(team_points AS DOUBLE), 2) AS "PF",
                ROUND(CAST(opponent_points AS DOUBLE), 2) AS "PA",
                ROUND(CAST(team_points AS DOUBLE) - CAST(opponent_points AS DOUBLE), 2) AS "Margin"
            FROM {self.table}
            WHERE manager = '{manager.replace("'", "''")}'
              AND CAST(year AS VARCHAR) = '{year}'
              AND (COALESCE(is_playoffs, 0) = 1 OR COALESCE(is_consolation, 0) = 1)
            ORDER BY week NULLS LAST
        """
        return run_query(query)

    def _get_summary_data(self, manager: str, year: str, section: str):
        """Get aggregated summary for a manager/year/section"""
        if section == "regular":
            section_filter = (
                "COALESCE(is_playoffs, 0) = 0 AND COALESCE(is_consolation, 0) = 0"
            )
        else:  # postseason
            section_filter = (
                "(COALESCE(is_playoffs, 0) = 1 OR COALESCE(is_consolation, 0) = 1)"
            )

        # Only count played games for PPG (where win or loss = 1)
        query = f"""
            SELECT
                ROUND(SUM(CAST(team_points AS DOUBLE)), 2) AS PF,
                ROUND(SUM(CAST(opponent_points AS DOUBLE)), 2) AS PA,
                ROUND(
                    SUM(CAST(team_points AS DOUBLE)) /
                    NULLIF(SUM(CASE WHEN win = 1 OR loss = 1 THEN 1 ELSE 0 END), 0),
                    2
                ) AS PPG,
                SUM(COALESCE(win, 0)) AS "Win",
                SUM(COALESCE(loss, 0)) AS "Loss"
            FROM {self.table}
            WHERE manager = '{manager.replace("'", "''")}'
              AND CAST(year AS VARCHAR) = '{year}'
              AND {section_filter}
        """

        return run_query(query)

    def _get_managers(self):
        """Get list of managers"""
        query = f"""
            SELECT DISTINCT manager 
            FROM {self.table} 
            WHERE manager IS NOT NULL 
            ORDER BY manager
        """
        df = run_query(query)
        return df["manager"].tolist() if not df.empty else []

    def _get_years(self, manager: str):
        """Get years for a specific manager"""
        query = f"""
            SELECT DISTINCT CAST(year AS VARCHAR) AS year
            FROM {self.table}
            WHERE manager = '{manager.replace("'", "''")}'
            ORDER BY year DESC
        """
        df = run_query(query)
        return df["year"].tolist() if not df.empty else []

    @st.fragment
    def display(self, prefix: str = "schedules"):
        """Display the schedules UI"""
        # Apply shared modern styles (includes CSS variables)
        apply_modern_styles()

        # Add theme-aware CSS using CSS variables
        st.markdown(
            """
            <style>
            /* Schedule section styling - uses CSS variables for theming */
            .schedule-section {
                background: var(--bg-primary, #ffffff);
                border-radius: var(--radius-md, 8px);
                padding: var(--space-lg, 1.5rem);
                margin: var(--space-md, 1rem) 0;
                border: 1px solid var(--border, #e0e0e0);
                /* NO shadow for static display */
            }

            .schedule-header {
                color: var(--text-primary, #333333);
                padding-bottom: var(--space-sm, 0.5rem);
                border-bottom: 2px solid var(--accent, #667eea);
                margin-bottom: var(--space-md, 1rem);
                font-weight: 600;
                font-size: 1.1rem;
            }

            /* Metric cards styling - static, no heavy shadows */
            div[data-testid="stMetric"] {
                background: var(--bg-secondary, #f8f9fa);
                padding: 0.75rem;
                border-radius: var(--radius-md, 8px);
                border: 1px solid var(--border, #e0e0e0);
            }

            /* Win/Loss cell styling */
            .win-cell {
                color: var(--success, #4CAF50);
                font-weight: bold;
            }
            .loss-cell {
                color: var(--error, #f44336);
                font-weight: bold;
            }

            /* Mobile responsive */
            @media (max-width: 768px) {
                .schedule-section {
                    padding: var(--space-md, 1rem);
                    margin: var(--space-sm, 0.5rem) 0;
                }
                .schedule-header {
                    font-size: 1rem;
                }
            }

            @media (max-width: 480px) {
                .schedule-header {
                    font-size: 0.9rem;
                }
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # Get managers
        managers = self._get_managers()
        if not managers:
            st.info("No managers found in schedule data.")
            return

        # Manager selector with better layout
        st.markdown("### Select Team & Season")
        col1, col2 = st.columns(2)
        with col1:
            selected_manager = st.selectbox(
                "Manager", managers, index=0, key=f"{prefix}_manager"
            )

        # Get years for selected manager
        years = self._get_years(selected_manager)
        if not years:
            st.info("No seasons found for the selected manager.")
            return

        with col2:
            selected_year = st.selectbox(
                "Year",
                years,
                index=0,  # Most recent year first (already sorted DESC)
                key=f"{prefix}_year",
            )

        st.markdown(f"### {selected_manager}'s {selected_year} Season")

        # Get theme-aware colors for pandas styling (requires actual values, not CSS vars)
        is_dark = detect_theme() == "dark"
        colors = {
            "win": "#10B981",  # success green
            "loss": "#EF4444",  # error red
            "text_muted": "#9CA3AF" if is_dark else "#6B7280",
        }

        # Helper function to style schedule rows
        def style_schedule_row(row):
            n_cols = len(row)
            styles = [""] * n_cols

            result = row.get("Result", "")
            margin = row.get("Margin", 0) if row.get("Margin") is not None else 0

            # Style Result column
            if "Result" in row.index:
                result_idx = list(row.index).index("Result")
                if result == "W":
                    styles[result_idx] = f'color: {colors["win"]}; font-weight: bold;'
                elif result == "L":
                    styles[result_idx] = f'color: {colors["loss"]}; font-weight: bold;'

            # Style Margin column
            if "Margin" in row.index:
                margin_idx = list(row.index).index("Margin")
                try:
                    margin_val = float(margin) if margin is not None else 0
                    if margin_val > 0:
                        styles[margin_idx] = (
                            f'color: {colors["win"]}; font-weight: bold;'
                        )
                    elif margin_val < 0:
                        styles[margin_idx] = (
                            f'color: {colors["loss"]}; font-weight: bold;'
                        )
                except (ValueError, TypeError):
                    pass

            return styles

        # Helper function to display summary metrics
        def display_summary(section_key: str):
            summary_df = self._get_summary_data(
                selected_manager, selected_year, section_key
            )

            if summary_df is not None and not summary_df.empty:
                try:
                    wins = (
                        int(summary_df["Win"].iloc[0])
                        if "Win" in summary_df.columns
                        else 0
                    )
                    losses = (
                        int(summary_df["Loss"].iloc[0])
                        if "Loss" in summary_df.columns
                        else 0
                    )
                    pf = (
                        float(summary_df["PF"].iloc[0])
                        if "PF" in summary_df.columns and summary_df["PF"].iloc[0]
                        else 0
                    )
                    pa = (
                        float(summary_df["PA"].iloc[0])
                        if "PA" in summary_df.columns and summary_df["PA"].iloc[0]
                        else 0
                    )
                    ppg = (
                        float(summary_df["PPG"].iloc[0])
                        if "PPG" in summary_df.columns and summary_df["PPG"].iloc[0]
                        else 0
                    )
                    point_diff = pf - pa

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        record_color = (
                            colors["win"]
                            if wins > losses
                            else (
                                colors["loss"]
                                if losses > wins
                                else colors["text_muted"]
                            )
                        )
                        st.markdown(
                            f"""
                            <div style="text-align: center; padding: 0.5rem;">
                                <div style="font-size: 0.85rem; color: var(--text-muted, {colors['text_muted']});">Record</div>
                                <div style="font-size: 1.8rem; font-weight: bold; color: {record_color};">{wins}-{losses}</div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.metric("PPG", f"{ppg:.2f}")
                    with col3:
                        st.metric("Total PF", f"{pf:.2f}")
                    with col4:
                        st.metric("Point Diff", f"{point_diff:+.2f}")
                except Exception:
                    pass

        # Create tabs for Regular Season and Postseason
        tab_regular, tab_postseason = st.tabs(["Regular Season", "Postseason"])

        with tab_regular:
            schedule_df = self._get_regular_season_data(selected_manager, selected_year)

            if schedule_df is None or schedule_df.empty:
                st.info("No regular season games found.")
            else:
                styled_df = schedule_df.style.apply(style_schedule_row, axis=1).format(
                    {"PF": "{:.2f}", "PA": "{:.2f}", "Margin": "{:+.2f}"}, na_rep="-"
                )

                st.dataframe(
                    styled_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(500, len(schedule_df) * 35 + 38),
                )

                display_summary("regular")

        with tab_postseason:
            schedule_df = self._get_postseason_data(selected_manager, selected_year)

            if schedule_df is None or schedule_df.empty:
                st.info("No postseason games found.")
            else:
                styled_df = schedule_df.style.apply(style_schedule_row, axis=1).format(
                    {"PF": "{:.2f}", "PA": "{:.2f}", "Margin": "{:+.2f}"}, na_rep="-"
                )

                st.dataframe(
                    styled_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(300, len(schedule_df) * 35 + 38),
                )

                display_summary("postseason")


@st.fragment
def display_schedules(df_dict=None, prefix: str = "schedules"):
    """
    Display schedules using direct MotherDuck queries.
    df_dict parameter is kept for backward compatibility but not used.
    """
    viewer = SchedulesViewer()
    viewer.display(prefix=prefix)
