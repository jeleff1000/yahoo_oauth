import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class SeasonOptimalLineupsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced season optimal lineups with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown(
            """
        <div class="tab-header">
        <h2>🎯 Season Optimal Lineups</h2>
        <p>Lineup efficiency analysis aggregated by season</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Check for required columns
        required_columns = [
            "manager",
            "year",
            "team_points",
            "opponent_points",
            "optimal_points",
            "win",
        ]

        if not all(col in self.df.columns for col in required_columns):
            st.error("❌ Some required columns are missing from the data.")
            return

        # Prepare data
        display_df = self._prepare_display_data()

        if display_df.empty:
            st.info("No optimal lineup data available with current filters")
            return

        # Calculate summary statistics
        stats = self._calculate_stats(display_df)

        # === ENHANCED TABLE DISPLAY ===
        self._render_enhanced_table(display_df, prefix)

        # === QUICK STATS SECTION (Below Table) ===
        st.markdown("---")
        self._render_quick_stats(stats)

        # === DOWNLOAD SECTION ===
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**💾 Export Data**")
        with col2:
            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"season_optimal_lineups_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True,
            )

    def _prepare_display_data(self) -> pd.DataFrame:
        """Prepare and format season-level data for display."""
        df = self.df.copy()

        # Filter to records with valid optimal_points
        df = df[df["optimal_points"].notna() & (df["optimal_points"] > 0)].copy()

        # Calculate derived metrics
        df["bench_points"] = df["optimal_points"] - df["team_points"]
        df["lineup_efficiency"] = df["team_points"] / df["optimal_points"] * 100

        # Calculate what result would have been with optimal lineup
        df["optimal_margin"] = df["optimal_points"] - df["opponent_points"]
        df["optimal_win"] = df["optimal_margin"] > 0

        # Did optimal lineup change the outcome?
        df["outcome_change"] = df["win"] != df["optimal_win"]

        # Low efficiency flag for Lucky Wins calculation
        df["low_efficiency"] = df["lineup_efficiency"] < 80

        # Aggregate by manager and year
        agg_dict = {
            "team_points": "sum",
            "optimal_points": "sum",
            "bench_points": "sum",
            "lineup_efficiency": "mean",
            "win": "sum",
            "optimal_win": "sum",
            "outcome_change": "sum",
            "optimal_margin": "sum",
            "low_efficiency": "sum",
        }

        display_df = df.groupby(["manager", "year"]).agg(agg_dict).reset_index()

        # Calculate additional season-level metrics
        display_df["games"] = df.groupby(["manager", "year"]).size().values
        display_df["losses"] = display_df["games"] - display_df["win"]
        display_df["optimal_losses"] = display_df["games"] - display_df["optimal_win"]

        # Calculate Lucky Wins: wins with low efficiency
        # Need to count wins where efficiency was low
        lucky_wins = (
            df[df["win"] & df["low_efficiency"]].groupby(["manager", "year"]).size()
        )
        display_df["lucky_wins"] = display_df.apply(
            lambda row: lucky_wins.get((row["manager"], row["year"]), 0), axis=1
        )

        # Calculate Missed Wins: losses that would have been wins
        missed_wins = (
            df[(~df["win"]) & df["optimal_win"]].groupby(["manager", "year"]).size()
        )
        display_df["missed_wins"] = display_df.apply(
            lambda row: missed_wins.get((row["manager"], row["year"]), 0), axis=1
        )

        # Rename columns for display
        display_df = display_df.rename(
            columns={
                "manager": "Manager",
                "year": "Year",
                "games": "Games",
                "win": "Wins",
                "losses": "Losses",
                "team_points": "Actual Pts",
                "optimal_points": "Optimal Pts",
                "bench_points": "Total Bench Pts",
                "lineup_efficiency": "Avg Efficiency",
                "optimal_win": "Optimal Wins",
                "optimal_losses": "Optimal Losses",
                "missed_wins": "Missed Wins",
                "lucky_wins": "Lucky Wins",
                "outcome_change": "Outcome Changes",
                "optimal_margin": "Optimal Margin",
            }
        )

        # Sort by most recent first
        display_df = display_df.sort_values(
            by=["Year", "Manager"], ascending=[False, True]
        ).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_seasons = len(df)
        if total_seasons == 0:
            return {}

        # Win stats
        actual_wins = df["Wins"].sum()
        optimal_wins = df["Optimal Wins"].sum()

        # Efficiency stats
        avg_efficiency = df["Avg Efficiency"].mean()
        high_efficiency_seasons = len(df[df["Avg Efficiency"] >= 90])
        avg_bench = df["Total Bench Pts"].sum() / df["Games"].sum()  # Per game
        total_bench = df["Total Bench Pts"].sum()

        # Outcome changes
        missed_wins = df["Missed Wins"].sum()
        lucky_wins = df["Lucky Wins"].sum()

        # Points
        total_games = df["Games"].sum()
        avg_actual = df["Actual Pts"].sum() / total_games
        avg_optimal = df["Optimal Pts"].sum() / total_games

        return {
            "total_seasons": total_seasons,
            "total_games": int(total_games),
            "actual_wins": int(actual_wins),
            "optimal_wins": int(optimal_wins),
            "avg_efficiency": avg_efficiency,
            "high_efficiency_seasons": high_efficiency_seasons,
            "avg_bench": avg_bench,
            "total_bench": total_bench,
            "missed_wins": int(missed_wins),
            "lucky_wins": int(lucky_wins),
            "avg_actual": avg_actual,
            "avg_optimal": avg_optimal,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str):
        """Render enhanced table with column configuration."""

        display_df = df.copy()

        # Configure column display
        column_config = {
            "Manager": st.column_config.TextColumn(
                "Manager", help="Manager name", width="medium"
            ),
            "Year": st.column_config.NumberColumn(
                "Year", help="Season year", format="%d", width="small"
            ),
            "Games": st.column_config.NumberColumn(
                "Games", help="Total games played", format="%d", width="small"
            ),
            "Wins": st.column_config.NumberColumn(
                "Wins", help="Actual wins", format="%d", width="small"
            ),
            "Losses": st.column_config.NumberColumn(
                "Losses", help="Actual losses", format="%d", width="small"
            ),
            "Actual Pts": st.column_config.NumberColumn(
                "Actual Pts",
                help="Total actual points scored",
                format="%.2f",
                width="small",
            ),
            "Optimal Pts": st.column_config.NumberColumn(
                "Optimal Pts",
                help="Total optimal lineup points",
                format="%.2f",
                width="small",
            ),
            "Total Bench Pts": st.column_config.NumberColumn(
                "Total Bench Pts",
                help="Total points left on bench",
                format="%.2f",
                width="small",
            ),
            "Avg Efficiency": st.column_config.NumberColumn(
                "Avg Efficiency",
                help="Average lineup efficiency %",
                format="%.1f%%",
                width="small",
            ),
            "Optimal Wins": st.column_config.NumberColumn(
                "Optimal Wins",
                help="Wins with optimal lineup",
                format="%d",
                width="small",
            ),
            "Optimal Losses": st.column_config.NumberColumn(
                "Optimal Losses",
                help="Losses with optimal lineup",
                format="%d",
                width="small",
            ),
            "Missed Wins": st.column_config.NumberColumn(
                "Missed Wins",
                help="Losses that would have been wins with optimal lineup",
                format="%d",
                width="small",
            ),
            "Lucky Wins": st.column_config.NumberColumn(
                "Lucky Wins",
                help="Wins despite poor lineup efficiency (<80%)",
                format="%d",
                width="small",
            ),
            "Outcome Changes": st.column_config.NumberColumn(
                "Outcome Changes",
                help="Games where optimal lineup changed outcome",
                format="%d",
                width="small",
            ),
            "Optimal Margin": st.column_config.NumberColumn(
                "Optimal Margin",
                help="Total optimal lineup margin",
                format="%.2f",
                width="small",
            ),
        }

        # Display the enhanced dataframe
        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500,
        )

    def _render_quick_stats(self, stats: dict):
        """Render quick statistics cards."""
        if not stats:
            return

        st.markdown("### 📈 Quick Stats")

        # Row 1: Core Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Seasons",
                f"{stats['total_seasons']:,}",
                help="Total seasons analyzed",
            )

        with col2:
            st.metric(
                "Total Games", f"{stats['total_games']:,}", help="Total games analyzed"
            )

        with col3:
            st.metric(
                "Avg Efficiency",
                f"{stats['avg_efficiency']:.1f}%",
                help="Average lineup efficiency percentage",
            )

        with col4:
            st.metric(
                "High Eff Seasons",
                f"{stats['high_efficiency_seasons']}",
                delta=f"{stats['high_efficiency_seasons']/stats['total_seasons']*100:.1f}%",
                help="Seasons with ≥90% avg efficiency",
            )

        # Row 2: Win Analysis
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Actual Wins", f"{stats['actual_wins']}", help="Wins with actual lineup"
            )

        with col6:
            st.metric(
                "Optimal Wins",
                f"{stats['optimal_wins']}",
                delta=f"+{stats['optimal_wins'] - stats['actual_wins']}",
                help="Wins with optimal lineup",
            )

        with col7:
            st.metric(
                "Missed Wins",
                f"{stats['missed_wins']}",
                help="Losses that would have been wins with optimal lineup",
            )

        with col8:
            st.metric(
                "Lucky Wins",
                f"{stats['lucky_wins']}",
                help="Wins despite poor lineup efficiency (<80%)",
            )

        # Row 3: Points Analysis
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Avg Actual/Game",
                f"{stats['avg_actual']:.1f}",
                help="Average actual points per game",
            )

        with col10:
            st.metric(
                "Avg Optimal/Game",
                f"{stats['avg_optimal']:.1f}",
                delta=f"+{stats['avg_optimal'] - stats['avg_actual']:.1f}",
                help="Average optimal points per game",
            )

        with col11:
            st.metric(
                "Avg Bench/Game",
                f"{stats['avg_bench']:.1f}",
                help="Average points left on bench per game",
            )

        with col12:
            st.metric(
                "Total Left on Bench",
                f"{stats['total_bench']:.0f}",
                help="Total points left on bench across all games",
            )


# Legacy function wrapper for backward compatibility
@st.fragment
def display_season_optimal_lineup(matchup_df: pd.DataFrame):
    """Legacy function wrapper - calls new class-based viewer."""
    viewer = SeasonOptimalLineupsViewer(matchup_df)
    viewer.display(prefix="season_optimal")
