import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class WeeklyOptimalLineupsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced weekly optimal lineups with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown(
            """
        <div class="tab-header">
        <h2>🎯 Weekly Optimal Lineups</h2>
        <p>Lineup efficiency analysis comparing actual scores to optimal possible lineups</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Check for required columns
        required_columns = [
            "manager",
            "opponent",
            "week",
            "year",
            "team_points",
            "opponent_points",
            "optimal_points",
            "win",
            "margin",
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
                file_name=f"weekly_optimal_lineups_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True,
            )

    def _prepare_display_data(self) -> pd.DataFrame:
        """Prepare and format data for display."""
        df = self.df.copy()

        # Filter to records with valid optimal_points
        df = df[df["optimal_points"].notna() & (df["optimal_points"] > 0)].copy()

        # Calculate derived metrics
        df["bench_points"] = df["optimal_points"] - df["team_points"]
        df["lineup_efficiency"] = (
            df["team_points"] / df["optimal_points"] * 100
        ).round(1)

        # Calculate what result would have been with optimal lineup
        df["optimal_margin"] = df["optimal_points"] - df["opponent_points"]
        df["optimal_win"] = df["optimal_margin"] > 0

        # Did optimal lineup change the outcome?
        df["outcome_change"] = df["win"] != df["optimal_win"]

        # Select columns to display
        columns_to_show = [
            "year",
            "week",
            "manager",
            "opponent",
            # Actual performance
            "team_points",
            "opponent_points",
            "win",
            "margin",
            # Optimal performance
            "optimal_points",
            "bench_points",
            "lineup_efficiency",
            # What if analysis
            "optimal_win",
            "optimal_margin",
            "outcome_change",
            # Context
            "playoff_round",
        ]

        # Filter to available columns
        available_cols = [col for col in columns_to_show if col in df.columns]
        display_df = df[available_cols].copy()

        # Format and rename columns
        display_df = display_df.rename(
            columns={
                "year": "Year",
                "week": "Week",
                "manager": "Manager",
                "opponent": "Opponent",
                # Actual
                "team_points": "Actual",
                "opponent_points": "Opp Pts",
                "win": "Result",
                "margin": "Margin",
                # Optimal
                "optimal_points": "Optimal",
                "bench_points": "Bench Pts",
                "lineup_efficiency": "Efficiency",
                # What if
                "optimal_win": "Opt Result",
                "optimal_margin": "Opt Margin",
                "outcome_change": "Changed",
                # Context
                "playoff_round": "Round",
            }
        )

        # Format numeric columns
        numeric_cols = [
            "Actual",
            "Opp Pts",
            "Margin",
            "Optimal",
            "Bench Pts",
            "Efficiency",
            "Opt Margin",
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

        # Sort by most recent first
        display_df["Year"] = display_df["Year"].astype(int)
        display_df["Week"] = display_df["Week"].astype(int)
        display_df = display_df.sort_values(
            by=["Year", "Week"], ascending=[False, False]
        ).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_games = len(df)
        if total_games == 0:
            return {}

        # Win stats
        actual_wins = df["Result"].sum()
        optimal_wins = df["Opt Result"].sum()

        # Efficiency stats
        avg_efficiency = df["Efficiency"].mean()
        perfect_lineups = len(df[df["Efficiency"] >= 98.0])
        avg_bench = df["Bench Pts"].mean()
        total_bench = df["Bench Pts"].sum()

        # Outcome changes
        would_have_won = len(df[(~df["Result"]) & (df["Opt Result"])])
        lucky_wins = len(df[(df["Result"]) & (df["Efficiency"] < 80)])

        # Points
        avg_actual = df["Actual"].mean()
        avg_optimal = df["Optimal"].mean()
        avg_lost_points = df["Bench Pts"].mean()

        # Efficiency distribution
        high_efficiency = len(df[df["Efficiency"] >= 90])
        medium_efficiency = len(df[(df["Efficiency"] >= 80) & (df["Efficiency"] < 90)])
        low_efficiency = len(df[df["Efficiency"] < 80])

        return {
            "total_games": total_games,
            "actual_wins": int(actual_wins),
            "optimal_wins": int(optimal_wins),
            "avg_efficiency": avg_efficiency,
            "perfect_lineups": perfect_lineups,
            "avg_bench": avg_bench,
            "total_bench": total_bench,
            "would_have_won": would_have_won,
            "lucky_wins": lucky_wins,
            "avg_actual": avg_actual,
            "avg_optimal": avg_optimal,
            "avg_lost_points": avg_lost_points,
            "high_efficiency": high_efficiency,
            "medium_efficiency": medium_efficiency,
            "low_efficiency": low_efficiency,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str):
        """Render enhanced table with column configuration."""

        # Create display dataframe with formatted values
        display_df = df.copy()

        # Format boolean columns - using quieter indicators
        display_df["Result"] = display_df["Result"].apply(lambda x: "W" if x else "L")

        display_df["Opt Result"] = display_df["Opt Result"].apply(
            lambda x: "W" if x else "L"
        )

        display_df["Changed"] = display_df["Changed"].apply(
            lambda x: "Yes" if x else ""
        )

        # Format efficiency as percentage
        if "Efficiency" in display_df.columns:
            display_df["Efficiency"] = display_df["Efficiency"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else ""
            )

        # Format playoff round
        if "Round" in display_df.columns:
            display_df["Round"] = (
                display_df["Round"]
                .fillna("")
                .apply(lambda x: f"🏆 {x.title()}" if x else "")
            )

        # Configure column display
        column_config = {
            "Year": st.column_config.NumberColumn(
                "Year", help="Season year", format="%d", width="small"
            ),
            "Week": st.column_config.NumberColumn(
                "Week", help="Week number", format="%d", width="small"
            ),
            "Manager": st.column_config.TextColumn(
                "Manager", help="Manager name", width="medium"
            ),
            "Opponent": st.column_config.TextColumn(
                "Opponent", help="Opponent name", width="medium"
            ),
            "Actual": st.column_config.NumberColumn(
                "Actual", help="Actual points scored", format="%.2f", width="small"
            ),
            "Opp Pts": st.column_config.NumberColumn(
                "Opp Pts", help="Opponent points", format="%.2f", width="small"
            ),
            "Result": st.column_config.TextColumn(
                "Result", help="Actual game result", width="small"
            ),
            "Margin": st.column_config.NumberColumn(
                "Margin", help="Actual point differential", format="%.2f", width="small"
            ),
            "Optimal": st.column_config.NumberColumn(
                "Optimal",
                help="Optimal lineup points (best possible)",
                format="%.2f",
                width="small",
            ),
            "Bench Pts": st.column_config.NumberColumn(
                "Bench Pts",
                help="Points left on bench (Optimal - Actual)",
                format="%.2f",
                width="small",
            ),
            "Efficiency": st.column_config.TextColumn(
                "Efficiency",
                help="Lineup efficiency % (Actual / Optimal)",
                width="small",
            ),
            "Opt Result": st.column_config.TextColumn(
                "Opt Result", help="Result with optimal lineup", width="small"
            ),
            "Opt Margin": st.column_config.NumberColumn(
                "Opt Margin", help="Optimal lineup margin", format="%.2f", width="small"
            ),
            "Changed": st.column_config.TextColumn(
                "Changed",
                help="Would optimal lineup have changed outcome?",
                width="small",
            ),
            "Round": st.column_config.TextColumn(
                "Round", help="Playoff round (if applicable)", width="medium"
            ),
        }

        # Display the enhanced dataframe
        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500,  # Fixed height for better scrolling
        )

    def _render_quick_stats(self, stats: dict):
        """Render quick statistics cards."""
        if not stats:
            return

        st.markdown("### 📈 Quick Stats")

        # Row 1: Core Efficiency Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total matchups analyzed",
            )

        with col2:
            st.metric(
                "Avg Efficiency",
                f"{stats['avg_efficiency']:.1f}%",
                help="Average lineup efficiency percentage",
            )

        with col3:
            st.metric(
                "Perfect Lineups",
                f"{stats['perfect_lineups']}",
                delta=f"{stats['perfect_lineups']/stats['total_games']*100:.1f}%",
                help="Lineups with 98%+ efficiency",
            )

        with col4:
            st.metric(
                "Avg Bench Pts",
                f"{stats['avg_bench']:.1f}",
                help="Average points left on bench",
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
                f"{stats['would_have_won']}",
                help="Losses that would have been wins with optimal lineup",
            )

        with col8:
            st.metric(
                "Lucky Wins",
                f"{stats['lucky_wins']}",
                help="Wins despite poor lineup efficiency (<80%)",
            )

        # Row 3: Points & Distribution
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Avg Actual",
                f"{stats['avg_actual']:.1f}",
                help="Average actual points per game",
            )

        with col10:
            st.metric(
                "Avg Optimal",
                f"{stats['avg_optimal']:.1f}",
                delta=f"+{stats['avg_optimal'] - stats['avg_actual']:.1f}",
                help="Average optimal points per game",
            )

        with col11:
            st.metric(
                "Total Left on Bench",
                f"{stats['total_bench']:.0f}",
                help="Total points left on bench across all games",
            )

        with col12:
            efficiency_breakdown = f"{stats['high_efficiency']}H / {stats['medium_efficiency']}M / {stats['low_efficiency']}L"
            st.metric(
                "Efficiency Dist",
                efficiency_breakdown,
                help="High (≥90%) / Medium (80-90%) / Low (<80%)",
            )


# Legacy function wrapper for backward compatibility
@st.fragment
def display_weekly_optimal_lineup(matchup_df: pd.DataFrame):
    """Legacy function wrapper - calls new class-based viewer."""
    viewer = WeeklyOptimalLineupsViewer(matchup_df)
    viewer.display(prefix="weekly_optimal")
