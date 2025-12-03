import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


@st.fragment
def display_alltime_optimal_lineup(matchup_df: pd.DataFrame):
    """Display enhanced career optimal lineup stats with improved UX."""
    apply_modern_styles()
    apply_theme_styles()

    # Header with description
    st.markdown(
        """
    <div class="tab-header">
    <h2>⚡ Career Optimal Lineups</h2>
    <p>Lifetime lineup efficiency, optimal performance, and what-if analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Required columns
    need_m = {
        "manager",
        "week",
        "year",
        "opponent",
        "team_points",
        "opponent_points",
        "win",
        "loss",
        "is_playoffs",
        "is_consolation",
        "optimal_points",
        "optimal_win",
        "optimal_loss",
    }
    miss_m = need_m - set(matchup_df.columns)
    if miss_m:
        st.error(f"❌ Matchup data missing: {sorted(miss_m)}")
        return

    # Prepare data
    display_df = _prepare_display_data(matchup_df)

    if display_df.empty:
        st.info("No optimal lineup data available")
        return

    # Calculate summary statistics
    stats = _calculate_stats(display_df)

    # === ENHANCED TABLE DISPLAY ===
    _render_enhanced_table(display_df)

    # === QUICK STATS SECTION (Below Table) ===
    st.markdown("---")
    _render_quick_stats(stats)

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
            file_name=f"career_optimal_lineups_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="career_optimal_download_csv",
            use_container_width=True,
        )


def _prepare_display_data(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and format career optimal lineup data for display."""
    df = matchup_df.copy()

    # Ensure proper types
    df["win"] = df["win"].astype(bool)
    df["loss"] = df["loss"].astype(bool)
    df["optimal_win"] = df["optimal_win"].astype(bool)
    df["optimal_loss"] = df["optimal_loss"].astype(bool)

    # Filter out rows with zero optimal points
    df = df[df["optimal_points"] != 0].copy()

    if df.empty:
        return pd.DataFrame()

    # Calculate efficiency and lost points
    df["efficiency"] = (df["team_points"] / df["optimal_points"] * 100).round(2)
    df["lost_points"] = df["optimal_points"] - df["team_points"]

    # Lucky Wins: Won despite poor efficiency (<80%)
    df["lucky_win"] = df["win"] & (df["efficiency"] < 80)

    # Aggregate by manager
    agg_dict = {
        "team_points": "sum",
        "optimal_points": "sum",
        "lost_points": "sum",
        "win": "sum",
        "loss": "sum",
        "optimal_win": "sum",
        "optimal_loss": "sum",
        "lucky_win": "sum",
        "efficiency": "mean",  # Average efficiency
    }

    display_df = df.groupby("manager").agg(agg_dict).reset_index()

    # Calculate additional metrics
    display_df["games"] = display_df["win"] + display_df["loss"]
    display_df["avg_team_points"] = (
        display_df["team_points"] / display_df["games"]
    ).round(2)
    display_df["avg_optimal_points"] = (
        display_df["optimal_points"] / display_df["games"]
    ).round(2)
    display_df["avg_lost_points"] = (
        display_df["lost_points"] / display_df["games"]
    ).round(2)

    # Wins/losses that changed with optimal lineup
    display_df["potential_wins"] = display_df["optimal_win"] - display_df["win"]
    display_df["saved_losses"] = display_df["loss"] - display_df["optimal_loss"]

    # Rename columns for display
    display_df = display_df.rename(
        columns={
            "manager": "Manager",
            "win": "Actual W",
            "loss": "Actual L",
            "optimal_win": "Optimal W",
            "optimal_loss": "Optimal L",
            "team_points": "Total Actual",
            "optimal_points": "Total Optimal",
            "lost_points": "Total Lost",
            "avg_team_points": "Avg Actual",
            "avg_optimal_points": "Avg Optimal",
            "avg_lost_points": "Avg Lost",
            "efficiency": "Efficiency %",
            "potential_wins": "Potential Wins",
            "saved_losses": "Saved Losses",
            "lucky_win": "Lucky Wins",
        }
    )

    # Sort by manager
    display_df = display_df.sort_values(by="Manager", ascending=True).reset_index(
        drop=True
    )

    return display_df


def _calculate_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics."""
    total_managers = len(df)
    if total_managers == 0:
        return {}

    total_games = df["games"].sum()
    total_lost_points = df["Total Lost"].sum()
    avg_efficiency = df["Efficiency %"].mean()

    # Potential wins/losses
    total_potential_wins = df["Potential Wins"].sum()
    total_saved_losses = df["Saved Losses"].sum()

    # Lucky wins
    total_lucky_wins = df["Lucky Wins"].sum()
    lucky_win_rate = (total_lucky_wins / total_games * 100) if total_games > 0 else 0

    # Best/worst efficiency
    best_efficiency = df["Efficiency %"].max()
    worst_efficiency = df["Efficiency %"].min()

    return {
        "total_managers": total_managers,
        "total_games": int(total_games),
        "total_lost_points": total_lost_points,
        "avg_efficiency": avg_efficiency,
        "total_potential_wins": int(total_potential_wins),
        "total_saved_losses": int(total_saved_losses),
        "total_lucky_wins": int(total_lucky_wins),
        "lucky_win_rate": lucky_win_rate,
        "best_efficiency": best_efficiency,
        "worst_efficiency": worst_efficiency,
    }


def _render_enhanced_table(df: pd.DataFrame):
    """Render enhanced table with column configuration."""

    # Configure column display
    column_config = {
        "Manager": st.column_config.TextColumn(
            "Manager", help="Manager name", width="medium"
        ),
        "Actual W": st.column_config.NumberColumn(
            "Actual W", help="Actual wins", format="%d", width="small"
        ),
        "Actual L": st.column_config.NumberColumn(
            "Actual L", help="Actual losses", format="%d", width="small"
        ),
        "Optimal W": st.column_config.NumberColumn(
            "Optimal W", help="Wins with optimal lineup", format="%d", width="small"
        ),
        "Optimal L": st.column_config.NumberColumn(
            "Optimal L", help="Losses with optimal lineup", format="%d", width="small"
        ),
        "Potential Wins": st.column_config.NumberColumn(
            "Potential Wins",
            help="Additional wins with optimal lineup",
            format="%d",
            width="small",
        ),
        "Saved Losses": st.column_config.NumberColumn(
            "Saved Losses",
            help="Losses avoided with optimal lineup",
            format="%d",
            width="small",
        ),
        "Lucky Wins": st.column_config.NumberColumn(
            "Lucky Wins",
            help="Wins despite poor efficiency (<80%)",
            format="%d",
            width="small",
        ),
        "Total Actual": st.column_config.NumberColumn(
            "Total Actual", help="Total actual points", format="%.2f", width="small"
        ),
        "Total Optimal": st.column_config.NumberColumn(
            "Total Optimal", help="Total optimal points", format="%.2f", width="small"
        ),
        "Total Lost": st.column_config.NumberColumn(
            "Total Lost",
            help="Total points left on bench",
            format="%.2f",
            width="small",
        ),
        "Avg Actual": st.column_config.NumberColumn(
            "Avg Actual",
            help="Average actual points per game",
            format="%.2f",
            width="small",
        ),
        "Avg Optimal": st.column_config.NumberColumn(
            "Avg Optimal",
            help="Average optimal points per game",
            format="%.2f",
            width="small",
        ),
        "Avg Lost": st.column_config.NumberColumn(
            "Avg Lost",
            help="Average points lost per game",
            format="%.2f",
            width="small",
        ),
        "Efficiency %": st.column_config.NumberColumn(
            "Efficiency %",
            help="Average lineup efficiency",
            format="%.2f%%",
            width="small",
        ),
    }

    # Display the enhanced dataframe
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=500,
    )


def _render_quick_stats(stats: dict):
    """Render quick statistics cards."""
    if not stats:
        return

    st.markdown("### 📈 Quick Stats")

    # Row 1: Overall Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Managers",
            f"{stats['total_managers']:,}",
            help="Total managers analyzed",
        )

    with col2:
        st.metric(
            "Total Games",
            f"{stats['total_games']:,}",
            help="Total games across all managers",
        )

    with col3:
        st.metric(
            "Avg Efficiency",
            f"{stats['avg_efficiency']:.2f}%",
            help="Average lineup efficiency",
        )

    with col4:
        st.metric(
            "Total Lost Points",
            f"{stats['total_lost_points']:,.2f}",
            help="Total points left on bench",
        )

    # Row 2: Efficiency Range
    st.markdown("---")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Best Efficiency",
            f"{stats['best_efficiency']:.2f}%",
            help="Highest average efficiency",
        )

    with col6:
        st.metric(
            "Worst Efficiency",
            f"{stats['worst_efficiency']:.2f}%",
            help="Lowest average efficiency",
        )

    with col7:
        efficiency_range = stats["best_efficiency"] - stats["worst_efficiency"]
        st.metric(
            "Efficiency Range",
            f"{efficiency_range:.2f}%",
            help="Difference between best and worst",
        )

    with col8:
        avg_lost_per_game = (
            stats["total_lost_points"] / stats["total_games"]
            if stats["total_games"] > 0
            else 0
        )
        st.metric(
            "Avg Lost/Game",
            f"{avg_lost_per_game:.2f}",
            help="Average points lost per game",
        )

    # Row 3: What-If Analysis
    st.markdown("---")
    col9, col10, col11, col12 = st.columns(4)

    with col9:
        st.metric(
            "Potential Wins",
            f"{stats['total_potential_wins']:,}",
            help="Additional wins with optimal lineups",
        )

    with col10:
        st.metric(
            "Saved Losses",
            f"{stats['total_saved_losses']:,}",
            help="Losses avoided with optimal lineups",
        )

    with col11:
        st.metric(
            "Lucky Wins",
            f"{stats['total_lucky_wins']:,}",
            help="Wins despite poor efficiency (<80%)",
        )

    with col12:
        st.metric(
            "Lucky Win Rate",
            f"{stats['lucky_win_rate']:.1f}%",
            help="% of games won despite low efficiency",
        )
