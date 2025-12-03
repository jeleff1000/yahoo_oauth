#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import (
    load_filtered_weekly_player_data,
)
from md.core import list_player_seasons


@st.fragment
def display_manager_spar_capture_rate(prefix=""):
    """
    Manager SPAR Capture Rate - Shows (Manager SPAR / Player SPAR) × 100

    Identifies which players managers benched during their best weeks.
    100% = perfect starts, <100% = left value on bench
    """
    st.header("🎯 Manager SPAR Capture Rate")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Lineup Management Audit:</strong> See what % of available SPAR value you captured.
    100% = perfect starts, <100% = benched during boom weeks.
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Year selection
    available_years = list_player_seasons()
    if not available_years:
        st.error("No player data found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Select Season",
            options=sorted(available_years, reverse=True),
            key=f"{prefix}_capture_year",
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_capture_position",
        )

    # Load weekly data and aggregate
    with st.spinner("Loading data..."):
        filters = {"year": [int(selected_year)], "rostered_only": True}
        if position != "All":
            filters["nfl_position"] = [position]

        weekly_data = load_filtered_weekly_player_data(filters=filters, limit=50000)

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for {selected_year}")
            return

    # Check for SPAR columns
    if (
        "player_spar" not in weekly_data.columns
        or "manager_spar" not in weekly_data.columns
    ):
        st.error("SPAR data not available. Make sure SPAR calculations have been run.")
        return

    # Convert to numeric
    weekly_data["player_spar"] = pd.to_numeric(
        weekly_data["player_spar"], errors="coerce"
    )
    weekly_data["manager_spar"] = pd.to_numeric(
        weekly_data["manager_spar"], errors="coerce"
    )

    # Aggregate to season level
    season_data = (
        weekly_data.groupby("player")
        .agg(
            {
                "player_spar": "sum",
                "manager_spar": "sum",
                "nfl_position": "first",
                "manager": "first",
            }
        )
        .reset_index()
    )

    season_data.columns = [
        "player",
        "player_spar",
        "manager_spar",
        "nfl_position",
        "manager",
    ]

    # Filter to players with SPAR > 0
    season_data = season_data[
        (season_data["player_spar"].notna()) & (season_data["player_spar"] > 0)
    ].copy()

    if season_data.empty:
        st.warning("No SPAR data available for selected filters.")
        return

    # Calculate capture rate
    season_data["capture_rate"] = (
        (season_data["manager_spar"] / season_data["player_spar"] * 100)
        .fillna(0)
        .clip(upper=100)  # Cap at 100%
    )

    # Calculate missed SPAR
    season_data["missed_spar"] = (
        season_data["player_spar"] - season_data["manager_spar"]
    )

    # Sort by most missed SPAR
    season_data = season_data.sort_values("missed_spar", ascending=False)

    # Limit to top 20
    top_n = 20
    display_data = season_data.head(top_n).copy()

    # Create the chart
    st.subheader(f"Top {len(display_data)} Players by Missed SPAR")

    fig = go.Figure()

    # Add captured SPAR (green)
    fig.add_trace(
        go.Bar(
            x=display_data["player"],
            y=display_data["manager_spar"],
            name="Captured SPAR",
            marker_color="#10B981",
            hovertemplate="<b>%{x}</b><br>Captured: %{y:.1f} SPAR<br><extra></extra>",
        )
    )

    # Add missed SPAR (red)
    fig.add_trace(
        go.Bar(
            x=display_data["player"],
            y=display_data["missed_spar"],
            name="Missed SPAR (Benched)",
            marker_color="#EF4444",
            hovertemplate="<b>%{x}</b><br>Missed: %{y:.1f} SPAR<br><extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Player",
        yaxis_title="SPAR Value",
        hovermode="x unified",
        showlegend=True,
        height=600,
        template="plotly_white",
        xaxis=dict(tickangle=-45),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_capture_bar")

    # Capture rate ranking
    st.subheader("📊 Capture Rate Rankings")

    # Color code by capture rate
    def get_capture_color(rate):
        if rate >= 90:
            return "#10B981"  # Green - excellent
        elif rate >= 75:
            return "#3B82F6"  # Blue - good
        elif rate >= 50:
            return "#F59E0B"  # Orange - mediocre
        else:
            return "#EF4444"  # Red - poor

    display_data["color"] = display_data["capture_rate"].apply(get_capture_color)

    # Horizontal bar chart of capture rates
    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            y=display_data["player"],
            x=display_data["capture_rate"],
            orientation="h",
            marker=dict(color=display_data["color"], line=dict(color="white", width=2)),
            text=display_data["capture_rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Capture Rate: %{x:.1f}%<br><extra></extra>",
        )
    )

    fig2.update_layout(
        xaxis_title="Capture Rate (%)",
        yaxis_title="Player",
        height=600,
        template="plotly_white",
        showlegend=False,
        yaxis=dict(autorange="reversed"),  # Top to bottom
    )

    st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_capture_rate_bar")

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_capture = display_data["capture_rate"].mean()
        st.metric("Avg Capture Rate", f"{avg_capture:.1f}%")

    with col2:
        total_missed = display_data["missed_spar"].sum()
        st.metric("Total Missed SPAR", f"{total_missed:.1f}")

    with col3:
        worst_capture = display_data["capture_rate"].min()
        st.metric("Worst Capture Rate", f"{worst_capture:.1f}%")

    # Detailed table
    with st.expander("📋 Detailed Data"):
        table_data = display_data[
            [
                "player",
                "nfl_position",
                "manager",
                "player_spar",
                "manager_spar",
                "missed_spar",
                "capture_rate",
            ]
        ].copy()

        table_data = table_data.rename(
            columns={
                "player": "Player",
                "nfl_position": "Pos",
                "manager": "Manager",
                "player_spar": "Player SPAR",
                "manager_spar": "Manager SPAR",
                "missed_spar": "Missed SPAR",
                "capture_rate": "Capture %",
            }
        )

        # Format numbers
        table_data["Player SPAR"] = table_data["Player SPAR"].apply(
            lambda x: f"{x:.1f}"
        )
        table_data["Manager SPAR"] = table_data["Manager SPAR"].apply(
            lambda x: f"{x:.1f}"
        )
        table_data["Missed SPAR"] = table_data["Missed SPAR"].apply(
            lambda x: f"{x:.1f}"
        )
        table_data["Capture %"] = table_data["Capture %"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(table_data, hide_index=True, use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown(
            """
        **Capture Rate** = (Manager SPAR / Player SPAR) × 100

        - **100%**: Perfect lineup management - started player every week they produced value
        - **75-99%**: Good - missed some value but mostly captured opportunities
        - **50-74%**: Mediocre - left significant value on bench
        - **<50%**: Poor - benched player during most of their good weeks

        **Common Reasons for Low Capture Rate:**
        - Benched during player's boom weeks
        - Player on bench early season, broke out later
        - Traded/dropped before breakout
        - Backup who became starter mid-season

        **Use This To:**
        - Identify which players you should have started more
        - Audit your lineup decisions
        - Learn from mistakes for next season
        """
        )
