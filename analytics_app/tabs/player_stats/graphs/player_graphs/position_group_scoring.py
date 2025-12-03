#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import list_player_seasons
from md.tab_data_access.players import load_players_season_data


@st.fragment
def display_position_group_scoring_graphs(prefix=""):
    """
    Position group scoring trends - optimized to use data_access layer.
    Compare performance across positions and managers.
    """
    st.header("📊 Position Group Scoring")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Analyze position performance:</strong> See how different positions perform across seasons.
    Filter by manager to see roster strength by position group.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection
    available_years = list_player_seasons()
    if not available_years:
        st.error("No player data found.")
        return

    min_year, max_year = min(available_years), max(available_years)

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=min_year, max_value=max_year,
                                     value=min_year, key=f"{prefix}_start_year")
    with col2:
        end_year = st.number_input("End Year", min_value=min_year, max_value=max_year,
                                   value=max_year, key=f"{prefix}_end_year")

    with st.spinner("Loading player data..."):
        # Load data for selected years
        years_to_load = list(range(int(start_year), int(end_year) + 1))

        player_data = load_players_season_data(
            year=years_to_load,
            rostered_only=False,  # Show all players, not just rostered
            sort_column="points",
            sort_direction="DESC"
        )

        if player_data.empty:
            st.warning("No data found for the selected years.")
            return

    # Manager filter
    managers = sorted([m for m in player_data["manager"].dropna().unique() if m and str(m).strip()])

    if managers:
        manager_options = ["All Players (1999+)", "All Rostered Players (2014+)"] + managers
        manager_filter = st.selectbox(
            "Filter by Manager",
            options=manager_options,
            key=f"{prefix}_manager",
            help="'All Players' shows complete history back to 1999. Manager filters only show 2014+ when rosters were tracked."
        )

        if manager_filter == "All Players (1999+)":
            # Keep all data, no filtering
            pass
        elif manager_filter == "All Rostered Players (2014+)":
            # Filter to only players who had a manager at some point
            player_data = player_data[player_data["manager"].notna()]
        else:
            # Specific manager selected
            player_data = player_data[player_data["manager"].str.contains(manager_filter, na=False)]
    else:
        st.info("No manager data available - showing all players from full history (1999+)")

    # Position selection
    position_order = ["QB", "RB", "WR", "TE", "K", "DEF"]
    available_positions = [pos for pos in position_order if pos in player_data["nfl_position"].unique()]

    with st.expander("Select Positions to Display", expanded=True):
        selected_positions = st.multiselect(
            "Choose positions",
            options=available_positions,
            default=[p for p in available_positions if p not in {"K", "DEF"}],
            key=f"{prefix}_positions"
        )

    if not selected_positions:
        st.info("Please select at least one position to display.")
        return

    # Filter by selected positions
    filtered = player_data[player_data["nfl_position"].isin(selected_positions)]

    if filtered.empty:
        st.warning("No data for selected positions.")
        return

    # Aggregate by position and year
    agg = filtered.groupby(["nfl_position", "year"], as_index=False).agg({
        "season_ppg": "mean",
        "points": "sum",
        "fantasy_games": "sum",
        "games_started": "sum"
    })

    agg = agg.rename(columns={"season_ppg": "avg_ppg"})
    agg["avg_ppg"] = agg["avg_ppg"].round(2)
    agg["points"] = agg["points"].round(2)

    # Create visualization
    st.subheader("Average Points Per Game by Position")

    fig = go.Figure()

    for position in selected_positions:
        pos_data = agg[agg["nfl_position"] == position].sort_values("year")

        if not pos_data.empty:
            fig.add_trace(go.Scatter(
                x=pos_data["year"],
                y=pos_data["avg_ppg"],
                mode="lines+markers",
                name=position,
                marker=dict(size=8),
                line=dict(width=3),
                hovertemplate=f"<b>{position}</b><br>Year: %{{x}}<br>Avg PPG: %{{y:.2f}}<br><extra></extra>"
            ))

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Points Per Game",
        hovermode="x unified",
        showlegend=True,
        height=500,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(tickmode='linear', dtick=1, showgrid=True)
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_position_group_ppg")

    # Summary statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Position Averages")
        summary = agg.groupby("nfl_position", as_index=False).agg({
            "avg_ppg": "mean",
            "points": "sum",
            "fantasy_games": "sum"
        })
        summary = summary.rename(columns={
            "avg_ppg": "Avg PPG",
            "points": "Total Points",
            "fantasy_games": "Total Games"
        })
        summary["Avg PPG"] = summary["Avg PPG"].round(2)
        summary = summary.sort_values("Avg PPG", ascending=False)
        st.dataframe(summary, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("📊 Year-over-Year Trends")
        # Calculate year-over-year change
        trends = []
        for pos in selected_positions:
            pos_data = agg[agg["nfl_position"] == pos].sort_values("year")
            if len(pos_data) >= 2:
                latest = pos_data.iloc[-1]["avg_ppg"]
                previous = pos_data.iloc[-2]["avg_ppg"]
                change = latest - previous
                pct_change = (change / previous * 100) if previous > 0 else 0
                trends.append({
                    "Position": pos,
                    "Latest PPG": round(latest, 2),
                    "Change": round(change, 2),
                    "% Change": round(pct_change, 1)
                })

        if trends:
            trends_df = pd.DataFrame(trends).sort_values("% Change", ascending=False)
            st.dataframe(trends_df, hide_index=True, use_container_width=True)
        else:
            st.info("Need at least 2 years of data to show trends.")

    # Detailed data table
    with st.expander("📋 View Detailed Data", expanded=False):
        pivot = agg.pivot(index="year", columns="nfl_position", values="avg_ppg")
        pivot = pivot[[p for p in position_order if p in pivot.columns]]
        pivot = pivot.round(2)
        st.dataframe(pivot, use_container_width=True)
