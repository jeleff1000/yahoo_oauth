#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from md.data_access import load_players_career_data, list_player_seasons


@st.fragment
def display_player_consistency_graph(prefix=""):
    """
    Analyze player consistency - variance, boom/bust rates, reliability.
    """
    st.header("🎯 Player Consistency Analysis")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Find reliable performers:</strong> Compare players by their consistency metrics.
    High PPG with low variance = reliable. High variance = boom/bust.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection with "All Seasons" option
    available_years = list_player_seasons()
    if not available_years:
        st.error("No player data found.")
        return

    min_year, max_year = min(available_years), max(available_years)

    year_options = ["All Seasons"] + list(range(max_year, min_year - 1, -1))

    col1, col2 = st.columns(2)
    with col1:
        year_selection = st.selectbox(
            "Select Season Range",
            options=year_options,
            index=0,
            key=f"{prefix}_year_range"
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_position"
        )

    # Determine years to load
    if year_selection == "All Seasons":
        years_to_load = list(range(min_year, max_year + 1))
    else:
        years_to_load = [int(year_selection)]

    # Roster filter options
    col3, col4 = st.columns(2)
    with col3:
        roster_filter = st.radio(
            "Data Range",
            options=["All Players (1999+)", "Rostered Only (2014+)"],
            index=0,
            key=f"{prefix}_roster_filter",
            help="All Players shows complete history. Rostered Only limits to when manager data was tracked."
        )

    with col4:
        started_filter = st.checkbox(
            "Started games only",
            value=True,
            key=f"{prefix}_started_only",
            help="Only count games where player was actually started (not benched)"
        )

    # Load data
    with st.spinner("Loading player data..."):
        player_data = load_players_career_data(
            year=years_to_load,
            position=None if position == "All" else position,
            rostered_only=(roster_filter == "Rostered Only (2014+)"),
            started_only=started_filter,
            sort_column="points",
            sort_direction="DESC"
        )

        if player_data.empty:
            st.warning("No data found for the selected filters.")
            return

    # Filter for minimum games (need meaningful sample)
    min_games = st.slider("Minimum games started", 5, 50, 10, key=f"{prefix}_min_games")
    player_data = player_data[player_data["games_started"] >= min_games]

    if player_data.empty:
        st.warning(f"No players with at least {min_games} games started.")
        return

    # Calculate metrics
    player_data["ppg"] = (player_data["points"] / player_data["games_started"]).round(2)
    if "win" in player_data.columns:
        player_data["win_rate"] = ((player_data["win"] / player_data["games_started"]) * 100).round(1)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Top Performers", "🎯 PPG vs Games", "📋 Full Stats"])

    with tab1:
        st.subheader("📊 Top Performers by Points Per Game")

        top_n = st.slider("Show top N players", 5, 50, 20, key=f"{prefix}_top_n")
        top_players = player_data.nlargest(top_n, "ppg")

        # Create horizontal bar chart for easier reading
        fig = go.Figure()

        top_players_sorted = top_players.sort_values("ppg")

        fig.add_trace(go.Bar(
            y=top_players_sorted["player"],
            x=top_players_sorted["ppg"],
            orientation='h',
            marker=dict(
                color=top_players_sorted["ppg"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="PPG")
            ),
            text=top_players_sorted["ppg"].apply(lambda x: f"{x:.2f}"),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>PPG: %{x:.2f}<br>Games: " +
                         top_players_sorted["games_started"].astype(str) + "<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title="Points Per Game",
            yaxis_title="",
            height=max(400, top_n * 25),
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_top_performers")

    with tab2:
        st.subheader("🎯 PPG vs Volume Analysis")
        st.caption("Find players who produce consistently over many games")

        # Scatter plot: PPG vs Games Started
        fig2 = px.scatter(
            player_data,
            x="games_started",
            y="ppg",
            color="nfl_position" if position == "All" else None,
            size="points",
            hover_name="player",
            hover_data={"ppg": ":.2f", "games_started": True, "points": ":.0f"},
            labels={"ppg": "Points Per Game", "games_started": "Games Started"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig2.update_layout(
            height=500,
            template="plotly_white"
        )

        # Add reference lines
        median_ppg = player_data["ppg"].median()
        median_games = player_data["games_started"].median()

        fig2.add_hline(y=median_ppg, line_dash="dash", line_color="gray", opacity=0.5)
        fig2.add_vline(x=median_games, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_ppg_vs_volume")

    with tab3:
        st.subheader("📋 Complete Player Statistics")

        # Prepare display dataframe
        display_cols = ["player", "nfl_position", "ppg", "points", "games_started"]
        if "win" in player_data.columns:
            display_cols.append("win_rate")

        available_cols = [c for c in display_cols if c in player_data.columns]
        display_df = player_data[available_cols].copy()

        # Rename columns for display
        column_mapping = {
            "player": "Player",
            "nfl_position": "Position",
            "ppg": "PPG",
            "points": "Total Points",
            "games_started": "Games",
            "win_rate": "Win %"
        }
        display_df = display_df.rename(columns=column_mapping)
        display_df = display_df.sort_values("PPG", ascending=False)

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "PPG": st.column_config.NumberColumn(format="%.2f"),
                "Total Points": st.column_config.NumberColumn(format="%.1f"),
                "Win %": st.column_config.NumberColumn(format="%.1f%%") if "Win %" in display_df.columns else None
            }
        )

    # Summary insights
    with st.expander("💡 Summary Insights", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Players", len(player_data))

        with col2:
            st.metric("Avg PPG", f"{player_data['ppg'].mean():.2f}")

        with col3:
            st.metric("Top PPG", f"{player_data['ppg'].max():.2f}")
            st.caption(player_data.nlargest(1, "ppg")["player"].iloc[0])

        with col4:
            if "win_rate" in player_data.columns:
                total_wins = player_data["win"].sum()
                total_games = player_data["games_started"].sum()
                win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
                st.metric("Avg Win Rate", f"{win_rate:.1f}%")
