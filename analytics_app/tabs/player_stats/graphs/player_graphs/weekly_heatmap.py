#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import list_player_seasons
from md.tab_data_access.players.weekly_player_data import (
    load_filtered_weekly_player_data,
)


@st.fragment
def display_weekly_performance_heatmap(prefix=""):
    """
    Calendar-style heatmap showing player performance week-by-week.
    Great for spotting consistency patterns and hot/cold streaks.
    """
    st.header("🗓️ Weekly Performance Heatmap")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Visual consistency tracker:</strong> See every game at a glance.
    Color intensity shows scoring levels - dark green = elite week, red = poor week.
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

    _min_year, _max_year = min(available_years), max(available_years)

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Select Season",
            options=sorted(available_years, reverse=True),
            key=f"{prefix}_heatmap_year",
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_heatmap_position",
        )

    # Player search
    player_search = st.text_input(
        "🔍 Search for players (comma separated):",
        placeholder="e.g., Patrick Mahomes, Christian McCaffrey",
        key=f"{prefix}_heatmap_search",
    ).strip()

    if not player_search:
        st.info("💡 Enter player name(s) to display their weekly heatmap")
        return

    # Load data
    with st.spinner("Loading weekly data..."):
        # Load all weeks for the selected year
        filters = {"year": [int(selected_year)], "rostered_only": False}
        if position != "All":
            filters["nfl_position"] = [position]

        weekly_data = load_filtered_weekly_player_data(
            filters=filters, limit=50000  # Get all data for the year
        )

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for {selected_year}")
            return

    # Parse player names and filter
    search_names = [name.strip() for name in player_search.split(",") if name.strip()]
    weekly_data["player_lower"] = weekly_data["player"].str.lower()
    search_lower = [n.lower() for n in search_names]

    filtered = weekly_data[
        weekly_data["player_lower"].apply(
            lambda x: any(search in x for search in search_lower)
        )
    ].copy()
    filtered = filtered.drop(columns=["player_lower"])

    if filtered.empty:
        st.warning(f"No players found matching: {', '.join(search_names)}")
        return

    st.success(f"✅ Found {len(filtered['player'].unique())} player(s)")

    # Options
    col3, col4 = st.columns(2)
    with col3:
        color_metric = st.radio(
            "Color by",
            ["Points", "PPG Differential"],
            key=f"{prefix}_heatmap_color",
            help="Points shows raw score. PPG Differential shows how far above/below season average.",
        )

    with col4:
        show_values = st.checkbox(
            "Show values on heatmap", value=True, key=f"{prefix}_heatmap_show_values"
        )

    # Prepare heatmap data
    players = filtered["player"].unique()

    for player_name in players:
        player_df = filtered[filtered["player"] == player_name].copy()

        # Ensure numeric types
        player_df["week"] = pd.to_numeric(player_df["week"], errors="coerce")
        player_df["points"] = pd.to_numeric(player_df["points"], errors="coerce")

        # Calculate season PPG for differential
        season_ppg = player_df["points"].mean()
        player_df["ppg_diff"] = player_df["points"] - season_ppg

        # Sort by week
        player_df = player_df.sort_values("week")

        # Get position and team for subtitle
        pos = (
            player_df["nfl_position"].iloc[0]
            if "nfl_position" in player_df.columns
            else ""
        )
        team = player_df["nfl_team"].iloc[0] if "nfl_team" in player_df.columns else ""
        manager = player_df["manager"].iloc[0] if "manager" in player_df.columns else ""

        # Create matrix for heatmap (1 row per player)
        weeks = player_df["week"].tolist()
        if color_metric == "Points":
            values = player_df["points"].tolist()
            colorbar_title = "Points"
        else:
            values = player_df["ppg_diff"].tolist()
            colorbar_title = "vs Avg"

        # Create heatmap
        fig = go.Figure()

        # Determine color scale based on metric
        if color_metric == "Points":
            colorscale = [
                [0.0, "#DC2626"],  # Red (low)
                [0.3, "#F59E0B"],  # Orange
                [0.5, "#FCD34D"],  # Yellow
                [0.7, "#10B981"],  # Green
                [1.0, "#059669"],  # Dark Green (high)
            ]
        else:
            # Diverging scale for differential (red=below avg, green=above avg)
            colorscale = [
                [0.0, "#DC2626"],  # Red (well below)
                [0.5, "#F3F4F6"],  # Gray (average)
                [1.0, "#059669"],  # Green (well above)
            ]

        # Create text labels
        if show_values:
            text_values = [[f"{v:.1f}" for v in values]]
        else:
            text_values = None

        fig.add_trace(
            go.Heatmap(
                z=[values],
                x=weeks,
                y=[player_name],
                colorscale=colorscale,
                text=text_values,
                texttemplate="%{text}" if show_values else None,
                textfont={"size": 10},
                hovertemplate="<b>Week %{x}</b><br>%{y}<br>"
                + colorbar_title
                + ": %{z:.2f}<extra></extra>",
                colorbar=dict(title=colorbar_title, thickness=15, len=0.7),
            )
        )

        # Update layout
        subtitle = f"{pos} - {team}" if pos and team else ""
        if manager:
            subtitle += f" | Rostered by: {manager}"

        fig.update_layout(
            title=dict(
                text=f"<b>{player_name}</b><br><sub>{subtitle}</sub>",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Week",
                tickmode="linear",
                tick0=1,
                dtick=1,
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
            ),
            yaxis=dict(title="", showticklabels=False),
            height=200,
            margin=dict(l=20, r=20, t=80, b=50),
            template="plotly_white",
        )

        st.plotly_chart(
            fig, use_container_width=True, key=f"{prefix}_heatmap_{player_name}"
        )

        # Summary stats
        with st.expander(f"📊 {player_name} - {selected_year} Summary", expanded=False):
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Games", len(player_df))
            with col2:
                st.metric("Total Pts", f"{player_df['points'].sum():.1f}")
            with col3:
                st.metric("Avg PPG", f"{season_ppg:.2f}")
            with col4:
                st.metric("Best Week", f"{player_df['points'].max():.1f}")
            with col5:
                st.metric("Worst Week", f"{player_df['points'].min():.1f}")

            # Week-by-week table
            st.caption("Week-by-week breakdown:")
            detail_cols = ["week", "points", "opponent"]
            if "nfl_position" in player_df.columns:
                detail_cols.insert(1, "nfl_position")

            available_cols = [c for c in detail_cols if c in player_df.columns]
            detail_df = player_df[available_cols].copy()

            # Rename for display
            detail_df = detail_df.rename(
                columns={
                    "week": "Week",
                    "points": "Points",
                    "opponent": "Opponent",
                    "nfl_position": "Position",
                }
            )

            st.dataframe(
                detail_df,
                hide_index=True,
                use_container_width=True,
                column_config={"Points": st.column_config.NumberColumn(format="%.1f")},
            )

    # Multi-player comparison option
    if len(players) > 1:
        st.markdown("---")
        st.subheader("📊 Multi-Player Comparison")

        # Create combined heatmap
        comparison_data = []
        for player_name in players:
            player_df = filtered[filtered["player"] == player_name].copy()
            player_df = player_df.sort_values("week")

            season_ppg = player_df["points"].mean()

            for _, row in player_df.iterrows():
                comparison_data.append(
                    {
                        "player": player_name,
                        "week": row["week"],
                        "points": row["points"],
                        "ppg_diff": row["points"] - season_ppg,
                    }
                )

        comp_df = pd.DataFrame(comparison_data)

        # Pivot for heatmap
        if color_metric == "Points":
            pivot_df = comp_df.pivot(index="player", columns="week", values="points")
        else:
            pivot_df = comp_df.pivot(index="player", columns="week", values="ppg_diff")

        fig_comp = go.Figure()

        fig_comp.add_trace(
            go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale=colorscale,
                hovertemplate="<b>%{y}</b><br>Week %{x}<br>"
                + colorbar_title
                + ": %{z:.2f}<extra></extra>",
                colorbar=dict(title=colorbar_title, thickness=15, len=0.7),
            )
        )

        fig_comp.update_layout(
            title=f"Player Comparison - {selected_year}",
            xaxis=dict(title="Week", tickmode="linear", tick0=1, dtick=1),
            yaxis=dict(title="Player"),
            height=max(300, len(players) * 60),
            template="plotly_white",
        )

        st.plotly_chart(
            fig_comp, use_container_width=True, key=f"{prefix}_heatmap_comparison"
        )
