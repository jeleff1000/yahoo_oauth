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
def display_player_radar_comparison(prefix=""):
    """
    Multi-dimensional radar/spider chart for comparing players.
    Shows PPG, Consistency, Win Rate, Peak, Floor, and Ceiling.
    """
    st.header("🕸️ Player Comparison Radar")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Multi-dimensional comparison:</strong> Compare players across 6 key metrics.
    Larger area = better overall player. Shape shows strengths/weaknesses.
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

    selected_year = st.selectbox(
        "Select Season",
        options=sorted(available_years, reverse=True),
        key=f"{prefix}_radar_year",
    )

    # Player search
    player_search = st.text_input(
        "🔍 Enter 2-5 players to compare (comma separated):",
        placeholder="e.g., Josh Allen, Jalen Hurts, Patrick Mahomes",
        key=f"{prefix}_radar_search",
    ).strip()

    if not player_search:
        st.info("💡 Enter 2-5 player names to compare")
        return

    # Parse players
    search_names = [name.strip() for name in player_search.split(",") if name.strip()]
    if len(search_names) < 2:
        st.warning("⚠️ Enter at least 2 players to compare")
        return
    if len(search_names) > 5:
        st.warning("⚠️ Limiting to first 5 players")
        search_names = search_names[:5]

    # Load weekly data to calculate metrics
    with st.spinner("Calculating player metrics..."):
        filters = {"year": [int(selected_year)], "rostered_only": False}

        weekly_data = load_filtered_weekly_player_data(filters=filters, limit=50000)

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for {selected_year}")
            return

    # Filter players
    weekly_data["player_lower"] = weekly_data["player"].str.lower()
    search_lower = [n.lower() for n in search_names]

    filtered = weekly_data[
        weekly_data["player_lower"].apply(
            lambda x: any(search in x for search in search_lower)
        )
    ].copy()

    if filtered.empty:
        st.warning(f"No players found matching: {', '.join(search_names)}")
        return

    filtered["points"] = pd.to_numeric(filtered["points"], errors="coerce")
    filtered = filtered.dropna(subset=["points"])

    players = filtered["player"].unique()
    if len(players) < 2:
        st.warning("Found less than 2 players. Need at least 2 for comparison.")
        return

    st.success(f"✅ Comparing {len(players)} player(s)")

    # Calculate metrics for each player
    player_metrics = []

    for player_name in players:
        player_df = filtered[filtered["player"] == player_name]
        points = player_df["points"]

        # Calculate metrics
        ppg = points.mean()
        consistency = 100 - (points.std() / ppg * 100) if ppg > 0 else 0  # Inverse CV%
        consistency = max(0, min(100, consistency))  # Clamp to 0-100

        peak = points.max()
        floor = points.min()
        points.median()

        # Win rate if available
        if "win" in player_df.columns:
            wins = pd.to_numeric(player_df["win"], errors="coerce").sum()
            win_rate = (wins / len(player_df) * 100) if len(player_df) > 0 else 0
        else:
            win_rate = 50  # Default if not available

        player_metrics.append(
            {
                "player": player_name,
                "PPG": ppg,
                "Consistency": consistency,
                "Win Rate": win_rate,
                "Peak Performance": peak,
                "Floor": floor,
                "Ceiling": peak,
            }
        )

    # Normalize metrics to 0-100 scale for radar chart
    metrics_df = pd.DataFrame(player_metrics)

    # Normalize each metric
    for metric in ["PPG", "Peak Performance", "Floor", "Ceiling"]:
        max_val = metrics_df[metric].max()
        if max_val > 0:
            metrics_df[f"{metric}_norm"] = metrics_df[metric] / max_val * 100
        else:
            metrics_df[f"{metric}_norm"] = 0

    # Consistency and Win Rate are already 0-100
    metrics_df["Consistency_norm"] = metrics_df["Consistency"]
    metrics_df["Win Rate_norm"] = metrics_df["Win Rate"]

    # Create radar chart
    fig = go.Figure()

    categories = [
        "PPG",
        "Consistency",
        "Win Rate",
        "Peak Performance",
        "Floor",
        "Ceiling",
    ]

    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    for idx, player_name in enumerate(players):
        player_row = metrics_df[metrics_df["player"] == player_name].iloc[0]

        values = [
            player_row["PPG_norm"],
            player_row["Consistency_norm"],
            player_row["Win Rate_norm"],
            player_row["Peak Performance_norm"],
            player_row["Floor_norm"],
            player_row["Ceiling_norm"],
        ]

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill="toself",
                name=player_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                fillcolor=colors[idx % len(colors)],
                opacity=0.3,
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<br><extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks="",
                gridcolor="rgba(128,128,128,0.2)",
            ),
            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        ),
        showlegend=True,
        title="Player Comparison (Normalized 0-100)",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_radar_chart")

    # Show raw metrics table
    st.subheader("📊 Raw Metrics")

    display_metrics = metrics_df[
        [
            "player",
            "PPG",
            "Consistency",
            "Win Rate",
            "Peak Performance",
            "Floor",
            "Ceiling",
        ]
    ].copy()
    display_metrics = display_metrics.rename(columns={"player": "Player"})

    # Format columns
    for col in ["PPG", "Peak Performance", "Floor", "Ceiling"]:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2f}")

    for col in ["Consistency", "Win Rate"]:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_metrics, hide_index=True, use_container_width=True)

    # Metric explanations
    with st.expander("📖 Understanding the Metrics", expanded=False):
        st.markdown(
            """
        **Radar Chart Metrics:**

        1. **PPG (Points Per Game)**: Average fantasy points per week
        2. **Consistency**: Inverse of coefficient of variation (100 = perfectly consistent)
        3. **Win Rate**: Percentage of games where player's team won
        4. **Peak Performance**: Highest single-game score
        5. **Floor**: Lowest single-game score
        6. **Ceiling**: Same as Peak (for symmetry)

        **How to Read:**
        - **Larger area** = Better overall player
        - **Balanced shape** = Well-rounded player
        - **Spiky shape** = Strong in some areas, weak in others
        - Compare shapes to see different player profiles
        """
        )
