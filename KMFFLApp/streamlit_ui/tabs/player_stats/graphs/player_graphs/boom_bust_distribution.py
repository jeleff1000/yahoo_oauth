#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from md.data_access import list_player_seasons
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data


@st.fragment
def display_boom_bust_distribution(prefix=""):
    """
    Histogram and violin plots showing scoring distribution.
    Identify boom/bust players vs consistent performers.
    """
    st.header("💥 Boom/Bust Analysis")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Understand player volatility:</strong> See the full distribution of weekly scores.
    Tight distributions = consistent. Wide distributions = boom/bust.
    </p>
    </div>
    """, unsafe_allow_html=True)

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
            key=f"{prefix}_boom_bust_year"
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_boom_bust_position"
        )

    # Player search
    player_search = st.text_input(
        "🔍 Search for players (comma separated, compare up to 5):",
        placeholder="e.g., Josh Allen, Lamar Jackson, Jalen Hurts",
        key=f"{prefix}_boom_bust_search"
    ).strip()

    if not player_search:
        st.info("💡 Enter player name(s) to analyze their scoring distribution")
        return

    # Load data
    with st.spinner("Loading weekly data..."):
        filters = {
            "year": [int(selected_year)],
            "rostered_only": False
        }
        if position != "All":
            filters["nfl_position"] = [position]

        weekly_data = load_filtered_weekly_player_data(
            filters=filters,
            limit=50000
        )

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for {selected_year}")
            return

    # Parse and filter players
    search_names = [name.strip() for name in player_search.split(",") if name.strip()]
    if len(search_names) > 5:
        st.warning("⚠️ Limiting to first 5 players for readability")
        search_names = search_names[:5]

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

    # Ensure numeric
    filtered['points'] = pd.to_numeric(filtered['points'], errors='coerce')
    filtered = filtered.dropna(subset=['points'])

    players = filtered['player'].unique()
    st.success(f"✅ Found {len(players)} player(s)")

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Histogram", "🎻 Violin Plot", "📈 Stats Summary"])

    with tab1:
        st.subheader("Score Distribution Histogram")
        st.caption("Shows how often players score in different point ranges")

        # Create overlaid histograms
        fig_hist = go.Figure()

        colors = px.colors.qualitative.Set2
        for idx, player_name in enumerate(players):
            player_df = filtered[filtered['player'] == player_name]

            fig_hist.add_trace(go.Histogram(
                x=player_df['points'],
                name=player_name,
                opacity=0.7,
                marker_color=colors[idx % len(colors)],
                nbinsx=15,
                hovertemplate='<b>%{x:.1f}-point range</b><br>Frequency: %{y}<br><extra></extra>'
            ))

        fig_hist.update_layout(
            barmode='overlay',
            xaxis_title="Points Scored",
            yaxis_title="Number of Games",
            height=500,
            template="plotly_white",
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig_hist, use_container_width=True, key=f"{prefix}_boom_bust_histogram")

    with tab2:
        st.subheader("Score Distribution Violin Plot")
        st.caption("Width shows frequency. Wider = more common scoring range")

        # Create violin plot
        fig_violin = go.Figure()

        for idx, player_name in enumerate(players):
            player_df = filtered[filtered['player'] == player_name]

            fig_violin.add_trace(go.Violin(
                y=player_df['points'],
                name=player_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[idx % len(colors)],
                opacity=0.6,
                hovertemplate='<b>%{y:.2f} points</b><extra></extra>'
            ))

        fig_violin.update_layout(
            yaxis_title="Points Scored",
            xaxis_title="Player",
            height=500,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_violin, use_container_width=True, key=f"{prefix}_boom_bust_violin")

    with tab3:
        st.subheader("Statistical Breakdown")

        # Calculate statistics for each player
        stats_data = []

        for player_name in players:
            player_df = filtered[filtered['player'] == player_name].copy()
            points = player_df['points']

            # Calculate statistics
            ppg = points.mean()
            std_dev = points.std()
            cv = (std_dev / ppg * 100) if ppg > 0 else 0  # Coefficient of variation
            median = points.median()
            q1 = points.quantile(0.25)
            q3 = points.quantile(0.75)
            iqr = q3 - q1
            min_pts = points.min()
            max_pts = points.max()
            range_pts = max_pts - min_pts

            # Boom/Bust metrics
            boom_threshold = ppg + std_dev  # Games well above average
            bust_threshold = ppg - std_dev  # Games well below average
            boom_games = (points >= boom_threshold).sum()
            bust_games = (points <= bust_threshold).sum()
            boom_rate = (boom_games / len(points) * 100) if len(points) > 0 else 0
            bust_rate = (bust_games / len(points) * 100) if len(points) > 0 else 0

            stats_data.append({
                'Player': player_name,
                'Games': len(points),
                'PPG': f"{ppg:.2f}",
                'Median': f"{median:.2f}",
                'Std Dev': f"{std_dev:.2f}",
                'CV %': f"{cv:.1f}",
                'Min': f"{min_pts:.1f}",
                'Max': f"{max_pts:.1f}",
                'Range': f"{range_pts:.1f}",
                'Boom Rate': f"{boom_rate:.1f}%",
                'Bust Rate': f"{bust_rate:.1f}%"
            })

        stats_df = pd.DataFrame(stats_data)

        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        # Explanation
        with st.expander("📖 How to Read These Stats", expanded=False):
            st.markdown("""
            **Key Metrics:**

            - **PPG**: Points Per Game (average)
            - **Median**: Middle value (50th percentile)
            - **Std Dev**: Standard deviation - higher = more volatile
            - **CV %**: Coefficient of Variation - volatility relative to average (lower = more consistent)
            - **Range**: Difference between best and worst game
            - **Boom Rate**: % of games well above average (PPG + 1 std dev)
            - **Bust Rate**: % of games well below average (PPG - 1 std dev)

            **Interpretation:**

            - **Consistent Player**: Low CV%, low Std Dev, narrow range
            - **Boom/Bust Player**: High CV%, high Std Dev, wide range, high boom & bust rates
            - **High Floor Player**: High minimum score, low bust rate
            - **High Ceiling Player**: High maximum score, high boom rate
            """)

        # Scoring ranges breakdown
        st.subheader("Scoring Ranges")
        st.caption("Games in each point range")

        # Define scoring ranges
        ranges = [
            ("0-5", 0, 5),
            ("5-10", 5, 10),
            ("10-15", 10, 15),
            ("15-20", 15, 20),
            ("20-25", 20, 25),
            ("25-30", 25, 30),
            ("30+", 30, 999)
        ]

        range_data = []
        for player_name in players:
            player_df = filtered[filtered['player'] == player_name]
            points = player_df['points']

            row = {'Player': player_name}
            for range_label, min_val, max_val in ranges:
                if max_val == 999:
                    count = (points >= min_val).sum()
                else:
                    count = ((points >= min_val) & (points < max_val)).sum()
                pct = (count / len(points) * 100) if len(points) > 0 else 0
                row[range_label] = f"{count} ({pct:.1f}%)"

            range_data.append(row)

        range_df = pd.DataFrame(range_data)
        st.dataframe(range_df, hide_index=True, use_container_width=True)

    # Consistency Rating
    st.markdown("---")
    st.subheader("🎯 Consistency Rating")

    rating_data = []
    for player_name in players:
        player_df = filtered[filtered['player'] == player_name]
        points = player_df['points']

        ppg = points.mean()
        std_dev = points.std()
        cv = (std_dev / ppg * 100) if ppg > 0 else 0

        # Determine rating based on CV%
        if cv < 20:
            rating = "⭐⭐⭐⭐⭐ Very Consistent"
            color = "#059669"
        elif cv < 30:
            rating = "⭐⭐⭐⭐ Consistent"
            color = "#10B981"
        elif cv < 40:
            rating = "⭐⭐⭐ Moderate"
            color = "#F59E0B"
        elif cv < 50:
            rating = "⭐⭐ Volatile"
            color = "#F97316"
        else:
            rating = "⭐ Very Volatile"
            color = "#DC2626"

        rating_data.append({
            'player': player_name,
            'ppg': ppg,
            'cv': cv,
            'rating': rating,
            'color': color
        })

    # Display ratings
    cols = st.columns(len(players))
    for idx, player_rating in enumerate(rating_data):
        with cols[idx]:
            st.markdown(f"""
            <div style="background: {player_rating['color']}; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h4 style="margin: 0; color: white;">{player_rating['player']}</h4>
                <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">
                    {player_rating['rating']}
                </p>
                <p style="margin: 0.25rem 0 0 0; color: white; font-size: 0.8rem;">
                    CV: {player_rating['cv']:.1f}% | PPG: {player_rating['ppg']:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)
