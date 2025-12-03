#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.core import list_player_seasons


@st.fragment
def display_spar_consistency_scatter(prefix=""):
    """
    SPAR Consistency Scatter - Total Player SPAR vs Standard Deviation

    X-axis: Total Player SPAR
    Y-axis: Std Dev of weekly SPAR
    Shows high SPAR + Low StdDev = consistent value; High SPAR + High StdDev = boom/bust
    """
    st.header("📊 SPAR Consistency Analysis")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Find reliable players:</strong> High SPAR + Low volatility = consistent value.
    High SPAR + High volatility = boom/bust player.
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
            key=f"{prefix}_consistency_year"
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_consistency_position"
        )

    # Minimum games filter
    min_games = st.slider(
        "Minimum Games Played",
        min_value=1,
        max_value=17,
        value=8,
        key=f"{prefix}_consistency_min_games",
        help="Filter to players who played at least this many games"
    )

    # Load weekly data
    with st.spinner("Loading weekly data..."):
        filters = {
            "year": [int(selected_year)],
            "rostered_only": True
        }
        if position != "All":
            filters["nfl_position"] = [position]

        weekly_data = load_filtered_weekly_player_data(filters=filters, limit=50000)

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for {selected_year}")
            return

    # Check for SPAR columns
    if 'player_weekly_spar' not in weekly_data.columns:
        st.error("Weekly SPAR data not available.")
        return

    # Convert to numeric
    weekly_data['player_weekly_spar'] = pd.to_numeric(weekly_data['player_weekly_spar'], errors='coerce')
    weekly_data['points'] = pd.to_numeric(weekly_data['points'], errors='coerce')

    # Calculate statistics per player
    player_stats = weekly_data.groupby('player').agg({
        'player_weekly_spar': ['sum', 'std', 'count', 'mean'],
        'points': 'sum',
        'nfl_position': 'first',
        'manager': 'first'
    }).reset_index()

    player_stats.columns = ['player', 'total_spar', 'spar_std', 'games', 'avg_spar', 'total_points', 'position', 'manager']

    # Filter by minimum games
    player_stats = player_stats[player_stats['games'] >= min_games].copy()

    if player_stats.empty:
        st.warning(f"No players with at least {min_games} games.")
        return

    # Fill NaN std (players with only 1 game)
    player_stats['spar_std'] = player_stats['spar_std'].fillna(0)

    # Calculate coefficient of variation
    player_stats['cv'] = np.where(
        player_stats['avg_spar'] > 0,
        (player_stats['spar_std'] / player_stats['avg_spar']) * 100,
        0
    )

    # Create scatter plot
    fig = go.Figure()

    if position == "All":
        positions = player_stats['position'].unique()
        colors = {'QB': '#3B82F6', 'RB': '#10B981', 'WR': '#F59E0B', 'TE': '#EF4444', 'K': '#8B5CF6', 'DEF': '#6B7280'}

        for pos in positions:
            pos_data = player_stats[player_stats['position'] == pos]
            fig.add_trace(go.Scatter(
                x=pos_data['total_spar'],
                y=pos_data['spar_std'],
                mode='markers',
                name=pos,
                marker=dict(
                    size=10,
                    color=colors.get(pos, '#6B7280'),
                    line=dict(width=1, color='white')
                ),
                text=pos_data['player'],
                customdata=pos_data[['avg_spar', 'cv', 'games']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Total SPAR: %{x:.2f}<br>' +
                             'Std Dev: %{y:.2f}<br>' +
                             'Avg SPAR/gm: %{customdata[0]:.2f}<br>' +
                             'CV: %{customdata[1]:.1f}%<br>' +
                             'Games: %{customdata[2]}<br>' +
                             '<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=player_stats['total_spar'],
            y=player_stats['spar_std'],
            mode='markers',
            marker=dict(
                size=12,
                color=player_stats['cv'],
                colorscale='RdYlGn_r',  # Red = high CV (volatile), Green = low CV (consistent)
                showscale=True,
                colorbar=dict(title="CV %"),
                line=dict(width=1, color='white')
            ),
            text=player_stats['player'],
            customdata=player_stats[['avg_spar', 'cv', 'games']],
            hovertemplate='<b>%{text}</b><br>' +
                         'Total SPAR: %{x:.2f}<br>' +
                         'Std Dev: %{y:.2f}<br>' +
                         'Avg SPAR/gm: %{customdata[0]:.2f}<br>' +
                         'CV: %{customdata[1]:.1f}%<br>' +
                         'Games: %{customdata[2]}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title="Total Player SPAR (Season Value)",
        yaxis_title="Standard Deviation (Volatility)",
        height=600,
        template="plotly_white",
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_consistency_scatter")

    # Quadrant analysis
    median_spar = player_stats['total_spar'].median()
    median_std = player_stats['spar_std'].median()

    # Categorize players
    player_stats['category'] = 'Unknown'
    player_stats.loc[
        (player_stats['total_spar'] >= median_spar) & (player_stats['spar_std'] < median_std),
        'category'
    ] = '🌟 Elite Consistent'
    player_stats.loc[
        (player_stats['total_spar'] >= median_spar) & (player_stats['spar_std'] >= median_std),
        'category'
    ] = '💥 High Value Boom/Bust'
    player_stats.loc[
        (player_stats['total_spar'] < median_spar) & (player_stats['spar_std'] < median_std),
        'category'
    ] = '📊 Low Value Consistent'
    player_stats.loc[
        (player_stats['total_spar'] < median_spar) & (player_stats['spar_std'] >= median_std),
        'category'
    ] = '❌ Low Value Volatile'

    # Display categories
    st.subheader("📋 Player Categories")

    tabs = st.tabs(["🌟 Elite Consistent", "💥 Boom/Bust", "📊 Low Consistent", "❌ Volatile"])

    categories = [
        ('🌟 Elite Consistent', 'High SPAR + Low volatility = reliable studs'),
        ('💥 High Value Boom/Bust', 'High SPAR but inconsistent = risky stars'),
        ('📊 Low Value Consistent', 'Low SPAR but steady = safe floor players'),
        ('❌ Low Value Volatile', 'Low SPAR + high volatility = bench/drop')
    ]

    for tab, (cat, desc) in zip(tabs, categories):
        with tab:
            st.caption(desc)
            cat_data = player_stats[player_stats['category'] == cat].copy()

            if not cat_data.empty:
                cat_data = cat_data.sort_values('total_spar', ascending=False)[['player', 'position', 'total_spar', 'spar_std', 'cv', 'games']].head(10)

                cat_data = cat_data.rename(columns={
                    'player': 'Player',
                    'position': 'Pos',
                    'total_spar': 'Total SPAR',
                    'spar_std': 'Std Dev',
                    'cv': 'CV %',
                    'games': 'Games'
                })

                cat_data['Total SPAR'] = cat_data['Total SPAR'].apply(lambda x: f"{x:.2f}")
                cat_data['Std Dev'] = cat_data['Std Dev'].apply(lambda x: f"{x:.2f}")
                cat_data['CV %'] = cat_data['CV %'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(cat_data, hide_index=True, use_container_width=True)
            else:
                st.info("No players in this category")

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Understanding SPAR Consistency:**

        - **X-axis**: Total Player SPAR (how much value above replacement they provided all season)
        - **Y-axis**: Standard Deviation (how much their weekly SPAR varied)
        - **Color (when single position)**: Coefficient of Variation % (relative volatility)

        **The Four Quadrants:**

        1. **🌟 Elite Consistent** (Top-Right, Low Volatility)
           - High total SPAR + Low week-to-week variance
           - These are your draft targets and keepers
           - Reliable studs who produce value every week

        2. **💥 High Value Boom/Bust** (Top-Right, High Volatility)
           - High total SPAR but inconsistent weekly performance
           - Great for best ball, risky for season-long
           - Can win weeks but also cost you playoffs

        3. **📊 Low Value Consistent** (Bottom-Left, Low Volatility)
           - Low total SPAR but steady week-to-week
           - Good bye-week fill-ins, safe floor
           - Won't win you weeks but won't lose them either

        4. **❌ Low Value Volatile** (Bottom-Left, High Volatility)
           - Low SPAR + high variance = worst category
           - Bench/drop candidates
           - Can't be trusted and don't provide enough upside

        **CV% (Coefficient of Variation):**
        - Lower is better (more consistent)
        - <30% = very consistent
        - 30-50% = moderate variance
        - >50% = boom/bust
        """)
