#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.data_access import list_player_seasons


@st.fragment
def display_position_spar_boxplot(prefix=""):
    """
    Position SPAR Box Plot - Distribution of Player SPAR by position

    Shows which positions generate most value above replacement
    Helps identify scarcity and draft strategy priorities
    """
    st.header("📦 Position SPAR Distribution")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Position Scarcity Analysis:</strong> See which positions have the most value above replacement.
    Wider boxes = more volatility, Higher medians = more valuable position.
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
            key=f"{prefix}_boxplot_year"
        )

    with col2:
        min_games = st.slider(
            "Minimum Games Played",
            min_value=1,
            max_value=17,
            value=8,
            key=f"{prefix}_boxplot_min_games",
            help="Filter to players who played at least this many games"
        )

    # Load weekly data
    with st.spinner("Loading data..."):
        filters = {
            "year": [int(selected_year)],
            "rostered_only": True
        }

        weekly_data = load_filtered_weekly_player_data(filters=filters, limit=50000)

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for {selected_year}")
            return

    # Check for SPAR columns
    if 'player_spar' not in weekly_data.columns:
        st.error("Player SPAR data not available.")
        return

    # Convert to numeric
    weekly_data['player_spar'] = pd.to_numeric(weekly_data['player_spar'], errors='coerce')

    # Aggregate to season level
    season_data = weekly_data.groupby('player').agg({
        'player_spar': 'sum',
        'nfl_position': 'first',
        'week': 'count'  # count weeks as games played
    }).reset_index()

    season_data.columns = ['player', 'player_spar', 'nfl_position', 'fantasy_games']

    # Filter by minimum games and valid data
    season_data = season_data[
        (season_data['fantasy_games'] >= min_games) &
        (season_data['player_spar'].notna())
    ].copy()

    if season_data.empty:
        st.warning(f"No data for selected filters (min {min_games} games).")
        return

    # Create box plot
    fig = go.Figure()

    # Define position order and colors
    position_order = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    colors = {
        'QB': '#3B82F6',
        'RB': '#10B981',
        'WR': '#F59E0B',
        'TE': '#EF4444',
        'K': '#8B5CF6',
        'DEF': '#6B7280'
    }

    # Filter to positions that exist in data
    positions = [p for p in position_order if p in season_data['nfl_position'].unique()]

    for pos in positions:
        pos_data = season_data[season_data['nfl_position'] == pos]

        fig.add_trace(go.Box(
            y=pos_data['player_spar'],
            name=pos,
            marker_color=colors.get(pos, '#6B7280'),
            boxmean='sd',  # Show mean and standard deviation
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Player SPAR: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        yaxis_title="Player SPAR (Value Above Replacement)",
        xaxis_title="Position",
        height=600,
        template="plotly_white",
        hovermode='closest',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_position_boxplot")

    # Summary statistics table
    st.subheader("📊 Position Statistics")

    summary_stats = []

    for pos in positions:
        pos_data = season_data[season_data['nfl_position'] == pos]['player_spar']

        summary_stats.append({
            'Position': pos,
            'Count': len(pos_data),
            'Mean': pos_data.mean(),
            'Median': pos_data.median(),
            'Std Dev': pos_data.std(),
            'Min': pos_data.min(),
            'Max': pos_data.max(),
            'Total SPAR': pos_data.sum()
        })

    summary_df = pd.DataFrame(summary_stats)

    # Format numbers
    summary_df['Mean'] = summary_df['Mean'].apply(lambda x: f"{x:.2f}")
    summary_df['Median'] = summary_df['Median'].apply(lambda x: f"{x:.2f}")
    summary_df['Std Dev'] = summary_df['Std Dev'].apply(lambda x: f"{x:.2f}")
    summary_df['Min'] = summary_df['Min'].apply(lambda x: f"{x:.2f}")
    summary_df['Max'] = summary_df['Max'].apply(lambda x: f"{x:.2f}")
    summary_df['Total SPAR'] = summary_df['Total SPAR'].apply(lambda x: f"{x:.1f}")

    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Top performers by position
    with st.expander("🌟 Top 5 by Position"):
        cols = st.columns(len(positions))

        for idx, pos in enumerate(positions):
            with cols[idx]:
                st.markdown(f"**{pos}**")
                pos_top = season_data[season_data['nfl_position'] == pos].nlargest(5, 'player_spar')[['player', 'player_spar']].copy()

                pos_top = pos_top.rename(columns={'player': 'Player', 'player_spar': 'SPAR'})
                pos_top['SPAR'] = pos_top['SPAR'].apply(lambda x: f"{x:.1f}")

                st.dataframe(pos_top, hide_index=True, use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Understanding Position SPAR Distribution:**

        - **Box**: The middle 50% of players (25th to 75th percentile)
        - **Line inside box**: Median SPAR for that position
        - **Diamond**: Mean (average) SPAR
        - **Whiskers**: Range of typical values (1.5x IQR)
        - **Dots**: Outliers (exceptional performers)

        **What This Tells You:**

        1. **Higher boxes** = More valuable position overall
           - Positions with higher median SPAR are more important to roster

        2. **Wider boxes** = More volatility within position
           - Positions with wide boxes have bigger difference between studs and duds
           - Narrow boxes = more consistent/predictable position

        3. **Total SPAR** = League-wide value generated
           - Shows which positions contribute most to winning

        4. **Standard Deviation** = Consistency metric
           - Lower std dev = more predictable position
           - Higher std dev = bigger gap between elite and replacement

        **Draft Strategy Implications:**

        - **High median + High std dev** = Draft studs early (big advantage over replacement)
        - **High median + Low std dev** = Can wait (less dropoff in talent)
        - **Low median + High std dev** = Streaming/waiver wire position
        - **Low median + Low std dev** = Least important position

        **Use This To:**
        - Identify position scarcity for draft prep
        - Understand where elite talent matters most
        - Plan which positions to target early vs late
        - Validate your draft strategy against actual SPAR data
        """)
