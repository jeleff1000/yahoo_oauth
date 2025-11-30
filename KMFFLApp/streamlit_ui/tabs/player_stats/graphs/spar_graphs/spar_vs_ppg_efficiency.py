#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.data_access import load_players_season_data, list_player_seasons


@st.fragment
def display_spar_vs_ppg_efficiency(prefix=""):
    """
    SPAR vs PPG Efficiency scatter

    X-axis: PPG
    Y-axis: Player SPAR
    Diagonal reference line: Expected SPAR for that PPG
    Above line = efficient, Below line = volume-based
    """
    st.header("⚡ SPAR vs PPG Efficiency")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Value vs Volume:</strong> High SPAR + Lower PPG = efficient scorer.
    High PPG + Lower SPAR = volume-dependent player.
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
            key=f"{prefix}_efficiency_year"
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_efficiency_position"
        )

    # Minimum games filter
    min_games = st.slider(
        "Minimum Games Played",
        min_value=1,
        max_value=17,
        value=8,
        key=f"{prefix}_efficiency_min_games"
    )

    # Load weekly SPAR data
    with st.spinner("Loading SPAR data..."):
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
    if 'player_spar' not in weekly_data.columns:
        st.error("SPAR data not available.")
        return

    # Convert to numeric
    weekly_data['player_spar'] = pd.to_numeric(weekly_data['player_spar'], errors='coerce')

    # Aggregate SPAR to season level
    spar_data = weekly_data.groupby('player').agg({
        'player_spar': 'sum',
        'nfl_position': 'first'
    }).reset_index()
    spar_data.columns = ['player', 'player_spar', 'nfl_position']

    # Load season data for PPG
    season_data = load_players_season_data(
        year=[int(selected_year)],
        rostered_only=True
    )

    if season_data is None or season_data.empty:
        st.warning(f"No season data found for {selected_year}")
        return

    # Merge SPAR with PPG data
    season_data = season_data.merge(spar_data, on='player', how='inner', suffixes=('', '_spar'))

    # Use nfl_position from SPAR data if available
    if 'nfl_position_spar' in season_data.columns:
        season_data['nfl_position'] = season_data['nfl_position_spar']

    # Check for required columns
    if 'season_ppg' not in season_data.columns:
        st.error("PPG data not available.")
        return

    # Convert to numeric
    season_data['player_spar'] = pd.to_numeric(season_data['player_spar'], errors='coerce')
    season_data['season_ppg'] = pd.to_numeric(season_data['season_ppg'], errors='coerce')
    season_data['fantasy_games'] = pd.to_numeric(season_data.get('fantasy_games', season_data.get('games_played', 0)), errors='coerce')

    # Filter by minimum games and valid data
    season_data = season_data[
        (season_data['fantasy_games'] >= min_games) &
        (season_data['player_spar'].notna()) &
        (season_data['season_ppg'].notna()) &
        (season_data['season_ppg'] > 0)
    ].copy()

    if season_data.empty:
        st.warning(f"No data for selected filters (min {min_games} games).")
        return

    # Calculate efficiency ratio
    # Higher ratio = more SPAR per point scored = more efficient
    season_data['efficiency_ratio'] = season_data['player_spar'] / season_data['season_ppg']

    # Create scatter plot
    fig = go.Figure()

    if position == "All":
        positions = season_data['nfl_position'].unique()
        colors = {'QB': '#3B82F6', 'RB': '#10B981', 'WR': '#F59E0B', 'TE': '#EF4444', 'K': '#8B5CF6', 'DEF': '#6B7280'}

        for pos in positions:
            pos_data = season_data[season_data['nfl_position'] == pos]
            fig.add_trace(go.Scatter(
                x=pos_data['season_ppg'],
                y=pos_data['player_spar'],
                mode='markers',
                name=pos,
                marker=dict(
                    size=10,
                    color=colors.get(pos, '#6B7280'),
                    line=dict(width=1, color='white')
                ),
                text=pos_data['player'],
                customdata=pos_data[['efficiency_ratio', 'fantasy_games']],
                hovertemplate='<b>%{text}</b><br>' +
                             'PPG: %{x:.2f}<br>' +
                             'Player SPAR: %{y:.2f}<br>' +
                             'Efficiency: %{customdata[0]:.3f}<br>' +
                             'Games: %{customdata[1]}<br>' +
                             '<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=season_data['season_ppg'],
            y=season_data['player_spar'],
            mode='markers',
            marker=dict(
                size=12,
                color=season_data['efficiency_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="SPAR/PPG"),
                line=dict(width=1, color='white')
            ),
            text=season_data['player'],
            customdata=season_data[['efficiency_ratio', 'fantasy_games']],
            hovertemplate='<b>%{text}</b><br>' +
                         'PPG: %{x:.2f}<br>' +
                         'Player SPAR: %{y:.2f}<br>' +
                         'Efficiency: %{customdata[0]:.3f}<br>' +
                         'Games: %{customdata[1]}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

    # Add trendline (simple linear regression)
    from numpy import polyfit, poly1d
    x = season_data['season_ppg'].values
    y = season_data['player_spar'].values
    z = polyfit(x, y, 1)
    p = poly1d(z)

    x_trend = [x.min(), x.max()]
    y_trend = [p(x.min()), p(x.max())]

    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        line=dict(color='gray', dash='dash', width=2),
        name='Expected SPAR',
        hovertemplate='Expected SPAR<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Points Per Game (PPG)",
        yaxis_title="Player SPAR (Value Above Replacement)",
        height=600,
        template="plotly_white",
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_efficiency_scatter")

    # Top/Bottom performers
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌟 Most Efficient (High SPAR/PPG)")
        top_efficiency = season_data.nlargest(10, 'efficiency_ratio')[[
            'player', 'nfl_position', 'season_ppg', 'player_spar', 'efficiency_ratio'
        ]].copy()

        top_efficiency = top_efficiency.rename(columns={
            'player': 'Player',
            'nfl_position': 'Pos',
            'season_ppg': 'PPG',
            'player_spar': 'SPAR',
            'efficiency_ratio': 'SPAR/PPG'
        })

        top_efficiency['PPG'] = top_efficiency['PPG'].apply(lambda x: f"{x:.2f}")
        top_efficiency['SPAR'] = top_efficiency['SPAR'].apply(lambda x: f"{x:.2f}")
        top_efficiency['SPAR/PPG'] = top_efficiency['SPAR/PPG'].apply(lambda x: f"{x:.3f}")

        st.dataframe(top_efficiency, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("📉 Least Efficient (Low SPAR/PPG)")
        bottom_efficiency = season_data.nsmallest(10, 'efficiency_ratio')[[
            'player', 'nfl_position', 'season_ppg', 'player_spar', 'efficiency_ratio'
        ]].copy()

        bottom_efficiency = bottom_efficiency.rename(columns={
            'player': 'Player',
            'nfl_position': 'Pos',
            'season_ppg': 'PPG',
            'player_spar': 'SPAR',
            'efficiency_ratio': 'SPAR/PPG'
        })

        bottom_efficiency['PPG'] = bottom_efficiency['PPG'].apply(lambda x: f"{x:.2f}")
        bottom_efficiency['SPAR'] = bottom_efficiency['SPAR'].apply(lambda x: f"{x:.2f}")
        bottom_efficiency['SPAR/PPG'] = bottom_efficiency['SPAR/PPG'].apply(lambda x: f"{x:.3f}")

        st.dataframe(bottom_efficiency, hide_index=True, use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Understanding SPAR vs PPG Efficiency:**

        - **X-axis**: Points Per Game (raw scoring volume)
        - **Y-axis**: Player SPAR (value above replacement)
        - **Trendline**: Expected SPAR for a given PPG level

        **Interpreting the Scatter:**

        1. **Above the trendline** = Efficient scorers
           - High SPAR relative to their PPG
           - Scoring in situations that matter more
           - Less touchdown-dependent, more consistent base production

        2. **On the trendline** = Expected efficiency
           - SPAR matches what you'd expect for their PPG
           - Standard value for their scoring level

        3. **Below the trendline** = Volume-dependent
           - Lower SPAR relative to their PPG
           - May rely on TDs or boom weeks
           - Less consistent value generation

        **Efficiency Ratio (SPAR/PPG):**
        - Higher = More efficient (more value per point scored)
        - Lower = Less efficient (points from less valuable situations)

        **Use This To:**
        - Find undervalued players (high efficiency, moderate PPG)
        - Identify touchdown-dependent players (low efficiency, high PPG)
        - Evaluate consistency vs volume scorers
        - Target draft sleepers (efficient scorers flying under the radar)
        """)
