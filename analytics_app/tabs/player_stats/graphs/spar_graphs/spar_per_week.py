#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.core import list_player_seasons


@st.fragment
def display_spar_per_week_played(prefix=""):
    """
    SPAR per Week Played scatter plot

    X-axis: Player SPAR/week (talent efficiency)
    Y-axis: Manager SPAR/week (usage efficiency)
    Diagonal line = perfect capture
    """
    st.header("📈 SPAR per Week Played")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Efficiency Analysis:</strong> Player SPAR/week (talent) vs Manager SPAR/week (usage).
    Points on diagonal = perfect capture. Below diagonal = benched during boom weeks.
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
            key=f"{prefix}_spar_week_year"
        )

    with col2:
        position = st.selectbox(
            "Select Position",
            options=["All", "QB", "RB", "WR", "TE", "K", "DEF"],
            key=f"{prefix}_spar_week_position"
        )

    # Load weekly data
    with st.spinner("Loading data..."):
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
    if 'player_spar' not in weekly_data.columns or 'manager_spar' not in weekly_data.columns:
        st.error("SPAR data not available.")
        return

    # Convert to numeric
    weekly_data['player_spar'] = pd.to_numeric(weekly_data['player_spar'], errors='coerce')
    weekly_data['manager_spar'] = pd.to_numeric(weekly_data['manager_spar'], errors='coerce')

    # Aggregate to season level
    season_data = weekly_data.groupby('player').agg({
        'player_spar': ['sum', 'count'],  # sum for total, count for weeks played
        'manager_spar': 'sum',
        'nfl_position': 'first',
        'manager': 'first'
    }).reset_index()

    season_data.columns = ['player', 'player_spar', 'weeks_played', 'manager_spar', 'nfl_position', 'manager']

    # Count weeks started (where manager_spar > 0 in weekly data)
    weeks_started = weekly_data[weekly_data['manager_spar'] > 0].groupby('player').size().reset_index(name='weeks_started')
    season_data = season_data.merge(weeks_started, on='player', how='left')
    season_data['weeks_started'] = season_data['weeks_started'].fillna(0)

    # Filter to players with data
    season_data = season_data[
        (season_data['player_spar'].notna()) &
        (season_data['weeks_played'] > 0) &
        (season_data['weeks_started'] > 0)
    ].copy()

    if season_data.empty:
        st.warning("No valid SPAR data for selected filters.")
        return

    # Calculate SPAR per week
    season_data['player_spar_per_week'] = season_data['player_spar'] / season_data['weeks_played']
    season_data['manager_spar_per_week'] = season_data['manager_spar'] / season_data['weeks_started']

    # Create scatter plot
    fig = go.Figure()

    # Color by position if "All" selected
    if position == "All":
        positions = season_data['nfl_position'].unique()
        colors = {'QB': '#3B82F6', 'RB': '#10B981', 'WR': '#F59E0B', 'TE': '#EF4444', 'K': '#8B5CF6', 'DEF': '#6B7280'}

        for pos in positions:
            pos_data = season_data[season_data['nfl_position'] == pos]
            fig.add_trace(go.Scatter(
                x=pos_data['player_spar_per_week'],
                y=pos_data['manager_spar_per_week'],
                mode='markers',
                name=pos,
                marker=dict(
                    size=10,
                    color=colors.get(pos, '#6B7280'),
                    line=dict(width=1, color='white')
                ),
                text=pos_data['player'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Player SPAR/wk: %{x:.2f}<br>' +
                             'Manager SPAR/wk: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=season_data['player_spar_per_week'],
            y=season_data['manager_spar_per_week'],
            mode='markers',
            marker=dict(
                size=12,
                color=season_data['player_spar'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Total<br>Player<br>SPAR"),
                line=dict(width=1, color='white')
            ),
            text=season_data['player'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Player SPAR/wk: %{x:.2f}<br>' +
                         'Manager SPAR/wk: %{y:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ))

    # Add diagonal reference line (perfect capture)
    max_val = max(
        season_data['player_spar_per_week'].max(),
        season_data['manager_spar_per_week'].max()
    )
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash', width=2),
        name='Perfect Capture',
        hovertemplate='Perfect Capture Line<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Player SPAR per Week (Talent Efficiency)",
        yaxis_title="Manager SPAR per Week (Usage Efficiency)",
        height=600,
        template="plotly_white",
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_spar_week_scatter")

    # Summary stats
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_player = season_data['player_spar_per_week'].mean()
        st.metric("Avg Player SPAR/wk", f"{avg_player:.2f}")

    with col2:
        avg_manager = season_data['manager_spar_per_week'].mean()
        st.metric("Avg Manager SPAR/wk", f"{avg_manager:.2f}")

    with col3:
        capture_rate = (avg_manager / avg_player * 100) if avg_player > 0 else 0
        st.metric("Overall Capture %", f"{capture_rate:.1f}%")

    # Top performers table
    with st.expander("📊 Top Performers by SPAR/Week"):
        top_data = season_data.nlargest(15, 'player_spar_per_week')[[
            'player', 'nfl_position', 'player_spar_per_week', 'manager_spar_per_week', 'weeks_played', 'weeks_started'
        ]].copy()

        top_data['capture_rate'] = (top_data['manager_spar_per_week'] / top_data['player_spar_per_week'] * 100).round(1)

        top_data = top_data.rename(columns={
            'player': 'Player',
            'nfl_position': 'Pos',
            'player_spar_per_week': 'Player SPAR/wk',
            'manager_spar_per_week': 'Manager SPAR/wk',
            'weeks_played': 'Wks Played',
            'weeks_started': 'Wks Started',
            'capture_rate': 'Capture %'
        })

        top_data['Player SPAR/wk'] = top_data['Player SPAR/wk'].apply(lambda x: f"{x:.2f}")
        top_data['Manager SPAR/wk'] = top_data['Manager SPAR/wk'].apply(lambda x: f"{x:.2f}")
        top_data['Capture %'] = top_data['Capture %'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(top_data, hide_index=True, use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Reading the Scatter Plot:**

        - **X-axis (Player SPAR/week)**: Player's talent/production per week played
        - **Y-axis (Manager SPAR/week)**: Value captured per week started
        - **Diagonal line**: Perfect capture (manager captured all available value)

        **Interpreting Points:**
        - **On the line**: Perfect lineup management
        - **Below the line**: Left value on bench (benched during boom weeks)
        - **Above the line**: Got lucky (started during player's best weeks, benched during duds)

        **Use This To:**
        - Identify high-talent players you under-utilized
        - See if you're timing starts correctly
        - Find consistent value generators (high X, high Y)
        """)
