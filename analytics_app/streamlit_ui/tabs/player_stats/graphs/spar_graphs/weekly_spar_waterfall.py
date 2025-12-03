#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.data_access import list_player_seasons


@st.fragment
def display_weekly_spar_waterfall(prefix=""):
    """
    Weekly SPAR Waterfall - Week-by-week bars showing Player vs Manager SPAR

    Gap between bars = value left on bench
    Color coded: Green when captured, Red when missed
    """
    st.header("💧 Weekly SPAR Waterfall")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Week-by-week breakdown:</strong> See exactly which weeks you left value on bench.
    Green bars = captured, Red bars = missed opportunities.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection
    available_years = list_player_seasons()
    if not available_years:
        st.error("No player data found.")
        return

    selected_year = st.selectbox(
        "Select Season",
        options=sorted(available_years, reverse=True),
        key=f"{prefix}_waterfall_year"
    )

    # Player search
    player_search = st.text_input(
        "🔍 Search for player:",
        placeholder="e.g., Lamar Jackson",
        key=f"{prefix}_waterfall_search"
    ).strip()

    if not player_search:
        st.info("💡 Enter a player name to see their weekly SPAR breakdown")
        return

    # Load weekly data
    with st.spinner("Loading weekly data..."):
        filters = {
            "year": [int(selected_year)],
            "player_query": player_search,
            "rostered_only": True
        }

        weekly_data = load_filtered_weekly_player_data(filters=filters, limit=1000)

        if weekly_data is None or weekly_data.empty:
            st.warning(f"No data found for '{player_search}' in {selected_year}")
            return

    # Check for SPAR columns
    if 'player_weekly_spar' not in weekly_data.columns or 'manager_weekly_spar' not in weekly_data.columns:
        st.error("Weekly SPAR data not available. Make sure SPAR calculations have been run.")
        return

    # Convert to numeric
    weekly_data['player_weekly_spar'] = pd.to_numeric(weekly_data['player_weekly_spar'], errors='coerce')
    weekly_data['manager_weekly_spar'] = pd.to_numeric(weekly_data['manager_weekly_spar'], errors='coerce')
    weekly_data['week'] = pd.to_numeric(weekly_data['week'], errors='coerce')
    weekly_data['points'] = pd.to_numeric(weekly_data['points'], errors='coerce')

    # Get unique players (in case of partial matches)
    players = weekly_data['player'].unique()
    if len(players) > 1:
        player_name = st.selectbox(
            "Multiple players found - select one:",
            options=players,
            key=f"{prefix}_waterfall_player_select"
        )
        weekly_data = weekly_data[weekly_data['player'] == player_name]
    else:
        player_name = players[0]

    # Sort by week
    weekly_data = weekly_data.sort_values('week')

    # Calculate missed SPAR
    weekly_data['missed_spar'] = weekly_data['player_weekly_spar'] - weekly_data['manager_weekly_spar']

    st.subheader(f"📊 {player_name} - {selected_year}")

    # Create grouped bar chart
    fig = go.Figure()

    # Player SPAR (total available)
    fig.add_trace(go.Bar(
        x=weekly_data['week'],
        y=weekly_data['player_weekly_spar'],
        name='Player SPAR (Available)',
        marker_color='#3B82F6',
        opacity=0.6,
        hovertemplate='<b>Week %{x}</b><br>Player SPAR: %{y:.2f}<br><extra></extra>'
    ))

    # Manager SPAR (captured) - color based on if started
    colors = ['#10B981' if ms > 0 else '#6B7280' for ms in weekly_data['manager_weekly_spar']]

    fig.add_trace(go.Bar(
        x=weekly_data['week'],
        y=weekly_data['manager_weekly_spar'],
        name='Manager SPAR (Captured)',
        marker_color=colors,
        hovertemplate='<b>Week %{x}</b><br>Manager SPAR: %{y:.2f}<br><extra></extra>'
    ))

    fig.update_layout(
        barmode='group',
        xaxis_title="Week",
        yaxis_title="SPAR Value",
        height=500,
        template="plotly_white",
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=1)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_waterfall_bars")

    # Missed SPAR bar chart
    st.subheader("🔴 Missed SPAR by Week")

    fig2 = go.Figure()

    # Color negative values (over-captured) differently
    colors2 = ['#EF4444' if x > 0 else '#3B82F6' for x in weekly_data['missed_spar']]

    fig2.add_trace(go.Bar(
        x=weekly_data['week'],
        y=weekly_data['missed_spar'],
        marker_color=colors2,
        hovertemplate='<b>Week %{x}</b><br>Missed SPAR: %{y:.2f}<br><extra></extra>',
        showlegend=False
    ))

    fig2.update_layout(
        xaxis_title="Week",
        yaxis_title="Missed SPAR (Player - Manager)",
        height=400,
        template="plotly_white",
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )

    st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_waterfall_missed")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_player = weekly_data['player_weekly_spar'].sum()
    total_manager = weekly_data['manager_weekly_spar'].sum()
    total_missed = weekly_data['missed_spar'].sum()
    capture_pct = (total_manager / total_player * 100) if total_player > 0 else 0

    with col1:
        st.metric("Total Player SPAR", f"{total_player:.1f}")

    with col2:
        st.metric("Total Manager SPAR", f"{total_manager:.1f}")

    with col3:
        st.metric("Total Missed SPAR", f"{total_missed:.1f}")

    with col4:
        st.metric("Capture Rate", f"{capture_pct:.1f}%")

    # Detailed weekly table
    with st.expander("📋 Weekly Breakdown"):
        table_data = weekly_data[[
            'week', 'points', 'player_weekly_spar', 'manager_weekly_spar', 'missed_spar'
        ]].copy()

        table_data = table_data.rename(columns={
            'week': 'Week',
            'points': 'Points',
            'player_weekly_spar': 'Player SPAR',
            'manager_weekly_spar': 'Manager SPAR',
            'missed_spar': 'Missed SPAR'
        })

        # Format numbers
        for col in ['Points', 'Player SPAR', 'Manager SPAR', 'Missed SPAR']:
            table_data[col] = table_data[col].apply(lambda x: f"{x:.2f}")

        st.dataframe(table_data, hide_index=True, use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Understanding the Waterfall:**

        - **Blue bars**: Player SPAR (total value available that week)
        - **Green bars**: Manager SPAR (value you captured by starting them)
        - **Gray bars**: Week player was benched (Manager SPAR = 0)

        **Missed SPAR Chart:**
        - **Red bars**: Positive = left value on bench
        - **Blue bars**: Negative = over-captured (started when they underperformed)
        - **Zero line**: Perfect capture

        **Use This To:**
        - Identify specific weeks where you made mistakes
        - See patterns (always bench during bye weeks, etc.)
        - Learn which players you should trust more
        """)
