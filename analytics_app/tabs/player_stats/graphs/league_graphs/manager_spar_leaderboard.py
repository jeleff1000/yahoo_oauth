#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.core import list_player_seasons


@st.fragment
def display_manager_spar_leaderboard(prefix=""):
    """
    Manager SPAR Leaderboard - Ranked by total SPAR captured/missed

    Shows league-wide lineup management efficiency
    Stacked bars: Manager SPAR (captured) + Missed SPAR
    """
    st.header("🏆 Manager SPAR Leaderboard")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>League-wide efficiency:</strong> See who captured the most value and who left points on the bench.
    Green = captured SPAR, Red = missed opportunities.
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
            key=f"{prefix}_leaderboard_year"
        )

    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=["Total Player SPAR", "Manager SPAR Captured", "Missed SPAR", "Capture Rate %"],
            key=f"{prefix}_leaderboard_sort"
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
    if 'player_spar' not in weekly_data.columns or 'manager_spar' not in weekly_data.columns:
        st.error("SPAR data not available.")
        return

    # Convert to numeric
    weekly_data['player_spar'] = pd.to_numeric(weekly_data['player_spar'], errors='coerce')
    weekly_data['manager_spar'] = pd.to_numeric(weekly_data['manager_spar'], errors='coerce')

    # Aggregate to season level by player
    season_data = weekly_data.groupby('player').agg({
        'player_spar': 'sum',
        'manager_spar': 'sum',
        'nfl_position': 'first',
        'manager': 'first'
    }).reset_index()

    season_data.columns = ['player', 'player_spar', 'manager_spar', 'nfl_position', 'manager']

    # Filter valid data
    season_data = season_data[
        (season_data['player_spar'].notna()) &
        (season_data['manager_spar'].notna()) &
        (season_data['manager'].notna())
    ].copy()

    if season_data.empty:
        st.warning("No valid SPAR data for selected filters.")
        return

    # Calculate missed SPAR
    season_data['missed_spar'] = season_data['player_spar'] - season_data['manager_spar']

    # Aggregate by manager
    manager_stats = season_data.groupby('manager').agg({
        'player_spar': 'sum',
        'manager_spar': 'sum',
        'missed_spar': 'sum'
    }).reset_index()

    manager_stats.columns = ['manager', 'total_player_spar', 'total_manager_spar', 'total_missed_spar']

    # Calculate capture rate
    manager_stats['capture_rate'] = (
        (manager_stats['total_manager_spar'] / manager_stats['total_player_spar'] * 100)
        .fillna(0)
    )

    # Sort based on user selection
    sort_column_map = {
        "Total Player SPAR": 'total_player_spar',
        "Manager SPAR Captured": 'total_manager_spar',
        "Missed SPAR": 'total_missed_spar',
        "Capture Rate %": 'capture_rate'
    }
    manager_stats = manager_stats.sort_values(sort_column_map[sort_by], ascending=False)

    # Create stacked bar chart
    fig = go.Figure()

    # Manager SPAR (captured) - green
    fig.add_trace(go.Bar(
        y=manager_stats['manager'],
        x=manager_stats['total_manager_spar'],
        name='Manager SPAR (Captured)',
        orientation='h',
        marker_color='#10B981',
        hovertemplate='<b>%{y}</b><br>Captured: %{x:.1f}<br><extra></extra>'
    ))

    # Missed SPAR - red
    fig.add_trace(go.Bar(
        y=manager_stats['manager'],
        x=manager_stats['total_missed_spar'],
        name='Missed SPAR (Benched)',
        orientation='h',
        marker_color='#EF4444',
        hovertemplate='<b>%{y}</b><br>Missed: %{x:.1f}<br><extra></extra>'
    ))

    fig.update_layout(
        barmode='stack',
        xaxis_title="SPAR Value",
        yaxis_title="Manager",
        height=max(400, len(manager_stats) * 30),
        template="plotly_white",
        hovermode='y unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_leaderboard_stacked")

    # Capture rate ranking
    st.subheader("📊 Capture Rate Rankings")

    capture_sorted = manager_stats.sort_values('capture_rate', ascending=False)

    fig2 = go.Figure()

    # Color code by capture rate
    colors = [
        '#10B981' if rate >= 85 else '#F59E0B' if rate >= 75 else '#EF4444'
        for rate in capture_sorted['capture_rate']
    ]

    fig2.add_trace(go.Bar(
        y=capture_sorted['manager'],
        x=capture_sorted['capture_rate'],
        orientation='h',
        marker_color=colors,
        text=capture_sorted['capture_rate'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Capture Rate: %{x:.1f}%<br><extra></extra>',
        showlegend=False
    ))

    fig2.update_layout(
        xaxis_title="Capture Rate (%)",
        yaxis_title="Manager",
        height=max(400, len(capture_sorted) * 30),
        template="plotly_white",
        hovermode='y unified',
        xaxis=dict(range=[0, 105])
    )

    st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_leaderboard_capture")

    # Detailed stats table
    st.subheader("📋 Detailed Statistics")

    table_data = manager_stats.copy()
    table_data = table_data.rename(columns={
        'manager': 'Manager',
        'total_player_spar': 'Player SPAR',
        'total_manager_spar': 'Manager SPAR',
        'total_missed_spar': 'Missed SPAR',
        'capture_rate': 'Capture %'
    })

    # Format numbers
    table_data['Player SPAR'] = table_data['Player SPAR'].apply(lambda x: f"{x:.1f}")
    table_data['Manager SPAR'] = table_data['Manager SPAR'].apply(lambda x: f"{x:.1f}")
    table_data['Missed SPAR'] = table_data['Missed SPAR'].apply(lambda x: f"{x:.1f}")
    table_data['Capture %'] = table_data['Capture %'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(table_data, hide_index=True, use_container_width=True)

    # Position breakdown
    with st.expander("📊 Breakdown by Position"):
        st.markdown("**Manager SPAR by Position**")

        # Aggregate by manager and position
        if 'nfl_position' in season_data.columns:
            position_breakdown = season_data.groupby(['manager', 'nfl_position']).agg({
                'player_spar': 'sum',
                'manager_spar': 'sum',
                'missed_spar': 'sum'
            }).reset_index()

            position_breakdown['capture_rate'] = (
                (position_breakdown['manager_spar'] / position_breakdown['player_spar'] * 100)
                .fillna(0)
            )

            # Pivot for display
            pivot_player = position_breakdown.pivot(
                index='manager',
                columns='nfl_position',
                values='player_spar'
            ).fillna(0)

            pivot_manager = position_breakdown.pivot(
                index='manager',
                columns='nfl_position',
                values='manager_spar'
            ).fillna(0)

            pivot_capture = position_breakdown.pivot(
                index='manager',
                columns='nfl_position',
                values='capture_rate'
            ).fillna(0)

            tab1, tab2, tab3 = st.tabs(["Player SPAR", "Manager SPAR", "Capture Rate"])

            with tab1:
                st.dataframe(pivot_player.style.format("{:.1f}"), use_container_width=True)

            with tab2:
                st.dataframe(pivot_manager.style.format("{:.1f}"), use_container_width=True)

            with tab3:
                st.dataframe(pivot_capture.style.format("{:.1f}%"), use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Understanding the Leaderboard:**

        - **Green bars**: Manager SPAR (value captured by starting players)
        - **Red bars**: Missed SPAR (value left on bench)
        - **Total bar width**: Total Player SPAR (roster talent)

        **Capture Rate:**
        - **85%+** = Excellent lineup management (green)
        - **75-85%** = Good management (yellow)
        - **<75%** = Poor management (red)

        **What This Shows:**

        1. **Wide bars** = More rostered talent (better draft/waivers)
        2. **More green** = Better lineup decisions (started the right players)
        3. **More red** = Left value on bench (poor start/sit choices)

        **Sorting Options:**

        - **Total Player SPAR**: Best roster talent (draft + waivers)
        - **Manager SPAR Captured**: Most value actually started
        - **Missed SPAR**: Most value left on bench (biggest mistakes)
        - **Capture Rate %**: Best lineup management efficiency

        **Use This To:**
        - Compare your roster management to league
        - Identify managers who draft well but manage poorly
        - Find managers who maximize their roster talent
        - Learn from efficient managers' strategies
        """)
