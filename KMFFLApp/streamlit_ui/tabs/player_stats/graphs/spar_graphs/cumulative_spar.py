#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.tab_data_access.players.weekly_player_data import load_filtered_weekly_player_data
from md.data_access import list_player_seasons


@st.fragment
def display_cumulative_spar_over_season(prefix=""):
    """
    Cumulative SPAR Over Season - Running total area chart

    Multiple lines: Player SPAR vs Manager SPAR accumulating week-by-week
    Divergence shows when you failed to capture value
    """
    st.header("📈 Cumulative SPAR Over Season")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Value accumulation:</strong> See how SPAR built up week-by-week.
    Diverging lines = missed opportunities. Parallel lines = perfect capture.
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
        key=f"{prefix}_cumulative_year"
    )

    # Player search
    player_search = st.text_input(
        "🔍 Search for players (comma separated, up to 5):",
        placeholder="e.g., Christian McCaffrey, Tyreek Hill",
        key=f"{prefix}_cumulative_search"
    ).strip()

    if not player_search:
        st.info("💡 Enter player name(s) to see cumulative SPAR")
        return

    # Parse players
    search_names = [name.strip() for name in player_search.split(",") if name.strip()]
    if len(search_names) > 5:
        st.warning("⚠️ Limiting to first 5 players")
        search_names = search_names[:5]

    # Load weekly data
    with st.spinner("Loading weekly data..."):
        filters = {
            "year": [int(selected_year)],
            "rostered_only": True
        }

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

    # Check for SPAR columns
    if 'player_weekly_spar' not in filtered.columns or 'manager_weekly_spar' not in filtered.columns:
        st.error("Weekly SPAR data not available.")
        return

    # Convert to numeric
    filtered['player_weekly_spar'] = pd.to_numeric(filtered['player_weekly_spar'], errors='coerce')
    filtered['manager_weekly_spar'] = pd.to_numeric(filtered['manager_weekly_spar'], errors='coerce')
    filtered['week'] = pd.to_numeric(filtered['week'], errors='coerce')

    players = filtered['player'].unique()
    st.success(f"✅ Found {len(players)} player(s)")

    # Create dual-axis chart with Player SPAR and Manager SPAR
    fig = go.Figure()

    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

    for idx, player_name in enumerate(players):
        player_df = filtered[filtered['player'] == player_name].sort_values('week')

        # Calculate cumulative sums
        player_df['cumulative_player_spar'] = player_df['player_weekly_spar'].cumsum()
        player_df['cumulative_manager_spar'] = player_df['manager_weekly_spar'].cumsum()

        color = colors[idx % len(colors)]

        # Player SPAR line (solid)
        fig.add_trace(go.Scatter(
            x=player_df['week'],
            y=player_df['cumulative_player_spar'],
            mode='lines+markers',
            name=f"{player_name} (Player SPAR)",
            line=dict(color=color, width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{fullData.name}</b><br>Week %{x}<br>Cumulative: %{y:.2f}<br><extra></extra>'
        ))

        # Manager SPAR line (dashed)
        fig.add_trace(go.Scatter(
            x=player_df['week'],
            y=player_df['cumulative_manager_spar'],
            mode='lines+markers',
            name=f"{player_name} (Manager SPAR)",
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='<b>%{fullData.name}</b><br>Week %{x}<br>Cumulative: %{y:.2f}<br><extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Cumulative SPAR",
        height=600,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis=dict(tickmode='linear', dtick=1)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_cumulative_lines")

    # Gap analysis (divergence)
    st.subheader("📊 Value Gap Analysis")

    for player_name in players:
        player_df = filtered[filtered['player'] == player_name].sort_values('week')

        # Calculate cumulative sums
        player_df['cumulative_player_spar'] = player_df['player_weekly_spar'].cumsum()
        player_df['cumulative_manager_spar'] = player_df['manager_weekly_spar'].cumsum()
        player_df['cumulative_gap'] = player_df['cumulative_player_spar'] - player_df['cumulative_manager_spar']

        final_gap = player_df['cumulative_gap'].iloc[-1] if not player_df.empty else 0

        # Create gap chart
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=player_df['week'],
            y=player_df['cumulative_gap'],
            fill='tozeroy',
            mode='lines',
            line=dict(color='#EF4444', width=2),
            hovertemplate=f'<b>{player_name}</b><br>Week %{{x}}<br>Cumulative Gap: %{{y:.2f}}<br><extra></extra>'
        ))

        fig2.update_layout(
            title=f"{player_name} - Cumulative Missed SPAR",
            xaxis_title="Week",
            yaxis_title="Cumulative Gap (Player - Manager SPAR)",
            height=300,
            template="plotly_white",
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        )

        st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_gap_{player_name}")

        st.caption(f"**Final Gap**: {final_gap:.2f} SPAR missed over the season")

    # Summary table
    with st.expander("📋 Season Totals"):
        summary_data = []

        for player_name in players:
            player_df = filtered[filtered['player'] == player_name]

            total_player = player_df['player_weekly_spar'].sum()
            total_manager = player_df['manager_weekly_spar'].sum()
            total_missed = total_player - total_manager
            capture_pct = (total_manager / total_player * 100) if total_player > 0 else 0

            summary_data.append({
                'Player': player_name,
                'Player SPAR': f"{total_player:.2f}",
                'Manager SPAR': f"{total_manager:.2f}",
                'Missed SPAR': f"{total_missed:.2f}",
                'Capture %': f"{capture_pct:.1f}%"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Explanation
    with st.expander("❓ How to Read This Chart"):
        st.markdown("""
        **Understanding Cumulative SPAR:**

        - **Solid lines**: Player SPAR (total value available week-by-week)
        - **Dashed lines**: Manager SPAR (value you captured)
        - **Gap between lines**: Missed opportunities accumulating over time

        **Patterns to Look For:**

        1. **Parallel lines**: Perfect capture - you started them every week
        2. **Diverging lines**: Missed value - you benched them during boom weeks
        3. **Converging lines**: Catching up - you started using them more effectively
        4. **Flat Manager line**: Player on your bench all season

        **Gap Analysis Chart:**
        - Shows cumulative missed SPAR over time
        - Growing gap = consistently missing value
        - Flat gap = good capture after initial mistakes

        **Use This To:**
        - See when players became valuable (steep slopes)
        - Identify when you failed to adapt (widening gaps)
        - Compare multiple players' value trajectories
        """)
