#!/usr/bin/env python3
"""
Drop Regret Analysis Tab

Analyzes dropped players who went on to produce significant value for other teams.
Uses pre-computed drop_regret_score and drop_regret_tier from the transactions pipeline.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Try to import chart theming
try:
    from shared.chart_themes import (
        apply_chart_theme,
        get_chart_colors,
        create_regret_bar_chart,
        create_horizontal_bar_chart,
        REGRET_COLORS,
    )
    HAS_CHART_THEMES = True
except ImportError:
    HAS_CHART_THEMES = False
    REGRET_COLORS = {
        'No Regret': '#2ca02c',
        'Minor Regret': '#98df8a',
        'Some Regret': '#ffbb78',
        'Big Regret': '#ff7f0e',
        'Major Regret': '#d62728',
        'Disaster': '#8b0000',
    }


def get_regret_emoji(tier: str) -> str:
    """Get emoji for regret tier."""
    emojis = {
        'No Regret': '✅',
        'Minor Regret': '😐',
        'Some Regret': '😕',
        'Big Regret': '😬',
        'Major Regret': '😰',
        'Disaster': '💀',
    }
    return emojis.get(tier, '❓')


def get_regret_description(tier: str) -> str:
    """Get description for regret tier."""
    descriptions = {
        'No Regret': 'Player produced minimal value after being dropped',
        'Minor Regret': 'Slight production after drop - not significant',
        'Some Regret': 'Player had some decent games post-drop',
        'Big Regret': 'Player produced meaningful fantasy value elsewhere',
        'Major Regret': 'Player became a quality starter after drop',
        'Disaster': 'Player became an elite performer after drop',
    }
    return descriptions.get(tier, 'Unknown tier')


@st.fragment
def display_drop_regret_analysis(transaction_df: pd.DataFrame) -> None:
    """Display drop regret analysis with themed visualizations."""

    st.markdown("### 😬 Drop Regret Analysis")
    st.markdown("*Players who produced after you dropped them*")

    df = transaction_df.copy()

    if df.empty:
        st.warning("No transaction data available.")
        return

    # Filter to drops only
    drops_df = df[df['transaction_type'] == 'drop'].copy()

    if drops_df.empty:
        st.info("No drop transactions found in the data.")
        return

    # Check for required columns
    has_regret_score = 'drop_regret_score' in drops_df.columns
    has_regret_tier = 'drop_regret_tier' in drops_df.columns

    if not has_regret_score and not has_regret_tier:
        st.warning("Drop regret metrics not available. Run the transaction pipeline with engagement metrics enabled.")
        return

    # Get filter options
    managers = sorted(drops_df['manager'].dropna().unique().tolist())
    years = sorted(drops_df['year'].dropna().unique().tolist(), reverse=True)

    # Filters row
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        manager_options = ["All Managers"] + managers
        selected_manager = st.selectbox("Manager", manager_options, key="regret_manager")
    with col2:
        year_options = ["All Years"] + [str(y) for y in years]
        selected_year = st.selectbox("Year", year_options, key="regret_year")

    # Apply filters
    filtered_df = drops_df.copy()
    if selected_manager != "All Managers":
        filtered_df = filtered_df[filtered_df['manager'] == selected_manager]
    if selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]

    if filtered_df.empty:
        st.warning("No drops match the selected filters.")
        return

    # Fill NA values
    if has_regret_score:
        filtered_df['drop_regret_score'] = filtered_df['drop_regret_score'].fillna(0)
    if has_regret_tier:
        filtered_df['drop_regret_tier'] = filtered_df['drop_regret_tier'].fillna('No Regret')

    # ========== SUMMARY METRICS ==========
    st.markdown("---")

    total_drops = len(filtered_df)

    # Calculate regret stats
    if has_regret_tier:
        tier_counts = filtered_df['drop_regret_tier'].value_counts().to_dict()
        disaster_count = tier_counts.get('Disaster', 0)
        major_count = tier_counts.get('Major Regret', 0)
        big_count = tier_counts.get('Big Regret', 0)
        no_regret_count = tier_counts.get('No Regret', 0)

        # Regrettable = Big Regret or worse
        regrettable_count = disaster_count + major_count + big_count
        regret_rate = (regrettable_count / total_drops * 100) if total_drops > 0 else 0
    else:
        tier_counts = {}
        regrettable_count = 0
        regret_rate = 0

    if has_regret_score:
        total_regret_spar = filtered_df['drop_regret_score'].sum()
        avg_regret_spar = filtered_df['drop_regret_score'].mean()
    else:
        total_regret_spar = 0
        avg_regret_spar = 0

    # Display metrics
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Drops", f"{total_drops:,}")
    with metric_cols[1]:
        st.metric("Regrettable Drops", f"{regrettable_count}", help="Big Regret or worse")
    with metric_cols[2]:
        st.metric("Regret Rate", f"{regret_rate:.1f}%")
    with metric_cols[3]:
        st.metric("SPAR Lost", f"{total_regret_spar:.1f}", help="Total SPAR produced by dropped players")

    # ========== VISUALIZATIONS ==========
    st.markdown("---")

    viz_col1, viz_col2 = st.columns(2)

    # Regret Tier Distribution Chart
    with viz_col1:
        st.markdown("#### Regret Distribution")

        if has_regret_tier and tier_counts:
            if HAS_CHART_THEMES:
                fig = create_regret_bar_chart(tier_counts, title="")
            else:
                # Manual chart creation
                tier_order = ['No Regret', 'Minor Regret', 'Some Regret', 'Big Regret', 'Major Regret', 'Disaster']
                tiers = [t for t in tier_order if t in tier_counts]
                counts = [tier_counts.get(t, 0) for t in tiers]
                colors = [REGRET_COLORS.get(t, '#808080') for t in tiers]

                fig = go.Figure(go.Bar(
                    x=tiers,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition='outside',
                ))
                fig.update_layout(
                    height=350,
                    xaxis_tickangle=-45,
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=80),
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regret tier data available.")

    # Manager Regret Leaderboard (if not filtered to single manager)
    with viz_col2:
        st.markdown("#### Manager Regret Leaderboard")

        if selected_manager == "All Managers" and has_regret_score:
            # Aggregate by manager
            manager_stats = drops_df.groupby('manager').agg({
                'drop_regret_score': ['sum', 'mean', 'count']
            }).round(1)
            manager_stats.columns = ['Total SPAR Lost', 'Avg SPAR Lost', 'Drop Count']
            manager_stats = manager_stats.reset_index()
            manager_stats = manager_stats.sort_values('Total SPAR Lost', ascending=True)

            # Horizontal bar chart
            if HAS_CHART_THEMES:
                fig = create_horizontal_bar_chart(
                    labels=manager_stats['manager'].tolist(),
                    values=manager_stats['Total SPAR Lost'].tolist(),
                    title="",
                    color_by_value=False
                )
                # Override to use red for regret
                fig.update_traces(marker_color='#d62728')
            else:
                fig = go.Figure(go.Bar(
                    y=manager_stats['manager'],
                    x=manager_stats['Total SPAR Lost'],
                    orientation='h',
                    marker_color='#d62728',
                    text=[f"{v:.1f}" for v in manager_stats['Total SPAR Lost']],
                    textposition='outside',
                ))
                fig.update_layout(
                    height=max(300, len(manager_stats) * 30),
                    margin=dict(l=100, r=40, t=40, b=40),
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show regret by year for single manager
            if has_regret_score:
                year_stats = filtered_df.groupby('year').agg({
                    'drop_regret_score': 'sum'
                }).round(1).reset_index()
                year_stats.columns = ['Year', 'SPAR Lost']
                year_stats = year_stats.sort_values('Year')

                fig = go.Figure(go.Bar(
                    x=year_stats['Year'].astype(str),
                    y=year_stats['SPAR Lost'],
                    marker_color='#d62728',
                    text=[f"{v:.1f}" for v in year_stats['SPAR Lost']],
                    textposition='outside',
                ))
                fig.update_layout(
                    height=300,
                    xaxis_title="Year",
                    yaxis_title="SPAR Lost",
                    margin=dict(l=40, r=40, t=40, b=40),
                )

                if HAS_CHART_THEMES:
                    fig = apply_chart_theme(fig)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No regret score data available.")

    # ========== WORST DROPS TABLE ==========
    st.markdown("---")
    st.markdown("#### 💀 Worst Drops")
    st.markdown("*Players who produced the most after being dropped*")

    # Sort by regret score descending
    if has_regret_score:
        worst_drops = filtered_df.nlargest(20, 'drop_regret_score')
    else:
        worst_drops = filtered_df.head(20)

    # Build display columns
    display_cols = ['manager', 'year', 'week', 'player_name']
    if 'position' in worst_drops.columns:
        display_cols.append('position')
    if has_regret_score:
        display_cols.append('drop_regret_score')
    if has_regret_tier:
        display_cols.append('drop_regret_tier')

    display_df = worst_drops[display_cols].copy()

    # Rename columns for display
    rename_map = {
        'manager': 'Manager',
        'year': 'Year',
        'week': 'Week',
        'player_name': 'Player',
        'position': 'Pos',
        'drop_regret_score': 'SPAR After Drop',
        'drop_regret_tier': 'Regret Level',
    }
    display_df = display_df.rename(columns=rename_map)

    # Add emoji column
    if 'Regret Level' in display_df.columns:
        display_df[''] = display_df['Regret Level'].apply(get_regret_emoji)
        # Reorder to put emoji first
        cols = display_df.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        display_df = display_df[cols]

    # Style the dataframe
    def style_regret_level(val):
        color = REGRET_COLORS.get(val, '#808080')
        return f'color: {color}; font-weight: bold'

    def style_spar(val):
        try:
            if float(val) >= 50:
                return 'color: #8b0000; font-weight: bold'
            elif float(val) >= 25:
                return 'color: #d62728; font-weight: bold'
            elif float(val) >= 10:
                return 'color: #ff7f0e'
            return ''
        except:
            return ''

    styled_df = display_df.style
    if 'Regret Level' in display_df.columns:
        styled_df = styled_df.map(style_regret_level, subset=['Regret Level'])
    if 'SPAR After Drop' in display_df.columns:
        styled_df = styled_df.map(style_spar, subset=['SPAR After Drop'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ========== REGRET TIER LEGEND ==========
    st.markdown("---")
    with st.expander("📖 Regret Tier Definitions"):
        for tier, color in REGRET_COLORS.items():
            emoji = get_regret_emoji(tier)
            desc = get_regret_description(tier)
            st.markdown(
                f"<span style='color: {color}; font-weight: bold;'>{emoji} {tier}</span>: {desc}",
                unsafe_allow_html=True
            )
