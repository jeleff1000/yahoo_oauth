#!/usr/bin/env python3
"""
playoff_simulation_enhanced.py - Comprehensive playoff simulation dashboard

Optimized for mobile/desktop, light/dark mode with responsive charts.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from md.data_access import run_query, T
from .table_styles import render_modern_table

# Import chart theming for light/dark mode support
import sys
from pathlib import Path

# Ensure streamlit_ui is in path for imports
_streamlit_ui_dir = Path(__file__).parent.parent.parent.parent.resolve()
if str(_streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(_streamlit_ui_dir))

from shared.chart_themes import get_chart_theme, apply_chart_theme, get_chart_colors


def _select_week_for_playoff(base_df: pd.DataFrame, prefix: str):
    """Week selection with auto-load option matching other predictive tabs."""
    mode = st.radio(
        "Selection Mode",
        ["Today's Date", "Specific Week"],
        horizontal=True,
        key=f"{prefix}_playoff_mode",
        index=0
    )

    if mode == "Today's Date":
        year = int(base_df['year'].max())
        week = int(base_df[base_df['year'] == year]['week'].max())
        st.caption(f"📅 Auto-selected Year {year}, Week {week}")
        return year, week, True
    else:
        years = sorted(base_df['year'].astype(int).unique(), reverse=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            year_choice = st.selectbox(
                "Year",
                ["Select Year"] + [str(y) for y in years],
                key=f"{prefix}_playoff_year"
            )

        if year_choice == "Select Year":
            return None, None, False

        year = int(year_choice)
        weeks = sorted(base_df[base_df['year'] == year]['week'].astype(int).unique())

        with col2:
            week_choice = st.selectbox(
                "Week",
                ["Select Week"] + [str(w) for w in weeks],
                key=f"{prefix}_playoff_week"
            )

        if week_choice == "Select Week":
            return None, None, False

        week = int(week_choice)
        return year, week, False


@st.fragment
def _render_playoff_dashboard(playoff_data: pd.DataFrame, year: int, week: int, prefix: str):
    """Render the playoff dashboard for a specific year and week."""
    # Filter data up to the selected week
    week_data = playoff_data[
        (playoff_data['year'] == year) &
        (playoff_data['week'] <= week)
    ].copy()

    if week_data.empty:
        st.info("No data available for selected year/week.")
        return

    # Create tabs - focused on unique analytics only
    tab1, tab2 = st.tabs([
        "🏆 Championship Path",
        "🔥 Critical Moments"
    ])

    with tab1:
        _display_championship_path(week_data, year, week, prefix)

    with tab2:
        _display_critical_moments(week_data, year, week, prefix)


@st.fragment
def display_playoff_simulation_dashboard(prefix=""):
    """Enhanced playoff simulation analysis with modern visualizations."""
    st.header("🏈 Playoff Simulation Dashboard")

    st.info("Deep-dive analytics: championship funnels, critical turning points, and comprehensive season trajectories.")

    # Load data
    with st.spinner("Loading playoff simulation data..."):
        playoff_data = run_query(f"""
            SELECT
                year, week, manager,
                avg_seed,
                p_playoffs, p_bye, p_semis, p_final, p_champ,
                exp_final_wins, exp_final_pf,
                is_playoffs, is_consolation,
                champion
            FROM {T['matchup']}
            WHERE is_playoffs = 0
              AND is_consolation = 0
              AND p_playoffs IS NOT NULL
            ORDER BY year DESC, week DESC, avg_seed
        """)

    if playoff_data.empty:
        st.warning("No playoff simulation data available.")
        return

    # Type conversion
    playoff_data['year'] = playoff_data['year'].astype(int)
    playoff_data['week'] = playoff_data['week'].astype(int)

    # Week selection
    year, week, auto_display = _select_week_for_playoff(playoff_data, prefix)

    if year is None or week is None:
        return

    # Auto-display or show button
    if auto_display:
        _render_playoff_dashboard(playoff_data, year, week, prefix)
    else:
        if st.button("Go", key=f"{prefix}_playoff_go"):
            _render_playoff_dashboard(playoff_data, year, week, prefix)


def _render_metric_cards(final_data: pd.DataFrame):
    """Render compact metric summary cards at the top."""
    # Get key stats
    top_playoff = final_data.nlargest(1, 'p_playoffs').iloc[0]
    top_champ = final_data.nlargest(1, 'p_champ').iloc[0]
    top_power = final_data.nlargest(1, 'exp_final_wins').iloc[0]

    # Get theme colors
    colors = get_chart_colors()

    # Create compact metric row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Top Playoff Odds",
            value=f"{top_playoff['p_playoffs']:.0f}%",
            delta=top_playoff.name
        )

    with col2:
        st.metric(
            label="Championship Favorite",
            value=f"{top_champ['p_champ']:.1f}%",
            delta=top_champ.name
        )

    with col3:
        st.metric(
            label="Projected Most Wins",
            value=f"{top_power['exp_final_wins']:.1f}",
            delta=top_power.name
        )


@st.fragment
def _display_championship_path(data, year, week, prefix):
    """Show path to championship with responsive design."""
    st.subheader(f"🏆 Championship Path - {year} Week {week}")

    # Use selected week data
    final_week = week
    final_data = data[data['week'] == final_week].groupby('manager').agg({
        'avg_seed': 'mean',
        'p_playoffs': 'mean',
        'p_bye': 'mean',
        'exp_final_wins': 'mean',
        'exp_final_pf': 'mean',
        'p_semis': 'mean',
        'p_final': 'mean',
        'p_champ': 'mean'
    }).sort_values('avg_seed', ascending=True)

    # Render metric cards first
    _render_metric_cards(final_data)

    st.markdown("---")

    # Reset index for table display
    final_data = final_data.reset_index()

    # Championship odds table - DISPLAYED FIRST
    st.markdown("### 🏆 Championship Odds & Projections")

    # Prepare data for render_modern_table
    table_df = final_data.copy()
    table_df = table_df.set_index('manager')

    # Define which columns get gradient coloring (probability columns)
    color_columns = ['p_playoffs', 'p_bye', 'p_semis', 'p_final', 'p_champ']

    # Define column display names
    column_names = {
        'avg_seed': 'Avg Seed',
        'p_playoffs': 'Playoff %',
        'p_bye': 'Bye %',
        'exp_final_wins': 'Exp Wins',
        'exp_final_pf': 'Exp PF',
        'p_semis': 'Semis %',
        'p_final': 'Final %',
        'p_champ': 'Champ %'
    }

    # Define format specs
    format_specs = {
        'avg_seed': '{:.2f}',
        'p_playoffs': '{:.1f}',
        'p_bye': '{:.1f}',
        'exp_final_wins': '{:.2f}',
        'exp_final_pf': '{:.1f}',
        'p_semis': '{:.1f}',
        'p_final': '{:.1f}',
        'p_champ': '{:.1f}'
    }

    # Display the modern table with column-based gradients
    render_modern_table(
        table_df,
        title="",
        color_columns=color_columns,
        reverse_columns=[],
        format_specs=format_specs,
        column_names=column_names,
        gradient_mode="column"
    )

    st.markdown("---")

    # Visualization section
    st.markdown("### 📊 Visual Analysis")

    # Get theme-aware colors
    colors = get_chart_colors()
    theme_config = get_chart_theme()

    # Show top 6 managers for cleaner mobile display
    managers = final_data.head(6)['manager'].tolist()

    # Create a simpler, more mobile-friendly chart
    fig = go.Figure()

    # Championship funnel - horizontal bar chart (better for mobile)
    for i, manager in enumerate(managers):
        mgr_row = final_data[final_data['manager'] == manager].iloc[0]
        stages = ['Win Title', 'Reach Final', 'Reach Semis', 'Make Playoffs']
        values = [
            mgr_row['p_champ'],
            mgr_row['p_final'],
            mgr_row['p_semis'],
            mgr_row['p_playoffs']
        ]

        fig.add_trace(go.Bar(
            name=manager,
            y=stages,
            x=values,
            orientation='h',
            text=[f"{v:.0f}%" for v in values],
            textposition='outside',
            hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>',
            marker_color=colors['categorical'][i % len(colors['categorical'])]
        ))

    # Apply theme and update layout
    fig.update_layout(
        height=350,  # Compact height for mobile
        barmode='group',
        hovermode='closest',
        margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        xaxis=dict(title="Probability (%)", range=[0, 105]),
    )
    apply_chart_theme(fig)

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_champ_funnel_{year}")


@st.fragment
def _display_critical_moments(data, year, week, prefix):
    """Critical turning points with theme-aware charts."""
    st.subheader(f"🔥 Critical Moments - {year} Week {week}")

    st.caption("Identify the biggest swings in playoff odds. Which weeks had the most drama?")

    # Get theme colors
    colors = get_chart_colors()

    # Calculate changes
    managers = data['manager'].unique()
    changes_list = []

    for manager in managers:
        mgr_data = data[data['manager'] == manager].sort_values('week')
        mgr_data['p_change'] = mgr_data['p_playoffs'].diff()
        mgr_data['champ_change'] = mgr_data['p_champ'].diff()
        changes_list.append(mgr_data)

    all_changes = pd.concat(changes_list)

    # Biggest swings
    biggest_gains = all_changes.nlargest(10, 'p_change')[
        ['week', 'manager', 'p_change', 'p_playoffs']
    ].copy()
    biggest_losses = all_changes.nsmallest(10, 'p_change')[
        ['week', 'manager', 'p_change', 'p_playoffs']
    ].copy()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**📈 Biggest Weekly Gains**")
        biggest_gains_display = biggest_gains.copy()
        biggest_gains_display.columns = ['Week', 'Manager', 'Change', 'New Odds']

        # Format the data
        biggest_gains_display['Change'] = biggest_gains_display['Change'].apply(lambda x: f"+{x:.1f}%")
        biggest_gains_display['New Odds'] = biggest_gains_display['New Odds'].apply(lambda x: f"{x:.1f}%")

        # Style the dataframe - theme adaptive
        def highlight_gains(s):
            if s.name == 'Change':
                return ['color: #28a745; font-weight: bold' for _ in s]
            elif s.name == 'Manager':
                return ['font-weight: bold' for _ in s]
            return ['' for _ in s]

        styled_gains = biggest_gains_display.style.apply(highlight_gains)
        st.dataframe(styled_gains, hide_index=True, use_container_width=True, height=400)

    with col2:
        st.markdown("**📉 Biggest Weekly Drops**")
        biggest_losses_display = biggest_losses.copy()
        biggest_losses_display.columns = ['Week', 'Manager', 'Change', 'New Odds']

        # Format the data
        biggest_losses_display['Change'] = biggest_losses_display['Change'].apply(lambda x: f"{x:.1f}%")
        biggest_losses_display['New Odds'] = biggest_losses_display['New Odds'].apply(lambda x: f"{x:.1f}%")

        # Style the dataframe - theme adaptive
        def highlight_losses(s):
            if s.name == 'Change':
                return ['color: #dc3545; font-weight: bold' for _ in s]
            elif s.name == 'Manager':
                return ['font-weight: bold' for _ in s]
            return ['' for _ in s]

        styled_losses = biggest_losses_display.style.apply(highlight_losses)
        st.dataframe(styled_losses, hide_index=True, use_container_width=True, height=400)

    # Volatility chart
    st.markdown("### 🎢 Season Volatility")
    st.caption("Higher = more roller coaster season")

    volatility = data.groupby('manager')['p_playoffs'].std().sort_values(ascending=False)

    # Use theme-aware colors for volatility levels
    bar_colors = []
    for x in volatility.values:
        if x > 15:
            bar_colors.append(colors['negative'])
        elif x > 10:
            bar_colors.append('#ff9800')  # Orange
        else:
            bar_colors.append(colors['positive'])

    fig = go.Figure(go.Bar(
        x=volatility.index,
        y=volatility.values,
        marker_color=bar_colors,
        text=[f"{v:.0f}" for v in volatility.values],
        textposition='outside',
        hovertemplate='%{x}<br>Volatility: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        height=300,  # Compact for mobile
        xaxis=dict(tickangle=-45, title=None),
        yaxis=dict(title="Std Dev (%)"),
        margin=dict(l=10, r=10, t=10, b=60),
        showlegend=False
    )
    apply_chart_theme(fig)

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_volatility_{year}")