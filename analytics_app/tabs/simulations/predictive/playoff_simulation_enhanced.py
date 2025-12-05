#!/usr/bin/env python3
"""
playoff_simulation_enhanced.py - Comprehensive playoff simulation dashboard

Optimized for mobile/desktop, light/dark mode with responsive charts.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from md.core import T, run_query
from .table_styles import render_modern_table
from ..shared.simulation_styles import (
    render_summary_tiles,
    render_section_header,
    render_group_card,
    close_card,
    render_summary_panel,
    render_manager_filter,
)
from ..shared.unified_header import render_delta_pill

# Import chart theming for light/dark mode support
import sys
from pathlib import Path

# Ensure analytics_app is in path for imports
_app_dir = Path(__file__).parent.parent.parent.parent.resolve()
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from shared.chart_themes import get_chart_theme, apply_chart_theme, get_chart_colors


def _select_week_for_playoff(base_df: pd.DataFrame, prefix: str):
    """Week selection with auto-load option matching other predictive tabs."""
    # Session state buttons instead of radio
    mode_key = f"{prefix}_playoff_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = 0

    modes = ["Today's Date", "Specific Week"]
    cols = st.columns(2)
    for idx, (col, name) in enumerate(zip(cols, modes)):
        with col:
            is_active = st.session_state[mode_key] == idx
            if st.button(
                name,
                key=f"{prefix}_mode_btn_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state[mode_key] = idx
                    st.rerun()

    mode = modes[st.session_state[mode_key]]

    if mode == "Today's Date":
        year = int(base_df["year"].max())
        week = int(base_df[base_df["year"] == year]["week"].max())
        st.caption(f"📅 Auto-selected Year {year}, Week {week}")
        return year, week, True
    else:
        years = sorted(base_df["year"].astype(int).unique(), reverse=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            year_choice = st.selectbox(
                "Year",
                ["Select Year"] + [str(y) for y in years],
                key=f"{prefix}_playoff_year",
            )

        if year_choice == "Select Year":
            return None, None, False

        year = int(year_choice)
        weeks = sorted(base_df[base_df["year"] == year]["week"].astype(int).unique())

        with col2:
            week_choice = st.selectbox(
                "Week",
                ["Select Week"] + [str(w) for w in weeks],
                key=f"{prefix}_playoff_week",
            )

        if week_choice == "Select Week":
            return None, None, False

        week = int(week_choice)
        return year, week, False


@st.fragment
def _render_playoff_dashboard(
    playoff_data: pd.DataFrame, year: int, week: int, prefix: str
):
    """Render the playoff dashboard for a specific year and week."""
    # Filter data up to the selected week
    week_data = playoff_data[
        (playoff_data["year"] == year) & (playoff_data["week"] <= week)
    ].copy()

    if week_data.empty:
        st.info("No data available for selected year/week.")
        return

    # Simulation Summary Panel (collapsible context)
    current_week_data = week_data[week_data["week"] == week]
    if not current_week_data.empty:
        num_managers = current_week_data["manager"].nunique()
        int(week_data["week"].max())
        avg_playoff_odds = current_week_data["p_playoffs"].mean()
        clinched = len(current_week_data[current_week_data["p_playoffs"] >= 99])
        eliminated = len(current_week_data[current_week_data["p_playoffs"] <= 1])

        render_summary_panel(
            "Simulation Context",
            [
                {"label": "Season Progress", "value": f"Week {week}"},
                {"label": "Teams", "value": str(num_managers)},
                {"label": "Avg Playoff Odds", "value": f"{avg_playoff_odds:.1f}%"},
                {"label": "Clinched", "value": str(clinched)},
                {"label": "Eliminated", "value": str(eliminated)},
            ],
            expanded=False,
        )

    # Create tabs - focused on unique analytics only
    tab1, tab2 = st.tabs(["Championship Path", "Critical Moments"])

    with tab1:
        _display_championship_path(week_data, year, week, prefix)

    with tab2:
        _display_critical_moments(week_data, year, week, prefix)


@st.fragment
def display_playoff_simulation_dashboard(
    prefix: str = "",
    year: Optional[int] = None,
    week: Optional[int] = None,
):
    """Enhanced playoff simulation analysis with modern visualizations.

    Args:
        prefix: Unique key prefix for session state
        year: Selected year (from unified header)
        week: Selected week (from unified header)
    """
    # Load data
    with st.spinner("Loading..."):
        playoff_data = run_query(
            f"""
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
        """
        )

    if playoff_data.empty:
        st.warning("No playoff simulation data available.")
        return

    # Type conversion
    playoff_data["year"] = playoff_data["year"].astype(int)
    playoff_data["week"] = playoff_data["week"].astype(int)

    # Use provided year/week or default to latest
    if year is None:
        year = int(playoff_data["year"].max())
    if week is None:
        week = int(playoff_data[playoff_data["year"] == year]["week"].max())

    # Render dashboard directly (year/week comes from unified header)
    _render_playoff_dashboard(playoff_data, year, week, prefix)


def _render_metric_cards(final_data: pd.DataFrame):
    """Render compact metric summary tiles at the top."""
    # Get key stats
    top_playoff = final_data.nlargest(1, "p_playoffs").iloc[0]
    top_champ = final_data.nlargest(1, "p_champ").iloc[0]
    top_power = final_data.nlargest(1, "exp_final_wins").iloc[0]
    top_seed = final_data.nsmallest(1, "avg_seed").iloc[0]

    # Use summary tiles for compact, visually consistent display
    render_summary_tiles(
        [
            {
                "icon": "",
                "label": "Highest Playoff Odds",
                "value": f"{top_playoff['p_playoffs']:.0f}%",
                "sublabel": top_playoff.name,
            },
            {
                "icon": "",
                "label": "Championship Favorite",
                "value": f"{top_champ['p_champ']:.1f}%",
                "sublabel": top_champ.name,
            },
            {
                "icon": "",
                "label": "Most Likely #1 Seed",
                "value": f"{top_seed['avg_seed']:.2f}",
                "sublabel": top_seed.name,
            },
            {
                "icon": "",
                "label": "Projected Most Wins",
                "value": f"{top_power['exp_final_wins']:.1f}",
                "sublabel": top_power.name,
            },
        ]
    )


@st.fragment
def _display_championship_path(data, year, week, prefix):
    """Show path to championship with responsive design."""
    # Use selected week data
    final_week = week
    week_data = data[data["week"] == final_week]

    # Check if we have data for this week
    if week_data.empty:
        st.info(f"No playoff simulation data available for Week {week}")
        return

    final_data = (
        week_data
        .groupby("manager")
        .agg(
            {
                "avg_seed": "mean",
                "p_playoffs": "mean",
                "p_bye": "mean",
                "exp_final_wins": "mean",
                "exp_final_pf": "mean",
                "p_semis": "mean",
                "p_final": "mean",
                "p_champ": "mean",
            }
        )
        .sort_values("avg_seed", ascending=True)
    )

    # Check if aggregation produced any data
    if final_data.empty:
        st.info(f"No playoff simulation data available for Week {week}")
        return

    # Get key stats for KPI hero
    top_playoff = final_data.nlargest(1, "p_playoffs").iloc[0]
    top_champ = final_data.nlargest(1, "p_champ").iloc[0]
    top_power = final_data.nlargest(1, "exp_final_wins").iloc[0]
    top_seed = final_data.nsmallest(1, "avg_seed").iloc[0]

    all_managers = list(final_data.index)

    # Title row with manager dropdown bound to it
    with st.container(border=True):
        title_cols = st.columns([3, 1])
        with title_cols[0]:
            st.markdown(f"### Championship Path — {year} Week {week}")
        with title_cols[1]:
            all_option = ["All Managers"] + list(all_managers)
            selected_manager = st.selectbox(
                "Manager",
                all_option,
                index=0,
                key=f"{prefix}_champ_manager_filter",
                label_visibility="collapsed",
            )
            if selected_manager == "All Managers":
                selected_manager = None

        # KPI Grid - 4 cards in a row
        kpi_html = (
            '<div class="sim-kpi-grid">'
            '<div class="sim-kpi-item">'
            f'<div class="sim-kpi-value">{top_playoff["p_playoffs"]:.0f}%</div>'
            '<div class="sim-kpi-label">HIGHEST PLAYOFF ODDS</div>'
            f'<div class="sim-kpi-owner">{top_playoff.name}</div>'
            '</div>'
            '<div class="sim-kpi-item">'
            f'<div class="sim-kpi-value">{top_champ["p_champ"]:.1f}%</div>'
            '<div class="sim-kpi-label">CHAMPIONSHIP FAVORITE</div>'
            f'<div class="sim-kpi-owner">{top_champ.name}</div>'
            '</div>'
            '<div class="sim-kpi-item">'
            f'<div class="sim-kpi-value">{top_seed["avg_seed"]:.2f}</div>'
            '<div class="sim-kpi-label">MOST LIKELY #1 SEED</div>'
            f'<div class="sim-kpi-owner">{top_seed.name}</div>'
            '</div>'
            '<div class="sim-kpi-item">'
            f'<div class="sim-kpi-value">{top_power["exp_final_wins"]:.1f}</div>'
            '<div class="sim-kpi-label">PROJECTED MOST WINS</div>'
            f'<div class="sim-kpi-owner">{top_power.name}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(kpi_html, unsafe_allow_html=True)

    # Reset index for table display
    final_data = final_data.reset_index()

    # Championship odds table in card
    with st.container(border=True):
        st.markdown("**Championship Odds & Projections**")

        # Prepare data for render_modern_table
        table_df = final_data.copy()
        table_df = table_df.set_index("manager")

        # Define which columns get gradient coloring (probability columns)
        color_columns = ["p_playoffs", "p_bye", "p_semis", "p_final", "p_champ"]

        # Define column display names
        column_names = {
            "avg_seed": "Avg Seed",
            "p_playoffs": "Playoff %",
            "p_bye": "Bye %",
            "exp_final_wins": "Exp Wins",
            "exp_final_pf": "Exp PF",
            "p_semis": "Semis %",
            "p_final": "Final %",
            "p_champ": "Champ %",
        }

        # Define format specs
        format_specs = {
            "avg_seed": "{:.2f}",
            "p_playoffs": "{:.1f}",
            "p_bye": "{:.1f}",
            "exp_final_wins": "{:.2f}",
            "exp_final_pf": "{:.1f}",
            "p_semis": "{:.1f}",
            "p_final": "{:.1f}",
            "p_champ": "{:.1f}",
        }

        # Display the modern table with column-based gradients
        render_modern_table(
            table_df,
            title="",
            color_columns=color_columns,
            reverse_columns=[],
            format_specs=format_specs,
            column_names=column_names,
            gradient_mode="column",
        )

    # Visualization section in bordered container
    with st.container(border=True):
        st.markdown("### Visual Analysis")

        # Get theme-aware colors
        colors = get_chart_colors()
        get_chart_theme()

        # Show top 6 managers for cleaner mobile display
        managers = final_data.head(6)["manager"].tolist()

        # Create a simpler, more mobile-friendly chart
        fig = go.Figure()

        # Championship funnel - horizontal bar chart (better for mobile)
        for i, manager in enumerate(managers):
            mgr_row = final_data[final_data["manager"] == manager].iloc[0]
            stages = ["Win Title", "Reach Final", "Reach Semis", "Make Playoffs"]
            values = [
                mgr_row["p_champ"],
                mgr_row["p_final"],
                mgr_row["p_semis"],
                mgr_row["p_playoffs"],
            ]

            fig.add_trace(
                go.Bar(
                    name=manager,
                    y=stages,
                    x=values,
                    orientation="h",
                    text=[f"{v:.0f}%" for v in values],
                    textposition="outside",
                    hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
                    marker_color=colors["categorical"][i % len(colors["categorical"])],
                )
            )

        # Apply theme and update layout
        fig.update_layout(
            height=350,  # Compact height for mobile
            barmode="group",
            hovermode="closest",
            margin=dict(l=10, r=60, t=30, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
            xaxis=dict(title="Probability (%)", range=[0, 105]),
        )
        apply_chart_theme(fig)

        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_champ_funnel_{year}")


@st.fragment
def _display_critical_moments(data, year, week, prefix):
    """Critical turning points with theme-aware charts."""
    # Subtitle caption above cards
    st.caption("Biggest weekly swings in playoff odds — who rode the roller coaster?")

    # Get theme colors
    colors = get_chart_colors()

    # Calculate changes
    managers = data["manager"].unique()
    changes_list = []

    for manager in managers:
        mgr_data = data[data["manager"] == manager].sort_values("week")
        mgr_data["p_change"] = mgr_data["p_playoffs"].diff()
        mgr_data["champ_change"] = mgr_data["p_champ"].diff()
        changes_list.append(mgr_data)

    all_changes = pd.concat(changes_list)

    # Biggest swings
    biggest_gains = all_changes.nlargest(5, "p_change")[
        ["week", "manager", "p_change", "p_playoffs"]
    ].copy()
    biggest_losses = all_changes.nsmallest(5, "p_change")[
        ["week", "manager", "p_change", "p_playoffs"]
    ].copy()

    # Calculate volatility for third column
    volatility = (
        data.groupby("manager")["p_playoffs"].std().sort_values(ascending=False)
    )

    # 3-column layout with equal heights via CSS class
    st.markdown('<div class="sim-equal-height-row">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        with st.container(border=True):
            st.markdown("**Biggest Gains**")
            for _, row in biggest_gains.iterrows():
                delta_html = render_delta_pill(f"+{row['p_change']:.1f}%", is_positive=True)
                st.markdown(
                    f"Wk {int(row['week'])} - **{row['manager']}** {delta_html}",
                    unsafe_allow_html=True,
                )

            # Show all as right-aligned text link
            show_all_key = f"{prefix}_show_all_gains"
            if show_all_key not in st.session_state:
                st.session_state[show_all_key] = False

            link_cols = st.columns([3, 1])
            with link_cols[1]:
                if st.button("Show all ›", key=f"{show_all_key}_btn", type="tertiary"):
                    st.session_state[show_all_key] = not st.session_state[show_all_key]

            if st.session_state[show_all_key]:
                all_gains = all_changes.nlargest(10, "p_change")[
                    ["week", "manager", "p_change"]
                ]
                for _, row in all_gains.iterrows():
                    st.caption(f"Wk {int(row['week'])}: {row['manager']} +{row['p_change']:.1f}%")

    with col2:
        with st.container(border=True):
            st.markdown("**Biggest Drops**")
            for _, row in biggest_losses.iterrows():
                delta_html = render_delta_pill(f"{row['p_change']:.1f}%", is_positive=False)
                st.markdown(
                    f"Wk {int(row['week'])} - **{row['manager']}** {delta_html}",
                    unsafe_allow_html=True,
                )

            # Show all as right-aligned text link
            show_all_key = f"{prefix}_show_all_drops"
            if show_all_key not in st.session_state:
                st.session_state[show_all_key] = False

            link_cols = st.columns([3, 1])
            with link_cols[1]:
                if st.button("Show all ›", key=f"{show_all_key}_btn", type="tertiary"):
                    st.session_state[show_all_key] = not st.session_state[show_all_key]

            if st.session_state[show_all_key]:
                all_losses = all_changes.nsmallest(10, "p_change")[
                    ["week", "manager", "p_change"]
                ]
                for _, row in all_losses.iterrows():
                    st.caption(f"Wk {int(row['week'])}: {row['manager']} {row['p_change']:.1f}%")

    with col3:
        with st.container(border=True):
            st.markdown("**Season Volatility**")
            st.caption("Higher = more roller coaster season")

            # Use theme-aware colors for volatility levels
            bar_colors = []
            for x in volatility.values:
                if x > 15:
                    bar_colors.append(colors["negative"])
                elif x > 10:
                    bar_colors.append("#ff9800")  # Orange
                else:
                    bar_colors.append(colors["positive"])

            fig = go.Figure(
                go.Bar(
                    x=volatility.index,
                    y=volatility.values,
                    marker_color=bar_colors,
                    text=[f"{v:.0f}" for v in volatility.values],
                    textposition="outside",
                    hovertemplate="%{x}<br>Volatility: %{y:.1f}%<extra></extra>",
                )
            )

            fig.update_layout(
                height=250,  # Compact for side panel
                xaxis=dict(tickangle=-45, title=None),
                yaxis=dict(title="Std Dev (%)", title_font=dict(size=10)),
                margin=dict(l=10, r=10, t=10, b=60),
                showlegend=False,
            )
            apply_chart_theme(fig)

            st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_volatility_{year}")

    st.markdown("</div>", unsafe_allow_html=True)
