#!/usr/bin/env python3
"""
draft_round_efficiency.py - Analyze efficiency by draft round
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from md.core import T, run_query


@st.fragment
def display_draft_round_efficiency(prefix=""):
    """Analyze draft efficiency by round"""
    st.subheader("📈 Round-by-Round Efficiency")

    with st.spinner("Loading draft data..."):
        draft_data = run_query(
            f"""
            SELECT
                year, manager, player, yahoo_position,
                round, pick, cost,
                COALESCE(TRY_CAST(is_keeper_status AS INTEGER), 0) as is_keeper,
                COALESCE(total_fantasy_points, 0) as points,
                COALESCE(season_ppg, 0) as season_ppg,
                COALESCE(spar, 0) as spar,
                COALESCE(draft_roi, 0) as draft_roi
            FROM {T['draft']}
            WHERE cost > 0 AND round IS NOT NULL
            ORDER BY year DESC, pick
        """
        )

    if draft_data.empty:
        st.warning("No draft data available.")
        return

    # Year selection
    years = sorted(draft_data["year"].unique(), reverse=True)
    selected_year = st.selectbox(
        "Select Year",
        options=["All Years"] + list(years),
        index=0,
        key=f"{prefix}_round_year",
    )

    # Filter data
    if selected_year != "All Years":
        data = draft_data[draft_data["year"] == selected_year].copy()
    else:
        data = draft_data.copy()

    # Calculate round statistics
    round_stats = (
        data.groupby("round")
        .agg(
            {
                "cost": ["mean", "sum"],
                "spar": ["mean", "sum"],
                "draft_roi": "mean",
                "player": "count",
            }
        )
        .round(2)
    )
    round_stats.columns = [
        "Avg Cost",
        "Total Spent",
        "Avg SPAR",
        "Total SPAR",
        "Avg SPAR/$",
        "Picks",
    ]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Round Statistics**")
        st.dataframe(round_stats, use_container_width=True, height=400)

    with col2:
        # Multi-metric round chart
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Average SPAR vs Cost by Round", "Average SPAR/$ by Round"),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        # Top chart: SPAR and Cost
        fig.add_trace(
            go.Scatter(
                x=round_stats.index,
                y=round_stats["Avg SPAR"],
                mode="lines+markers",
                name="Avg SPAR",
                line=dict(color="green", width=3),
                marker=dict(size=10),
                hovertemplate="Round %{x}<br>Avg SPAR: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=round_stats.index,
                y=round_stats["Avg Cost"],
                mode="lines+markers",
                name="Avg Cost",
                line=dict(color="blue", width=3, dash="dash"),
                marker=dict(size=10),
                hovertemplate="Round %{x}<br>Avg Cost: $%{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Bottom chart: SPAR/$
        fig.add_trace(
            go.Bar(
                x=round_stats.index,
                y=round_stats["Avg SPAR/$"],
                marker_color=[
                    "darkgreen" if x > 1 else "lightgreen" if x > 0 else "red"
                    for x in round_stats["Avg SPAR/$"]
                ],
                name="Avg SPAR/$",
                showlegend=False,
                text=round_stats["Avg SPAR/$"].round(2),
                texttemplate="%{text}",
                textposition="outside",
                hovertemplate="Round %{x}<br>Avg SPAR/$: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Round", row=2, col=1)
        fig.update_yaxes(title_text="SPAR", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="SPAR/$", row=2, col=1)

        fig.update_layout(height=600, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_round_eff_chart")

    # Round winners/losers
    st.markdown("### 🎯 Best and Worst Rounds")

    if not round_stats.empty and len(round_stats) > 0:
        col1, col2, col3 = st.columns(3)

        best_roi_round = round_stats["Avg SPAR/$"].idxmax()
        worst_roi_round = round_stats["Avg SPAR/$"].idxmin()
        best_spar_round = round_stats["Avg SPAR"].idxmax()

        with col1:
            st.metric(
                "Best SPAR/$ Round",
                f"Round {best_roi_round}",
                delta=f"{round_stats.loc[best_roi_round, 'Avg SPAR/$']:.2f} SPAR/$",
            )
        with col2:
            st.metric(
                "Worst SPAR/$ Round",
                f"Round {worst_roi_round}",
                delta=f"{round_stats.loc[worst_roi_round, 'Avg SPAR/$']:.2f} SPAR/$",
            )
        with col3:
            st.metric(
                "Best SPAR Round",
                f"Round {best_spar_round}",
                delta=f"{round_stats.loc[best_spar_round, 'Avg SPAR']:.1f} SPAR",
            )
    else:
        st.info("No round data available for comparison.")
