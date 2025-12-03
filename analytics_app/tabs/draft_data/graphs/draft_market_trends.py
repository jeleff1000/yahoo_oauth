#!/usr/bin/env python3
"""
draft_market_trends.py - Market inefficiencies and trends over time
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from md.core import T, run_query


@st.fragment
def display_draft_market_trends(prefix=""):
    """Analyze market inefficiencies and trends over time"""
    st.subheader("🔥 Market Trends & Inefficiencies")

    with st.spinner("Loading draft data..."):
        draft_data = run_query(
            f"""
            SELECT
                year, manager, player, yahoo_position,
                round, pick, cost, cost_bucket,
                COALESCE(total_fantasy_points, 0) as points,
                COALESCE(season_ppg, 0) as season_ppg,
                COALESCE(spar, 0) as spar,
                COALESCE(draft_roi, 0) as draft_roi
            FROM {T['draft']}
            WHERE cost > 0
            ORDER BY year DESC, pick
        """
        )

    if draft_data.empty:
        st.warning("No draft data available.")
        return

    # Year-over-year trends
    yearly_trends = (
        draft_data.groupby(["year", "yahoo_position"])
        .agg({"cost": "mean", "spar": "mean", "draft_roi": "mean"})
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Cost by Position Over Time**")
        fig = go.Figure()

        positions = sorted(yearly_trends["yahoo_position"].unique())
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, pos in enumerate(positions):
            pos_data = yearly_trends[yearly_trends["yahoo_position"] == pos]
            fig.add_trace(
                go.Scatter(
                    x=pos_data["year"],
                    y=pos_data["cost"],
                    mode="lines+markers",
                    name=pos,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{pos}</b><br>"
                    + "Year: %{x}<br>"
                    + "Avg Cost: $%{y:.1f}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Cost ($)",
            height=400,
            hovermode="x unified",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_cost_trends")

    with col2:
        st.markdown("**Average SPAR/$ by Position Over Time**")
        fig = go.Figure()

        for i, pos in enumerate(positions):
            pos_data = yearly_trends[yearly_trends["yahoo_position"] == pos]
            fig.add_trace(
                go.Scatter(
                    x=pos_data["year"],
                    y=pos_data["draft_roi"],
                    mode="lines+markers",
                    name=pos,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{pos}</b><br>"
                    + "Year: %{x}<br>"
                    + "Avg SPAR/$: %{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Average SPAR/$",
            height=400,
            hovermode="x unified",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_roi_trends")

    # Cost bucket analysis
    st.markdown("### 💰 Value by Cost Bucket")

    bucket_data = draft_data[draft_data["cost_bucket"].notna()].copy()
    bucket_stats = (
        bucket_data.groupby("cost_bucket")
        .agg({"cost": "mean", "spar": "mean", "draft_roi": "mean", "player": "count"})
        .reset_index()
    )
    bucket_stats.columns = [
        "Cost Bucket",
        "Avg Cost",
        "Avg SPAR",
        "Avg SPAR/$",
        "Count",
    ]
    bucket_stats = bucket_stats.sort_values("Cost Bucket")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=bucket_stats["Cost Bucket"],
            y=bucket_stats["Avg SPAR/$"],
            name="Avg SPAR/$",
            marker_color=[
                "darkgreen" if x > 1 else "lightgreen" if x > 0 else "red"
                for x in bucket_stats["Avg SPAR/$"]
            ],
            text=bucket_stats["Avg SPAR/$"].round(2),
            texttemplate="%{text}",
            textposition="outside",
            hovertemplate="Bucket %{x}<br>Avg SPAR/$: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=bucket_stats["Cost Bucket"],
            y=bucket_stats["Count"],
            name="# Picks",
            mode="lines+markers",
            line=dict(color="blue", width=3),
            marker=dict(size=10),
            hovertemplate="Bucket %{x}<br>Picks: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Cost Bucket")
    fig.update_yaxes(title_text="Average SPAR/$", secondary_y=False)
    fig.update_yaxes(title_text="Number of Picks", secondary_y=True)
    fig.update_layout(height=400, hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_bucket_value")

    # Market insights
    st.markdown("### 💡 Market Insights")

    if not bucket_stats.empty and len(bucket_stats) > 0:
        col1, col2, col3 = st.columns(3)

        best_bucket = bucket_stats.loc[bucket_stats["Avg SPAR/$"].idxmax()]
        worst_bucket = bucket_stats.loc[bucket_stats["Avg SPAR/$"].idxmin()]
        most_popular = bucket_stats.loc[bucket_stats["Count"].idxmax()]

        with col1:
            st.metric(
                "Best SPAR/$ Bucket",
                f"Bucket {int(best_bucket['Cost Bucket'])}",
                delta=f"{best_bucket['Avg SPAR/$']:.2f} SPAR/$",
            )
        with col2:
            st.metric(
                "Worst SPAR/$ Bucket",
                f"Bucket {int(worst_bucket['Cost Bucket'])}",
                delta=f"{worst_bucket['Avg SPAR/$']:.2f} SPAR/$",
            )
        with col3:
            st.metric(
                "Most Popular Bucket",
                f"Bucket {int(most_popular['Cost Bucket'])}",
                delta=f"{int(most_popular['Count'])} picks",
            )
    else:
        st.info("No cost bucket data available for insights.")
