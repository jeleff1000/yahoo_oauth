#!/usr/bin/env python3
"""
Scoring Distribution Analysis with premium UI.

Improvements:
- Consistency rank badges
- Better box plot spacing with distinct mean/median colors
- Quadrant shading on scatter plot
- Better label contrast
- Trendline on scatter
- Percentile-based rankings for scalability
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from md.core import T, list_seasons, run_query

# Consistent color palette
CHART_COLORS = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6",
    "#8B5CF6", "#EF4444", "#14B8A6", "#F97316", "#84CC16",
    "#06B6D4", "#A855F7",
]


def _get_chart_color(index: int) -> str:
    return CHART_COLORS[index % len(CHART_COLORS)]


def _render_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 0.875rem;">
            <h3 style="margin: 0 0 0.375rem 0; font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">
                {title}
            </h3>
            <p style="margin: 0; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_consistency_rank_badge(rank: int, total: int) -> str:
    """Generate consistency rank badge based on percentile."""
    percentile = (total - rank + 1) / total * 100

    if percentile >= 90:
        return f"#1 Most Consistent"
    elif percentile >= 75:
        return f"#{rank} Very Consistent"
    elif percentile >= 50:
        return f"#{rank}"
    elif percentile >= 25:
        return f"#{rank} Variable"
    else:
        return f"#{rank} Volatile"


@st.fragment
def display_scoring_distribution_graph(prefix=""):
    """Analyze scoring distribution with premium UI."""

    # Header
    st.markdown(
        """
        <div style="
            background: rgba(0,0,0,0.05);
            border: 1px solid var(--border, #374151);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h2 style="margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 600;">
                Scoring Distribution Analysis
            </h2>
            <p style="margin: 0; color: var(--text-secondary, #9CA3AF); font-size: 0.85rem; line-height: 1.5;">
                Analyze scoring patterns and consistency.
                <strong>Tight distributions = consistent</strong>; wide spreads = volatile.
                The best teams are both high-scoring AND consistent.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Year selection
    available_years = list_seasons()
    if not available_years:
        st.error("No data available.")
        return

    year_options = ["All Seasons"] + available_years
    selected_year = st.selectbox("Select Season", options=year_options, key=f"{prefix}_year")

    # Load data
    with st.spinner("Loading scoring data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT manager, year, week, team_points, opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE manager IS NOT NULL AND team_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT manager, week, team_points, opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE year = {int(selected_year)} AND manager IS NOT NULL AND team_points IS NOT NULL
                ORDER BY manager, week
            """
        data = run_query(query)
        if data.empty:
            st.warning("No data found.")
            return

    # Calculate statistics
    stats = data.groupby("manager").agg({
        "team_points": ["mean", "std", "min", "max", "median", "count"],
        "win": "sum"
    }).round(2)
    stats.columns = ["avg_points", "std_dev", "min_points", "max_points", "median_points", "games", "wins"]
    stats = stats.reset_index()
    stats["win_pct"] = (stats["wins"] / stats["games"] * 100).round(1)
    stats["coefficient_of_variation"] = ((stats["std_dev"] / stats["avg_points"]) * 100).round(1)

    # Rank by consistency (lower CV = more consistent)
    stats["consistency_rank"] = stats["coefficient_of_variation"].rank(method="min").astype(int)
    stats = stats.sort_values("avg_points", ascending=False)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Box Plots", "Scatter Analysis", "Stats Table"])

    # ==================== BOX PLOTS ====================
    with tab1:
        _render_section_header(
            "Score Distribution by Manager",
            "Box shows 25th-75th percentile. Line = median, diamond = mean. Hover for details.",
        )

        fig = go.Figure()

        # Sort by median for visual clarity
        manager_medians = data.groupby("manager")["team_points"].median().sort_values(ascending=False)

        for i, manager in enumerate(manager_medians.index):
            manager_scores = data[data["manager"] == manager]["team_points"]
            color = _get_chart_color(i)

            fig.add_trace(
                go.Box(
                    y=manager_scores,
                    name=manager,
                    boxmean="sd",
                    marker=dict(color=color, outliercolor=color),
                    line=dict(color=color),
                    fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}",
                    hovertemplate=(
                        f"<b>{manager}</b><br>"
                        "Score: %{y:.1f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Add league average line
        league_avg = data["team_points"].mean()
        fig.add_hline(
            y=league_avg,
            line_dash="dash",
            line_color="rgba(255,255,255,0.5)",
            line_width=2,
            annotation_text=f"League Avg: {league_avg:.1f}",
            annotation_position="right",
            annotation_font_size=10,
        )

        fig.update_layout(
            yaxis_title="Team Points",
            showlegend=False,
            height=550,
            template="plotly_white",
            margin=dict(l=50, r=20, t=30, b=100),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
        )

        st.markdown('<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_boxplot")
        st.markdown("</div>", unsafe_allow_html=True)

        # Consistency rank badges
        st.markdown(
            "<p style='font-size: 0.85rem; font-weight: 600; margin: 1rem 0 0.5rem 0;'>Consistency Rankings</p>",
            unsafe_allow_html=True,
        )

        # Sort by consistency for badges
        consistency_sorted = stats.sort_values("coefficient_of_variation")
        total_managers = len(consistency_sorted)

        badge_cols = st.columns(min(6, total_managers))
        for i, (_, row) in enumerate(consistency_sorted.head(6).iterrows()):
            with badge_cols[i]:
                rank = row["consistency_rank"]
                cv = row["coefficient_of_variation"]

                # Color based on rank
                if rank <= total_managers * 0.25:
                    badge_color = "#10B981"
                    badge_bg = "rgba(16,185,129,0.15)"
                elif rank <= total_managers * 0.5:
                    badge_color = "#3B82F6"
                    badge_bg = "rgba(59,130,246,0.15)"
                else:
                    badge_color = "#6B7280"
                    badge_bg = "rgba(107,114,128,0.15)"

                st.markdown(
                    f"""
                    <div style="
                        background: {badge_bg};
                        border: 1px solid {badge_color};
                        border-radius: 8px;
                        padding: 0.5rem;
                        text-align: center;
                    ">
                        <div style="font-weight: 600; font-size: 0.8rem; color: {badge_color};">
                            #{rank}
                        </div>
                        <div style="font-size: 0.75rem; font-weight: 500; margin-top: 0.25rem;">
                            {row['manager']}
                        </div>
                        <div style="font-size: 0.65rem; color: var(--text-muted);">
                            {cv:.1f}% CV
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ==================== SCATTER ANALYSIS ====================
    with tab2:
        _render_section_header(
            "Consistency vs Performance Analysis",
            "Find the sweet spot: high scoring AND consistent. Lower CV% = more consistent.",
        )

        fig2 = go.Figure()

        # Calculate quadrant boundaries
        avg_cv = stats["coefficient_of_variation"].median()
        avg_pts = stats["avg_points"].median()

        # Add quadrant shading
        cv_min = stats["coefficient_of_variation"].min() - 2
        cv_max = stats["coefficient_of_variation"].max() + 2
        pts_min = stats["avg_points"].min() - 5
        pts_max = stats["avg_points"].max() + 5

        # Elite quadrant (top-left: high points, low CV)
        fig2.add_shape(type="rect", x0=cv_min, x1=avg_cv, y0=avg_pts, y1=pts_max,
                      fillcolor="rgba(16,185,129,0.08)", line_width=0)
        # High variance winners (top-right)
        fig2.add_shape(type="rect", x0=avg_cv, x1=cv_max, y0=avg_pts, y1=pts_max,
                      fillcolor="rgba(245,158,11,0.08)", line_width=0)
        # Consistent mediocrity (bottom-left)
        fig2.add_shape(type="rect", x0=cv_min, x1=avg_cv, y0=pts_min, y1=avg_pts,
                      fillcolor="rgba(156,163,175,0.08)", line_width=0)
        # Boom or bust (bottom-right)
        fig2.add_shape(type="rect", x0=avg_cv, x1=cv_max, y0=pts_min, y1=avg_pts,
                      fillcolor="rgba(239,68,68,0.08)", line_width=0)

        # Add scatter points with better contrast
        for i, (_, row) in enumerate(stats.iterrows()):
            color = _get_chart_color(i)

            fig2.add_trace(
                go.Scatter(
                    x=[row["coefficient_of_variation"]],
                    y=[row["avg_points"]],
                    mode="markers+text",
                    name=row["manager"],
                    text=[row["manager"]],
                    textposition="top center",
                    textfont=dict(size=11, color="white"),
                    marker=dict(
                        size=max(12, row["games"] / 3),
                        color=color,
                        line=dict(width=2, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>{row['manager']}</b><br>"
                        f"Avg: {row['avg_points']:.1f} PPG<br>"
                        f"CV: {row['coefficient_of_variation']:.1f}%<br>"
                        f"Win%: {row['win_pct']:.1f}%<br>"
                        f"Games: {row['games']}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        # Add trendline
        if len(stats) >= 2:
            z = np.polyfit(stats["coefficient_of_variation"], stats["avg_points"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(cv_min, cv_max, 50)
            fig2.add_trace(
                go.Scatter(
                    x=x_trend, y=p(x_trend),
                    mode="lines",
                    line=dict(dash="dot", color="rgba(255,255,255,0.4)", width=2),
                    name="Trend",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add quadrant dividers
        fig2.add_hline(y=avg_pts, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
        fig2.add_vline(x=avg_cv, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

        # Add quadrant labels (softened for less visual noise)
        fig2.add_annotation(text="Elite<br>(High + Consistent)", x=0.12, y=0.95,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=9, color="rgba(16,185,129,0.6)"),
                           bgcolor="rgba(16,185,129,0.08)", borderpad=5)
        fig2.add_annotation(text="High Variance<br>Winners", x=0.88, y=0.95,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=9, color="rgba(245,158,11,0.6)"),
                           bgcolor="rgba(245,158,11,0.08)", borderpad=5)
        fig2.add_annotation(text="Consistent<br>Mediocrity", x=0.12, y=0.05,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=9, color="rgba(156,163,175,0.6)"),
                           bgcolor="rgba(156,163,175,0.08)", borderpad=5)
        fig2.add_annotation(text="Boom or<br>Bust", x=0.88, y=0.05,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=9, color="rgba(239,68,68,0.6)"),
                           bgcolor="rgba(239,68,68,0.08)", borderpad=5)

        fig2.update_layout(
            xaxis_title="Coefficient of Variation % (lower = more consistent)",
            yaxis_title="Average Points (higher = better)",
            height=550,
            template="plotly_white",
            xaxis=dict(gridcolor="rgba(128,128,128,0.1)", range=[cv_min, cv_max]),
            yaxis=dict(gridcolor="rgba(128,128,128,0.1)", range=[pts_min, pts_max]),
            margin=dict(l=50, r=20, t=30, b=50),
        )

        st.markdown('<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_scatter")
        st.markdown("</div>", unsafe_allow_html=True)

        # Interpretation helper
        elite_managers = stats[
            (stats["coefficient_of_variation"] < avg_cv) & (stats["avg_points"] > avg_pts)
        ]["manager"].tolist()

        if elite_managers:
            st.markdown(
                f"""
                <div style="
                    background: rgba(16,185,129,0.1);
                    border-left: 3px solid #10B981;
                    padding: 0.5rem 0.75rem;
                    margin: 0.75rem 0;
                    font-size: 0.8rem;
                ">
                    <strong>Elite performers:</strong> {', '.join(elite_managers[:3])}
                    — High scoring AND consistent.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==================== STATS TABLE ====================
    with tab3:
        _render_section_header("Manager Statistics", "Complete breakdown of scoring patterns and consistency")

        display_df = stats[[
            "manager", "avg_points", "median_points", "std_dev", "coefficient_of_variation",
            "min_points", "max_points", "wins", "games", "win_pct", "consistency_rank"
        ]].copy()
        display_df.columns = [
            "Manager", "Avg", "Median", "Std Dev", "CV %",
            "Min", "Max", "Wins", "Games", "Win %", "Consistency Rank"
        ]

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                "CV %": st.column_config.NumberColumn(format="%.1f%%", help="Coefficient of Variation - lower = more consistent"),
                "Avg": st.column_config.NumberColumn(format="%.2f"),
                "Median": st.column_config.NumberColumn(format="%.2f"),
                "Consistency Rank": st.column_config.NumberColumn(help="Ranked by CV% - #1 is most consistent"),
            },
        )

    # Key Insights
    with st.expander("Key Insights", expanded=False):
        most_consistent = stats.nsmallest(1, "coefficient_of_variation").iloc[0]
        most_volatile = stats.nlargest(1, "coefficient_of_variation").iloc[0]
        highest_avg = stats.nlargest(1, "avg_points").iloc[0]
        highest_ceiling = stats.nlargest(1, "max_points").iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Most Consistent", most_consistent["manager"], f"{most_consistent['coefficient_of_variation']:.1f}% CV")
        with col2:
            st.metric("Highest Average", highest_avg["manager"], f"{highest_avg['avg_points']:.1f} PPG")
        with col3:
            st.metric("Highest Ceiling", highest_ceiling["manager"], f"{highest_ceiling['max_points']:.1f} pts")
        with col4:
            st.metric("Most Volatile", most_volatile["manager"], f"{most_volatile['coefficient_of_variation']:.1f}% CV")
