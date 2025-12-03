#!/usr/bin/env python3
"""
Unified scoring trends visualization - replaces all_time_scoring_graphs and weekly_scoring_graphs.
Provides weekly, cumulative, and season average views in one component.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import T, list_seasons, run_query


@st.fragment
def display_scoring_trends(df_dict=None, prefix=""):
    """
    Unified scoring trends with multiple view modes.
    """
    st.header("📈 Scoring Trends")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Track scoring patterns:</strong> View weekly performance, cumulative career totals, or season averages.
    Identify hot streaks, slumps, and long-term trends.
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # View mode selection
    col1, col2 = st.columns([2, 3])

    with col1:
        view_mode = st.radio(
            "View Mode",
            [
                "📊 Weekly (Single Season)",
                "📈 Cumulative (All-Time)",
                "📅 Season Averages",
            ],
            key=f"{prefix}_view_mode",
            horizontal=False,
        )

    with col2:
        # Year selection (only for weekly mode)
        if view_mode == "📊 Weekly (Single Season)":
            available_years = list_seasons()
            if not available_years:
                st.error("No data available.")
                return

            selected_year = st.selectbox(
                "Select Season",
                options=available_years,
                index=0,  # Most recent
                key=f"{prefix}_year",
            )
        else:
            selected_year = None

    # Load data based on view mode
    with st.spinner("Loading scoring data..."):
        if view_mode == "📊 Weekly (Single Season)":
            query = f"""
                SELECT
                    week,
                    manager,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY week, manager
            """
        elif view_mode == "📈 Cumulative (All-Time)":
            query = f"""
                SELECT
                    year,
                    week,
                    manager,
                    team_points,
                    ROW_NUMBER() OVER (PARTITION BY manager ORDER BY year, week) as game_number
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:  # Season Averages
            query = f"""
                SELECT
                    year,
                    manager,
                    AVG(team_points) as avg_points,
                    MIN(team_points) as min_points,
                    MAX(team_points) as max_points,
                    STDDEV(team_points) as std_dev,
                    COUNT(*) as games
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                GROUP BY year, manager
                ORDER BY year, manager
            """

        data = run_query(query)

        if data.empty:
            st.warning("No data found.")
            return

    # Manager selection
    managers = sorted(data["manager"].unique())
    selected_managers = st.multiselect(
        "Select Managers to Display",
        options=managers,
        default=managers[:5] if len(managers) >= 5 else managers,
        key=f"{prefix}_managers",
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Render based on view mode
    if view_mode == "📊 Weekly (Single Season)":
        _render_weekly_view(filtered_data, selected_year, prefix)
    elif view_mode == "📈 Cumulative (All-Time)":
        _render_cumulative_view(filtered_data, prefix)
    else:
        _render_season_averages_view(filtered_data, prefix)


def _render_weekly_view(data: pd.DataFrame, year: str, prefix: str):
    """Render weekly scoring for a single season."""
    st.subheader(f"📊 Week-by-Week Performance - {year}")

    # Create line chart
    fig = go.Figure()

    for manager in data["manager"].unique():
        manager_data = data[data["manager"] == manager].sort_values("week")

        fig.add_trace(
            go.Scatter(
                x=manager_data["week"],
                y=manager_data["team_points"],
                mode="lines+markers",
                name=manager,
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate=f"<b>{manager}</b><br>Week: %{{x}}<br>Points: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Team Points",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_weekly_chart")

    # Statistics table
    with st.expander("📊 Week-by-Week Data", expanded=False):
        try:
            pivot = data.pivot(index="week", columns="manager", values="team_points")
            pivot = pivot.round(2)

            # Add summary rows
            summary_df = pd.DataFrame(
                {
                    col: [
                        pivot[col].mean(),
                        pivot[col].max(),
                        pivot[col].min(),
                        pivot[col].std(),
                    ]
                    for col in pivot.columns
                },
                index=["Average", "Max", "Min", "Std Dev"],
            )

            pivot = pd.concat([pivot, summary_df])
            st.dataframe(pivot, use_container_width=True)
        except Exception:
            st.dataframe(
                data[["week", "manager", "team_points"]],
                hide_index=True,
                use_container_width=True,
            )

    # Insights
    with st.expander("💡 Weekly Insights", expanded=False):
        stats = (
            data.groupby("manager")
            .agg({"team_points": ["mean", "max", "min", "std"], "win": "sum"})
            .round(2)
        )
        stats.columns = ["Avg", "Best Week", "Worst Week", "Consistency", "Wins"]
        stats = stats.sort_values("Avg", ascending=False)

        col1, col2, col3 = st.columns(3)

        with col1:
            best_avg = stats.nlargest(1, "Avg")
            st.metric(
                "Highest Average",
                best_avg.index[0],
                f"{best_avg.iloc[0]['Avg']:.2f} PPG",
            )

        with col2:
            best_week = stats.nlargest(1, "Best Week")
            st.metric(
                "Best Single Week",
                best_week.index[0],
                f"{best_week.iloc[0]['Best Week']:.2f} pts",
            )

        with col3:
            most_consistent = stats.nsmallest(1, "Consistency")
            st.metric(
                "Most Consistent",
                most_consistent.index[0],
                f"{most_consistent.iloc[0]['Consistency']:.2f} std dev",
            )


def _render_cumulative_view(data: pd.DataFrame, prefix: str):
    """Render cumulative all-time scoring progression."""
    st.subheader("📈 Career Scoring Progression")

    # Calculate cumulative points
    data = data.copy()
    data["cumulative_points"] = data.groupby("manager")["team_points"].cumsum()

    # Create cumulative chart
    fig = go.Figure()

    for manager in data["manager"].unique():
        manager_data = data[data["manager"] == manager]

        fig.add_trace(
            go.Scatter(
                x=manager_data["game_number"],
                y=manager_data["cumulative_points"],
                mode="lines",
                name=manager,
                line=dict(width=3),
                hovertemplate=f"<b>{manager}</b><br>Game: %{{x}}<br>Total: %{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Career Games Played",
        yaxis_title="Cumulative Points",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_cumulative_chart")

    # Career totals table
    st.subheader("📊 Career Totals")

    career_stats = (
        data.groupby("manager").agg({"team_points": ["sum", "mean", "count"]}).round(2)
    )

    career_stats.columns = ["Total Points", "Avg PPG", "Games"]
    career_stats = career_stats.sort_values("Total Points", ascending=False)
    career_stats["Total Points"] = career_stats["Total Points"].astype(int)

    st.dataframe(career_stats, use_container_width=True)

    # Career milestones
    with st.expander("💡 Career Milestones", expanded=False):
        top_scorer = career_stats.index[0]
        top_total = career_stats.iloc[0]["Total Points"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("All-Time Leader", top_scorer, f"{top_total:,} points")

        with col2:
            highest_avg = career_stats.nlargest(1, "Avg PPG")
            st.metric(
                "Highest PPG",
                highest_avg.index[0],
                f"{highest_avg.iloc[0]['Avg PPG']:.2f}",
            )

        with col3:
            most_games = career_stats.nlargest(1, "Games")
            st.metric(
                "Most Games", most_games.index[0], f"{int(most_games.iloc[0]['Games'])}"
            )


def _render_season_averages_view(data: pd.DataFrame, prefix: str):
    """Render season-by-season average scoring trends."""
    st.subheader("📅 Season Average Trends")

    # Create line chart with error bars
    fig = go.Figure()

    for manager in data["manager"].unique():
        manager_data = data[data["manager"] == manager].sort_values("year")

        fig.add_trace(
            go.Scatter(
                x=manager_data["year"],
                y=manager_data["avg_points"],
                mode="lines+markers",
                name=manager,
                line=dict(width=2),
                marker=dict(size=10),
                error_y=dict(
                    type="data",
                    array=manager_data["std_dev"],
                    visible=True,
                    thickness=1.5,
                    width=4,
                ),
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Year: %{x}<br>"
                    "Avg: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Points Per Game",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        showlegend=True,
    )
    fig.update_xaxes(tickmode="linear", dtick=1)

    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_season_avg_chart")

    # Season-by-season table
    with st.expander("📊 Season Statistics", expanded=False):
        pivot = data.pivot(index="year", columns="manager", values="avg_points")
        pivot = pivot.round(2)
        st.dataframe(pivot, use_container_width=True)

    # Trend analysis
    with st.expander("📈 Trend Analysis", expanded=False):
        trends = []
        for manager in data["manager"].unique():
            manager_data = data[data["manager"] == manager].sort_values("year")
            if len(manager_data) >= 2:
                latest = manager_data.iloc[-1]["avg_points"]
                previous = manager_data.iloc[-2]["avg_points"]
                change = latest - previous

                # Calculate overall trend (slope)
                years = manager_data["year"].values
                avg_pts = manager_data["avg_points"].values
                if len(years) > 1:
                    slope = (avg_pts[-1] - avg_pts[0]) / (years[-1] - years[0])
                    trend = (
                        "📈 Improving"
                        if slope > 1
                        else "📉 Declining" if slope < -1 else "➡️ Stable"
                    )
                else:
                    trend = "➡️ Stable"

                trends.append(
                    {
                        "Manager": manager,
                        "Latest PPG": f"{latest:.2f}",
                        "YoY Change": f"{change:+.2f}",
                        "Trend": trend,
                    }
                )

        if trends:
            trends_df = pd.DataFrame(trends)
            st.dataframe(trends_df, hide_index=True, use_container_width=True)
