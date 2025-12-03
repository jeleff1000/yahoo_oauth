#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.data_access import run_query, T, list_seasons


@st.fragment
def display_scoring_distribution_graph(prefix=""):
    """
    Analyze scoring distribution - who's consistent vs volatile.
    """
    st.header("📊 Scoring Distribution Analysis")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Analyze scoring patterns:</strong> See the distribution of team scores.
    Consistent managers have tight distributions; volatile managers have wide spreads.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection with "All Seasons" option
    available_years = list_seasons()
    if not available_years:
        st.error("No data available.")
        return

    year_options = ["All Seasons"] + available_years
    selected_year = st.selectbox(
        "Select Season",
        options=year_options,
        key=f"{prefix}_year"
    )

    # Load matchup data
    with st.spinner("Loading scoring data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT
                    manager,
                    year,
                    week,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT
                    manager,
                    week,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY manager, week
            """
        data = run_query(query)

        if data.empty:
            st.warning("No data found for selected year.")
            return

    # Calculate statistics by manager
    stats = data.groupby("manager").agg({
        "team_points": ["mean", "std", "min", "max", "count"],
        "win": "sum"
    }).round(2)

    stats.columns = ["avg_points", "std_dev", "min_points", "max_points", "games", "wins"]
    stats = stats.reset_index()
    stats["win_pct"] = (stats["wins"] / stats["games"] * 100).round(1)
    stats["consistency_score"] = (stats["avg_points"] / (stats["std_dev"] + 1)).round(2)
    stats["coefficient_of_variation"] = ((stats["std_dev"] / stats["avg_points"]) * 100).round(1)  # Lower = more consistent
    stats = stats.sort_values("avg_points", ascending=False)

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Box Plots", "📈 Scatter Analysis", "📋 Stats Table"])

    with tab1:
        st.subheader("Score Distribution by Manager")
        st.caption("Visualize scoring range and consistency for each manager")

        # Create box plot
        fig = go.Figure()

        # Sort managers by median for better visual
        manager_medians = data.groupby("manager")["team_points"].median().sort_values(ascending=False)

        for manager in manager_medians.index:
            manager_scores = data[data["manager"] == manager]["team_points"]

            fig.add_trace(go.Box(
                y=manager_scores,
                name=manager,
                boxmean='sd',
                marker_color='lightseagreen',
                hovertemplate="<b>%{fullData.name}</b><br>Score: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            yaxis_title="Team Points",
            showlegend=False,
            height=500,
            template="plotly_white",
            title=dict(
                text=f"Distribution - {selected_year}",
                x=0.5,
                xanchor='center'
            )
        )
        fig.update_xaxes(tickangle=-45)

        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_boxplot")

        st.caption("💡 Box: 25th-75th percentile | Line in box: median | Diamond: mean | Whiskers: range (excl. outliers)")

    with tab2:
        st.subheader("Consistency vs Performance Analysis")
        st.caption("Find the sweet spot: high scoring AND consistent")

        # Scatter plot: avg points vs coefficient of variation
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=stats["coefficient_of_variation"],
            y=stats["avg_points"],
            mode="markers+text",
            text=stats["manager"],
            textposition="top center",
            marker=dict(
                size=stats["games"] / 2,  # Size by number of games
                color=stats["win_pct"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Win %"),
                line=dict(width=1, color='white')
            ),
            hovertemplate="<b>%{text}</b><br>Avg: %{y:.2f}<br>CV: %{x:.1f}%<br>Win%: " +
                         stats["win_pct"].astype(str) + "%<br>Games: " +
                         stats["games"].astype(str) + "<extra></extra>"
        ))

        fig2.update_layout(
            xaxis_title="Coefficient of Variation (lower = more consistent)",
            yaxis_title="Average Points (higher = better)",
            height=550,
            template="plotly_white"
        )

        # Add quadrant lines
        avg_cv = stats["coefficient_of_variation"].median()
        avg_pts = stats["avg_points"].median()

        fig2.add_hline(y=avg_pts, line_dash="dash", line_color="gray", opacity=0.5)
        fig2.add_vline(x=avg_cv, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig2.add_annotation(text="🎯 Elite<br>(High + Consistent)", xref="paper", yref="paper",
                           x=0.15, y=0.95, showarrow=False, bgcolor="rgba(144,238,144,0.3)", borderpad=4)
        fig2.add_annotation(text="💪 High Variance<br>Winners", xref="paper", yref="paper",
                           x=0.85, y=0.95, showarrow=False, bgcolor="rgba(255,255,0,0.2)", borderpad=4)
        fig2.add_annotation(text="😐 Consistent<br>Mediocrity", xref="paper", yref="paper",
                           x=0.15, y=0.05, showarrow=False, bgcolor="rgba(211,211,211,0.3)", borderpad=4)
        fig2.add_annotation(text="🎲 Boom or Bust", xref="paper", yref="paper",
                           x=0.85, y=0.05, showarrow=False, bgcolor="rgba(255,0,0,0.2)", borderpad=4)

        st.plotly_chart(fig2, use_container_width=True, key=f"{prefix}_scatter")

    with tab3:
        st.subheader("Manager Statistics")

        display_df = stats[[
            "manager", "avg_points", "std_dev", "coefficient_of_variation",
            "min_points", "max_points", "wins", "games", "win_pct"
        ]].copy()

        display_df.columns = [
            "Manager", "Avg Points", "Std Dev", "CV %",
            "Min", "Max", "Wins", "Games", "Win %"
        ]

        # Color code by performance
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                "CV %": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Lower = more consistent"
                ),
                "Avg Points": st.column_config.NumberColumn(format="%.2f")
            }
        )

    # Insights
    with st.expander("💡 Key Insights", expanded=False):
        most_consistent = stats.nsmallest(1, "coefficient_of_variation").iloc[0]
        most_volatile = stats.nlargest(1, "coefficient_of_variation").iloc[0]
        highest_avg = stats.nlargest(1, "avg_points").iloc[0]
        highest_ceiling = stats.nlargest(1, "max_points").iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Most Consistent",
                most_consistent["manager"],
                f"{most_consistent['coefficient_of_variation']:.1f}% CV"
            )

        with col2:
            st.metric(
                "Highest Average",
                highest_avg["manager"],
                f"{highest_avg['avg_points']:.1f} PPG"
            )

        with col3:
            st.metric(
                "Highest Ceiling",
                highest_ceiling["manager"],
                f"{highest_ceiling['max_points']:.1f} pts"
            )

        with col4:
            st.metric(
                "Most Volatile",
                most_volatile["manager"],
                f"{most_volatile['coefficient_of_variation']:.1f}% CV"
            )
