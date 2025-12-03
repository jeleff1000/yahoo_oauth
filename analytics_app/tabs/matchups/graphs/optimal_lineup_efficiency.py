#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from md.core import T, list_seasons, run_query


@st.fragment
def display_optimal_lineup_efficiency_graph(df_dict=None, prefix=""):
    """
    Analyze lineup efficiency - how well managers set their optimal lineups.
    Shows points left on bench and coaching decision quality.
    """
    st.header("🎯 Optimal Lineup Efficiency")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Maximize your roster:</strong> See how efficiently you set lineups. Are you leaving points on the bench?
    Track lineup decisions and their impact on wins/losses.
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
    selected_year = st.selectbox(
        "Select Season", options=year_options, key=f"{prefix}_year"
    )

    # Load data
    with st.spinner("Loading lineup efficiency data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT
                    year,
                    week,
                    manager,
                    team_points,
                    optimal_points,
                    bench_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    CASE
                        WHEN optimal_points > 0 THEN (team_points / optimal_points * 100)
                        ELSE 100
                    END as lineup_efficiency
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                  AND optimal_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT
                    week,
                    manager,
                    team_points,
                    optimal_points,
                    bench_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    CASE
                        WHEN optimal_points > 0 THEN (team_points / optimal_points * 100)
                        ELSE 100
                    END as lineup_efficiency
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND team_points IS NOT NULL
                  AND optimal_points IS NOT NULL
                ORDER BY manager, week
            """
        data = run_query(query)

        if data.empty:
            st.warning("No optimal lineup data found.")
            return

    # Calculate additional metrics
    data["points_left_on_bench"] = data["optimal_points"] - data["team_points"]
    data["could_have_won"] = (
        (data["win"] == 0) & (data["optimal_points"] > data["opponent_points"])
    ).astype(int)

    # Manager selection
    managers = sorted(data["manager"].unique())
    selected_managers = st.multiselect(
        "Select Managers to Display",
        options=managers,
        default=managers[:3] if len(managers) >= 3 else managers,
        key=f"{prefix}_managers",
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["📈 Efficiency Trends", "😢 Missed Opportunities", "📊 Statistics"]
    )

    with tab1:
        st.subheader("Lineup Efficiency Over Time")
        st.caption("100% = Perfect lineup every week")

        # Determine x-axis
        if selected_year == "All Seasons":
            # Use game number for all-time view
            filtered_data = filtered_data.sort_values(["manager", "year", "week"])
            filtered_data["game_number"] = (
                filtered_data.groupby("manager").cumcount() + 1
            )
            x_col = "game_number"
            x_title = "Game Number"
        else:
            x_col = "week"
            x_title = "Week"

        # Line chart of efficiency
        fig_eff = go.Figure()

        for manager in selected_managers:
            manager_data = filtered_data[
                filtered_data["manager"] == manager
            ].sort_values(x_col)

            fig_eff.add_trace(
                go.Scatter(
                    x=manager_data[x_col],
                    y=manager_data["lineup_efficiency"],
                    mode="lines+markers",
                    name=manager,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{manager}</b><br>"
                        f"{x_title}: %{{x}}<br>"
                        "Efficiency: %{y:.1f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Add reference lines
        fig_eff.add_hline(
            y=100,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text="Perfect (100%)",
            annotation_position="right",
        )
        fig_eff.add_hline(
            y=90,
            line_dash="dot",
            line_color="orange",
            opacity=0.3,
            annotation_text="Good (90%)",
            annotation_position="right",
        )

        fig_eff.update_layout(
            xaxis_title=x_title,
            yaxis_title="Lineup Efficiency (%)",
            hovermode="x unified",
            height=500,
            template="plotly_white",
            showlegend=True,
            yaxis=dict(range=[0, 105]),
        )

        st.plotly_chart(
            fig_eff, use_container_width=True, key=f"{prefix}_efficiency_trend"
        )

        # Rolling average efficiency
        with st.expander("📊 Rolling Average (Last 4 Weeks)", expanded=False):
            fig_rolling = go.Figure()

            for manager in selected_managers:
                manager_data = filtered_data[
                    filtered_data["manager"] == manager
                ].sort_values(x_col)
                manager_data["rolling_eff"] = (
                    manager_data["lineup_efficiency"]
                    .rolling(window=4, min_periods=1)
                    .mean()
                )

                fig_rolling.add_trace(
                    go.Scatter(
                        x=manager_data[x_col],
                        y=manager_data["rolling_eff"],
                        mode="lines",
                        name=manager,
                        line=dict(width=3),
                        hovertemplate=f"<b>{manager}</b><br>{x_title}: %{{x}}<br>4-Week Avg: %{{y:.1f}}%<extra></extra>",
                    )
                )

            fig_rolling.add_hline(
                y=90, line_dash="dash", line_color="gray", opacity=0.5
            )

            fig_rolling.update_layout(
                xaxis_title=x_title,
                yaxis_title="Rolling 4-Week Efficiency (%)",
                hovermode="x unified",
                height=400,
                template="plotly_white",
                showlegend=True,
            )

            st.plotly_chart(
                fig_rolling, use_container_width=True, key=f"{prefix}_rolling"
            )

    with tab2:
        st.subheader("😢 Games You Could Have Won")
        st.caption("Losses where optimal lineup would have secured victory")

        # Calculate missed wins by manager
        missed_wins = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            total_losses = len(manager_data[manager_data["win"] == 0])
            games_could_win = manager_data["could_have_won"].sum()
            total_bench_pts = manager_data["points_left_on_bench"].sum()
            avg_bench_pts = manager_data["points_left_on_bench"].mean()

            # Close losses where bench would have helped
            close_losses = manager_data[
                (manager_data["win"] == 0)
                & (
                    manager_data["points_left_on_bench"]
                    > abs(manager_data["team_points"] - manager_data["opponent_points"])
                )
            ]

            missed_wins.append(
                {
                    "Manager": manager,
                    "Total Losses": total_losses,
                    "Could Have Won": int(games_could_win),
                    "Close Losses": len(close_losses),
                    "Avg Bench Points": avg_bench_pts,
                    "Total Bench Points": total_bench_pts,
                    "Missed Win %": (
                        (games_could_win / total_losses * 100)
                        if total_losses > 0
                        else 0
                    ),
                }
            )

        missed_df = pd.DataFrame(missed_wins)

        # Bar chart of missed opportunities
        fig_missed = go.Figure()

        fig_missed.add_trace(
            go.Bar(
                name="Losses (Good Lineup)",
                x=missed_df["Manager"],
                y=missed_df["Total Losses"] - missed_df["Could Have Won"],
                marker_color="lightcoral",
                hovertemplate="<b>%{x}</b><br>Unavoidable Losses: %{y}<extra></extra>",
            )
        )

        fig_missed.add_trace(
            go.Bar(
                name="Could Have Won",
                x=missed_df["Manager"],
                y=missed_df["Could Have Won"],
                marker_color="darkred",
                text=missed_df["Could Have Won"],
                textposition="inside",
                hovertemplate="<b>%{x}</b><br>Could Have Won: %{y}<extra></extra>",
            )
        )

        fig_missed.update_layout(
            barmode="stack",
            xaxis_title="",
            yaxis_title="Number of Losses",
            height=400,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig_missed, use_container_width=True, key=f"{prefix}_missed")

        # Points left on bench
        st.subheader("📉 Points Left on Bench")

        fig_bench = go.Figure()

        bench_sorted = missed_df.sort_values("Total Bench Points", ascending=True)

        fig_bench.add_trace(
            go.Bar(
                y=bench_sorted["Manager"],
                x=bench_sorted["Total Bench Points"],
                orientation="h",
                marker=dict(
                    color=bench_sorted["Total Bench Points"],
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="Points"),
                ),
                text=bench_sorted["Total Bench Points"].apply(lambda x: f"{x:.0f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Total Bench Points: %{x:.0f}<extra></extra>",
            )
        )

        fig_bench.update_layout(
            xaxis_title="Total Points Left on Bench",
            yaxis_title="",
            height=max(300, len(missed_df) * 40),
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(fig_bench, use_container_width=True, key=f"{prefix}_bench")

        # Detailed table
        with st.expander("📊 Detailed Missed Opportunities", expanded=False):
            display_missed = missed_df.copy()
            display_missed["Avg Bench Points"] = display_missed[
                "Avg Bench Points"
            ].round(2)
            display_missed["Total Bench Points"] = display_missed[
                "Total Bench Points"
            ].round(2)
            display_missed["Missed Win %"] = display_missed["Missed Win %"].round(1)

            st.dataframe(display_missed, hide_index=True, use_container_width=True)

    with tab3:
        st.subheader("📊 Efficiency Statistics")

        # Calculate comprehensive stats
        efficiency_stats = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            wins = manager_data[manager_data["win"] == 1]
            losses = manager_data[manager_data["win"] == 0]

            efficiency_stats.append(
                {
                    "Manager": manager,
                    "Avg Efficiency": manager_data["lineup_efficiency"].mean(),
                    "Best Week": manager_data["lineup_efficiency"].max(),
                    "Worst Week": manager_data["lineup_efficiency"].min(),
                    "Std Dev": manager_data["lineup_efficiency"].std(),
                    "Win Efficiency": (
                        wins["lineup_efficiency"].mean() if len(wins) > 0 else 0
                    ),
                    "Loss Efficiency": (
                        losses["lineup_efficiency"].mean() if len(losses) > 0 else 0
                    ),
                    "Weeks at 100%": len(
                        manager_data[manager_data["lineup_efficiency"] >= 99.5]
                    ),
                    "Weeks Below 90%": len(
                        manager_data[manager_data["lineup_efficiency"] < 90]
                    ),
                }
            )

        stats_df = pd.DataFrame(efficiency_stats).round(2)
        stats_df = stats_df.sort_values("Avg Efficiency", ascending=False)

        # Display stats table
        st.dataframe(
            stats_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Avg Efficiency": st.column_config.NumberColumn(format="%.2f%%"),
                "Best Week": st.column_config.NumberColumn(format="%.2f%%"),
                "Worst Week": st.column_config.NumberColumn(format="%.2f%%"),
                "Std Dev": st.column_config.NumberColumn(format="%.2f"),
                "Win Efficiency": st.column_config.NumberColumn(format="%.2f%%"),
                "Loss Efficiency": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        # Efficiency vs Wins scatter
        st.subheader("🔍 Efficiency vs Win Rate")

        win_rate_data = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            win_rate_data.append(
                {
                    "manager": manager,
                    "avg_efficiency": manager_data["lineup_efficiency"].mean(),
                    "win_rate": manager_data["win"].mean() * 100,
                    "games": len(manager_data),
                }
            )

        win_rate_df = pd.DataFrame(win_rate_data)

        fig_scatter = go.Figure()

        fig_scatter.add_trace(
            go.Scatter(
                x=win_rate_df["avg_efficiency"],
                y=win_rate_df["win_rate"],
                mode="markers+text",
                text=win_rate_df["manager"],
                textposition="top center",
                marker=dict(
                    size=win_rate_df["games"] / 2,
                    color=win_rate_df["win_rate"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Win %"),
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Avg Efficiency: %{x:.1f}%<br>"
                    "Win Rate: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig_scatter.update_layout(
            xaxis_title="Average Lineup Efficiency (%)",
            yaxis_title="Win Rate (%)",
            height=450,
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(fig_scatter, use_container_width=True, key=f"{prefix}_scatter")

    # Key insights
    with st.expander("💡 Key Insights", expanded=False):
        best_coach = stats_df.iloc[0]
        most_missed = missed_df.nlargest(1, "Could Have Won").iloc[0]
        most_bench_pts = missed_df.nlargest(1, "Avg Bench Points").iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🎯 Best Coach",
                best_coach["Manager"],
                f"{best_coach['Avg Efficiency']:.1f}% efficiency",
            )

        with col2:
            st.metric(
                "😢 Most Missed Wins",
                most_missed["Manager"],
                f"{int(most_missed['Could Have Won'])} games",
            )

        with col3:
            st.metric(
                "📉 Most Bench Points",
                most_bench_pts["Manager"],
                f"{most_bench_pts['Avg Bench Points']:.1f} PPG",
            )
