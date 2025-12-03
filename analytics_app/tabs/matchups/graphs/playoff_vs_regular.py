#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from md.core import T, list_seasons, run_query


@st.fragment
def display_playoff_vs_regular_graph(df_dict=None, prefix=""):
    """
    Compare playoff performance vs regular season - who steps up when it matters?
    """
    st.header("🏆 Playoff vs Regular Season Performance")

    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>When it matters most:</strong> See who rises to the occasion in playoffs vs regular season.
    Are you a playoff performer or a regular season hero?
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection
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

    # Load data
    with st.spinner("Loading playoff data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT
                    year,
                    week,
                    manager,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    CASE
                        WHEN is_playoffs = 1 THEN 'Playoff'
                        ELSE 'Regular Season'
                    END as game_type
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT
                    week,
                    manager,
                    team_points,
                    opponent_points,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    CASE
                        WHEN is_playoffs = 1 THEN 'Playoff'
                        ELSE 'Regular Season'
                    END as game_type
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND team_points IS NOT NULL
                ORDER BY manager, week
            """
        data = run_query(query)

        if data.empty:
            st.warning("No data found.")
            return

    # Check if we have playoff data
    playoff_data = data[data["game_type"] == "Playoff"]
    if playoff_data.empty:
        st.warning("No playoff data found for selected period.")
        return

    # Manager selection
    managers = sorted(data["manager"].unique())
    selected_managers = st.multiselect(
        "Select Managers to Display",
        options=managers,
        default=managers[:5] if len(managers) >= 5 else managers,
        key=f"{prefix}_managers"
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Comparison", "🏆 Playoff Performance", "📈 Trends"])

    with tab1:
        st.subheader("Regular Season vs Playoff Stats")

        # Calculate stats by game type
        comparison_stats = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            for game_type in ["Regular Season", "Playoff"]:
                type_data = manager_data[manager_data["game_type"] == game_type]

                if len(type_data) > 0:
                    comparison_stats.append({
                        "Manager": manager,
                        "Game Type": game_type,
                        "Games": len(type_data),
                        "Wins": type_data["win"].sum(),
                        "Losses": len(type_data) - type_data["win"].sum(),
                        "Win %": type_data["win"].mean() * 100,
                        "Avg Points": type_data["team_points"].mean(),
                        "Avg Opp Points": type_data["opponent_points"].mean(),
                        "Consistency (Std Dev)": type_data["team_points"].std()
                    })

        comp_df = pd.DataFrame(comparison_stats)

        if comp_df.empty:
            st.warning("Not enough data for comparison.")
            return

        # Side-by-side comparison bar chart
        st.subheader("Average Points: Regular Season vs Playoffs")

        fig_comp = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Average Points Per Game", "Win Percentage"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Pivot data for easier plotting
        pivot_points = comp_df.pivot(index="Manager", columns="Game Type", values="Avg Points")
        pivot_winpct = comp_df.pivot(index="Manager", columns="Game Type", values="Win %")

        # Add traces for points
        if "Regular Season" in pivot_points.columns:
            fig_comp.add_trace(
                go.Bar(
                    name='Regular Season',
                    x=pivot_points.index,
                    y=pivot_points["Regular Season"],
                    marker_color='lightblue',
                    text=pivot_points["Regular Season"].round(2),
                    textposition='outside'
                ),
                row=1, col=1
            )

        if "Playoff" in pivot_points.columns:
            fig_comp.add_trace(
                go.Bar(
                    name='Playoff',
                    x=pivot_points.index,
                    y=pivot_points["Playoff"],
                    marker_color='gold',
                    text=pivot_points["Playoff"].round(2),
                    textposition='outside'
                ),
                row=1, col=1
            )

        # Add traces for win %
        if "Regular Season" in pivot_winpct.columns:
            fig_comp.add_trace(
                go.Bar(
                    name='Regular Season',
                    x=pivot_winpct.index,
                    y=pivot_winpct["Regular Season"],
                    marker_color='lightblue',
                    text=pivot_winpct["Regular Season"].round(1).astype(str) + '%',
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=2
            )

        if "Playoff" in pivot_winpct.columns:
            fig_comp.add_trace(
                go.Bar(
                    name='Playoff',
                    x=pivot_winpct.index,
                    y=pivot_winpct["Playoff"],
                    marker_color='gold',
                    text=pivot_winpct["Playoff"].round(1).astype(str) + '%',
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=2
            )

        fig_comp.update_xaxes(tickangle=-45)
        fig_comp.update_layout(
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_comp, use_container_width=True, key=f"{prefix}_comparison")

        # Detailed comparison table
        with st.expander("📊 Detailed Stats", expanded=False):
            display_comp = comp_df.copy()
            display_comp['Win %'] = display_comp['Win %'].round(1)
            display_comp['Avg Points'] = display_comp['Avg Points'].round(2)
            display_comp['Avg Opp Points'] = display_comp['Avg Opp Points'].round(2)
            display_comp['Consistency (Std Dev)'] = display_comp['Consistency (Std Dev)'].round(2)

            st.dataframe(display_comp, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("🏆 Playoff Performance Breakdown")

        # Playoff-specific analysis
        playoff_only = filtered_data[filtered_data["game_type"] == "Playoff"]

        # Calculate playoff stats
        playoff_stats = []
        for manager in selected_managers:
            manager_playoff = playoff_only[playoff_only["manager"] == manager]

            if len(manager_playoff) > 0:
                playoff_stats.append({
                    "Manager": manager,
                    "Playoff Games": len(manager_playoff),
                    "Playoff Wins": manager_playoff["win"].sum(),
                    "Playoff Win %": manager_playoff["win"].mean() * 100,
                    "Avg Playoff PPG": manager_playoff["team_points"].mean(),
                    "Best Playoff Game": manager_playoff["team_points"].max(),
                    "Worst Playoff Game": manager_playoff["team_points"].min()
                })

        playoff_df = pd.DataFrame(playoff_stats).round(2)
        playoff_df = playoff_df.sort_values("Playoff Win %", ascending=False)

        # Playoff performance bar chart
        fig_playoff = go.Figure()

        playoff_sorted = playoff_df.sort_values("Playoff Win %", ascending=True)

        fig_playoff.add_trace(go.Bar(
            y=playoff_sorted['Manager'],
            x=playoff_sorted['Playoff Win %'],
            orientation='h',
            marker=dict(
                color=playoff_sorted['Playoff Win %'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Win %")
            ),
            text=playoff_sorted['Playoff Win %'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Playoff Win %: %{x:.1f}%<extra></extra>"
        ))

        fig_playoff.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        fig_playoff.update_layout(
            xaxis_title="Playoff Win Percentage",
            yaxis_title="",
            height=max(300, len(playoff_df) * 40),
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_playoff, use_container_width=True, key=f"{prefix}_playoff")

        # Playoff records
        st.subheader("📋 Playoff Records")

        st.dataframe(
            playoff_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Playoff Win %": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Playoff PPG": st.column_config.NumberColumn(format="%.2f"),
                "Best Playoff Game": st.column_config.NumberColumn(format="%.2f"),
                "Worst Playoff Game": st.column_config.NumberColumn(format="%.2f")
            }
        )

    with tab3:
        st.subheader("📈 Performance Differential")
        st.caption("How much better/worse are you in playoffs vs regular season?")

        # Calculate differential
        differential_stats = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]

            regular = manager_data[manager_data["game_type"] == "Regular Season"]
            playoff = manager_data[manager_data["game_type"] == "Playoff"]

            if len(regular) > 0 and len(playoff) > 0:
                ppg_diff = playoff["team_points"].mean() - regular["team_points"].mean()
                win_pct_diff = (playoff["win"].mean() - regular["win"].mean()) * 100

                differential_stats.append({
                    "Manager": manager,
                    "PPG Difference": ppg_diff,
                    "Win % Difference": win_pct_diff,
                    "Regular PPG": regular["team_points"].mean(),
                    "Playoff PPG": playoff["team_points"].mean(),
                    "Regular Win %": regular["win"].mean() * 100,
                    "Playoff Win %": playoff["win"].mean() * 100,
                    "Category": "🔥 Playoff Riser" if ppg_diff > 0 else "📉 Playoff Dropper"
                })

        if not differential_stats:
            st.info("Not enough data to calculate differentials.")
        else:
            diff_df = pd.DataFrame(differential_stats).round(2)

            # Differential bar chart
            fig_diff = go.Figure()

            diff_sorted = diff_df.sort_values("PPG Difference", ascending=True)

            colors = ['green' if x > 0 else 'red' for x in diff_sorted['PPG Difference']]

            fig_diff.add_trace(go.Bar(
                y=diff_sorted['Manager'],
                x=diff_sorted['PPG Difference'],
                orientation='h',
                marker_color=colors,
                text=diff_sorted['PPG Difference'].apply(lambda x: f"{x:+.1f}"),
                textposition='outside',
                hovertemplate="<b>%{y}</b><br>PPG Diff: %{x:+.1f}<extra></extra>"
            ))

            fig_diff.add_vline(x=0, line_color="black", line_width=2)

            fig_diff.update_layout(
                xaxis_title="PPG Difference (Playoff - Regular Season)",
                yaxis_title="",
                height=max(300, len(diff_df) * 40),
                template="plotly_white",
                showlegend=False
            )

            st.plotly_chart(fig_diff, use_container_width=True, key=f"{prefix}_diff")

            # Scatter plot: Regular vs Playoff performance
            st.subheader("🔍 Regular vs Playoff PPG")

            fig_scatter = go.Figure()

            fig_scatter.add_trace(go.Scatter(
                x=diff_df['Regular PPG'],
                y=diff_df['Playoff PPG'],
                mode='markers+text',
                text=diff_df['Manager'],
                textposition='top center',
                marker=dict(
                    size=15,
                    color=diff_df['PPG Difference'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Diff"),
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Regular PPG: %{x:.1f}<br>"
                    "Playoff PPG: %{y:.1f}<br>"
                    "<extra></extra>"
                )
            ))

            # Add 45-degree reference line (equal performance)
            min_val = min(diff_df['Regular PPG'].min(), diff_df['Playoff PPG'].min())
            max_val = max(diff_df['Regular PPG'].max(), diff_df['Playoff PPG'].max())

            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig_scatter.update_layout(
                xaxis_title="Regular Season PPG",
                yaxis_title="Playoff PPG",
                height=500,
                template="plotly_white",
                showlegend=False
            )

            st.plotly_chart(fig_scatter, use_container_width=True, key=f"{prefix}_scatter")

            # Differential table
            with st.expander("📊 Full Differential Stats", expanded=False):
                st.dataframe(diff_df, hide_index=True, use_container_width=True)

    # Key insights
    with st.expander("💡 Key Insights", expanded=False):
        if differential_stats:
            best_riser = max(differential_stats, key=lambda x: x["PPG Difference"])
            worst_dropper = min(differential_stats, key=lambda x: x["PPG Difference"])
            best_playoff_winpct = playoff_df.nlargest(1, "Playoff Win %").iloc[0] if not playoff_df.empty else None

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🔥 Biggest Playoff Riser",
                    best_riser["Manager"],
                    f"+{best_riser['PPG Difference']:.1f} PPG"
                )

            with col2:
                if best_playoff_winpct is not None:
                    st.metric(
                        "🏆 Best Playoff Win %",
                        best_playoff_winpct["Manager"],
                        f"{best_playoff_winpct['Playoff Win %']:.1f}%"
                    )

            with col3:
                st.metric(
                    "📉 Biggest Playoff Dropper",
                    worst_dropper["Manager"],
                    f"{worst_dropper['PPG Difference']:.1f} PPG"
                )
