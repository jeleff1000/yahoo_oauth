#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from md.core import T, list_seasons, run_query

# Shared color palette for consistency across visualizations
CHART_COLORS = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6",
    "#8B5CF6", "#EF4444", "#14B8A6", "#F97316", "#84CC16",
    "#06B6D4", "#A855F7",
]


def _get_manager_color(manager: str, managers: list) -> str:
    """Get consistent color for a manager."""
    if manager in managers:
        idx = managers.index(manager)
        return CHART_COLORS[idx % len(CHART_COLORS)]
    return CHART_COLORS[0]


def _render_manager_selector(managers: list, key: str, default_count: int = 5) -> list:
    """Render manager selection with card wrapper and Select All/Clear All."""
    default_selection = managers[:default_count] if len(managers) >= default_count else managers

    st.markdown("""
    <div class="manager-selector-card">
        <div class="manager-selector-header">
            <span class="manager-selector-title">Select Managers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Select All", key=f"{key}_all", use_container_width=True):
            st.session_state[f"{key}_managers"] = managers
    with col2:
        if st.button("Clear All", key=f"{key}_clear", use_container_width=True):
            st.session_state[f"{key}_managers"] = []

    current_selection = st.session_state.get(f"{key}_managers", default_selection)
    selected = st.multiselect(
        "Managers",
        options=managers,
        default=current_selection,
        key=f"{key}_managers_widget",
        label_visibility="collapsed",
    )

    return selected


def _render_summary_insight(text: str, icon: str = "") -> None:
    """Render a summary insight sentence at the top of a module."""
    st.markdown(f"""
    <div style="background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius-md);
                padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.9rem; color: var(--text-secondary);">
        {icon} {text}
    </div>
    """, unsafe_allow_html=True)


@st.fragment
def display_margin_of_victory_graph(df_dict=None, prefix=""):
    """
    Analyze margin of victory/defeat - identify blowout winners, nail-biters, and unlucky losers.
    """
    st.markdown("""
    <div class="chart-title-container">
        <h2 class="chart-title">Margin of Victory Analysis</h2>
        <p class="chart-subtitle">Beyond W-L records: see how managers win and lose. Identify dominant performers, close-game clutch players, and unlucky losers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Year selection
    available_years = list_seasons()
    if not available_years:
        st.error("No data available.")
        return

    year_options = ["All Seasons"] + available_years
    selected_year = st.selectbox(
        "Select Season", options=year_options, key=f"{prefix}_year"
    )

    # Load data - close_game is a preset column from source data
    with st.spinner("Loading margin data..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT
                    manager,
                    year,
                    week,
                    team_points,
                    opponent_points,
                    (team_points - opponent_points) as margin,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    close_game
                FROM {T['matchup']}
                WHERE manager IS NOT NULL
                  AND team_points IS NOT NULL
                  AND opponent_points IS NOT NULL
                ORDER BY manager, year, week
            """
        else:
            query = f"""
                SELECT
                    manager,
                    week,
                    team_points,
                    opponent_points,
                    (team_points - opponent_points) as margin,
                    CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                    close_game
                FROM {T['matchup']}
                WHERE year = {int(selected_year)}
                  AND manager IS NOT NULL
                  AND team_points IS NOT NULL
                  AND opponent_points IS NOT NULL
                ORDER BY manager, week
            """
        data = run_query(query)

        if data.empty:
            st.warning("No data found.")
            return

    # Manager selection
    managers = sorted(data["manager"].unique())
    selected_managers = _render_manager_selector(managers, f"{prefix}_mov", default_count=5)

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)].copy()

    # Pre-calculate league-wide stats for insights
    all_close = data[data["close_game"] == 1]
    league_close_win_rate = (all_close["win"].sum() / len(all_close) * 100) if len(all_close) > 0 else 50
    avg_margin = data["margin"].mean()

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Distribution", "Close Games", "Statistics"])

    with tab1:
        _render_distribution_tab(filtered_data, selected_managers, managers, prefix, avg_margin)

    with tab2:
        _render_close_games_tab(filtered_data, selected_managers, managers, prefix, league_close_win_rate)

    with tab3:
        _render_statistics_tab(filtered_data, selected_managers, managers, prefix)


def _render_distribution_tab(filtered_data: pd.DataFrame, selected_managers: list, all_managers: list, prefix: str, avg_margin: float):
    """Render the distribution analysis tab."""
    st.markdown("""
    <div class="chart-title-container">
        <h3 class="chart-title">Victory Margin Distribution</h3>
        <p class="chart-subtitle">Positive = Win, Negative = Loss. See your pattern of dominance vs close games.</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary insight
    most_dominant = None
    highest_avg = float('-inf')
    for mgr in selected_managers:
        mgr_data = filtered_data[filtered_data["manager"] == mgr]
        wins = mgr_data[mgr_data["win"] == 1]
        if len(wins) > 0:
            avg_win_margin = wins["margin"].mean()
            if avg_win_margin > highest_avg:
                highest_avg = avg_win_margin
                most_dominant = mgr

    if most_dominant:
        _render_summary_insight(
            f"<strong>{most_dominant}</strong> has the highest average win margin at <strong>+{highest_avg:.1f} pts</strong>. "
            f"League average margin is {avg_margin:+.1f} pts.",
            ""
        )

    # Histogram display mode toggle
    display_mode = st.radio(
        "Histogram Display",
        options=["Overlay", "Side-by-Side", "Density"],
        horizontal=True,
        key=f"{prefix}_hist_mode",
        help="Overlay stacks histograms, Side-by-Side groups them, Density shows smooth curves"
    )

    # Create histogram based on mode
    fig_hist = go.Figure()

    if display_mode == "Density":
        # KDE-style density curves
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]
            color = _get_manager_color(manager, all_managers)

            # Use histogram with normalized density
            fig_hist.add_trace(
                go.Histogram(
                    x=manager_data["margin"],
                    name=manager,
                    opacity=0.6,
                    nbinsx=40,
                    histnorm="probability density",
                    marker_color=color,
                    hovertemplate="<b>%{fullData.name}</b><br>Margin: %{x:.1f}<br>Density: %{y:.3f}<extra></extra>",
                )
            )
        barmode = "overlay"
    elif display_mode == "Side-by-Side":
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]
            color = _get_manager_color(manager, all_managers)

            fig_hist.add_trace(
                go.Histogram(
                    x=manager_data["margin"],
                    name=manager,
                    nbinsx=20,
                    marker_color=color,
                    hovertemplate="<b>%{fullData.name}</b><br>Margin: %{x:.1f}<br>Count: %{y}<extra></extra>",
                )
            )
        barmode = "group"
    else:  # Overlay
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager]
            color = _get_manager_color(manager, all_managers)

            fig_hist.add_trace(
                go.Histogram(
                    x=manager_data["margin"],
                    name=manager,
                    opacity=0.5,
                    nbinsx=30,
                    marker_color=color,
                    hovertemplate="<b>%{fullData.name}</b><br>Margin: %{x:.1f}<br>Count: %{y}<extra></extra>",
                )
            )
        barmode = "overlay"

    # Add reference line at 0
    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(239, 68, 68, 0.8)",
        line_width=2,
        annotation_text="Even",
        annotation_position="top",
    )

    fig_hist.update_layout(
        xaxis_title="Margin of Victory (Points)",
        yaxis_title="Number of Games" if display_mode != "Density" else "Density",
        barmode=barmode,
        height=450,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        xaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_hist, use_container_width=True, key=f"{prefix}_hist")

    # Box plot for comparison
    st.markdown("""
    <div class="chart-title-container" style="margin-top: 1.5rem;">
        <h3 class="chart-title">Margin Comparison (Box Plot)</h3>
        <p class="chart-subtitle">Compare the spread and median margins across managers.</p>
    </div>
    """, unsafe_allow_html=True)

    fig_box = go.Figure()

    for manager in selected_managers:
        manager_data = filtered_data[filtered_data["manager"] == manager]
        color = _get_manager_color(manager, all_managers)

        fig_box.add_trace(
            go.Box(
                y=manager_data["margin"],
                name=manager,
                boxmean="sd",
                marker_color=color,
                line_color=color,
                hovertemplate="<b>%{fullData.name}</b><br>Margin: %{y:.1f}<extra></extra>",
            )
        )

    fig_box.add_hline(y=0, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", line_width=1)

    fig_box.update_layout(
        yaxis_title="Margin of Victory (Points)",
        showlegend=False,
        height=400,
        template="plotly_white",
        xaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)"),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_box, use_container_width=True, key=f"{prefix}_box")


def _render_close_games_tab(filtered_data: pd.DataFrame, selected_managers: list, all_managers: list, prefix: str, league_close_win_rate: float):
    """Render the close games analysis tab."""
    st.markdown("""
    <div class="chart-title-container">
        <h3 class="chart-title">Close Game Analysis</h3>
        <p class="chart-subtitle">Close games as defined in source data - who's clutch and who's unlucky?</p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate close game stats using preset close_game column
    close_stats = []
    for manager in selected_managers:
        manager_data = filtered_data[filtered_data["manager"] == manager]

        close_games = manager_data[manager_data["close_game"] == 1]
        close_wins = close_games[close_games["win"] == 1]
        close_losses = close_games[close_games["win"] == 0]

        blowout_games = manager_data[manager_data["close_game"] == 0]
        blowout_wins = blowout_games[blowout_games["win"] == 1]
        blowout_losses = blowout_games[blowout_games["win"] == 0]

        close_win_pct = (len(close_wins) / len(close_games) * 100) if len(close_games) > 0 else 0
        blowout_win_pct = (len(blowout_wins) / len(blowout_games) * 100) if len(blowout_games) > 0 else 0
        blowout_loss_pct = (len(blowout_losses) / len(manager_data) * 100) if len(manager_data) > 0 else 0

        close_stats.append(
            {
                "Manager": manager,
                "Close Wins": len(close_wins),
                "Close Losses": len(close_losses),
                "Close Games": len(close_games),
                "Close Win %": close_win_pct,
                "Blowout Wins": len(blowout_wins),
                "Blowout Losses": len(blowout_losses),
                "Blowout Win %": blowout_win_pct,
                "Blowout Loss Rate": blowout_loss_pct,
                "Total Games": len(manager_data),
            }
        )

    close_df = pd.DataFrame(close_stats)

    # Summary insight
    if not close_df.empty:
        most_clutch = close_df[close_df["Close Games"] >= 3].nlargest(1, "Close Win %")
        most_unlucky = close_df.nlargest(1, "Close Losses")

        insight_parts = []
        if not most_clutch.empty:
            mc = most_clutch.iloc[0]
            insight_parts.append(f"<strong>{mc['Manager']}</strong> is the most clutch with a <strong>{mc['Close Win %']:.0f}%</strong> close game win rate")
        if not most_unlucky.empty:
            mu = most_unlucky.iloc[0]
            insight_parts.append(f"<strong>{mu['Manager']}</strong> has suffered the most close losses (<strong>{int(mu['Close Losses'])}</strong>)")

        if insight_parts:
            _render_summary_insight(
                f"{'. '.join(insight_parts)}. League average close game win rate is {league_close_win_rate:.0f}%.",
                ""
            )

    # Blowout metrics row
    st.markdown("##### Blowout Performance")
    cols = st.columns(len(selected_managers))
    for i, manager in enumerate(selected_managers):
        with cols[i]:
            row = close_df[close_df["Manager"] == manager].iloc[0]
            blowout_rate = row["Blowout Win %"]
            loss_rate = row["Blowout Loss Rate"]
            color = _get_manager_color(manager, all_managers)

            st.markdown(f"""
            <div style="background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius-md);
                        padding: 0.75rem; text-align: center; border-left: 3px solid {color};">
                <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.25rem;">{manager}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">
                    Blowout W: <span style="color: var(--success);">{blowout_rate:.0f}%</span> |
                    Blown Out: <span style="color: var(--error);">{loss_rate:.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # Stacked bar chart of close vs blowout
    st.markdown("##### Game Breakdown by Type")

    fig_close = go.Figure()

    fig_close.add_trace(
        go.Bar(
            name="Close Wins",
            x=close_df["Manager"],
            y=close_df["Close Wins"],
            marker_color="rgba(16, 185, 129, 0.7)",
            text=close_df["Close Wins"],
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Close Wins: %{y}<extra></extra>",
        )
    )

    fig_close.add_trace(
        go.Bar(
            name="Close Losses",
            x=close_df["Manager"],
            y=close_df["Close Losses"],
            marker_color="rgba(239, 68, 68, 0.5)",
            text=close_df["Close Losses"],
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Close Losses: %{y}<extra></extra>",
        )
    )

    fig_close.add_trace(
        go.Bar(
            name="Blowout Wins",
            x=close_df["Manager"],
            y=close_df["Blowout Wins"],
            marker_color="rgba(16, 185, 129, 1)",
            text=close_df["Blowout Wins"],
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Blowout Wins: %{y}<extra></extra>",
        )
    )

    fig_close.add_trace(
        go.Bar(
            name="Blowout Losses",
            x=close_df["Manager"],
            y=close_df["Blowout Losses"],
            marker_color="rgba(239, 68, 68, 1)",
            text=close_df["Blowout Losses"],
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Blowout Losses: %{y}<extra></extra>",
        )
    )

    fig_close.update_layout(
        barmode="stack",
        xaxis_title="",
        yaxis_title="Number of Games",
        height=400,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        xaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)"),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_close, use_container_width=True, key=f"{prefix}_close")

    # Close game win percentage horizontal bar chart
    st.markdown("##### Close Game Win %")

    fig_close_pct = go.Figure()

    close_df_sorted = close_df.sort_values("Close Win %", ascending=True)

    # Add bars with individual colors
    for _, row in close_df_sorted.iterrows():
        color = _get_manager_color(row["Manager"], all_managers)
        fig_close_pct.add_trace(
            go.Bar(
                y=[row["Manager"]],
                x=[row["Close Win %"]],
                orientation="h",
                marker_color=color,
                text=f"{row['Close Win %']:.1f}%",
                textposition="outside",
                hovertemplate=f"<b>{row['Manager']}</b><br>Close Win %: {row['Close Win %']:.1f}%<br>Close Games: {row['Close Games']}<extra></extra>",
                showlegend=False,
            )
        )

    # Add 50% reference line
    fig_close_pct.add_vline(
        x=50,
        line_dash="dash",
        line_color="rgba(156, 163, 175, 0.8)",
        line_width=2,
        annotation_text="50%",
        annotation_position="top",
    )

    fig_close_pct.update_layout(
        xaxis_title="Win Percentage in Close Games",
        yaxis_title="",
        height=max(300, len(close_df) * 45),
        template="plotly_white",
        showlegend=False,
        xaxis=dict(range=[0, 105], gridcolor="rgba(128, 128, 128, 0.1)"),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_close_pct, use_container_width=True, key=f"{prefix}_close_pct"
    )

    # Display table
    with st.expander("Detailed Close Game Stats", expanded=False):
        display_close = close_df.copy()
        display_close["Close Win %"] = display_close["Close Win %"].round(1)
        display_close["Blowout Win %"] = display_close["Blowout Win %"].round(1)
        st.dataframe(
            display_close[["Manager", "Close Wins", "Close Losses", "Close Games", "Close Win %", "Blowout Wins", "Blowout Losses", "Blowout Win %"]],
            hide_index=True,
            use_container_width=True
        )


def _render_statistics_tab(filtered_data: pd.DataFrame, selected_managers: list, all_managers: list, prefix: str):
    """Render the statistics analysis tab."""
    st.markdown("""
    <div class="chart-title-container">
        <h3 class="chart-title">Margin Statistics</h3>
        <p class="chart-subtitle">Detailed breakdown of average margins, biggest wins, and worst losses.</p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate stats by manager
    margin_stats = []
    for manager in selected_managers:
        manager_data = filtered_data[filtered_data["manager"] == manager]

        wins = manager_data[manager_data["win"] == 1]
        losses = manager_data[manager_data["win"] == 0]

        margin_stats.append(
            {
                "Manager": manager,
                "Avg Margin (All)": manager_data["margin"].mean(),
                "Avg Win Margin": wins["margin"].mean() if len(wins) > 0 else 0,
                "Avg Loss Margin": losses["margin"].mean() if len(losses) > 0 else 0,
                "Biggest Win": manager_data["margin"].max(),
                "Worst Loss": manager_data["margin"].min(),
                "Std Dev": manager_data["margin"].std(),
            }
        )

    stats_df = pd.DataFrame(margin_stats).round(2)

    # Summary insight
    if not stats_df.empty:
        most_dominant = stats_df.nlargest(1, "Avg Win Margin").iloc[0]
        most_consistent = stats_df.nsmallest(1, "Std Dev").iloc[0]
        biggest_win = stats_df.nlargest(1, "Biggest Win").iloc[0]

        _render_summary_insight(
            f"<strong>{most_dominant['Manager']}</strong> wins by an average of <strong>+{most_dominant['Avg Win Margin']:.1f} pts</strong>. "
            f"<strong>{most_consistent['Manager']}</strong> has the most consistent margins (lowest variance). "
            f"Biggest win: <strong>{biggest_win['Manager']}</strong> with a <strong>+{biggest_win['Biggest Win']:.0f} pt</strong> victory.",
            ""
        )

    # Display stats table
    st.dataframe(
        stats_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Avg Margin (All)": st.column_config.NumberColumn(format="%.2f"),
            "Avg Win Margin": st.column_config.NumberColumn(format="+%.2f"),
            "Avg Loss Margin": st.column_config.NumberColumn(format="%.2f"),
            "Biggest Win": st.column_config.NumberColumn(format="+%.2f"),
            "Worst Loss": st.column_config.NumberColumn(format="%.2f"),
            "Std Dev": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # Bar chart comparing average margins
    st.markdown("""
    <div class="chart-title-container" style="margin-top: 1.5rem;">
        <h3 class="chart-title">Average Win vs Loss Margin</h3>
        <p class="chart-subtitle">Compare how decisively managers win vs how badly they lose.</p>
    </div>
    """, unsafe_allow_html=True)

    fig_avg_margin = go.Figure()

    # Sort by avg win margin
    stats_df_sorted = stats_df.sort_values("Avg Win Margin", ascending=True)

    fig_avg_margin.add_trace(
        go.Bar(
            name="Avg Win Margin",
            y=stats_df_sorted["Manager"],
            x=stats_df_sorted["Avg Win Margin"],
            orientation="h",
            marker_color="rgba(16, 185, 129, 0.8)",
            text=stats_df_sorted["Avg Win Margin"].apply(lambda x: f"+{x:.1f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg Win Margin: +%{x:.1f}<extra></extra>",
        )
    )

    fig_avg_margin.add_trace(
        go.Bar(
            name="Avg Loss Margin",
            y=stats_df_sorted["Manager"],
            x=stats_df_sorted["Avg Loss Margin"],
            orientation="h",
            marker_color="rgba(239, 68, 68, 0.8)",
            text=stats_df_sorted["Avg Loss Margin"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg Loss Margin: %{x:.1f}<extra></extra>",
        )
    )

    fig_avg_margin.update_layout(
        barmode="group",
        xaxis_title="Average Margin (Points)",
        yaxis_title="",
        height=max(350, len(stats_df) * 50),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)", zeroline=True, zerolinecolor="rgba(156, 163, 175, 0.5)"),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_avg_margin, use_container_width=True, key=f"{prefix}_avg_margin"
    )

    # Key insights - consolidated at the bottom
    st.markdown("##### Key Takeaways")

    # Find the most dominant winner (highest avg win margin)
    most_dominant = stats_df.nlargest(1, "Avg Win Margin").iloc[0]

    # Find the unluckiest (most close losses)
    close_stats = []
    for manager in selected_managers:
        manager_data = filtered_data[filtered_data["manager"] == manager]
        close_games = manager_data[manager_data["close_game"] == 1]
        close_losses = close_games[close_games["win"] == 0]
        close_wins = close_games[close_games["win"] == 1]
        close_win_pct = (len(close_wins) / len(close_games) * 100) if len(close_games) > 0 else 0
        close_stats.append({
            "Manager": manager,
            "Close Losses": len(close_losses),
            "Close Games": len(close_games),
            "Close Win %": close_win_pct,
        })
    close_df = pd.DataFrame(close_stats)

    most_unlucky = close_df.nlargest(1, "Close Losses").iloc[0]

    # Find clutch performer (highest close game win %)
    clutch_candidates = close_df[close_df["Close Games"] >= 3]
    if not clutch_candidates.empty:
        most_clutch = clutch_candidates.nlargest(1, "Close Win %").iloc[0]
    else:
        most_clutch = None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Most Dominant",
            most_dominant["Manager"],
            f"+{most_dominant['Avg Win Margin']:.1f} pts/win",
        )

    with col2:
        if most_clutch is not None:
            st.metric(
                "Most Clutch",
                most_clutch["Manager"],
                f"{most_clutch['Close Win %']:.0f}% in close games",
            )
        else:
            st.info("Not enough close games for analysis")

    with col3:
        st.metric(
            "Most Close Losses",
            most_unlucky["Manager"],
            f"{int(most_unlucky['Close Losses'])} games",
        )
