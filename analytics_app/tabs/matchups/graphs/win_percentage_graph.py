#!/usr/bin/env python3
"""
Win Percentage visualization with premium UI design.

Improvements:
- Consistent header hierarchy and spacing tokens
- Dynamic Y-axis scaling for better readability
- Stronger .500 baseline with interpretation helpers
- Manager chips in card wrapper with Select All/Clear All
- Clamp early games toggle for cumulative view
- Milestone markers and smoothing options
- Thinner bar charts with sorting options
- Narrative cues for data interpretation
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from md.core import T, list_seasons, run_query

# Design tokens - consistent with scoring_trends.py
CHART_COLORS = [
    "#6366F1",  # Indigo
    "#EC4899",  # Pink
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#3B82F6",  # Blue
    "#8B5CF6",  # Violet
    "#EF4444",  # Red
    "#14B8A6",  # Teal
    "#F97316",  # Orange
    "#84CC16",  # Lime
    "#06B6D4",  # Cyan
    "#A855F7",  # Purple
]

# Softer colors for wins/losses
WIN_COLOR = "#22C55E"  # Green-500
LOSS_COLOR = "#E04B4B"  # Softer red


def _get_chart_color(index: int) -> str:
    """Get a contrast-safe color for chart series."""
    return CHART_COLORS[index % len(CHART_COLORS)]


def _render_section_header(title: str, subtitle: str, icon: str = "") -> None:
    """Render a section header with consistent styling."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <h3 style="margin: 0 0 0.375rem 0; font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">
                {icon_html}{title}
            </h3>
            <p style="margin: 0; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_manager_selector(managers: list, key: str, default_count: int = 3) -> list:
    """Render manager selector in a card wrapper with Select All/Clear All."""
    st.markdown(
        """
        <div style="
            background: var(--card-bg, rgba(255,255,255,0.03));
            border: 1px solid var(--border, #374151);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-subtle, #2D3748);
            ">
                <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-primary, #F9FAFB);">
                    Managers Selected
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Select All", key=f"{key}_all", use_container_width=True):
            st.session_state[key] = managers
            st.rerun()
    with col2:
        if st.button("Clear All", key=f"{key}_clear", use_container_width=True):
            st.session_state[key] = []
            st.rerun()

    default = managers[:default_count] if len(managers) >= default_count else managers
    selected = st.multiselect(
        "Select managers",
        options=managers,
        default=st.session_state.get(key, default),
        key=key,
        label_visibility="collapsed",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    return selected


def _get_base_layout(height: int = 500) -> dict:
    """Get standardized chart layout."""
    return {
        "height": height,
        "template": "plotly_white",
        "hovermode": "x unified",
        "showlegend": True,
        "legend": dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.02)",
            bordercolor="rgba(128,128,128,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
        "margin": dict(l=50, r=20, t=30, b=50),
        "xaxis": dict(gridcolor="rgba(128,128,128,0.1)", gridwidth=1),
        "yaxis": dict(gridcolor="rgba(128,128,128,0.1)", gridwidth=1),
        "font": dict(family="system-ui, -apple-system, sans-serif"),
    }


@st.fragment
def display_win_percentage_graph(df_dict, prefix=""):
    """Display win percentage trends with premium UI design."""

    # Header with description card
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
                Win Percentage Trends
            </h2>
            <p style="margin: 0; color: var(--text-secondary, #9CA3AF); font-size: 0.85rem; line-height: 1.5;">
                Track winning trends year-over-year and cumulatively.
                <strong>Above .500</strong> indicates a winning record.
                Early career games swing wildly; trends stabilize after ~40 games.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs with better spacing
    st.markdown("<div style='margin-top: 0.75rem;'>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(
        ["Year-by-Year", "Cumulative Career", "Win/Loss Records"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== YEAR-BY-YEAR TAB ====================
    with tab1:
        _render_yearly_view(prefix)

    # ==================== CUMULATIVE CAREER TAB ====================
    with tab2:
        _render_cumulative_view(prefix)

    # ==================== WIN/LOSS RECORDS TAB ====================
    with tab3:
        _render_winloss_view(prefix)


def _render_yearly_view(prefix: str):
    """Render year-by-year win percentage view."""
    _render_section_header(
        "Season Win Percentage by Year",
        "Compare win rates across seasons. The .500 line marks league average.",
    )

    # Load data
    with st.spinner("Loading win percentage data..."):
        query = f"""
            SELECT
                year,
                manager,
                SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) as wins,
                COUNT(*) as games
            FROM {T['matchup']}
            WHERE manager IS NOT NULL AND team_points IS NOT NULL
            GROUP BY year, manager
            ORDER BY year, manager
        """
        data = run_query(query)
        if data.empty:
            st.warning("No data found.")
            return

    data["win_pct"] = (data["wins"] / data["games"] * 100).round(1)

    # Manager selection in card
    managers = sorted(data["manager"].unique())
    selected_managers = _render_manager_selector(
        managers, f"{prefix}_yearly_managers", default_count=3
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Chart options
    col1, col2 = st.columns(2)
    with col1:
        show_trend_lines = st.checkbox(
            "Show Trend Lines",
            key=f"{prefix}_yearly_trends",
            help="Add linear regression trend lines",
        )
    with col2:
        dynamic_yaxis = st.checkbox(
            "Dynamic Y-Axis",
            value=True,
            key=f"{prefix}_yearly_dynamic",
            help="Scale Y-axis to data range instead of 0-100",
        )

    # Create figure
    fig = go.Figure()

    # Calculate dynamic y-axis range
    if dynamic_yaxis:
        min_pct = filtered_data["win_pct"].min()
        max_pct = filtered_data["win_pct"].max()
        y_min = max(0, min_pct - 10)
        y_max = min(100, max_pct + 10)
    else:
        y_min, y_max = 0, 100

    # Add manager traces
    for i, manager in enumerate(selected_managers):
        manager_data = filtered_data[filtered_data["manager"] == manager].sort_values("year")
        color = _get_chart_color(i)

        fig.add_trace(
            go.Scatter(
                x=manager_data["year"],
                y=manager_data["win_pct"],
                mode="lines+markers",
                name=manager,
                line=dict(width=3, color=color),
                marker=dict(size=10),
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Year: %{x}<br>"
                    "Win %: %{y:.1f}%<br>"
                    f"Record: {manager_data['wins'].iloc[0]}-{manager_data['games'].iloc[0] - manager_data['wins'].iloc[0]}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Add trend line if enabled
        if show_trend_lines and len(manager_data) >= 2:
            years = manager_data["year"].values.astype(float)
            pcts = manager_data["win_pct"].values
            z = np.polyfit(years, pcts, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=manager_data["year"],
                    y=p(years),
                    mode="lines",
                    name=f"{manager} trend",
                    line=dict(width=2, dash="dot", color=color),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add .500 reference line (stronger styling)
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(255,255,255,0.6)",
        line_width=2,
        annotation_text=".500 (league average)",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="rgba(255,255,255,0.8)",
    )

    # Apply layout
    layout = _get_base_layout(height=500)
    layout["xaxis_title"] = "Season"
    layout["yaxis_title"] = "Win Percentage"
    layout["yaxis"]["range"] = [y_min, y_max]
    fig.update_layout(**layout)
    fig.update_xaxes(tickmode="linear", dtick=1)

    # Mobile-friendly container
    st.markdown('<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_winpct_line")
    st.markdown("</div>", unsafe_allow_html=True)

    # Interpretation helper
    best_manager = filtered_data.groupby("manager")["win_pct"].mean().idxmax()
    best_avg = filtered_data.groupby("manager")["win_pct"].mean().max()
    st.markdown(
        f"""
        <div style="
            background: var(--card-bg, rgba(255,255,255,0.03));
            border-left: 3px solid var(--accent, #818CF8);
            padding: 0.5rem 0.75rem;
            margin: 0.75rem 0;
            font-size: 0.8rem;
            color: var(--text-secondary, #9CA3AF);
        ">
            <strong>{best_manager}</strong> leads with an average of <strong>{best_avg:.1f}%</strong> win rate across seasons.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expandable sections
    with st.expander("Year-over-Year Records", expanded=False):
        pivot = filtered_data.pivot(index="year", columns="manager", values="win_pct").round(1)
        st.dataframe(pivot, use_container_width=True)

    with st.expander("Trend Analysis", expanded=False):
        trends = []
        for manager in selected_managers:
            manager_data = filtered_data[filtered_data["manager"] == manager].sort_values("year")
            if len(manager_data) >= 2:
                latest = manager_data.iloc[-1]["win_pct"]
                previous = manager_data.iloc[-2]["win_pct"]
                change = latest - previous
                years = manager_data["year"].values
                pcts = manager_data["win_pct"].values
                slope = (pcts[-1] - pcts[0]) / (years[-1] - years[0]) if len(years) > 1 else 0
                trend = "Improving" if slope > 0 else "Declining" if slope < 0 else "Stable"
                trends.append({
                    "Manager": manager,
                    "Latest": f"{latest:.1f}%",
                    "YoY Change": f"{change:+.1f}%",
                    "Trend": trend,
                })
        if trends:
            st.dataframe(pd.DataFrame(trends), hide_index=True, use_container_width=True)


def _render_cumulative_view(prefix: str):
    """Render cumulative career win percentage view."""
    _render_section_header(
        "All-Time Win Percentage Progression",
        "Track career win % game-by-game. Early games swing wildly; trends stabilize after ~40 games.",
    )

    # Load data
    with st.spinner("Loading cumulative data..."):
        query = f"""
            SELECT
                year, week, manager,
                CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win,
                ROW_NUMBER() OVER (PARTITION BY manager ORDER BY year, week) as game_number
            FROM {T['matchup']}
            WHERE manager IS NOT NULL AND team_points IS NOT NULL
            ORDER BY manager, year, week
        """
        data = run_query(query)
        if data.empty:
            st.warning("No data found.")
            return

    # Calculate running totals
    data["cumulative_wins"] = data.groupby("manager")["win"].cumsum()
    data["cumulative_win_pct"] = (data["cumulative_wins"] / data["game_number"] * 100).round(2)

    # Manager selection
    managers = sorted(data["manager"].unique())
    selected_managers = _render_manager_selector(
        managers, f"{prefix}_cumulative_managers", default_count=3
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Chart options
    col1, col2, col3 = st.columns(3)
    with col1:
        clamp_early = st.checkbox(
            "Stabilize Early Games",
            key=f"{prefix}_clamp",
            help="Start from game 10 to reduce early volatility",
        )
    with col2:
        smooth_lines = st.checkbox(
            "Smooth (5-game avg)",
            key=f"{prefix}_smooth",
            help="Apply moving average smoothing",
        )
    with col3:
        show_milestones = st.checkbox(
            "Show Milestones",
            value=True,
            key=f"{prefix}_milestones",
            help="Mark 50, 100, 150 game milestones",
        )

    # Apply clamping
    if clamp_early:
        filtered_data = filtered_data[filtered_data["game_number"] >= 10].copy()

    # Create figure
    fig = go.Figure()

    # Add milestone markers
    if show_milestones:
        max_games = filtered_data["game_number"].max()
        for milestone in [50, 100, 150, 200]:
            if milestone <= max_games:
                fig.add_vline(
                    x=milestone,
                    line_dash="dot",
                    line_color="rgba(128,128,128,0.2)",
                    line_width=1,
                    annotation_text=f"{milestone} games",
                    annotation_position="top",
                    annotation_font_size=9,
                    annotation_font_color="rgba(200,200,200,0.7)",
                )

    # Add manager traces
    for i, manager in enumerate(selected_managers):
        manager_data = filtered_data[filtered_data["manager"] == manager].copy()
        color = _get_chart_color(i)

        y_values = manager_data["cumulative_win_pct"]
        if smooth_lines and len(manager_data) >= 5:
            y_values = manager_data["cumulative_win_pct"].rolling(5, min_periods=1).mean()

        fig.add_trace(
            go.Scatter(
                x=manager_data["game_number"],
                y=y_values,
                mode="lines",
                name=manager,
                line=dict(width=2.5, color=color),
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Game: %{x}<br>"
                    "Win %: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Add .500 reference line
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(255,255,255,0.6)",
        line_width=2,
        annotation_text=".500 (break-even)",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="rgba(255,255,255,0.8)",
    )

    # Layout
    layout = _get_base_layout(height=500)
    layout["xaxis_title"] = "Career Game Number"
    layout["yaxis_title"] = "Cumulative Win %"

    # Dynamic y-axis
    min_pct = filtered_data["cumulative_win_pct"].min()
    max_pct = filtered_data["cumulative_win_pct"].max()
    layout["yaxis"]["range"] = [max(0, min_pct - 5), min(100, max_pct + 5)]

    fig.update_layout(**layout)

    st.markdown('<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_cumulative_line")
    st.markdown("</div>", unsafe_allow_html=True)

    # Current standings with interpretation
    st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
    _render_section_header("Current All-Time Records", "Final career standings for selected managers")

    records = []
    for manager in selected_managers:
        final = data[data["manager"] == manager].iloc[-1]
        total_games = int(final["game_number"])
        total_wins = int(final["cumulative_wins"])
        total_losses = total_games - total_wins
        final_pct = final["cumulative_win_pct"]
        status = "Above .500" if final_pct > 50 else "Below .500" if final_pct < 50 else "At .500"
        records.append({
            "Manager": manager,
            "Record": f"{total_wins}-{total_losses}",
            "Games": total_games,
            "Win %": f"{final_pct:.2f}%",
            "Status": status,
        })

    records_df = pd.DataFrame(records).sort_values(
        "Win %", ascending=False, key=lambda x: x.str.rstrip("%").astype(float)
    )
    st.dataframe(records_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Recent form
    with st.expander("Recent Form (Last 20 Games)", expanded=False):
        recent_form = []
        for manager in selected_managers:
            manager_games = data[data["manager"] == manager]
            if len(manager_games) >= 20:
                last_20 = manager_games.tail(20)
                recent_wins = last_20["win"].sum()
                recent_pct = (recent_wins / 20 * 100).round(1)
                overall_pct = manager_games.iloc[-1]["cumulative_win_pct"]
                diff = recent_pct - overall_pct
                form = "Hot" if diff > 5 else "Cold" if diff < -5 else "Average"
                recent_form.append({
                    "Manager": manager,
                    "Last 20": f"{recent_pct:.1f}%",
                    "Career": f"{overall_pct:.2f}%",
                    "Diff": f"{diff:+.1f}%",
                    "Form": form,
                })
        if recent_form:
            st.dataframe(pd.DataFrame(recent_form), hide_index=True, use_container_width=True)
        else:
            st.info("Need at least 20 games for recent form analysis.")


def _render_winloss_view(prefix: str):
    """Render win/loss records view with bar charts."""
    _render_section_header(
        "Win/Loss Records Comparison",
        "Compare total wins, losses, and win percentages across managers.",
    )

    # Year selection
    available_years = list_seasons()
    if not available_years:
        st.error("No data available.")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        year_options = ["All Seasons"] + available_years
        selected_year = st.selectbox("Select Season", options=year_options, key=f"{prefix}_wl_year")

    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=["Win %", "Total Wins", "Games Played", "Manager Name"],
            key=f"{prefix}_wl_sort",
        )

    # Load data
    with st.spinner("Loading records..."):
        if selected_year == "All Seasons":
            query = f"""
                SELECT manager,
                    SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) as losses,
                    COUNT(*) as games
                FROM {T['matchup']}
                WHERE manager IS NOT NULL AND team_points IS NOT NULL
                GROUP BY manager
            """
        else:
            query = f"""
                SELECT manager,
                    SUM(CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN team_points <= opponent_points THEN 1 ELSE 0 END) as losses,
                    COUNT(*) as games
                FROM {T['matchup']}
                WHERE year = {int(selected_year)} AND manager IS NOT NULL AND team_points IS NOT NULL
                GROUP BY manager
            """
        wl_data = run_query(query)
        if wl_data.empty:
            st.warning("No data found.")
            return

    wl_data["win_pct"] = (wl_data["wins"] / wl_data["games"] * 100).round(1)

    # Apply sorting
    sort_map = {
        "Win %": ("win_pct", True),
        "Total Wins": ("wins", True),
        "Games Played": ("games", True),
        "Manager Name": ("manager", True),
    }
    sort_col, ascending = sort_map[sort_by]
    wl_sorted = wl_data.sort_values(sort_col, ascending=not ascending if sort_col != "manager" else ascending)

    # Bar thickness based on manager count
    bar_height = max(25, min(40, 400 // len(wl_data)))

    # Stacked bar chart
    st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
    _render_section_header("Total Wins and Losses", "Stacked comparison of win/loss records")

    fig_stack = go.Figure()

    fig_stack.add_trace(
        go.Bar(
            y=wl_sorted["manager"],
            x=wl_sorted["wins"],
            name="Wins",
            orientation="h",
            marker=dict(color=WIN_COLOR, line=dict(width=0)),
            text=wl_sorted["wins"],
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{y}</b><br>Wins: %{x}<extra></extra>",
        )
    )

    fig_stack.add_trace(
        go.Bar(
            y=wl_sorted["manager"],
            x=wl_sorted["losses"],
            name="Losses",
            orientation="h",
            marker=dict(color=LOSS_COLOR, line=dict(width=0)),
            text=wl_sorted["losses"],
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{y}</b><br>Losses: %{x}<extra></extra>",
        )
    )

    fig_stack.update_layout(
        barmode="stack",
        xaxis_title="Games",
        yaxis_title="",
        height=max(350, len(wl_data) * bar_height),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=100, r=20, t=40, b=40),
        bargap=0.3,
    )

    # Add subtle row separators via shapes
    for i in range(len(wl_sorted) - 1):
        fig_stack.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=i + 0.5, y1=i + 0.5,
            line=dict(color="rgba(128,128,128,0.15)", width=1),
        )

    st.plotly_chart(fig_stack, use_container_width=True, key=f"{prefix}_wl_stack")
    st.markdown("</div>", unsafe_allow_html=True)

    # Win percentage heatbar
    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    _render_section_header("Win Percentage Comparison", "Color-coded by performance. 50% = league average.")

    wl_pct_sorted = wl_sorted.sort_values("win_pct", ascending=True)

    fig_pct = go.Figure()

    fig_pct.add_trace(
        go.Bar(
            y=wl_pct_sorted["manager"],
            x=wl_pct_sorted["win_pct"],
            orientation="h",
            marker=dict(
                color=wl_pct_sorted["win_pct"],
                colorscale="RdYlGn",
                cmin=30,
                cmax=70,
                showscale=True,
                colorbar=dict(title="Win %", ticksuffix="%"),
            ),
            text=wl_pct_sorted.apply(
                lambda r: f"{r['win_pct']:.1f}% ({r['wins']}-{r['losses']})", axis=1
            ),
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Win %%: %{x:.1f}%%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add 50% vertical line
    fig_pct.add_vline(
        x=50,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        line_width=2,
        annotation_text="50%",
        annotation_position="top",
        annotation_font_size=10,
    )

    fig_pct.update_layout(
        xaxis_title="Win Percentage",
        yaxis_title="",
        height=max(350, len(wl_data) * bar_height),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=100, r=80, t=40, b=40),
        xaxis=dict(range=[0, max(100, wl_pct_sorted["win_pct"].max() + 10)]),
        bargap=0.3,
    )

    st.plotly_chart(fig_pct, use_container_width=True, key=f"{prefix}_wl_pct")
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed records accordion
    st.markdown("<div style='margin-top: 1.5rem; padding-top: 1rem;'>", unsafe_allow_html=True)
    with st.expander("Detailed Records", expanded=False):
        display_df = wl_sorted[["manager", "wins", "losses", "games", "win_pct"]].copy()
        display_df.columns = ["Manager", "Wins", "Losses", "Games", "Win %"]
        display_df["Rank"] = range(1, len(display_df) + 1)
        display_df = display_df[["Rank", "Manager", "Wins", "Losses", "Games", "Win %"]]

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={"Win %": st.column_config.NumberColumn(format="%.1f%%")},
        )

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"win_loss_records_{selected_year.replace(' ', '_')}.csv",
            mime="text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)
