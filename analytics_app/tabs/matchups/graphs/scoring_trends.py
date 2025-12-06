#!/usr/bin/env python3
"""
Unified scoring trends visualization - replaces all_time_scoring_graphs and weekly_scoring_graphs.
Provides weekly, cumulative, and season average views in one component.

Design improvements:
- Segmented control mode picker with helper text
- Enhanced manager selection with card wrapper, Select All/Clear All
- Better chart titles with subtitles
- Highlight manager mode, benchmarks, season markers
- Mobile-optimized horizontal scrolling
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from md.core import T, list_seasons, run_query

# Design tokens - contrast-safe colors for charts
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


def _get_chart_color(index: int) -> str:
    """Get a contrast-safe color for chart series."""
    return CHART_COLORS[index % len(CHART_COLORS)]


def _render_chart_title(title: str, subtitle: str) -> None:
    """Render a chart title with subtitle following design framework."""
    st.markdown(
        f"""
        <div style="margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">
                {title}
            </h3>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: var(--text-secondary);">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_segmented_control(options: list, key: str, descriptions: list = None) -> str:
    """
    Render a segmented control (pill-style selector) with optional descriptions.

    Args:
        options: List of option labels
        key: Streamlit session state key
        descriptions: Optional list of helper text for each option

    Returns:
        Selected option value
    """
    # Initialize session state
    if key not in st.session_state:
        st.session_state[key] = options[0]

    # Generate unique button keys
    cols = st.columns(len(options))

    for i, (col, opt) in enumerate(zip(cols, options)):
        with col:
            is_selected = st.session_state[key] == opt

            # Button styling based on selection
            button_type = "primary" if is_selected else "secondary"

            if st.button(
                opt,
                key=f"{key}_btn_{i}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state[key] = opt
                st.rerun()

            # Show description if provided
            if descriptions and i < len(descriptions):
                st.markdown(
                    f"<p style='font-size: 0.65rem; color: var(--text-muted); text-align: center; margin-top: 0.25rem;'>{descriptions[i]}</p>",
                    unsafe_allow_html=True,
                )

    return st.session_state[key]


def _render_manager_selector(
    managers: list,
    key: str,
    default_count: int = 5,
) -> list:
    """
    Render an enhanced manager multi-select with card wrapper and Select All/Clear All.

    Args:
        managers: List of available managers
        key: Streamlit session state key
        default_count: Number of managers to select by default

    Returns:
        List of selected managers
    """
    # Card wrapper
    st.markdown(
        """
        <div style="
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            ">
                <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-primary);">
                    Managers Selected
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Select All / Clear All buttons
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("Select All", key=f"{key}_select_all", use_container_width=True):
            st.session_state[key] = managers
            st.rerun()

    with col2:
        if st.button("Clear All", key=f"{key}_clear_all", use_container_width=True):
            st.session_state[key] = []
            st.rerun()

    # Multi-select
    default = managers[:default_count] if len(managers) >= default_count else managers

    selected = st.multiselect(
        "Select managers",
        options=managers,
        default=st.session_state.get(key, default),
        key=key,
        label_visibility="collapsed",
    )

    # Show count badge if many selected
    if len(selected) > 5:
        st.markdown(
            f"""
            <span style="
                background: var(--accent-subtle);
                border: 1px solid var(--accent);
                padding: 0.25rem 0.5rem;
                border-radius: 12px;
                font-size: 0.75rem;
                color: var(--text-primary);
            ">
                {len(selected)} managers selected
            </span>
            """,
            unsafe_allow_html=True,
        )

    # Close card wrapper
    st.markdown("</div>", unsafe_allow_html=True)

    return selected


def _get_base_chart_layout(height: int = 500) -> dict:
    """Get standardized chart layout with design improvements."""
    return {
        "height": height,
        "template": "plotly_white",
        "hovermode": "x unified",
        "showlegend": True,
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        "margin": dict(l=50, r=20, t=40, b=50),
        "xaxis": dict(
            gridcolor="rgba(128, 128, 128, 0.1)",
            gridwidth=1,
            showgrid=True,
        ),
        "yaxis": dict(
            gridcolor="rgba(128, 128, 128, 0.1)",
            gridwidth=1,
            showgrid=True,
        ),
        "font": dict(family="system-ui, -apple-system, sans-serif"),
    }


@st.fragment
def display_scoring_trends(df_dict=None, prefix=""):
    """
    Unified scoring trends with multiple view modes.
    """
    # Header with description card
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.05) 100%);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h2 style="margin: 0 0 0.5rem 0; font-size: 1.25rem; font-weight: 600;">
                Scoring Trends
            </h2>
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.85rem;">
                Track scoring patterns: View weekly performance, cumulative career totals, or season averages.
                Identify hot streaks, slumps, and long-term trends.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mode selection with segmented control
    st.markdown(
        "<p style='font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);'>View Mode</p>",
        unsafe_allow_html=True,
    )

    view_mode = _render_segmented_control(
        options=["Weekly", "Cumulative", "Averages"],
        key=f"{prefix}_view_mode_seg",
        descriptions=[
            "Week-by-week scores",
            "Total points over career",
            "Per-season average",
        ],
    )

    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

    # Year selection (only for weekly mode)
    selected_year = None
    if view_mode == "Weekly":
        available_years = list_seasons()
        if not available_years:
            st.error("No data available.")
            return

        selected_year = st.selectbox(
            "Select Season",
            options=available_years,
            index=0,
            key=f"{prefix}_year",
        )

    # Load data based on view mode
    with st.spinner("Loading scoring data..."):
        if view_mode == "Weekly":
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
        elif view_mode == "Cumulative":
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
        else:  # Averages
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

    # Manager selection with enhanced UI
    managers = sorted(data["manager"].unique())
    selected_managers = _render_manager_selector(
        managers=managers,
        key=f"{prefix}_managers",
        default_count=5,
    )

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    filtered_data = data[data["manager"].isin(selected_managers)]

    # Render based on view mode
    if view_mode == "Weekly":
        _render_weekly_view(filtered_data, selected_year, selected_managers, prefix)
    elif view_mode == "Cumulative":
        _render_cumulative_view(filtered_data, selected_managers, prefix)
    else:
        _render_season_averages_view(filtered_data, selected_managers, prefix)


def _render_weekly_view(data: pd.DataFrame, year: str, selected_managers: list, prefix: str):
    """Render weekly scoring for a single season with enhanced features."""

    # Chart title with subtitle
    _render_chart_title(
        f"Week-by-Week Performance - {year}",
        "Compare manager scoring volatility and peak weeks. Click a manager in the legend to highlight.",
    )

    # Chart options
    col1, col2, col3 = st.columns(3)

    with col1:
        highlight_manager = st.selectbox(
            "Highlight Manager",
            options=["None"] + selected_managers,
            key=f"{prefix}_highlight",
            help="Select a manager to emphasize their line",
        )

    with col2:
        show_benchmarks = st.checkbox(
            "Show Benchmarks",
            key=f"{prefix}_benchmarks",
            help="Show league average and high per week",
        )

    with col3:
        smooth_lines = st.checkbox(
            "Smooth Curves",
            key=f"{prefix}_smooth",
            help="Use smooth curves instead of jagged lines",
        )

    # Create figure
    fig = go.Figure()

    # Calculate benchmarks if needed
    if show_benchmarks:
        weekly_stats = data.groupby("week")["team_points"].agg(["mean", "max"]).reset_index()

        # League average line
        fig.add_trace(
            go.Scatter(
                x=weekly_stats["week"],
                y=weekly_stats["mean"],
                mode="lines",
                name="League Avg",
                line=dict(width=2, dash="dash", color="rgba(128, 128, 128, 0.6)"),
                hovertemplate="<b>League Avg</b><br>Week: %{x}<br>Avg: %{y:.2f}<extra></extra>",
            )
        )

        # League high line
        fig.add_trace(
            go.Scatter(
                x=weekly_stats["week"],
                y=weekly_stats["max"],
                mode="lines",
                name="League High",
                line=dict(width=1, dash="dot", color="rgba(16, 185, 129, 0.5)"),
                hovertemplate="<b>League High</b><br>Week: %{x}<br>High: %{y:.2f}<extra></extra>",
            )
        )

    # Add manager traces - highlighted manager gets glow effect
    highlighted_idx = None
    for i, manager in enumerate(data["manager"].unique()):
        manager_data = data[data["manager"] == manager].sort_values("week")

        # Determine opacity based on highlight mode (40% dimming for non-highlighted)
        if highlight_manager != "None":
            is_highlighted = manager == highlight_manager
            opacity = 1.0 if is_highlighted else 0.35
            line_width = 4.5 if is_highlighted else 1.5
            marker_size = 10 if is_highlighted else 6
            if is_highlighted:
                highlighted_idx = i
        else:
            opacity = 1.0
            line_width = 2.5
            marker_size = 8

        color = _get_chart_color(i)
        line_shape = "spline" if smooth_lines else "linear"

        # Add glow effect behind highlighted manager
        if highlight_manager != "None" and manager == highlight_manager:
            # Shadow/glow trace (wider, semi-transparent)
            fig.add_trace(
                go.Scatter(
                    x=manager_data["week"],
                    y=manager_data["team_points"],
                    mode="lines",
                    name=f"{manager}_glow",
                    line=dict(width=12, color=color, shape=line_shape),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=manager_data["week"],
                y=manager_data["team_points"],
                mode="lines+markers",
                name=manager,
                line=dict(width=line_width, color=color, shape=line_shape),
                marker=dict(size=marker_size),
                opacity=opacity,
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Week: %{x}<br>"
                    "Points: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Apply layout
    layout = _get_base_chart_layout(height=500)
    layout["xaxis_title"] = "Week"
    layout["yaxis_title"] = "Team Points"
    fig.update_layout(**layout)

    # Mobile-friendly chart container
    st.markdown(
        '<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_weekly_chart")
    st.markdown("</div>", unsafe_allow_html=True)

    # Tooltip summary cards
    st.markdown(
        "<p style='font-size: 0.8rem; font-weight: 600; margin: 1rem 0 0.5rem 0;'>Quick Stats</p>",
        unsafe_allow_html=True,
    )

    stats = (
        data.groupby("manager")
        .agg({"team_points": ["mean", "max", "min", "std"], "win": "sum"})
        .round(2)
    )
    stats.columns = ["Avg", "High", "Low", "StdDev", "Wins"]

    # Display as compact metric cards
    stat_cols = st.columns(min(len(selected_managers), 4))
    for i, manager in enumerate(selected_managers[:4]):
        if manager in stats.index:
            with stat_cols[i]:
                manager_stats = stats.loc[manager]
                st.markdown(
                    f"""
                    <div style="
                        background: var(--bg-secondary);
                        border: 1px solid var(--border);
                        border-radius: 6px;
                        padding: 0.5rem;
                        text-align: center;
                    ">
                        <div style="font-weight: 600; font-size: 0.8rem; color: {_get_chart_color(selected_managers.index(manager))};">
                            {manager}
                        </div>
                        <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.25rem;">
                            Avg: {manager_stats['Avg']:.1f} | High: {manager_stats['High']:.1f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Detailed statistics (collapsible)
    with st.expander("Detailed Week-by-Week Data", expanded=False):
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


def _render_cumulative_view(data: pd.DataFrame, selected_managers: list, prefix: str):
    """Render cumulative all-time scoring progression with enhanced features."""

    # Chart title with subtitle
    _render_chart_title(
        "Career Scoring Progression",
        "Track total lifetime points across all seasons. Steeper lines indicate higher scoring.",
    )

    # Options
    col1, col2 = st.columns(2)

    with col1:
        normalize_by_games = st.checkbox(
            "Normalize by Games Played",
            key=f"{prefix}_normalize",
            help="Show points per game instead of raw totals for fair comparison",
        )

    with col2:
        show_season_markers = st.checkbox(
            "Show Season Boundaries",
            key=f"{prefix}_season_markers",
            value=True,
            help="Display vertical lines at each season boundary",
        )

    # Calculate cumulative points
    data = data.copy()

    if normalize_by_games:
        data["cumulative_points"] = data.groupby("manager")["team_points"].cumsum()
        data["display_value"] = data["cumulative_points"] / data["game_number"]
        y_title = "Average Points Per Game"
    else:
        data["cumulative_points"] = data.groupby("manager")["team_points"].cumsum()
        data["display_value"] = data["cumulative_points"]
        y_title = "Cumulative Points"

    # Create figure
    fig = go.Figure()

    # Add season boundary markers if enabled
    if show_season_markers and not normalize_by_games:
        # Find game numbers where seasons change
        season_boundaries = []
        for manager in data["manager"].unique():
            manager_data = data[data["manager"] == manager].copy()
            manager_data["year_change"] = manager_data["year"].diff().fillna(0) != 0
            boundaries = manager_data[manager_data["year_change"]]["game_number"].tolist()
            season_boundaries.extend(boundaries)

        # Get unique boundaries (approximately)
        if season_boundaries:
            unique_boundaries = sorted(set(int(b) for b in season_boundaries))

            # Get years for labels
            years = sorted(data["year"].unique())

            for i, boundary in enumerate(unique_boundaries[:len(years)-1]):
                year_label = years[i + 1] if i + 1 < len(years) else ""
                fig.add_vline(
                    x=boundary,
                    line_dash="dot",
                    line_color="rgba(128, 128, 128, 0.15)",
                    line_width=1,
                    annotation_text=str(year_label),
                    annotation_position="top",
                    annotation_font_size=10,
                    annotation_font_color="rgba(200, 200, 200, 0.9)",
                    annotation_font_weight=500,
                )

    # Add manager traces
    for i, manager in enumerate(data["manager"].unique()):
        manager_data = data[data["manager"] == manager]

        fig.add_trace(
            go.Scatter(
                x=manager_data["game_number"],
                y=manager_data["display_value"],
                mode="lines",
                name=manager,
                line=dict(width=3, color=_get_chart_color(i)),
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Game: %{x}<br>"
                    f"{'Avg PPG' if normalize_by_games else 'Total'}: %{{y:,.{'2f' if normalize_by_games else '0f'}}}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Apply layout
    layout = _get_base_chart_layout(height=500)
    layout["xaxis_title"] = "Career Games Played"
    layout["yaxis_title"] = y_title
    fig.update_layout(**layout)

    # Mobile-friendly chart container
    st.markdown(
        '<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_cumulative_chart")
    st.markdown("</div>", unsafe_allow_html=True)

    # Career totals table
    st.markdown(
        "<p style='font-size: 0.9rem; font-weight: 600; margin: 1rem 0 0.5rem 0;'>Career Totals</p>",
        unsafe_allow_html=True,
    )

    career_stats = (
        data.groupby("manager").agg({"team_points": ["sum", "mean", "count"]}).round(2)
    )

    career_stats.columns = ["Total Points", "Avg PPG", "Games"]
    career_stats = career_stats.sort_values("Total Points", ascending=False)
    career_stats["Total Points"] = career_stats["Total Points"].astype(int)

    st.dataframe(career_stats, use_container_width=True)

    # Career milestones
    with st.expander("Career Milestones", expanded=False):
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


def _render_season_averages_view(data: pd.DataFrame, selected_managers: list, prefix: str):
    """Render season-by-season average scoring trends with enhanced features."""

    # Chart title with subtitle
    _render_chart_title(
        "Season Average Trends",
        "Compare scoring efficiency across seasons. Error bars show variability (std dev).",
    )

    # Options
    col1, col2, col3 = st.columns(3)

    with col1:
        show_error_bars = st.checkbox(
            "Show Error Bars",
            key=f"{prefix}_error_bars",
            value=True,
            help="Display standard deviation as error bars",
        )

    with col2:
        compare_mode = st.checkbox(
            "Compare Two Managers",
            key=f"{prefix}_compare_mode",
            help="Highlight two managers, fade others",
        )

    with col3:
        show_rolling_avg = st.checkbox(
            "3-Year Rolling Avg",
            key=f"{prefix}_rolling_avg",
            help="Show smoothed 3-year rolling average",
        )

    # Compare mode selection
    compare_managers = []
    if compare_mode:
        compare_managers = st.multiselect(
            "Select two managers to compare",
            options=selected_managers,
            max_selections=2,
            key=f"{prefix}_compare_select",
        )

    # Create figure
    fig = go.Figure()

    # Add alternating season background highlights (subtle 3% opacity)
    years = sorted(data["year"].unique())
    for i, year in enumerate(years):
        if i % 2 == 0:
            fig.add_vrect(
                x0=year - 0.5,
                x1=year + 0.5,
                fillcolor="rgba(128, 128, 128, 0.03)",
                line_width=0,
            )

    # Calculate rolling average if needed
    if show_rolling_avg:
        for manager in data["manager"].unique():
            manager_data = data[data["manager"] == manager].sort_values("year").copy()
            if len(manager_data) >= 3:
                manager_data["rolling_avg"] = manager_data["avg_points"].rolling(3, min_periods=1).mean()

                # Determine opacity for compare mode
                if compare_mode and compare_managers:
                    opacity = 0.8 if manager in compare_managers else 0.15
                else:
                    opacity = 0.6

                idx = selected_managers.index(manager) if manager in selected_managers else 0

                fig.add_trace(
                    go.Scatter(
                        x=manager_data["year"],
                        y=manager_data["rolling_avg"],
                        mode="lines",
                        name=f"{manager} (3yr avg)",
                        line=dict(width=2, dash="dot", color=_get_chart_color(idx)),
                        opacity=opacity,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{manager} 3-Year Avg</b><br>"
                            "Year: %{x}<br>"
                            "Rolling Avg: %{y:.2f}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

    # Add manager traces with trend arrows
    trend_info = []
    compare_annotations = []

    for i, manager in enumerate(data["manager"].unique()):
        manager_data = data[data["manager"] == manager].sort_values("year")

        # Determine opacity for compare mode (more dramatic: 15% for non-compared)
        if compare_mode and compare_managers:
            is_compared = manager in compare_managers
            opacity = 1.0 if is_compared else 0.15
            line_width = 4 if is_compared else 1.2
            marker_size = 12 if is_compared else 6
            # Store annotation info for compared managers
            if is_compared and len(manager_data) > 0:
                last_year = manager_data.iloc[-1]["year"]
                last_avg = manager_data.iloc[-1]["avg_points"]
                compare_annotations.append((last_year, last_avg, manager, _get_chart_color(i)))
        else:
            opacity = 1.0
            line_width = 2.5
            marker_size = 10

        color = _get_chart_color(i)

        # Error bar configuration
        error_y_config = None
        if show_error_bars:
            error_y_config = dict(
                type="data",
                array=manager_data["std_dev"],
                visible=True,
                thickness=1.5,
                width=4,
                color=color,
            )

        fig.add_trace(
            go.Scatter(
                x=manager_data["year"],
                y=manager_data["avg_points"],
                mode="lines+markers",
                name=manager,
                line=dict(width=line_width, color=color),
                marker=dict(size=marker_size),
                opacity=opacity,
                error_y=error_y_config,
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Year: %{x}<br>"
                    "Avg: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Calculate trend for this manager
        if len(manager_data) >= 3:
            recent_3 = manager_data.tail(3)["avg_points"].values
            if len(recent_3) == 3:
                trend_slope = (recent_3[-1] - recent_3[0]) / 2
                if trend_slope > 2:
                    trend = ("up", recent_3[-1])
                elif trend_slope < -2:
                    trend = ("down", recent_3[-1])
                else:
                    trend = ("stable", recent_3[-1])
                trend_info.append((manager, trend[0], trend[1], color))

    # Add annotations for compared managers at rightmost point
    if compare_mode and compare_annotations:
        for year, avg, name, color in compare_annotations:
            fig.add_annotation(
                x=year + 0.3,
                y=avg,
                text=f"<b>{name}</b><br>{avg:.1f}",
                showarrow=False,
                font=dict(size=10, color=color),
                align="left",
                xanchor="left",
            )

    # Apply layout
    layout = _get_base_chart_layout(height=500)
    layout["xaxis_title"] = "Season"
    layout["yaxis_title"] = "Average Points Per Game"
    fig.update_layout(**layout)
    fig.update_xaxes(tickmode="linear", dtick=1)

    # Mobile-friendly chart container
    st.markdown(
        '<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_season_avg_chart")
    st.markdown("</div>", unsafe_allow_html=True)

    # Trend indicators
    if trend_info:
        st.markdown(
            "<p style='font-size: 0.8rem; font-weight: 600; margin: 1rem 0 0.5rem 0;'>3-Year Trends</p>",
            unsafe_allow_html=True,
        )

        trend_cols = st.columns(min(len(trend_info), 6))
        trend_icons = {"up": "trending_up", "down": "trending_down", "stable": "trending_flat"}
        trend_arrows = {"up": "^", "down": "v", "stable": "-"}

        for i, (manager, trend, latest, color) in enumerate(trend_info[:6]):
            with trend_cols[i]:
                arrow = trend_arrows[trend]
                arrow_color = "#10B981" if trend == "up" else "#EF4444" if trend == "down" else "#6B7280"

                st.markdown(
                    f"""
                    <div style="
                        background: var(--bg-secondary);
                        border: 1px solid var(--border);
                        border-radius: 6px;
                        padding: 0.5rem;
                        text-align: center;
                    ">
                        <div style="font-weight: 600; font-size: 0.75rem; color: {color};">
                            {manager}
                        </div>
                        <div style="font-size: 1.2rem; color: {arrow_color}; font-weight: bold;">
                            {arrow}
                        </div>
                        <div style="font-size: 0.65rem; color: var(--text-muted);">
                            {latest:.1f} PPG
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Season statistics table (collapsible)
    with st.expander("Season Statistics", expanded=False):
        pivot = data.pivot(index="year", columns="manager", values="avg_points")
        pivot = pivot.round(2)
        st.dataframe(pivot, use_container_width=True)

    # Trend analysis (collapsible)
    with st.expander("Trend Analysis", expanded=False):
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
                        "Improving"
                        if slope > 1
                        else "Declining" if slope < -1 else "Stable"
                    )
                else:
                    trend = "Stable"

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
