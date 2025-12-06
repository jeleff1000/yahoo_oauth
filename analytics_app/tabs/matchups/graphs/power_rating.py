#!/usr/bin/env python3
"""
Power Rating Dashboard with premium UI.

Improvements:
- Tier zones (Elite, Competitive, Average, Struggling) using percentiles
- Spotlight mode to highlight single manager
- Smoothing toggle (rolling average)
- Week-over-week arrows in current ratings
- Better styling and interpretation helpers
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from md.core import T, run_query

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


def _render_manager_selector(managers: list, key: str, default_count: int = 5) -> list:
    """Manager selector in card with Select All/Clear All."""
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
                font-size: 0.85rem; font-weight: 600;
                color: var(--text-primary, #F9FAFB);
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-subtle, #2D3748);
            ">
                Managers Selected
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


@st.fragment
def display_power_rating_graph(df_dict=None, prefix="graphs_manager_power_rating"):
    """Display power rating trends with premium UI."""

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
                Power Rating Dashboard
            </h2>
            <p style="margin: 0; color: var(--text-secondary, #9CA3AF); font-size: 0.85rem; line-height: 1.5;">
                Power ratings measure team strength based on performance.
                <strong>Higher = stronger</strong>. Ratings adjust after each game based on results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading power rating data..."):
        query = f"""
            SELECT year, week, manager, power_rating, team_points,
                CASE WHEN team_points > opponent_points THEN 1 ELSE 0 END as win
            FROM {T['matchup']}
            WHERE manager IS NOT NULL AND power_rating IS NOT NULL
            ORDER BY year, week, manager
        """
        df = run_query(query)
        if df is None or df.empty:
            st.warning("No power rating data found.")
            return

    # Create cumulative week column
    df = df.sort_values(["year", "week"])
    year_week_map = df[["year", "week"]].drop_duplicates().sort_values(["year", "week"]).reset_index(drop=True)
    year_week_map["cumulative_week"] = range(1, len(year_week_map) + 1)
    df = df.merge(year_week_map, on=["year", "week"], how="left")

    available_years = sorted(df["year"].unique())
    managers_available = sorted(df["manager"].unique())

    # Controls row
    col1, col2 = st.columns(2)
    with col1:
        year_range = st.select_slider(
            "Year Range",
            options=available_years,
            value=(available_years[0], available_years[-1]),
            key=f"{prefix}_year_range",
        )
    start_year, end_year = year_range

    with col2:
        # Chart options
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            show_smoothed = st.checkbox("Smooth Lines", key=f"{prefix}_smooth", help="5-week rolling average")
        with opt_col2:
            show_zones = st.checkbox("Show Tiers", value=True, key=f"{prefix}_zones", help="Elite/Competitive/Average/Struggling zones")

    # Manager selection
    selected_managers = _render_manager_selector(managers_available, f"{prefix}_managers", default_count=5)

    if not selected_managers:
        st.info("Please select at least one manager.")
        return

    # Spotlight mode
    spotlight_manager = st.selectbox(
        "Spotlight Manager (others fade)",
        options=["None"] + selected_managers,
        key=f"{prefix}_spotlight",
        help="Highlight one manager, fade others to 20%",
    )

    # Filter data
    df_filtered = df[
        (df["year"] >= start_year) & (df["year"] <= end_year) & (df["manager"].isin(selected_managers))
    ].copy()

    if df_filtered.empty:
        st.info("No data after applying filters.")
        return

    # --- POWER RATING TREND CHART ---
    _render_section_header(
        "Power Rating Over Time",
        "Track how team strength evolves. Vertical lines mark season boundaries.",
    )

    fig = go.Figure()

    # Calculate percentile-based tier zones
    if show_zones:
        all_ratings = df_filtered["power_rating"]
        p25 = all_ratings.quantile(0.25)
        p50 = all_ratings.quantile(0.50)
        p75 = all_ratings.quantile(0.75)
        p90 = all_ratings.quantile(0.90)

        # Add tier zones as horizontal bands
        y_min = all_ratings.min() - 5
        y_max = all_ratings.max() + 5

        zones = [
            (p90, y_max, "rgba(16, 185, 129, 0.08)", "Elite"),
            (p75, p90, "rgba(59, 130, 246, 0.06)", "Competitive"),
            (p50, p75, "rgba(156, 163, 175, 0.05)", "Average"),
            (y_min, p50, "rgba(239, 68, 68, 0.05)", "Struggling"),
        ]

        for y0, y1, color, label in zones:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)

        # Add tier labels on right
        fig.add_annotation(x=1.02, y=p90, text="Elite", xref="paper", yref="y",
                          showarrow=False, font=dict(size=9, color="rgba(16,185,129,0.8)"), xanchor="left")
        fig.add_annotation(x=1.02, y=p75, text="Competitive", xref="paper", yref="y",
                          showarrow=False, font=dict(size=9, color="rgba(59,130,246,0.8)"), xanchor="left")
        fig.add_annotation(x=1.02, y=p50, text="Average", xref="paper", yref="y",
                          showarrow=False, font=dict(size=9, color="rgba(156,163,175,0.8)"), xanchor="left")

    # Add manager traces
    for i, manager in enumerate(selected_managers):
        manager_data = df_filtered[df_filtered["manager"] == manager].sort_values("cumulative_week")
        if manager_data.empty:
            continue

        color = _get_chart_color(i)

        # Determine opacity based on spotlight mode
        if spotlight_manager != "None":
            opacity = 1.0 if manager == spotlight_manager else 0.2
            line_width = 3.5 if manager == spotlight_manager else 1.5
        else:
            opacity = 1.0
            line_width = 2.5

        y_values = manager_data["power_rating"]
        if show_smoothed:
            y_values = manager_data["power_rating"].rolling(5, min_periods=1).mean()

        fig.add_trace(
            go.Scatter(
                x=manager_data["cumulative_week"],
                y=y_values,
                mode="lines",
                name=manager,
                line=dict(width=line_width, color=color),
                opacity=opacity,
                hovertemplate=(
                    f"<b>{manager}</b><br>"
                    "Year: %{customdata[0]} Week: %{customdata[1]}<br>"
                    "Power Rating: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                customdata=manager_data[["year", "week"]].values,
            )
        )

    # Add year boundary markers
    year_boundaries = df_filtered.groupby("year")["cumulative_week"].min().reset_index()
    for _, row in year_boundaries.iterrows():
        fig.add_vline(
            x=row["cumulative_week"],
            line_dash="dot",
            line_color="rgba(128,128,128,0.2)",
            line_width=1,
            annotation_text=str(int(row["year"])),
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color="rgba(200,200,200,0.7)",
        )

    fig.update_layout(
        xaxis_title="Cumulative Week",
        yaxis_title="Power Rating",
        hovermode="closest",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=0.99,
                   bgcolor="rgba(0,0,0,0.02)", bordercolor="rgba(128,128,128,0.2)",
                   borderwidth=1, font=dict(size=10)),
        margin=dict(l=50, r=80, t=30, b=50),
        xaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
    )

    st.markdown('<div style="overflow-x: auto; -webkit-overflow-scrolling: touch;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_trend")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- CURRENT RATINGS TABLE WITH ARROWS ---
    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        _render_section_header("Current Power Ratings", "Latest ratings with week-over-week change")

        # Get latest and previous ratings
        latest_ratings = df_filtered.sort_values("cumulative_week").groupby("manager").last().reset_index()

        # Get previous week ratings for comparison
        max_week = df_filtered["cumulative_week"].max()
        prev_ratings = df_filtered[df_filtered["cumulative_week"] == max_week - 1].set_index("manager")["power_rating"].to_dict()

        display_data = []
        for _, row in latest_ratings.iterrows():
            manager = row["manager"]
            current = row["power_rating"]
            prev = prev_ratings.get(manager, current)
            change = current - prev

            # Arrow indicator
            if change > 0.5:
                arrow = "^"
                arrow_color = "#10B981"
            elif change < -0.5:
                arrow = "v"
                arrow_color = "#EF4444"
            else:
                arrow = "-"
                arrow_color = "#6B7280"

            display_data.append({
                "Manager": manager,
                "Rating": f"{current:.2f}",
                "Change": f"{arrow} {change:+.1f}",
                "change_val": change,
            })

        display_df = pd.DataFrame(display_data).sort_values("Rating", ascending=False, key=lambda x: x.astype(float))

        # Render with custom styling
        for _, row in display_df.iterrows():
            change_val = row["change_val"]
            color = "#10B981" if change_val > 0.5 else "#EF4444" if change_val < -0.5 else "#6B7280"
            st.markdown(
                f"""
                <div style="
                    display: flex; justify-content: space-between; align-items: center;
                    padding: 0.5rem 0.75rem; margin-bottom: 0.25rem;
                    background: var(--card-bg, rgba(255,255,255,0.03));
                    border-radius: 6px; border: 1px solid var(--border-subtle, #2D3748);
                ">
                    <span style="font-weight: 500;">{row['Manager']}</span>
                    <span>
                        <span style="font-weight: 600;">{row['Rating']}</span>
                        <span style="color: {color}; margin-left: 0.5rem; font-size: 0.85rem;">{row['Change']}</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        _render_section_header("Power Rating Range", "Min, max, and average across selected period")

        rating_stats = df_filtered.groupby("manager")["power_rating"].agg(["min", "max", "mean", "std"]).round(2)
        rating_stats = rating_stats.sort_values("mean", ascending=False).reset_index()
        rating_stats.columns = ["Manager", "Min", "Max", "Avg", "Volatility"]

        st.dataframe(
            rating_stats,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Volatility": st.column_config.NumberColumn(format="%.1f", help="Standard deviation - lower = more stable"),
            },
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- CORRELATION SCATTER ---
    with st.expander("Power Rating vs Performance", expanded=False):
        correlation_data = df_filtered.groupby("manager").agg({
            "power_rating": "mean", "win": "mean", "team_points": "mean"
        }).reset_index()
        correlation_data["win_pct"] = (correlation_data["win"] * 100).round(1)
        correlation_data["avg_rating"] = correlation_data["power_rating"].round(2)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=correlation_data["avg_rating"],
                y=correlation_data["win_pct"],
                mode="markers+text",
                text=correlation_data["manager"],
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(
                    size=15,
                    color=correlation_data["win_pct"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Win %"),
                    line=dict(width=2, color="white"),
                ),
            )
        )

        # Add trendline
        if len(correlation_data) >= 2:
            z = np.polyfit(correlation_data["avg_rating"], correlation_data["win_pct"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(correlation_data["avg_rating"].min(), correlation_data["avg_rating"].max(), 50)
            fig_scatter.add_trace(
                go.Scatter(x=x_trend, y=p(x_trend), mode="lines",
                          line=dict(dash="dot", color="rgba(128,128,128,0.5)", width=2),
                          name="Trend", showlegend=False)
            )

        fig_scatter.update_layout(
            xaxis_title="Average Power Rating",
            yaxis_title="Win Percentage",
            height=400,
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(fig_scatter, use_container_width=True, key=f"{prefix}_correlation")

    # --- KEY INSIGHTS ---
    with st.expander("Key Insights", expanded=False):
        if len(latest_ratings) >= 1:
            strongest = latest_ratings.sort_values("power_rating", ascending=False).iloc[0]
            weakest = latest_ratings.sort_values("power_rating", ascending=True).iloc[0]

            # Calculate biggest improver
            if start_year != end_year:
                start_ratings = df_filtered[df_filtered["year"] == start_year].groupby("manager")["power_rating"].mean()
                end_ratings = df_filtered[df_filtered["year"] == end_year].groupby("manager")["power_rating"].mean()
                common = set(start_ratings.index) & set(end_ratings.index)

                if common:
                    improvements = {m: end_ratings[m] - start_ratings[m] for m in common}
                    best_improver = max(improvements, key=improvements.get)
                    improvement = improvements[best_improver]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strongest Team", strongest["manager"], f"{strongest['power_rating']:.2f}")
                    with col2:
                        st.metric("Most Improved", best_improver, f"+{improvement:.2f}")
                    with col3:
                        st.metric("Needs Work", weakest["manager"], f"{weakest['power_rating']:.2f}")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Strongest Team", strongest["manager"], f"{strongest['power_rating']:.2f}")
                    with col2:
                        st.metric("Needs Work", weakest["manager"], f"{weakest['power_rating']:.2f}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Strongest Team", strongest["manager"], f"{strongest['power_rating']:.2f}")
                with col2:
                    st.metric("Needs Work", weakest["manager"], f"{weakest['power_rating']:.2f}")
