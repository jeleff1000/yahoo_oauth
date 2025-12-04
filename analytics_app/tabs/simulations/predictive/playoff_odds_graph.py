#!/usr/bin/env python3
"""
playoff_odds_graph.py - Enhanced weekly playoff odds visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

METRIC_LABELS = {
    "p_playoffs": "Playoff Odds",
    "p_bye": "First Round Bye",
    "exp_final_wins": "Expected Final Wins",
    "p_semis": "Semifinal Odds",
    "p_final": "Championship Game",
    "p_champ": "Championship Win",
}

METRIC_DESCRIPTIONS = {
    "p_playoffs": "Probability of making the playoffs",
    "p_bye": "Probability of earning a first-round bye",
    "exp_final_wins": "Projected final win total",
    "p_semis": "Probability of reaching the semifinals",
    "p_final": "Probability of reaching the championship game",
    "p_champ": "Probability of winning the championship",
}


class PlayoffOddsViewer:
    def __init__(self, matchup_data_df):
        self.df = matchup_data_df.copy()

    @st.fragment
    def display(self, year: int = None, week: int = None):
        """Display weekly playoff odds tracker.

        Args:
            year: Pre-selected year (from unified header)
            week: Pre-selected week (from unified header)
        """
        st.subheader("Weekly Playoff Odds Tracker")

        st.info("Track how playoff odds evolved week-by-week throughout the season")

        df = self.df[self.df["is_consolation"] == 0].copy()

        # Clean data
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df = df.dropna(subset=["year", "week"])

        if df.empty:
            st.info("No valid playoff odds data available.")
            return

        df["year"] = df["year"].astype(int)
        df["week"] = df["week"].astype(int)

        seasons = sorted(df["year"].unique())
        if not seasons:
            st.info("No seasons available.")
            return

        _min_year, max_year = int(seasons[0]), int(seasons[-1])
        min_week, max_week = int(df["week"].min()), int(df["week"].max())

        # Controls
        st.markdown("### ⚙️ Filter Options")

        # Quick filter dropdown
        quick_preset = st.selectbox(
            "Quick Preset",
            ["Current Season", "Last 3 Years", "Last 5 Years", "All Time", "Custom"],
            index=4,
            key="quick_preset_odds",
        )

        # Determine default years based on preset
        if quick_preset == "Current Season":
            default_years = [max_year]
        elif quick_preset == "Last 3 Years":
            default_years = [y for y in seasons if y >= max_year - 2]
        elif quick_preset == "Last 5 Years":
            default_years = [y for y in seasons if y >= max_year - 4]
        elif quick_preset == "All Time":
            default_years = seasons
        else:  # Custom
            default_years = [max_year]

        # Year selection
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_years = st.multiselect(
                "Select Seasons",
                options=seasons,
                default=default_years,
                key="year_multiselect",
                help="Select one or more seasons to analyze",
                disabled=(quick_preset != "Custom"),
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            show_champions = st.checkbox("🏆 Champs", value=False, key="champions_only")

        # Week range selection
        col1, col2 = st.columns(2)

        with col1:
            start_week = st.selectbox(
                "Start Week",
                options=list(range(min_week, max_week + 1)),
                index=0,
                key="start_week_select",
            )

        with col2:
            end_week = st.selectbox(
                "End Week",
                options=list(range(min_week, max_week + 1)),
                index=max_week - min_week,
                key="end_week_select",
            )

        if not selected_years:
            st.info("Please select at least one season.")
            return

        # Filter data
        filtered_df = df[
            (df["year"].isin(selected_years))
            & (df["week"] >= start_week)
            & (df["week"] <= end_week)
        ].copy()

        if show_champions:
            filtered_df = filtered_df[filtered_df["champion"] == 1]

        if filtered_df.empty:
            st.warning("No data matches your filters.")
            return

        # Manager selection
        managers = sorted(filtered_df["manager"].unique())

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_mgrs = st.multiselect(
                "Select Managers (leave empty for all)",
                managers,
                default=[],
                key="manager_select",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Select All", key="select_all_mgrs"):
                selected_mgrs = managers

        effective_mgrs = managers if len(selected_mgrs) == 0 else selected_mgrs
        plot_df = filtered_df[filtered_df["manager"].isin(effective_mgrs)].copy()

        if plot_df.empty:
            st.info("No data for selected managers.")
            return

        # Metric selection with session state buttons
        st.markdown("---")
        st.markdown("### 📈 Choose Metric to Display")

        metric_key = "metric_select"
        if metric_key not in st.session_state:
            st.session_state[metric_key] = 0

        metric_keys = list(METRIC_LABELS.keys())
        metric_cols = st.columns(len(metric_keys))
        for idx, (col, key) in enumerate(zip(metric_cols, metric_keys)):
            with col:
                is_active = st.session_state[metric_key] == idx
                if st.button(
                    METRIC_LABELS[key],
                    key=f"metric_btn_{idx}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        st.session_state[metric_key] = idx
                        st.rerun()

        selected_metric = metric_keys[st.session_state[metric_key]]

        # Create visualization
        self._create_enhanced_chart(plot_df, selected_metric, selected_years)

        # Summary statistics
        st.markdown("---")
        self._display_summary_stats(plot_df, selected_metric)

    def _create_enhanced_chart(self, data, metric, selected_years):
        """Create enhanced line chart."""
        data = data.sort_values(["manager", "year", "week"])

        fig = go.Figure()

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        managers = sorted(data["manager"].unique())

        # Check if data needs scaling (0-1 to 0-100)
        is_percent_metric = metric != "exp_final_wins"
        needs_scaling = False
        if is_percent_metric and not data.empty:
            sample = data[metric].dropna()
            if not sample.empty and sample.max() <= 1.01:
                needs_scaling = True

        for i, manager in enumerate(managers):
            mgr_data = data[data["manager"] == manager].sort_values(["year", "week"])

            for year in mgr_data["year"].unique():
                year_data = mgr_data[mgr_data["year"] == year]

                y_values = year_data[metric].values
                if is_percent_metric and needs_scaling:
                    y_values = y_values * 100

                fig.add_trace(
                    go.Scatter(
                        x=year_data["week"],
                        y=y_values,
                        mode="lines+markers",
                        name=manager,
                        legendgroup=manager,
                        showlegend=bool(year == mgr_data["year"].min()),
                        line=dict(width=3, color=colors[i % len(colors)]),
                        marker=dict(size=8),
                        hovertemplate=(
                            f"<b>{manager}</b><br>"
                            + f"Year: {year}<br>"
                            + "Week: %{x}<br>"
                            + f"{METRIC_LABELS[metric]}: %{{y:.1f}}"
                            + ("%" if is_percent_metric else "")
                            + "<extra></extra>"
                        ),
                    )
                )

        # Layout - create title based on selected years
        if len(selected_years) == 1:
            title = f"{METRIC_LABELS[metric]} - {selected_years[0]} Season"
        elif len(selected_years) <= 3:
            years_str = ", ".join(str(y) for y in sorted(selected_years))
            title = f"{METRIC_LABELS[metric]} - Seasons {years_str}"
        else:
            min_year = min(selected_years)
            max_year = max(selected_years)
            title = f"{METRIC_LABELS[metric]} - {len(selected_years)} Seasons ({min_year}-{max_year})"

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18}},
            xaxis_title="Week",
            yaxis_title=METRIC_LABELS[metric] + (" (%)" if is_percent_metric else ""),
            hovermode="x unified",
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            font=dict(size=12),
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=True, dtick=1),
            yaxis=dict(showgrid=True),
        )

        # Reference lines
        if metric == "p_playoffs" and needs_scaling:
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="orange",
                annotation_text="50% - Coin Flip",
            )
        elif metric == "p_champ" and needs_scaling:
            fig.add_hline(
                y=10,
                line_dash="dash",
                line_color="red",
                annotation_text="10% - Long Shot",
            )

        st.plotly_chart(fig, use_container_width=True)

    @st.fragment
    def _display_summary_stats(self, data, metric):
        """Display summary statistics."""
        st.markdown("### 📊 Summary Statistics")

        # Calculate stats
        manager_stats = (
            data.groupby("manager")
            .agg({metric: ["mean", "max", "min", "std"], "week": "count"})
            .round(4)
        )

        manager_stats.columns = ["Average", "Peak", "Low", "Volatility", "Weeks"]

        # Scale if needed
        is_percent_metric = metric != "exp_final_wins"
        needs_scaling = False
        if is_percent_metric and not data.empty:
            sample = data[metric].dropna()
            if not sample.empty and sample.max() <= 1.01:
                needs_scaling = True

        if is_percent_metric and needs_scaling:
            for col in ["Average", "Peak", "Low", "Volatility"]:
                manager_stats[col] = manager_stats[col] * 100

        manager_stats = manager_stats.sort_values("Average", ascending=False)

        # Display metrics - responsive layout
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            best_avg = manager_stats["Average"].idxmax()
            best_val = manager_stats.loc[best_avg, "Average"]
            suffix = "%" if is_percent_metric else " wins"
            st.metric("Highest Average", best_avg, f"{best_val:.1f}{suffix}")

        with col2:
            best_peak = manager_stats["Peak"].idxmax()
            peak_val = manager_stats.loc[best_peak, "Peak"]
            st.metric("Best Peak", best_peak, f"{peak_val:.1f}{suffix}")

        with col3:
            most_volatile = manager_stats["Volatility"].idxmax()
            volatile_val = manager_stats.loc[most_volatile, "Volatility"]
            st.metric("Most Volatile", most_volatile, f"±{volatile_val:.1f}{suffix}")

        with col4:
            league_avg = manager_stats["Average"].mean()
            st.metric("League Average", f"{league_avg:.1f}{suffix}")

        # Full table
        st.markdown("**Detailed Statistics**")

        styled_df = manager_stats.copy()
        if is_percent_metric:
            for col in ["Average", "Peak", "Low", "Volatility"]:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:.1f}%")
        else:
            for col in ["Average", "Peak", "Low", "Volatility"]:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}")

        st.dataframe(styled_df, use_container_width=True)
