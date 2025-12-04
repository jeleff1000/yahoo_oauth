#!/usr/bin/env python3
"""
yearly_playoff_graph.py - Year-over-year playoff trends showing ebbs and flows
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

METRIC_LABELS = {
    "p_playoffs": "Playoff Odds",
    "p_bye": "First Round Bye",
    "exp_final_wins": "Expected Final Wins",
    "p_semis": "Semifinal Odds",
    "p_final": "Championship Game",
    "p_champ": "Championship Win",
}


class PlayoffOddsCumulativeViewer:
    def __init__(self, matchup_data_df):
        self.df = matchup_data_df.copy()

    @st.fragment
    def display(self, year: int = None, week: int = None):
        """Display multi-season playoff trends.

        Args:
            year: Pre-selected year (from unified header)
            week: Pre-selected week (from unified header)
        """
        st.subheader("Multi-Season Playoff Trends")

        st.caption("Track how managers' playoff fortunes changed year-over-year")

        df = self.df.copy()
        if df.empty:
            st.info("No data available.")
            return

        if "week" not in df.columns or "year" not in df.columns:
            st.error("Required columns 'week' and 'year' not found.")
            return

        # Clean data
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df = df.dropna(subset=["year", "week"]).copy()
        df["year"] = df["year"].astype(int)
        df["week"] = df["week"].astype(int)

        years = sorted(df["year"].unique())
        if not years:
            st.info("No years available.")
            return

        # Filters - mobile friendly stacked layout
        metric = st.selectbox(
            "📊 Metric to Track",
            list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS[k],
            key="metric_select_cum",
            help="Choose which playoff metric to compare year-over-year",
        )

        # Session state buttons instead of radio
        trend_key = "trend_type_cum"
        if trend_key not in st.session_state:
            st.session_state[trend_key] = 0

        trend_types = ["Season Average", "Peak Value"]
        trend_cols = st.columns(2)
        for idx, (col, name) in enumerate(zip(trend_cols, trend_types)):
            with col:
                is_active = st.session_state[trend_key] == idx
                if st.button(
                    name,
                    key=f"trend_btn_{idx}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        st.session_state[trend_key] = idx
                        st.rerun()

        trend_type = trend_types[st.session_state[trend_key]]

        # Manager selection
        managers = sorted(df["manager"].unique())

        # Handle "Select All" button
        if "cum_select_all_clicked" not in st.session_state:
            st.session_state.cum_select_all_clicked = False

        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("All", key="select_all_cum", use_container_width=True):
                st.session_state.cum_select_all_clicked = True
                st.rerun()

        with col1:
            default_mgrs = managers if st.session_state.cum_select_all_clicked else []
            selected_mgrs = st.multiselect(
                "Managers",
                managers,
                default=default_mgrs,
                key="select_cum",
                label_visibility="collapsed",
                placeholder="Select managers (empty = all)",
            )

        if st.session_state.cum_select_all_clicked:
            st.session_state.cum_select_all_clicked = False

        effective_mgrs = managers if len(selected_mgrs) == 0 else selected_mgrs

        # Create visualization
        self._create_year_over_year_chart(df, metric, effective_mgrs, years, trend_type)

        # Stats table
        with st.expander("📊 Year-over-Year Stats", expanded=False):
            self._display_yoy_stats(df, metric, effective_mgrs, years)

    def _create_year_over_year_chart(self, data, metric, managers, years, trend_type):
        """Create year-over-year line chart showing ebbs and flows"""

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

        for i, manager in enumerate(managers):
            mgr_data = data[data["manager"] == manager]

            year_values = []
            for year in years:
                year_data = mgr_data[mgr_data["year"] == year]

                if year_data.empty:
                    year_values.append(None)
                    continue

                if trend_type == "Season Average":
                    value = year_data[metric].mean()
                elif trend_type == "Peak Value":
                    value = year_data[metric].max()
                else:
                    value = None

                year_values.append(value)

            # Plot the trend
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=year_values,
                    mode="lines+markers",
                    name=manager,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=10),
                    hovertemplate=(
                        f"<b>{manager}</b><br>"
                        + "Year: %{x}<br>"
                        + f"{METRIC_LABELS[metric]}: %{{y:.1f}}"
                        + ("%" if metric != "exp_final_wins" else "")
                        + "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"{METRIC_LABELS[metric]} - {trend_type} by Year",
            xaxis_title="Season",
            yaxis_title=METRIC_LABELS[metric]
            + (" (%)" if metric != "exp_final_wins" else ""),
            hovermode="x unified",
            height=500,
            xaxis=dict(showgrid=True, dtick=1),
            yaxis=dict(showgrid=True),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig, use_container_width=True, key=f"yoy_{metric}_{trend_type}")

    @st.fragment
    def _display_yoy_stats(self, data, metric, managers, years):
        """Display year-over-year statistics table"""

        stats_list = []

        for manager in managers:
            mgr_data = data[data["manager"] == manager]

            for year in years:
                year_data = mgr_data[mgr_data["year"] == year]

                if year_data.empty:
                    continue

                avg_value = year_data[metric].mean()
                peak_value = year_data[metric].max()

                stats_list.append(
                    {
                        "Manager": manager,
                        "Year": year,
                        "Average": avg_value,
                        "Peak": peak_value,
                    }
                )

        if not stats_list:
            st.info("No data available")
            return

        stats_df = pd.DataFrame(stats_list)

        # Pivot to show years as columns
        pivot_avg = stats_df.pivot(index="Manager", columns="Year", values="Average")
        pivot_peak = stats_df.pivot(index="Manager", columns="Year", values="Peak")

        # Calculate year-over-year changes for average
        year_list = sorted(years)
        for i in range(1, len(year_list)):
            prev_year = year_list[i - 1]
            curr_year = year_list[i]
            if prev_year in pivot_avg.columns and curr_year in pivot_avg.columns:
                pivot_avg[f"Δ{curr_year}"] = (
                    pivot_avg[curr_year] - pivot_avg[prev_year]
                ).round(2)

        # Mobile-friendly stacked layout
        st.markdown("**📊 Season Averages by Year**")
        st.dataframe(pivot_avg.round(2), use_container_width=True)

        st.markdown("**🎯 Peak Values by Year**")
        st.dataframe(pivot_peak.round(2), use_container_width=True)

        # Biggest improvers/decliners - based on average
        change_cols = [col for col in pivot_avg.columns if "Δ" in str(col)]
        if change_cols:
            st.markdown("**📈 Biggest Year-over-Year Changes (Average)**")
            for col in change_cols:
                # Skip if all values are NaN
                if pivot_avg[col].isna().all():
                    continue

                best = pivot_avg[col].idxmax()
                worst = pivot_avg[col].idxmin()

                # Skip if indices are NaN
                if pd.isna(best) or pd.isna(worst):
                    continue

                best_val = pivot_avg.loc[best, col]
                worst_val = pivot_avg.loc[worst, col]
                suffix = "%" if metric != "exp_final_wins" else " wins"

                # Stacked for mobile
                st.caption(f"{col} ⬆️ **{best}** {best_val:+.1f}{suffix}")
                st.caption(f"{col} ⬇️ **{worst}** {worst_val:+.1f}{suffix}")
