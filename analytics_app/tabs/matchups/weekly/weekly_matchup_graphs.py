import streamlit as st
import altair as alt


class SeasonMatchupStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display_weekly_graphs(self, prefix=""):
        st.header("Team Points by Week")
        required_cols = {
            "Manager",
            "week",
            "team_points",
            "year",
            "opponent",
            "Final Playoff Seed",
        }
        if (
            self.df is not None
            and required_cols.issubset(self.df.columns)
            and not self.df.empty
        ):
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                managers = sorted(self.df["Manager"].unique().tolist())
                selected_managers = st.multiselect(
                    "Select Manager(s)",
                    managers,
                    default=[],
                    key=f"{prefix}_managers_graph",
                )
                if not selected_managers:
                    selected_managers = managers
            with col2:
                opponents = sorted(self.df["opponent"].unique().tolist())
                selected_opponents = st.multiselect(
                    "Select Opponent(s)",
                    opponents,
                    default=[],
                    key=f"{prefix}_opponents_graph",
                )
                if not selected_opponents:
                    selected_opponents = opponents
            with col3:
                years = sorted(self.df["year"].astype(int).unique().tolist())
                selected_years = st.multiselect(
                    "Select Year(s)", years, default=[], key=f"{prefix}_years_graph"
                )
                if not selected_years:
                    selected_years = years
            with col4:
                seeds = sorted(self.df["Final Playoff Seed"].dropna().unique().tolist())
                selected_seeds = st.multiselect(
                    "Select Final Playoff Seed(s)",
                    seeds,
                    default=[],
                    key=f"{prefix}_seeds_graph",
                )
                if not selected_seeds:
                    selected_seeds = seeds

            # Filter the DataFrame
            df_filtered = self.df[
                self.df["Manager"].isin(selected_managers)
                & self.df["opponent"].isin(selected_opponents)
                & self.df["year"].isin(selected_years)
                & self.df["Final Playoff Seed"].isin(selected_seeds)
            ]

            if df_filtered.empty:
                st.write("No data available for selected filters.")
                return

            x_axis_week = st.toggle("Show Season Week on X-axis", value=False)
            x_field = "week" if x_axis_week else "Cumulative Week"

            df_sorted = df_filtered.sort_values(["year", "week"])
            df_sorted["team_points"] = df_sorted["team_points"].round(2)

            if not x_axis_week:
                x_axis = alt.Axis(title="Cumulative Week")
            else:
                min_week = int(df_sorted["week"].min())
                max_week = int(df_sorted["week"].max())
                x_axis = alt.Axis(
                    format="d", values=list(range(min_week, max_week + 1)), title="week"
                )

            x_type = "Q" if x_axis_week else "O"

            df_sorted = df_sorted.sort_values(x_field)
            points = (
                alt.Chart(df_sorted)
                .mark_point(size=60)
                .encode(
                    x=alt.X(f"{x_field}:{x_type}", axis=x_axis),
                    y=alt.Y("team_points:Q", scale=alt.Scale(domain=[50, 220])),
                    color="Manager:N",
                )
            )

            if x_axis_week:
                # Default (all data) average
                week_avg_all = (
                    self.df.groupby("week")["team_points"].mean().reset_index()
                )
                week_avg_all["team_points"] = week_avg_all["team_points"].round(2)
                avg_line_all = (
                    alt.Chart(week_avg_all)
                    .mark_line(strokeDash=[5, 5], size=5, color="black")
                    .encode(
                        x=alt.X("week:Q", axis=x_axis),
                        y=alt.Y("team_points:Q", scale=alt.Scale(domain=[50, 220])),
                    )
                )
                # Filtered average
                week_avg_filtered = (
                    df_sorted.groupby("week")["team_points"].mean().reset_index()
                )
                week_avg_filtered["team_points"] = week_avg_filtered[
                    "team_points"
                ].round(2)
                avg_line_filtered = (
                    alt.Chart(week_avg_filtered)
                    .mark_line(strokeDash=[5, 5], size=5, color="red")
                    .encode(
                        x=alt.X("week:Q", axis=x_axis),
                        y=alt.Y("team_points:Q", scale=alt.Scale(domain=[50, 220])),
                    )
                )
                chart = points + avg_line_all + avg_line_filtered
            else:
                # Default (all data) cumulative avg
                df_all = self.df.copy()
                df_all = df_all.sort_values(["year", "week"])
                df_all["League Cumulative Avg"] = (
                    df_all.groupby("year")["team_points"]
                    .expanding()
                    .mean()
                    .reset_index(level=0, drop=True)
                ).round(2)
                league_avg_all = (
                    df_all.groupby(["year", "Cumulative Week"])
                    .agg({"League Cumulative Avg": "last"})
                    .reset_index()
                )
                cumulative_line_all = (
                    alt.Chart(league_avg_all)
                    .mark_line(strokeDash=[5, 5], size=5, color="black")
                    .encode(
                        x=alt.X("Cumulative Week:O", axis=x_axis),
                        y=alt.Y(
                            "League Cumulative Avg:Q", scale=alt.Scale(domain=[50, 220])
                        ),
                        detail="year:N",
                    )
                )
                # Filtered cumulative avg
                df_sorted["League Cumulative Avg"] = (
                    df_sorted.groupby("year")["team_points"]
                    .expanding()
                    .mean()
                    .reset_index(level=0, drop=True)
                ).round(2)
                league_avg_filtered = (
                    df_sorted.groupby(["year", "Cumulative Week"])
                    .agg({"League Cumulative Avg": "last"})
                    .reset_index()
                )
                cumulative_line_filtered = (
                    alt.Chart(league_avg_filtered)
                    .mark_line(strokeDash=[5, 5], size=5, color="red")
                    .encode(
                        x=alt.X("Cumulative Week:O", axis=x_axis),
                        y=alt.Y(
                            "League Cumulative Avg:Q", scale=alt.Scale(domain=[50, 220])
                        ),
                        detail="year:N",
                    )
                )
                year_boundaries = (
                    df_sorted.groupby("year")["Cumulative Week"].min().reset_index()
                )
                year_rules = (
                    alt.Chart(year_boundaries)
                    .mark_rule(color="gray", strokeDash=[2, 2])
                    .encode(x=alt.X("Cumulative Week:O", axis=x_axis))
                )
                year_labels = (
                    alt.Chart(year_boundaries)
                    .mark_text(dy=-225, fontSize=12, font="sans-serif", color="black")
                    .encode(
                        x=alt.X("Cumulative Week:O"), y=alt.value(220), text="year:N"
                    )
                )
                chart = (
                    points
                    + cumulative_line_all
                    + cumulative_line_filtered
                    + year_rules
                    + year_labels
                )

            chart = chart.properties(width=800, height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write(
                "Required columns (`Manager`, `week`, `team_points`, `year`, `opponent`, `Final Playoff Seed`) are missing or DataFrame is empty."
            )


@st.fragment
def display_weekly_graphs(df, prefix=""):
    viewer = SeasonMatchupStatsViewer(df)
    viewer.display_weekly_graphs(prefix=prefix)
