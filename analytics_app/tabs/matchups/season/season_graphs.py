import streamlit as st
import altair as alt


class SeasonMatchupStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        st.header("Cumulative Team Points by Week")
        required_cols = {"Manager", "week", "team_points"}
        if (
            self.df is not None
            and required_cols.issubset(self.df.columns)
            and not self.df.empty
        ):
            df_sorted = self.df.sort_values(["Manager", "week"])
            df_sorted["Cumulative Team Points"] = df_sorted.groupby("Manager")[
                "team_points"
            ].cumsum()
            df_sorted = df_sorted.rename(columns={"week": "Cumulative Week"})
            chart = (
                alt.Chart(df_sorted)
                .mark_line()
                .encode(
                    x="Cumulative Week:Q",
                    y="Cumulative Team Points:Q",
                    color="Manager:N",
                )
                .properties(width=800, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write(
                "Required columns ('Manager', 'week', 'team_points') are missing or DataFrame is empty."
            )


@st.fragment
def display_season_graphs(df, prefix=""):
    viewer = SeasonMatchupStatsViewer(df)
    viewer.display(prefix=prefix)
