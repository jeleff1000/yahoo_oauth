import streamlit as st
import pandas as pd


class TopWeeksViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self):
        st.header("Top Weeks")

        # Create columns for checkboxes
        col1, col2 = st.columns(2)
        with col1:
            regular_season = st.checkbox(
                "Regular Season", value=True, key="top_weeks_regular_season"
            )
        with col2:
            playoffs = st.checkbox("Playoffs", value=False, key="top_weeks_playoffs")

        if self.df is not None:
            # Create columns for dropdowns with narrower widths
            col3, col4 = st.columns([1, 1])
            with col3:
                years = ["All Years"] + sorted(self.df["year"].unique().tolist())
                selected_year = st.selectbox(
                    "Select Year", years, index=0, key="top_weeks_year"
                )
            with col4:
                managers = ["All Managers"] + sorted(
                    self.df["Manager"].unique().tolist()
                )
                selected_manager = st.selectbox(
                    "Select Manager", managers, index=0, key="top_weeks_manager"
                )

            # Filter data based on checkboxes
            if regular_season and playoffs:
                filtered_df = self.df[(self.df["is_consolation"] == 0)]
            elif regular_season:
                filtered_df = self.df[
                    (self.df["is_playoffs"] == 0) & (self.df["is_consolation"] == 0)
                ]
            elif playoffs:
                filtered_df = self.df[
                    (self.df["is_playoffs"] == 1) & (self.df["is_consolation"] == 0)
                ]
            else:
                filtered_df = pd.DataFrame()  # No data to display

            # Filter data based on selected year and Manager
            if selected_year != "All Years":
                filtered_df = filtered_df[filtered_df["year"] == selected_year]
            if selected_manager != "All Managers":
                filtered_df = filtered_df[filtered_df["Manager"] == selected_manager]

            if not filtered_df.empty:
                # Work on a local copy and de-duplicate matchups so each game appears once.
                if "opponent" in filtered_df.columns:
                    dfv = filtered_df.copy()
                    dfv["match_key"] = dfv.apply(
                        lambda r: "|".join(
                            sorted(
                                [
                                    str(r["Manager"]).lower(),
                                    str(r.get("opponent", "")).lower(),
                                ]
                            )
                        ),
                        axis=1,
                    )
                    dfv = dfv.sort_values(
                        by=["year", "week", "match_key", "team_points"],
                        ascending=[True, True, True, False],
                    )
                    dfv = dfv.drop_duplicates(
                        subset=["year", "week", "match_key"], keep="first"
                    )
                else:
                    dfv = filtered_df.copy()

                # Calculate margin
                dfv["margin"] = dfv["team_points"] - dfv["opponent_score"]

                # Ensure year column does not have commas (render as string)
                dfv["year"] = dfv["year"].astype(str)

                # Add champion column based on the Champion column in the data source
                dfv["champion"] = dfv.get("Champion", 0) == 1

                # Convert win column to boolean
                dfv["win"] = dfv["win"] == 1

                # Sort by team_points and display
                dfv = dfv.sort_values(by="team_points", ascending=False)
                st.dataframe(
                    dfv[
                        [
                            "Manager",
                            "week",
                            "year",
                            "team_points",
                            "opponent_score",
                            "margin",
                            "win",
                            "champion",
                        ]
                    ]
                )
            else:
                st.write("No data available for the selected filters")
        else:
            st.write("No data available")
