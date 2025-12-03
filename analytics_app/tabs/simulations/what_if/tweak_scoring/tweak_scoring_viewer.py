import streamlit as st
import pandas as pd
import numpy as np
from tabs.matchups.weekly.weekly_matchup_overview import WeeklyMatchupDataViewer
from .shuffle_scores import calculate_std_dev, tweak_scores, calculate_playoff_seed
from .shuffle_schedule import shuffle_schedule
from ..shuffle_schedules.table_styles import render_modern_table


class TweakScoringViewer(WeeklyMatchupDataViewer):
    def __init__(self, matchup_data_df):
        super().__init__(matchup_data_df)
        self.df = matchup_data_df

    @st.fragment
    def display(self):
        if self.df is None:
            st.info("No data available")
            return

        st.subheader("Tweak Scoring")

        col1, col2 = st.columns([1, 3])
        with col1:
            years = sorted(self.df["year"].unique().tolist())
            years_list = ["All Years"] + years
            default_year = years[-1] if years else "All Years"
            selected_year = st.selectbox(
                "Select Year",
                years_list,
                index=years_list.index(default_year),
                key="tweak_scoring_year",
            )

        col3, col4 = st.columns(2)
        with col3:
            show_regular_season = st.checkbox(
                "Regular Season", value=True, key="regular_season_checkbox"
            )
        with col4:
            show_postseason = st.checkbox(
                "Postseason", value=False, key="postseason_checkbox"
            )

        col5, col6 = st.columns(2)
        with col5:
            tweak_scores_flag = st.checkbox("Tweak Scores", key="tweak_scores_checkbox")
        with col6:
            shuffle_schedule_flag = st.checkbox(
                "Shuffle Schedule", key="shuffle_schedule_checkbox"
            )

        if st.button("Simulate"):
            # Vectorized filtering using pandas
            mask = pd.Series(True, index=self.df.index)

            if selected_year != "All Years":
                mask &= self.df["year"] == selected_year

            if show_regular_season and show_postseason:
                mask &= (
                    (self.df["is_playoffs"] == 0) & (self.df["is_consolation"] == 0)
                ) | (self.df["is_playoffs"] == 1)
            elif show_regular_season:
                mask &= (self.df["is_playoffs"] == 0) & (self.df["is_consolation"] == 0)
            elif show_postseason:
                mask &= self.df["is_playoffs"] == 1

            filtered_df = self.df[mask].copy()

            if filtered_df.empty:
                st.warning("No data matches the selected filters.")
                return

            # Apply schedule shuffle if requested
            if shuffle_schedule_flag:
                filtered_df = shuffle_schedule(filtered_df)

            # Initialize tweaked points column
            if "tweaked_team_points" not in filtered_df.columns:
                filtered_df["tweaked_team_points"] = filtered_df["team_points"]

            # Apply score tweaking if requested
            if tweak_scores_flag:
                std_dev_df = calculate_std_dev(
                    filtered_df, selected_year, show_regular_season, show_postseason
                )
                filtered_df = tweak_scores(filtered_df, std_dev_df)

            # Initialize simulation columns if not present
            if "Sim_Wins" not in filtered_df.columns:
                filtered_df["Sim_Wins"] = 0
            if "Sim_Losses" not in filtered_df.columns:
                filtered_df["Sim_Losses"] = 0

            # Calculate playoff seeds
            filtered_df = calculate_playoff_seed(filtered_df)

            if "Sim_Playoff_Seed" not in filtered_df.columns:
                filtered_df["Sim_Playoff_Seed"] = np.nan

            # Optimized aggregation using pandas
            # Get playoff_seed_to_date from the latest week for each manager/year
            latest_week_idx = filtered_df.groupby(["year", "manager"])["week"].idxmax()
            seeds_df = filtered_df.loc[
                latest_week_idx, ["year", "manager", "playoff_seed_to_date"]
            ]

            # Aggregate stats
            aggregated_df = (
                filtered_df.groupby(["year", "manager"], as_index=False)
                .agg(
                    {
                        "team_points": "sum",
                        "tweaked_team_points": "sum",
                        "Sim_Wins": "sum",
                        "Sim_Losses": "sum",
                        "win": "sum",
                        "loss": "sum",
                        "Sim_Playoff_Seed": "max",
                    }
                )
                .rename(
                    columns={
                        "team_points": "Points",
                        "tweaked_team_points": "Sim_Points",
                        "Sim_Wins": "Sim_W",
                        "Sim_Losses": "Sim_L",
                        "win": "Wins",
                        "loss": "Losses",
                        "Sim_Playoff_Seed": "Sim_Seed",
                    }
                )
            )

            # Merge with seeds
            aggregated_df = aggregated_df.merge(
                seeds_df.rename(columns={"playoff_seed_to_date": "Seed"}),
                on=["year", "manager"],
                how="left",
            )

            # Convert year to string for display
            aggregated_df["year"] = aggregated_df["year"].astype(str)

            # Sort by actual seed (best to worst)
            aggregated_df = aggregated_df.sort_values(by=["Seed"])

            # Define columns for display
            if selected_year == "All Years":
                actual_df = aggregated_df[
                    ["year", "manager", "Seed", "Wins", "Losses", "Points"]
                ].copy()
                sim_df = aggregated_df[
                    ["year", "manager", "Sim_Seed", "Sim_W", "Sim_L", "Sim_Points"]
                ].copy()

                # Set multi-index for actual
                actual_df = actual_df.set_index(["year", "manager"])
                actual_df.index.names = ["Year", "Manager"]

                # Set multi-index for simulated (sorted by Sim_Seed)
                sim_df = sim_df.sort_values(by=["Sim_Seed"]).set_index(
                    ["year", "manager"]
                )
                sim_df.index.names = ["Year", "Manager"]
            else:
                actual_df = aggregated_df[
                    ["manager", "Seed", "Wins", "Losses", "Points"]
                ].copy()
                sim_df = aggregated_df[
                    ["manager", "Sim_Seed", "Sim_W", "Sim_L", "Sim_Points"]
                ].copy()

                # Set index for actual
                actual_df = actual_df.set_index("manager")
                actual_df.index.name = "Manager"

                # Set index for simulated (sorted by Sim_Seed)
                sim_df = sim_df.sort_values(by=["Sim_Seed"]).set_index("manager")
                sim_df.index.name = "Manager"

            # Store results in session state so they persist across reruns (e.g., when sorting)
            st.session_state.tweak_scoring_actual_df = actual_df
            st.session_state.tweak_scoring_sim_df = sim_df

        # Display results if they exist in session state
        if (
            "tweak_scoring_actual_df" in st.session_state
            and "tweak_scoring_sim_df" in st.session_state
        ):
            st.markdown("---")

            # Display results side by side
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Actual Results**")
                render_modern_table(
                    st.session_state.tweak_scoring_actual_df,
                    title="",
                    color_columns=["Wins", "Points"],
                    reverse_columns=["Seed", "Losses"],
                    format_specs={
                        "Seed": "{:.0f}",
                        "Wins": "{:.0f}",
                        "Losses": "{:.0f}",
                        "Points": "{:.1f}",
                    },
                    column_names={
                        "Seed": "Seed",
                        "Wins": "W",
                        "Losses": "L",
                        "Points": "Pts",
                    },
                    gradient_by_column=True,
                )

            with col2:
                st.markdown("**Simulated Results**")
                render_modern_table(
                    st.session_state.tweak_scoring_sim_df,
                    title="",
                    color_columns=["Sim_W", "Sim_Points"],
                    reverse_columns=["Sim_Seed", "Sim_L"],
                    format_specs={
                        "Sim_Seed": "{:.0f}",
                        "Sim_W": "{:.0f}",
                        "Sim_L": "{:.0f}",
                        "Sim_Points": "{:.1f}",
                    },
                    column_names={
                        "Sim_Seed": "Seed",
                        "Sim_W": "W",
                        "Sim_L": "L",
                        "Sim_Points": "Pts",
                    },
                    gradient_by_column=True,
                )
