import streamlit as st
import pandas as pd


class SeasonInjuryStatsViewer:
    def __init__(self):
        pass

    @st.fragment
    def display(self, merged_data):
        st.header("Season Injury Stats")

        merged_data["player"] = merged_data["player"].astype(str)
        merged_data["manager"] = merged_data["manager"].astype(str)
        merged_data["report_status"] = merged_data["report_status"].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            players = st.multiselect(
                "Search by Player",
                options=sorted(merged_data["player"].unique()),
                key="player_multiselect",
            )
        with col2:
            managers = st.multiselect(
                "Search by manager",
                options=sorted(merged_data["manager"].unique()),
                key="manager_multiselect",
            )

        col3, col4 = st.columns(2)
        with col3:
            years = sorted(merged_data["year"].unique())
            year = st.multiselect(
                "Select Year", options=["All"] + list(years), key="year_multiselect"
            )
        with col4:
            report_statuses = sorted(merged_data["report_status"].unique())
            report_status = st.multiselect(
                "Select Report Status",
                options=["All"] + list(report_statuses),
                key="report_status_multiselect",
            )

        filtered_data = merged_data[
            (
                merged_data["player"].isin(players)
                if players
                else merged_data["player"].notna()
            )
            & (
                merged_data["manager"].isin(managers)
                if managers
                else merged_data["manager"].notna()
            )
            & (
                (merged_data["year"].isin(year))
                if year and "All" not in year
                else merged_data["year"].notna()
            )
            & (
                (merged_data["report_status"].isin(report_status))
                if report_status and "All" not in report_status
                else merged_data["report_status"].notna()
            )
        ]

        aggregated_data = filtered_data.pivot_table(
            index=["player", "year"],
            columns="report_status",
            aggfunc="size",
            fill_value=0,
        ).reset_index()

        for status in ["Questionable", "Doubtful", "Out"]:
            if status not in aggregated_data.columns:
                aggregated_data[status] = 0

        aggregated_data.columns = ["player", "year"] + list(aggregated_data.columns[2:])

        # Choose position column
        position_col = None
        if "nfl_position" in filtered_data.columns:
            position_col = "nfl_position"
        elif "position" in filtered_data.columns:
            position_col = "position"

        # Group additional columns, only if they exist
        group_cols = {"manager": "first"}
        if "nfl_team" in filtered_data.columns:
            group_cols["nfl_team"] = "first"
        if position_col:
            group_cols[position_col] = "first"

        additional_columns = (
            filtered_data.groupby(["player", "year"]).agg(group_cols).reset_index()
        )
        aggregated_data = pd.merge(
            aggregated_data, additional_columns, on=["player", "year"]
        )

        # Build columns to display, only if they exist
        columns_to_display = ["year", "player"]
        if "nfl_team" in aggregated_data.columns:
            columns_to_display.append("nfl_team")
        columns_to_display.append("manager")
        if position_col and position_col in aggregated_data.columns:
            columns_to_display.append(position_col)
        columns_to_display += ["Questionable", "Doubtful", "Out"]
        columns_to_display = [
            col for col in columns_to_display if col in aggregated_data.columns
        ]

        aggregated_data = aggregated_data[columns_to_display]
        aggregated_data["year"] = aggregated_data["year"].astype(str)
        st.dataframe(aggregated_data, hide_index=True)
