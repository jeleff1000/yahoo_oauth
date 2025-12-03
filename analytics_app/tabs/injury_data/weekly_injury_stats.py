import streamlit as st


class WeeklyInjuryStatsViewer:
    def __init__(self):
        pass

    @st.fragment
    def display(self, merged_data):
        st.header("Weekly Injury Stats")

        required_columns = ["player", "manager", "report_status", "year"]
        missing_columns = [
            col for col in required_columns if col not in merged_data.columns
        ]

        if missing_columns:
            st.error(f"Missing columns in data: {', '.join(missing_columns)}")
            return

        merged_data["player"] = merged_data["player"].astype(str)
        merged_data["manager"] = merged_data["manager"].astype(str)
        merged_data["report_status"] = merged_data["report_status"].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            players = st.multiselect(
                "Search by Player", options=sorted(merged_data["player"].unique())
            )
        with col2:
            managers = st.multiselect(
                "Search by manager", options=sorted(merged_data["manager"].unique())
            )

        col3, col4 = st.columns(2)
        with col3:
            years = sorted(merged_data["year"].unique())
            year = st.multiselect("Select year", options=["All"] + list(years))
        with col4:
            report_statuses = sorted(merged_data["report_status"].unique())
            report_status = st.multiselect(
                "Select Report Status", options=["All"] + list(report_statuses)
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

        # Build columns to display, checking for existence
        columns_to_display = [
            "week",
            "year",
            "nfl_team",
            "player",
            "manager",
            (
                "nfl_position"
                if "nfl_position" in filtered_data.columns
                else ("position" if "position" in filtered_data.columns else None)
            ),
            "fantasy_position",
            "report_primary_injury",
            "report_secondary_injury",
            "report_status",
            "practice_status",
        ]
        columns_to_display = [
            col for col in columns_to_display if col and col in filtered_data.columns
        ]

        filtered_data = filtered_data[columns_to_display]
        filtered_data["year"] = filtered_data["year"].astype(str)
        st.dataframe(filtered_data, hide_index=True)
