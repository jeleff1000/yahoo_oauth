import streamlit as st


class CareerInjuryStatsViewer:
    def __init__(self):
        pass

    @st.fragment
    def display(self, merged_data):
        st.header("Career Injury Stats")

        merged_data["player"] = merged_data["player"].astype(str)
        merged_data["manager"] = merged_data["manager"].astype(str)
        merged_data["report_status"] = merged_data["report_status"].astype(str)

        # Choose position column
        position_col = None
        if "nfl_position" in merged_data.columns:
            position_col = "nfl_position"
            merged_data[position_col] = merged_data[position_col].astype(str)
        elif "position" in merged_data.columns:
            position_col = "position"
            merged_data[position_col] = merged_data[position_col].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            players = st.multiselect(
                "Search by Player",
                options=sorted(merged_data["player"].unique()),
                key="career_player_multiselect",
            )
        with col2:
            manager_search = st.text_input(
                "Search by manager", key="career_manager_search"
            )

        report_statuses = sorted(merged_data["report_status"].unique())
        report_status = st.multiselect(
            "Select Report Status",
            options=["All"] + list(report_statuses),
            key="career_report_status_multiselect",
        )

        filtered_data = merged_data[
            (
                merged_data["player"].isin(players)
                if players
                else merged_data["player"].notna()
            )
            & (
                merged_data["manager"].str.contains(manager_search, case=False)
                if manager_search
                else merged_data["manager"].notna()
            )
            & (
                (merged_data["report_status"].isin(report_status))
                if report_status and "All" not in report_status
                else merged_data["report_status"].notna()
            )
        ]

        # Aggregate the data by player and position
        index_cols = ["player"]
        if position_col:
            index_cols.append(position_col)
        aggregated_data = filtered_data.pivot_table(
            index=index_cols, columns="report_status", aggfunc="size", fill_value=0
        ).reset_index()

        for status in ["Questionable", "Doubtful", "Out"]:
            if status not in aggregated_data.columns:
                aggregated_data[status] = 0

        # Build columns to display
        columns_to_display = ["player"]
        if position_col:
            columns_to_display.append(position_col)
        columns_to_display += ["Questionable", "Doubtful", "Out"]
        columns_to_display = [
            col for col in columns_to_display if col in aggregated_data.columns
        ]

        aggregated_data = aggregated_data[columns_to_display]
        st.dataframe(aggregated_data, hide_index=True)
