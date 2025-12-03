import streamlit as st
import pandas as pd
import re


class EveryonesScheduleViewer:
    """
    Shows what each manager's record would be if they played everyone else's schedule.
    This reveals the impact of schedule strength on season outcomes.
    """

    def __init__(self, matchup_data_df):
        self.df = matchup_data_df

    @st.fragment
    def display(self):
        st.subheader("📊 Your Scores vs Everyone's Schedules")
        st.caption(
            "What would your record be if you played someone else's schedule? Shows cross-schedule performance."
        )

        if self.df is None or self.df.empty:
            st.info("No data available for schedule comparison.")
            return

        # Compact filters - all on one row
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])

        with col1:
            years = sorted(self.df["year"].astype(int).unique().tolist())
            years_list = ["All"] + years
            default_year = years[-1] if years else "All"
            selected_year = st.selectbox(
                "Year",
                years_list,
                index=years_list.index(default_year),
                key="everyones_schedule_year_dropdown",
            )

        with col2:
            viewer_type = st.selectbox(
                "View",
                ["Win-Loss Record", "Win Percentage"],
                key="schedule_comp_stat_type",
            )

        with col3:
            include_regular_season = st.checkbox(
                "Regular", value=True, key="everyones_schedule_include_regular_season"
            )

        with col4:
            include_postseason = st.checkbox(
                "Playoffs", value=False, key="everyones_schedule_include_postseason"
            )

        # Filter data
        mask = pd.Series(True, index=self.df.index)
        if selected_year != "All":
            mask &= self.df["year"] == int(selected_year)

        season_mask = pd.Series(False, index=self.df.index)
        if include_regular_season:
            season_mask |= (self.df["is_playoffs"] == 0) & (
                self.df["is_consolation"] == 0
            )
        if include_postseason:
            season_mask |= (self.df["is_playoffs"] == 1) | (
                self.df["is_consolation"] == 1
            )

        mask &= season_mask
        filtered_df = self.df[mask]

        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
            return

        # Display the selected view
        if viewer_type == "Win-Loss Record":
            self.display_record(filtered_df)
        elif viewer_type == "Win Percentage":
            self.display_win_percentage(filtered_df)

    @st.fragment
    def display_record(self, filtered_df):
        """Display W-L records with each schedule"""
        st.subheader("Win-Loss Records vs Each Schedule")

        # Extract schedule columns
        all_cols = filtered_df.columns
        win_cols = [c for c in all_cols if re.match(r"w_vs_(.+)_sched", c)]
        opponent_names = sorted(
            [re.match(r"w_vs_(.+)_sched", c).group(1) for c in win_cols]
        )

        if not opponent_names:
            st.warning("No schedule comparison data found.")
            return

        # Validate opponents have data
        valid_opponents = []
        for opponent in opponent_names:
            w_col = f"w_vs_{opponent}_sched"
            l_col = f"l_vs_{opponent}_sched"
            if w_col in filtered_df.columns and l_col in filtered_df.columns:
                if (filtered_df[w_col].sum() + filtered_df[l_col].sum()) > 0:
                    valid_opponents.append(opponent)

        if not valid_opponents:
            st.warning("No valid schedule data found.")
            return

        # Build result matrix
        managers = sorted(filtered_df["manager"].unique())
        result_data = {"Manager": managers}

        for opponent in valid_opponents:
            w_col = f"w_vs_{opponent}_sched"
            l_col = f"l_vs_{opponent}_sched"

            records = []
            for manager in managers:
                manager_rows = filtered_df[filtered_df["manager"] == manager]
                wins = int(manager_rows[w_col].sum())
                losses = int(manager_rows[l_col].sum())
                records.append(f"{wins}-{losses}")

            result_data[f"{opponent.title()}'s Schedule"] = records

        result_df = pd.DataFrame(result_data)

        # Style and display
        html = self._style_table(result_df, "Win-Loss Record", is_record=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption(
            "💡 **How to read:** Each cell shows a manager's record if they played that column's schedule. Highlighted cells show your actual schedule performance."
        )

    @st.fragment
    def display_win_percentage(self, filtered_df):
        """Display win percentages with each schedule"""
        st.subheader("Win Percentages vs Each Schedule")

        # Extract schedule columns
        all_cols = filtered_df.columns
        win_cols = [c for c in all_cols if re.match(r"w_vs_(.+)_sched", c)]
        opponent_names = sorted(
            [re.match(r"w_vs_(.+)_sched", c).group(1) for c in win_cols]
        )

        if not opponent_names:
            st.warning("No schedule comparison data found.")
            return

        # Validate opponents have data
        valid_opponents = []
        for opponent in opponent_names:
            w_col = f"w_vs_{opponent}_sched"
            l_col = f"l_vs_{opponent}_sched"
            if w_col in filtered_df.columns and l_col in filtered_df.columns:
                if (filtered_df[w_col].sum() + filtered_df[l_col].sum()) > 0:
                    valid_opponents.append(opponent)

        if not valid_opponents:
            st.warning("No valid schedule data found.")
            return

        # Build result matrix
        managers = sorted(filtered_df["manager"].unique())
        result_data = {"Manager": managers}

        for opponent in valid_opponents:
            w_col = f"w_vs_{opponent}_sched"
            l_col = f"l_vs_{opponent}_sched"

            percentages = []
            for manager in managers:
                manager_rows = filtered_df[filtered_df["manager"] == manager]
                wins = int(manager_rows[w_col].sum())
                losses = int(manager_rows[l_col].sum())
                total = wins + losses
                if total > 0:
                    pct = (wins / total) * 100
                    percentages.append(f"{pct:.1f}%")
                else:
                    percentages.append("—")

            result_data[f"{opponent.title()}'s Schedule"] = percentages

        result_df = pd.DataFrame(result_data)

        # Style and display
        html = self._style_table(result_df, "Win Percentage", is_percentage=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption(
            "💡 **How to read:** Percentage of games won if each manager played that schedule. Higher is better."
        )

    @st.fragment
    def display_games_played(self, filtered_df):
        """Display total games in each schedule comparison"""
        st.subheader("Games Played")

        # Extract schedule columns
        all_cols = filtered_df.columns
        win_cols = [c for c in all_cols if re.match(r"w_vs_(.+)_sched", c)]
        opponent_names = sorted(
            [re.match(r"w_vs_(.+)_sched", c).group(1) for c in win_cols]
        )

        if not opponent_names:
            st.warning("No schedule comparison data found.")
            return

        # Validate opponents have data
        valid_opponents = []
        for opponent in opponent_names:
            w_col = f"w_vs_{opponent}_sched"
            l_col = f"l_vs_{opponent}_sched"
            if w_col in filtered_df.columns and l_col in filtered_df.columns:
                if (filtered_df[w_col].sum() + filtered_df[l_col].sum()) > 0:
                    valid_opponents.append(opponent)

        if not valid_opponents:
            st.warning("No valid schedule data found.")
            return

        # Build result matrix
        managers = sorted(filtered_df["manager"].unique())
        result_data = {"Manager": managers}

        for opponent in valid_opponents:
            w_col = f"w_vs_{opponent}_sched"
            l_col = f"l_vs_{opponent}_sched"

            games = []
            for manager in managers:
                manager_rows = filtered_df[filtered_df["manager"] == manager]
                wins = int(manager_rows[w_col].sum())
                losses = int(manager_rows[l_col].sum())
                total = wins + losses
                games.append(str(total))

            result_data[f"{opponent.title()}'s Schedule"] = games

        result_df = pd.DataFrame(result_data)

        # Style and display
        html = self._style_table(result_df, "Games Played", is_numeric=False)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Total games in each schedule comparison.")

    def _style_table(
        self, df, title, is_record=False, is_percentage=False, is_numeric=False
    ):
        """Create a styled HTML table with diagonal highlighting for own schedule"""

        html = """
        <style>
            .schedule-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            .schedule-table thead tr {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #ffffff;
                text-align: center;
                font-weight: bold;
            }
            .schedule-table th,
            .schedule-table td {
                padding: 12px 8px;
                text-align: center;
                border: 1px solid #ddd;
            }
            .schedule-table tbody tr {
                border-bottom: 1px solid #dddddd;
            }
            .schedule-table tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            .schedule-table tbody tr:hover {
                background-color: #e8e8e8;
                cursor: pointer;
            }
            .schedule-table .row-header {
                background-color: #667eea;
                color: white;
                font-weight: bold;
                text-align: left;
                padding-left: 15px;
            }
            .winning-record {
                background-color: #d4edda !important;
                color: #155724;
                font-weight: bold;
            }
            .losing-record {
                background-color: #f8d7da !important;
                color: #721c24;
                font-weight: bold;
            }
            .even-record {
                background-color: #fff3cd !important;
                color: #856404;
                font-weight: bold;
            }
            .own-schedule {
                background-color: #ffeb3b !important;
                color: #000;
                font-weight: bold;
                border: 2px solid #ffc107 !important;
            }
        </style>
        <table class="schedule-table">
            <thead>
                <tr>
                    <th>Manager ↓ Playing Schedule →</th>
        """

        for col in df.columns[1:]:  # Skip 'Manager' column
            html += f"<th>{col}</th>"

        html += "</tr></thead><tbody>"

        for idx, row in df.iterrows():
            manager = row["Manager"]
            html += f"<tr><td class='row-header'>{manager}</td>"

            for col in df.columns[1:]:
                value = row[col]
                cell_class = ""

                # Highlight diagonal (own schedule)
                if manager.lower() in col.lower():
                    cell_class = "own-schedule"
                elif value == "—":
                    cell_class = ""
                elif is_record and "-" in str(value):
                    # Parse W-L record
                    parts = str(value).split("-")
                    if len(parts) == 2:
                        wins = int(parts[0])
                        losses = int(parts[1])
                        if wins > losses:
                            cell_class = "winning-record"
                        elif losses > wins:
                            cell_class = "losing-record"
                        else:
                            cell_class = "even-record"
                elif is_percentage and "%" in str(value):
                    # Color code percentages
                    try:
                        pct = float(str(value).replace("%", ""))
                        if pct >= 60:
                            cell_class = "winning-record"
                        elif pct <= 40:
                            cell_class = "losing-record"
                        else:
                            cell_class = "even-record"
                    except Exception:
                        pass

                html += f"<td class='{cell_class}'>{value}</td>"

            html += "</tr>"

        html += "</tbody></table>"

        return html
