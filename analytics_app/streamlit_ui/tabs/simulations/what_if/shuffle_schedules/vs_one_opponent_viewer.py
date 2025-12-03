import streamlit as st
import pandas as pd


class VsOneOpponentViewer:
    """
    Head-to-head matchup matrix showing what would happen if each manager
    played every opponent for the entire season (schedule shuffle simulation).
    """

    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self):
        st.subheader("📊 Head-to-Head Simulation Matrix")
        st.caption("What if you played every opponent for an entire season? Shows simulated matchup records.")

        if self.df is None or self.df.empty:
            st.info("No data available for head-to-head simulation.")
            return

        # Compact filters - all on one row
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])

        with col1:
            years = sorted(self.df['year'].astype(int).unique())
            years_list = ["All"] + years
            default_year = years[-1] if years else "All"
            selected_year = st.selectbox("Year", years_list,
                                         index=years_list.index(default_year),
                                         key="vs_one_opponent_year_dropdown")

        with col2:
            viewer_type = st.selectbox(
                "View",
                [
                    "Win-Loss Record",
                    "Win Percentage"
                ],
                key="vs_opponent_stat_type"
            )

        with col3:
            include_regular_season = st.checkbox("Regular", value=True,
                                                 key="include_regular_season")

        with col4:
            include_postseason = st.checkbox("Playoffs", value=False,
                                             key="include_postseason")

        # Filter data
        mask = pd.Series(True, index=self.df.index)
        if selected_year != "All":
            mask &= (self.df['year'] == int(selected_year))

        season_mask = pd.Series(False, index=self.df.index)
        if include_regular_season:
            season_mask |= ((self.df['is_playoffs'] == 0) & (self.df['is_consolation'] == 0))
        if include_postseason:
            season_mask |= ((self.df['is_playoffs'] == 1) | (self.df['is_consolation'] == 1))

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
        """Display W-L records"""
        st.subheader("Win-Loss Records")

        # Extract win/loss columns
        all_cols = filtered_df.columns
        win_cols = [c for c in all_cols if c.startswith("w_vs_") and not c.endswith("_sched")]
        loss_cols = [c for c in all_cols if c.startswith("l_vs_") and not c.endswith("_sched")]

        win_suffixes = set(c[5:] for c in win_cols)
        loss_suffixes = set(c[5:] for c in loss_cols)
        suffixes = sorted(win_suffixes & loss_suffixes)

        if not suffixes:
            st.warning("No head-to-head data found.")
            return

        # Filter to only opponents with actual data (not all 0-0)
        valid_opponents = []
        for suffix in suffixes:
            w_col = f"w_vs_{suffix}"
            l_col = f"l_vs_{suffix}"
            if (filtered_df[w_col].sum() + filtered_df[l_col].sum()) > 0:
                valid_opponents.append(suffix)

        if not valid_opponents:
            st.warning("No valid opponent data found for the selected filters.")
            return

        # Build result matrix - only use valid opponents
        managers = sorted(filtered_df['manager'].unique())
        result_data = {'Manager': managers}

        for opponent in valid_opponents:
            w_col = f"w_vs_{opponent}"
            l_col = f"l_vs_{opponent}"

            records = []
            for manager in managers:
                manager_data = filtered_df[filtered_df['manager'] == manager]
                wins = int(manager_data[w_col].sum()) if w_col in manager_data.columns else 0
                losses = int(manager_data[l_col].sum()) if l_col in manager_data.columns else 0
                records.append(f"{wins}-{losses}")

            result_data[opponent.title()] = records

        result_df = pd.DataFrame(result_data)

        # Style and display
        html = self._style_table(result_df, "Win-Loss Record", is_record=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Each row shows a manager's simulated record against every opponent. Green = winning record, red = losing record.")

    @st.fragment
    def display_win_percentage(self, filtered_df):
        """Display win percentages"""
        st.subheader("Win Percentages")

        # Extract win/loss columns
        all_cols = filtered_df.columns
        win_cols = [c for c in all_cols if c.startswith("w_vs_") and not c.endswith("_sched")]
        loss_cols = [c for c in all_cols if c.startswith("l_vs_") and not c.endswith("_sched")]

        win_suffixes = set(c[5:] for c in win_cols)
        loss_suffixes = set(c[5:] for c in loss_cols)
        suffixes = sorted(win_suffixes & loss_suffixes)

        if not suffixes:
            st.warning("No head-to-head data found.")
            return

        # Filter to only opponents with actual data (not all 0-0)
        valid_opponents = []
        for suffix in suffixes:
            w_col = f"w_vs_{suffix}"
            l_col = f"l_vs_{suffix}"
            if (filtered_df[w_col].sum() + filtered_df[l_col].sum()) > 0:
                valid_opponents.append(suffix)

        if not valid_opponents:
            st.warning("No valid opponent data found for the selected filters.")
            return

        # Build result matrix - only use valid opponents
        managers = sorted(filtered_df['manager'].unique())
        result_data = {'Manager': managers}

        for opponent in valid_opponents:
            w_col = f"w_vs_{opponent}"
            l_col = f"l_vs_{opponent}"

            percentages = []
            for manager in managers:
                manager_data = filtered_df[filtered_df['manager'] == manager]
                wins = int(manager_data[w_col].sum()) if w_col in manager_data.columns else 0
                losses = int(manager_data[l_col].sum()) if l_col in manager_data.columns else 0
                total = wins + losses
                if total > 0:
                    pct = (wins / total) * 100
                    percentages.append(f"{pct:.1f}%")
                else:
                    percentages.append("—")

            result_data[opponent.title()] = percentages

        result_df = pd.DataFrame(result_data)

        # Style and display
        html = self._style_table(result_df, "Win Percentage", is_percentage=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Percentage of simulated games won by each manager against each opponent. Higher is better.")

    @st.fragment
    def display_games_played(self, filtered_df):
        """Display total games played in simulation"""
        st.subheader("Games Played")

        # Extract win/loss columns
        all_cols = filtered_df.columns
        win_cols = [c for c in all_cols if c.startswith("w_vs_") and not c.endswith("_sched")]
        loss_cols = [c for c in all_cols if c.startswith("l_vs_") and not c.endswith("_sched")]

        win_suffixes = set(c[5:] for c in win_cols)
        loss_suffixes = set(c[5:] for c in loss_cols)
        suffixes = sorted(win_suffixes & loss_suffixes)

        if not suffixes:
            st.warning("No head-to-head data found.")
            return

        # Filter to only opponents with actual data (not all 0-0)
        valid_opponents = []
        for suffix in suffixes:
            w_col = f"w_vs_{suffix}"
            l_col = f"l_vs_{suffix}"
            if (filtered_df[w_col].sum() + filtered_df[l_col].sum()) > 0:
                valid_opponents.append(suffix)

        if not valid_opponents:
            st.warning("No valid opponent data found for the selected filters.")
            return

        # Build result matrix - only use valid opponents
        managers = sorted(filtered_df['manager'].unique())
        result_data = {'Manager': managers}

        for opponent in valid_opponents:
            w_col = f"w_vs_{opponent}"
            l_col = f"l_vs_{opponent}"

            games = []
            for manager in managers:
                manager_data = filtered_df[filtered_df['manager'] == manager]
                wins = int(manager_data[w_col].sum()) if w_col in manager_data.columns else 0
                losses = int(manager_data[l_col].sum()) if l_col in manager_data.columns else 0
                total = wins + losses
                games.append(str(total))

            result_data[opponent.title()] = games

        result_df = pd.DataFrame(result_data)

        # Style and display
        html = self._style_table(result_df, "Games Played", is_numeric=False)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Total simulated games between each manager and opponent.")

    def _style_table(self, df, title, is_record=False, is_percentage=False, is_numeric=False):
        """Create a styled HTML table"""

        html = f"""
        <style>
            .h2h-sim-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }}
            .h2h-sim-table thead tr {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #ffffff;
                text-align: center;
                font-weight: bold;
            }}
            .h2h-sim-table th,
            .h2h-sim-table td {{
                padding: 12px 8px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .h2h-sim-table tbody tr {{
                border-bottom: 1px solid #dddddd;
            }}
            .h2h-sim-table tbody tr:nth-of-type(even) {{
                background-color: #f3f3f3;
            }}
            .h2h-sim-table tbody tr:hover {{
                background-color: #e8e8e8;
                cursor: pointer;
            }}
            .h2h-sim-table .row-header {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
                text-align: left;
                padding-left: 15px;
            }}
            .winning-record {{
                background-color: #d4edda !important;
                color: #155724;
                font-weight: bold;
            }}
            .losing-record {{
                background-color: #f8d7da !important;
                color: #721c24;
                font-weight: bold;
            }}
            .even-record {{
                background-color: #fff3cd !important;
                color: #856404;
                font-weight: bold;
            }}
        </style>
        <table class="h2h-sim-table">
            <thead>
                <tr>
                    <th>Manager ↓ vs Opponent →</th>
        """

        for col in df.columns[1:]:  # Skip 'Manager' column
            html += f"<th>{col}</th>"

        html += "</tr></thead><tbody>"

        for idx, row in df.iterrows():
            html += f"<tr><td class='row-header'>{row['Manager']}</td>"
            for col in df.columns[1:]:
                value = row[col]
                cell_class = ""

                if value == "—":
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
                    except:
                        pass

                html += f"<td class='{cell_class}'>{value}</td>"

            html += "</tr>"

        html += "</tbody></table>"

        return html
