import pandas as pd
import streamlit as st

class SeasonHeadToHeadViewer:
    """
    Enhanced head-to-head viewer for season matchups with multiple stat views and styled HTML tables.
    """

    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        st.header("🤝 Head-to-Head Matchup Matrix")

        if self.df is None or self.df.empty:
            st.info("No data available for head-to-head view.")
            return

        required_columns = ['manager', 'opponent', 'win', 'loss', 'team_points', 'opponent_points', 'margin']
        if not all(column in self.df.columns for column in required_columns):
            st.error("Required columns are missing from the data source")
            return

        # Stat type selector
        st.markdown("""
        **Choose what to display:** Select how you want to compare managers head-to-head.
        Each cell shows the stat for the row manager against the column opponent.
        """)

        col1, col2 = st.columns([1, 3])
        with col1:
            viewer_type = st.selectbox(
                "📊 Stat Type",
                [
                    "Win-Loss Record",
                    "Win %",
                    "Total Points For",
                    "Avg Points Per Season",
                    "Total Margin",
                    "Avg Margin Per Season",
                    "Highest Season Score",
                    "Lowest Season Score",
                    "Most Recent Result",
                    "Head-to-Head Streak"
                ],
                key=f"{prefix}_h2h_stat_type"
            )

        # Display the selected view
        if viewer_type == "Win-Loss Record":
            self.display_record()
        elif viewer_type == "Win %":
            self.display_win_percentage()
        elif viewer_type == "Total Points For":
            self.display_total_points()
        elif viewer_type == "Avg Points Per Season":
            self.display_per_game()
        elif viewer_type == "Total Margin":
            self.display_margin()
        elif viewer_type == "Avg Margin Per Season":
            self.display_avg_margin()
        elif viewer_type == "Highest Season Score":
            self.display_highest_score()
        elif viewer_type == "Lowest Season Score":
            self.display_lowest_score()
        elif viewer_type == "Most Recent Result":
            self.display_most_recent()
        elif viewer_type == "Head-to-Head Streak":
            self.display_streak()

    @st.fragment
    def display_record(self):
        """Display W-L record in each matchup"""
        st.subheader("Win-Loss Records")

        wins = self.df.pivot_table(
            index='manager', columns='opponent', values='win', aggfunc='sum', fill_value=0
        )
        losses = self.df.pivot_table(
            index='manager', columns='opponent', values='loss', aggfunc='sum', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        record_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    record_table.at[manager, opponent] = "—"
                else:
                    w = wins.at[manager, opponent] if manager in wins.index and opponent in wins.columns else 0
                    l = losses.at[manager, opponent] if manager in losses.index and opponent in losses.columns else 0
                    record_table.at[manager, opponent] = f"{int(w)}-{int(l)}"

        html = self._style_table(record_table, "Win-Loss Record", is_record=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Rows show each manager's record against opponents (columns). Green cells indicate winning records, red indicates losing records.")

    @st.fragment
    def display_win_percentage(self):
        """Display win percentage in each matchup"""
        st.subheader("Win Percentage")

        wins = self.df.pivot_table(
            index='manager', columns='opponent', values='win', aggfunc='sum', fill_value=0
        )
        losses = self.df.pivot_table(
            index='manager', columns='opponent', values='loss', aggfunc='sum', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        win_pct_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    win_pct_table.at[manager, opponent] = "—"
                else:
                    w = wins.at[manager, opponent] if manager in wins.index and opponent in wins.columns else 0
                    l = losses.at[manager, opponent] if manager in losses.index and opponent in losses.columns else 0
                    total = w + l
                    if total > 0:
                        pct = (w / total) * 100
                        win_pct_table.at[manager, opponent] = f"{pct:.2f}%"
                    else:
                        win_pct_table.at[manager, opponent] = "—"

        html = self._style_table(win_pct_table, "Win Percentage", is_percentage=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Percentage of seasons won by row manager against column opponent. Higher is better.")

    @st.fragment
    def display_total_points(self):
        """Display total points scored in matchups"""
        st.subheader("Total Points Scored")

        pivot_table = self.df.pivot_table(
            index='manager', columns='opponent', values='team_points', aggfunc='sum', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        points_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    points_table.at[manager, opponent] = "—"
                else:
                    pts = pivot_table.at[manager, opponent] if manager in pivot_table.index and opponent in pivot_table.columns else 0
                    points_table.at[manager, opponent] = f"{pts:.2f}"

        html = self._style_table(points_table, "Total Points Scored", is_numeric=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Total points scored by row manager in all seasons against column opponent.")

    @st.fragment
    def display_per_game(self):
        """Display average points per season"""
        st.subheader("Average Points Per Season")

        pivot_table = self.df.pivot_table(
            index='manager', columns='opponent', values='team_points', aggfunc='mean', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        avg_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    avg_table.at[manager, opponent] = "—"
                else:
                    avg = pivot_table.at[manager, opponent] if manager in pivot_table.index and opponent in pivot_table.columns else 0
                    avg_table.at[manager, opponent] = f"{avg:.2f}"

        html = self._style_table(avg_table, "Avg Points Per Season", is_numeric=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Average points per season scored by row manager against column opponent.")

    @st.fragment
    def display_margin(self):
        """Display total margin of victory/defeat"""
        st.subheader("Total Margin")

        pivot_table = self.df.pivot_table(
            index='manager', columns='opponent', values='margin', aggfunc='sum', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        margin_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    margin_table.at[manager, opponent] = "—"
                else:
                    margin = pivot_table.at[manager, opponent] if manager in pivot_table.index and opponent in pivot_table.columns else 0
                    margin_table.at[manager, opponent] = f"{margin:.2f}"

        html = self._style_table(margin_table, "Total Margin", is_margin=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Cumulative point differential for row manager vs column opponent. Positive = net wins, negative = net losses.")

    @st.fragment
    def display_avg_margin(self):
        """Display average margin per season"""
        st.subheader("Average Margin Per Season")

        pivot_table = self.df.pivot_table(
            index='manager', columns='opponent', values='margin', aggfunc='mean', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        avg_margin_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    avg_margin_table.at[manager, opponent] = "—"
                else:
                    avg_margin = pivot_table.at[manager, opponent] if manager in pivot_table.index and opponent in pivot_table.columns else 0
                    avg_margin_table.at[manager, opponent] = f"{avg_margin:.2f}"

        html = self._style_table(avg_margin_table, "Avg Margin Per Season", is_margin=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Average point differential per season for row manager vs column opponent.")

    @st.fragment
    def display_highest_score(self):
        """Display highest score in matchups against each opponent"""
        st.subheader("Highest Season Score Against Each Opponent")

        pivot_table = self.df.pivot_table(
            index='manager', columns='opponent', values='team_points', aggfunc='max', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        highest_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    highest_table.at[manager, opponent] = "—"
                else:
                    pts = pivot_table.at[manager, opponent] if manager in pivot_table.index and opponent in pivot_table.columns else 0
                    highest_table.at[manager, opponent] = f"{pts:.2f}"

        html = self._style_table(highest_table, "Highest Score", is_numeric=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Highest single-season score by row manager against column opponent.")

    @st.fragment
    def display_lowest_score(self):
        """Display lowest score in matchups against each opponent"""
        st.subheader("Lowest Season Score Against Each Opponent")

        df_filtered = self.df[self.df['team_points'] > 0].copy()

        pivot_table = df_filtered.pivot_table(
            index='manager', columns='opponent', values='team_points', aggfunc='min', fill_value=0
        )

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        lowest_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    lowest_table.at[manager, opponent] = "—"
                else:
                    pts = pivot_table.at[manager, opponent] if manager in pivot_table.index and opponent in pivot_table.columns else 0
                    if pts > 0:
                        lowest_table.at[manager, opponent] = f"{pts:.2f}"
                    else:
                        lowest_table.at[manager, opponent] = "—"

        html = self._style_table(lowest_table, "Lowest Score", is_numeric=False)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Lowest single-season score by row manager against column opponent.")

    @st.fragment
    def display_most_recent(self):
        """Display most recent matchup result"""
        st.subheader("Most Recent Season Result")

        df_sorted = self.df.sort_values(['year'], ascending=False)

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        recent_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    recent_table.at[manager, opponent] = "—"
                else:
                    matchups = df_sorted[(df_sorted['manager'] == manager) & (df_sorted['opponent'] == opponent)]
                    if not matchups.empty:
                        latest = matchups.iloc[0]
                        result = "W" if latest['win'] else "L"
                        score = f"{latest['team_points']:.2f}"
                        recent_table.at[manager, opponent] = f"{result} ({score})"
                    else:
                        recent_table.at[manager, opponent] = "—"

        html = self._style_table(recent_table, "Most Recent Result", is_result=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Most recent season result for row manager vs column opponent. Format: W/L (score).")

    @st.fragment
    def display_streak(self):
        """Display current head-to-head streak"""
        st.subheader("Current Head-to-Head Streak")

        df_sorted = self.df.sort_values(['manager', 'opponent', 'year'])

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        streak_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    streak_table.at[manager, opponent] = "—"
                else:
                    matchups = df_sorted[(df_sorted['manager'] == manager) & (df_sorted['opponent'] == opponent)]
                    if not matchups.empty:
                        results = matchups['win'].tolist()
                        if results:
                            current_result = results[-1]
                            streak = 1
                            for i in range(len(results) - 2, -1, -1):
                                if results[i] == current_result:
                                    streak += 1
                                else:
                                    break
                            result_char = "W" if current_result else "L"
                            streak_table.at[manager, opponent] = f"{result_char}{streak}"
                        else:
                            streak_table.at[manager, opponent] = "—"
                    else:
                        streak_table.at[manager, opponent] = "—"

        html = self._style_table(streak_table, "Current Streak", is_streak=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Current streak for row manager vs column opponent. W3 = 3-season win streak, L2 = 2-season losing streak.")

    def _style_table(self, df, title, is_record=False, is_percentage=False, is_numeric=False, is_margin=False, is_result=False, is_streak=False):
        """Create a styled HTML table"""

        html = f"""
        <style>
            .h2h-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }}
            .h2h-table thead tr {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #ffffff;
                text-align: center;
                font-weight: bold;
            }}
            .h2h-table th,
            .h2h-table td {{
                padding: 12px 8px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .h2h-table tbody tr {{
                border-bottom: 1px solid #dddddd;
            }}
            .h2h-table tbody tr:nth-of-type(even) {{
                background-color: #f3f3f3;
            }}
            .h2h-table tbody tr:hover {{
                background-color: #e8e8e8;
                cursor: pointer;
            }}
            .h2h-table .row-header {{
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
            .positive-margin {{
                background-color: #d4edda !important;
                color: #155724;
            }}
            .negative-margin {{
                background-color: #f8d7da !important;
                color: #721c24;
            }}
            .high-value {{
                background-color: #d1ecf1 !important;
                color: #0c5460;
                font-weight: bold;
            }}
            .win-streak {{
                background-color: #d4edda !important;
                color: #155724;
                font-weight: bold;
            }}
            .loss-streak {{
                background-color: #f8d7da !important;
                color: #721c24;
                font-weight: bold;
            }}
        </style>
        <table class="h2h-table">
            <thead>
                <tr>
                    <th>Manager ↓ vs Opponent →</th>
        """

        for col in df.columns:
            html += f"<th>{col}</th>"

        html += "</tr></thead><tbody>"

        for idx in df.index:
            html += f"<tr><td class='row-header'>{idx}</td>"
            for col in df.columns:
                value = df.at[idx, col]
                cell_class = ""

                if value == "—":
                    cell_class = ""
                elif is_record and "-" in str(value):
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
                elif is_margin:
                    try:
                        margin = float(value)
                        if margin > 0:
                            cell_class = "positive-margin"
                        elif margin < 0:
                            cell_class = "negative-margin"
                    except:
                        pass
                elif is_result and "(" in str(value):
                    if value.startswith("W"):
                        cell_class = "winning-record"
                    elif value.startswith("L"):
                        cell_class = "losing-record"
                elif is_streak and value != "—":
                    if value.startswith("W"):
                        cell_class = "win-streak"
                    elif value.startswith("L"):
                        cell_class = "loss-streak"
                elif is_numeric:
                    try:
                        val = float(value)
                        numeric_vals = []
                        for v in df.values.flatten():
                            if v != "—":
                                try:
                                    numeric_vals.append(float(v))
                                except:
                                    pass
                        if numeric_vals and val >= sorted(numeric_vals)[-len(numeric_vals)//4]:
                            cell_class = "high-value"
                    except:
                        pass

                html += f"<td class='{cell_class}'>{value}</td>"

            html += "</tr>"

        html += "</tbody></table>"

        return html

