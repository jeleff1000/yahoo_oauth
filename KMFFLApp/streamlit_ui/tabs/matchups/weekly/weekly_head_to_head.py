import pandas as pd
import streamlit as st
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class WeeklyHeadToHeadViewer:
    """
    Enhanced head-to-head viewer with multiple stat views and themed HTML tables.
    """

    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced head-to-head matchup matrix with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>🤝 Head-to-Head Matrix</h2>
        <p>Compare manager performance head-to-head across multiple statistical views</p>
        </div>
        """, unsafe_allow_html=True)

        if self.df is None or self.df.empty:
            st.info("No data available for head-to-head view")
            return

        required_columns = ['manager', 'opponent', 'win', 'loss', 'team_points', 'opponent_points', 'margin']
        if not all(column in self.df.columns for column in required_columns):
            st.error("❌ Required columns are missing from the data source")
            return

        # Display viewing count
        total_matchups = len(self.df)
        unique_matchups = len(self.df.groupby(['manager', 'opponent']))
        st.markdown(f"**Viewing {total_matchups:,} total matchups ({unique_matchups:,} unique pairings)**")

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
                    "Avg Points Per Game",
                    "Total Margin",
                    "Avg Margin Per Game",
                    "Highest Score",
                    "Lowest Score",
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
        elif viewer_type == "Avg Points Per Game":
            self.display_per_game()
        elif viewer_type == "Total Margin":
            self.display_margin()
        elif viewer_type == "Avg Margin Per Game":
            self.display_avg_margin()
        elif viewer_type == "Highest Score":
            self.display_highest_score()
        elif viewer_type == "Lowest Score":
            self.display_lowest_score()
        elif viewer_type == "Most Recent Result":
            self.display_most_recent()
        elif viewer_type == "Head-to-Head Streak":
            self.display_streak()

    @st.fragment
    def display_record(self):
        """Display W-L record in each matchup"""
        st.subheader("Win-Loss Records")

        # Create pivot tables for wins and losses
        wins = self.df.pivot_table(
            index='manager', columns='opponent', values='win', aggfunc='sum', fill_value=0
        )
        losses = self.df.pivot_table(
            index='manager', columns='opponent', values='loss', aggfunc='sum', fill_value=0
        )

        # Combine into W-L format
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

        # Style the HTML table
        html = self._style_table(record_table, "Win-Loss Record", is_record=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Rows show each manager's record against opponents (columns). Green cells indicate winning records, red indicates losing records.")

        # Add download button
        self._add_download_button(record_table, "h2h_record")

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

        st.caption("💡 **How to read:** Percentage of games won by row manager against column opponent. Higher is better.")

        self._add_download_button(win_pct_table, "h2h_win_pct")

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

        st.caption("💡 **How to read:** Total points scored by row manager in all games against column opponent.")

        self._add_download_button(points_table, "h2h_total_pts")

    @st.fragment
    def display_per_game(self):
        """Display average points per game"""
        st.subheader("Average Points Per Game")

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

        html = self._style_table(avg_table, "Avg Points Per Game", is_numeric=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Average points per game scored by row manager against column opponent.")

        self._add_download_button(avg_table, "h2h_avg_pts")

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

        self._add_download_button(margin_table, "h2h_total_margin")

    @st.fragment
    def display_avg_margin(self):
        """Display average margin per game"""
        st.subheader("Average Margin Per Game")

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

        html = self._style_table(avg_margin_table, "Avg Margin Per Game", is_margin=True)
        st.markdown(html, unsafe_allow_html=True)

        st.caption("💡 **How to read:** Average point differential per game for row manager vs column opponent.")

        self._add_download_button(avg_margin_table, "h2h_avg_margin")

    @st.fragment
    def display_highest_score(self):
        """Display highest score in matchups against each opponent"""
        st.subheader("Highest Score Against Each Opponent")

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

        st.caption("💡 **How to read:** Highest single-game score by row manager against column opponent.")

        self._add_download_button(highest_table, "h2h_highest")

    @st.fragment
    def display_lowest_score(self):
        """Display lowest score in matchups against each opponent"""
        st.subheader("Lowest Score Against Each Opponent")

        # Filter out zero scores which might be data errors
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

        st.caption("💡 **How to read:** Lowest single-game score by row manager against column opponent.")

        self._add_download_button(lowest_table, "h2h_lowest")

    @st.fragment
    def display_most_recent(self):
        """Display most recent matchup result"""
        st.subheader("Most Recent Matchup Result")

        # Get most recent matchup for each manager-opponent pair
        df_sorted = self.df.sort_values(['year', 'week'], ascending=False)

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

        st.caption("💡 **How to read:** Most recent result for row manager vs column opponent. Format: W/L (score).")

        self._add_download_button(recent_table, "h2h_recent")

    @st.fragment
    def display_streak(self):
        """Display current head-to-head streak"""
        st.subheader("Current Head-to-Head Streak")

        df_sorted = self.df.sort_values(['manager', 'opponent', 'year', 'week'])

        managers = sorted(set(self.df['manager']).union(self.df['opponent']))
        streak_table = pd.DataFrame(index=managers, columns=managers)

        for manager in managers:
            for opponent in managers:
                if manager == opponent:
                    streak_table.at[manager, opponent] = "—"
                else:
                    matchups = df_sorted[(df_sorted['manager'] == manager) & (df_sorted['opponent'] == opponent)]
                    if not matchups.empty:
                        # Calculate current streak
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

        st.caption("💡 **How to read:** Current streak for row manager vs column opponent. W3 = 3-game win streak, L2 = 2-game losing streak.")

        self._add_download_button(streak_table, "h2h_streak")

    def _add_download_button(self, df, suffix):
        """Add download button for the current view"""
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**💾 Export Data**")
        with col2:
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"h2h_{suffix}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"download_{suffix}",
                use_container_width=True
            )

    def _style_table(self, df, title, is_record=False, is_percentage=False, is_numeric=False, is_margin=False, is_result=False, is_streak=False):
        """Create a styled HTML table with theme-aware colors"""

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
                background-color: #ffffff;
            }}
            .h2h-table tbody tr:nth-of-type(even) {{
                background-color: #f9f9f9;
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

            /* Dark mode support */
            @media (prefers-color-scheme: dark) {{
                .h2h-table tbody tr {{
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                }}
                .h2h-table tbody tr:nth-of-type(even) {{
                    background-color: #2a2a2a;
                }}
                .h2h-table tbody tr:hover {{
                    background-color: #3a3a3a;
                }}
                .h2h-table th,
                .h2h-table td {{
                    border: 1px solid #444;
                }}
                .winning-record {{
                    background-color: #1e4620 !important;
                    color: #90ee90;
                }}
                .losing-record {{
                    background-color: #4a1e1e !important;
                    color: #ff9999;
                }}
                .even-record {{
                    background-color: #4a4420 !important;
                    color: #ffeb99;
                }}
                .positive-margin {{
                    background-color: #1e4620 !important;
                    color: #90ee90;
                }}
                .negative-margin {{
                    background-color: #4a1e1e !important;
                    color: #ff9999;
                }}
                .high-value {{
                    background-color: #1e3a4a !important;
                    color: #99ccff;
                }}
                .win-streak {{
                    background-color: #1e4620 !important;
                    color: #90ee90;
                }}
                .loss-streak {{
                    background-color: #4a1e1e !important;
                    color: #ff9999;
                }}
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
                elif is_margin:
                    # Color code margins
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
                    # Highlight high values
                    try:
                        val = float(value)
                        # Find max for comparison
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
