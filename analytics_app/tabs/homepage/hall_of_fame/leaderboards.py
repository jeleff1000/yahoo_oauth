"""
Leaderboards - Merged Top Teams + Top Players

Provides a unified view of:
- Career Leaders (all-time stats)
- Top Seasons (best single seasons)
- Top Weeks (highest single-week performances)
"""

import streamlit as st
import duckdb


class LeaderboardsViewer:
    """Unified leaderboards combining teams and player stats."""

    def __init__(self, df):
        self.df = df
        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("matchups", self.df)

    @st.fragment
    def display(self):
        st.markdown(
            """
            <div class='hof-gradient-header hof-header-blue'>
                <h2>👑 Leaderboards</h2>
                <p>All-time career leaders, legendary seasons, and explosive weeks</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if self.df is None or self.df.empty:
            st.info("📊 No data available")
            return

        tabs = st.tabs(["🏅 Career", "📈 Seasons", "⚡ Weeks"])

        with tabs[0]:
            self._display_career_leaders()
        with tabs[1]:
            self._display_top_seasons()
        with tabs[2]:
            self._display_top_weeks()

    @st.fragment
    def _display_career_leaders(self):
        st.markdown("### 🏅 All-Time Career Leaders")

        try:
            query = """
                SELECT
                    manager,
                    COUNT(*) as total_games,
                    SUM(CAST(win AS INT)) as total_wins,
                    SUM(CAST(team_points AS DOUBLE)) as total_points,
                    ROUND(AVG(CAST(team_points AS DOUBLE)), 2) as ppg,
                    ROUND(SUM(CAST(win AS INT)) * 100.0 / COUNT(*), 1) as win_pct,
                    SUM(CAST(champion AS INT)) as championships
                FROM matchups
                WHERE COALESCE(is_consolation, 0) = 0
                GROUP BY manager
                HAVING COUNT(*) >= 10
                ORDER BY total_points DESC
            """

            leaders = self.con.execute(query).fetchdf()

            if leaders.empty:
                st.info("No career data available")
                return

            # Top 3 compact
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for i, col in enumerate(cols):
                if i < len(leaders):
                    row = leaders.iloc[i]
                    rings = (
                        "🏆" * int(row["championships"])
                        if row["championships"] > 0
                        else ""
                    )
                    with col:
                        st.markdown(
                            f"""
                            <div style='text-align: center; padding: 0.5rem;'>
                                <div style='font-size: 1.1rem;'>{medals[i]} <b>{row['manager']}</b></div>
                                <div style='color: var(--success); font-weight: 700;'>{row['total_points']:,.0f} pts</div>
                                <div style='font-size: 0.75rem; color: var(--text-muted);'>{row['ppg']:.1f} PPG · {row['win_pct']:.0f}% {rings}</div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

            # Full leaderboard (Top 10)
            st.markdown("#### 📊 Full Career Leaderboard")
            display_df = leaders.head(10).copy()
            display_df.columns = [
                "Manager",
                "Games",
                "Wins",
                "Total Pts",
                "PPG",
                "Win %",
                "Titles",
            ]
            display_df["Titles"] = display_df["Titles"].apply(
                lambda x: "🏆 " * int(x) if x > 0 else "-"
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading career leaders: {e}")

    @st.fragment
    def _display_top_seasons(self):
        st.markdown("### 📈 Greatest Seasons")

        # Filter tabs
        season_tabs = st.tabs(["Regular Season", "Playoffs", "All Games"])

        for tab_idx, tab in enumerate(season_tabs):
            with tab:
                self._render_seasons_content(tab_idx)

    def _render_seasons_content(self, tab_idx):
        """Render seasons content for given tab index."""
        try:
            # Build WHERE clause based on tab
            if tab_idx == 0:  # Regular Season
                where_clause = "is_playoffs = 0 AND COALESCE(is_consolation, 0) = 0"
            elif tab_idx == 1:  # Playoffs
                where_clause = "is_playoffs = 1"
            else:  # All Games
                where_clause = "COALESCE(is_consolation, 0) = 0"

            query = f"""
                WITH season_stats AS (
                    SELECT
                        manager,
                        CAST(year AS INT) as year,
                        SUM(CAST(team_points AS DOUBLE)) as total_points,
                        SUM(CAST(win AS INT)) as total_wins,
                        COUNT(*) as games,
                        ROUND(AVG(CAST(team_points AS DOUBLE)), 2) as ppg,
                        MAX(CAST(champion AS INT)) as is_champion
                    FROM matchups
                    WHERE {where_clause}
                    GROUP BY manager, year
                    HAVING COUNT(*) >= 5
                )
                SELECT
                    manager,
                    year,
                    total_points,
                    total_wins,
                    games - total_wins as total_losses,
                    ppg,
                    is_champion
                FROM season_stats
                ORDER BY total_points DESC
                LIMIT 10
            """

            top_seasons = self.con.execute(query).fetchdf()

            if top_seasons.empty:
                st.info("No season data available")
                return

            # Display as table
            display_df = top_seasons.copy()
            display_df["year"] = display_df["year"].astype(str)
            display_df.columns = [
                "Manager",
                "Year",
                "Total Points",
                "Wins",
                "Losses",
                "PPG",
                "Champion",
            ]
            display_df["Champion"] = display_df["Champion"].map({1: "🏆", 0: ""})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading season data: {e}")

    @st.fragment
    def _display_top_weeks(self):
        st.markdown("### ⚡ Highest Single-Week Performances")

        try:

            def get_top_weeks(is_playoff):
                query = f"""
                    SELECT
                        manager,
                        CAST(year AS INT) as year,
                        CAST(week AS INT) as week,
                        CAST(team_points AS DOUBLE) as points,
                        CASE WHEN win = 1 THEN 'W' ELSE 'L' END as result
                    FROM matchups
                    WHERE COALESCE(is_consolation, 0) = 0
                        AND CAST(is_playoffs AS INT) = {is_playoff}
                    ORDER BY team_points DESC
                    LIMIT 10
                """
                return self.con.execute(query).fetchdf()

            reg_weeks = get_top_weeks(0)
            playoff_weeks = get_top_weeks(1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Regular Season")
                if not reg_weeks.empty:
                    display_df = reg_weeks.copy()
                    display_df.columns = ["Manager", "Year", "Wk", "Points", "W/L"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No regular season data")

            with col2:
                st.markdown("#### Playoffs")
                if not playoff_weeks.empty:
                    display_df = playoff_weeks.copy()
                    display_df.columns = ["Manager", "Year", "Wk", "Points", "W/L"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No playoff data")

        except Exception as e:
            st.error(f"Error loading top weeks: {e}")
