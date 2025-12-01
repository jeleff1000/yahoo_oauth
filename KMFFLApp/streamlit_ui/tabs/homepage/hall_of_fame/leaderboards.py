"""
Leaderboards - Merged Top Teams + Top Players

Provides a unified view of:
- Career Leaders (all-time stats)
- Top Seasons (best single seasons)
- Top Weeks (highest single-week performances)
"""

import streamlit as st
import pandas as pd
import duckdb

from .components import leader_card, season_card, week_card, narrative_callout


class LeaderboardsViewer:
    """Unified leaderboards combining teams and player stats."""

    def __init__(self, df):
        self.df = df
        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("matchups", self.df)

    @st.fragment
    def display(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-blue'>
                <h2>👑 Leaderboards</h2>
                <p>All-time career leaders, legendary seasons, and explosive weeks</p>
            </div>
        """, unsafe_allow_html=True)

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

            # Top 3 with big cards
            st.markdown("#### 🏆 Top 3 All-Time")
            cols = st.columns(3)
            for i, col in enumerate(cols):
                if i < len(leaders):
                    row = leaders.iloc[i]
                    if i == 0:
                        border = "#FFD700"
                        medal = "🥇"
                    elif i == 1:
                        border = "#C0C0C0"
                        medal = "🥈"
                    else:
                        border = "#CD7F32"
                        medal = "🥉"

                    with col:
                        st.markdown(f"""
                            <div class='hof-leader-card' style='border: 2px solid {border};'>
                                <div class='leader-medal'>{medal}</div>
                                <div class='leader-name'>{row['manager']}</div>
                                <div class='leader-score'>{row['total_points']:,.0f}</div>
                                <div class='leader-label'>Total Points</div>
                                <div style='font-size: 0.7rem; margin-top: 0.25rem; color: var(--text-muted);'>
                                    PPG: <b>{row['ppg']:.1f}</b> · Win%: <b>{row['win_pct']:.0f}%</b><br>
                                    Rings: <b>{int(row['championships'])}</b> · Games: <b>{int(row['total_games'])}</b>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # Narrative callout
            if len(leaders) >= 1:
                top = leaders.iloc[0]
                gap = top['total_points'] - leaders.iloc[1]['total_points'] if len(leaders) > 1 else 0
                st.markdown(narrative_callout(
                    f"{top['manager']} leads all-time with {top['total_points']:,.0f} total points - "
                    f"a {gap:,.0f} point lead over the competition!",
                    "📊"
                ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Full leaderboard (Top 10)
            st.markdown("#### 📊 Full Career Leaderboard")
            display_df = leaders.head(10).copy()
            display_df.columns = ['Manager', 'Games', 'Wins', 'Total Pts', 'PPG', 'Win %', 'Titles']
            display_df['Titles'] = display_df['Titles'].apply(lambda x: '🏆 ' * int(x) if x > 0 else '-')
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

            # KPIs
            col1, col2, col3 = st.columns(3)
            with col1:
                best = top_seasons.iloc[0]
                st.metric("Best Season", f"{best['total_points']:.1f} pts",
                          delta=f"{best['manager']} ({int(best['year'])})")
            with col2:
                avg_ppg = top_seasons['ppg'].mean()
                st.metric("Avg PPG (Top 10)", f"{avg_ppg:.1f}")
            with col3:
                champs = len(top_seasons[top_seasons['is_champion'] == 1])
                st.metric("Champions in Top 10", champs)

            st.markdown("<br>", unsafe_allow_html=True)

            # Display top 3 as cards
            st.markdown("#### 🥇 Top 3 Seasons")
            cols = st.columns(3)
            for i, col in enumerate(cols):
                if i < len(top_seasons):
                    row = top_seasons.iloc[i]
                    with col:
                        st.markdown(season_card(
                            rank=i + 1,
                            manager=row['manager'],
                            year=int(row['year']),
                            total_points=row['total_points'],
                            wins=int(row['total_wins']),
                            losses=int(row['total_losses']),
                            ppg=row['ppg'],
                            is_champion=row['is_champion'] == 1
                        ), unsafe_allow_html=True)

            # Table for 4-10
            if len(top_seasons) > 3:
                st.markdown("#### 📋 Positions 4-10")
                rest = top_seasons.iloc[3:10].copy()
                rest['year'] = rest['year'].astype(str)
                rest.columns = ['Manager', 'Year', 'Total Points', 'Wins', 'Losses', 'PPG', 'Champion']
                rest['Champion'] = rest['Champion'].map({1: '🏆', 0: ''})
                st.dataframe(rest, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading season data: {e}")

    @st.fragment
    def _display_top_weeks(self):
        st.markdown("### ⚡ Highest Single-Week Performances")

        # Filter tabs
        week_tabs = st.tabs(["Regular Season", "Playoffs"])

        for tab_idx, tab in enumerate(week_tabs):
            with tab:
                self._render_weeks_content(tab_idx)

    def _render_weeks_content(self, tab_idx):
        """Render weeks content for given tab index."""
        try:
            is_playoff = tab_idx  # 0 = Regular Season, 1 = Playoffs
            game_type = "playoffs" if is_playoff else "regular season"

            query = f"""
                SELECT
                    manager,
                    CAST(year AS INT) as year,
                    CAST(week AS INT) as week,
                    CAST(team_points AS DOUBLE) as points,
                    CASE WHEN win = 1 THEN 'W' ELSE 'L' END as result,
                    opponent
                FROM matchups
                WHERE COALESCE(is_consolation, 0) = 0
                    AND CAST(is_playoffs AS INT) = {is_playoff}
                ORDER BY team_points DESC
                LIMIT 10
            """

            weeks = self.con.execute(query).fetchdf()

            if weeks.empty:
                st.info(f"No {game_type} weeks available")
                return

            # Narrative for top week
            top = weeks.iloc[0]
            result_text = "won" if top['result'] == 'W' else "lost"
            st.markdown(narrative_callout(
                f"The highest {game_type} score ever: {top['manager']} dropped {top['points']:.1f} points "
                f"in Week {int(top['week'])}, {int(top['year'])} and {result_text} against {top['opponent']}!",
                "🔥"
            ), unsafe_allow_html=True)

            # Display all 10 as cards
            for i, row in weeks.iterrows():
                st.markdown(week_card(
                    manager=row['manager'],
                    year=int(row['year']),
                    week=int(row['week']),
                    points=row['points'],
                    result=row['result'],
                    is_playoff=(is_playoff == 1)
                ), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading top weeks: {e}")
