# tabs/hall_of_fame/top_players.py
import streamlit as st
import pandas as pd
import duckdb


class TopPlayersViewer:
    def __init__(self, df):
        self.df = df
        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("matchups", self.df)

    @st.fragment
    def display(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-blue'>
                <h2>👑 Top Players</h2>
                <p>The greatest managers and their legendary performances</p>
            </div>
        """, unsafe_allow_html=True)

        if self.df is None or self.df.empty:
            st.info("📊 No player data available")
            return

        # Create subtabs
        player_tabs = st.tabs([
            "🏅 Career Leaders",
            "📈 Single Season Bests",
            "⚡ Single Week Explosions",
            "🎯 Consistency Kings"
        ])

        with player_tabs[0]:
            self._display_career_leaders()

        with player_tabs[1]:
            self._display_season_bests()

        with player_tabs[2]:
            self._display_single_week()

        with player_tabs[3]:
            self._display_consistency()

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

            if not leaders.empty:
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
                                <div class='hof-leader-card' style='border: 3px solid {border};'>
                                    <div class='leader-medal'>{medal}</div>
                                    <div class='leader-name'>{row['manager']}</div>
                                    <div class='leader-score'>{row['total_points']:,.0f}</div>
                                    <div class='leader-label'>Total Points</div>
                                    <div class='leader-stats'>
                                        <div>
                                            <div class='stat-label'>PPG</div>
                                            <div class='stat-value'>{row['ppg']:.1f}</div>
                                        </div>
                                        <div>
                                            <div class='stat-label'>Win %</div>
                                            <div class='stat-value'>{row['win_pct']:.0f}%</div>
                                        </div>
                                        <div>
                                            <div class='stat-label'>Rings</div>
                                            <div class='stat-value'>{int(row['championships'])} 🏆</div>
                                        </div>
                                        <div>
                                            <div class='stat-label'>Games</div>
                                            <div class='stat-value'>{int(row['total_games'])}</div>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Full leaderboard
                st.markdown("#### 📊 Full Career Leaderboard")
                display_df = leaders.copy()
                display_df.columns = ['Manager', 'Games', 'Wins', 'Total Pts', 'PPG', 'Win %', 'Titles']
                display_df['Titles'] = display_df['Titles'].apply(lambda x: '🏆 ' * int(x) if x > 0 else '-')
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading career leaders: {e}")

    @st.fragment
    def _display_season_bests(self):
        st.markdown("### 📈 Best Single Seasons")

        try:
            query = """
                SELECT
                    manager,
                    CAST(year AS INT) as year,
                    SUM(CAST(team_points AS DOUBLE)) as total_points,
                    SUM(CAST(win AS INT)) as wins,
                    COUNT(*) as games,
                    ROUND(AVG(CAST(team_points AS DOUBLE)), 2) as ppg,
                    MAX(CAST(team_points AS DOUBLE)) as best_week,
                    MAX(CAST(champion AS INT)) as champion
                FROM matchups
                WHERE COALESCE(is_consolation, 0) = 0
                GROUP BY manager, year
                ORDER BY total_points DESC
                LIMIT 15
            """

            seasons = self.con.execute(query).fetchdf()

            if not seasons.empty:
                for i, row in seasons.head(15).iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        champ_badge = " 🏆" if row['champion'] == 1 else ""
                        st.markdown(f"""
                            <div class='hof-timeline-card'>
                                <div class='timeline-year'>#{i+1} • {int(row['year'])}</div>
                                <div class='timeline-winner'>{row['manager']}{champ_badge}</div>
                                <div class='timeline-details' style='display: flex; gap: 1.5rem;'>
                                    <span><b>{row['total_points']:.1f}</b> pts</span>
                                    <span>{int(row['wins'])}-{int(row['games'] - row['wins'])}</span>
                                    <span>{row['ppg']:.1f} PPG</span>
                                    <span>Best: {row['best_week']:.1f}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading season bests: {e}")

    @st.fragment
    def _display_single_week(self):
        st.markdown("### ⚡ Highest Single-Week Performances")

        try:
            query = """
                SELECT
                    manager,
                    CAST(year AS INT) as year,
                    CAST(week AS INT) as week,
                    CAST(team_points AS DOUBLE) as points,
                    CASE WHEN win = 1 THEN 'W' ELSE 'L' END as result,
                    CAST(is_playoffs AS INT) as playoffs
                FROM matchups
                WHERE COALESCE(is_consolation, 0) = 0
                ORDER BY team_points DESC
                LIMIT 20
            """

            weeks = self.con.execute(query).fetchdf()

            if not weeks.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🔥 Regular Season")
                    reg = weeks[weeks['playoffs'] == 0].head(10)
                    for i, row in reg.iterrows():
                        result_color = "#059669" if row['result'] == 'W' else "#DC2626"
                        st.markdown(f"""
                            <div class='hof-week-card'>
                                <div class='week-info'>
                                    <div>
                                        <span class='week-manager'>{row['manager']}</span>
                                        <span class='week-meta'> • {int(row['year'])} Wk{int(row['week'])}</span>
                                    </div>
                                    <div>
                                        <span class='week-score'>{row['points']:.1f}</span>
                                        <span style='color: {result_color}; font-weight: bold; margin-left: 0.5rem;'>
                                            {row['result']}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 🏆 Playoffs")
                    playoff = weeks[weeks['playoffs'] == 1].head(10)
                    if not playoff.empty:
                        for i, row in playoff.iterrows():
                            result_color = "#059669" if row['result'] == 'W' else "#DC2626"
                            st.markdown(f"""
                                <div class='hof-week-card hof-week-card-playoff'>
                                    <div class='week-info'>
                                        <div>
                                            <span class='week-manager'>{row['manager']}</span>
                                            <span class='week-meta'> • {int(row['year'])} Wk{int(row['week'])}</span>
                                        </div>
                                        <div>
                                            <span class='week-score'>{row['points']:.1f}</span>
                                            <span style='color: {result_color}; font-weight: bold; margin-left: 0.5rem;'>
                                                {row['result']}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No playoff games in top 20")

        except Exception as e:
            st.error(f"Error loading single week performances: {e}")

    @st.fragment
    def _display_consistency(self):
        st.markdown("### 🎯 Consistency Leaders")
        st.info("💡 Managers with the most reliable week-to-week performances")

        try:
            # Calculate standard deviation and consistency metrics
            query = """
                WITH manager_stats AS (
                    SELECT
                        manager,
                        CAST(team_points AS DOUBLE) as points
                    FROM matchups
                    WHERE COALESCE(is_consolation, 0) = 0
                )
                SELECT
                    manager,
                    COUNT(*) as games,
                    ROUND(AVG(points), 2) as avg_points,
                    ROUND(STDDEV(points), 2) as std_dev,
                    ROUND(AVG(points) / NULLIF(STDDEV(points), 0), 2) as consistency_score
                FROM manager_stats
                GROUP BY manager
                HAVING COUNT(*) >= 20
                ORDER BY consistency_score DESC
            """

            consistency = self.con.execute(query).fetchdf()

            if not consistency.empty:
                st.markdown("#### 📊 Most Consistent Performers")
                st.caption("*Higher consistency score = more reliable week-to-week performance*")

                display_df = consistency.head(15).copy()
                display_df.columns = ['Manager', 'Games', 'Avg Pts', 'Std Dev', 'Consistency Score']
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Show insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    most_consistent = consistency.iloc[0]
                    st.metric("Most Consistent", most_consistent['manager'],
                              delta=f"Score: {most_consistent['consistency_score']:.2f}")
                with col2:
                    highest_avg = consistency.nlargest(1, 'avg_points').iloc[0]
                    st.metric("Highest Avg (Top 15)", highest_avg['manager'],
                              delta=f"{highest_avg['avg_points']:.1f} PPG")
                with col3:
                    lowest_dev = consistency.nsmallest(1, 'std_dev').iloc[0]
                    st.metric("Lowest Variance", lowest_dev['manager'],
                              delta=f"±{lowest_dev['std_dev']:.1f} pts")

        except Exception as e:
            st.error(f"Error loading consistency data: {e}")