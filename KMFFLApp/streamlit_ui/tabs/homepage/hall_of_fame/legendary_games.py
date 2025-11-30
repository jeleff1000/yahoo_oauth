# tabs/hall_of_fame/legendary_games.py
import streamlit as st
import duckdb


class LegendaryGamesViewer:
    def __init__(self, df):
        self.df = df
        # Ensure commonly-referenced columns exist so DuckDB queries don't fail
        if self.df is not None:
            # provide sensible defaults when columns are missing
            if 'is_consolation' not in self.df.columns:
                self.df['is_consolation'] = 0
            if 'manager_proj_score' not in self.df.columns:
                self.df['manager_proj_score'] = None
            if 'opponent_proj_score' not in self.df.columns:
                self.df['opponent_proj_score'] = None
            if 'win' not in self.df.columns:
                self.df['win'] = None
        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("matchups", self.df)

    @st.fragment
    def display(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-orange'>
                <h2>🎮 Legendary Games</h2>
                <p>Unforgettable matchups, historic showdowns, and epic battles</p>
            </div>
        """, unsafe_allow_html=True)

        if self.df is None or self.df.empty:
            st.info("📊 No matchup data available")
            return

        # Create subtabs
        game_tabs = st.tabs([
            "🔥 Extreme Scoring",
            "😱 Closest Games",
            "🎯 Upsets",
            "⚔️ Classic Rivalries"
        ])

        with game_tabs[0]:
            self._display_highest_scoring()

        with game_tabs[1]:
            self._display_closest_games()

        with game_tabs[2]:
            self._display_upsets()

        with game_tabs[3]:
            self._display_rivalries()

    @st.fragment
    def _display_highest_scoring(self):
        st.markdown("### 🔥 Extreme Scoring Games")

        try:
            # Base dedupe query (used as a CTE in each per-category query)
            base_cte = """
                WITH unique_games AS (
                    SELECT
                        CAST(year AS INT) as year,
                        CAST(week AS INT) as week,
                        manager,
                        opponent,
                        CAST(team_points AS DOUBLE) as team_pts,
                        CAST(opponent_points AS DOUBLE) as opp_pts,
                        CAST(is_playoffs AS INT) as is_playoffs,
                        CASE
                            WHEN manager < opponent THEN manager || '|' || opponent
                            ELSE opponent || '|' || manager
                        END as match_key
                    FROM matchups
                    WHERE COALESCE(is_consolation, 0) = 0
                ),
                deduped AS (
                    SELECT *,
                        team_pts + opp_pts as combined_score,
                        CASE WHEN team_pts > opp_pts THEN manager ELSE opponent END as winner,
                        CASE WHEN team_pts > opp_pts THEN opponent ELSE manager END as loser,
                        CASE WHEN team_pts > opp_pts THEN team_pts ELSE opp_pts END as winner_score,
                        CASE WHEN team_pts > opp_pts THEN opp_pts ELSE team_pts END as loser_score,
                        ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                    FROM unique_games
                )
                SELECT year, week, winner, loser, winner_score, loser_score, combined_score, is_playoffs
                FROM deduped
                WHERE rn = 1
            """

            # Per-category queries (limit 5 each)
            # Append additional filter using AND because base_cte ends with WHERE rn = 1
            high_reg_query = base_cte + "\nAND is_playoffs = 0\nORDER BY combined_score DESC\nLIMIT 5"
            high_playoff_query = base_cte + "\nAND is_playoffs = 1\nORDER BY combined_score DESC\nLIMIT 5"
            low_reg_query = base_cte + "\nAND is_playoffs = 0\nORDER BY combined_score ASC\nLIMIT 5"
            low_playoff_query = base_cte + "\nAND is_playoffs = 1\nORDER BY combined_score ASC\nLIMIT 5"

            # Execute queries (use try/except separately to avoid one failure blocking others)
            high_reg = self.con.execute(high_reg_query).fetchdf()
            high_playoff = self.con.execute(high_playoff_query).fetchdf()
            low_reg = self.con.execute(low_reg_query).fetchdf()
            low_playoff = self.con.execute(low_playoff_query).fetchdf()

            # Display highest / extreme scoring
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏈 Regular Season")
                if high_reg is not None and not high_reg.empty:
                    for i, row in high_reg.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card'>
                                <div class='game-header'>
                                    <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                    <span class='game-stat' style='color: var(--success, #059669);'>{row['combined_score']:.1f} Total</span>
                                </div>
                                <div style='font-size: 0.95rem;'>
                                    <div style='margin-bottom: 0.2rem;'>
                                        <span class='team-name'>✅ {row['winner']}</span>
                                        <span style='float: right;' class='team-score'>{row['winner_score']:.1f}</span>
                                    </div>
                                    <div>
                                        <span class='loser-name'>❌ {row['loser']}</span>
                                        <span style='float: right;' class='loser-score'>{row['loser_score']:.1f}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No regular season games found")

            with col2:
                st.markdown("#### 🏆 Playoffs")
                if high_playoff is not None and not high_playoff.empty:
                    for i, row in high_playoff.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card hof-game-card-playoff'>
                                <div class='game-header'>
                                    <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                    <span class='game-stat' style='color: var(--success, #059669);'>{row['combined_score']:.1f} Total</span>
                                </div>
                                <div style='font-size: 0.95rem;'>
                                    <div style='margin-bottom: 0.2rem;'>
                                        <span class='team-name'>✅ {row['winner']}</span>
                                        <span style='float: right;' class='team-score'>{row['winner_score']:.1f}</span>
                                    </div>
                                    <div>
                                        <span class='loser-name'>❌ {row['loser']}</span>
                                        <span style='float: right;' class='loser-score'>{row['loser_score']:.1f}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No playoff games found")

            # Lowest scoring section (ice cube emoji)
            st.markdown("---")
            st.markdown("### 🧊 Lowest Scoring Games")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### 🏈 Regular Season")
                if low_reg is not None and not low_reg.empty:
                    for i, row in low_reg.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card'>
                                <div class='game-header'>
                                    <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                    <span class='game-stat' style='color: var(--success, #059669);'>{row['combined_score']:.1f} Total</span>
                                </div>
                                <div style='font-size: 0.95rem;'>
                                    <div style='margin-bottom: 0.2rem;'>
                                        <span class='team-name'>✅ {row['winner']}</span>
                                        <span style='float: right;' class='team-score'>{row['winner_score']:.1f}</span>
                                    </div>
                                    <div>
                                        <span class='loser-name'>❌ {row['loser']}</span>
                                        <span style='float: right;' class='loser-score'>{row['loser_score']:.1f}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No regular season games found")

            with col4:
                st.markdown("#### 🏆 Playoffs")
                if low_playoff is not None and not low_playoff.empty:
                    for i, row in low_playoff.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card hof-game-card-playoff'>
                                <div class='game-header'>
                                    <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                    <span class='game-stat' style='color: var(--success, #059669);'>{row['combined_score']:.1f} Total</span>
                                </div>
                                <div style='font-size: 0.95rem;'>
                                    <div style='margin-bottom: 0.2rem;'>
                                        <span class='team-name'>✅ {row['winner']}</span>
                                        <span style='float: right;' class='team-score'>{row['winner_score']:.1f}</span>
                                    </div>
                                    <div>
                                        <span class='loser-name'>❌ {row['loser']}</span>
                                        <span style='float: right;' class='loser-score'>{row['loser_score']:.1f}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No playoff games found")

        except Exception as e:
            st.error(f"Error loading highest scoring games: {e}")
            import traceback
            st.code(traceback.format_exc())

    @st.fragment
    def _display_closest_games(self):
        st.markdown("### 😱 Nail-Biters: Closest Games Ever")

        try:
            query = """
                WITH unique_games AS (
                    SELECT
                        CAST(year AS INT) as year,
                        CAST(week AS INT) as week,
                        manager,
                        opponent,
                        CAST(team_points AS DOUBLE) as team_pts,
                        CAST(opponent_points AS DOUBLE) as opp_pts,
                        CAST(is_playoffs AS INT) as is_playoffs,
                        CASE
                            WHEN manager < opponent THEN manager || '|' || opponent
                            ELSE opponent || '|' || manager
                        END as match_key
                    FROM matchups
                    WHERE COALESCE(is_consolation, 0) = 0
                ),
                deduped AS (
                    SELECT *,
                        ABS(team_pts - opp_pts) as margin,
                        CASE WHEN team_pts > opp_pts THEN manager ELSE opponent END as winner,
                        CASE WHEN team_pts > opp_pts THEN opponent ELSE manager END as loser,
                        CASE WHEN team_pts > opp_pts THEN team_pts ELSE opp_pts END as winner_score,
                        CASE WHEN team_pts > opp_pts THEN opp_pts ELSE team_pts END as loser_score,
                        ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                    FROM unique_games
                )
                SELECT year, week, winner, loser, winner_score, loser_score, margin, is_playoffs
                FROM deduped
                WHERE rn = 1
                ORDER BY margin ASC
                LIMIT 30
            """

            games = self.con.execute(query).fetchdf()

            if not games.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🏈 Regular Season")
                    reg = games[games['is_playoffs'] == 0].head(10)
                    for i, row in reg.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card' style='border-left: 3px solid var(--error, #EF4444);'>
                                <div class='game-header'>
                                    <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                    <span class='game-stat' style='color: var(--error, #EF4444);'>±{row['margin']:.2f}</span>
                                </div>
                                <div style='font-size: 0.95rem;'>
                                    <div style='margin-bottom: 0.2rem;'>
                                        <span class='team-name'>✅ {row['winner']}</span>
                                        <span style='float: right;' class='team-score'>{row['winner_score']:.1f}</span>
                                    </div>
                                    <div>
                                        <span class='loser-name'>❌ {row['loser']}</span>
                                        <span style='float: right;' class='loser-score'>{row['loser_score']:.1f}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 🏆 Playoffs")
                    playoff = games[games['is_playoffs'] == 1].head(10)
                    if not playoff.empty:
                        for i, row in playoff.iterrows():
                            st.markdown(f"""
                                <div class='hof-game-card hof-game-card-playoff' style='border-left: 3px solid var(--error, #EF4444);'>
                                    <div class='game-header'>
                                        <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                        <span class='game-stat' style='color: var(--error, #EF4444);'>±{row['margin']:.2f}</span>
                                    </div>
                                    <div style='font-size: 0.95rem;'>
                                        <div style='margin-bottom: 0.2rem;'>
                                            <span class='team-name'>✅ {row['winner']}</span>
                                            <span style='float: right;' class='team-score'>{row['winner_score']:.1f}</span>
                                        </div>
                                        <div>
                                            <span class='loser-name'>❌ {row['loser']}</span>
                                            <span style='float: right;' class='loser-score'>{row['loser_score']:.1f}</span>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No playoff games in top 30")

        except Exception as e:
            st.error(f"Error loading closest games: {e}")
            import traceback
            st.code(traceback.format_exc())

    @st.fragment
    def _display_upsets(self):
        st.markdown("### 🎯 Greatest Upsets")

        if 'manager_proj_score' not in self.df.columns or 'opponent_proj_score' not in self.df.columns:
            st.warning("💡 Upset detection requires projected points data")
            return

        try:
            query = """
                WITH unique_games AS (
                    SELECT
                        CAST(year AS INT) as year,
                        CAST(week AS INT) as week,
                        manager,
                        opponent,
                        CAST(team_points AS DOUBLE) as team_pts,
                        CAST(opponent_points AS DOUBLE) as opp_pts,
                        CAST(manager_proj_score AS DOUBLE) as team_proj,
                        CAST(opponent_proj_score AS DOUBLE) as opp_proj,
                        CAST(is_playoffs AS INT) as is_playoffs,
                        CAST(win AS INT) as win,
                        CASE
                            WHEN manager < opponent THEN manager || '|' || opponent
                            ELSE opponent || '|' || manager
                        END as match_key
                    FROM matchups
                    WHERE COALESCE(is_consolation, 0) = 0
                        AND manager_proj_score IS NOT NULL
                        AND opponent_proj_score IS NOT NULL
                ),
                deduped AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                    FROM unique_games
                ),
                upsets AS (
                    SELECT *,
                        CASE WHEN team_proj > opp_proj THEN manager ELSE opponent END as favored,
                        CASE WHEN team_proj > opp_proj THEN opponent ELSE manager END as underdog,
                        CASE WHEN team_proj > opp_proj THEN team_proj ELSE opp_proj END as favored_proj,
                        CASE WHEN team_proj > opp_proj THEN opp_proj ELSE team_proj END as underdog_proj,
                        CASE WHEN team_proj > opp_proj THEN team_pts ELSE opp_pts END as favored_actual,
                        CASE WHEN team_proj > opp_proj THEN opp_pts ELSE team_pts END as underdog_actual,
                        ABS(team_proj - opp_proj) as proj_diff
                    FROM deduped
                    WHERE rn = 1
                        AND (
                            (team_proj > opp_proj AND team_pts < opp_pts) OR
                            (team_proj < opp_proj AND team_pts > opp_pts)
                        )
                        AND ABS(team_proj - opp_proj) > 5
                )
                SELECT year, week, favored, underdog, favored_proj, underdog_proj,
                       favored_actual, underdog_actual, proj_diff, is_playoffs
                FROM upsets
                ORDER BY proj_diff DESC
                LIMIT 30
            """

            upsets = self.con.execute(query).fetchdf()

            if not upsets.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🏈 Regular Season")
                    reg = upsets[upsets['is_playoffs'] == 0].head(10)
                    for i, row in reg.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card' style='border-left: 3px solid var(--accent, #8B5CF6);'>
                                <div class='game-header'>
                                    <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                    <span class='game-stat' style='color: var(--accent, #8B5CF6);'>Upset by {row['proj_diff']:.1f}</span>
                                </div>
                                <div style='font-size: 0.9rem;'>
                                    <div style='margin-bottom: 0.3rem; padding-bottom: 0.3rem; border-bottom: 1px solid rgba(255,255,255,0.2);'>
                                        <div class='loser-name' style='font-size: 0.8rem; margin-bottom: 0.2rem;'>Favored (Lost)</div>
                                        <span class='team-name'>{row['favored']}</span>
                                        <span style='float: right;'>
                                            <span class='loser-score' style='font-size: 0.85rem;'>Proj: {row['favored_proj']:.1f}</span>
                                            <span class='team-score' style='margin-left: 0.5rem;'>{row['favored_actual']:.1f}</span>
                                        </span>
                                    </div>
                                    <div>
                                        <div style='color: var(--success, #059669); font-size: 0.8rem; margin-bottom: 0.2rem;'>Underdog (Won)</div>
                                        <span class='team-name' style='color: var(--success, #059669);'>{row['underdog']}</span>
                                        <span style='float: right;'>
                                            <span class='loser-score' style='font-size: 0.85rem;'>Proj: {row['underdog_proj']:.1f}</span>
                                            <span style='margin-left: 0.5rem; color: var(--success, #059669); font-weight: bold;'>{row['underdog_actual']:.1f}</span>
                                        </span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 🏆 Playoffs")
                    playoff = upsets[upsets['is_playoffs'] == 1].head(10)
                    if not playoff.empty:
                        for i, row in playoff.iterrows():
                            st.markdown(f"""
                                <div class='hof-game-card hof-game-card-playoff' style='border-left: 3px solid var(--accent, #8B5CF6);'>
                                    <div class='game-header'>
                                        <span class='game-date'>{int(row['year'])} Week {int(row['week'])}</span>
                                        <span class='game-stat' style='color: var(--accent, #8B5CF6);'>Upset by {row['proj_diff']:.1f}</span>
                                    </div>
                                    <div style='font-size: 0.9rem;'>
                                        <div style='margin-bottom: 0.3rem; padding-bottom: 0.3rem; border-bottom: 1px solid rgba(255,255,255,0.2);'>
                                            <div class='loser-name' style='font-size: 0.8rem; margin-bottom: 0.2rem;'>Favored (Lost)</div>
                                            <span class='team-name'>{row['favored']}</span>
                                            <span style='float: right;'>
                                                <span class='loser-score' style='font-size: 0.85rem;'>Proj: {row['favored_proj']:.1f}</span>
                                                <span class='team-score' style='margin-left: 0.5rem;'>{row['favored_actual']:.1f}</span>
                                            </span>
                                        </div>
                                        <div>
                                            <div style='color: var(--success, #059669); font-size: 0.8rem; margin-bottom: 0.2rem;'>Underdog (Won)</div>
                                            <span class='team-name' style='color: var(--success, #059669);'>{row['underdog']}</span>
                                            <span style='float: right;'>
                                                <span class='loser-score' style='font-size: 0.85rem;'>Proj: {row['underdog_proj']:.1f}</span>
                                                <span style='margin-left: 0.5rem; color: var(--success, #059669); font-weight: bold;'>{row['underdog_actual']:.1f}</span>
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No playoff upsets in top 30")
            else:
                st.info("No major upsets found (requiring 5+ point projection difference)")

        except Exception as e:
            st.error(f"Error loading upsets: {e}")
            import traceback
            st.code(traceback.format_exc())

    @st.fragment
    def _display_rivalries(self):
        st.markdown("### ⚔️ Classic Rivalry Matchups")
        st.info("💡 Most frequently played matchups with win/loss records")

        try:
            query = """
                WITH unique_games AS (
                    SELECT
                        CAST(year AS INT) as year,
                        CAST(week AS INT) as week,
                        manager,
                        opponent,
                        CAST(team_points AS DOUBLE) as team_pts,
                        CAST(opponent_points AS DOUBLE) as opp_pts,
                        CAST(is_playoffs AS INT) as is_playoffs,
                        CASE
                            WHEN LOWER(manager) < LOWER(opponent) THEN LOWER(manager) || '|' || LOWER(opponent)
                            ELSE LOWER(opponent) || '|' || LOWER(manager)
                        END as match_key,
                        CASE
                            WHEN LOWER(manager) < LOWER(opponent) THEN manager
                            ELSE opponent
                        END as team_a,
                        CASE
                            WHEN LOWER(manager) < LOWER(opponent) THEN opponent
                            ELSE manager
                        END as team_b
                    FROM matchups
                    WHERE COALESCE(is_consolation, 0) = 0
                ),
                deduped AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                    FROM unique_games
                ),
                rivalry_stats AS (
                    SELECT
                        team_a,
                        team_b,
                        is_playoffs,
                        COUNT(*) as total_games,
                        SUM(CASE
                            WHEN (LOWER(manager) = LOWER(team_a) AND team_pts > opp_pts) OR
                                 (LOWER(opponent) = LOWER(team_a) AND opp_pts > team_pts)
                            THEN 1 ELSE 0 END) as team_a_wins,
                        ROUND(AVG(team_pts + opp_pts), 1) as avg_combined_score
                    FROM deduped
                    WHERE rn = 1
                    GROUP BY team_a, team_b, is_playoffs
                    HAVING COUNT(*) >= 2
                )
                SELECT team_a, team_b, total_games, team_a_wins,
                       total_games - team_a_wins as team_b_wins,
                       avg_combined_score, is_playoffs
                FROM rivalry_stats
                ORDER BY is_playoffs DESC, total_games DESC, avg_combined_score DESC
                LIMIT 30
            """

            rivalries = self.con.execute(query).fetchdf()

            if not rivalries.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🏈 Regular Season Rivalries")
                    reg = rivalries[rivalries['is_playoffs'] == 0].head(10)
                    if not reg.empty:
                        for i, row in reg.iterrows():
                            st.markdown(f"""
                                <div class='hof-rivalry-card'>
                                    <div class='rivalry-header'>
                                        <span class='rivalry-teams'>{row['team_a']} vs {row['team_b']}</span>
                                        <span class='rivalry-games'>{int(row['total_games'])} games</span>
                                    </div>
                                    <div class='rivalry-stats'>
                                        <span>{row['team_a']}: <b>{int(row['team_a_wins'])}</b></span>
                                        <span>{row['team_b']}: <b>{int(row['team_b_wins'])}</b></span>
                                        <span style='color: var(--text-muted, #a5f3fc);'>Avg: {row['avg_combined_score']:.1f}</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No regular season rivalries found (min 2 games)")

                with col2:
                    st.markdown("#### 🏆 Playoff Rivalries")
                    playoff = rivalries[rivalries['is_playoffs'] == 1].head(10)
                    if not playoff.empty:
                        for i, row in playoff.iterrows():
                            st.markdown(f"""
                                <div class='hof-rivalry-card hof-game-card-playoff'>
                                    <div class='rivalry-header'>
                                        <span class='rivalry-teams'>{row['team_a']} vs {row['team_b']}</span>
                                        <span class='rivalry-games'>{int(row['total_games'])} games</span>
                                    </div>
                                    <div class='rivalry-stats'>
                                        <span>{row['team_a']}: <b>{int(row['team_a_wins'])}</b></span>
                                        <span>{row['team_b']}: <b>{int(row['team_b_wins'])}</b></span>
                                        <span style='color: var(--text-muted, #a5f3fc);'>Avg: {row['avg_combined_score']:.1f}</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No playoff rivalries found (min 2 games)")

        except Exception as e:
            st.error(f"Error loading rivalries: {e}")
            import traceback
            st.code(traceback.format_exc())
