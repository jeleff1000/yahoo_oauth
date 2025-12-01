# tabs/hall_of_fame/legendary_games.py
import streamlit as st
import duckdb

from .components import game_card, upset_card, rivalry_card, narrative_callout


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
        st.markdown("### 🔥 Highest Scoring Games")

        try:
            # Query for both regular season and playoffs
            def get_high_games(is_playoff):
                query = f"""
                    WITH unique_games AS (
                        SELECT
                            CAST(year AS INT) as year,
                            CAST(week AS INT) as week,
                            manager, opponent,
                            CAST(team_points AS DOUBLE) as team_pts,
                            CAST(opponent_points AS DOUBLE) as opp_pts,
                            CASE WHEN manager < opponent THEN manager || '|' || opponent
                                 ELSE opponent || '|' || manager END as match_key
                        FROM matchups
                        WHERE COALESCE(is_consolation, 0) = 0 AND CAST(is_playoffs AS INT) = {is_playoff}
                    ),
                    deduped AS (
                        SELECT *, team_pts + opp_pts as combined,
                            CASE WHEN team_pts > opp_pts THEN manager ELSE opponent END as winner,
                            CASE WHEN team_pts > opp_pts THEN opponent ELSE manager END as loser,
                            CASE WHEN team_pts > opp_pts THEN team_pts ELSE opp_pts END as win_pts,
                            CASE WHEN team_pts > opp_pts THEN opp_pts ELSE team_pts END as lose_pts,
                            ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                        FROM unique_games
                    )
                    SELECT year, week, winner, loser, win_pts, lose_pts, combined
                    FROM deduped WHERE rn = 1 ORDER BY combined DESC LIMIT 5
                """
                return self.con.execute(query).fetchdf()

            reg_games = get_high_games(0)
            playoff_games = get_high_games(1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Regular Season")
                if not reg_games.empty:
                    for _, row in reg_games.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card'>
                                <div class='game-header'><span class='game-date'>{int(row['year'])} Wk{int(row['week'])}</span>
                                <span class='game-stat' style='color: var(--success);'>{row['combined']:.1f}</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <div>✅ {row['winner']} <span style='float:right;'>{row['win_pts']:.1f}</span></div>
                                    <div style='color: var(--text-muted);'>❌ {row['loser']} <span style='float:right;'>{row['lose_pts']:.1f}</span></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No games found")

            with col2:
                st.markdown("#### Playoffs")
                if not playoff_games.empty:
                    for _, row in playoff_games.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card hof-game-card-playoff'>
                                <div class='game-header'><span class='game-date'>{int(row['year'])} Wk{int(row['week'])}</span>
                                <span class='game-stat' style='color: var(--success);'>{row['combined']:.1f}</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <div>✅ {row['winner']} <span style='float:right;'>{row['win_pts']:.1f}</span></div>
                                    <div style='color: var(--text-muted);'>❌ {row['loser']} <span style='float:right;'>{row['lose_pts']:.1f}</span></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No playoff games found")

        except Exception as e:
            st.error(f"Error loading highest scoring games: {e}")

    @st.fragment
    def _display_closest_games(self):
        st.markdown("### 😱 Closest Games")

        try:
            def get_closest_games(is_playoff):
                query = f"""
                    WITH unique_games AS (
                        SELECT CAST(year AS INT) as year, CAST(week AS INT) as week,
                            manager, opponent,
                            CAST(team_points AS DOUBLE) as team_pts,
                            CAST(opponent_points AS DOUBLE) as opp_pts,
                            CASE WHEN manager < opponent THEN manager || '|' || opponent
                                 ELSE opponent || '|' || manager END as match_key
                        FROM matchups
                        WHERE COALESCE(is_consolation, 0) = 0 AND CAST(is_playoffs AS INT) = {is_playoff}
                    ),
                    deduped AS (
                        SELECT *, ABS(team_pts - opp_pts) as margin,
                            CASE WHEN team_pts > opp_pts THEN manager ELSE opponent END as winner,
                            CASE WHEN team_pts > opp_pts THEN opponent ELSE manager END as loser,
                            CASE WHEN team_pts > opp_pts THEN team_pts ELSE opp_pts END as win_pts,
                            CASE WHEN team_pts > opp_pts THEN opp_pts ELSE team_pts END as lose_pts,
                            ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                        FROM unique_games
                    )
                    SELECT year, week, winner, loser, win_pts, lose_pts, margin
                    FROM deduped WHERE rn = 1 ORDER BY margin ASC LIMIT 5
                """
                return self.con.execute(query).fetchdf()

            reg_games = get_closest_games(0)
            playoff_games = get_closest_games(1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Regular Season")
                if not reg_games.empty:
                    for _, row in reg_games.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card'>
                                <div class='game-header'><span class='game-date'>{int(row['year'])} Wk{int(row['week'])}</span>
                                <span class='game-stat' style='color: var(--accent);'>±{row['margin']:.2f}</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <div>✅ {row['winner']} <span style='float:right;'>{row['win_pts']:.1f}</span></div>
                                    <div style='color: var(--text-muted);'>❌ {row['loser']} <span style='float:right;'>{row['lose_pts']:.1f}</span></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No games found")

            with col2:
                st.markdown("#### Playoffs")
                if not playoff_games.empty:
                    for _, row in playoff_games.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card hof-game-card-playoff'>
                                <div class='game-header'><span class='game-date'>{int(row['year'])} Wk{int(row['week'])}</span>
                                <span class='game-stat' style='color: var(--accent);'>±{row['margin']:.2f}</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <div>✅ {row['winner']} <span style='float:right;'>{row['win_pts']:.1f}</span></div>
                                    <div style='color: var(--text-muted);'>❌ {row['loser']} <span style='float:right;'>{row['lose_pts']:.1f}</span></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No playoff games found")

        except Exception as e:
            st.error(f"Error loading closest games: {e}")

    @st.fragment
    def _display_upsets(self):
        st.markdown("### 🎯 Greatest Upsets")

        if 'manager_proj_score' not in self.df.columns or 'opponent_proj_score' not in self.df.columns:
            st.warning("Upset detection requires projected points data")
            return

        try:
            def get_upsets(is_playoff):
                query = f"""
                    WITH unique_games AS (
                        SELECT CAST(year AS INT) as year, CAST(week AS INT) as week,
                            manager, opponent,
                            CAST(team_points AS DOUBLE) as team_pts,
                            CAST(opponent_points AS DOUBLE) as opp_pts,
                            CAST(manager_proj_score AS DOUBLE) as team_proj,
                            CAST(opponent_proj_score AS DOUBLE) as opp_proj,
                            CASE WHEN manager < opponent THEN manager || '|' || opponent
                                 ELSE opponent || '|' || manager END as match_key
                        FROM matchups
                        WHERE COALESCE(is_consolation, 0) = 0 AND CAST(is_playoffs AS INT) = {is_playoff}
                            AND manager_proj_score IS NOT NULL AND opponent_proj_score IS NOT NULL
                    ),
                    deduped AS (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                        FROM unique_games
                    ),
                    upsets AS (
                        SELECT *,
                            CASE WHEN team_proj > opp_proj THEN manager ELSE opponent END as favored,
                            CASE WHEN team_proj > opp_proj THEN opponent ELSE manager END as underdog,
                            CASE WHEN team_proj > opp_proj THEN opp_pts ELSE team_pts END as underdog_pts,
                            ABS(team_proj - opp_proj) as proj_diff
                        FROM deduped
                        WHERE rn = 1 AND ABS(team_proj - opp_proj) > 5
                            AND ((team_proj > opp_proj AND team_pts < opp_pts) OR (team_proj < opp_proj AND team_pts > opp_pts))
                    )
                    SELECT year, week, favored, underdog, underdog_pts, proj_diff
                    FROM upsets ORDER BY proj_diff DESC LIMIT 5
                """
                return self.con.execute(query).fetchdf()

            reg_upsets = get_upsets(0)
            playoff_upsets = get_upsets(1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Regular Season")
                if not reg_upsets.empty:
                    for _, row in reg_upsets.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card' style='border-left: 3px solid var(--accent);'>
                                <div class='game-header'><span class='game-date'>{int(row['year'])} Wk{int(row['week'])}</span>
                                <span class='game-stat' style='color: var(--accent);'>+{row['proj_diff']:.1f} upset</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <div style='color: var(--success);'>🏆 {row['underdog']} <span style='float:right;'>{row['underdog_pts']:.1f}</span></div>
                                    <div style='color: var(--text-muted);'>📉 {row['favored']} (favored)</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No major upsets found")

            with col2:
                st.markdown("#### Playoffs")
                if not playoff_upsets.empty:
                    for _, row in playoff_upsets.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card hof-game-card-playoff' style='border-left: 3px solid var(--accent);'>
                                <div class='game-header'><span class='game-date'>{int(row['year'])} Wk{int(row['week'])}</span>
                                <span class='game-stat' style='color: var(--accent);'>+{row['proj_diff']:.1f} upset</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <div style='color: var(--success);'>🏆 {row['underdog']} <span style='float:right;'>{row['underdog_pts']:.1f}</span></div>
                                    <div style='color: var(--text-muted);'>📉 {row['favored']} (favored)</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No playoff upsets found")

        except Exception as e:
            st.error(f"Error loading upsets: {e}")

    @st.fragment
    def _display_rivalries(self):
        st.markdown("### ⚔️ Top Rivalries")

        try:
            def get_rivalries(is_playoff):
                query = f"""
                    WITH unique_games AS (
                        SELECT CAST(year AS INT) as year, CAST(week AS INT) as week,
                            manager, opponent,
                            CAST(team_points AS DOUBLE) as team_pts,
                            CAST(opponent_points AS DOUBLE) as opp_pts,
                            CASE WHEN LOWER(manager) < LOWER(opponent) THEN manager ELSE opponent END as team_a,
                            CASE WHEN LOWER(manager) < LOWER(opponent) THEN opponent ELSE manager END as team_b,
                            CASE WHEN LOWER(manager) < LOWER(opponent) THEN LOWER(manager) || '|' || LOWER(opponent)
                                 ELSE LOWER(opponent) || '|' || LOWER(manager) END as match_key
                        FROM matchups
                        WHERE COALESCE(is_consolation, 0) = 0 AND CAST(is_playoffs AS INT) = {is_playoff}
                    ),
                    deduped AS (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY team_pts DESC) as rn
                        FROM unique_games
                    ),
                    rivalry_stats AS (
                        SELECT team_a, team_b, COUNT(*) as games,
                            SUM(CASE WHEN (LOWER(manager) = LOWER(team_a) AND team_pts > opp_pts) OR
                                         (LOWER(opponent) = LOWER(team_a) AND opp_pts > team_pts)
                                THEN 1 ELSE 0 END) as a_wins
                        FROM deduped WHERE rn = 1 GROUP BY team_a, team_b HAVING COUNT(*) >= 2
                    )
                    SELECT team_a, team_b, games, a_wins, games - a_wins as b_wins
                    FROM rivalry_stats ORDER BY games DESC LIMIT 5
                """
                return self.con.execute(query).fetchdf()

            reg_rivals = get_rivalries(0)
            playoff_rivals = get_rivalries(1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Regular Season")
                if not reg_rivals.empty:
                    for _, row in reg_rivals.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card'>
                                <div class='game-header'><span class='game-date'>{row['team_a']} vs {row['team_b']}</span>
                                <span class='game-stat'>{int(row['games'])} games</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <span>{row['team_a']}: <b>{int(row['a_wins'])}</b></span> ·
                                    <span>{row['team_b']}: <b>{int(row['b_wins'])}</b></span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No rivalries found")

            with col2:
                st.markdown("#### Playoffs")
                if not playoff_rivals.empty:
                    for _, row in playoff_rivals.iterrows():
                        st.markdown(f"""
                            <div class='hof-game-card hof-game-card-playoff'>
                                <div class='game-header'><span class='game-date'>{row['team_a']} vs {row['team_b']}</span>
                                <span class='game-stat'>{int(row['games'])} games</span></div>
                                <div style='font-size: 0.9rem;'>
                                    <span>{row['team_a']}: <b>{int(row['a_wins'])}</b></span> ·
                                    <span>{row['team_b']}: <b>{int(row['b_wins'])}</b></span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No playoff rivalries found")

        except Exception as e:
            st.error(f"Error loading rivalries: {e}")
