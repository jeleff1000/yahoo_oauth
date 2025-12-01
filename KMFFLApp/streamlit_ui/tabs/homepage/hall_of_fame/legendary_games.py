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
        st.markdown("### 🔥 Extreme Scoring Games")

        scoring_tabs = st.tabs(["Regular Season", "Playoffs"])

        for tab_idx, tab in enumerate(scoring_tabs):
            with tab:
                self._render_scoring_content(tab_idx)

    def _render_scoring_content(self, tab_idx):
        """Render scoring content for given tab index."""
        is_playoff = tab_idx
        game_type = "Playoffs" if is_playoff else "Regular Season"

        try:
            # Query for highest scoring
            high_query = f"""
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
                WHERE rn = 1 AND is_playoffs = {is_playoff}
                ORDER BY combined_score DESC
                LIMIT 10
            """

            low_query = f"""
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
                WHERE rn = 1 AND is_playoffs = {is_playoff}
                ORDER BY combined_score ASC
                LIMIT 10
            """

            high_games = self.con.execute(high_query).fetchdf()
            low_games = self.con.execute(low_query).fetchdf()

            # Highest scoring section
            st.markdown(f"#### 🔥 Highest Scoring ({game_type})")

            if high_games is not None and not high_games.empty:
                # Narrative callout
                top = high_games.iloc[0]
                st.markdown(narrative_callout(
                    f"The highest scoring {game_type.lower()} game ever: {top['winner']} vs {top['loser']} "
                    f"combined for {top['combined_score']:.1f} points in Week {int(top['week'])}, {int(top['year'])}!",
                    "🔥"
                ), unsafe_allow_html=True)

                for _, row in high_games.iterrows():
                    st.markdown(game_card(
                        winner=row['winner'],
                        loser=row['loser'],
                        winner_pts=row['winner_score'],
                        loser_pts=row['loser_score'],
                        year=int(row['year']),
                        week=int(row['week']),
                        is_playoff=(is_playoff == 1),
                        highlight_stat=f"{row['combined_score']:.1f} Total"
                    ), unsafe_allow_html=True)
            else:
                st.info(f"No {game_type.lower()} games found")

            # Lowest scoring section
            st.markdown("---")
            st.markdown(f"#### 🧊 Lowest Scoring ({game_type})")

            if low_games is not None and not low_games.empty:
                for _, row in low_games.iterrows():
                    st.markdown(game_card(
                        winner=row['winner'],
                        loser=row['loser'],
                        winner_pts=row['winner_score'],
                        loser_pts=row['loser_score'],
                        year=int(row['year']),
                        week=int(row['week']),
                        is_playoff=(is_playoff == 1),
                        highlight_stat=f"{row['combined_score']:.1f} Total"
                    ), unsafe_allow_html=True)
            else:
                st.info(f"No {game_type.lower()} games found")

        except Exception as e:
            st.error(f"Error loading highest scoring games: {e}")
            import traceback
            st.code(traceback.format_exc())

    @st.fragment
    def _display_closest_games(self):
        st.markdown("### 😱 Nail-Biters: Closest Games Ever")

        closest_tabs = st.tabs(["Regular Season", "Playoffs"])

        for tab_idx, tab in enumerate(closest_tabs):
            with tab:
                self._render_closest_content(tab_idx)

    def _render_closest_content(self, tab_idx):
        """Render closest games content for given tab index."""
        is_playoff = tab_idx
        game_type = "Playoffs" if is_playoff else "Regular Season"

        try:
            query = f"""
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
                WHERE rn = 1 AND is_playoffs = {is_playoff}
                ORDER BY margin ASC
                LIMIT 10
            """

            games = self.con.execute(query).fetchdf()

            if not games.empty:
                # Narrative callout
                top = games.iloc[0]
                st.markdown(narrative_callout(
                    f"The closest {game_type.lower()} game ever: {top['winner']} beat {top['loser']} by just "
                    f"{top['margin']:.2f} points in Week {int(top['week'])}, {int(top['year'])}!",
                    "😱"
                ), unsafe_allow_html=True)

                for _, row in games.iterrows():
                    st.markdown(game_card(
                        winner=row['winner'],
                        loser=row['loser'],
                        winner_pts=row['winner_score'],
                        loser_pts=row['loser_score'],
                        year=int(row['year']),
                        week=int(row['week']),
                        is_playoff=(is_playoff == 1),
                        highlight_stat=f"±{row['margin']:.2f}"
                    ), unsafe_allow_html=True)
            else:
                st.info(f"No {game_type.lower()} games found")

        except Exception as e:
            st.error(f"Error loading closest games: {e}")
            import traceback
            st.code(traceback.format_exc())

    @st.fragment
    def _display_upsets(self):
        st.markdown("### 🎯 Greatest Upsets")

        if 'manager_proj_score' not in self.df.columns or 'opponent_proj_score' not in self.df.columns:
            st.warning("Upset detection requires projected points data")
            return

        upsets_tabs = st.tabs(["Regular Season", "Playoffs"])

        for tab_idx, tab in enumerate(upsets_tabs):
            with tab:
                self._render_upsets_content(tab_idx)

    def _render_upsets_content(self, tab_idx):
        """Render upsets content for given tab index."""
        is_playoff = tab_idx
        game_type = "Playoffs" if is_playoff else "Regular Season"

        try:
            query = f"""
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
                        AND is_playoffs = {is_playoff}
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
                LIMIT 10
            """

            upsets = self.con.execute(query).fetchdf()

            if not upsets.empty:
                # Narrative callout
                top = upsets.iloc[0]
                st.markdown(narrative_callout(
                    f"Biggest {game_type.lower()} upset: {top['underdog']} was projected to lose by {top['proj_diff']:.1f} points "
                    f"to {top['favored']} in Week {int(top['week'])}, {int(top['year'])} - but pulled off the win!",
                    "🎯"
                ), unsafe_allow_html=True)

                for _, row in upsets.iterrows():
                    st.markdown(upset_card(
                        favored=row['favored'],
                        underdog=row['underdog'],
                        favored_pts=row['favored_actual'],
                        underdog_pts=row['underdog_actual'],
                        favored_proj=row['favored_proj'],
                        underdog_proj=row['underdog_proj'],
                        year=int(row['year']),
                        week=int(row['week']),
                        proj_diff=row['proj_diff'],
                        is_playoff=(is_playoff == 1)
                    ), unsafe_allow_html=True)
            else:
                st.info(f"No major {game_type.lower()} upsets found (requiring 5+ point projection difference)")

        except Exception as e:
            st.error(f"Error loading upsets: {e}")
            import traceback
            st.code(traceback.format_exc())

    @st.fragment
    def _display_rivalries(self):
        st.markdown("### ⚔️ Classic Rivalry Matchups")

        rivalry_tabs = st.tabs(["Regular Season", "Playoffs"])

        for tab_idx, tab in enumerate(rivalry_tabs):
            with tab:
                self._render_rivalries_content(tab_idx)

    def _render_rivalries_content(self, tab_idx):
        """Render rivalries content for given tab index."""
        is_playoff = tab_idx
        game_type = "Playoffs" if is_playoff else "Regular Season"

        try:
            query = f"""
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
                    WHERE rn = 1 AND is_playoffs = {is_playoff}
                    GROUP BY team_a, team_b, is_playoffs
                    HAVING COUNT(*) >= 2
                )
                SELECT team_a, team_b, total_games, team_a_wins,
                       total_games - team_a_wins as team_b_wins,
                       avg_combined_score, is_playoffs
                FROM rivalry_stats
                ORDER BY total_games DESC, avg_combined_score DESC
                LIMIT 10
            """

            rivalries = self.con.execute(query).fetchdf()

            if not rivalries.empty:
                # Narrative callout for top rivalry
                top = rivalries.iloc[0]
                st.markdown(narrative_callout(
                    f"The most frequent {game_type.lower()} matchup: {top['team_a']} vs {top['team_b']} "
                    f"have played {int(top['total_games'])} times! Current record: {int(top['team_a_wins'])}-{int(top['team_b_wins'])}",
                    "⚔️"
                ), unsafe_allow_html=True)

                for _, row in rivalries.iterrows():
                    st.markdown(rivalry_card(
                        team_a=row['team_a'],
                        team_b=row['team_b'],
                        team_a_wins=int(row['team_a_wins']),
                        team_b_wins=int(row['team_b_wins']),
                        total_games=int(row['total_games']),
                        avg_combined=row['avg_combined_score'],
                        is_playoff=(is_playoff == 1)
                    ), unsafe_allow_html=True)
            else:
                st.info(f"No {game_type.lower()} rivalries found (min 2 games)")

        except Exception as e:
            st.error(f"Error loading rivalries: {e}")
            import traceback
            st.code(traceback.format_exc())
