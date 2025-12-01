# tabs/hall_of_fame/records.py
"""
Records - All-Time League Records & Achievements

Consolidated view of:
- All-Time Records (team + scoring)
- Streaks (win/loss streaks)
- Tough Luck (high-scoring losses)
"""

import streamlit as st
import pandas as pd
import duckdb

from .components import record_card, narrative_callout


class RecordsViewer:
    def __init__(self, df):
        self.df = df
        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("matchups", self.df)

    @st.fragment
    def display(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-purple'>
                <h2>📊 League Records</h2>
                <p>All-time records, streaks, and memorable moments</p>
            </div>
        """, unsafe_allow_html=True)

        if self.df is None or self.df.empty:
            st.info("📊 No data available")
            return

        tabs = st.tabs(["🏆 All-Time", "🎯 Streaks", "📉 Tough Luck"])

        with tabs[0]:
            self._display_all_time_records()
        with tabs[1]:
            self._display_streaks()
        with tabs[2]:
            self._display_tough_luck()

    @st.fragment
    def _display_all_time_records(self):
        """Consolidated All-Time Records combining team + scoring records."""
        st.markdown("### 🏆 All-Time Records")

        try:
            # Get key records using DuckDB
            highest_week_query = """
                SELECT manager, year, week, team_points
                FROM matchups
                ORDER BY team_points DESC
                LIMIT 1
            """
            highest_week = self.con.execute(highest_week_query).fetchdf()

            lowest_week_query = """
                SELECT manager, year, week, team_points
                FROM matchups
                WHERE team_points > 0
                ORDER BY team_points ASC
                LIMIT 1
            """
            lowest_week = self.con.execute(lowest_week_query).fetchdf()

            best_season_query = """
                SELECT
                    manager,
                    CAST(year AS INT) as year,
                    SUM(CAST(win AS INT)) as wins,
                    ROUND(AVG(CAST(team_points AS DOUBLE)), 2) as ppg
                FROM matchups
                WHERE COALESCE(is_consolation, 0) = 0
                GROUP BY manager, year
                HAVING COUNT(*) >= 5
                ORDER BY ppg DESC
                LIMIT 1
            """
            best_season = self.con.execute(best_season_query).fetchdf()

            biggest_blowout_query = """
                SELECT
                    manager as winner,
                    opponent as loser,
                    team_points as winner_pts,
                    opponent_points as loser_pts,
                    year, week,
                    CAST(team_points AS DOUBLE) - CAST(opponent_points AS DOUBLE) as margin
                FROM matchups
                WHERE win = 1
                ORDER BY margin DESC
                LIMIT 1
            """
            biggest_blowout = self.con.execute(biggest_blowout_query).fetchdf()

            # Display record cards in a 2x2 grid
            col1, col2 = st.columns(2)

            with col1:
                if not highest_week.empty:
                    row = highest_week.iloc[0]
                    st.markdown(record_card(
                        title="🔥 Highest Single Week",
                        holder=row['manager'],
                        value=f"{row['team_points']:.1f} pts",
                        year=int(row['year']),
                        context=f"Week {int(row['week'])}"
                    ), unsafe_allow_html=True)

                if not best_season.empty:
                    row = best_season.iloc[0]
                    st.markdown(record_card(
                        title="📈 Best Season PPG",
                        holder=row['manager'],
                        value=f"{row['ppg']:.2f} ppg",
                        year=int(row['year']),
                        context=f"{int(row['wins'])} wins"
                    ), unsafe_allow_html=True)

            with col2:
                if not lowest_week.empty:
                    row = lowest_week.iloc[0]
                    st.markdown(record_card(
                        title="❄️ Lowest Single Week",
                        holder=row['manager'],
                        value=f"{row['team_points']:.1f} pts",
                        year=int(row['year']),
                        context=f"Week {int(row['week'])}"
                    ), unsafe_allow_html=True)

                if not biggest_blowout.empty:
                    row = biggest_blowout.iloc[0]
                    st.markdown(record_card(
                        title="💥 Biggest Blowout",
                        holder=row['winner'],
                        value=f"+{row['margin']:.1f} margin",
                        year=int(row['year']),
                        context=f"vs {row['loser']}"
                    ), unsafe_allow_html=True)

            # Narrative callout
            if not highest_week.empty:
                hw = highest_week.iloc[0]
                st.markdown(narrative_callout(
                    f"{hw['manager']}'s {hw['team_points']:.1f}-point explosion in Week {int(hw['week'])}, "
                    f"{int(hw['year'])} remains the all-time single-week record!",
                    "🎯"
                ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # League Totals Section (folded from Miscellaneous)
            st.markdown("#### 📊 League Totals")

            totals_query = """
                SELECT
                    COUNT(DISTINCT year) as seasons,
                    COUNT(*) / 2 as total_games,
                    SUM(CAST(team_points AS DOUBLE)) as total_points,
                    ROUND(AVG(CAST(team_points AS DOUBLE)), 1) as avg_score
                FROM matchups
            """
            totals = self.con.execute(totals_query).fetchdf()

            if not totals.empty:
                row = totals.iloc[0]
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Seasons", int(row['seasons']))
                with cols[1]:
                    st.metric("Games", f"{int(row['total_games']):,}")
                with cols[2]:
                    st.metric("Total Points", f"{row['total_points']:,.0f}")
                with cols[3]:
                    st.metric("Avg Score", f"{row['avg_score']:.1f}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Scoring Milestones
            st.markdown("#### 🎯 Scoring Milestones")
            milestones = [(150, "150+"), (140, "140+"), (130, "130+"), (100, "100+")]
            milestone_cols = st.columns(4)

            for i, (threshold, label) in enumerate(milestones):
                with milestone_cols[i]:
                    count_query = f"""
                        SELECT COUNT(*) as cnt FROM matchups
                        WHERE CAST(team_points AS DOUBLE) >= {threshold}
                    """
                    count = self.con.execute(count_query).fetchdf().iloc[0]['cnt']
                    st.metric(f"{label} Weeks", int(count))

        except Exception as e:
            st.error(f"Error loading records: {e}")

    @st.fragment
    def _display_streaks(self):
        st.markdown("### 🎯 Win & Loss Streaks")

        try:
            # Calculate streaks for each manager using pandas (streak logic is complex)
            streak_data = []

            for manager in self.df['manager'].unique():
                manager_games = self.df[self.df['manager'] == manager].sort_values(['year', 'week'])

                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0
                current_type = None

                for _, game in manager_games.iterrows():
                    if game['win'] == 1:
                        if current_type == 'win':
                            current_streak += 1
                        else:
                            current_streak = 1
                            current_type = 'win'
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        if current_type == 'loss':
                            current_streak += 1
                        else:
                            current_streak = 1
                            current_type = 'loss'
                        max_loss_streak = max(max_loss_streak, current_streak)

                streak_data.append({
                    'manager': manager,
                    'win_streak': max_win_streak,
                    'loss_streak': max_loss_streak
                })

            streak_df = pd.DataFrame(streak_data)

            # Top record holders
            best_win = streak_df.nlargest(1, 'win_streak')
            worst_loss = streak_df.nlargest(1, 'loss_streak')

            col1, col2 = st.columns(2)

            with col1:
                if not best_win.empty:
                    row = best_win.iloc[0]
                    st.markdown(record_card(
                        title="🔥 Longest Win Streak",
                        holder=row['manager'],
                        value=f"{int(row['win_streak'])} games"
                    ), unsafe_allow_html=True)

            with col2:
                if not worst_loss.empty:
                    row = worst_loss.iloc[0]
                    st.markdown(record_card(
                        title="💀 Longest Loss Streak",
                        holder=row['manager'],
                        value=f"{int(row['loss_streak'])} games"
                    ), unsafe_allow_html=True)

            # Narrative
            if not best_win.empty:
                bw = best_win.iloc[0]
                st.markdown(narrative_callout(
                    f"{bw['manager']} holds the record for consecutive wins with an impressive "
                    f"{int(bw['win_streak'])}-game win streak!",
                    "🔥"
                ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Show both streak tables side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🏆 Win Streaks")
                win_streaks = streak_df.nlargest(10, 'win_streak')[['manager', 'win_streak']].copy()
                win_streaks.columns = ['Manager', 'Streak']
                st.dataframe(win_streaks, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("#### 📉 Loss Streaks")
                loss_streaks = streak_df.nlargest(10, 'loss_streak')[['manager', 'loss_streak']].copy()
                loss_streaks.columns = ['Manager', 'Streak']
                st.dataframe(loss_streaks, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error calculating streaks: {e}")

    @st.fragment
    def _display_tough_luck(self):
        st.markdown("### 📉 Tough Luck Records")

        st.markdown(narrative_callout(
            "Sometimes you put up a great score and still lose. These are the unluckiest losses in league history.",
            "😢"
        ), unsafe_allow_html=True)

        try:
            # Get highest scoring losses using DuckDB
            tough_luck_query = """
                SELECT
                    manager,
                    opponent,
                    CAST(team_points AS DOUBLE) as score,
                    CAST(opponent_points AS DOUBLE) as opp_score,
                    CAST(year AS INT) as year,
                    CAST(week AS INT) as week,
                    ROUND(CAST(opponent_points AS DOUBLE) - CAST(team_points AS DOUBLE), 1) as margin
                FROM matchups
                WHERE win = 0
                ORDER BY team_points DESC
                LIMIT 10
            """
            tough_losses = self.con.execute(tough_luck_query).fetchdf()

            if tough_losses.empty:
                st.info("No tough luck data available")
                return

            # Top record card
            top = tough_losses.iloc[0]
            st.markdown(record_card(
                title="😭 Unluckiest Loss Ever",
                holder=top['manager'],
                value=f"{top['score']:.1f} pts (L)",
                year=int(top['year']),
                context=f"Lost to {top['opponent']} by {top['margin']:.1f}"
            ), unsafe_allow_html=True)

            # Summary KPIs
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Highest Losing Score", f"{tough_losses['score'].max():.1f}")
            with col2:
                st.metric("Avg (Top 10)", f"{tough_losses['score'].mean():.1f}")
            with col3:
                # Count 120+ losses
                high_loss_query = """
                    SELECT COUNT(*) as cnt FROM matchups
                    WHERE win = 0 AND CAST(team_points AS DOUBLE) >= 120
                """
                high_losses = self.con.execute(high_loss_query).fetchdf().iloc[0]['cnt']
                st.metric("120+ Point Losses", int(high_losses))

            st.markdown("<br>", unsafe_allow_html=True)

            # Full table
            st.markdown("#### 😢 Top 10 Highest Scoring Losses")
            display_df = tough_losses.copy()
            display_df['year'] = display_df['year'].astype(str)
            display_df.columns = ['Manager', 'Opponent', 'Score', 'Opp Score', 'Year', 'Week', 'Lost By']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error calculating tough luck records: {e}")
