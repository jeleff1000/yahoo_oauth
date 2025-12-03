import streamlit as st
import pandas as pd
import duckdb
from .top_weeks import TopWeeksViewer


class TopTeamsViewer:
    def __init__(self, df):
        self.df = df
        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("matchups", self.df)

    @st.fragment
    def display(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-green'>
                <h2>⭐ Top Teams</h2>
                <p>The most dominant seasons and explosive weeks in league history</p>
            </div>
        """, unsafe_allow_html=True)

        if self.df is None or self.df.empty:
            st.info("📊 No data available")
            return

        # Create tabs
        tab1, tab2 = st.tabs(["🏆 Top Seasons", "⚡ Top Weeks"])

        with tab1:
            self.display_top_seasons()

        with tab2:
            top_weeks_viewer = TopWeeksViewer(self.df)
            top_weeks_viewer.display()

    @st.fragment
    def display_top_seasons(self):
        st.markdown("### 🏆 Greatest Seasons")

        # Filter controls
        with st.expander("🔍 Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                regular_season = st.checkbox("Regular Season", value=True, key="top_seasons_regular")
            with col2:
                playoffs = st.checkbox("Playoffs", value=False, key="top_seasons_playoffs")
            with col3:
                min_games = st.number_input("Min Games", min_value=1, value=10, key="top_seasons_min_games")

            col4, col5 = st.columns(2)
            with col4:
                years = ["All Years"] + sorted([str(y) for y in self.df['year'].unique().tolist()])
                selected_year = st.selectbox("Year", years, index=0, key="top_seasons_year")
            with col5:
                managers = ["All Managers"] + sorted(self.df['manager'].unique().tolist())
                selected_manager = st.selectbox("Manager", managers, index=0, key="top_seasons_manager")

        try:
            # Build SQL query
            where_conditions = []
            if regular_season and not playoffs:
                where_conditions.append("is_playoffs = 0 AND COALESCE(is_consolation, 0) = 0")
            elif playoffs and not regular_season:
                where_conditions.append("is_playoffs = 1")
            elif not regular_season and not playoffs:
                st.warning("Select at least one game type")
                return
            else:
                where_conditions.append("COALESCE(is_consolation, 0) = 0")

            if selected_year != "All Years":
                where_conditions.append(f"CAST(year AS VARCHAR) = '{selected_year}'")
            if selected_manager != "All Managers":
                where_conditions.append(f"manager = '{selected_manager}'")

            where_clause = " AND ".join(where_conditions)

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
                    HAVING COUNT(*) >= {min_games}
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
                LIMIT 25
            """

            top_seasons = self.con.execute(query).fetchdf()

            if not top_seasons.empty:
                # Display KPIs
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Season", f"{top_seasons.iloc[0]['total_points']:.1f} pts",
                              delta=f"{top_seasons.iloc[0]['manager']} ({int(top_seasons.iloc[0]['year'])})")
                with col2:
                    avg_ppg = top_seasons['ppg'].mean()
                    st.metric("Avg PPG (Top 25)", f"{avg_ppg:.1f}")
                with col3:
                    champs_in_top = len(top_seasons[top_seasons['is_champion'] == 1])
                    st.metric("Champions in Top 25", champs_in_top)
                with col4:
                    best_record = top_seasons.nlargest(1, 'total_wins')
                    st.metric("Best Record", f"{int(best_record.iloc[0]['total_wins'])}-{int(best_record.iloc[0]['total_losses'])}")

                st.markdown("<br>", unsafe_allow_html=True)

                # Display as modern cards for top 10, table for rest
                top_10 = top_seasons.head(10)

                st.markdown("#### 🥇 Top 10 Seasons")
                for i in range(0, len(top_10), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(top_10):
                            row = top_10.iloc[i + j]
                            rank = i + j + 1

                            # Medal emoji for top 3
                            if rank == 1:
                                medal = "🥇"
                                border_color = "#FFD700"
                            elif rank == 2:
                                medal = "🥈"
                                border_color = "#C0C0C0"
                            elif rank == 3:
                                medal = "🥉"
                                border_color = "#CD7F32"
                            else:
                                medal = f"#{rank}"
                                border_color = "#E5E7EB"

                            champ_badge = " 🏆" if row['is_champion'] == 1 else ""

                            with col:
                                st.markdown(f"""
                                    <div class='hof-season-card' style='border-left: 4px solid {border_color};'>
                                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                            <span class='season-rank'>{medal}</span>
                                            <span class='season-year'>{int(row['year'])}</span>
                                        </div>
                                        <div class='season-manager'>{row['manager']}{champ_badge}</div>
                                        <div style='display: flex; gap: 1rem; margin-top: 0.5rem;'>
                                            <div>
                                                <div class='season-stat-label'>Total Pts</div>
                                                <div class='season-stat-value stat-highlight'>{row['total_points']:.1f}</div>
                                            </div>
                                            <div>
                                                <div class='season-stat-label'>Record</div>
                                                <div class='season-stat-value'>{int(row['total_wins'])}-{int(row['total_losses'])}</div>
                                            </div>
                                            <div>
                                                <div class='season-stat-label'>PPG</div>
                                                <div class='season-stat-value'>{row['ppg']:.1f}</div>
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                # Show rest in table if applicable
                if len(top_seasons) > 10:
                    with st.expander(f"📊 View Ranks 11-{len(top_seasons)}"):
                        rest = top_seasons.iloc[10:].copy()
                        rest['year'] = rest['year'].astype(str)
                        rest.columns = ['Manager', 'Year', 'Total Points', 'Wins', 'Losses', 'PPG', 'Champion']
                        rest['Champion'] = rest['Champion'].map({1: '🏆', 0: ''})
                        st.dataframe(rest, use_container_width=True, hide_index=True)

            else:
                st.info("No data available for the selected filters")

        except Exception as e:
            st.error(f"Error loading season data: {e}")
            import traceback
            st.code(traceback.format_exc())