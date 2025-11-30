# tabs/hall_of_fame/records.py
import streamlit as st
import pandas as pd


class RecordsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self):
        st.markdown("""
            <div class="hof-records-header">
                <h2>📊 League Records</h2>
                <p>All-time records, milestones, and statistical achievements</p>
            </div>
        """, unsafe_allow_html=True)

        if self.df is None or self.df.empty:
            st.info("📊 No data available")
            return

        # Create subtabs
        record_tabs = st.tabs([
            "🏆 Team Records",
            "📈 Scoring Records",
            "🎯 Streaks",
            "📉 Tough Luck",
            "🎪 Miscellaneous"
        ])

        with record_tabs[0]:
            self._display_team_records()

        with record_tabs[1]:
            self._display_scoring_records()

        with record_tabs[2]:
            self._display_streaks()

        with record_tabs[3]:
            self._display_tough_luck()

        with record_tabs[4]:
            self._display_miscellaneous()

    @st.fragment
    def _display_team_records(self):
        st.markdown("### 🏆 Team Records")

        try:
            # Aggregate season-level stats and correctly count games per season
            season_records = self.df.groupby(['manager', 'year']).agg(
                win=('win', 'sum'),
                team_points=('team_points', 'sum'),
                games=('win', 'count')
            ).reset_index()

            # Compute per-season PPG (points per game)
            season_records['ppg'] = (season_records['team_points'] / season_records['games']).round(2)

            best_record = season_records.nlargest(1, 'win')
            worst_record = season_records.nsmallest(1, 'win')
            # For the "Most Points" card we want highest PPG instead of raw total points
            most_points = season_records.nlargest(1, 'ppg')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🥇 Best Single Season")
                if not best_record.empty:
                    # Ensure year prints without commas (render as int/string)
                    yr = best_record.iloc[0]['year']
                    st.write(f"**{best_record.iloc[0]['manager']}** ({int(yr) if pd.notna(yr) else yr})")
                    st.metric("Wins", int(best_record.iloc[0]['win']))

            with col2:
                # Show highest PPG (season) rather than raw total points
                st.markdown("#### 📈 Highest PPG (Season)")
                if not most_points.empty:
                    yr = most_points.iloc[0]['year']
                    st.write(f"**{most_points.iloc[0]['manager']}** ({int(yr) if pd.notna(yr) else yr})")
                    st.metric("PPG", f"{most_points.iloc[0]['ppg']:.2f}")

            with col3:
                st.markdown("#### 🎯 Highest Win %")
                if not season_records.empty:
                    # season_records already includes a correct 'games' column from the aggregation above
                    # Compute win% safely, round to 1 decimal, and clamp to 100% to guard against bad data
                    season_records['win_pct'] = (season_records['win'] / season_records['games'] * 100).round(1).clip(upper=100.0)
                    best_pct = season_records.nlargest(1, 'win_pct')
                    yr = best_pct.iloc[0]['year']
                    st.write(f"**{best_pct.iloc[0]['manager']}** ({int(yr) if pd.notna(yr) else yr})")
                    st.metric("Win %", f"{best_pct.iloc[0]['win_pct']:.1f}%")

            # All-time records table
            st.markdown("---")
            st.markdown("#### 📋 Top Single Seasons (by PPG)")
            # Sort by PPG and show PPG plus total points
            top_seasons = season_records.nlargest(10, 'ppg')[
                ['manager', 'year', 'win', 'games', 'team_points', 'ppg']
            ].copy()
            top_seasons.columns = ['Manager', 'Year', 'Wins', 'Games', 'Total Points', 'PPG']
            top_seasons['Total Points'] = top_seasons['Total Points'].round(1)
            top_seasons['PPG'] = top_seasons['PPG'].round(2)
            # Ensure Year is a string to avoid comma formatting
            top_seasons['Year'] = top_seasons['Year'].astype(str)
            st.dataframe(top_seasons[['Manager', 'Year', 'Wins', 'Games', 'PPG', 'Total Points']], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error calculating team records: {e}")

    @st.fragment
    def _display_scoring_records(self):
        st.markdown("### 📈 Scoring Records")

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔥 Highest Single Week")
                highest_week = self.df.nlargest(1, 'team_points')
                if not highest_week.empty:
                    row = highest_week.iloc[0]
                    st.write(f"**{row['manager']}**")
                    yr = row['year']
                    st.metric("Points", f"{row['team_points']:.1f}",
                              delta=f"Week {int(row['week'])}, {int(yr) if pd.notna(yr) else yr}")

                # Add a red down arrow to highlight the lowest single-week score
                st.markdown("#### ❄️ Lowest Single Week <span style='color:red'>🔻</span>", unsafe_allow_html=True)
                lowest_week = self.df.nsmallest(1, 'team_points')
                if not lowest_week.empty:
                    row = lowest_week.iloc[0]
                    st.write(f"**{row['manager']}**")
                    yr = row['year']
                    st.metric("Points", f"{row['team_points']:.1f}",
                              delta=f"Week {int(row['week'])}, {int(yr) if pd.notna(yr) else yr}")

            with col2:
                st.markdown("#### 📊 PPG Leaders (Career)")
                career_ppg = self.df.groupby('manager').agg({
                    'team_points': ['sum', 'count', 'mean']
                }).reset_index()
                career_ppg.columns = ['Manager', 'Total Points', 'Games', 'PPG']
                career_ppg = career_ppg[career_ppg['Games'] >= 10]  # Min 10 games
                career_ppg['PPG'] = career_ppg['PPG'].round(2)
                career_ppg = career_ppg.nlargest(5, 'PPG')
                st.dataframe(career_ppg[['Manager', 'PPG', 'Games']],
                             use_container_width=True, hide_index=True)

            # Weekly scoring distribution
            st.markdown("---")
            st.markdown("#### 🎯 Scoring Milestones")

            milestone_cols = st.columns(4)
            milestones = [
                (150, "150+ Point Weeks"),
                (140, "140+ Point Weeks"),
                (130, "130+ Point Weeks"),
                (100, "100+ Point Weeks")
            ]

            for i, (threshold, label) in enumerate(milestones):
                with milestone_cols[i]:
                    count = len(self.df[self.df['team_points'] >= threshold])
                    st.metric(label, count)

        except Exception as e:
            st.error(f"Error calculating scoring records: {e}")

    @st.fragment
    def _display_streaks(self):
        st.markdown("### 🎯 Streaks")

        try:
            # Calculate streaks for each manager
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
                    'Manager': manager,
                    'Longest Win Streak': max_win_streak,
                    'Longest Loss Streak': max_loss_streak
                })

            streak_df = pd.DataFrame(streak_data)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔥 Longest Winning Streaks")
                win_streaks = streak_df.nlargest(10, 'Longest Win Streak')
                st.dataframe(win_streaks[['Manager', 'Longest Win Streak']],
                             use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### 💀 Longest Losing Streaks")
                loss_streaks = streak_df.nlargest(10, 'Longest Loss Streak')
                st.dataframe(loss_streaks[['Manager', 'Longest Loss Streak']],
                             use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error calculating streaks: {e}")

    @st.fragment
    def _display_tough_luck(self):
        st.markdown("### 📉 Tough Luck Records")
        st.info("💡 Games where teams scored well but still lost")

        try:
            # High scoring losses
            losses = self.df[self.df['win'] == 0].copy()
            # Deduplicate matchups so we don't show the same game twice
            if 'opponent' in losses.columns:
                lv = losses.copy()
                lv['match_key'] = lv.apply(lambda r: '|'.join(sorted([str(r['manager']).lower(), str(r.get('opponent','')).lower()])), axis=1)
                lv = lv.sort_values(by=['year', 'week', 'match_key', 'team_points'], ascending=[True, True, True, False])
                lv = lv.drop_duplicates(subset=['year', 'week', 'match_key'], keep='first')
            else:
                lv = losses.copy()

            tough_losses = lv.nlargest(10, 'team_points')[['year', 'week', 'manager', 'opponent', 'team_points', 'opponent_points']].copy()
            tough_losses['margin'] = (tough_losses['opponent_points'] - tough_losses['team_points']).round(2)
            tough_losses = tough_losses.round(2)
            tough_losses.columns = ['Year', 'Week', 'Manager', 'Opponent', 'Score', 'Opp Score', 'Lost By']
            tough_losses['Year'] = tough_losses['Year'].astype(str)

            st.markdown("#### 😢 Highest Scoring Losses")
            st.dataframe(tough_losses, use_container_width=True, hide_index=True)

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Highest Losing Score", f"{tough_losses['Score'].max():.1f}")
            with col2:
                st.metric("Avg Top 20", f"{tough_losses['Score'].mean():.1f}")
            with col3:
                # Count unique games with team_points >= 120
                if 'opponent' in self.df.columns:
                    dfv = self.df.copy()
                    dfv['match_key'] = dfv.apply(lambda r: '|'.join(sorted([str(r['manager']).lower(), str(r.get('opponent','')).lower()])), axis=1)
                    unique_games = dfv.drop_duplicates(subset=['year', 'week', 'match_key'])
                    unlucky = len(unique_games[unique_games['team_points'] >= 120])
                else:
                    unlucky = len(self.df[self.df['team_points'] >= 120])
                st.metric("120+ Point Losses", unlucky)

        except Exception as e:
            st.error(f"Error calculating tough luck records: {e}")

    @st.fragment
    def _display_miscellaneous(self):
        st.markdown("### 🎪 Miscellaneous Records & Fun Facts")

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 League Totals")
                # Compute unique game count (deduplicate paired rows if present)
                if 'opponent' in self.df.columns:
                    dfv = self.df.copy()
                    dfv['match_key'] = dfv.apply(lambda r: '|'.join(sorted([str(r['manager']).lower(), str(r.get('opponent','')).lower()])), axis=1)
                    unique_games = dfv.drop_duplicates(subset=['year', 'week', 'match_key'])
                    total_games = len(unique_games)
                else:
                    total_games = len(self.df)

                total_points = self.df['team_points'].sum()
                avg_score = self.df['team_points'].mean()

                st.metric("Total Games Played", f"{total_games}")
                st.metric("Total Points Scored", f"{total_points:,.0f}")
                st.metric("Average Game Score", f"{avg_score:.1f}")

                # Years active (format safely)
                years = sorted(self.df['year'].unique())
                if years:
                    st.metric("Seasons Tracked", len(years), delta=f"{int(years[0])} - {int(years[-1])}")
                else:
                    st.metric("Seasons Tracked", 0)

            with col2:
                st.markdown("#### 🎲 Interesting Stats")

                # Closest average margin
                self.df['margin'] = abs(self.df['team_points'] - self.df['opponent_points'])
                avg_margin = self.df['margin'].mean()
                st.metric("Average Victory Margin", f"{avg_margin:.1f} pts")

                # Blowouts
                blowouts = len(self.df[self.df['margin'] >= 50])
                st.metric("50+ Point Blowouts", blowouts)

                # Perfect weeks (if available)
                if 'team_projected_points' in self.df.columns:
                    self.df['accuracy'] = abs(self.df['team_points'] - self.df['team_projected_points'])
                    perfect = len(self.df[self.df['accuracy'] <= 1])
                    st.metric("Within 1pt of Projection", perfect)

        except Exception as e:
            st.error(f"Error calculating miscellaneous records: {e}")
