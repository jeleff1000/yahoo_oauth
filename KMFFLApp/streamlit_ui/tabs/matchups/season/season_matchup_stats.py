import streamlit as st
import pandas as pd

class SeasonMatchupStatsViewer:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def _get_final_playoff_result(group):
        """Get the final playoff/consolation result for a season."""
        # Build filter for playoff/consolation games
        playoff_filter = group['playoff_round'].notna() & (group['playoff_round'] != '') if 'playoff_round' in group.columns else pd.Series([False] * len(group), index=group.index)

        consolation_filter = pd.Series([False] * len(group), index=group.index)
        if 'consolation_round' in group.columns:
            consolation_filter = group['consolation_round'].notna() & (group['consolation_round'] != '')

        # Filter to playoff/consolation games only
        playoff_games = group[playoff_filter | consolation_filter]

        if playoff_games.empty:
            return ''

        # Get the last playoff/consolation game (championship, 3rd place game, etc.)
        last_game = playoff_games.sort_values('week', ascending=False).iloc[0]

        playoff_round = last_game['playoff_round'] if 'playoff_round' in last_game.index else ''
        consolation_round = last_game['consolation_round'] if 'consolation_round' in last_game.index else ''
        win = last_game['win'] if 'win' in last_game.index else False

        # Determine which round to use
        round_name = ''
        if playoff_round and pd.notna(playoff_round) and str(playoff_round).strip() and str(playoff_round).lower() != 'none':
            round_name = str(playoff_round).strip()
        elif consolation_round and pd.notna(consolation_round) and str(consolation_round).strip() and str(consolation_round).lower() != 'none':
            round_name = str(consolation_round).strip()

        if not round_name:
            return ''

        # Format the round name
        # Convert from snake_case to Title Case (e.g., "fifth_place_game" -> "Fifth Place Game")
        round_name = round_name.replace('_', ' ').title()

        # Combine with outcome
        outcome = "Won" if win else "Lost"
        return f"{outcome} {round_name}"

    @st.fragment
    def display(self, prefix=""):
        st.header("Season Matchup Stats")
        if 'win' in self.df.columns:
            self.df['win'] = self.df['win'] == 1
            self.df['loss'] = self.df['win'] == 0

            # Toggle for aggregation type
            aggregation_type = st.toggle("Per Game", value=False, key=f"{prefix}_aggregation_type")
            aggregation_func = 'mean' if aggregation_type else 'sum'

            # Compute Sacko flag and playoff result before aggregation
            sacko_flags = []
            playoff_results = []

            for (manager, year), group in self.df.groupby(['manager', 'year']):
                # Sacko flag
                if 'sacko' in self.df.columns:
                    sacko_flag = group['sacko'].eq(1).any()
                else:
                    sacko_flag = False
                sacko_flags.append({'manager': manager, 'year': year, 'sacko': sacko_flag})

                # Playoff result
                playoff_result = self._get_final_playoff_result(group)
                playoff_results.append({'manager': manager, 'year': year, 'playoff_result': playoff_result})

            sacko_df = pd.DataFrame(sacko_flags)
            playoff_df = pd.DataFrame(playoff_results)

            # Build aggregation dictionary
            # Standard aggregations (sum or mean based on toggle)
            agg_columns = [
                'team_points', 'opponent_points', 'win', 'loss',
                'champion', 'is_playoffs', 'margin', 'total_matchup_score', 'teams_beat_this_week',
                'opponent_teams_beat_this_week', 'close_margin', 'above_league_median', 'below_league_median',
                'above_opponent_median', 'below_opponent_median'
            ]
            agg_dict = {col: aggregation_func for col in agg_columns if col in self.df.columns}

            # GPA is always mean
            if 'gpa' in self.df.columns:
                agg_dict['gpa'] = 'mean'

            # Do main aggregation first
            aggregated_df = self.df.groupby(['manager', 'year']).agg(agg_dict).reset_index()

            # Add additional stats separately after the main aggregation
            if 'team_points' in self.df.columns:
                best_week = self.df.groupby(['manager', 'year'])['team_points'].max().reset_index()
                best_week.columns = ['manager', 'year', 'team_points_max']
                aggregated_df = pd.merge(aggregated_df, best_week, on=['manager', 'year'], how='left')

                worst_week = self.df.groupby(['manager', 'year'])['team_points'].min().reset_index()
                worst_week.columns = ['manager', 'year', 'team_points_min']
                aggregated_df = pd.merge(aggregated_df, worst_week, on=['manager', 'year'], how='left')

                std_dev = self.df.groupby(['manager', 'year'])['team_points'].std().reset_index()
                std_dev.columns = ['manager', 'year', 'team_points_std']
                aggregated_df = pd.merge(aggregated_df, std_dev, on=['manager', 'year'], how='left')

            # Get final playoff seed (last value of season)
            if 'final_playoff_seed' in self.df.columns:
                def last_non_null(series):
                    non_null = series.dropna()
                    return non_null.iloc[-1] if len(non_null) > 0 else None

                final_seed = self.df.sort_values('week').groupby(['manager', 'year'])['final_playoff_seed'].apply(last_non_null).reset_index()
                final_seed.columns = ['manager', 'year', 'final_playoff_seed']
                aggregated_df = pd.merge(aggregated_df, final_seed, on=['manager', 'year'], how='left')

            # Calculate derived stats
            if 'win' in aggregated_df.columns and 'loss' in aggregated_df.columns:
                aggregated_df['games'] = aggregated_df['win'] + aggregated_df['loss']
                aggregated_df['win_pct'] = aggregated_df['win'] / aggregated_df['games']

            if 'team_points' in aggregated_df.columns and 'games' in aggregated_df.columns:
                aggregated_df['ppg'] = aggregated_df['team_points'] / aggregated_df['games']

            if 'opponent_points' in aggregated_df.columns and 'games' in aggregated_df.columns:
                aggregated_df['papg'] = aggregated_df['opponent_points'] / aggregated_df['games']

            if 'team_points' in aggregated_df.columns and 'opponent_points' in aggregated_df.columns:
                aggregated_df['point_diff'] = aggregated_df['team_points'] - aggregated_df['opponent_points']

            # Merge Sacko and Playoff Result columns
            aggregated_df = pd.merge(aggregated_df, sacko_df, on=['manager', 'year'], how='left')
            aggregated_df = pd.merge(aggregated_df, playoff_df, on=['manager', 'year'], how='left')

            # Rounding
            if aggregation_type:
                columns_to_round_2 = [c for c in [
                    'team_points', 'opponent_points', 'margin', 'total_matchup_score', 'teams_beat_this_week',
                    'opponent_teams_beat_this_week', 'gpa'
                ] if c in aggregated_df.columns]
                columns_to_round_3 = [c for c in [
                    'close_margin', 'above_league_median', 'below_league_median', 'above_opponent_median',
                    'below_opponent_median'
                ] if c in aggregated_df.columns]
                aggregated_df[columns_to_round_2] = aggregated_df[columns_to_round_2].round(2)
                aggregated_df[columns_to_round_3] = aggregated_df[columns_to_round_3].round(3)
                if 'win' in aggregated_df.columns:
                    aggregated_df['win'] = aggregated_df['win'].round(3)
                if 'loss' in aggregated_df.columns:
                    aggregated_df['loss'] = aggregated_df['loss'].round(3)

            aggregated_df['year'] = aggregated_df['year'].astype(str)

            # Playoff/Champ flags
            if 'champion' in aggregated_df.columns:
                aggregated_df['champion_check'] = aggregated_df['champion'] > 0
            if 'is_playoffs' in aggregated_df.columns:
                aggregated_df['team_made_playoffs'] = aggregated_df['is_playoffs'] > 0

            # Display columns (replaced quarterfinal/semifinal/championship with playoff_result)
            display_cols = [
                'manager', 'year', 'win', 'loss', 'win_pct',
                'ppg', 'papg', 'point_diff',
                'team_points', 'opponent_points',
                'team_points_max', 'team_points_min', 'team_points_std',
                'final_playoff_seed', 'team_made_playoffs', 'playoff_result',
                'champion_check', 'sacko'
            ]
            display_df = aggregated_df[[c for c in display_cols if c in aggregated_df.columns]].copy()

            display_df = display_df.rename(columns={
                'manager': 'Manager',
                'year': 'Year',
                'win': 'W',
                'loss': 'L',
                'win_pct': 'Win %',
                'ppg': 'PPG',
                'papg': 'PA/G',
                'point_diff': 'Diff',
                'team_points': 'PF',
                'opponent_points': 'PA',
                'team_points_max': 'Best',
                'team_points_min': 'Worst',
                'team_points_std': 'Std Dev',
                'final_playoff_seed': 'Seed',
                'team_made_playoffs': 'Playoffs',
                'playoff_result': 'Playoff Result',
                'champion_check': 'Champ',
                'sacko': 'Sacko'
            })

            # Format boolean/text columns
            if 'Playoff Result' in display_df.columns:
                display_df['Playoff Result'] = display_df['Playoff Result'].apply(
                    lambda x: f'🏆 {x}' if x else ''
                )

            if 'Playoffs' in display_df.columns:
                display_df['Playoffs'] = display_df['Playoffs'].apply(
                    lambda x: 'Yes' if x else ''
                )

            if 'Champ' in display_df.columns:
                display_df['Champ'] = display_df['Champ'].apply(
                    lambda x: '🏆 Champion' if x else ''
                )

            if 'Sacko' in display_df.columns:
                display_df['Sacko'] = display_df['Sacko'].apply(
                    lambda x: '💩 Sacko' if x else ''
                )

            # Format Win % as percentage
            if 'Win %' in display_df.columns:
                display_df['Win %'] = (display_df['Win %'] * 100).round(1)

            # Configure column display
            column_config = {
                'Manager': st.column_config.TextColumn(
                    'Manager',
                    help='Manager name',
                    width='medium'
                ),
                'Year': st.column_config.TextColumn(
                    'Year',
                    help='Season year',
                    width='small'
                ),
                'W': st.column_config.NumberColumn(
                    'W',
                    help='Wins',
                    format='%.1f' if aggregation_type else '%d',
                    width='small'
                ),
                'L': st.column_config.NumberColumn(
                    'L',
                    help='Losses',
                    format='%.1f' if aggregation_type else '%d',
                    width='small'
                ),
                'Win %': st.column_config.NumberColumn(
                    'Win %',
                    help='Winning percentage',
                    format='%.1f%%',
                    width='small'
                ),
                'PPG': st.column_config.NumberColumn(
                    'PPG',
                    help='Points per game',
                    format='%.2f',
                    width='small'
                ),
                'PA/G': st.column_config.NumberColumn(
                    'PA/G',
                    help='Points against per game',
                    format='%.2f',
                    width='small'
                ),
                'Diff': st.column_config.NumberColumn(
                    'Diff',
                    help='Point differential (PF - PA)',
                    format='%.1f',
                    width='small'
                ),
                'PF': st.column_config.NumberColumn(
                    'PF',
                    help='Total points for',
                    format='%.2f',
                    width='small'
                ),
                'PA': st.column_config.NumberColumn(
                    'PA',
                    help='Total points against',
                    format='%.2f',
                    width='small'
                ),
                'Best': st.column_config.NumberColumn(
                    'Best',
                    help='Best single-week score',
                    format='%.2f',
                    width='small'
                ),
                'Worst': st.column_config.NumberColumn(
                    'Worst',
                    help='Worst single-week score',
                    format='%.2f',
                    width='small'
                ),
                'Std Dev': st.column_config.NumberColumn(
                    'Std Dev',
                    help='Standard deviation (consistency)',
                    format='%.2f',
                    width='small'
                ),
                'Seed': st.column_config.NumberColumn(
                    'Seed',
                    help='Final playoff seed (regular season standing)',
                    format='%d',
                    width='small'
                ),
                'Playoffs': st.column_config.TextColumn(
                    'Playoffs',
                    help='Made playoffs',
                    width='small'
                ),
                'Playoff Result': st.column_config.TextColumn(
                    'Playoff Result',
                    help='Final playoff/consolation game result',
                    width='medium'
                ),
                'Champ': st.column_config.TextColumn(
                    'Champ',
                    help='Won championship',
                    width='medium'
                ),
                'Sacko': st.column_config.TextColumn(
                    'Sacko',
                    help='Finished last place',
                    width='medium'
                ),
            }

            st.dataframe(
                display_df,
                column_config=column_config,
                hide_index=True,
                use_container_width=True,
                height=500
            )

            # Download button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Table as CSV",
                data=csv,
                file_name="season_matchup_stats.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv"
            )

            # Summary metrics
            st.markdown("---")
            st.subheader("📊 Quick Stats")
            total_seasons = len(display_df)

            # Calculate summary stats
            avg_win_pct = display_df['Win %'].mean() if 'Win %' in display_df.columns and total_seasons > 0 else 0
            avg_ppg = display_df['PPG'].mean() if 'PPG' in display_df.columns and total_seasons > 0 else 0
            avg_papg = display_df['PA/G'].mean() if 'PA/G' in display_df.columns and total_seasons > 0 else 0
            best_season = display_df['Best'].max() if 'Best' in display_df.columns and total_seasons > 0 else 0
            championships = len(display_df[display_df['Champ'] == '🏆 Champion']) if 'Champ' in display_df.columns else 0
            sackos = len(display_df[display_df['Sacko'] == '💩 Sacko']) if 'Sacko' in display_df.columns else 0

            # Row 1: Core metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Seasons", f"{total_seasons:,}")
            with col2:
                st.metric("Avg Win %", f"{avg_win_pct:.1f}%")
            with col3:
                st.metric("Championships", championships)
            with col4:
                st.metric("Sackos", sackos)

            # Row 2: Scoring metrics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Avg PPG", f"{avg_ppg:.2f}")
            with col6:
                st.metric("Avg PA/G", f"{avg_papg:.2f}")
            with col7:
                margin = avg_ppg - avg_papg
                st.metric("Avg Margin/G", f"{margin:+.2f}")
            with col8:
                st.metric("Best Week (All)", f"{best_season:.2f}")
        else:
            st.write("The required column 'win' is not available in the data.")
