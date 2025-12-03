import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class SeasonAdvancedStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced season advanced stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>📊 Season Advanced Stats</h2>
        <p>Deep dive into performance metrics, competition stats, and league comparisons</p>
        </div>
        """, unsafe_allow_html=True)

        required_columns = [
            'manager', 'opponent', 'week', 'year', 'team_points', 'opponent_points', 'win',
            'margin', 'total_matchup_score', 'teams_beat_this_week', 'opponent_teams_beat_this_week',
            'close_margin', 'above_league_median', 'below_league_median', 'above_opponent_median',
            'below_opponent_median', 'gpa', 'league_weekly_mean', 'league_weekly_median',
            'personal_season_mean', 'personal_season_median', 'winning_streak', 'losing_streak',
            'abs_proj_score_error'
        ]

        available_columns = self.df.columns.tolist()
        missing_columns = [col for col in required_columns if col not in available_columns]

        if missing_columns:
            st.error(f"❌ Some required columns are missing: {missing_columns}")
            return

        # Prepare data
        display_df = self._prepare_display_data(prefix)

        if display_df.empty:
            st.info("No advanced stats data available with current filters")
            return

        # Calculate summary statistics
        stats = self._calculate_stats(display_df)

        # === AGGREGATION TOGGLE ===
        aggregation_type = st.toggle(
            "Per Game Averages",
            value=False,
            key=f"{prefix}_aggregation_type",
            help="Toggle between season totals and per-game averages"
        )

        # === ENHANCED TABLE DISPLAY ===
        self._render_enhanced_table(display_df, prefix, aggregation_type)

        # === QUICK STATS SECTION (Below Table) ===
        st.markdown("---")
        self._render_quick_stats(stats)

        # === DOWNLOAD SECTION ===
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**💾 Export Data**")
        with col2:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"season_advanced_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_data(self, prefix) -> pd.DataFrame:
        """Prepare and format season-level data for display."""
        df = self.df.copy()

        # Calculate win/loss
        df['win'] = df['team_points'] > df['opponent_points']
        df['loss'] = df['team_points'] <= df['opponent_points']

        # Calculate close game wins/losses (games decided by <10 points)
        df['close_game'] = df['close_margin'] == 1
        df['close_win'] = df['close_game'] & df['win']
        df['close_loss'] = df['close_game'] & df['loss']

        # Calculate blowout wins/losses (margin >20 points)
        df['blowout_win'] = (df['win']) & (abs(df['margin']) > 20)
        df['blowout_loss'] = (df['loss']) & (abs(df['margin']) > 20)

        # Calculate accuracy tiers (based on absolute projection error)
        df['accurate'] = df['abs_proj_score_error'] <= 10  # Within 10 points
        df['moderate'] = (df['abs_proj_score_error'] > 10) & (df['abs_proj_score_error'] <= 20)
        df['inaccurate'] = df['abs_proj_score_error'] > 20

        # Aggregate by manager and year
        agg_dict = {
            'team_points': 'sum',
            'opponent_points': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'margin': 'sum',
            'teams_beat_this_week': 'sum',
            'opponent_teams_beat_this_week': 'sum',
            'close_margin': 'sum',
            'close_win': 'sum',
            'close_loss': 'sum',
            'blowout_win': 'sum',
            'blowout_loss': 'sum',
            'above_league_median': 'sum',
            'below_league_median': 'sum',
            'above_opponent_median': 'sum',
            'below_opponent_median': 'sum',
            'accurate': 'sum',
            'moderate': 'sum',
            'inaccurate': 'sum',
            'abs_proj_score_error': 'mean',  # Average error per game
            'gpa': 'mean',
            'winning_streak': 'max',
            'losing_streak': 'max'
        }

        display_df = df.groupby(['manager', 'year']).agg(agg_dict).reset_index()

        # Calculate Close Win %
        display_df['close_win_pct'] = ((display_df['close_win'] / display_df['close_margin']) * 100).fillna(0).round(1)

        # Calculate accuracy percentage
        display_df['games'] = display_df['win'] + display_df['loss']
        display_df['accuracy_pct'] = ((display_df['accurate'] / display_df['games']) * 100).fillna(0).round(1)

        # Rename columns
        display_df = display_df.rename(columns={
            'manager': 'Manager',
            'year': 'Year',
            'win': 'W',
            'loss': 'L',
            'team_points': 'PF',
            'opponent_points': 'PA',
            'margin': 'Margin',
            'teams_beat_this_week': 'All-Play W',
            'opponent_teams_beat_this_week': 'Opp All-Play W',
            'close_margin': 'Close Games',
            'close_win': 'Close W',
            'close_loss': 'Close L',
            'close_win_pct': 'Close W %',
            'blowout_win': 'Blowout W',
            'blowout_loss': 'Blowout L',
            'above_league_median': 'Above Median',
            'below_league_median': 'Below Median',
            'above_opponent_median': 'vs Opp Avg+',
            'below_opponent_median': 'vs Opp Avg-',
            'abs_proj_score_error': 'Avg Error/G',
            'accuracy_pct': 'Accuracy %',
            'gpa': 'GPA',
            'winning_streak': 'Win Streak',
            'losing_streak': 'Loss Streak'
        })

        # Drop intermediate calculation columns
        display_df = display_df.drop(columns=['accurate', 'moderate', 'inaccurate', 'games'], errors='ignore')

        # Sort by most recent first
        display_df = display_df.sort_values(
            by=['Year', 'Manager'],
            ascending=[False, True]
        ).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_seasons = len(df)
        if total_seasons == 0:
            return {}

        # Calculate total games across all seasons
        total_games = df['games'].sum() if 'games' in df.columns else 0

        # GPA stats
        avg_gpa = df['GPA'].mean()
        high_gpa = len(df[df['GPA'] >= 3.0])
        low_gpa = len(df[df['GPA'] < 2.0])

        # Close game stats
        total_close = df['Close Games'].sum()
        total_close_w = df['Close W'].sum()
        total_close_l = df['Close L'].sum()
        avg_close_win_pct = df['Close W %'].mean()

        # Blowout stats
        total_blowout_w = df['Blowout W'].sum()
        total_blowout_l = df['Blowout L'].sum()

        # Competition stats
        avg_all_play_w = df['All-Play W'].mean()

        # League comparison
        above_median_rate = (df['Above Median'].sum() / (df['Above Median'].sum() + df['Below Median'].sum())) * 100 if (df['Above Median'].sum() + df['Below Median'].sum()) > 0 else 0

        # Streaks
        longest_win_streak = df['Win Streak'].max()
        longest_loss_streak = df['Loss Streak'].max()

        # Accuracy stats
        avg_error = df['Avg Error/G'].mean() if 'Avg Error/G' in df.columns else 0
        avg_accuracy_pct = df['Accuracy %'].mean() if 'Accuracy %' in df.columns else 0

        # Accuracy distribution (percentage of total games)
        total_accurate = df['accurate'].sum() if 'accurate' in df.columns else 0
        total_moderate = df['moderate'].sum() if 'moderate' in df.columns else 0
        total_inaccurate = df['inaccurate'].sum() if 'inaccurate' in df.columns else 0

        accurate_pct = (total_accurate / total_games * 100) if total_games > 0 else 0
        moderate_pct = (total_moderate / total_games * 100) if total_games > 0 else 0
        inaccurate_pct = (total_inaccurate / total_games * 100) if total_games > 0 else 0

        return {
            'total_seasons': total_seasons,
            'total_games': int(total_games),
            'avg_gpa': avg_gpa,
            'high_gpa': high_gpa,
            'low_gpa': low_gpa,
            'total_close': int(total_close),
            'total_close_w': int(total_close_w),
            'total_close_l': int(total_close_l),
            'avg_close_win_pct': avg_close_win_pct,
            'total_blowout_w': int(total_blowout_w),
            'total_blowout_l': int(total_blowout_l),
            'avg_all_play_w': avg_all_play_w,
            'above_median_rate': above_median_rate,
            'longest_win_streak': int(longest_win_streak),
            'longest_loss_streak': int(longest_loss_streak),
            'avg_error': avg_error,
            'avg_accuracy_pct': avg_accuracy_pct,
            'accurate_pct': accurate_pct,
            'moderate_pct': moderate_pct,
            'inaccurate_pct': inaccurate_pct,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str, per_game: bool):
        """Render enhanced table with column configuration."""

        display_df = df.copy()

        # Apply per-game averaging if toggled
        if per_game:
            games = display_df['W'] + display_df['L']

            # Columns to average
            avg_cols = [
                'PF', 'PA', 'Margin', 'All-Play W', 'Opp All-Play W',
                'Above Median', 'Below Median', 'vs Opp Avg+', 'vs Opp Avg-'
            ]
            for col in avg_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] / games

            # Rate columns
            rate_cols = ['W', 'L', 'Close Games', 'Close W', 'Close L', 'Blowout W', 'Blowout L']
            for col in rate_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] / games

        # Configure column display
        column_config = {
            'Manager': st.column_config.TextColumn(
                'Manager',
                help='Manager name',
                width='medium'
            ),
            'Year': st.column_config.NumberColumn(
                'Year',
                help='Season year',
                format='%d',
                width='small'
            ),
            'W': st.column_config.NumberColumn(
                'W',
                help='Wins',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'L': st.column_config.NumberColumn(
                'L',
                help='Losses',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'PF': st.column_config.NumberColumn(
                'PF',
                help='Points for',
                format='%.2f',
                width='small'
            ),
            'PA': st.column_config.NumberColumn(
                'PA',
                help='Points against',
                format='%.2f',
                width='small'
            ),
            'Margin': st.column_config.NumberColumn(
                'Margin',
                help='Point differential',
                format='%.2f',
                width='small'
            ),
            'All-Play W': st.column_config.NumberColumn(
                'All-Play W',
                help='All-play wins (weekly rank-based record vs entire league)',
                format='%.2f' if per_game else '%.1f',
                width='small'
            ),
            'Opp All-Play W': st.column_config.NumberColumn(
                'Opp All-Play W',
                help="Opponents' all-play wins",
                format='%.2f' if per_game else '%.1f',
                width='small'
            ),
            'Close Games': st.column_config.NumberColumn(
                'Close Games',
                help='Games decided by <10 points',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Close W': st.column_config.NumberColumn(
                'Close W',
                help='Wins in close games (<10 pts)',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Close L': st.column_config.NumberColumn(
                'Close L',
                help='Losses in close games (<10 pts)',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Close W %': st.column_config.NumberColumn(
                'Close W %',
                help='Win percentage in close games',
                format='%.1f%%',
                width='small'
            ),
            'Blowout W': st.column_config.NumberColumn(
                'Blowout W',
                help='Wins by >20 points (dominant victories)',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Blowout L': st.column_config.NumberColumn(
                'Blowout L',
                help='Losses by >20 points',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Above Median': st.column_config.NumberColumn(
                'Above Median',
                help='Games scored above league median',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Below Median': st.column_config.NumberColumn(
                'Below Median',
                help='Games scored below league median',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'vs Opp Avg+': st.column_config.NumberColumn(
                'vs Opp Avg+',
                help='Games scored above opponent season average',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'vs Opp Avg-': st.column_config.NumberColumn(
                'vs Opp Avg-',
                help='Games scored below opponent season average',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Avg Error/G': st.column_config.NumberColumn(
                'Avg Error/G',
                help='Average projection error per game',
                format='%.2f',
                width='small'
            ),
            'Accuracy %': st.column_config.NumberColumn(
                'Accuracy %',
                help='% of games within 10 points of projection',
                format='%.1f%%',
                width='small'
            ),
            'GPA': st.column_config.NumberColumn(
                'GPA',
                help='Grade point average (performance metric)',
                format='%.2f',
                width='small'
            ),
            'Win Streak': st.column_config.NumberColumn(
                'Win Streak',
                help='Longest winning streak',
                format='%d',
                width='small'
            ),
            'Loss Streak': st.column_config.NumberColumn(
                'Loss Streak',
                help='Longest losing streak',
                format='%d',
                width='small'
            ),
        }

        # Display the enhanced dataframe
        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500
        )

    def _render_quick_stats(self, stats: dict):
        """Render quick statistics cards."""
        if not stats:
            return

        st.markdown("### 📈 Quick Stats")

        # Row 1: GPA Stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Seasons",
                f"{stats['total_seasons']:,}",
                help="Total seasons analyzed"
            )

        with col2:
            st.metric(
                "Avg GPA",
                f"{stats['avg_gpa']:.2f}",
                help="Average grade point average"
            )

        with col3:
            st.metric(
                "High GPA Seasons",
                f"{stats['high_gpa']}",
                delta=f"{stats['high_gpa']/stats['total_seasons']*100:.1f}%",
                help="Seasons with GPA ≥ 3.0"
            )

        with col4:
            st.metric(
                "Low GPA Seasons",
                f"{stats['low_gpa']}",
                help="Seasons with GPA < 2.0"
            )

        # Row 2: Close Game Stats
        st.markdown("---")
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Close Games",
                f"{stats['total_close']}",
                help="Total games decided by <10 points"
            )

        with col6:
            close_record = f"{stats['total_close_w']}-{stats['total_close_l']}"
            st.metric(
                "Close Game Record",
                close_record,
                delta=f"{stats['avg_close_win_pct']:.1f}%",
                help="Win-loss record in close games"
            )

        with col7:
            st.metric(
                "Blowout W",
                f"{stats['total_blowout_w']}",
                help="Total wins by >20 points"
            )

        with col8:
            st.metric(
                "Blowout L",
                f"{stats['total_blowout_l']}",
                help="Total losses by >20 points"
            )

        # Row 3: Competition Stats
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Avg All-Play W",
                f"{stats['avg_all_play_w']:.1f}",
                help="Average all-play wins (per season)"
            )

        with col10:
            st.metric(
                "Above Median Rate",
                f"{stats['above_median_rate']:.1f}%",
                help="Percentage of games above league median"
            )

        with col11:
            st.metric(
                "Longest Win Streak",
                f"{stats['longest_win_streak']}",
                help="Longest winning streak (games)"
            )

        with col12:
            st.metric(
                "Longest Loss Streak",
                f"{stats['longest_loss_streak']}",
                help="Longest losing streak (games)"
            )

        # Row 4: Projection Accuracy Distribution
        st.markdown("---")
        col13, col14, col15, col16 = st.columns(4)

        with col13:
            st.metric(
                "Avg Error/Game",
                f"{stats['avg_error']:.2f}",
                help="Average projection error per game"
            )

        with col14:
            st.metric(
                "Accurate (≤10)",
                f"{stats['accurate_pct']:.1f}%",
                help="Games within 10 points of projection"
            )

        with col15:
            st.metric(
                "Moderate (10-20)",
                f"{stats['moderate_pct']:.1f}%",
                help="Games 10-20 points off projection"
            )

        with col16:
            st.metric(
                "Inaccurate (>20)",
                f"{stats['inaccurate_pct']:.1f}%",
                help="Games more than 20 points off projection"
            )
