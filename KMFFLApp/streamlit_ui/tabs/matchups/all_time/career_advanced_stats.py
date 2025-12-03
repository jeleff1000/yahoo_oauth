import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class SeasonAdvancedStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced career advanced stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>📊 Career Advanced Stats</h2>
        <p>Lifetime performance metrics, competition stats, and league comparisons</p>
        </div>
        """, unsafe_allow_html=True)

        required_columns = [
            'manager', 'opponent', 'week', 'year', 'team_points', 'opponent_points', 'win',
            'margin', 'total_matchup_score', 'teams_beat_this_week', 'opponent_teams_beat_this_week',
            'close_margin', 'above_league_median', 'below_league_median', 'above_opponent_median',
            'below_opponent_median', 'gpa', 'league_weekly_mean', 'league_weekly_median',
            'personal_season_mean', 'personal_season_median', 'winning_streak', 'losing_streak'
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
            help="Toggle between career totals and per-game averages"
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
                file_name=f"career_advanced_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_data(self, prefix) -> pd.DataFrame:
        """Prepare and format career-level data for display."""
        df = self.df.copy()

        # Calculate win/loss
        df['win'] = df['team_points'] > df['opponent_points']
        df['loss'] = df['team_points'] <= df['opponent_points']

        # Calculate close game wins/losses (games decided by <10 points)
        df['close_game'] = df['close_margin'] == 1
        df['close_win'] = df['close_game'] & df['win']
        df['close_loss'] = df['close_game'] & df['loss']

        # Aggregate by manager
        agg_dict = {
            'team_points': 'sum',
            'opponent_points': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'margin': 'sum',
            'total_matchup_score': 'sum',
            'teams_beat_this_week': 'sum',
            'opponent_teams_beat_this_week': 'sum',
            'close_margin': 'sum',
            'close_win': 'sum',
            'close_loss': 'sum',
            'above_league_median': 'sum',
            'below_league_median': 'sum',
            'above_opponent_median': 'sum',
            'below_opponent_median': 'sum',
            'gpa': 'mean',
            'league_weekly_mean': 'mean',
            'league_weekly_median': 'median',
            'personal_season_mean': 'mean',
            'personal_season_median': 'median',
            'winning_streak': 'max',
            'losing_streak': 'max'
        }

        display_df = df.groupby('manager').agg(agg_dict).reset_index()

        # Rename columns
        display_df = display_df.rename(columns={
            'manager': 'Manager',
            'win': 'W',
            'loss': 'L',
            'team_points': 'PF',
            'opponent_points': 'PA',
            'margin': 'Margin',
            'total_matchup_score': 'Total Score',
            'teams_beat_this_week': 'Wins vs All',
            'opponent_teams_beat_this_week': 'Opp Wins vs All',
            'close_margin': 'Close Games',
            'close_win': 'Close W',
            'close_loss': 'Close L',
            'above_league_median': 'Above Lg Avg',
            'below_league_median': 'Below Lg Avg',
            'above_opponent_median': 'Above Opp Avg',
            'below_opponent_median': 'Below Opp Avg',
            'gpa': 'GPA',
            'league_weekly_mean': 'Lg Mean',
            'league_weekly_median': 'Lg Median',
            'personal_season_mean': 'Team Mean',
            'personal_season_median': 'Team Median',
            'winning_streak': 'Win Streak',
            'losing_streak': 'Loss Streak'
        })

        # Sort by manager
        display_df = display_df.sort_values(by='Manager', ascending=True).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_managers = len(df)
        if total_managers == 0:
            return {}

        total_games = df['W'].sum() + df['L'].sum() if 'W' in df.columns and 'L' in df.columns else 0

        # GPA stats
        avg_gpa = df['GPA'].mean()
        high_gpa = len(df[df['GPA'] >= 3.0])
        low_gpa = len(df[df['GPA'] < 2.0])

        # Close game stats
        total_close = df['Close Games'].sum()
        total_close_w = df['Close W'].sum()
        total_close_l = df['Close L'].sum()

        # Competition stats
        avg_wins_vs_all = df['Wins vs All'].mean()

        # League comparison
        above_avg_rate = (df['Above Lg Avg'].sum() / (df['Above Lg Avg'].sum() + df['Below Lg Avg'].sum()) * 100) if (df['Above Lg Avg'].sum() + df['Below Lg Avg'].sum()) > 0 else 0

        # Streaks
        longest_win_streak = df['Win Streak'].max()
        longest_loss_streak = df['Loss Streak'].max()

        return {
            'total_managers': total_managers,
            'total_games': int(total_games),
            'avg_gpa': avg_gpa,
            'high_gpa': high_gpa,
            'low_gpa': low_gpa,
            'total_close': int(total_close),
            'total_close_w': int(total_close_w),
            'total_close_l': int(total_close_l),
            'avg_wins_vs_all': avg_wins_vs_all,
            'above_avg_rate': above_avg_rate,
            'longest_win_streak': int(longest_win_streak),
            'longest_loss_streak': int(longest_loss_streak),
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str, per_game: bool):
        """Render enhanced table with column configuration."""

        display_df = df.copy()

        # Apply per-game averaging if toggled
        if per_game:
            games = display_df['W'] + display_df['L']

            # Columns to average
            avg_cols = [
                'PF', 'PA', 'Margin', 'Total Score', 'Wins vs All', 'Opp Wins vs All',
                'Above Lg Avg', 'Below Lg Avg', 'Above Opp Avg', 'Below Opp Avg'
            ]
            for col in avg_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] / games

            # Rate columns
            rate_cols = ['W', 'L', 'Close Games', 'Close W', 'Close L']
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
            'W': st.column_config.NumberColumn(
                'W',
                help='Total wins',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'L': st.column_config.NumberColumn(
                'L',
                help='Total losses',
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
            'Total Score': st.column_config.NumberColumn(
                'Total Score',
                help='Combined points (PF + PA)',
                format='%.2f',
                width='small'
            ),
            'Wins vs All': st.column_config.NumberColumn(
                'Wins vs All',
                help='Head-to-head record vs all teams',
                format='%.2f' if per_game else '%.1f',
                width='small'
            ),
            'Opp Wins vs All': st.column_config.NumberColumn(
                'Opp Wins vs All',
                help="Opponents' wins vs all teams",
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
            'Above Lg Avg': st.column_config.NumberColumn(
                'Above Lg Avg',
                help='Games scored above league average',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Below Lg Avg': st.column_config.NumberColumn(
                'Below Lg Avg',
                help='Games scored below league average',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Above Opp Avg': st.column_config.NumberColumn(
                'Above Opp Avg',
                help='Games scored above opponent average',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Below Opp Avg': st.column_config.NumberColumn(
                'Below Opp Avg',
                help='Games scored below opponent average',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'GPA': st.column_config.NumberColumn(
                'GPA',
                help='Career grade point average',
                format='%.2f',
                width='small'
            ),
            'Lg Mean': st.column_config.NumberColumn(
                'Lg Mean',
                help='Career league-wide mean score',
                format='%.2f',
                width='small'
            ),
            'Lg Median': st.column_config.NumberColumn(
                'Lg Median',
                help='Career league-wide median score',
                format='%.2f',
                width='small'
            ),
            'Team Mean': st.column_config.NumberColumn(
                'Team Mean',
                help='Team career mean score',
                format='%.2f',
                width='small'
            ),
            'Team Median': st.column_config.NumberColumn(
                'Team Median',
                help='Team career median score',
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
                "Total Managers",
                f"{stats['total_managers']:,}",
                help="Total managers analyzed"
            )

        with col2:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total games across all managers"
            )

        with col3:
            st.metric(
                "Avg GPA",
                f"{stats['avg_gpa']:.2f}",
                help="Average career grade point average"
            )

        with col4:
            st.metric(
                "High GPA Managers",
                f"{stats['high_gpa']}",
                delta=f"{stats['high_gpa']/stats['total_managers']*100:.1f}%",
                help="Managers with GPA ≥ 3.0"
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
            close_pct = (stats['total_close_w'] / stats['total_close'] * 100) if stats['total_close'] > 0 else 0
            st.metric(
                "Close Game Record",
                close_record,
                delta=f"{close_pct:.1f}%",
                help="Win-loss record in close games"
            )

        with col7:
            st.metric(
                "Longest Win Streak",
                f"{stats['longest_win_streak']}",
                help="Longest winning streak (games)"
            )

        with col8:
            st.metric(
                "Longest Loss Streak",
                f"{stats['longest_loss_streak']}",
                help="Longest losing streak (games)"
            )

        # Row 3: Competition Stats
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Avg Wins vs All",
                f"{stats['avg_wins_vs_all']:.1f}",
                help="Average wins vs all teams (career)"
            )

        with col10:
            st.metric(
                "Above Avg Rate",
                f"{stats['above_avg_rate']:.1f}%",
                help="Percentage of games above league average"
            )

        with col11:
            st.metric(
                "GPA Elite %",
                f"{stats['high_gpa']/stats['total_managers']*100:.1f}%",
                help="Percentage of elite GPA managers (≥3.0)"
            )

        with col12:
            close_win_rate = (stats['total_close_w'] / stats['total_close'] * 100) if stats['total_close'] > 0 else 0
            st.metric(
                "Clutch Factor",
                f"{close_win_rate:.1f}%",
                help="Win % in close games (<10 pts)"
            )
