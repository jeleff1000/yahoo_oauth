import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class WeeklyAdvancedStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced weekly advanced stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>🔬 Weekly Advanced Stats</h2>
        <p>Performance metrics, efficiency, league context, and momentum indicators</p>
        </div>
        """, unsafe_allow_html=True)

        # Check for required columns
        required_columns = ['manager', 'week', 'year']
        if not all(col in self.df.columns for col in required_columns):
            st.error("❌ Required columns are missing from the data.")
            return

        # Prepare data
        display_df = self._prepare_display_data()

        if display_df.empty:
            st.info("No advanced stats available with current filters")
            return

        # Calculate summary statistics
        stats = self._calculate_stats(display_df)

        # === ENHANCED TABLE DISPLAY ===
        st.markdown(f"**Viewing {len(display_df):,} matchups**")
        self._render_enhanced_table(display_df, prefix)

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
                file_name=f"weekly_advanced_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_data(self) -> pd.DataFrame:
        """Prepare and format data for display."""
        df = self.df.copy()

        # Select columns to display - focus on ADVANCED metrics only
        columns_to_show = [
            'year', 'week', 'manager', 'opponent',
            # Performance Metrics
            'gpa', 'grade', 'power_rating',
            # Efficiency Metrics
            'lineup_efficiency', 'bench_points',
            # League Context
            'weekly_rank', 'above_league_median', 'league_weekly_median',
            # Competition
            'close_margin', 'total_matchup_score',
            # Momentum
            'winning_streak', 'losing_streak',
            # Playoff
            'playoff_round',
            # Win Probability (if available)
            'win_probability',
        ]

        # Filter to available columns
        available_cols = [col for col in columns_to_show if col in df.columns]
        display_df = df[available_cols].copy()

        # Format and rename columns with clearer headers
        display_df = display_df.rename(columns={
            'year': 'Year',
            'week': 'Week',
            'manager': 'Manager',
            'opponent': 'Opponent',
            # Performance
            'gpa': 'GPA',
            'grade': 'Grade',
            'power_rating': 'Power',
            # Efficiency
            'lineup_efficiency': 'Efficiency',
            'bench_points': 'Bench Pts',
            # Context
            'weekly_rank': 'Rank',
            'above_league_median': 'Above Avg',
            'league_weekly_median': 'League Avg',
            # Competition
            'close_margin': 'Close Game',
            'total_matchup_score': 'Combined',
            # Momentum
            'winning_streak': 'W Streak',
            'losing_streak': 'L Streak',
            # Playoff
            'playoff_round': 'Round',
            # Probability
            'win_probability': 'Win Prob',
        })

        # Format numeric columns
        numeric_cols = ['Power', 'League Avg', 'Combined', 'GPA', 'Bench Pts', 'Efficiency', 'Win Prob']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

        # Sort by most recent first
        display_df['Year'] = display_df['Year'].astype(int)
        display_df['Week'] = display_df['Week'].astype(int)
        display_df = display_df.sort_values(
            by=['Year', 'Week'],
            ascending=[False, False]
        ).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_games = len(df)
        if total_games == 0:
            return {}

        stats = {'total_games': total_games}

        # Performance averages
        if 'GPA' in df.columns:
            stats['avg_gpa'] = df['GPA'].mean()
            stats['max_gpa'] = df['GPA'].max()
            stats['min_gpa'] = df['GPA'].min()

        if 'Power' in df.columns:
            stats['avg_power'] = df['Power'].mean()
            stats['max_power'] = df['Power'].max()

        # Efficiency
        if 'Efficiency' in df.columns:
            stats['avg_efficiency'] = df['Efficiency'].mean()
            stats['perfect_lineups'] = len(df[df['Efficiency'] >= 98.0]) if df['Efficiency'].notna().any() else 0

        if 'Bench Pts' in df.columns:
            stats['avg_bench'] = df['Bench Pts'].mean()
            stats['total_bench'] = df['Bench Pts'].sum()

        # Competition
        if 'Close Game' in df.columns:
            stats['close_games'] = (df['Close Game'] == True).sum()

        if 'Rank' in df.columns:
            stats['avg_rank'] = df['Rank'].mean()
            stats['top_3_weeks'] = len(df[df['Rank'] <= 3])
            stats['bottom_3_weeks'] = len(df[df['Rank'] >= 6])

        # Streaks
        if 'W Streak' in df.columns:
            stats['max_win_streak'] = df['W Streak'].max()
        if 'L Streak' in df.columns:
            stats['max_loss_streak'] = df['L Streak'].max()

        return stats

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str):
        """Render enhanced table with column configuration."""

        # Create display dataframe with formatted values
        display_df = df.copy()

        # Format boolean columns
        if 'Close Game' in display_df.columns:
            display_df['Close Game'] = display_df['Close Game'].apply(
                lambda x: '✓' if x else ''
            )

        if 'Above Avg' in display_df.columns:
            display_df['Above Avg'] = display_df['Above Avg'].apply(
                lambda x: '✓' if x else '✗'
            )

        # Format playoff round
        if 'Round' in display_df.columns:
            display_df['Round'] = display_df['Round'].fillna('').apply(
                lambda x: f'🏆 {x.title()}' if x else ''
            )

        # Format grades with letter grades
        if 'Grade' in display_df.columns:
            display_df['Grade'] = display_df['Grade'].fillna('')

        # Format efficiency as percentage
        if 'Efficiency' in display_df.columns:
            display_df['Efficiency'] = display_df['Efficiency'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else ''
            )

        # Format win probability as percentage
        if 'Win Prob' in display_df.columns:
            display_df['Win Prob'] = display_df['Win Prob'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else ''
            )

        # Configure column display
        column_config = {
            'Year': st.column_config.NumberColumn(
                'Year',
                help='Season year',
                format='%d',
                width='small'
            ),
            'Week': st.column_config.NumberColumn(
                'Week',
                help='Week number',
                format='%d',
                width='small'
            ),
            'Manager': st.column_config.TextColumn(
                'Manager',
                help='Manager name',
                width='medium'
            ),
            'Opponent': st.column_config.TextColumn(
                'Opponent',
                help='Opponent name',
                width='medium'
            ),
            'GPA': st.column_config.NumberColumn(
                'GPA',
                help='Grade Point Average (0.0-4.0 scale)',
                format='%.2f',
                width='small'
            ),
            'Grade': st.column_config.TextColumn(
                'Grade',
                help='Letter grade (A, B, C, D, F)',
                width='small'
            ),
            'Power': st.column_config.NumberColumn(
                'Power',
                help='Power rating (team strength)',
                format='%.1f',
                width='small'
            ),
            'Efficiency': st.column_config.TextColumn(
                'Efficiency',
                help='Lineup efficiency (actual/optimal)',
                width='small'
            ),
            'Bench Pts': st.column_config.NumberColumn(
                'Bench Pts',
                help='Points left on bench',
                format='%.1f',
                width='small'
            ),
            'Rank': st.column_config.NumberColumn(
                'Rank',
                help='Weekly league ranking (1 = best)',
                format='%d',
                width='small'
            ),
            'Above Avg': st.column_config.TextColumn(
                'Above Avg',
                help='Scored above league average',
                width='small'
            ),
            'League Avg': st.column_config.NumberColumn(
                'League Avg',
                help='League average score this week',
                format='%.1f',
                width='small'
            ),
            'Close Game': st.column_config.TextColumn(
                'Close Game',
                help='Decided by < 10 points',
                width='small'
            ),
            'Combined': st.column_config.NumberColumn(
                'Combined',
                help='Total points scored in matchup',
                format='%.1f',
                width='small'
            ),
            'W Streak': st.column_config.NumberColumn(
                'W Streak',
                help='Current winning streak',
                format='%d',
                width='small'
            ),
            'L Streak': st.column_config.NumberColumn(
                'L Streak',
                help='Current losing streak',
                format='%d',
                width='small'
            ),
            'Round': st.column_config.TextColumn(
                'Round',
                help='Playoff round (if applicable)',
                width='medium'
            ),
            'Win Prob': st.column_config.TextColumn(
                'Win Prob',
                help='Pre-game win probability',
                width='small'
            ),
        }

        # Display the enhanced dataframe
        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500  # Fixed height for better scrolling
        )

    def _render_quick_stats(self, stats: dict):
        """Render quick statistics cards."""
        if not stats:
            return

        st.markdown("### 📈 Quick Stats")

        # Row 1: Performance Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total matchups analyzed"
            )

        with col2:
            if 'avg_gpa' in stats:
                st.metric(
                    "Avg GPA",
                    f"{stats['avg_gpa']:.2f}",
                    help="Average grade point average"
                )

        with col3:
            if 'avg_power' in stats:
                st.metric(
                    "Avg Power",
                    f"{stats['avg_power']:.1f}",
                    help="Average power rating"
                )

        with col4:
            if 'avg_efficiency' in stats:
                st.metric(
                    "Avg Efficiency",
                    f"{stats['avg_efficiency']:.1f}%",
                    help="Average lineup efficiency"
                )

        # Row 2: Context & Competition
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            if 'avg_rank' in stats:
                st.metric(
                    "Avg Rank",
                    f"{stats['avg_rank']:.1f}",
                    help="Average weekly ranking"
                )

        with col6:
            if 'top_3_weeks' in stats:
                st.metric(
                    "Top 3 Finishes",
                    f"{stats['top_3_weeks']}",
                    help="Weeks ranked in top 3"
                )

        with col7:
            if 'close_games' in stats:
                st.metric(
                    "Close Games",
                    f"{stats['close_games']}",
                    help="Games decided by < 10 pts"
                )

        with col8:
            if 'avg_bench' in stats:
                st.metric(
                    "Avg Bench Pts",
                    f"{stats['avg_bench']:.1f}",
                    help="Average points left on bench"
                )

        # Row 3: Extremes & Streaks
        if any(key in stats for key in ['max_gpa', 'max_power', 'max_win_streak', 'perfect_lineups']):
            st.markdown("---")
            col9, col10, col11, col12 = st.columns(4)

            with col9:
                if 'max_gpa' in stats:
                    st.metric(
                        "Best GPA",
                        f"{stats['max_gpa']:.2f}",
                        help="Highest single-week GPA"
                    )

            with col10:
                if 'perfect_lineups' in stats:
                    st.metric(
                        "Perfect Lineups",
                        f"{stats['perfect_lineups']}",
                        help="Weeks with 98%+ efficiency"
                    )

            with col11:
                if 'max_win_streak' in stats:
                    st.metric(
                        "Longest Win Streak",
                        f"{int(stats['max_win_streak'])}W",
                        help="Longest consecutive wins"
                    )

            with col12:
                if 'total_bench' in stats:
                    st.metric(
                        "Total Bench Pts",
                        f"{stats['total_bench']:.0f}",
                        help="Total points left on bench"
                    )
