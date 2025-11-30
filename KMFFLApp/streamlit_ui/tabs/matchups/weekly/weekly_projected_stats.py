import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class WeeklyProjectedStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced weekly projected stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>📈 Weekly Projected Stats</h2>
        <p>Pre-game projections, actual results, and projection accuracy metrics</p>
        </div>
        """, unsafe_allow_html=True)

        # Check for required columns
        required_columns = [
            'manager', 'opponent', 'team_points', 'opponent_points',
            'team_projected_points', 'opponent_projected_points',
            'expected_odds', 'margin', 'expected_spread', 'week', 'year',
            'proj_score_error', 'abs_proj_score_error'
        ]

        # Support both new and old column names
        if 'manager_proj_score' in self.df.columns and 'team_projected_points' not in self.df.columns:
            self.df['team_projected_points'] = self.df['manager_proj_score']
        if 'opponent_proj_score' in self.df.columns and 'opponent_projected_points' not in self.df.columns:
            self.df['opponent_projected_points'] = self.df['opponent_proj_score']

        if not all(col in self.df.columns for col in required_columns):
            st.error("❌ Some required columns are missing from the data.")
            return

        # Prepare data
        display_df = self._prepare_display_data()

        if display_df.empty:
            st.info("No projected stats available with current filters")
            return

        # Calculate summary statistics
        stats = self._calculate_stats(display_df)

        # === ENHANCED TABLE DISPLAY ===
        st.markdown(f"**Viewing {len(display_df):,} matchups**")
        st.caption("💡 **Accuracy Tiers:** Excellent (<5 pts error) • Good (5-10 pts) • Fair (10-20 pts) • Poor (>20 pts)")
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
                file_name=f"weekly_projected_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_data(self) -> pd.DataFrame:
        """Prepare and format data for display."""
        df = self.df.copy()

        # Calculate derived columns
        df['win'] = df['team_points'] > df['opponent_points']
        df['projected_win'] = df['team_projected_points'] > df['opponent_projected_points']
        df['above_projection'] = df['team_points'] > df['team_projected_points']
        df['beat_spread'] = (df['team_points'] - df['opponent_points']) > df['expected_spread']
        df['projection_diff'] = df['team_points'] - df['team_projected_points']

        # Opponent projection performance
        df['opp_proj_diff'] = df['opponent_points'] - df['opponent_projected_points']

        # Game context
        df['favored'] = df['expected_odds'] > 0.5

        # Surprise outcomes (if columns exist)
        if 'underdog_wins' in df.columns:
            df['surprise_win'] = df['underdog_wins'] == 1
        else:
            df['surprise_win'] = (df['win'] == True) & (df['favored'] == False)

        if 'favorite_losses' in df.columns:
            df['surprise_loss'] = df['favorite_losses'] == 1
        else:
            df['surprise_loss'] = (df['win'] == False) & (df['favored'] == True)

        df['surprise_outcome'] = df['surprise_win'] | df['surprise_loss']

        # Projection accuracy tier
        df['accuracy_tier'] = df['abs_proj_score_error'].apply(self._categorize_accuracy)

        # Select columns to display - reorganized by logical groups
        columns_to_show = [
            # Context
            'year', 'week', 'manager', 'opponent',
            'favored', 'expected_odds',

            # Your Performance
            'team_points', 'team_projected_points', 'projection_diff',
            'above_projection',

            # Opponent Performance
            'opponent_points', 'opponent_projected_points', 'opp_proj_diff',

            # Game Outcome
            'margin', 'expected_spread', 'beat_spread',
            'win', 'projected_win', 'surprise_outcome',

            # Accuracy Metrics
            'proj_score_error', 'abs_proj_score_error', 'accuracy_tier',
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
            # Context
            'favored': 'Favored?',
            'expected_odds': 'Win %',
            # Your Performance
            'team_points': 'Your Pts',
            'team_projected_points': 'Your Proj',
            'projection_diff': 'Your +/-',
            'above_projection': 'Beat Proj?',
            # Opponent Performance
            'opponent_points': 'Opp Pts',
            'opponent_projected_points': 'Opp Proj',
            'opp_proj_diff': 'Opp +/-',
            # Game Outcome
            'margin': 'Margin',
            'expected_spread': 'Spread',
            'beat_spread': 'Beat Spread?',
            'win': 'Result',
            'projected_win': 'Proj Result',
            'surprise_outcome': 'Upset?',
            # Accuracy
            'proj_score_error': 'Error',
            'abs_proj_score_error': 'Abs Error',
            'accuracy_tier': 'Accuracy',
        })

        # Format numeric columns
        numeric_cols = [
            'Your Pts', 'Your Proj', 'Your +/-',
            'Opp Pts', 'Opp Proj', 'Opp +/-',
            'Margin', 'Spread', 'Win %', 'Error', 'Abs Error'
        ]
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

    @staticmethod
    def _categorize_accuracy(abs_error):
        """Categorize projection accuracy based on absolute error."""
        if pd.isna(abs_error):
            return 'N/A'
        elif abs_error < 5:
            return '⭐ Excellent'
        elif abs_error < 10:
            return '✓ Good'
        elif abs_error < 20:
            return '~ Fair'
        else:
            return '✗ Poor'

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_games = len(df)
        if total_games == 0:
            return {}

        wins = df['Result'].sum()
        losses = total_games - wins
        projected_wins = df['Proj Result'].sum()

        avg_margin = df['Margin'].mean()
        avg_error = df['Error'].mean()
        avg_abs_error = df['Abs Error'].mean()

        beat_proj_count = df['Beat Proj?'].sum()
        beat_spread_count = df['Beat Spread?'].sum()

        beat_proj_pct = (beat_proj_count / total_games * 100) if total_games > 0 else 0
        beat_spread_pct = (beat_spread_count / total_games * 100) if total_games > 0 else 0

        # Projection accuracy
        accurate_projections = len(df[df['Result'] == df['Proj Result']])
        projection_accuracy = (accurate_projections / total_games * 100) if total_games > 0 else 0

        # Upset statistics
        upset_count = df['Upset?'].sum() if 'Upset?' in df.columns else 0
        upset_rate = (upset_count / total_games * 100) if total_games > 0 else 0

        # When favored
        favored_games = df[df['Favored?'] == True] if 'Favored?' in df.columns else pd.DataFrame()
        favored_wins = favored_games['Result'].sum() if len(favored_games) > 0 else 0
        favored_win_pct = (favored_wins / len(favored_games) * 100) if len(favored_games) > 0 else 0

        # When underdog
        underdog_games = df[df['Favored?'] == False] if 'Favored?' in df.columns else pd.DataFrame()
        underdog_wins = underdog_games['Result'].sum() if len(underdog_games) > 0 else 0
        underdog_win_pct = (underdog_wins / len(underdog_games) * 100) if len(underdog_games) > 0 else 0

        # Accuracy tier distribution
        accuracy_dist = df['Accuracy'].value_counts().to_dict() if 'Accuracy' in df.columns else {}

        return {
            'total_games': total_games,
            'wins': int(wins),
            'losses': int(losses),
            'projected_wins': int(projected_wins),
            'projected_losses': int(total_games - projected_wins),
            'avg_margin': avg_margin,
            'avg_error': avg_error,
            'avg_abs_error': avg_abs_error,
            'beat_proj_count': int(beat_proj_count),
            'beat_spread_count': int(beat_spread_count),
            'beat_proj_pct': beat_proj_pct,
            'beat_spread_pct': beat_spread_pct,
            'projection_accuracy': projection_accuracy,
            'upset_count': int(upset_count),
            'upset_rate': upset_rate,
            'favored_games': len(favored_games),
            'favored_wins': int(favored_wins),
            'favored_win_pct': favored_win_pct,
            'underdog_games': len(underdog_games),
            'underdog_wins': int(underdog_wins),
            'underdog_win_pct': underdog_win_pct,
            'accuracy_dist': accuracy_dist,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str):
        """Render enhanced table with column configuration."""

        # Create display dataframe with formatted values
        display_df = df.copy()

        # Format boolean columns
        if 'Result' in display_df.columns:
            display_df['Result'] = display_df['Result'].apply(
                lambda x: '✓ Win' if x else '✗ Loss'
            )

        if 'Proj Result' in display_df.columns:
            display_df['Proj Result'] = display_df['Proj Result'].apply(
                lambda x: '✓ Win' if x else '✗ Loss'
            )

        if 'Beat Proj?' in display_df.columns:
            display_df['Beat Proj?'] = display_df['Beat Proj?'].apply(
                lambda x: '✓' if x else '✗'
            )

        if 'Beat Spread?' in display_df.columns:
            display_df['Beat Spread?'] = display_df['Beat Spread?'].apply(
                lambda x: '✓' if x else '✗'
            )

        if 'Favored?' in display_df.columns:
            display_df['Favored?'] = display_df['Favored?'].apply(
                lambda x: '⭐ Yes' if x else 'Underdog'
            )

        if 'Upset?' in display_df.columns:
            display_df['Upset?'] = display_df['Upset?'].apply(
                lambda x: '🎯 Upset!' if x else ''
            )

        # Format Win % as percentage
        if 'Win %' in display_df.columns:
            display_df['Win %'] = display_df['Win %'].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else ''
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
            'Favored?': st.column_config.TextColumn(
                'Favored?',
                help='Were you favored to win?',
                width='small'
            ),
            'Win %': st.column_config.TextColumn(
                'Win %',
                help='Pre-game win probability',
                width='small'
            ),
            'Your Pts': st.column_config.NumberColumn(
                'Your Pts',
                help='Actual points scored',
                format='%.2f',
                width='small'
            ),
            'Your Proj': st.column_config.NumberColumn(
                'Your Proj',
                help='Projected points',
                format='%.2f',
                width='small'
            ),
            'Your +/-': st.column_config.NumberColumn(
                'Your +/-',
                help='Actual minus Projected (+ = beat projection)',
                format='%.2f',
                width='small'
            ),
            'Beat Proj?': st.column_config.TextColumn(
                'Beat Proj?',
                help='Scored above projection',
                width='small'
            ),
            'Opp Pts': st.column_config.NumberColumn(
                'Opp Pts',
                help='Opponent actual points',
                format='%.2f',
                width='small'
            ),
            'Opp Proj': st.column_config.NumberColumn(
                'Opp Proj',
                help='Opponent projected points',
                format='%.2f',
                width='small'
            ),
            'Opp +/-': st.column_config.NumberColumn(
                'Opp +/-',
                help='Opponent actual vs projected',
                format='%.2f',
                width='small'
            ),
            'Margin': st.column_config.NumberColumn(
                'Margin',
                help='Point differential (Your Pts - Opp Pts)',
                format='%.2f',
                width='small'
            ),
            'Spread': st.column_config.NumberColumn(
                'Spread',
                help='Expected point spread (+ = favored)',
                format='%.2f',
                width='small'
            ),
            'Beat Spread?': st.column_config.TextColumn(
                'Beat Spread?',
                help='Beat the spread',
                width='small'
            ),
            'Result': st.column_config.TextColumn(
                'Result',
                help='Actual game result',
                width='small'
            ),
            'Proj Result': st.column_config.TextColumn(
                'Proj Result',
                help='Projected game result',
                width='small'
            ),
            'Upset?': st.column_config.TextColumn(
                'Upset?',
                help='Result defied expectations',
                width='small'
            ),
            'Error': st.column_config.NumberColumn(
                'Error',
                help='Projection error (actual - projected)',
                format='%.2f',
                width='small'
            ),
            'Abs Error': st.column_config.NumberColumn(
                'Abs Error',
                help='Absolute projection error',
                format='%.2f',
                width='small'
            ),
            'Accuracy': st.column_config.TextColumn(
                'Accuracy',
                help='Projection accuracy tier',
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

        # Row 1: Record & Accuracy
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total matchups analyzed"
            )

        with col2:
            actual_record = f"{stats['wins']}-{stats['losses']}"
            st.metric(
                "Actual Record",
                actual_record,
                help="Actual win-loss record"
            )

        with col3:
            projected_record = f"{stats['projected_wins']}-{stats['projected_losses']}"
            st.metric(
                "Projected Record",
                projected_record,
                help="What projections predicted"
            )

        with col4:
            st.metric(
                "Projection Accuracy",
                f"{stats['projection_accuracy']:.1f}%",
                help="How often projections predicted the correct winner"
            )

        # Row 2: Performance vs Projections
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Avg Error",
                f"{stats['avg_error']:.2f}",
                help="Average projection error (positive = beat projection)"
            )

        with col6:
            st.metric(
                "Avg Abs Error",
                f"{stats['avg_abs_error']:.2f}",
                help="Average absolute projection error (always positive)"
            )

        with col7:
            st.metric(
                "Beat Projection",
                f"{stats['beat_proj_pct']:.1f}%",
                delta=f"{stats['beat_proj_count']} games",
                help="Games where actual score beat projection"
            )

        with col8:
            st.metric(
                "Beat Spread",
                f"{stats['beat_spread_pct']:.1f}%",
                delta=f"{stats['beat_spread_count']} games",
                help="Games where margin beat the spread"
            )

        # Row 3: Upset & Context Stats
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Upset Games",
                f"{stats['upset_count']}",
                delta=f"{stats['upset_rate']:.1f}%",
                help="Games with surprise outcomes"
            )

        with col10:
            if stats['favored_games'] > 0:
                st.metric(
                    "As Favorite",
                    f"{stats['favored_wins']}-{stats['favored_games']-stats['favored_wins']}",
                    delta=f"{stats['favored_win_pct']:.1f}%",
                    help="Record when favored to win"
                )

        with col11:
            if stats['underdog_games'] > 0:
                st.metric(
                    "As Underdog",
                    f"{stats['underdog_wins']}-{stats['underdog_games']-stats['underdog_wins']}",
                    delta=f"{stats['underdog_win_pct']:.1f}%",
                    help="Record when underdog"
                )

        with col12:
            st.metric(
                "Avg Margin",
                f"{stats['avg_margin']:.2f}",
                help="Average point differential per game"
            )

        # Row 4: Accuracy Distribution
        if stats.get('accuracy_dist'):
            st.markdown("---")
            st.markdown("**📊 Projection Accuracy Distribution**")
            acc_cols = st.columns(len(stats['accuracy_dist']))
            for idx, (tier, count) in enumerate(sorted(stats['accuracy_dist'].items())):
                with acc_cols[idx]:
                    st.metric(tier, f"{count}")
