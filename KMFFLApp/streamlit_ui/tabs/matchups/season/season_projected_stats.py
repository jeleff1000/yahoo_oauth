import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class SeasonProjectedStatsViewer:
    def __init__(self, filtered_df, original_df):
        self.df = filtered_df.copy()
        self.original_df = original_df.copy()

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced season projected stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>🎯 Season Projected Stats</h2>
        <p>Projection accuracy and upset outcomes aggregated by season</p>
        </div>
        """, unsafe_allow_html=True)

        # Check for required columns
        required_columns = [
            'manager', 'year', 'team_points', 'opponent_points',
            'team_projected_points', 'opponent_projected_points',
            'expected_odds', 'margin', 'expected_spread',
            'proj_score_error', 'abs_proj_score_error',
            'proj_wins', 'above_proj_score', 'below_proj_score', 'win_vs_spread',
            'underdog_wins', 'favorite_losses'
        ]

        # Map old column names to new if needed
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
            st.info("No projected stats data available with current filters")
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
        st.markdown(f"**Viewing {len(display_df):,} seasons**")
        self._render_enhanced_table(display_df, prefix, aggregation_type)

        # === QUICK STATS SECTION (Below Table) ===
        st.markdown("---")
        self._render_quick_stats(stats, aggregation_type)

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
                file_name=f"season_projected_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_data(self) -> pd.DataFrame:
        """Prepare and format season-level data for display."""
        df = self.df.copy()

        # Calculate win/loss
        df['win'] = df['team_points'] > df['opponent_points']
        df['loss'] = df['team_points'] <= df['opponent_points']

        # Calculate derived metrics
        df['favored'] = df['expected_odds'] > 0.5
        df['surprise_outcome'] = ((df['underdog_wins'] == 1) | (df['favorite_losses'] == 1))
        df['opp_proj_diff'] = df['opponent_points'] - df['opponent_projected_points']

        # Calculate accuracy tiers (based on absolute projection error per game)
        df['accurate'] = df['abs_proj_score_error'] <= 10  # Within 10 points
        df['moderate'] = (df['abs_proj_score_error'] > 10) & (df['abs_proj_score_error'] <= 20)
        df['inaccurate'] = df['abs_proj_score_error'] > 20

        # Aggregate by manager and year
        agg_dict = {
            'team_points': 'sum',
            'opponent_points': 'sum',
            'team_projected_points': 'sum',
            'opponent_projected_points': 'sum',
            'expected_odds': 'mean',
            'margin': 'sum',
            'expected_spread': 'mean',
            'win': 'sum',
            'loss': 'sum',
            'proj_wins': 'sum',
            'above_proj_score': 'sum',
            'below_proj_score': 'sum',
            'win_vs_spread': 'sum',
            'proj_score_error': 'sum',
            'abs_proj_score_error': 'mean',  # Average error per game
            'underdog_wins': 'sum',
            'favorite_losses': 'sum',
            'favored': 'mean',  # % of games favored
            'surprise_outcome': 'sum',  # Count of upsets
            'opp_proj_diff': 'sum',
            'accurate': 'sum',  # Games within 10 points
            'moderate': 'sum',  # Games 10-20 points off
            'inaccurate': 'sum',  # Games >20 points off
        }

        display_df = df.groupby(['manager', 'year']).agg(agg_dict).reset_index()

        # Calculate season-level derived metrics
        display_df['proj_diff'] = display_df['team_points'] - display_df['team_projected_points']
        display_df['beat_projection'] = display_df['proj_diff'] > 0
        display_df['proj_win'] = display_df['expected_odds'] > 0.5  # Majority favored
        display_df['proj_margin'] = display_df['team_projected_points'] - display_df['opponent_projected_points']

        # Drop intermediate calculation columns
        display_df = display_df.drop(columns=['accurate', 'moderate', 'inaccurate'], errors='ignore')

        # Sort by most recent first
        display_df = display_df.sort_values(
            by=['year', 'manager'],
            ascending=[False, True]
        ).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_seasons = len(df)
        if total_seasons == 0:
            return {}

        # Calculate total games across all seasons
        total_games = df['win'].sum() + df['loss'].sum()

        # Win stats
        total_wins = df['win'].sum()
        total_losses = df['loss'].sum()
        proj_wins = df['proj_wins'].sum()

        # Projection accuracy
        avg_error = df['abs_proj_score_error'].mean()
        beat_proj = df['beat_projection'].sum()

        # Upset stats
        underdog_wins = df['underdog_wins'].sum()
        favorite_losses = df['favorite_losses'].sum()
        total_surprises = df['surprise_outcome'].sum()

        # Spread performance
        beat_spread = df['win_vs_spread'].sum()

        # Accuracy distribution (percentage of total games)
        total_accurate = df['accurate'].sum() if 'accurate' in df.columns else 0
        total_moderate = df['moderate'].sum() if 'moderate' in df.columns else 0
        total_inaccurate = df['inaccurate'].sum() if 'inaccurate' in df.columns else 0

        accurate_pct = (total_accurate / total_games * 100) if total_games > 0 else 0
        moderate_pct = (total_moderate / total_games * 100) if total_games > 0 else 0
        inaccurate_pct = (total_inaccurate / total_games * 100) if total_games > 0 else 0

        # Points
        avg_pf = df['team_points'].sum() / total_seasons
        avg_proj = df['team_projected_points'].sum() / total_seasons

        return {
            'total_seasons': total_seasons,
            'total_games': int(total_games),
            'total_wins': int(total_wins),
            'total_losses': int(total_losses),
            'proj_wins': proj_wins,
            'avg_error': avg_error,
            'beat_proj': beat_proj,
            'underdog_wins': underdog_wins,
            'favorite_losses': favorite_losses,
            'total_surprises': total_surprises,
            'beat_spread': beat_spread,
            'accurate_pct': accurate_pct,
            'moderate_pct': moderate_pct,
            'inaccurate_pct': inaccurate_pct,
            'avg_pf': avg_pf,
            'avg_proj': avg_proj,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str, per_game: bool):
        """Render enhanced table with column configuration."""

        display_df = df.copy()

        # Apply per-game averaging if toggled
        if per_game:
            # Calculate games played
            display_df['games'] = display_df['win'] + display_df['loss']

            # Columns to average
            avg_cols = [
                'team_points', 'opponent_points', 'team_projected_points',
                'opponent_projected_points', 'margin', 'proj_diff',
                'opp_proj_diff', 'proj_score_error'
            ]
            for col in avg_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] / display_df['games']

            # Rate columns (already rates, just label differently)
            rate_cols = ['win', 'loss', 'proj_wins', 'above_proj_score', 'below_proj_score',
                        'win_vs_spread', 'underdog_wins', 'favorite_losses', 'surprise_outcome']
            for col in rate_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] / display_df['games']

        # Rename columns for display
        display_df = display_df.rename(columns={
            'manager': 'Manager',
            'year': 'Year',
            'win': 'W',
            'loss': 'L',
            'team_points': 'Your Pts',
            'team_projected_points': 'Your Proj',
            'proj_diff': 'Your +/-',
            'above_proj_score': 'Beat Proj',
            'opponent_points': 'Opp Pts',
            'opponent_projected_points': 'Opp Proj',
            'opp_proj_diff': 'Opp +/-',
            'margin': 'Margin',
            'expected_spread': 'Proj Spread',
            'win_vs_spread': 'Beat Spread',
            'favored': 'Favored %',
            'expected_odds': 'Avg Win %',
            'proj_wins': 'Proj W',
            'underdog_wins': 'Upset W',
            'favorite_losses': 'Upset L',
            'surprise_outcome': 'Total Upsets',
            'proj_score_error': 'Total Error',
            'abs_proj_score_error': 'Avg Error',
            'accuracy_tier': 'Accuracy',
        })

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
                format='%.1f' if per_game else '%d',
                width='small'
            ),
            'L': st.column_config.NumberColumn(
                'L',
                help='Losses',
                format='%.1f' if per_game else '%d',
                width='small'
            ),
            'Your Pts': st.column_config.NumberColumn(
                'Your Pts',
                help='Total points scored' if not per_game else 'Average points per game',
                format='%.2f',
                width='small'
            ),
            'Your Proj': st.column_config.NumberColumn(
                'Your Proj',
                help='Total projected points' if not per_game else 'Average projected points per game',
                format='%.2f',
                width='small'
            ),
            'Your +/-': st.column_config.NumberColumn(
                'Your +/-',
                help='Points above/below projection',
                format='%.2f',
                width='small'
            ),
            'Beat Proj': st.column_config.NumberColumn(
                'Beat Proj',
                help='Games beat projection' if not per_game else 'Rate of beating projection',
                format='%.2f' if per_game else'%d',
                width='small'
            ),
            'Opp Pts': st.column_config.NumberColumn(
                'Opp Pts',
                help='Opponent points',
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
                help='Opponent points above/below projection',
                format='%.2f',
                width='small'
            ),
            'Margin': st.column_config.NumberColumn(
                'Margin',
                help='Point differential',
                format='%.2f',
                width='small'
            ),
            'Proj Spread': st.column_config.NumberColumn(
                'Proj Spread',
                help='Average projected spread',
                format='%.2f',
                width='small'
            ),
            'Beat Spread': st.column_config.NumberColumn(
                'Beat Spread',
                help='Games beat the spread' if not per_game else 'Rate of beating spread',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Favored %': st.column_config.NumberColumn(
                'Favored %',
                help='Percentage of games favored',
                format='%.1f%%',
                width='small'
            ),
            'Avg Win %': st.column_config.NumberColumn(
                'Avg Win %',
                help='Average win probability',
                format='%.1f%%',
                width='small'
            ),
            'Proj W': st.column_config.NumberColumn(
                'Proj W',
                help='Projected wins',
                format='%.1f',
                width='small'
            ),
            'Upset W': st.column_config.NumberColumn(
                'Upset W',
                help='Wins as underdog',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Upset L': st.column_config.NumberColumn(
                'Upset L',
                help='Losses as favorite',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Total Upsets': st.column_config.NumberColumn(
                'Total Upsets',
                help='Total upset outcomes',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Total Error': st.column_config.NumberColumn(
                'Total Error',
                help='Cumulative projection error',
                format='%.2f',
                width='small'
            ),
            'Avg Error': st.column_config.NumberColumn(
                'Avg Error',
                help='Average absolute projection error per game',
                format='%.2f',
                width='small'
            ),
            'Accuracy': st.column_config.TextColumn(
                'Accuracy',
                help='Projection accuracy tier',
                width='small'
            ),
        }

        # Format percentage columns
        if 'Favored %' in display_df.columns:
            display_df['Favored %'] = (display_df['Favored %'] * 100).round(1)
        if 'Avg Win %' in display_df.columns:
            display_df['Avg Win %'] = (display_df['Avg Win %'] * 100).round(1)

        # Display the enhanced dataframe
        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500
        )

    def _render_quick_stats(self, stats: dict, per_game: bool):
        """Render quick statistics cards."""
        if not stats:
            return

        st.markdown("### 📈 Quick Stats")

        # Row 1: Core Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Seasons",
                f"{stats['total_seasons']:,}",
                help="Total seasons analyzed"
            )

        with col2:
            st.metric(
                "Actual Wins",
                f"{stats['total_wins']:,}",
                help="Total actual wins"
            )

        with col3:
            st.metric(
                "Projected Wins",
                f"{stats['proj_wins']:.1f}",
                delta=f"{stats['proj_wins'] - stats['total_wins']:.1f}",
                help="Total projected wins"
            )

        with col4:
            st.metric(
                "Avg Error/Game",
                f"{stats['avg_error']:.2f}",
                help="Average projection error per game"
            )

        # Row 2: Performance Metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Beat Projection",
                f"{stats['beat_proj']}",
                help="Seasons where actual > projected"
            )

        with col6:
            st.metric(
                "Beat Spread",
                f"{stats['beat_spread']}",
                help="Games beat the spread"
            )

        with col7:
            st.metric(
                "Avg Season PF",
                f"{stats['avg_pf']:.1f}",
                help="Average points for per season"
            )

        with col8:
            st.metric(
                "Avg Season Proj",
                f"{stats['avg_proj']:.1f}",
                delta=f"{stats['avg_proj'] - stats['avg_pf']:.1f}",
                help="Average projected points per season"
            )

        # Row 3: Upset Stats
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Total Upsets",
                f"{stats['total_surprises']}",
                help="Total unexpected outcomes"
            )

        with col10:
            st.metric(
                "Upset Wins",
                f"{stats['underdog_wins']}",
                help="Wins as underdog"
            )

        with col11:
            st.metric(
                "Upset Losses",
                f"{stats['favorite_losses']}",
                help="Losses as favorite"
            )

        with col12:
            upset_rate = (stats['total_surprises'] / (stats['total_wins'] + stats['total_losses']) * 100) if (stats['total_wins'] + stats['total_losses']) > 0 else 0
            st.metric(
                "Upset Rate",
                f"{upset_rate:.1f}%",
                help="Percentage of games with upset outcomes"
            )

        # Row 4: Accuracy Distribution
        st.markdown("---")
        st.markdown("**Projection Accuracy Distribution (% of Games)**")
        col13, col14, col15, col16 = st.columns(4)

        with col13:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total games across all seasons"
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


# Legacy function wrapper for backward compatibility
@st.fragment
def display_season_projected_stats(matchup_df: pd.DataFrame, original_df: pd.DataFrame):
    """Legacy function wrapper - calls new class-based viewer."""
    viewer = SeasonProjectedStatsViewer(matchup_df, original_df)
    viewer.display(prefix="season_projected")
