import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class SeasonProjectedStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced career projected stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>🎯 Career Projected Stats</h2>
        <p>Lifetime projection accuracy, spread performance, and upset analysis</p>
        </div>
        """, unsafe_allow_html=True)

        required_columns = [
            'manager', 'opponent', 'team_points', 'opponent_points', 'team_projected_points',
            'opponent_projected_points', 'expected_odds', 'margin', 'expected_spread',
            'proj_score_error', 'abs_proj_score_error'
        ]

        available_columns = self.df.columns.tolist()
        missing_columns = [col for col in required_columns if col not in available_columns]

        if missing_columns:
            st.error(f"❌ Some required columns are missing: {missing_columns}")
            return

        # Prepare data
        display_df = self._prepare_display_data(prefix)

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
            help="Toggle between career totals and per-game averages"
        )

        # === ENHANCED TABLE DISPLAY ===
        st.markdown(f"**Viewing {len(display_df):,} managers**")
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
                file_name=f"career_projected_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
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

        # Projected outcomes
        df['proj_win'] = df['team_projected_points'] > df['opponent_projected_points']
        df['proj_loss'] = df['team_projected_points'] <= df['opponent_projected_points']

        # Performance vs projection
        df['above_proj'] = df['team_points'] > df['team_projected_points']
        df['below_proj'] = df['team_points'] <= df['team_projected_points']

        # Spread performance
        df['win_vs_spread'] = (df['team_points'] - df['opponent_points']) > df['expected_spread']
        df['loss_vs_spread'] = (df['team_points'] - df['opponent_points']) <= df['expected_spread']

        # Upset tracking
        df['underdog'] = df['team_projected_points'] < df['opponent_projected_points']
        df['favorite'] = df['team_projected_points'] > df['opponent_projected_points']
        df['underdog_win'] = df['underdog'] & df['win']
        df['favorite_loss'] = df['favorite'] & df['loss']

        # Accuracy tiers (based on absolute error)
        df['accurate'] = df['abs_proj_score_error'] <= 10  # Within 10 points
        df['moderate'] = (df['abs_proj_score_error'] > 10) & (df['abs_proj_score_error'] <= 20)
        df['inaccurate'] = df['abs_proj_score_error'] > 20

        # Aggregate by manager
        agg_dict = {
            'team_points': 'sum',
            'opponent_points': 'sum',
            'team_projected_points': 'sum',
            'opponent_projected_points': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'proj_win': 'sum',
            'proj_loss': 'sum',
            'above_proj': 'sum',
            'below_proj': 'sum',
            'win_vs_spread': 'sum',
            'loss_vs_spread': 'sum',
            'underdog_win': 'sum',
            'favorite_loss': 'sum',
            'accurate': 'sum',
            'moderate': 'sum',
            'inaccurate': 'sum',
            'margin': 'sum',
            'expected_spread': 'sum',
            'proj_score_error': 'sum',
            'abs_proj_score_error': 'sum',
            'expected_odds': 'mean'
        }

        display_df = df.groupby('manager').agg(agg_dict).reset_index()

        # Calculate derived metrics
        display_df['games'] = display_df['win'] + display_df['loss']
        display_df['proj_accuracy'] = (display_df['accurate'] / display_df['games'] * 100).round(1)
        display_df['spread_win_rate'] = (display_df['win_vs_spread'] / display_df['games'] * 100).round(1)

        # Rename columns
        display_df = display_df.rename(columns={
            'manager': 'Manager',
            'win': 'W',
            'loss': 'L',
            'team_points': 'PF',
            'opponent_points': 'PA',
            'team_projected_points': 'Proj PF',
            'opponent_projected_points': 'Proj PA',
            'proj_win': 'Proj W',
            'proj_loss': 'Proj L',
            'above_proj': 'Above Proj',
            'below_proj': 'Below Proj',
            'win_vs_spread': 'W vs Spread',
            'loss_vs_spread': 'L vs Spread',
            'underdog_win': 'Upset W',
            'favorite_loss': 'Upset L',
            'accurate': 'Accurate',
            'moderate': 'Moderate',
            'inaccurate': 'Inaccurate',
            'margin': 'Margin',
            'expected_spread': 'Exp Margin',
            'proj_score_error': 'Proj Error',
            'abs_proj_score_error': 'Abs Proj Error',
            'expected_odds': 'Avg Odds',
            'proj_accuracy': 'Accuracy %',
            'spread_win_rate': 'Spread W %'
        })

        # Sort by manager
        display_df = display_df.sort_values(by='Manager', ascending=True).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_managers = len(df)
        if total_managers == 0:
            return {}

        total_games = df['games'].sum()

        # Projection accuracy
        overall_accuracy = (df['Accurate'].sum() / total_games * 100) if total_games > 0 else 0
        avg_abs_error = df['Abs Proj Error'].sum() / total_games if total_games > 0 else 0

        # Spread performance
        total_spread_wins = df['W vs Spread'].sum()
        spread_win_rate = (total_spread_wins / total_games * 100) if total_games > 0 else 0

        # Upset tracking
        total_upsets = df['Upset W'].sum()
        total_favorite_losses = df['Upset L'].sum()
        upset_rate = (total_upsets / total_games * 100) if total_games > 0 else 0

        # Performance vs projection
        total_above = df['Above Proj'].sum()
        above_proj_rate = (total_above / total_games * 100) if total_games > 0 else 0

        # Accuracy tiers
        accurate_pct = (df['Accurate'].sum() / total_games * 100) if total_games > 0 else 0
        moderate_pct = (df['Moderate'].sum() / total_games * 100) if total_games > 0 else 0
        inaccurate_pct = (df['Inaccurate'].sum() / total_games * 100) if total_games > 0 else 0

        return {
            'total_managers': total_managers,
            'total_games': int(total_games),
            'overall_accuracy': overall_accuracy,
            'avg_abs_error': avg_abs_error,
            'spread_win_rate': spread_win_rate,
            'total_upsets': int(total_upsets),
            'total_favorite_losses': int(total_favorite_losses),
            'upset_rate': upset_rate,
            'above_proj_rate': above_proj_rate,
            'accurate_pct': accurate_pct,
            'moderate_pct': moderate_pct,
            'inaccurate_pct': inaccurate_pct,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str, per_game: bool):
        """Render enhanced table with column configuration."""

        display_df = df.copy()

        # Apply per-game averaging if toggled
        if per_game:
            games = display_df['games']

            # Columns to average
            avg_cols = [
                'PF', 'PA', 'Proj PF', 'Proj PA', 'Margin', 'Exp Margin',
                'Proj Error', 'Abs Proj Error'
            ]
            for col in avg_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col] / games

            # Rate columns
            rate_cols = [
                'W', 'L', 'Proj W', 'Proj L', 'Above Proj', 'Below Proj',
                'W vs Spread', 'L vs Spread', 'Upset W', 'Upset L',
                'Accurate', 'Moderate', 'Inaccurate'
            ]
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
                help='Actual wins',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'L': st.column_config.NumberColumn(
                'L',
                help='Actual losses',
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
            'Proj PF': st.column_config.NumberColumn(
                'Proj PF',
                help='Projected points for',
                format='%.2f',
                width='small'
            ),
            'Proj PA': st.column_config.NumberColumn(
                'Proj PA',
                help='Projected points against',
                format='%.2f',
                width='small'
            ),
            'Proj W': st.column_config.NumberColumn(
                'Proj W',
                help='Projected wins',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Proj L': st.column_config.NumberColumn(
                'Proj L',
                help='Projected losses',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Above Proj': st.column_config.NumberColumn(
                'Above Proj',
                help='Games scored above projection',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Below Proj': st.column_config.NumberColumn(
                'Below Proj',
                help='Games scored below projection',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'W vs Spread': st.column_config.NumberColumn(
                'W vs Spread',
                help='Wins against the spread',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'L vs Spread': st.column_config.NumberColumn(
                'L vs Spread',
                help='Losses against the spread',
                format='%.2f' if per_game else '%d',
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
            'Accurate': st.column_config.NumberColumn(
                'Accurate',
                help='Games with ≤10 point projection error',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Moderate': st.column_config.NumberColumn(
                'Moderate',
                help='Games with 10-20 point projection error',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Inaccurate': st.column_config.NumberColumn(
                'Inaccurate',
                help='Games with >20 point projection error',
                format='%.2f' if per_game else '%d',
                width='small'
            ),
            'Margin': st.column_config.NumberColumn(
                'Margin',
                help='Actual point differential',
                format='%.2f',
                width='small'
            ),
            'Exp Margin': st.column_config.NumberColumn(
                'Exp Margin',
                help='Expected point differential',
                format='%.2f',
                width='small'
            ),
            'Proj Error': st.column_config.NumberColumn(
                'Proj Error',
                help='Projection error (actual - projected)',
                format='%.2f',
                width='small'
            ),
            'Abs Proj Error': st.column_config.NumberColumn(
                'Abs Proj Error',
                help='Absolute projection error',
                format='%.2f',
                width='small'
            ),
            'Avg Odds': st.column_config.NumberColumn(
                'Avg Odds',
                help='Average win probability',
                format='%.3f',
                width='small'
            ),
            'Accuracy %': st.column_config.NumberColumn(
                'Accuracy %',
                help='% of games within 10 points of projection',
                format='%.1f%%',
                width='small'
            ),
            'Spread W %': st.column_config.NumberColumn(
                'Spread W %',
                help='Win rate against the spread',
                format='%.1f%%',
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

        # Row 1: Overall Metrics
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
                "Avg Accuracy",
                f"{stats['overall_accuracy']:.1f}%",
                help="% of games within 10 points of projection"
            )

        with col4:
            st.metric(
                "Avg Error",
                f"{stats['avg_abs_error']:.2f}",
                help="Average absolute projection error"
            )

        # Row 2: Accuracy Tiers
        st.markdown("---")
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Accurate (≤10)",
                f"{stats['accurate_pct']:.1f}%",
                help="Games within 10 points"
            )

        with col6:
            st.metric(
                "Moderate (10-20)",
                f"{stats['moderate_pct']:.1f}%",
                help="Games 10-20 points off"
            )

        with col7:
            st.metric(
                "Inaccurate (>20)",
                f"{stats['inaccurate_pct']:.1f}%",
                help="Games more than 20 points off"
            )

        with col8:
            st.metric(
                "Above Proj Rate",
                f"{stats['above_proj_rate']:.1f}%",
                help="% of games above projection"
            )

        # Row 3: Spread Performance
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Spread Win Rate",
                f"{stats['spread_win_rate']:.1f}%",
                help="Win rate against the spread"
            )

        with col10:
            st.metric(
                "Total Upsets",
                f"{stats['total_upsets']:,}",
                help="Wins as underdog"
            )

        with col11:
            st.metric(
                "Upset Rate",
                f"{stats['upset_rate']:.1f}%",
                help="% of games won as underdog"
            )

        with col12:
            st.metric(
                "Favorite Losses",
                f"{stats['total_favorite_losses']:,}",
                help="Losses as favorite"
            )
