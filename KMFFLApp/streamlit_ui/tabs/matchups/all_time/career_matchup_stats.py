import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class CareerMatchupStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced career matchup stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>📊 Career Matchup Stats</h2>
        <p>Lifetime performance metrics, playoff history, and championship achievements</p>
        </div>
        """, unsafe_allow_html=True)

        if 'win' not in self.df.columns:
            st.error("❌ The required column 'win' is not available in the data.")
            return

        # Prepare data
        display_df = self._prepare_display_data(prefix)

        if display_df.empty:
            st.info("No career matchup data available with current filters")
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
                file_name=f"career_matchup_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_data(self, prefix) -> pd.DataFrame:
        """Prepare and format career-level data for display."""
        df = self.df.copy()

        # Ensure win/loss are boolean
        df['win'] = df['win'] == 1
        df['loss'] = df['win'] == 0

        # Create manager-year key for tracking unique seasons
        df['manager_year'] = df['manager'].astype(str) + '_' + df['year'].astype(str)

        # Count unique years and games per manager
        unique_years = df.groupby('manager')['year'].nunique()
        unique_games = df.groupby('manager')['manager_year'].apply(lambda x: len(df[df['manager_year'].isin(x)]))

        df['unique_years'] = df['manager'].map(unique_years)
        df['unique_games'] = df['manager'].map(unique_games)

        # Count championships, sackos, playoffs per season, then sum
        champion_count = df.groupby(['manager', 'year'])['champion'].max().groupby('manager').sum() if 'champion' in df.columns else None
        sacko_count = df.groupby(['manager', 'year'])['sacko'].max().groupby('manager').sum() if 'sacko' in df.columns else None
        playoffs_count = df.groupby(['manager', 'year'])['is_playoffs'].max().groupby('manager').sum() if 'is_playoffs' in df.columns else None

        # Main aggregation
        agg_dict = {
            'team_points': 'sum',
            'opponent_points': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'unique_years': 'first',
            'unique_games': 'first'
        }

        # Filter to only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        aggregated_df = df.groupby('manager').agg(agg_dict).reset_index()

        # Add championship/sacko/playoff counts
        if champion_count is not None:
            aggregated_df['champion'] = champion_count.values
        if sacko_count is not None:
            aggregated_df['sacko'] = sacko_count.values
        if playoffs_count is not None:
            aggregated_df['is_playoffs'] = playoffs_count.values

        # Add stats calculated separately
        if 'team_points' in df.columns:
            best_week = df.groupby('manager')['team_points'].max().reset_index()
            best_week.columns = ['manager', 'team_points_max']
            aggregated_df = pd.merge(aggregated_df, best_week, on='manager', how='left')

            worst_week = df.groupby('manager')['team_points'].min().reset_index()
            worst_week.columns = ['manager', 'team_points_min']
            aggregated_df = pd.merge(aggregated_df, worst_week, on='manager', how='left')

            std_dev = df.groupby('manager')['team_points'].std().reset_index()
            std_dev.columns = ['manager', 'team_points_std']
            aggregated_df = pd.merge(aggregated_df, std_dev, on='manager', how='left')

        # Calculate derived stats
        if 'win' in aggregated_df.columns and 'loss' in aggregated_df.columns:
            aggregated_df['games'] = aggregated_df['win'] + aggregated_df['loss']
            aggregated_df['win_pct'] = (aggregated_df['win'] / aggregated_df['games'] * 100).round(1)

        if 'team_points' in aggregated_df.columns and 'games' in aggregated_df.columns:
            aggregated_df['ppg'] = (aggregated_df['team_points'] / aggregated_df['games']).round(2)

        if 'opponent_points' in aggregated_df.columns and 'games' in aggregated_df.columns:
            aggregated_df['papg'] = (aggregated_df['opponent_points'] / aggregated_df['games']).round(2)

        if 'team_points' in aggregated_df.columns and 'opponent_points' in aggregated_df.columns:
            aggregated_df['point_diff'] = (aggregated_df['team_points'] - aggregated_df['opponent_points']).round(1)

        # Championship/playoff rates (per season)
        if 'champion' in aggregated_df.columns and 'unique_years' in aggregated_df.columns:
            aggregated_df['champ_rate'] = (aggregated_df['champion'] / aggregated_df['unique_years'] * 100).round(1)

        if 'sacko' in aggregated_df.columns and 'unique_years' in aggregated_df.columns:
            aggregated_df['sacko_rate'] = (aggregated_df['sacko'] / aggregated_df['unique_years'] * 100).round(1)

        if 'is_playoffs' in aggregated_df.columns and 'unique_years' in aggregated_df.columns:
            aggregated_df['playoff_rate'] = (aggregated_df['is_playoffs'] / aggregated_df['unique_years'] * 100).round(1)

        # Rename columns for display
        rename_dict = {
            'manager': 'Manager',
            'unique_years': 'Seasons',
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
            'champion': 'Champs',
            'is_playoffs': 'Playoffs',
            'sacko': 'Sackos',
            'champ_rate': 'Champ %',
            'playoff_rate': 'Playoff %',
            'sacko_rate': 'Sacko %'
        }
        aggregated_df = aggregated_df.rename(columns=rename_dict)

        # Select display columns
        display_cols = [
            'Manager', 'Seasons', 'W', 'L', 'Win %',
            'PPG', 'PA/G', 'Diff',
            'PF', 'PA',
            'Best', 'Worst', 'Std Dev',
            'Playoffs', 'Playoff %',
            'Champs', 'Champ %',
            'Sackos', 'Sacko %'
        ]
        display_df = aggregated_df[[c for c in display_cols if c in aggregated_df.columns]].copy()

        # Sort by manager
        display_df = display_df.sort_values(by='Manager', ascending=True).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_managers = len(df)
        if total_managers == 0:
            return {}

        total_games = df['games'].sum() if 'games' in df.columns else 0
        total_seasons = df['Seasons'].sum() if 'Seasons' in df.columns else 0

        # Win rate
        avg_win_pct = df['Win %'].mean() if 'Win %' in df.columns and total_managers > 0 else 0

        # Scoring
        avg_ppg = df['PPG'].mean() if 'PPG' in df.columns and total_managers > 0 else 0
        avg_papg = df['PA/G'].mean() if 'PA/G' in df.columns and total_managers > 0 else 0
        best_week = df['Best'].max() if 'Best' in df.columns and total_managers > 0 else 0

        # Championships/playoffs/sackos
        total_champs = df['Champs'].sum() if 'Champs' in df.columns else 0
        total_sackos = df['Sackos'].sum() if 'Sackos' in df.columns else 0
        total_playoffs = df['Playoffs'].sum() if 'Playoffs' in df.columns else 0

        # Rates
        avg_champ_rate = df['Champ %'].mean() if 'Champ %' in df.columns and total_managers > 0 else 0
        avg_playoff_rate = df['Playoff %'].mean() if 'Playoff %' in df.columns and total_managers > 0 else 0
        avg_sacko_rate = df['Sacko %'].mean() if 'Sacko %' in df.columns and total_managers > 0 else 0

        return {
            'total_managers': total_managers,
            'total_games': int(total_games),
            'total_seasons': int(total_seasons),
            'avg_win_pct': avg_win_pct,
            'avg_ppg': avg_ppg,
            'avg_papg': avg_papg,
            'best_week': best_week,
            'total_champs': int(total_champs),
            'total_sackos': int(total_sackos),
            'total_playoffs': int(total_playoffs),
            'avg_champ_rate': avg_champ_rate,
            'avg_playoff_rate': avg_playoff_rate,
            'avg_sacko_rate': avg_sacko_rate,
        }

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str, per_game: bool):
        """Render enhanced table with column configuration."""

        display_df = df.copy()

        # Apply per-game averaging if toggled (only for cumulative stats)
        if per_game:
            games = display_df['games']

            # Columns to average per game
            avg_cols = ['PF', 'PA', 'Diff']
            for col in avg_cols:
                if col in display_df.columns:
                    display_df[col] = (display_df[col] / games).round(2)

            # W/L become rates
            if 'W' in display_df.columns:
                display_df['W'] = (display_df['W'] / games).round(3)
            if 'L' in display_df.columns:
                display_df['L'] = (display_df['L'] / games).round(3)

        # Configure column display
        column_config = {
            'Manager': st.column_config.TextColumn(
                'Manager',
                help='Manager name',
                width='medium'
            ),
            'Seasons': st.column_config.NumberColumn(
                'Seasons',
                help='Number of seasons played',
                format='%d',
                width='small'
            ),
            'W': st.column_config.NumberColumn(
                'W',
                help='Total wins',
                format='%.3f' if per_game else '%d',
                width='small'
            ),
            'L': st.column_config.NumberColumn(
                'L',
                help='Total losses',
                format='%.3f' if per_game else '%d',
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
                help='Point differential',
                format='%.1f' if per_game else '%.2f',
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
            'Playoffs': st.column_config.NumberColumn(
                'Playoffs',
                help='Playoff appearances',
                format='%d',
                width='small'
            ),
            'Playoff %': st.column_config.NumberColumn(
                'Playoff %',
                help='Playoff rate (per season)',
                format='%.1f%%',
                width='small'
            ),
            'Champs': st.column_config.NumberColumn(
                'Champs',
                help='Championships won',
                format='%d',
                width='small'
            ),
            'Champ %': st.column_config.NumberColumn(
                'Champ %',
                help='Championship rate (per season)',
                format='%.1f%%',
                width='small'
            ),
            'Sackos': st.column_config.NumberColumn(
                'Sackos',
                help='Last place finishes',
                format='%d',
                width='small'
            ),
            'Sacko %': st.column_config.NumberColumn(
                'Sacko %',
                help='Sacko rate (per season)',
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
                help="Total managers in league history"
            )

        with col2:
            st.metric(
                "Total Seasons",
                f"{stats['total_seasons']:,}",
                help="Combined seasons across all managers"
            )

        with col3:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total games played"
            )

        with col4:
            st.metric(
                "Avg Win %",
                f"{stats['avg_win_pct']:.1f}%",
                help="Average winning percentage"
            )

        # Row 2: Scoring Metrics
        st.markdown("---")
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Avg PPG",
                f"{stats['avg_ppg']:.2f}",
                help="Average points per game"
            )

        with col6:
            st.metric(
                "Avg PA/G",
                f"{stats['avg_papg']:.2f}",
                help="Average points against per game"
            )

        with col7:
            margin = stats['avg_ppg'] - stats['avg_papg']
            st.metric(
                "Avg Margin/G",
                f"{margin:+.2f}",
                help="Average point differential per game"
            )

        with col8:
            st.metric(
                "Best Week (All)",
                f"{stats['best_week']:.2f}",
                help="Highest single-week score"
            )

        # Row 3: Championships & Playoffs
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "Championships",
                f"{stats['total_champs']:,}",
                help="Total championships"
            )

        with col10:
            st.metric(
                "Avg Champ Rate",
                f"{stats['avg_champ_rate']:.1f}%",
                help="Average championship rate per season"
            )

        with col11:
            st.metric(
                "Playoff Apps",
                f"{stats['total_playoffs']:,}",
                help="Total playoff appearances"
            )

        with col12:
            st.metric(
                "Avg Playoff Rate",
                f"{stats['avg_playoff_rate']:.1f}%",
                help="Average playoff rate per season"
            )

        # Row 4: Performance Metrics
        st.markdown("---")
        col13, col14, col15, col16 = st.columns(4)

        with col13:
            st.metric(
                "Sackos",
                f"{stats['total_sackos']:,}",
                help="Total last place finishes"
            )

        with col14:
            st.metric(
                "Avg Sacko Rate",
                f"{stats['avg_sacko_rate']:.1f}%",
                help="Average sacko rate per season"
            )

        with col15:
            if stats['total_seasons'] > 0:
                games_per_season = stats['total_games'] / stats['total_seasons']
                st.metric(
                    "Games/Season",
                    f"{games_per_season:.1f}",
                    help="Average games per season"
                )

        with col16:
            competitiveness = (stats['avg_champ_rate'] + stats['avg_playoff_rate']) / 2 if stats['avg_playoff_rate'] > 0 else 0
            st.metric(
                "Competitiveness",
                f"{competitiveness:.1f}%",
                help="Average of championship and playoff rates"
            )
