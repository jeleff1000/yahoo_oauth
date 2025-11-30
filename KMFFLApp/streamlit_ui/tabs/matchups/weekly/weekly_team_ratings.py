import streamlit as st
import pandas as pd
from typing import List
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class WeeklyTeamRatingsViewer:
    """
    Weekly team ratings viewer showing power ratings and playoff probabilities.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Drop duplicate-named columns, keep first occurrence
        self.base_df = df.loc[:, ~df.columns.duplicated()].copy()

        # Known numeric columns for coercion/formatting
        self.numeric_cols: List[str] = [
            "team_points", "win", "loss", "opponent_points",
            "avg_seed", "p_playoffs", "p_bye", "exp_final_wins",
            "p_semis", "p_final", "p_champ", "exp_final_pf",
            "power_rating", "power rating",
            # Seed distribution probabilities
            "x1_seed", "x2_seed", "x3_seed", "x4_seed",
            "x5_seed", "x6_seed", "x7_seed", "x8_seed",
            "x9_seed", "x10_seed",
        ]
        for c in self.numeric_cols:
            if c in self.base_df.columns:
                self.base_df[c] = pd.to_numeric(self.base_df[c], errors="coerce")

        # Ensure 'year' and 'week' types if present
        for c in ["year", "week"]:
            if c in self.base_df.columns:
                self.base_df[c] = pd.to_numeric(self.base_df[c], errors="ignore")

    def _present(self, cols: List[str]) -> List[str]:
        """Return columns that exist in the dataframe."""
        return [c for c in cols if c in self.base_df.columns]

    @st.fragment
    def display(self, prefix: str = "weekly_team_ratings") -> None:
        """Display enhanced weekly team ratings with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>⭐ Weekly Team Ratings</h2>
        <p>Power ratings and playoff probability projections based on current performance</p>
        </div>
        """, unsafe_allow_html=True)

        if self.base_df.empty:
            st.info("No team ratings data available with current filters")
            return

        # View mode selector
        st.markdown("**View Mode:**")
        view_mode = st.radio(
            "Select what to display",
            ["Overview", "Seed Distribution"],
            horizontal=True,
            key=f"{prefix}_view_mode",
            help="Overview shows playoff odds, Seed Distribution shows probability of each final seed"
        )

        if view_mode == "Overview":
            self._display_overview(prefix)
        else:
            self._display_seed_distribution(prefix)

    def _display_overview(self, prefix: str):
        """Display overview with power ratings and playoff odds."""
        # Prepare data
        display_df = self._prepare_overview_data()

        if display_df.empty:
            st.info("No team ratings data available")
            return

        # Calculate summary statistics
        stats = self._calculate_stats(display_df)

        # === ENHANCED TABLE DISPLAY ===
        st.markdown(f"**Viewing {len(display_df):,} team ratings**")
        st.caption("💡 **Power Rating** measures team strength. **Playoff probabilities** show likelihood of advancing based on simulations.")
        self._render_overview_table(display_df, prefix)

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
                file_name=f"weekly_team_ratings_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _display_seed_distribution(self, prefix: str):
        """Display detailed seed probability distribution."""
        # Prepare data
        display_df = self._prepare_seed_distribution_data()

        if display_df.empty:
            st.info("No seed distribution data available")
            return

        st.markdown(f"**Viewing {len(display_df):,} team seed probabilities**")
        st.caption("💡 Each column shows the probability (%) of finishing in that playoff seed position")
        self._render_seed_distribution_table(display_df, prefix)

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
                file_name=f"seed_distribution_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_seed_csv",
                use_container_width=True
            )

    def _prepare_overview_data(self) -> pd.DataFrame:
        """Prepare overview data with power ratings and playoff odds."""
        df = self.base_df.copy()

        # Sort by power_rating if available, otherwise by team_points
        sort_candidates = ["power_rating", "power rating", "team_points"]
        sort_col = next((c for c in sort_candidates if c in df.columns), None)
        if sort_col:
            df = df.sort_values(by=sort_col, ascending=False)

        # Select columns for overview
        priority_order = [
            "manager", "week", "year",
            "power_rating", "power rating",
            "avg_seed",
            "p_playoffs", "p_bye", "p_semis", "p_final", "p_champ",
            "exp_final_wins", "exp_final_pf",
        ]

        show_cols = self._present(priority_order)
        if not show_cols:
            return pd.DataFrame()

        display_df = df[show_cols].copy()

        # Rename columns for clarity
        rename_map = {
            'manager': 'Manager',
            'week': 'Week',
            'year': 'Year',
            'power_rating': 'Power',
            'power rating': 'Power',
            'avg_seed': 'Exp Seed',
            'p_playoffs': 'Playoffs %',
            'p_bye': 'Bye %',
            'p_semis': 'Semis %',
            'p_final': 'Finals %',
            'p_champ': 'Champ %',
            'exp_final_wins': 'Exp Wins',
            'exp_final_pf': 'Exp PF',
        }
        display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

        # Format year and week
        if 'Year' in display_df.columns:
            display_df['Year'] = display_df['Year'].astype(int)
        if 'Week' in display_df.columns:
            display_df['Week'] = display_df['Week'].astype(int)

        # Ensure numeric types
        numeric_cols_to_round = ['Power', 'Exp Seed', 'Exp Wins', 'Exp PF']
        for col in numeric_cols_to_round:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

        # Probabilities should already be 0-100, but verify
        prob_cols = ['Playoffs %', 'Bye %', 'Semis %', 'Finals %', 'Champ %']
        for col in prob_cols:
            if col in display_df.columns:
                # Check if values are between 0 and 1 (need conversion)
                if display_df[col].dropna().between(0, 1).all():
                    display_df[col] = display_df[col] * 100

        return display_df

    def _prepare_seed_distribution_data(self) -> pd.DataFrame:
        """Prepare seed distribution data."""
        df = self.base_df.copy()

        # Select seed distribution columns
        seed_cols = [f"x{i}_seed" for i in range(1, 11)]
        available_seed_cols = [c for c in seed_cols if c in df.columns]

        if not available_seed_cols:
            return pd.DataFrame()

        priority_order = ["manager", "week", "year"] + available_seed_cols
        show_cols = self._present(priority_order)

        display_df = df[show_cols].copy()

        # Rename columns
        rename_map = {
            'manager': 'Manager',
            'week': 'Week',
            'year': 'Year',
        }

        # Add seed column renames
        for i in range(1, 11):
            old_name = f"x{i}_seed"
            if old_name in display_df.columns:
                rename_map[old_name] = f"Seed {i} %"

        display_df = display_df.rename(columns=rename_map)

        # Format year and week
        if 'Year' in display_df.columns:
            display_df['Year'] = display_df['Year'].astype(int)
        if 'Week' in display_df.columns:
            display_df['Week'] = display_df['Week'].astype(int)

        # Sort by most likely top seed
        if 'Seed 1 %' in display_df.columns:
            display_df = display_df.sort_values('Seed 1 %', ascending=False)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_teams = len(df)
        if total_teams == 0:
            return {}

        stats = {'total_teams': total_teams}

        if 'Power' in df.columns:
            stats['avg_power'] = df['Power'].mean()
            stats['max_power'] = df['Power'].max()
            stats['min_power'] = df['Power'].min()

        if 'Playoffs %' in df.columns:
            stats['avg_playoff'] = df['Playoffs %'].mean()
            stats['high_playoff'] = len(df[df['Playoffs %'] >= 75])
            stats['low_playoff'] = len(df[df['Playoffs %'] < 25])
            stats['locked_in'] = len(df[df['Playoffs %'] >= 99])
            stats['eliminated'] = len(df[df['Playoffs %'] <= 1])

        if 'Champ %' in df.columns:
            stats['avg_champ'] = df['Champ %'].mean()
            stats['max_champ'] = df['Champ %'].max()
            stats['title_favorite'] = df.loc[df['Champ %'].idxmax(), 'Manager'] if len(df) > 0 else None

        if 'Exp Seed' in df.columns:
            stats['avg_exp_seed'] = df['Exp Seed'].mean()

        if 'Exp Wins' in df.columns:
            stats['avg_exp_wins'] = df['Exp Wins'].mean()
            stats['max_exp_wins'] = df['Exp Wins'].max()

        return stats

    def _render_overview_table(self, df: pd.DataFrame, prefix: str):
        """Render overview table with column configuration."""

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
            'Power': st.column_config.NumberColumn(
                'Power',
                help='Power rating (higher = stronger team)',
                format='%.2f',
                width='small'
            ),
            'Exp Seed': st.column_config.NumberColumn(
                'Exp Seed',
                help='Expected playoff seed (1 = best)',
                format='%.2f',
                width='small'
            ),
            'Playoffs %': st.column_config.NumberColumn(
                'Playoffs %',
                help='Probability of making playoffs',
                format='%.1f%%',
                width='small'
            ),
            'Bye %': st.column_config.NumberColumn(
                'Bye %',
                help='Probability of getting first round bye',
                format='%.1f%%',
                width='small'
            ),
            'Semis %': st.column_config.NumberColumn(
                'Semis %',
                help='Probability of reaching semifinals',
                format='%.1f%%',
                width='small'
            ),
            'Finals %': st.column_config.NumberColumn(
                'Finals %',
                help='Probability of reaching championship game',
                format='%.1f%%',
                width='small'
            ),
            'Champ %': st.column_config.NumberColumn(
                'Champ %',
                help='Probability of winning championship',
                format='%.1f%%',
                width='small'
            ),
            'Exp Wins': st.column_config.NumberColumn(
                'Exp Wins',
                help='Expected final win total',
                format='%.2f',
                width='small'
            ),
            'Exp PF': st.column_config.NumberColumn(
                'Exp PF',
                help='Expected final points for',
                format='%.2f',
                width='small'
            ),
        }

        # Display the enhanced dataframe
        st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500  # Fixed height for better scrolling
        )

    def _render_seed_distribution_table(self, df: pd.DataFrame, prefix: str):
        """Render seed distribution table with column configuration."""

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
        }

        # Add seed columns
        for i in range(1, 11):
            col_name = f"Seed {i} %"
            if col_name in df.columns:
                column_config[col_name] = st.column_config.NumberColumn(
                    col_name,
                    help=f'Probability of finishing as {i} seed',
                    format='%.1f%%',
                    width='small'
                )

        # Display the enhanced dataframe
        st.dataframe(
            df,
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

        # Row 1: Power & Seed Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Teams",
                f"{stats['total_teams']:,}",
                help="Total team ratings analyzed"
            )

        with col2:
            if 'avg_power' in stats:
                st.metric(
                    "Avg Power",
                    f"{stats['avg_power']:.2f}",
                    help="Average power rating"
                )

        with col3:
            if 'max_power' in stats:
                st.metric(
                    "Max Power",
                    f"{stats['max_power']:.2f}",
                    help="Highest power rating"
                )

        with col4:
            if 'avg_exp_seed' in stats:
                st.metric(
                    "Avg Exp Seed",
                    f"{stats['avg_exp_seed']:.2f}",
                    help="Average expected playoff seed"
                )

        # Row 2: Playoff Probabilities
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            if 'avg_playoff' in stats:
                st.metric(
                    "Avg Playoff %",
                    f"{stats['avg_playoff']:.1f}%",
                    help="Average playoff probability"
                )

        with col6:
            if 'locked_in' in stats:
                st.metric(
                    "Locked In",
                    f"{stats['locked_in']}",
                    help="Teams with ≥99% playoff odds"
                )

        with col7:
            if 'eliminated' in stats:
                st.metric(
                    "Eliminated",
                    f"{stats['eliminated']}",
                    help="Teams with ≤1% playoff odds"
                )

        with col8:
            if 'avg_champ' in stats:
                st.metric(
                    "Avg Champ %",
                    f"{stats['avg_champ']:.1f}%",
                    help="Average championship probability"
                )

        # Row 3: Championship Race
        if 'title_favorite' in stats and stats['title_favorite']:
            st.markdown("---")
            col9, col10, col11, col12 = st.columns(4)

            with col9:
                st.metric(
                    "Title Favorite",
                    stats['title_favorite'],
                    help="Team with highest championship odds"
                )

            with col10:
                if 'max_champ' in stats:
                    st.metric(
                        "Best Champ Odds",
                        f"{stats['max_champ']:.1f}%",
                        help="Highest championship probability"
                    )

            with col11:
                if 'avg_exp_wins' in stats:
                    st.metric(
                        "Avg Exp Wins",
                        f"{stats['avg_exp_wins']:.2f}",
                        help="Average expected final wins"
                    )

            with col12:
                if 'max_exp_wins' in stats:
                    st.metric(
                        "Most Exp Wins",
                        f"{stats['max_exp_wins']:.1f}",
                        help="Highest expected win total"
                    )
