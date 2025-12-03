from __future__ import annotations

from typing import Dict, List, Callable
import pandas as pd
import streamlit as st
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class SeasonTeamRatingsViewer:
    """
    Season-level Streamlit viewer for team ratings and playoff projections.
    Aggregates weekly data to season level with proper handling of cumulative vs average metrics.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.base_df = df.loc[:, ~df.columns.duplicated()].copy()

        # Coerce numerics where applicable
        self.numeric_cols: List[str] = [
            "team_points", "win", "loss", "opponent_points",
            "avg_seed", "p_playoffs", "p_bye", "exp_final_wins",
            "p_semis", "p_final", "p_champ",
            "x1_seed", "final_playoff_seed",
            "shuffle_1_seed", "shuffle_avg_wins",
            "wins_vs_shuffle_wins", "shuffle_avg_playoffs",
            "shuffle_avg_bye", "shuffle_avg_seed",
            "seed_vs_shuffle_seed", "power_rating",
            "week", "year",
        ]
        for c in self.numeric_cols:
            if c in self.base_df.columns:
                self.base_df[c] = pd.to_numeric(self.base_df[c], errors="coerce")

        # Aggregate to season level
        self.agg_df = self._aggregate_by_season(self.base_df)

    @staticmethod
    def _last_non_null(s: pd.Series):
        """Get the last non-null value from a series."""
        s = s.dropna()
        return s.iloc[-1] if not s.empty else pd.NA

    def _aggregate_by_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate weekly data to season level with proper handling of different metric types."""
        if df.empty:
            return df.copy()

        df_sorted = df.copy()
        if "week" in df_sorted.columns:
            df_sorted = df_sorted.sort_values(["manager", "year", "week"])

        group_keys = [k for k in ["manager", "year"] if k in df_sorted.columns]
        if not group_keys:
            return df_sorted.copy()

        present = set(df_sorted.columns)

        # CUMULATIVE STATS (sum across season)
        sum_cols = [c for c in ["win", "loss", "team_points", "opponent_points"] if c in present]

        # AVERAGE STATS (mean across season - these are weekly probabilities/ratings)
        mean_cols = [c for c in [
            "power_rating",
            "exp_final_wins", "avg_seed",
            "p_playoffs", "p_bye", "p_semis", "p_final", "p_champ",
        ] if c in present]

        # STREAKS (max for winning, max for losing since it's already a positive number)
        max_cols = [c for c in ["winning_streak", "losing_streak"] if c in present]

        # END-OF-SEASON VALUES (take last week's value)
        # Shuffle simulations are already averaged across all possible schedules
        last_cols = [c for c in [
            "shuffle_1_seed", "shuffle_avg_wins", "wins_vs_shuffle_wins",
            "shuffle_avg_playoffs", "shuffle_avg_bye", "shuffle_avg_seed",
            "seed_vs_shuffle_seed",
        ] if c in present]

        agg_map: Dict[str, Callable] = {}
        agg_map.update({c: "sum" for c in sum_cols})
        agg_map.update({c: "mean" for c in mean_cols})
        agg_map.update({c: "max" for c in max_cols})
        for c in last_cols:
            agg_map[c] = self._last_non_null

        grouped = df_sorted.groupby(group_keys, dropna=False).agg(agg_map).reset_index()

        # Calculate derived metrics
        if "win" in grouped.columns and "loss" in grouped.columns:
            grouped["games"] = grouped["win"] + grouped["loss"]
            grouped["win_pct"] = (grouped["win"] / grouped["games"] * 100).round(1)

        if "team_points" in grouped.columns and "games" in grouped.columns:
            grouped["ppg"] = (grouped["team_points"] / grouped["games"]).round(2)

        if "opponent_points" in grouped.columns and "games" in grouped.columns:
            grouped["papg"] = (grouped["opponent_points"] / grouped["games"]).round(2)

        return grouped

    @st.fragment
    def display(self, prefix: str = "season_team_ratings") -> None:
        """Display season team ratings with improved formatting."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown("""
        <div class="tab-header">
        <h2>📊 Season Team Ratings</h2>
        <p>Playoff projections, power ratings, and schedule shuffle simulations aggregated by season</p>
        </div>
        """, unsafe_allow_html=True)

        if self.agg_df.empty:
            st.info("No team ratings data available")
            return

        # Sort by year (desc) then power rating (desc)
        df = self.agg_df.copy()
        sort_cols = []
        if "year" in df.columns:
            sort_cols.append("year")
        if "power_rating" in df.columns:
            sort_cols.append("power_rating")

        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=[False, False] if len(sort_cols) == 2 else False)

        # Prepare display dataframe
        display_df = self._prepare_display_df(df)

        # Configure columns
        column_config = self._get_column_config()

        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=500
        )

        # Download section
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**💾 Export Data**")
        with col2:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"season_team_ratings_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True
            )

    def _prepare_display_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and rename columns for display."""
        display_df = df.copy()

        # Define column order and rename mapping
        column_mapping = {
            # Identifiers
            'manager': 'Manager',
            'year': 'Year',

            # Season Record
            'win': 'W',
            'loss': 'L',
            'win_pct': 'Win %',

            # Season Scoring
            'team_points': 'PF',
            'opponent_points': 'PA',
            'ppg': 'PPG',
            'papg': 'PA/G',

            # Power & Projections
            'power_rating': 'Power Rating',
            'avg_seed': 'Avg Seed',
            'exp_final_wins': 'Exp Wins',

            # Playoff Probabilities (averaged across season)
            'p_playoffs': 'Playoff %',
            'p_bye': 'Bye %',
            'p_semis': 'Semis %',
            'p_final': 'Final %',
            'p_champ': 'Champ %',

            # Shuffle Simulations (alternate schedule outcomes)
            'shuffle_1_seed': 'Shuffle Top Seed',
            'shuffle_avg_wins': 'Shuffle Avg W',
            'wins_vs_shuffle_wins': 'W vs Shuffle',
            'shuffle_avg_playoffs': 'Shuffle Playoff %',
            'shuffle_avg_bye': 'Shuffle Bye %',
            'shuffle_avg_seed': 'Shuffle Avg Seed',
            'seed_vs_shuffle_seed': 'Seed vs Shuffle',

            # Streaks
            'winning_streak': 'Win Streak',
            'losing_streak': 'Loss Streak',
        }

        # Rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in display_df.columns}
        display_df = display_df.rename(columns=rename_dict)

        # Define preferred column order
        preferred_order = [
            'Manager', 'Year', 'W', 'L', 'Win %',
            'PF', 'PA', 'PPG', 'PA/G',
            'Power Rating', 'Avg Seed', 'Exp Wins',
            'Playoff %', 'Bye %', 'Semis %', 'Final %', 'Champ %',
            'Shuffle Top Seed', 'Shuffle Avg W', 'W vs Shuffle',
            'Shuffle Playoff %', 'Shuffle Bye %', 'Shuffle Avg Seed', 'Seed vs Shuffle',
            'Win Streak', 'Loss Streak',
        ]

        # Keep only columns that exist in preferred order
        existing_cols = [col for col in preferred_order if col in display_df.columns]

        # Add any remaining columns not in preferred order
        remaining_cols = [col for col in display_df.columns if col not in existing_cols]

        display_df = display_df[existing_cols + remaining_cols]

        return display_df

    def _get_column_config(self) -> Dict[str, st.column_config.Column]:
        """Get column configuration for enhanced display."""
        return {
            # Identifiers
            'Manager': st.column_config.TextColumn('Manager', help='Manager name', width='medium'),
            'Year': st.column_config.NumberColumn('Year', help='Season year', format='%d', width='small'),

            # Season Record
            'W': st.column_config.NumberColumn('W', help='Wins', format='%d', width='small'),
            'L': st.column_config.NumberColumn('L', help='Losses', format='%d', width='small'),
            'Win %': st.column_config.NumberColumn('Win %', help='Winning percentage', format='%.1f%%', width='small'),

            # Season Scoring (TOTALS)
            'PF': st.column_config.NumberColumn('PF', help='Total points for (season total)', format='%.2f', width='small'),
            'PA': st.column_config.NumberColumn('PA', help='Total points against (season total)', format='%.2f', width='small'),
            'PPG': st.column_config.NumberColumn('PPG', help='Points per game', format='%.2f', width='small'),
            'PA/G': st.column_config.NumberColumn('PA/G', help='Points against per game', format='%.2f', width='small'),

            # Power & Projections (AVERAGES)
            'Power Rating': st.column_config.NumberColumn('Power Rating', help='Average power rating across season', format='%.2f', width='small'),
            'Avg Seed': st.column_config.NumberColumn('Avg Seed', help='Average projected seed across season', format='%.2f', width='small'),
            'Exp Wins': st.column_config.NumberColumn('Exp Wins', help='Average expected final wins', format='%.2f', width='small'),

            # Playoff Probabilities (AVERAGES across season)
            'Playoff %': st.column_config.NumberColumn('Playoff %', help='Average playoff probability across season', format='%.1f%%', width='small'),
            'Bye %': st.column_config.NumberColumn('Bye %', help='Average bye probability across season', format='%.1f%%', width='small'),
            'Semis %': st.column_config.NumberColumn('Semis %', help='Average semifinal probability across season', format='%.1f%%', width='small'),
            'Final %': st.column_config.NumberColumn('Final %', help='Average final probability across season', format='%.1f%%', width='small'),
            'Champ %': st.column_config.NumberColumn('Champ %', help='Average championship probability across season', format='%.1f%%', width='small'),

            # Shuffle Simulations (what would happen with every possible schedule)
            'Shuffle Top Seed': st.column_config.NumberColumn('Shuffle Top Seed', help='Times finished 1st seed across all possible schedules', format='%d', width='small'),
            'Shuffle Avg W': st.column_config.NumberColumn('Shuffle Avg W', help='Average wins across all possible schedules', format='%.2f', width='small'),
            'W vs Shuffle': st.column_config.NumberColumn('W vs Shuffle', help='Actual wins minus shuffle average (luck factor)', format='%.2f', width='small'),
            'Shuffle Playoff %': st.column_config.NumberColumn('Shuffle Playoff %', help='Playoff % across all possible schedules', format='%.1f%%', width='small'),
            'Shuffle Bye %': st.column_config.NumberColumn('Shuffle Bye %', help='Bye % across all possible schedules', format='%.1f%%', width='small'),
            'Shuffle Avg Seed': st.column_config.NumberColumn('Shuffle Avg Seed', help='Average seed across all possible schedules', format='%.2f', width='small'),
            'Seed vs Shuffle': st.column_config.NumberColumn('Seed vs Shuffle', help='Actual seed minus shuffle average (schedule luck)', format='%.2f', width='small'),

            # Streaks
            'Win Streak': st.column_config.NumberColumn('Win Streak', help='Longest winning streak', format='%d', width='small'),
            'Loss Streak': st.column_config.NumberColumn('Loss Streak', help='Longest losing streak', format='%d', width='small'),
        }
