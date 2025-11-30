from __future__ import annotations

from typing import Dict, List, Callable
import pandas as pd
import streamlit as st


class CareerTeamRatingsViewer:
    """
    Career-level Streamlit viewer for team ratings.
    - Weekly -> season: apply season rules (last-week-of-season for specific fields).
    - Season -> career: sum or average per requirements.
      * Sum across seasons: wins_vs_shuffle_wins, seed_vs_shuffle_seed
      * Average across seasons: Final Playoff Seed
    - Hides `week`, `year`, and `opponent` from display.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Deduplicate same-named columns
        self.base_df = df.loc[:, ~df.columns.duplicated()].copy()

        # Coerce numerics
        self.numeric_cols: List[str] = [
            "team_points", "win", "loss", "opponent_points",
            "avg_seed", "p_playoffs", "p_bye", "exp_final_wins",
            "p_semis", "p_final", "p_champ",
            "x1_seed", "Final Playoff Seed",
            "shuffle_1_seed", "shuffle_avg_wins",
            "wins_vs_shuffle_wins", "shuffle_avg_playoffs",
            "shuffle_avg_bye", "shuffle_avg_seed",
            "seed_vs_shuffle_seed", "power_rating", "power rating",
            "week", "year",
        ]
        for c in self.numeric_cols:
            if c in self.base_df.columns:
                self.base_df[c] = pd.to_numeric(self.base_df[c], errors="coerce")

        # Columns not to show at career level
        self.hidden_cols: List[str] = ["Opponent Week", "OpponentYear", "week", "opponent", "year"]

        # Leading display order (career)
        self.leading_order: List[str] = [
            "manager", "win", "loss",
            "team_points", "opponent_points", "power_rating", "power rating",
        ]

        # Remaining original known columns (career, excludes hidden)
        self.secondary_order: List[str] = [
            "Winning Streak", "Losing Streak",
            "avg_seed", "Final Playoff Seed",
            "p_playoffs", "p_bye", "exp_final_wins", "p_semis", "p_final", "p_champ",
            "shuffle_1_seed", "shuffle_avg_wins", "wins_vs_shuffle_wins",
            "shuffle_avg_playoffs", "shuffle_avg_bye", "shuffle_avg_seed",
            "seed_vs_shuffle_seed",
        ]

        # Build career aggregation
        season_df = self._aggregate_by_season(self.base_df)
        self.agg_df = self._aggregate_career(season_df)

    @staticmethod
    def _last_non_null(s: pd.Series):
        s = s.dropna()
        return s.iloc[-1] if not s.empty else pd.NA

    def _aggregate_by_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply season-level rules per [manager, year]."""
        if df.empty:
            return df.copy()

        df_sorted = df.copy()
        sort_keys = [k for k in ["manager", "year", "week"] if k in df_sorted.columns]
        if sort_keys:
            df_sorted = df_sorted.sort_values(sort_keys)

        group_keys = [k for k in ["manager", "year"] if k in df_sorted.columns]
        if not group_keys:
            return df_sorted.copy()

        present = set(df_sorted.columns)

        # Sums
        sum_cols = [c for c in ["win", "loss"] if c in present]

        # Means (season averages)
        mean_cols = [c for c in [
            "team_points", "opponent_points", "power_rating", "power rating",
            "exp_final_wins", "avg_seed",
            "p_playoffs", "p_bye", "p_semis", "p_final", "p_champ",
        ] if c in present]
        if "shuffle_1_seed" in present:
            mean_cols.append("shuffle_1_seed")

        # Streaks
        max_cols = [c for c in ["Winning Streak"] if c in present]
        min_cols = [c for c in ["Losing Streak"] if c in present]

        # Last-week-of-season values
        last_cols = [c for c in [
            "Final Playoff Seed",
            "x1_seed",  # optional fallback
            "shuffle_avg_wins",
            "wins_vs_shuffle_wins",
            "shuffle_avg_playoffs", "shuffle_avg_bye", "shuffle_avg_seed",
            "seed_vs_shuffle_seed",
        ] if c in present]

        agg_map: Dict[str, Callable] = {}
        agg_map.update({c: "sum" for c in sum_cols})
        agg_map.update({c: "mean" for c in mean_cols})
        agg_map.update({c: "max" for c in max_cols})
        agg_map.update({c: "min" for c in min_cols})
        for c in last_cols:
            agg_map[c] = self._last_non_null

        grouped = df_sorted.groupby(group_keys, dropna=False).agg(agg_map).reset_index()

        # If Final Playoff Seed missing but x1_seed exists, create from x1_seed
        if "Final Playoff Seed" not in grouped.columns and "x1_seed" in grouped.columns:
            grouped["Final Playoff Seed"] = grouped["x1_seed"]

        return grouped

    def _aggregate_career(self, season_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate season-level rows to career-level per manager."""
        if season_df.empty:
            return season_df.copy()

        present = set(season_df.columns)
        group_keys = ["manager"]

        # Sums across seasons (per requirement)
        sum_cols = [c for c in [
            "win", "loss",
            "wins_vs_shuffle_wins",
            "seed_vs_shuffle_seed",
        ] if c in present]

        # Averages across seasons
        mean_cols = [c for c in [
            "team_points", "opponent_points", "power_rating", "power rating",
            "exp_final_wins", "avg_seed",
            "p_playoffs", "p_bye", "p_semis", "p_final", "p_champ",
            "Final Playoff Seed",
            "shuffle_avg_wins", "shuffle_avg_playoffs", "shuffle_avg_bye", "shuffle_avg_seed",
            "shuffle_1_seed",
        ] if c in present]

        max_cols = [c for c in ["Winning Streak"] if c in present]
        min_cols = [c for c in ["Losing Streak"] if c in present]

        agg_map: Dict[str, Callable] = {}
        agg_map.update({c: "sum" for c in sum_cols})
        agg_map.update({c: "mean" for c in mean_cols})
        agg_map.update({c: "max" for c in max_cols})
        agg_map.update({c: "min" for c in min_cols})

        grouped = season_df.groupby(group_keys, dropna=False).agg(agg_map).reset_index()
        return grouped

    def _present(self, df: pd.DataFrame, cols: List[str]) -> List[str]:
        return [c for c in cols if c in df.columns]

    def _prob_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in ["p_playoffs", "p_bye", "p_semis", "p_final", "p_champ"] if c in df.columns]

    @st.fragment
    def display(self, prefix: str = "career_team_ratings") -> None:
        st.subheader("Career Team Ratings")

        if self.agg_df.empty:
            st.info("No data available.")
            return

        # Internal sort
        sort_candidates = ["power_rating", "power rating", "team_points"]
        sort_col = next((c for c in sort_candidates if c in self.agg_df.columns), None)
        df = self.agg_df.copy()
        if sort_col:
            df = df.sort_values(by=sort_col, ascending=False)

        # Only known columns: leading, then secondary; hide unwanted
        leading = self._present(df, self.leading_order)
        secondary = [c for c in self._present(df, self.secondary_order) if c not in leading]
        show_cols = [c for c in (leading + secondary) if c not in self.hidden_cols]

        if not show_cols:
            st.warning("No known columns to display.")
            return

        df_show = df[show_cols].copy()

        # Percent formatting for probabilities
        prob_cols = self._prob_cols(df_show)
        column_config: Dict[str, st.column_config.Column] = {}
        for c in prob_cols:
            column_config[c] = st.column_config.NumberColumn(c, format="%.1f%%")
            if df_show[c].dropna().between(0, 1).all():
                df_show[c] = df_show[c] * 100.0

        # Numeric formatting
        for c in ["team_points", "power_rating", "power rating", "exp_final_wins", "avg_seed",
                  "shuffle_avg_wins", "shuffle_avg_playoffs", "shuffle_avg_bye", "shuffle_avg_seed"]:
            if c in df_show.columns:
                column_config[c] = st.column_config.NumberColumn(c, format="%.2f")
        for c in ["win", "loss", "wins_vs_shuffle_wins", "shuffle_1_seed", "seed_vs_shuffle_seed",
                  "opponent_points", "Final Playoff Seed"]:
            if c in df_show.columns:
                column_config[c] = st.column_config.NumberColumn(c, format="%d")

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            column_config=column_config or None,
        )