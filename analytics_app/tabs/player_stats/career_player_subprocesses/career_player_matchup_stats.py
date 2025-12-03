import streamlit as st
import pandas as pd


class CombinedMatchupStatsViewer:
    """
    Displays matchup-related stats for career aggregated data.
    Career data is already aggregated across all years per player.
    Matches the structure and column display of weekly matchup stats.
    """

    def __init__(self, player_data):
        """player_data is career-aggregated dataframe"""
        if player_data is not None and not player_data.empty:
            # CRITICAL: Remove duplicate columns from source data immediately
            if player_data.columns.duplicated().any():
                player_data = player_data.loc[
                    :, ~player_data.columns.duplicated(keep="first")
                ]
        self.player_data = (
            player_data.copy() if player_data is not None else pd.DataFrame()
        )

    def display(self, prefix):
        df = self.player_data.copy()

        # Filter out unrostered players - matchup stats only show rostered players
        if "manager" in df.columns:
            df = df[df["manager"] != "Unrostered"].copy()
            df = df[df["manager"].notna()].copy()
            df = df[df["manager"] != ""].copy()

        if df.empty:
            st.warning("No rostered players found for matchup stats")
            return

        # Ensure correct dtypes
        numeric_cols = [
            "points",
            "win",
            "loss",
            "playoff_wins",
            "playoff_losses",
            "championships",
            "ppg_all_time",
            "fantasy_games",
            "games_started",
            "weeks_rostered",
            "weeks_on_bench",
            "weeks_on_ir",
            "optimal_player",
            "league_wide_optimal_player",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rename columns to be more user-friendly
        column_renames = {
            "player": "Player",
            "points": "Pts",
            "nfl_position": "Pos",
            "games_started": "Started",
            "weeks_rostered": "Rostered",
            "weeks_on_bench": "On Bench",
            "weeks_on_ir": "On IR",
            "win": "Wins",
            "loss": "Losses",
            "playoff_wins": "Playoff Wins",
            "playoff_losses": "Playoff Losses",
            "championships": "Championships",
            "ppg_all_time": "PPG",
            "manager": "Manager",
            "optimal_player": "Optimal",
            "league_wide_optimal_player": "League Optimal",
        }

        df = df.rename(columns=column_renames)

        # Convert counts to integers (career has total counts, not boolean)
        for col in [
            "Wins",
            "Losses",
            "Playoff Wins",
            "Playoff Losses",
            "Championships",
            "Started",
            "Rostered",
            "On Bench",
            "On IR",
            "Optimal",
            "League Optimal",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Count unique managers (for cases where player had multiple managers)
        if "Manager" in df.columns:

            def count_managers(manager_str):
                if pd.isna(manager_str) or str(manager_str).strip() == "":
                    return 0
                # Split by comma and count unique names
                names = [n.strip() for n in str(manager_str).split(",") if n.strip()]
                return len(set(names))

            df["Unique Mgrs"] = df["Manager"].apply(count_managers)

        # Column order: Player, Pts, PPG, Pos, Started, Rostered, On Bench, On IR, Optimal, League Optimal, Wins, Losses, Playoff Wins, Playoff Losses, Championships, Unique Mgrs, Manager
        display_cols = [
            "Player",
            "Pts",
            "PPG",
            "Pos",
            "Started",
            "Rostered",
            "On Bench",
            "On IR",
            "Optimal",
            "League Optimal",
            "Wins",
            "Losses",
            "Playoff Wins",
            "Playoff Losses",
            "Championships",
            "Unique Mgrs",
            "Manager",
        ]

        # Filter to only existing columns
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()

        # Ensure proper data types for display
        if "Pts" in display_df.columns:
            display_df["Pts"] = display_df["Pts"].round(2)
        if "PPG" in display_df.columns:
            display_df["PPG"] = display_df["PPG"].round(2)

        # Sort by manager ASC, points DESC
        sort_cols = []
        sort_order = []

        if "Manager" in display_df.columns:
            sort_cols.append("Manager")
            sort_order.append(True)  # ASC - alphabetical
        if "Pts" in display_df.columns:
            sort_cols.append("Pts")
            sort_order.append(False)  # DESC - highest points first

        if sort_cols:
            display_df = display_df.sort_values(
                by=sort_cols, ascending=sort_order
            ).reset_index(drop=True)

        # CRITICAL: Remove any duplicate column names to prevent React error #185
        if display_df.columns.duplicated().any():
            display_df = display_df.loc[:, ~display_df.columns.duplicated(keep="first")]

        # Configure column display
        column_config = {}

        # Numeric columns
        numeric_config = {
            "Pts": ("Pts", "%.2f", "Total career fantasy points"),
            "PPG": ("PPG", "%.2f", "Career points per game"),
            "Started": ("Started", "%d", "Total games started"),
            "Rostered": ("Rostered", "%d", "Total weeks rostered"),
            "On Bench": ("On Bench", "%d", "Total weeks on bench"),
            "On IR": ("On IR", "%d", "Total weeks on IR"),
            "Optimal": ("Optimal", "%d", "Times in optimal lineup"),
            "League Optimal": (
                "League Optimal",
                "%d",
                "Times in league-wide optimal lineup",
            ),
            "Wins": ("Wins", "%d", "Total career wins"),
            "Losses": ("Losses", "%d", "Total career losses"),
            "Playoff Wins": ("Playoff Wins", "%d", "Total playoff wins"),
            "Playoff Losses": ("Playoff Losses", "%d", "Total playoff losses"),
            "Championships": ("Championships", "%d", "Total championships won"),
            "Unique Mgrs": ("Unique Mgrs", "%d", "Number of different managers"),
        }

        for col, (label, fmt, help_text) in numeric_config.items():
            if col in display_df.columns:
                column_config[col] = st.column_config.NumberColumn(
                    label, format=fmt, help=help_text
                )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config=column_config,
        )

    def get_combined_stats(self):
        """Return the processed matchup stats dataframe without displaying it."""
        # Just call display method's logic but return the dataframe instead
        df = self.player_data.copy()

        # Filter out unrostered players - matchup stats only show rostered players
        if "manager" in df.columns:
            df = df[df["manager"] != "Unrostered"].copy()
            df = df[df["manager"].notna()].copy()
            df = df[df["manager"] != ""].copy()

        if df.empty:
            return pd.DataFrame()

        # Ensure correct dtypes
        numeric_cols = [
            "points",
            "win",
            "loss",
            "playoff_wins",
            "playoff_losses",
            "championships",
            "ppg_all_time",
            "fantasy_games",
            "games_started",
            "weeks_rostered",
            "weeks_on_bench",
            "weeks_on_ir",
            "optimal_player",
            "league_wide_optimal_player",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rename columns to be more user-friendly
        column_renames = {
            "player": "Player",
            "points": "Pts",
            "nfl_position": "Pos",
            "games_started": "Started",
            "weeks_rostered": "Rostered",
            "weeks_on_bench": "On Bench",
            "weeks_on_ir": "On IR",
            "win": "Wins",
            "loss": "Losses",
            "playoff_wins": "Playoff Wins",
            "playoff_losses": "Playoff Losses",
            "championships": "Championships",
            "ppg_all_time": "PPG",
            "manager": "Manager",
            "optimal_player": "Optimal",
            "league_wide_optimal_player": "League Optimal",
        }

        df = df.rename(columns=column_renames)

        # Convert counts to integers (career has total counts, not boolean)
        for col in [
            "Wins",
            "Losses",
            "Playoff Wins",
            "Playoff Losses",
            "Championships",
            "Started",
            "Rostered",
            "On Bench",
            "On IR",
            "Optimal",
            "League Optimal",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Count unique managers (for cases where player had multiple managers)
        if "Manager" in df.columns:

            def count_managers(manager_str):
                if pd.isna(manager_str) or str(manager_str).strip() == "":
                    return 0
                # Split by comma and count unique names
                names = [n.strip() for n in str(manager_str).split(",") if n.strip()]
                return len(set(names))

            df["Unique Mgrs"] = df["Manager"].apply(count_managers)

        # Column order: Player, Pts, PPG, Pos, Started, Rostered, On Bench, On IR, Optimal, League Optimal, Wins, Losses, Playoff Wins, Playoff Losses, Championships, Unique Mgrs, Manager
        display_cols = [
            "Player",
            "Pts",
            "PPG",
            "Pos",
            "Started",
            "Rostered",
            "On Bench",
            "On IR",
            "Optimal",
            "League Optimal",
            "Wins",
            "Losses",
            "Playoff Wins",
            "Playoff Losses",
            "Championships",
            "Unique Mgrs",
            "Manager",
        ]

        # Filter to only existing columns
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()

        # Ensure proper data types for display
        if "Pts" in display_df.columns:
            display_df["Pts"] = display_df["Pts"].round(2)
        if "PPG" in display_df.columns:
            display_df["PPG"] = display_df["PPG"].round(2)

        # Sort by manager ASC, points DESC
        sort_cols = []
        sort_order = []

        if "Manager" in display_df.columns:
            sort_cols.append("Manager")
            sort_order.append(True)  # ASC - alphabetical
        if "Pts" in display_df.columns:
            sort_cols.append("Pts")
            sort_order.append(False)  # DESC - highest points first

        if sort_cols:
            display_df = display_df.sort_values(
                by=sort_cols, ascending=sort_order
            ).reset_index(drop=True)

        # CRITICAL: Remove any duplicate column names to prevent React error #185
        if display_df.columns.duplicated().any():
            display_df = display_df.loc[:, ~display_df.columns.duplicated(keep="first")]

        return display_df
