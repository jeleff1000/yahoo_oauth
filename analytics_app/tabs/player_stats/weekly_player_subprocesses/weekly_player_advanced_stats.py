import pandas as pd


def get_advanced_stats(
    player_data: pd.DataFrame, position: str = "All"
) -> pd.DataFrame:
    """
    Returns weekly advanced stats partitioned by nfl_position (actual NFL position).
    Uses nfl_position instead of fantasy_position since it goes back to 1999 and is more relevant for stats.

    Args:
        player_data: DataFrame with player stats
        position: Optional position filter - when "All" or None, shows only core columns with SPAR
    """

    df = player_data.copy()

    # Normalize position value (strip whitespace, handle None)
    if position:
        position = str(position).strip()
    else:
        position = "All"

    # CRITICAL: Remove duplicate columns from source data immediately
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Coerce dtypes - ensure ALL numeric columns are numeric before any processing
    numeric_columns = [
        "year",
        "week",
        "points",
        "spar",
        "player_spar",
        "manager_spar",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "attempts",
        "completions",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "passing_epa",
        "passing_cpoe",
        "pacr",
        "rushing_yards",
        "carries",
        "rushing_tds",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "rushing_epa",
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "receiving_fumbles",
        "receiving_fumbles_lost",
        "receiving_first_downs",
        "receiving_epa",
        "target_share",
        "wopr",
        "racr",
        "receiving_2pt_conversions",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "air_yards_share",
        "fg_made",
        "fg_att",
        "fg_pct",
        "fg_long",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_miss",
        "pat_made",
        "pat_att",
        "pat_missed",
        "def_sacks",
        "def_sack",
        "def_sack_yards",
        "def_qb_hits",
        "def_interceptions",
        "def_interception_yards",
        "def_pass_defended",
        "def_tackles_solo",
        "def_tackle_assists",
        "def_tackles_with_assist",
        "def_tackles_for_loss",
        "def_tackles_for_loss_yards",
        "def_fumbles",
        "def_fumbles_forced",
        "def_safeties",
        "def_tds",
        "def_td",
        "pts_allow",
        "dst_points_allowed",
        "points_allowed",
        "pass_yds_allowed",
        "passing_yds_allowed",
        "rushing_yds_allowed",
        "total_yds_allowed",
        "def_yds_allow",
        "fumble_recovery_opp",
        "fumble_recovery_tds",
        "kick_and_punt_ret_td",
        "3_and_outs",
        "4_dwn_stops",
    ]

    for col in numeric_columns:
        if col in df.columns:
            try:
                if isinstance(df[col], pd.Series):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    # Coerce dtypes
    if "year" in df.columns:
        df["year"] = df["year"].astype(str)
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    if "points" in df.columns:
        df["points"] = pd.to_numeric(df["points"], errors="coerce")

    # Rename columns to be more user-friendly
    column_renames = {
        "player": "Player",
        "nfl_team": "Team",
        "opponent_nfl_team": "Vs",
        "week": "Week",
        "year": "Year",
        "manager": "Manager",
        "points": "Points",
        "spar": "SPAR",
        "player_spar": "Player SPAR",
        "manager_spar": "Manager SPAR",
        "nfl_position": "Position",
        "fantasy_position": "Roster Slot",
        "passing_yards": "Pass Yds",
        "passing_tds": "Pass TD",
        "passing_interceptions": "INT Thrown",
        "attempts": "Pass Att",
        "completions": "Comp",
        "passing_air_yards": "Air Yds",
        "passing_yards_after_catch": "Pass YAC",
        "passing_first_downs": "Pass 1D",
        "passing_epa": "Pass EPA",
        "passing_cpoe": "CPOE",
        "pacr": "PACR",
        "rushing_yards": "Rush Yds",
        "carries": "Rush Att",
        "rushing_tds": "Rush TD",
        "rushing_fumbles": "Rush Fum",
        "rushing_fumbles_lost": "Rush FL",
        "rushing_first_downs": "Rush 1D",
        "rushing_epa": "Rush EPA",
        "passing_2pt_conversions": "Pass 2PT",
        "rushing_2pt_conversions": "Rush 2PT",
        "receptions": "Rec",
        "receiving_yards": "Rec Yds",
        "receiving_tds": "Rec TD",
        "targets": "Targets",
        "receiving_fumbles": "Rec Fum",
        "receiving_fumbles_lost": "Rec FL",
        "receiving_first_downs": "Rec 1D",
        "receiving_epa": "Rec EPA",
        "target_share": "Tgt %",
        "wopr": "WOPR",
        "racr": "RACR",
        "receiving_2pt_conversions": "Rec 2PT",
        "receiving_air_yards": "Rec Air Yds",
        "receiving_yards_after_catch": "Rec YAC",
        "air_yards_share": "Air Yds %",
        "fg_made": "FGM",
        "fg_att": "FGA",
        "fg_pct": "FG%",
        "fg_long": "Long",
        "fg_made_0_19": "FG 0-19",
        "fg_made_20_29": "FG 20-29",
        "fg_made_30_39": "FG 30-39",
        "fg_made_40_49": "FG 40-49",
        "fg_made_50_59": "FG 50+",
        "fg_miss": "FG Miss",
        "pat_made": "XPM",
        "pat_att": "XPA",
        "pat_missed": "XP Miss",
        "def_sacks": "Sacks",
        "def_sack": "Sacks",  # Same as above - pandas will handle
        "def_sack_yards": "Sack Yds",
        "def_qb_hits": "QB Hits",
        "def_interceptions": "INT",
        "def_interception_yards": "INT Yds",
        "def_pass_defended": "PD",
        "def_tackles_solo": "Solo Tkl",
        "def_tackle_assists": "Ast Tkl",
        "def_tackles_with_assist": "Total Tkl",
        "def_tackles_for_loss": "TFL",
        "def_tackles_for_loss_yards": "TFL Yds",
        "def_fumbles": "Def Fum Rec",  # Changed from 'Fum Rec' to avoid duplicate
        "def_fumbles_forced": "FF",
        "def_safeties": "Saf",
        "def_tds": "Def TD",
        "def_td": "Def TD",  # Same as above - pandas will handle
        "pts_allow": "PA",
        "dst_points_allowed": "Pts Allow",
        "points_allowed": "Total PA",
        "pass_yds_allowed": "Pass Yds Allow",
        "passing_yds_allowed": "Pass Yds Allow",  # Same as above - pandas will handle
        "rushing_yds_allowed": "Rush Yds Allow",
        "total_yds_allowed": "Total Yds Allow",
        "def_yds_allow": "Total Yds Allow",  # Same as above - pandas will handle
        "fumble_recovery_opp": "Fum Rec",
        "fumble_recovery_tds": "Fum TD",
        "kick_and_punt_ret_td": "Ret TD",
        "3_and_outs": "3&Outs",
        "4_dwn_stops": "4D Stops",
    }

    df = df.rename(columns=column_renames)

    # After renaming, ensure the renamed numeric columns are still numeric
    renamed_numeric_cols = [
        "Week",
        "Points",
        "Pass Yds",
        "Pass TD",
        "Pass Att",
        "Comp",
        "Air Yds",
        "Pass YAC",
        "Pass 1D",
        "Pass EPA",
        "CPOE",
        "PACR",
        "Rush Yds",
        "Rush Att",
        "Rush TD",
        "Rush Fum",
        "Rush FL",
        "Rush 1D",
        "Rush EPA",
        "Rec",
        "Rec Yds",
        "Rec TD",
        "Targets",
        "Rec Fum",
        "Rec FL",
        "Rec 1D",
        "Rec EPA",
        "Tgt %",
        "WOPR",
        "RACR",
        "Rec Air Yds",
        "Rec YAC",
        "Air Yds %",
        "FGM",
        "FGA",
        "FG%",
        "Long",
        "FG 0-19",
        "FG 20-29",
        "FG 30-39",
        "FG 40-49",
        "FG 50+",
        "FG Miss",
        "XPM",
        "XPA",
        "XP Miss",
        "Sacks",
        "Sack Yds",
        "QB Hits",
        "INT",
        "INT Yds",
        "PD",
        "Solo Tkl",
        "Ast Tkl",
        "Total Tkl",
        "TFL",
        "TFL Yds",
        "Fum Rec",
        "FF",
        "Fum TD",
        "Saf",
        "Def TD",
        "Ret TD",
        "PA",
        "Total PA",
        "Pass Yds Allow",
        "Rush Yds Allow",
        "Total Yds Allow",
        "3&Outs",
        "4D Stops",
    ]

    for col in renamed_numeric_cols:
        if col in df.columns:
            try:
                if isinstance(df[col], pd.Series):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    # Use nfl_position for grouping (goes back to 1999, not dependent on fantasy roster)
    if "Position" not in df.columns:
        df["Position"] = "UNK"
    else:
        df["Position"] = df["Position"].fillna("UNK").astype(str)
        df["Position"] = df["Position"].replace(["", "nan", "None", "<NA>"], "UNK")

    # If no position filter, show only core columns with SPAR stats
    if position == "All":
        core_columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "SPAR",
            "Player SPAR",
            "Manager SPAR",
            "Position",
            "Roster Slot",
        ]
        existing_cols = [c for c in core_columns if c in df.columns]
        result = df[existing_cols].copy()
        result = result.sort_values("Points", ascending=False, na_position="last")
        return result

    # Process each position separately to avoid index issues
    results = []

    for pos in df["Position"].unique():
        pos_df = df[df["Position"] == pos].copy()

        if pos == "QB":
            columns = [
                "Player",
                "Team",
                "Vs",
                "Week",
                "Year",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
                "Pass Yds",
                "Pass TD",
                "INT Thrown",
                "Comp",
                "Pass Att",
                "Comp%",
                "Air Yds",
                "Pass YAC",
                "Pass 1D",
                "Pass EPA",
                "CPOE",
                "PACR",
                "Rush Yds",
                "Rush Att",
                "YPC",
                "Rush TD",
                "Rush Fum",
                "Rush FL",
                "Rush 1D",
                "Rush EPA",
                "Pass 2PT",
                "Rush 2PT",
            ]
            # Calculate derived stats
            if "Comp" in pos_df.columns and "Pass Att" in pos_df.columns:
                comp = pd.to_numeric(pos_df["Comp"], errors="coerce")
                pass_att = pd.to_numeric(pos_df["Pass Att"], errors="coerce")
                pos_df["Comp%"] = (comp / pass_att * 100).round(1)
            if "Rush Yds" in pos_df.columns and "Rush Att" in pos_df.columns:
                rush_yds = pd.to_numeric(pos_df["Rush Yds"], errors="coerce")
                rush_att = pd.to_numeric(pos_df["Rush Att"], errors="coerce")
                pos_df["YPC"] = (rush_yds / rush_att).round(1)

        elif pos == "RB":
            columns = [
                "Player",
                "Team",
                "Vs",
                "Week",
                "Year",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
                "Rush Yds",
                "Rush Att",
                "YPC",
                "Rush TD",
                "Rush Fum",
                "Rush FL",
                "Rush 1D",
                "Rush EPA",
                "Rush 2PT",
                "Rec",
                "Rec Yds",
                "YPR",
                "Rec TD",
                "Targets",
                "Catch%",
                "Rec Fum",
                "Rec FL",
                "Rec 1D",
                "Rec EPA",
                "Rec 2PT",
                "Tgt %",
                "WOPR",
                "RACR",
                "Rec Air Yds",
                "Rec YAC",
                "Air Yds %",
            ]
            # Calculate derived stats
            if "Rush Yds" in pos_df.columns and "Rush Att" in pos_df.columns:
                rush_yds = pd.to_numeric(pos_df["Rush Yds"], errors="coerce")
                rush_att = pd.to_numeric(pos_df["Rush Att"], errors="coerce")
                pos_df["YPC"] = (rush_yds / rush_att).round(1)
            if "Rec Yds" in pos_df.columns and "Rec" in pos_df.columns:
                rec_yds = pd.to_numeric(pos_df["Rec Yds"], errors="coerce")
                rec = pd.to_numeric(pos_df["Rec"], errors="coerce")
                pos_df["YPR"] = (rec_yds / rec).round(1)
            if "Rec" in pos_df.columns and "Targets" in pos_df.columns:
                rec = pd.to_numeric(pos_df["Rec"], errors="coerce")
                targets = pd.to_numeric(pos_df["Targets"], errors="coerce")
                pos_df["Catch%"] = (rec / targets * 100).round(1)

        elif pos in ["WR", "TE"]:
            columns = [
                "Player",
                "Team",
                "Vs",
                "Week",
                "Year",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
                "Rec",
                "Rec Yds",
                "YPR",
                "Rec TD",
                "Targets",
                "Catch%",
                "Rec Fum",
                "Rec FL",
                "Rec 1D",
                "Rec EPA",
                "Rec 2PT",
                "Tgt %",
                "WOPR",
                "RACR",
                "Rec Air Yds",
                "Rec YAC",
                "Air Yds %",
                "Rush Yds",
                "Rush Att",
                "YPC",
                "Rush TD",
                "Rush 2PT",
            ]
            # Calculate derived stats
            if "Rec Yds" in pos_df.columns and "Rec" in pos_df.columns:
                rec_yds = pd.to_numeric(pos_df["Rec Yds"], errors="coerce")
                rec = pd.to_numeric(pos_df["Rec"], errors="coerce")
                pos_df["YPR"] = (rec_yds / rec).round(1)
            if "Rec" in pos_df.columns and "Targets" in pos_df.columns:
                rec = pd.to_numeric(pos_df["Rec"], errors="coerce")
                targets = pd.to_numeric(pos_df["Targets"], errors="coerce")
                pos_df["Catch%"] = (rec / targets * 100).round(1)
            if "Rush Yds" in pos_df.columns and "Rush Att" in pos_df.columns:
                rush_yds = pd.to_numeric(pos_df["Rush Yds"], errors="coerce")
                rush_att = pd.to_numeric(pos_df["Rush Att"], errors="coerce")
                pos_df["YPC"] = (rush_yds / rush_att).round(1)

        elif pos == "K":
            columns = [
                "Player",
                "Team",
                "Vs",
                "Week",
                "Year",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
                "FGM",
                "FGA",
                "FG%",
                "Long",
                "FG Miss",
                "FG 0-19",
                "FG 20-29",
                "FG 30-39",
                "FG 40-49",
                "FG 50+",
                "XPM",
                "XPA",
                "XP Miss",
            ]

        elif pos == "DEF":
            columns = [
                "Player",
                "Team",
                "Vs",
                "Week",
                "Year",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
                "Sacks",
                "Sack Yds",
                "QB Hits",
                "INT",
                "INT Yds",
                "PD",
                "Solo Tkl",
                "Ast Tkl",
                "Total Tkl",
                "TFL",
                "TFL Yds",
                "Def Fum Rec",
                "Fum Rec",
                "FF",
                "Fum TD",
                "Saf",
                "Def TD",
                "Ret TD",
                "PA",
                "Total PA",
                "Pass Yds Allow",
                "Rush Yds Allow",
                "Total Yds Allow",
                "3&Outs",
                "4D Stops",
            ]
        else:
            # For other positions or unknown, show basic columns
            columns = [
                "Player",
                "Team",
                "Vs",
                "Week",
                "Year",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
            ]

        # Select only columns that exist
        keep = [c for c in columns if c in pos_df.columns]

        # Remove duplicate columns to avoid concat errors
        result_df = pos_df[keep]

        # Ensure no duplicate column names
        if result_df.columns.duplicated().any():
            # Keep only the first occurrence of each column
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        results.append(result_df)

    # Concatenate all positions
    if results:
        result = pd.concat(results, ignore_index=True)

        # Sort by Points DESC only
        if "Points" in result.columns:
            result = result.sort_values("Points", ascending=False, na_position="last")
            result = result.reset_index(drop=True)

        # CRITICAL: Remove any duplicate column names to prevent React error #185
        if result.columns.duplicated().any():
            result = result.loc[:, ~result.columns.duplicated(keep="first")]
    else:
        result = pd.DataFrame()

    return result
