import pandas as pd


def get_advanced_stats(player_data, position):
    """
    Return advanced stats for career view adapted from season advanced logic.
    Data is already aggregated across all years from the database. This function
    renames columns to the weekly-friendly labels and selects position-specific
    columns (no Week or Year).

    Position parameter controls which columns to display (like weekly view).
    """
    if player_data is None or player_data.empty:
        return pd.DataFrame()

    df = player_data.copy()

    # CRITICAL: Remove duplicate columns from source data immediately
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Normalize position value (strip whitespace, handle None)
    if position:
        position = str(position).strip()
    else:
        position = "All"

    # If no position filter, show only core columns with SPAR stats
    if position == "All":
        # Rename core columns first
        core_renames = {
            "player": "Player",
            "nfl_team": "Team",
            "manager_primary": "Manager",
            "manager": "Manager",
            "points": "Points",
            "spar": "SPAR",
            "player_spar": "Player SPAR",
            "manager_spar": "Manager SPAR",
            "nfl_position": "Position",
            "fantasy_position": "Roster Slot",
        }
        df = df.rename(columns=core_renames)

        core_columns = [
            "Player",
            "Team",
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

        # Remove any duplicate columns
        if result.columns.duplicated().any():
            result = result.loc[:, ~result.columns.duplicated(keep="first")]

        # Sort by Points desc
        if "Points" in result.columns:
            result = result.sort_values(
                "Points", ascending=False, na_position="last"
            ).reset_index(drop=True)

        return result

    # Filter by position if specified (like weekly does)
    if position and position != "All" and "nfl_position" in df.columns:
        df = df[df["nfl_position"] == position].copy()
        if df.empty:
            return pd.DataFrame()

    # Coerce dtypes for common advanced columns
    numeric_columns = [
        "points",
        "spar",
        "player_spar",
        "manager_spar",
        "pass_yds",
        "pass_td",
        "passing_interceptions",
        "attempts",
        "completions",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "passing_epa",
        "passing_cpoe",
        "pacr",
        "rush_yds",
        "rush_att",
        "rush_td",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "rushing_epa",
        "rushing_2pt_conversions",
        "rec",
        "rec_yds",
        "rec_td",
        "targets",
        "receiving_fumbles",
        "receiving_fumbles_lost",
        "receiving_first_downs",
        "receiving_epa",
        "target_share",
        "wopr",
        "racr",
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
        "pts_allow",
        "dst_points_allowed",
        "points_allowed",
        "pass_yds_allowed",
        "rushing_yds_allowed",
        "total_yds_allowed",
        "fumble_recovery_opp",
        "fumble_recovery_tds",
        "kick_and_punt_ret_td",
        "3_and_outs",
        "4_dwn_stops",
    ]

    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    # Rename to weekly-friendly labels (adapted for career: no Year/Week)
    column_renames = {
        "player": "Player",
        "nfl_team": "Team",
        "manager_primary": "Manager",
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
        "rush_att": "Rush Att",
        "carries": "Rush Att",
        "rushing_tds": "Rush TD",
        "rushing_fumbles": "Rush Fum",
        "rushing_fumbles_lost": "Rush FL",
        "rushing_first_downs": "Rush 1D",
        "rushing_epa": "Rush EPA",
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
        "def_fumbles": "Fum Rec",
        "def_fumbles_forced": "FF",
        "def_safeties": "Saf",
        "def_tds": "Def TD",
        "pts_allow": "PA",
        "dst_points_allowed": "Pts Allow",
        "points_allowed": "Total PA",
        "pass_yds_allowed": "Pass Yds Allow",
        "rushing_yds_allowed": "Rush Yds Allow",
        "total_yds_allowed": "Total Yds Allow",
        "fumble_recovery_opp": "Fum Rec",
        "fumble_recovery_tds": "Fum TD",
        "kick_and_punt_ret_td": "Ret TD",
        "3_and_outs": "3&Outs",
        "4_dwn_stops": "4D Stops",
    }

    df = df.rename(columns=column_renames)

    # Ensure Position exists
    if "Position" not in df.columns:
        df["Position"] = df.get("nfl_position", "UNK")
    df["Position"] = df["Position"].fillna("UNK").astype(str)
    df["Position"] = df["Position"].replace(["", "nan", "None", "<NA>"], "UNK")

    results = []
    for pos in df["Position"].unique():
        pos_df = df[df["Position"] == pos].copy()

        if pos == "QB":
            columns = [
                "Player",
                "Team",
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
                "Rush 2PT",
            ]
            # Derived
            if "Comp" in pos_df.columns and "Pass Att" in pos_df.columns:
                comp = pd.to_numeric(pos_df["Comp"], errors="coerce")
                pass_att = pd.to_numeric(pos_df["Pass Att"], errors="coerce")
                pos_df["Comp%"] = (comp / pass_att * 100).round(1)
            if "Rush Yds" in pos_df.columns and "Rush Att" in pos_df.columns:
                pos_df["YPC"] = (
                    pd.to_numeric(pos_df["Rush Yds"], errors="coerce")
                    / pd.to_numeric(pos_df["Rush Att"], errors="coerce")
                ).round(1)

        elif pos == "RB":
            columns = [
                "Player",
                "Team",
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
            if "Rush Yds" in pos_df.columns and "Rush Att" in pos_df.columns:
                pos_df["YPC"] = (
                    pd.to_numeric(pos_df["Rush Yds"], errors="coerce")
                    / pd.to_numeric(pos_df["Rush Att"], errors="coerce")
                ).round(1)
            if "Rec Yds" in pos_df.columns and "Rec" in pos_df.columns:
                pos_df["YPR"] = (
                    pd.to_numeric(pos_df["Rec Yds"], errors="coerce")
                    / pd.to_numeric(pos_df["Rec"], errors="coerce")
                ).round(1)
            if "Rec" in pos_df.columns and "Targets" in pos_df.columns:
                pos_df["Catch%"] = (
                    pd.to_numeric(pos_df["Rec"], errors="coerce")
                    / pd.to_numeric(pos_df["Targets"], errors="coerce")
                    * 100
                ).round(1)

        elif pos in ["WR", "TE"]:
            columns = [
                "Player",
                "Team",
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
            if "Rec Yds" in pos_df.columns and "Rec" in pos_df.columns:
                pos_df["YPR"] = (
                    pd.to_numeric(pos_df["Rec Yds"], errors="coerce")
                    / pd.to_numeric(pos_df["Rec"], errors="coerce")
                ).round(1)
            if "Rec" in pos_df.columns and "Targets" in pos_df.columns:
                pos_df["Catch%"] = (
                    pd.to_numeric(pos_df["Rec"], errors="coerce")
                    / pd.to_numeric(pos_df["Targets"], errors="coerce")
                    * 100
                ).round(1)
            if "Rush Yds" in pos_df.columns and "Rush Att" in pos_df.columns:
                pos_df["YPC"] = (
                    pd.to_numeric(pos_df["Rush Yds"], errors="coerce")
                    / pd.to_numeric(pos_df["Rush Att"], errors="coerce")
                ).round(1)

        elif pos == "K":
            columns = [
                "Player",
                "Team",
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
            columns = [
                "Player",
                "Team",
                "Manager",
                "Points",
                "SPAR",
                "Player SPAR",
                "Manager SPAR",
                "Position",
                "Roster Slot",
            ]

        keep = [c for c in columns if c in pos_df.columns]
        result_df = pos_df[keep].copy()
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        results.append(result_df)

    if results:
        result = pd.concat(results, ignore_index=True)
        # Sort by Points desc only
        if "Points" in result.columns:
            result = result.sort_values(
                by="Points", ascending=False, na_position="last"
            ).reset_index(drop=True)
    else:
        result = pd.DataFrame()

    return result
