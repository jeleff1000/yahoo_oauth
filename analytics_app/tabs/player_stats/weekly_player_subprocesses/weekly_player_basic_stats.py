import pandas as pd


def get_basic_stats(player_data: pd.DataFrame, position: str) -> pd.DataFrame:
    """Return weekly player stats with only the most basic essential stats."""
    df = player_data.copy()

    # Normalize position value (strip whitespace, handle None)
    if position:
        position = str(position).strip()
    else:
        position = "All"

    # CRITICAL: Remove duplicate columns from source data immediately
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Dtypes - ensure numeric columns are actually numeric, not strings
    numeric_cols = [
        "year",
        "week",
        "points",
        "spar",
        "player_spar",
        "manager_spar",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "fg_att",
        "fg_made",
        "pat_made",
        "pat_att",
        "def_sacks",
        "def_interceptions",
        "pts_allow",
        "def_tds",
        "attempts",
        "completions",
        "carries",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Year and week as strings for display
    if "year" in df.columns:
        df["year"] = df["year"].astype("Int64").astype("string")
    if "week" in df.columns:
        df["week"] = df["week"].astype("Int64")

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
        "passing_interceptions": "Pass INT",
        "rushing_yards": "Rush Yds",
        "rushing_tds": "Rush TD",
        "carries": "Att",
        "receptions": "Rec",
        "receiving_yards": "Rec Yds",
        "receiving_tds": "Rec TD",
        "targets": "Tgt",
        "completions": "Comp",
        "attempts": "Pass Att",
        "fg_made": "FGM",
        "fg_att": "FGA",
        "pat_made": "XPM",
        "pat_att": "XPA",
        "def_sacks": "Sacks",
        "def_interceptions": "INT",
        "pts_allow": "PA",
        "def_tds": "TD",
        "opponent": "Opp Manager",
        "bye": "Bye",
        "is_started": "Started",
        "optimal_player": "Optimal",
    }

    df = df.rename(columns=column_renames)

    # CRITICAL: Remove any duplicate columns created during rename
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Columns by position - ONLY BASIC STATS (NO SPAR in basic stats!)
    if position == "QB":
        columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "Position",
            "Pass Yds",
            "Pass TD",
            "Pass INT",
            "Comp",
            "Pass Att",
            "Rush Yds",
            "Rush TD",
        ]
    elif position in ["RB", "W/R/T"]:
        columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "Position",
            "Rush Yds",
            "Rush TD",
            "Att",
            "Rec",
            "Rec Yds",
            "Rec TD",
            "Tgt",
        ]
    elif position in ["WR", "TE"]:
        columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "Position",
            "Rec",
            "Rec Yds",
            "Rec TD",
            "Tgt",
            "Rush Yds",
            "Rush TD",
        ]
    elif position == "K":
        columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "Position",
            "FGM",
            "FGA",
            "XPM",
            "XPA",
        ]
    elif position == "DEF":
        columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "Position",
            "Sacks",
            "INT",
            "PA",
            "TD",
        ]
    else:
        # No position filter - show ONLY core columns
        columns = [
            "Player",
            "Team",
            "Vs",
            "Week",
            "Year",
            "Manager",
            "Points",
            "Position",
            "Roster Slot",
        ]

    # Remove duplicates from columns list (shouldn't happen, but safeguard)
    seen = set()
    columns_unique = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            columns_unique.append(col)

    existing = [c for c in columns_unique if c in df.columns]

    # Sort by Points DESC only
    if "Points" in df.columns:
        df = df.sort_values("Points", ascending=False, na_position="last")

    result_df = df[existing].copy()

    # CRITICAL: Remove any duplicate column names to prevent React error #185
    if result_df.columns.duplicated().any():
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep="first")]

    # Ensure all column names are strings
    result_df.columns = [str(col) for col in result_df.columns]

    # Final check - verify no duplicates remain
    assert (
        not result_df.columns.duplicated().any()
    ), f"Duplicate columns detected: {result_df.columns[result_df.columns.duplicated()].tolist()}"

    return result_df
