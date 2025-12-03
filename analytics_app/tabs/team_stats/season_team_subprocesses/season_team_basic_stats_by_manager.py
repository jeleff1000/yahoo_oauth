#!/usr/bin/env python3
import pandas as pd


def get_basic_stats(team_data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns season basic team stats aggregated by manager only.
    Includes Total TD and Total Yards columns.
    """
    if team_data is None or team_data.empty:
        return pd.DataFrame()

    df = team_data.copy()

    # CRITICAL: Remove duplicate columns
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Ensure numeric columns
    numeric_cols = [
        "year",
        "points",
        "player_spar",
        "season_ppg",
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
        "def_td",
        "attempts",
        "completions",
        "rush_att",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate Total TDs and Total Yards
    df["total_tds"] = (
        pd.to_numeric(df.get("passing_tds", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("rushing_tds", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("receiving_tds", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("def_td", 0), errors="coerce").fillna(0)
    )

    df["other_tds"] = (
        pd.to_numeric(df.get("rushing_tds", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("receiving_tds", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("def_td", 0), errors="coerce").fillna(0)
    )

    df["total_yards"] = (
        pd.to_numeric(df.get("passing_yards", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("rushing_yards", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("receiving_yards", 0), errors="coerce").fillna(0)
    )

    df["rush_rec_yards"] = pd.to_numeric(
        df.get("rushing_yards", 0), errors="coerce"
    ).fillna(0) + pd.to_numeric(df.get("receiving_yards", 0), errors="coerce").fillna(0)

    # Rename columns
    column_renames = {
        "manager": "Manager",
        "year": "Year",
        "points": "Points",
        "player_spar": "SPAR",  # Use player_spar to avoid double aggregation
        "season_ppg": "PPG",
        "total_tds": "Total TD",
        "passing_tds": "Pass TD",
        "other_tds": "Other TD",
        "total_yards": "Total Yds",
        "passing_yards": "Pass Yds",
        "rush_rec_yards": "Rush+Rec Yds",
        "passing_interceptions": "INT",
        "rushing_yards": "Rush Yds",
        "rushing_tds": "Rush TD",
        "rush_att": "Att",
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
        "def_interceptions": "Def INT",
        "pts_allow": "PA",
        "def_td": "Def TD",
    }

    df = df.rename(columns=column_renames)

    # CRITICAL: Remove any duplicate columns created during rename
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Ensure Year formatting as string without commas
    if "Year" in df.columns:
        df["Year"] = (
            pd.to_numeric(df["Year"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.replace(",", "")
        )

    # Select columns - minimal default view matching the new style
    columns = [
        "Manager",
        "Year",
        "Points",
        "SPAR",
        "PPG",
        "Pass TD",
        "Other TD",
        "Pass Yds",
        "Rush+Rec Yds",
        "Rec",
    ]

    existing = [c for c in columns if c in df.columns]
    existing = list(dict.fromkeys(existing))

    # Sorting: Year desc, Points desc
    sort_cols = [c for c in ["Year", "Points"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False, False], na_position="last")

    result_df = df[existing].copy()

    # CRITICAL: Remove any duplicate column names
    if result_df.columns.duplicated().any():
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep="first")]

    result_df.columns = [str(col) for col in result_df.columns]

    assert (
        not result_df.columns.duplicated().any()
    ), f"Duplicate columns detected: {result_df.columns[result_df.columns.duplicated()].tolist()}"

    return result_df
