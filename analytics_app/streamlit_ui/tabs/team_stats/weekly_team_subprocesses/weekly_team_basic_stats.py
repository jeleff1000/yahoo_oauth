#!/usr/bin/env python3
import pandas as pd

from ..shared.table_formatting import add_derived_metrics, enhance_table_data


def get_basic_stats(team_data: pd.DataFrame, position: str = "All") -> pd.DataFrame:
    """
    Returns weekly basic team stats partitioned by fantasy_position.
    Shows aggregated stats for each manager's position group (e.g., all WRs owned by Daniel).
    """
    if team_data is None or team_data.empty:
        return pd.DataFrame()

    df = team_data.copy()

    # CRITICAL: Remove duplicate columns from source data immediately
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Filter by position if specified
    if position and position != "All" and "fantasy_position" in df.columns:
        df = df[df["fantasy_position"] == position].copy()
        if df.empty:
            return pd.DataFrame()

    # Ensure numeric columns
    numeric_cols = [
        'year', 'week', 'points', 'spar', 'player_spar', 'manager_spar', 'passing_yards', 'passing_tds', 'passing_interceptions',
        'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds',
        'targets', 'fg_att', 'fg_made', 'pat_made', 'pat_att', 'def_sacks',
        'def_interceptions', 'pts_allow', 'def_tds', 'attempts', 'completions', 'carries'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename columns to be more user-friendly
    column_renames = {
        'manager': 'Manager',
        'week': 'Week',
        'year': 'Year',
        'points': 'Points',
        'player_spar': 'SPAR',  # Use player_spar for weekly view
        'fantasy_position': 'Position',
        'passing_yards': 'Pass Yds',
        'passing_tds': 'Pass TD',
        'passing_interceptions': 'Pass INT',
        'rushing_yards': 'Rush Yds',
        'rushing_tds': 'Rush TD',
        'carries': 'Att',
        'receptions': 'Rec',
        'receiving_yards': 'Rec Yds',
        'receiving_tds': 'Rec TD',
        'targets': 'Tgt',
        'completions': 'Comp',
        'attempts': 'Pass Att',
        'fg_made': 'FGM',
        'fg_att': 'FGA',
        'pat_made': 'XPM',
        'pat_att': 'XPA',
        'def_sacks': 'Sacks',
        'def_interceptions': 'Def INT',
        'pts_allow': 'PA',
        'def_tds': 'TD',
    }

    df = df.rename(columns=column_renames)

    # CRITICAL: Remove any duplicate columns created during rename
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Ensure Year and Week formatting
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(str)
    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').astype('Int64')

    # Add derived metrics based on position (Comp%, YPC, YPR, etc.)
    df = add_derived_metrics(df, position)

    # Create combined columns for "All" position view
    if position == "All":
        # Rush+Rec Yds
        rush_yds = df['Rush Yds'].fillna(0) if 'Rush Yds' in df.columns else 0
        rec_yds = df['Rec Yds'].fillna(0) if 'Rec Yds' in df.columns else 0
        df['Rush+Rec Yds'] = rush_yds + rec_yds

        # Other TD (Rush + Rec TDs)
        rush_td = df['Rush TD'].fillna(0) if 'Rush TD' in df.columns else 0
        rec_td = df['Rec TD'].fillna(0) if 'Rec TD' in df.columns else 0
        df['Other TD'] = rush_td + rec_td

    # Columns by position (now including derived metrics!)
    if position == "QB":
        columns = ["Manager","Position","Week","Year","Points","SPAR",
                   "Pass Yds","Pass TD","Pass INT","Comp","Pass Att","Comp%","YPA","TD%","INT%",
                   "Rush Yds","Rush TD","Att"]
    elif position in ["RB","W/R/T"]:
        columns = ["Manager","Position","Week","Year","Points","SPAR",
                   "Rush Yds","Rush TD","Att","YPC",
                   "Rec","Rec Yds","Rec TD","Tgt","Catch%","YPR","YPRT"]
    elif position in ["WR","TE"]:
        columns = ["Manager","Position","Week","Year","Points","SPAR",
                   "Rec","Rec Yds","Rec TD","Tgt","Catch%","YPR","YPRT",
                   "Rush Yds","Rush TD","Att","YPC"]
    elif position == "K":
        columns = ["Manager","Position","Week","Year","Points",
                   "FGM","FGA","FG%","XPM","XPA"]
    elif position == "DEF":
        columns = ["Manager","Position","Week","Year","Points",
                   "Sacks","Def INT","PA","TD"]
    else:
        # All positions - minimal default view
        columns = ["Manager","Position","Week","Year","Points","SPAR",
                   "Pass TD","Other TD","Pass Yds","Rush+Rec Yds","Rec"]

    existing = [c for c in columns if c in df.columns]
    existing = list(dict.fromkeys(existing))

    # Sorting: Year desc, Week desc, Points desc
    sort_cols = [c for c in ['Year', 'Week', 'Points'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False, False, False], na_position='last')

    result_df = df[existing].copy()

    # CRITICAL: Remove any duplicate column names to prevent React error #185
    if result_df.columns.duplicated().any():
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')]

    result_df.columns = [str(col) for col in result_df.columns]

    assert not result_df.columns.duplicated().any(), f"Duplicate columns detected: {result_df.columns[result_df.columns.duplicated()].tolist()}"

    return result_df
