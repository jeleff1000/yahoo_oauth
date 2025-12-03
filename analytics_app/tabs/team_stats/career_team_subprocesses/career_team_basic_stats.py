#!/usr/bin/env python3
import pandas as pd


def get_basic_stats(team_data: pd.DataFrame, position: str = "All") -> pd.DataFrame:
    """
    Return basic stats for career team view.
    Shows aggregated stats for each manager's position group across all years.
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
        'total_points', 'player_spar', 'ppg_all_time', 'passing_yards', 'passing_tds', 'passing_interceptions',
        'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds',
        'targets', 'fg_att', 'fg_made', 'pat_made', 'pat_att', 'def_sacks',
        'def_interceptions', 'pts_allow', 'def_td', 'attempts', 'completions', 'rush_att',
        'fum_lost', 'games_played'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename columns
    column_renames = {
        'manager': 'Manager',
        'total_points': 'Points',
        'player_spar': 'SPAR',  # Use player_spar to avoid double aggregation
        'ppg_all_time': 'PPG',
        'fantasy_position': 'Position',
        'passing_yards': 'Pass Yds',
        'passing_tds': 'Pass TD',
        'passing_interceptions': 'Pass INT',
        'rushing_yards': 'Rush Yds',
        'rushing_tds': 'Rush TD',
        'rush_att': 'Att',
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
        'def_td': 'TD',
        'fum_lost': 'Fum',
        'games_played': 'GP'
    }

    df = df.rename(columns=column_renames)

    # CRITICAL: Remove any duplicate columns created during rename
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

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

    # Columns by position (removed GP - it's being double-aggregated at data layer)
    if position == "QB":
        columns = ["Manager","Position","Points","SPAR","PPG",
                   "Pass Yds","Pass TD","Pass INT","Comp","Pass Att","Rush Yds","Rush TD","Att","Fum"]
    elif position in ["RB","W/R/T"]:
        columns = ["Manager","Position","Points","SPAR","PPG",
                   "Rush Yds","Rush TD","Att","Rec","Rec Yds","Rec TD","Tgt","Fum"]
    elif position in ["WR","TE"]:
        columns = ["Manager","Position","Points","SPAR","PPG",
                   "Rec","Rec Yds","Rec TD","Tgt","Rush Yds","Rush TD","Att","Fum"]
    elif position == "K":
        columns = ["Manager","Position","Points","SPAR","PPG",
                   "FGM","FGA","XPM","XPA"]
    elif position == "DEF":
        columns = ["Manager","Position","Points","SPAR","PPG",
                   "Sacks","Def INT","PA","TD"]
    else:
        # All positions - minimal default view
        columns = ["Manager","Position","Points","SPAR","PPG",
                   "Pass TD","Other TD","Pass Yds","Rush+Rec Yds","Rec"]

    existing = [c for c in columns if c in df.columns]
    existing = list(dict.fromkeys(existing))

    result_df = df[existing].copy()

    # CRITICAL: Remove any duplicate column names to prevent React error #185
    if result_df.columns.duplicated().any():
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')]

    # Round numeric displays
    for c in ['Points','PPG']:
        if c in result_df.columns:
            result_df[c] = pd.to_numeric(result_df[c], errors='coerce')
            result_df[c] = result_df[c].round(2 if c == 'PPG' else 1)

    result_df.columns = [str(col) for col in result_df.columns]

    assert not result_df.columns.duplicated().any(), f"Duplicate columns detected: {result_df.columns[result_df.columns.duplicated()].tolist()}"

    return result_df
