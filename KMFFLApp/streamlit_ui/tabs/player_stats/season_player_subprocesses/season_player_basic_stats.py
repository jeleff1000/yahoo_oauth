#!/usr/bin/env python3
import pandas as pd


def get_basic_stats(player_data, position):
    """
    Return basic stats for season view.
    Data is already aggregated by player+year, so no grouping needed.
    Aligns column names and per-position columns with the Weekly basic stats view,
    but season removes Week and keeps Year.

    Position parameter controls which columns to display (like weekly view).
    """
    if player_data is None or player_data.empty:
        return pd.DataFrame()

    df = player_data.copy()

    # CRITICAL: Remove duplicate columns from source data immediately
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Filter by position if specified (like weekly does)
    if position and position != "All" and "nfl_position" in df.columns:
        df = df[df["nfl_position"] == position].copy()
        if df.empty:
            return pd.DataFrame()

    # Ensure numeric columns including season_ppg
    numeric_cols = [
        'year', 'points', 'spar', 'player_spar', 'manager_spar', 'season_ppg', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds',
        'receptions', 'receiving_yards', 'receiving_tds', 'targets', 'fg_att', 'fg_made', 'pat_made', 'pat_att',
        'def_sacks', 'def_interceptions', 'pts_allow', 'def_td', 'attempts', 'completions', 'rush_att', 'fum_lost',
        'games_started', 'fantasy_games', 'carries'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename columns to match weekly naming
    # CRITICAL: Use manager_primary if available, otherwise manager
    if 'manager_primary' in df.columns and 'manager' not in df.columns:
        df = df.rename(columns={'manager_primary': 'manager'})

    # CRITICAL: Prefer rush_att over carries for 'Att'
    if 'rush_att' in df.columns and 'carries' in df.columns:
        df = df.drop(columns=['carries'])
    elif 'carries' in df.columns and 'rush_att' not in df.columns:
        df = df.rename(columns={'carries': 'rush_att'})

    # CRITICAL: Prefer passing_interceptions for INT, rename def_interceptions to Def INT
    column_renames = {
        'player': 'Player',
        'nfl_team': 'Team',
        'year': 'Year',
        'manager': 'Manager',
        'points': 'Points',
        'spar': 'SPAR',
        'player_spar': 'Player SPAR',
        'manager_spar': 'Manager SPAR',
        'nfl_position': 'Position',
        'fantasy_position': 'Roster Slot',
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
        'opponent': 'Opp Manager',
        'started': 'Started',
        'optimal_player': 'Optimal',
        'fum_lost': 'Fum',
        'season_ppg': 'PPG'
    }

    df = df.rename(columns=column_renames)

    # CRITICAL: Remove any duplicate columns created during rename
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Ensure Year formatting as string without commas
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64').astype(str).str.replace(',', '')

    # Columns by position - include PPG and games for season stats (NO SPAR in basic stats!)
    if position == "QB":
        columns = ["Player","Team","Year","Manager","Points","PPG","Position",
                   "Pass Yds","Pass TD","Pass INT","Comp","Pass Att","Rush Yds","Rush TD","Att","Fum"]
    elif position in ["RB","W/R/T"]:
        columns = ["Player","Team","Year","Manager","Points","PPG","Position",
                   "Rush Yds","Rush TD","Att","Rec","Rec Yds","Rec TD","Tgt","Fum"]
    elif position in ["WR","TE"]:
        columns = ["Player","Team","Year","Manager","Points","PPG","Position",
                   "Rec","Rec Yds","Rec TD","Tgt","Rush Yds","Rush TD","Att","Fum"]
    elif position == "K":
        columns = ["Player","Team","Year","Manager","Points","PPG","Position",
                   "FGM","FGA","XPM","XPA"]
    elif position == "DEF":
        columns = ["Player","Team","Year","Manager","Points","PPG","Position",
                   "Sacks","Def INT","PA","TD"]
    else:
        # No position filter - show ONLY core columns
        columns = ["Player","Team","Year","Manager","Points","PPG","Position","Roster Slot"]

    existing = [c for c in columns if c in df.columns]

    # CRITICAL: Remove any duplicates from existing list to prevent duplicate column selection
    # Preserve order by using dict.fromkeys()
    existing = list(dict.fromkeys(existing))

    # Sorting: Points desc only (year is already grouped)
    if 'Points' in df.columns:
        df = df.sort_values(by='Points', ascending=False, na_position='last')

    result_df = df[existing].copy()

    # CRITICAL: Remove any duplicate column names to prevent React error #185
    if result_df.columns.duplicated().any():
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')]

    # Ensure all column names are strings
    result_df.columns = [str(col) for col in result_df.columns]

    # Final check - verify no duplicates remain
    assert not result_df.columns.duplicated().any(), f"Duplicate columns detected: {result_df.columns[result_df.columns.duplicated()].tolist()}"

    return result_df
