import pandas as pd
import numpy as np
import json
from datetime import date, datetime
from numbers import Number

# ---------------- Sanitize helpers ----------------

def _is_scalar(v):
    if isinstance(v, (Number, str, bool, type(None), pd.Timestamp, pd.Timedelta, date, datetime, np.generic)):
        return True
    try:
        return np.isscalar(v)
    except Exception:
        return False


def _sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    # normalize names & drop dups
    d.columns = [str(c) for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()]
    # stringify any non-scalar objects in object columns
    for c in d.columns:
        if d[c].dtype == "object":
            sample = d[c].head(32)
            if sample.map(lambda x: not _is_scalar(x)).any():
                def _safe(v):
                    if _is_scalar(v): return v
                    try:
                        return json.dumps(v, default=str)
                    except Exception:
                        return str(v)
                d[c] = d[c].map(_safe)
    # flatten index
    if d.index.nlevels > 1 or not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index(drop=True)
    return d


# ---------------- Basic stats (career) ----------------

def get_basic_stats(player_data: pd.DataFrame, position: str = "All") -> pd.DataFrame:
    """
    Career Basic stats aligned with Weekly Basic layout but without Week/Year.
    Uses career aggregates like total_points (Points) and ppg_all_time (PPG).

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

    # aliases
    if "nfl_position" not in df.columns and "position" in df.columns:
        df["nfl_position"] = df["position"]
    if "total_points" not in df.columns and "points" in df.columns:
        df["total_points"] = df["points"]
    if "games_played" not in df.columns and "fantasy_games" in df.columns:
        df["games_played"] = df["fantasy_games"]
    if "ppg_all_time" not in df.columns and "ppg_all_time" in df.columns:
        df["ppg_all_time"] = df.get("ppg_all_time")

    # Ensure numeric columns
    numeric_cols = [
        'points', 'total_points', 'spar', 'player_spar', 'manager_spar', 'ppg_all_time', 'passing_yards', 'passing_tds', 'passing_interceptions',
        'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds', 'targets', 'fg_att', 'fg_made',
        'pat_made', 'pat_att', 'def_sacks', 'def_interceptions', 'pts_allow', 'def_td', 'attempts', 'completions', 'rush_att', 'carries', 'fum_lost',
        'games_started', 'fantasy_games', 'games_played'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # CRITICAL: Handle duplicate mappings before renaming
    # Use manager_primary if available, otherwise manager
    if 'manager_primary' in df.columns and 'manager' not in df.columns:
        df = df.rename(columns={'manager_primary': 'manager'})

    # Use total_points if available, otherwise points
    if 'total_points' in df.columns and 'points' in df.columns:
        df = df.drop(columns=['points'])
    elif 'points' in df.columns and 'total_points' not in df.columns:
        df = df.rename(columns={'points': 'total_points'})

    # Prefer rush_att over carries for 'Att'
    if 'rush_att' in df.columns and 'carries' in df.columns:
        df = df.drop(columns=['carries'])
    elif 'carries' in df.columns and 'rush_att' not in df.columns:
        df = df.rename(columns={'carries': 'rush_att'})

    # Rename to weekly-friendly names
    # CRITICAL: Prefer passing_interceptions for Pass INT, rename def_interceptions to Def INT
    column_renames = {
        'player': 'Player',
        'nfl_team': 'Team',
        'manager': 'Manager',
        'total_points': 'Points',
        'spar': 'SPAR',
        'player_spar': 'Player SPAR',
        'manager_spar': 'Manager SPAR',
        'ppg_all_time': 'PPG',
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
        'fum_lost': 'Fum',
        'games_played': 'GP'
    }
    df = df.rename(columns=column_renames)

    # CRITICAL: Remove any duplicate columns created during rename
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Position-specific columns similar to season basic (no Week/Year/Vs, NO SPAR in basic stats!)
    if position == "QB":
        columns = ["Player","Team","Manager","Points","PPG","Position",
                   "Pass Yds","Pass TD","Pass INT","Comp","Pass Att","Rush Yds","Rush TD","Att","Fum","GP"]
    elif position in ["RB","W/R/T"]:
        columns = ["Player","Team","Manager","Points","PPG","Position",
                   "Rush Yds","Rush TD","Att","Rec","Rec Yds","Rec TD","Tgt","Fum","GP"]
    elif position in ["WR","TE"]:
        columns = ["Player","Team","Manager","Points","PPG","Position",
                   "Rec","Rec Yds","Rec TD","Tgt","Rush Yds","Rush TD","Att","Fum","GP"]
    elif position == "K":
        columns = ["Player","Team","Manager","Points","PPG","Position",
                   "FGM","FGA","XPM","XPA","GP"]
    elif position == "DEF":
        columns = ["Player","Team","Manager","Points","PPG","Position",
                   "Sacks","Def INT","PA","TD","GP"]
    else:
        # No position filter - show ONLY core columns
        columns = ["Player","Team","Manager","Points","PPG","Position","Roster Slot","GP"]

    existing = [c for c in columns if c in df.columns]

    # CRITICAL: Remove any duplicates from existing list to prevent duplicate column selection
    existing = list(dict.fromkeys(existing))

    out = df[existing].copy()

    # CRITICAL: Remove any duplicate column names to prevent React error #185
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep='first')]

    # Sort by Points DESC only
    if 'Points' in out.columns:
        out = out.sort_values('Points', ascending=False, na_position='last')

    # Round numeric displays
    for c in ['Points','PPG']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
            out[c] = out[c].round(2 if c == 'PPG' else 1)

    if 'GP' in out.columns:
        out['GP'] = pd.to_numeric(out['GP'], errors='coerce').round(0)

    # Ensure all column names are strings
    out.columns = [str(col) for col in out.columns]

    # Final check - verify no duplicates remain
    assert not out.columns.duplicated().any(), f"Duplicate columns detected: {out.columns[out.columns.duplicated()].tolist()}"

    return _sanitize_df_for_streamlit(out)
