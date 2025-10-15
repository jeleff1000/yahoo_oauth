#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd

# ---------------------------
# Paths (relative to this file)
# ---------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_ROOT = THIS_FILE.parent
DATA_ROOT = SCRIPT_ROOT.parent.parent / "fantasy_football_data" / "player_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Config / Maps
# ---------------------------
TEAM_CODE_MAP: Dict[str, str] = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV"}

TEAM_LOGO_MAP: Dict[str, str] = {
    "ARI": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/Arizona_Cardinals_logo.svg/179px-Arizona_Cardinals_logo.svg.png",
    "ATL": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Atlanta_Falcons_logo.svg/192px-Atlanta_Falcons_logo.svg.png",
    "BAL": "https://upload.wikimedia.org/wikipedia/en/thumb/1/16/Baltimore_Ravens_logo.svg/193px-Baltimore_Ravens_logo.svg.png",
    "BUF": "https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Buffalo_Bills_logo.svg/189px-Buffalo_Bills_logo.svg.png",
    "CAR": "https://upload.wikimedia.org/wikipedia/en/thumb/1/1c/Carolina_Panthers_logo.svg/100px-Carolina_Panthers_logo.svg.png",
    "CHI": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Chicago_Bears_logo.svg/100px-Chicago_Bears_logo.svg.png",
    "CIN": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Cincinnati_Bengals_logo.svg/100px-Cincinnati_Bengals_logo.svg.png",
    "CLE": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d9/Cleveland_Browns_logo.svg/100px-Cleveland_Browns_logo.svg.png",
    "DAL": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Dallas_Cowboys.svg/100px-Dallas_Cowboys.svg.png",
    "DEN": "https://upload.wikimedia.org/wikipedia/en/thumb/4/44/Denver_Broncos_logo.svg/100px-Denver_Broncos_logo.svg.png",
    "DET": "https://upload.wikimedia.org/wikipedia/en/thumb/7/71/Detroit_Lions_logo.svg/100px-Detroit_Lions_logo.svg.png",
    "GB":  "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Green_Bay_Packers_logo.svg/100px-Green_Bay_Packers_logo.svg.png",
    "HOU": "https://upload.wikimedia.org/wikipedia/en/thumb/2/28/Houston_Texans_logo.svg/100px-Houston_Texans_logo.svg.png",
    "IND": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Indianapolis_Colts_logo.svg/100px-Indianapolis_Colts_logo.svg.png",
    "JAX": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/Jacksonville_Jaguars_logo.svg/100px-Jacksonville_Jaguars_logo.svg.png",
    "KC":  "https://upload.wikimedia.org/wikipedia/en/thumb/e/e1/Kansas_City_Chiefs_logo.svg/100px-Kansas_City_Chiefs_logo.svg.png",
    "LAC": "https://upload.wikimedia.org/wikipedia/en/thumb/7/72/NFL_Chargers_logo.svg/100px-NFL_Chargers_logo.svg.png",
    "LAR": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/Los_Angeles_Rams_logo.svg/100px-Los_Angeles_Rams_logo.svg.png",
    "MIA": "https://upload.wikimedia.org/wikipedia/en/thumb/3/37/Miami_Dolphins_logo.svg/100px-Miami_Dolphins_logo.svg.png",
    "MIN": "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Minnesota_Vikings_logo.svg/98px-Minnesota_Vikings_logo.svg.png",
    "NE":  "https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/New_England_Patriots_logo.svg/100px-New_England_Patriots_logo.svg.png",
    "NO":  "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/New_Orleans_Saints_logo.svg/98px-New_Orleans_Saints_logo.svg.png",
    "NYG": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/New_York_Giants_logo.svg/100px-New_York_Giants_logo.svg.png",
    "NYJ": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/New_York_Jets_logo.svg/100px-New_York_Jets_logo.svg.png",
    "LV":  "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Las_Vegas_Raiders_logo.svg/150px-Las_Vegas_Raiders_logo.svg.png",
    "PHI": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Philadelphia_Eagles_logo.svg/100px-Philadelphia_Eagles_logo.svg.png",
    "PIT": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Pittsburgh_Steelers_logo.svg/100px-Pittsburgh_Steelers_logo.svg.png",
    "SF":  "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/San_Francisco_49ers_logo.svg/100px-San_Francisco_49ers_logo.svg.png",
    "SEA": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Seattle_Seahawks_logo.svg/100px-Seattle_Seahawks_logo.svg.png",
    "TB":  "https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/Tampa_Bay_Buccaneers_logo.svg/100px-Tampa_Bay_Buccaneers_logo.svg.png",
    "TEN": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c1/Tennessee_Titans_logo.svg/100px-Tennessee_Titans_logo.svg.png",
    "WAS": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Washington_football_team_wlogo.svg/1024px-Washington_football_team_wlogo.svg.png",
}

DISPLAY_BASE: Dict[str, str] = {
    "ARI": "Arizona","ATL": "Atlanta","BAL": "Baltimore","BUF": "Buffalo",
    "CAR": "Carolina","CHI": "Chicago","CIN": "Cincinnati","CLE": "Cleveland",
    "DAL": "Dallas","DEN": "Denver","DET": "Detroit","GB": "Green Bay",
    "HOU": "Houston","IND": "Indianapolis","JAX": "Jacksonville","KC": "Kansas City",
    "LAC": "Chargers","LAR": "Rams","MIA": "Miami","MIN": "Minnesota",
    "NE": "New England","NO": "New Orleans","NYG": "Giants","NYJ": "Jets",
    "LV": "Las Vegas","PHI": "Philadelphia","PIT": "Pittsburgh","SF": "San Francisco",
    "SEA": "Seattle","TB": "Tampa Bay","TEN": "Tennessee","WAS": "Washington",
}

BLOCKED_COL_RENAMES = {
    "fg_blocked": "blocked_fg",
    "fg_blocked_list": "blocked_fg_list",
    "fg_blocked_distance": "blocked_fg_distance",
    "pat_blocked": "blocked_pat",
    "gwfg_blocked": "blocked_gwfg",
}

# Opponent→this row mappings
ALLOWED_MAP = {
    "passing_tds": "passing_tds_allowed",
    "rushing_tds": "rushing_tds_allowed",
    "receiving_tds": "receiving_tds_allowed",
    "special_teams_tds": "special_teams_tds_allowed",
    "def_tds": "def_tds_allowed",
    "fg_made": "fg_made_allowed",
    "pat_made": "pat_made_allowed",
    "passing_2pt_conversions": "passing_2pt_conversions_allowed",
    "rushing_2pt_conversions": "rushing_2pt_conversions_allowed",
    "receiving_2pt_conversions": "receiving_2pt_conversions_allowed",
    # NEW: yards allowed
    "passing_yards": "pass_yds_allowed",
    "rushing_yards": "rushing_yds_allowed",
}

REQUESTED_COLS: List[str] = [
    # identifiers
    "year","week","nfl_team","season_type","opponent_nfl_team","nfl_position",
    # original defensive/special teams columns
    "special_teams_tds","def_tackles_solo","def_tackles_with_assist","def_tackle_assists",
    "def_tackles_for_loss","def_tackles_for_loss_yards","def_fumbles_forced",
    "def_sacks","def_sack_yards","def_qb_hits","def_interceptions","def_interception_yards",
    "def_pass_defended","def_tds","def_fumbles","def_safeties","misc_yards",
    "fumble_recovery_own","fumble_recovery_yards_own","fumble_recovery_opp",
    "fumble_recovery_yards_opp","fumble_recovery_tds","penalties","penalty_yards",
    "timeouts","punt_returns","punt_return_yards","kickoff_returns","kickoff_return_yards",
    # blocked (renamed & flipped earlier)
    "blocked_fg","blocked_fg_list","blocked_fg_distance","blocked_pat","blocked_gwfg",
    # allowed columns
    "passing_tds_allowed","rushing_tds_allowed","receiving_tds_allowed",
    "special_teams_tds_allowed","def_tds_allowed",
    "fg_made_allowed","pat_made_allowed",
    "passing_2pt_conversions_allowed","rushing_2pt_conversions_allowed","receiving_2pt_conversions_allowed",
    "two_pt_conversions_allowed",
    # NEW yards allowed
    "pass_yds_allowed","rushing_yds_allowed","total_yds_allowed",
    # totals
    "points_allowed","dst_points_allowed",
    # --- Yahoo alignment additions ---
    "pts_allow","def_yds_allow","blk_kick","fum_rec",
    # --- Yahoo DST buckets (points) ---
    "pts_allow_0","pts_allow_1_6","pts_allow_7_13","pts_allow_14_20",
    "pts_allow_21_27","pts_allow_28_34","pts_allow_35_plus",
    # --- Yahoo DST buckets (yards) ---
    "yds_allow_neg","yds_allow_0_99","yds_allow_100_199","yds_allow_200_299",
    "yds_allow_300_399","yds_allow_400_499","yds_allow_500_plus",
]

# ---------------------------
# Core helpers
# ---------------------------
def build_url(year: int) -> str:
    return f"https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{year}.parquet"

def load_year_df(year: int) -> pd.DataFrame:
    return pd.read_parquet(build_url(year))

def normalize_team_codes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["team","opponent_team"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_CODE_MAP)
    return df

def _game_key_columns(df: pd.DataFrame) -> List[str]:
    """Prefer game_id when present to avoid duplicate join issues."""
    if "game_id" in df.columns:
        keys: List[str] = ["game_id"]
        for c in ["season", "week", "season_type"]:
            if c in df.columns:
                keys.append(c)
    else:
        keys = [c for c in ["season","week","season_type"] if c in df.columns]
    keys += ["team","opponent_team"]
    return keys

def flip_blocked_to_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Shift blocked kick stats to the opponent row (rename as blocked_*)."""
    key_cols = _game_key_columns(df)
    for old_col, new_col in BLOCKED_COL_RENAMES.items():
        if old_col not in df.columns:
            df[new_col] = pd.NA
            continue
        rhs = df[key_cols + [old_col]].copy()
        rhs = rhs.rename(columns={"team": "team_swapped", "opponent_team": "opponent_team_swapped"})
        m = df.merge(
            rhs.rename(columns={old_col: f"__opp_{old_col}"}),
            how="left",
            left_on=key_cols,
            right_on=[("opponent_team_swapped" if c == "team" else "team_swapped" if c == "opponent_team" else c) for c in key_cols],
            copy=False,
            validate="m:m"
        )
        df[new_col] = m[f"__opp_{old_col}"]
    return df

def add_allowed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create *_allowed columns by pulling the opponent's value via a self-merge."""
    key_cols = _game_key_columns(df)

    # Ensure all source columns exist & numeric where applicable
    for c in ALLOWED_MAP.keys():
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    rhs = df[key_cols + list(ALLOWED_MAP.keys())].copy()
    rhs = rhs.rename(columns={"team": "team_swapped", "opponent_team": "opponent_team_swapped"})
    m = df.merge(
        rhs,
        how="left",
        left_on=key_cols,
        right_on=[("opponent_team_swapped" if c == "team" else "team_swapped" if c == "opponent_team" else c) for c in key_cols],
        suffixes=("", "_opp"),
        validate="m:m",
        copy=False
    )

    # Assign *_allowed columns from opponent values
    for src_col, dst_col in ALLOWED_MAP.items():
        m[dst_col] = pd.to_numeric(m[src_col + "_opp"], errors="coerce").fillna(0)

    # Two-point conversions allowed: rushing + max(passing, receiving)
    m["two_pt_conversions_allowed"] = (
        m["rushing_2pt_conversions_allowed"]
        + m[["passing_2pt_conversions_allowed","receiving_2pt_conversions_allowed"]].max(axis=1)
    )

    # NEW: total yards allowed
    m["pass_yds_allowed"] = pd.to_numeric(m["pass_yds_allowed"], errors="coerce").fillna(0)
    m["rushing_yds_allowed"] = pd.to_numeric(m["rushing_yds_allowed"], errors="coerce").fillna(0)
    m["total_yds_allowed"] = (m["pass_yds_allowed"] + m["rushing_yds_allowed"]).astype(int)

    # Points allowed (no double-count of receiving TDs)
    tds_allowed = (
        m["passing_tds_allowed"]
        + m["rushing_tds_allowed"]
        + m["special_teams_tds_allowed"]
        + m["def_tds_allowed"]
    )
    fg_allowed = m["fg_made_allowed"]
    pat_allowed = m["pat_made_allowed"]
    two_pt_allowed = m["two_pt_conversions_allowed"]

    m["points_allowed"] = (6 * tds_allowed + 3 * fg_allowed + 1 * pat_allowed + 2 * two_pt_allowed).astype(int)
    m["dst_points_allowed"] = (m["points_allowed"] - 6 * m["def_tds_allowed"]).astype(int)

    # --- Yahoo-aligned extras ---
    m["pts_allow"] = m["dst_points_allowed"]
    m["def_yds_allow"] = m["total_yds_allowed"]

    # blk_kick = blocked_fg + blocked_pat + blocked_gwfg (post-flip columns added later)
    m["blk_kick"] = 0

    # fum_rec (Yahoo) sourced from fumble_recovery_opp
    if "fumble_recovery_opp" in m.columns:
        m["fum_rec"] = pd.to_numeric(m["fumble_recovery_opp"], errors="coerce").fillna(0)
    else:
        m["fum_rec"] = 0

    # Clean helper columns
    drop_helpers = [c for c in m.columns if c.endswith("_opp") or c.endswith("_swapped")]
    df = m.drop(columns=drop_helpers, errors="ignore")
    return df

def add_player_and_logo(df: pd.DataFrame) -> pd.DataFrame:
    # Force 'player' to city nickname (not "Defense"); drop any name/position fields from source.
    df["player"] = df["team"].map(DISPLAY_BASE).fillna(df["team"])
    for col in ["player_name", "player_display_name", "position"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["headshot_url"] = df["team"].map(TEAM_LOGO_MAP)
    return df

def _add_yahoo_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create Yahoo-style points/yards allowed bucket flags from pts_allow & def_yds_allow."""
    # Initialize all bucket columns to 0
    for c in [
        "pts_allow_0","pts_allow_1_6","pts_allow_7_13","pts_allow_14_20",
        "pts_allow_21_27","pts_allow_28_34","pts_allow_35_plus",
        "yds_allow_neg","yds_allow_0_99","yds_allow_100_199","yds_allow_200_299",
        "yds_allow_300_399","yds_allow_400_499","yds_allow_500_plus",
    ]:
        df[c] = 0

    # Points buckets use pts_allow (Yahoo's DST points-allowed notion)
    pa = pd.to_numeric(df.get("pts_allow", 0), errors="coerce").fillna(0).astype(int)
    df.loc[pa == 0, "pts_allow_0"] = 1
    df.loc[(pa >= 1) & (pa <= 6), "pts_allow_1_6"] = 1
    df.loc[(pa >= 7) & (pa <= 13), "pts_allow_7_13"] = 1
    df.loc[(pa >= 14) & (pa <= 20), "pts_allow_14_20"] = 1
    df.loc[(pa >= 21) & (pa <= 27), "pts_allow_21_27"] = 1
    df.loc[(pa >= 28) & (pa <= 34), "pts_allow_28_34"] = 1
    df.loc[pa >= 35, "pts_allow_35_plus"] = 1

    # Yardage buckets use def_yds_allow
    ya = pd.to_numeric(df.get("def_yds_allow", 0), errors="coerce").fillna(0).astype(int)
    df.loc[ya < 0, "yds_allow_neg"] = 1
    df.loc[(ya >= 0) & (ya <= 99), "yds_allow_0_99"] = 1
    df.loc[(ya >= 100) & (ya <= 199), "yds_allow_100_199"] = 1
    df.loc[(ya >= 200) & (ya <= 299), "yds_allow_200_299"] = 1
    df.loc[(ya >= 300) & (ya <= 399), "yds_allow_300_399"] = 1
    df.loc[(ya >= 400) & (ya <= 499), "yds_allow_400_499"] = 1
    df.loc[ya >= 500, "yds_allow_500_plus"] = 1

    return df

def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"season":"year","team":"nfl_team","opponent_team":"opponent_nfl_team"})

    # Set nfl_position = "DEF" for all rows
    df["nfl_position"] = "DEF"

    # Compute blk_kick now that blocked_* columns exist on this (flipped) row
    for c in ["blocked_fg","blocked_pat","blocked_gwfg"]:
        if c not in df.columns:
            df[c] = 0
    df["blk_kick"] = (
        pd.to_numeric(df["blocked_fg"], errors="coerce").fillna(0)
        + pd.to_numeric(df["blocked_pat"], errors="coerce").fillna(0)
        + pd.to_numeric(df["blocked_gwfg"], errors="coerce").fillna(0)
    ).astype(int)

    # Ensure presence of all requested cols (fill later if needed)
    for c in REQUESTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Build Yahoo buckets from pts_allow/def_yds_allow
    df = _add_yahoo_bins(df)

    ordered = df[REQUESTED_COLS + ["player","headshot_url"]]
    return ordered

def process_one_year(year: int, week: int | None) -> pd.DataFrame:
    df = load_year_df(year)
    df = normalize_team_codes(df)
    df = flip_blocked_to_opponent(df)
    df = add_allowed_columns(df)
    df = add_player_and_logo(df)
    out = finalize_columns(df)
    if week and week != 0:
        out = out[out["week"] == week]
    return out

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        year_in = int(input("Enter the season year (0 = start from current year and go backward): ").strip())
        week_in = int(input("Enter the week number (0 = all available weeks): ").strip())
    except Exception as e:
        print(f"Invalid input: {e}", file=sys.stderr)
        sys.exit(1)

    frames: List[pd.DataFrame] = []

    if year_in == 0:
        y = datetime.now().year
        while y >= 1900:  # safety floor
            try:
                print(f"Loading {y} ...")
                frames.append(process_one_year(y, week_in))
                y -= 1
            except Exception as e:
                print(f"Stopping at {y}: could not load year ({e}).")
                break
    else:
        try:
            frames.append(process_one_year(year_in, week_in))
        except Exception as e:
            print(f"Failed to load {year_in}: {e}", file=sys.stderr)
            sys.exit(2)

    if not frames:
        print("No data loaded.")
        sys.exit(0)

    combined = pd.concat(frames, ignore_index=True)

    year_tag = "multi_year" if year_in == 0 else str(year_in)
    week_tag = "allweeks" if week_in == 0 else f"w{week_in}"

    csv_path = DATA_ROOT / f"defense_stats_{year_tag}_{week_tag}.csv"
    parquet_path = DATA_ROOT / f"defense_stats_{year_tag}_{week_tag}.parquet"

    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Parquet: {parquet_path}")
    print(f"Rows: {len(combined):,}")

if __name__ == "__main__":
    main()
