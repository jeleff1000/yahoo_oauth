#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import re

# Bring in the shared name utilities
from imports_and_utils import clean_name  # type: ignore

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

# Columns that often contain strings or NaNs; force to string to avoid pyarrow ArrowTypeError
STRINGY_LIST_COLS = [
    "fg_made_list", "fg_missed_list", "fg_blocked_list",
    "fg_made_distance", "fg_missed_distance", "fg_blocked_distance",
]

# Final column order (after renames below) — keep your original payload but ensure Yahoo-aligned fields exist
REQUESTED_COLS: List[str] = [
    "NFL_player_id",                      # renamed from player_id
    "player",                             # cleaned with imports_and_utils.clean_name
    "player_last_name",                   # strict normalized (lowercase, no hyphens/suffixes)
    "nfl_position",                       # renamed from position
    "position_group",
    "headshot_url",
    "year","week","season_type","nfl_team","opponent_nfl_team",

    # Core passing
    "pass_yds","pass_td","passing_interceptions",     # renamed from passing_*
    "completions","attempts","passing_air_yards","passing_yards_after_catch",
    "passing_first_downs","passing_epa","passing_cpoe","passing_2pt_conversions","pacr",

    # Core rushing
    "rush_att","rush_yds","rush_td","rushing_fumbles","rushing_fumbles_lost",
    "rushing_first_downs","rushing_epa","rushing_2pt_conversions",

    # Core receiving
    "rec","targets","rec_yds","rec_td","receiving_fumbles","receiving_fumbles_lost",
    "receiving_air_yards","receiving_yards_after_catch","receiving_first_downs",
    "receiving_epa","receiving_2pt_conversions","racr","target_share","air_yards_share","wopr",

    # Special-teams / DEF spill
    "ret_td",                             # renamed from special_teams_tds
    "def_tackles_solo","def_tackles_with_assist","def_tackle_assists",
    "def_tackles_for_loss","def_tackles_for_loss_yards","def_fumbles_forced",
    "def_sacks","def_sack_yards","def_qb_hits","def_interceptions","def_interception_yards",
    "def_pass_defended","def_tds","def_fumbles","def_safeties",

    # Fumble recovery & misc
    "misc_yards","fumble_recovery_own","fumble_recovery_yards_own","fumble_recovery_opp",
    "fumble_recovery_yards_opp","fum_ret_td",          # renamed from fumble_recovery_tds
    "penalties","penalty_yards",

    # Returns
    "punt_returns","punt_return_yards","kickoff_returns","kickoff_return_yards",

    # FG/PAT
    "fg_made","fg_att","fg_miss","fg_blocked","fg_long","fg_pct",
    "fg_made_0_19","fg_made_20_29","fg_made_30_39","fg_made_40_49","fg_made_50_59","fg_made_60_",
    "fg_missed_0_19","fg_missed_20_29","fg_missed_30_39","fg_missed_40_49","fg_missed_50_59","fg_missed_60_",
    "fg_made_list","fg_missed_list","fg_blocked_list","fg_made_distance","fg_missed_distance","fg_blocked_distance",
    "pat_made","pat_att","pat_missed","pat_blocked","pat_pct",
    "gwfg_made","gwfg_att","gwfg_missed","gwfg_blocked","gwfg_distance",

    # Yahoo-align derived
    "2-pt",                               # rushing + passing + receiving 2pt
    "fum_lost",                           # rushing_fumbles_lost + receiving_fumbles_lost
    "fg_yds",                             # sum of made distances; fallback by bins if list missing

    # Points variants (kept if present)
    "fantasy_points_zero_ppr","fantasy_points_ppr","fantasy_points_half_ppr",
]

# ---------------------------
# Last-name normalizer (strict, matches Yahoo script behavior)
# ---------------------------
_PUNCT_LAST_RE = re.compile(r"[.\-'\u2019]")  # includes smart apostrophe
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "ivs"}

def normalize_last_name_from_full(full_name: str) -> str:
    """
    From a full name, produce a normalized last name:
      - lowercase
      - remove hyphens/apostrophes/periods
      - strip suffix tokens like jr/sr/ii/iii/iv/v/... (also accepts 'ivs')
    """
    if not isinstance(full_name, str):
        return ""
    s = full_name.strip().lower()
    if not s:
        return ""
    # strip inner punctuation first
    s = _PUNCT_LAST_RE.sub("", s)
    # squeeze spaces
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    tokens = s.split(" ")
    # drop trailing suffix tokens
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    if not tokens:
        return ""
    return tokens[-1]

# ---------------------------
# Helpers
# ---------------------------
def build_url(year: int) -> str:
    return f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"

def load_year_df(year: int) -> pd.DataFrame:
    return pd.read_csv(build_url(year), low_memory=False)

def normalize_team_codes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("team", "opponent", "opponent_team"):
        if col in df.columns:
            df[col] = df[col].replace(TEAM_CODE_MAP)
    return df

def rename_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"season": "year", "team": "nfl_team"}
    if "opponent" in df.columns:
        rename_map["opponent"] = "opponent_nfl_team"
    elif "opponent_team" in df.columns:
        rename_map["opponent_team"] = "opponent_nfl_team"
    df = df.rename(columns=rename_map)
    return df

def add_half_ppr_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("fantasy_points", "fantasy_points_ppr"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Rename the raw fantasy_points column to represent 0-PPR scoring.  Older data sources
    # provide a non-PPR points total under the "fantasy_points" name.  To avoid confusion
    # with full PPR, we map it to `fantasy_points_zero_ppr` and derive half-PPR accordingly.
    df = df.rename(columns={"fantasy_points": "fantasy_points_zero_ppr"})
    # Derive half‑PPR by averaging zero‑PPR and full PPR totals when available
    df["fantasy_points_half_ppr"] = (df["fantasy_points_zero_ppr"] + df["fantasy_points_ppr"]) / 2.0
    return df

def coerce_stringy_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in STRINGY_LIST_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype("string").fillna("")
    return df

def _clean_player_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use imports_and_utils.clean_name to standardize the full 'player' name,
    then derive a strict-normalized 'player_last_name'.
    """
    # Determine source name column (prefer display name)
    src_col = None
    if "player_display_name" in df.columns:
        src_col = "player_display_name"
    elif "player_name" in df.columns:
        src_col = "player_name"
    elif "player" in df.columns:
        src_col = "player"

    if src_col is None:
        # ensure both columns exist
        df["player"] = ""
        df["player_last_name"] = ""
        return df

    # Clean full name
    df["player"] = df[src_col].fillna("").map(lambda s: clean_name(str(s)))
    # Strict normalized last name
    df["player_last_name"] = df["player"].map(normalize_last_name_from_full)
    return df

def derive_yahoo_aligned_fields(df: pd.DataFrame) -> pd.DataFrame:
    # --- ID & names ---
    if "player_id" in df.columns:
        df = df.rename(columns={"player_id": "NFL_player_id"})

    # Clean 'player' + 'player_last_name' using shared util + strict normalizer
    df = _clean_player_columns(df)

    # position → nfl_position
    if "position" in df.columns:
        df = df.rename(columns={"position": "nfl_position"})

    # Passing → pass_yds, pass_td, passing_interceptions
    rename_map = {
        "passing_yards": "pass_yds",
        "passing_tds": "pass_td",
        # keep interceptions consistent with your Yahoo side
        "interceptions": "passing_interceptions",              # rare alt name
        "passing_interceptions": "passing_interceptions",
        # Rushing
        "carries": "rush_att",
        "rushing_yards": "rush_yds",
        "rushing_tds": "rush_td",
        # Receiving
        "receptions": "rec",
        "receiving_yards": "rec_yds",
        "receiving_tds": "rec_td",
        # ST TDs
        "special_teams_tds": "ret_td",
        # FG missed should be fg_miss
        "fg_missed": "fg_miss",
        # fumble recovery tds
        "fumble_recovery_tds": "fum_ret_td",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Aggregated 2-pt: rushing + passing + receiving
    for c in ["rushing_2pt_conversions","passing_2pt_conversions","receiving_2pt_conversions"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["2-pt"] = (df["rushing_2pt_conversions"] + df["passing_2pt_conversions"] + df["receiving_2pt_conversions"]).astype(int)

    # fum_lost = rushing_fumbles_lost + receiving_fumbles_lost (if present)
    for c in ["rushing_fumbles_lost","receiving_fumbles_lost"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["fum_lost"] = (df["rushing_fumbles_lost"] + df["receiving_fumbles_lost"]).astype(int)

    # FG yards: prefer parsing fg_made_distance list; fallback to bin midpoints
    def _sum_distance_list(s: pd.Series) -> pd.Series:
        # strings like "20,33,48"; tolerate blanks
        def parse(x: Any) -> int:
            if pd.isna(x) or (isinstance(x, str) and x.strip() == ""):
                return 0
            if isinstance(x, (int, float)):
                return int(x)
            total = 0
            for part in str(x).split(","):
                part = part.strip()
                try:
                    if part:
                        total += int(part)
                except ValueError:
                    pass
            return total
        return s.apply(parse)

    if "fg_made_distance" in df.columns:
        df["fg_yds"] = _sum_distance_list(df["fg_made_distance"])
    else:
        # fallback via bins (use bin midpoints)
        mids = {
            "fg_made_0_19": 19,
            "fg_made_20_29": 25,
            "fg_made_30_39": 35,
            "fg_made_40_49": 45,
            "fg_made_50_59": 55,
            "fg_made_60_": 60,
        }
        for c in mids:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["fg_yds"] = (
            df["fg_made_0_19"] * mids["fg_made_0_19"]
            + df["fg_made_20_29"] * mids["fg_made_20_29"]
            + df["fg_made_30_39"] * mids["fg_made_30_39"]
            + df["fg_made_40_49"] * mids["fg_made_40_49"]
            + df["fg_made_50_59"] * mids["fg_made_50_59"]
            + df["fg_made_60_"]   * mids["fg_made_60_"]
        ).astype(int)

    return df

def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all requested columns exist (older years may miss some)
    for c in REQUESTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[REQUESTED_COLS]

def process_one_year(year: int, week: int | None) -> pd.DataFrame:
    df = load_year_df(year)
    df = normalize_team_codes(df)
    df = rename_core_columns(df)
    df = add_half_ppr_and_rename(df)
    df = coerce_stringy_list_cols(df)  # avoid ArrowTypeError on parquet
    # ---- Yahoo-aligned transforms ----
    df = derive_yahoo_aligned_fields(df)
    # ----------------------------------
    if week and week != 0 and "week" in df.columns:
        df = df[df["week"] == week]
    out = finalize_columns(df)
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

    # Dynamic filenames
    year_tag = "multi_year" if year_in == 0 else str(year_in)
    week_tag = "allweeks" if week_in == 0 else f"w{week_in}"

    csv_path = DATA_ROOT / f"player_stats_{year_tag}_{week_tag}.csv"
    parquet_path = DATA_ROOT / f"player_stats_{year_tag}_{week_tag}.parquet"

    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Parquet: {parquet_path}")
    print(f"Rows: {len(combined):,}")

if __name__ == "__main__":
    main()
