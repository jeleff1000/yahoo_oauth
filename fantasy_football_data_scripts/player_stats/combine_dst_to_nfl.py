#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict
import importlib
import pandas as pd

# ---------------------------------
# Paths (relative to this file)
# ---------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_ROOT = THIS_FILE.parent
DATA_ROOT = SCRIPT_ROOT.parent.parent / "fantasy_football_data" / "player_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Make sibling modules importable (defense_stats.py, nfl_offense_stats.py)
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    defense_mod = importlib.import_module("defense_stats")
    offense_mod = importlib.import_module("nfl_offense_stats")
except Exception as e:
    print(
        "Failed to import sibling modules. Ensure this file sits next to "
        "`defense_stats.py` and `nfl_offense_stats.py`.\n",
        file=sys.stderr,
    )
    raise

# For parquet safety (ArrowTypeError), coerce these to string when present
STRINGY_LIST_COLS = [
    # offense
    "fg_made_list", "fg_missed_list", "fg_blocked_list",
    "fg_made_distance", "fg_missed_distance", "fg_blocked_distance",
    # defense
    "blocked_fg_list", "blocked_fg_distance",
]

# Identity / ordering helpers
# Use only 'player' for naming; drop player_name/player_display_name/position/position_group entirely.
# Keep player_last_name if it exists (offense) for downstream matching.
ID_FIRST = [
    "NFL_player_id", "player", "player_last_name", "headshot_url",
    "year", "week", "season_type", "nfl_team", "opponent_nfl_team",
]

# ---------------------------------
# Loaders (reuse your existing modules)
# ---------------------------------
def load_offense_year(year: int, week: int | None) -> pd.DataFrame:
    return offense_mod.process_one_year(year, week)

def load_defense_year(year: int, week: int | None) -> pd.DataFrame:
    return defense_mod.process_one_year(year, week)

# ---------------------------------
# Shaping / coercion
# ---------------------------------
def offense_fix_player(off: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'player' exists for offense rows; drop name/position variants.
    Priority for player: player (if exists) > player_display_name > player_name.
    Keep player_last_name if already present (used downstream).
    """
    out = off.copy()

    if "player" not in out.columns:
        if "player_display_name" in out.columns:
            out["player"] = out["player_display_name"].astype(str)
        elif "player_name" in out.columns:
            out["player"] = out["player_name"].astype(str)
        else:
            out["player"] = pd.NA

    # Normalize dtype
    out["player"] = out["player"].astype("string")

    # Drop unwanted name/position columns
    drop_cols = [c for c in ["player_name", "player_display_name", "position", "position_group"] if c in out.columns]
    if drop_cols:
        out.drop(columns=drop_cols, inplace=True)

    return out

def defense_to_player_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep defense rows as-is for naming: 'player' already set to the city in defense file.
    Do NOT append 'Defense'; remove name/position variants entirely.
    Provide a stable NFL_player_id for DEF rows if missing.
    """
    out = df.copy()

    # Ensure a 'player' label exists (defense script already sets city; fallback to team code)
    if "player" not in out.columns:
        base = out["nfl_team"] if "nfl_team" in out.columns else pd.Series("", index=out.index)
        out["player"] = base.astype("string")

    # Create deterministic NFL_player_id for DEF rows if not present
    if "NFL_player_id" not in out.columns:
        if "nfl_team" in out.columns:
            out["NFL_player_id"] = ("DEF-" + out["nfl_team"].astype(str)).astype("string")
        else:
            out["NFL_player_id"] = pd.Series(["DEF-UNK"] * len(out), dtype="string")

    # Ensure headshot_url exists
    if "headshot_url" not in out.columns:
        out["headshot_url"] = pd.NA

    # Remove extra name/position columns if present
    drop_cols = [c for c in ["player_name", "player_display_name", "position", "position_group"] if c in out.columns]
    if drop_cols:
        out.drop(columns=drop_cols, inplace=True)

    # Normalize dtype
    out["player"] = out["player"].astype("string")

    return out

def coerce_stringy_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure *_list and *_distance fields are strings for parquet."""
    out = df.copy()
    for c in STRINGY_LIST_COLS:
        if c in out.columns:
            try:
                out[c] = out[c].astype("string")
            except Exception:
                out[c] = out[c].astype(str).astype("string")
            out[c] = out[c].fillna("")
    return out

# ---------- NEW: dtype-safe concat helpers to avoid FutureWarning ----------
def _build_dtype_map(df_list: List[pd.DataFrame]) -> Dict[str, pd.api.extensions.ExtensionDtype | str]:
    """
    For each column appearing in any df, pick a stable dtype preference:
      - Prefer the first df where the column exists and is NOT all-NA.
      - Fall back to that df's dtype if all-NA but dtype is known.
      - Finally default to 'string' as a safe, parquet-friendly dtype.
    """
    dtype_map: Dict[str, pd.api.extensions.ExtensionDtype | str] = {}
    for df in df_list:
        for c in df.columns:
            if c in dtype_map:
                continue
            ser = df[c]
            if not ser.isna().all():
                dtype_map[c] = ser.dtype
            else:
                dtype_map[c] = ser.dtype
    return dtype_map

def _ensure_columns_with_dtype(df: pd.DataFrame, cols: List[str], dtype_map: Dict[str, pd.api.extensions.ExtensionDtype | str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            dt = dtype_map.get(c, "string")
            try:
                out[c] = pd.Series(pd.NA, index=out.index, dtype=dt)
            except Exception:
                out[c] = pd.Series(pd.NA, index=out.index, dtype="string")
        else:
            # If column exists but is generic 'object' and we picked a better dtype, try to align
            dt = dtype_map.get(c)
            if dt is not None and out[c].dtype == "object":
                try:
                    out[c] = out[c].astype(dt)
                except Exception:
                    # last resort: leave it as-is; we'll coerce objects -> string before parquet
                    pass
    return out

def stack_union(offense: pd.DataFrame, defense_player_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Union columns and stack (no row merges) with explicit dtypes to avoid
    pandas' FutureWarning about all-NA dtype inference at concat time.
    """
    # Build union of columns
    all_cols: Set[str] = set(offense.columns) | set(defense_player_rows.columns)

    # Prefer identity columns first, then the rest sorted
    ordered = ID_FIRST + sorted([c for c in all_cols if c not in ID_FIRST])

    # Build a dtype map from what we already have
    dtype_map = _build_dtype_map([offense, defense_player_rows])

    # Ensure both frames have all columns, created with stable dtypes (typed NA, not bare NA)
    off = _ensure_columns_with_dtype(offense, ordered, dtype_map).reindex(columns=ordered)
    dfn = _ensure_columns_with_dtype(defense_player_rows, ordered, dtype_map).reindex(columns=ordered)

    # Concat without the deprecation warning
    stacked = pd.concat([off, dfn], ignore_index=True, copy=False)

    # Optional: drop columns that are all-NA across the result (never populated by either side)
    # Comment this out if you prefer to keep every unioned column:
    stacked = stacked.loc[:, ~stacked.isna().all(axis=0)]

    return stacked
# -------------------------------------------------------------------------

# ---------------------------------
# Driver (year/week semantics match your other scripts)
# ---------------------------------
def load_all(year_in: int, week_in: int) -> pd.DataFrame:
    offense_frames: List[pd.DataFrame] = []
    defense_frames: List[pd.DataFrame] = []

    if year_in == 0:
        y = datetime.now().year
        while y >= 1900:
            try:
                print(f"Loading offense {y} ...")
                offense_frames.append(load_offense_year(y, week_in))
                print(f"Loading defense {y} ...")
                defense_frames.append(load_defense_year(y, week_in))
                y -= 1
            except Exception as e:
                print(f"Stopping at {y}: could not load year ({e}).")
                break
    else:
        offense_frames.append(load_offense_year(year_in, week_in))
        defense_frames.append(load_defense_year(year_in, week_in))

    if not offense_frames:
        raise RuntimeError("No offense data loaded.")
    if not defense_frames:
        raise RuntimeError("No defense data loaded.")

    offense_all = pd.concat(offense_frames, ignore_index=True)
    defense_all = pd.concat(defense_frames, ignore_index=True)

    # Normalize naming columns:
    offense_all = offense_fix_player(offense_all)
    defense_as_players = defense_to_player_shape(defense_all)

    # Parquet safety for list-ish columns
    offense_all = coerce_stringy_cols(offense_all)
    defense_as_players = coerce_stringy_cols(defense_as_players)

    # Stack with union schema (dtype-safe)
    merged = stack_union(offense_all, defense_as_players)

    # Final small cleanup: make sure critical identity cols are string dtype
    for col in ["NFL_player_id", "player", "player_last_name", "headshot_url", "nfl_team", "opponent_nfl_team", "season_type"]:
        if col in merged.columns and merged[col].dtype == "object":
            merged[col] = merged[col].astype("string")

    return merged

def main():
    # Accept flags so upstream callers don't get prompted again
    ap = argparse.ArgumentParser(description="Combine NFL offense + defense into a single player-shaped dataset.")
    ap.add_argument("--year", type=int, default=None, help="0 = start from current year and go backward")
    ap.add_argument("--week", type=int, default=None, help="0 = all available weeks")
    args = ap.parse_args()

    if args.year is None:
        try:
            year_in = int(input("Enter the season year (0 = start from current year and go backward): ").strip())
        except Exception as e:
            print(f"Invalid input: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        year_in = args.year

    if args.week is None:
        try:
            week_in = int(input("Enter the week number (0 = all available weeks): ").strip())
        except Exception as e:
            print(f"Invalid input: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        week_in = args.week

    try:
        merged = load_all(year_in, week_in)
    except Exception as e:
        print(f"Merge failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Dynamic filenames
    year_tag = "multi_year" if year_in == 0 else str(year_in)
    week_tag = "allweeks" if week_in == 0 else f"w{week_in}"

    csv_path = DATA_ROOT / f"nfl_stats_merged_{year_tag}_{week_tag}.csv"
    parquet_path = DATA_ROOT / f"nfl_stats_merged_{year_tag}_{week_tag}.parquet"

    merged.to_csv(csv_path, index=False)

    # Coerce any remaining object columns to string (defensive for pyarrow)
    for c in merged.columns:
        if merged[c].dtype == "object":
            try:
                merged[c] = merged[c].astype("string")
            except Exception:
                # fallback: cast to str then to "string"
                merged[c] = merged[c].astype(str).astype("string")

    merged.to_parquet(parquet_path, index=False)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Parquet: {parquet_path}")
    print(f"Rows: {len(merged):,}")

if __name__ == "__main__":
    main()
