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

# Add scripts directory to sys.path to enable multi_league.core.league_context import
SCRIPTS_DIR = SCRIPT_ROOT.parent.parent  # .../fantasy_football_data_scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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
    Ensure 'player' exists for offense rows; rename position to nfl_position.
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

    # Preserve position as nfl_position (critical for distinguishing same-named players)
    if "position" in out.columns and "nfl_position" not in out.columns:
        out["nfl_position"] = out["position"].astype("string")

    # Do NOT create fantasy_position from NFL data - it should only come from Yahoo
    # fantasy_position represents the roster slot (QB, RB1, FLEX, BN, etc.) from Yahoo

    # Drop unwanted name columns (but keep nfl_position)
    drop_cols = [c for c in ["player_name", "player_display_name", "position", "position_group"] if c in out.columns]
    if drop_cols:
        out.drop(columns=drop_cols, inplace=True)

    return out

def defense_to_player_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep defense rows as-is for naming: 'player' already set to the city in defense file.
    Do NOT append 'Defense'; preserve nfl_position and fantasy_position='DEF' if present.
    Note: fantasy_position='DEF' is correct for defense rows (set by defense_stats.py).
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

    # Preserve nfl_position and fantasy_position if already present (from defense_stats.py)
    # but drop the generic 'position' and 'position_group' variants
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
            try:
                ser = df[c]
                # Handle edge case: if df[c] returns DataFrame (duplicate columns), take first
                if isinstance(ser, pd.DataFrame):
                    ser = ser.iloc[:, 0]
                if not ser.isna().all():
                    dtype_map[c] = ser.dtype
                else:
                    dtype_map[c] = ser.dtype
            except Exception:
                # Fallback to string for problematic columns
                dtype_map[c] = "string"
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
            try:
                # If column exists but is generic 'object' and we picked a better dtype, try to align
                col_data = out[c]
                # Handle duplicate columns: take first if DataFrame returned
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                    out[c] = col_data

                dt = dtype_map.get(c)
                if dt is not None and col_data.dtype == "object":
                    try:
                        out[c] = col_data.astype(dt)
                    except Exception:
                        # last resort: leave it as-is; we'll coerce objects -> string before parquet
                        pass
            except Exception:
                # Skip problematic columns
                pass
    return out

def stack_union(offense: pd.DataFrame, defense_player_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Union columns and stack (no row merges) with explicit dtypes to avoid
    pandas' FutureWarning about all-NA dtype inference at concat time.
    """
    # Check for duplicate columns
    off_dupes = set([c for c in offense.columns if list(offense.columns).count(c) > 1])
    def_dupes = set([c for c in defense_player_rows.columns if list(defense_player_rows.columns).count(c) > 1])
    if off_dupes:
        print(f"[warn] Offense has duplicate columns: {off_dupes}")
        offense = offense.loc[:, ~offense.columns.duplicated()]
    if def_dupes:
        print(f"[warn] Defense has duplicate columns: {def_dupes}")
        defense_player_rows = defense_player_rows.loc[:, ~defense_player_rows.columns.duplicated()]

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
    # NOTE: Column names are now standardized at source (nfl_offense_stats.py)
    # No need to rename after union - offense and defense use same column names
    merged = stack_union(offense_all, defense_as_players)

    # Legacy rename for old column names (player_name/player_display_name)
    # These were never standardized at source, so we keep the rename here
    legacy_renames = {
        'player_name': 'player_name_orig',
        'player_display_name': 'player_display_name_orig'
    }
    renames_to_apply = {old: new for old, new in legacy_renames.items() if old in merged.columns}
    if renames_to_apply:
        merged = merged.rename(columns=renames_to_apply)
        print(f"[normalize] Renamed legacy columns: {list(renames_to_apply.keys())} -> {list(renames_to_apply.values())}")

    # Final small cleanup: make sure critical identity cols are string dtype
    # NOTE: Skip duplicate columns here - they'll be handled in duplicate removal
    for col in ["NFL_player_id", "player", "player_last_name", "headshot_url", "nfl_team", "opponent_nfl_team", "season_type"]:
        if col in merged.columns:
            try:
                col_data = merged[col]
                # Handle duplicate columns: take first if DataFrame returned
                if isinstance(col_data, pd.DataFrame):
                    # Don't modify duplicate columns here - skip them
                    continue
                if col_data.dtype == "object":
                    merged[col] = col_data.astype("string")
            except Exception:
                pass

    return merged

def main():
    # Accept flags so upstream callers don't get prompted again
    ap = argparse.ArgumentParser(description="Combine NFL offense + defense into a single player-shaped dataset.")
    ap.add_argument("--year", type=int, default=None, help="0 = start from current year and go backward")
    ap.add_argument("--week", type=int, default=None, help="0 = all available weeks")
    ap.add_argument("--context", type=str, default=None, help="Path to league_context.json (multi-league support)")
    args = ap.parse_args()

    # If context provided, override DATA_ROOT for this module and its dependencies
    if args.context:
        # Try to import LeagueContext
        try:
            from multi_league.core.league_context import LeagueContext  # type: ignore
        except Exception:
            try:
                from league_context import LeagueContext  # type: ignore
            except Exception:
                LeagueContext = None  # type: ignore
        if LeagueContext is not None:
            try:
                ctx = LeagueContext.load(args.context)
                # Update this module's DATA_ROOT
                global DATA_ROOT
                DATA_ROOT = Path(ctx.player_data_directory)
                DATA_ROOT.mkdir(parents=True, exist_ok=True)
                # Propagate the override into the imported modules so they write into the same root
                try:
                    defense_mod.DATA_ROOT = DATA_ROOT
                    defense_mod.DATA_ROOT.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                try:
                    offense_mod.DATA_ROOT = DATA_ROOT
                    offense_mod.DATA_ROOT.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                print(f"[context] Using league: {ctx.league_name} ({ctx.league_id})")
                print(f"[context] Output: {DATA_ROOT}")
            except Exception as e:
                print(f"[context][warn] Failed to load context: {e}")
        else:
            print(f"[context][warn] LeagueContext not available; ignoring --context")

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

    # Dynamic filenames - FIXED: Use "all_weeks" with underscore to match merge script expectations
    year_tag = "multi_year" if year_in == 0 else str(year_in)
    week_tag = "all_weeks" if week_in == 0 else f"week_{week_in}"

    # Remove duplicate columns by merging their data
    if any(merged.columns.duplicated()):
        dupes = set([c for c in merged.columns if list(merged.columns).count(c) > 1])
        print(f"[warn] Merging duplicate columns: {list(dupes)[:10]}..." if len(dupes) > 10 else f"[warn] Merging duplicate columns: {list(dupes)}")

        # DEBUG: Check what's in each column before merging
        if 'nfl_team' in dupes:
            nfl_team_positions = [i for i, c in enumerate(merged.columns) if c == 'nfl_team']
            print(f"[DEBUG] Found nfl_team at positions: {nfl_team_positions}")
            print(f"[DEBUG] Merged has {len(merged)} rows, {len(merged.columns)} columns")
            print(f"[DEBUG] Column 128 name: {merged.columns[128] if len(merged.columns) > 128 else 'N/A'}")
            print(f"[DEBUG] Column 132 name: {merged.columns[132] if len(merged.columns) > 132 else 'N/A'}")

        # Build deduplicated dataframe by merging duplicate column data
        result_cols = {}
        processed_names = set()

        for i, col_name in enumerate(merged.columns):
            if col_name in processed_names:
                continue  # Already processed this column name

            if col_name in dupes:
                # Find all positions of this duplicate column
                col_positions = [j for j, c in enumerate(merged.columns) if c == col_name]

                # DEBUG: Print what we're starting with
                if col_name == 'nfl_team':
                    for idx, pos in enumerate(col_positions):
                        count = merged.iloc[:, pos].notna().sum()
                        print(f"  DEBUG: nfl_team[{pos}] before merge has {count:,} values")

                # Merge data from all duplicates by taking first non-NA value
                # Start with the first column and fill NAs from subsequent columns
                merged_col = merged.iloc[:, col_positions[0]].copy()
                for pos in col_positions[1:]:
                    # For each row, if current value is NA, take from next column
                    mask = merged_col.isna()
                    other_col = merged.iloc[:, pos].copy()
                    merged_col[mask] = other_col[mask]

                result_cols[col_name] = merged_col
                non_na = merged_col.notna().sum()
                print(f"  Merged '{col_name}' from {len(col_positions)} columns -> {non_na:,} values")
                processed_names.add(col_name)
            else:
                # Not a duplicate, keep as-is
                result_cols[col_name] = merged.iloc[:, i].copy()

        # Reconstruct dataframe with merged columns
        merged = pd.DataFrame(result_cols, index=merged.index)

    csv_path = DATA_ROOT / f"nfl_stats_merged_{year_tag}_{week_tag}.csv"
    parquet_path = DATA_ROOT / f"nfl_stats_merged_{year_tag}_{week_tag}.parquet"

    merged.to_csv(csv_path, index=False)

    # CRITICAL: Convert numeric stat columns FIRST before string conversion
    # This prevents dst_points_allowed and other numeric columns from becoming strings
    numeric_stat_cols = [
        # DST/DEF basic stats
        'dst_points_allowed', 'dst_yards_allowed', 'dst_sacks', 'dst_interceptions',
        'dst_fumbles', 'dst_safeties', 'dst_tds', 'dst_blocked_kicks', 'dst_ret_tds',
        'def_interceptions', 'def_fumbles', 'def_sacks', 'def_safeties',
        'def_tds', 'def_2pt_returns', 'def_points_allowed', 'def_yards_allowed',
        # DEF detailed stats (from defense_stats.py)
        'def_sack_yards', 'def_qb_hits', 'def_interception_yards', 'def_pass_defended',
        'def_tackles_solo', 'def_tackles_with_assist', 'def_tackle_assists',
        'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced',
        'special_teams_tds', 'fum_rec', 'fum_ret_td',
        # Points/yards allowed
        'points_allowed', 'pts_allow', 'pass_yds_allowed', 'passing_yds_allowed',
        'rush_yds_allowed', 'rushing_yds_allowed', 'total_yds_allowed',
        'passing_tds_allowed', 'rushing_tds_allowed', 'receiving_tds_allowed',
        'passing_2pt_conversions_allowed', 'rushing_2pt_conversions_allowed',
        'receiving_2pt_conversions_allowed', 'fg_made_allowed', 'pat_made_allowed',
        # Points/yards allowed brackets
        'pts_allow_0', 'pts_allow_1_6', 'pts_allow_7_13', 'pts_allow_14_20',
        'pts_allow_21_27', 'pts_allow_28_34', 'pts_allow_35_plus',
        'yds_allow_0_99', 'yds_allow_100_199', 'yds_allow_200_299',
        'yds_allow_300_399', 'yds_allow_400_499', 'yds_allow_500_plus',
        # Offensive stats
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
        'carries', 'rushing_yards', 'rushing_tds',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
        'fantasy_points', 'fantasy_points_ppr', 'fantasy_points_half_ppr',
    ]

    for col in numeric_stat_cols:
        if col in merged.columns and not pd.api.types.is_numeric_dtype(merged[col]):
            try:
                merged[col] = pd.to_numeric(merged[col], errors='coerce')
            except Exception:
                pass

    # NOW coerce remaining object columns to string (defensive for pyarrow)
    # Skip columns that are already numeric (we just converted them above)
    for c in merged.columns:
        try:
            col_data = merged[c]
            # Handle duplicate columns: take first if DataFrame returned
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
                merged[c] = col_data
            # Only convert to string if NOT numeric
            if col_data.dtype == "object" and not pd.api.types.is_numeric_dtype(merged[c]):
                try:
                    merged[c] = col_data.astype("string")
                except Exception:
                    # fallback: cast to str then to "string"
                    merged[c] = col_data.astype(str).astype("string")
        except Exception:
            pass

    merged.to_parquet(parquet_path, index=False)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Parquet: {parquet_path}")
    print(f"Rows: {len(merged):,}")

if __name__ == "__main__":
    main()
