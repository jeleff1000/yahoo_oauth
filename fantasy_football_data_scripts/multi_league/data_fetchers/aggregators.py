#!/usr/bin/env python3
"""
Data Aggregators

Functions for aggregating and normalizing weekly/yearly data files into
canonical parquet outputs. These operations are commonly needed for both
initial imports and incremental updates.

Extracted from initial_import_v2.py for modularity and reusability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional
import pandas as pd

# Import data normalization utilities from the same package
from multi_league.core.data_normalization import (
    normalize_numeric_columns,
    write_parquet_robust,
    add_composite_keys
)

# Try to import LeagueContext
try:
    from multi_league.core.league_context import LeagueContext
except ImportError:
    LeagueContext = None


def _load_ctx(context_path: str):
    """Load league context from JSON file."""
    if LeagueContext is None:
        raise ImportError("LeagueContext not available")
    return LeagueContext.load(context_path)


# Required columns for matchup data
REQUIRED_MATCHUP_COLS = [
    "year", "week", "manager", "opponent", "team_name", "opponent_team",
    "team_points", "opponent_points", "is_playoffs", "is_consolation",
    "manager_week"
]

OPTIONAL_MATCHUP_COLS_DEFAULTS = {
    "is_playoffs": False,
    "is_consolation": False,
}


def merge_matchups_to_parquet(
    context: str,
    years: Optional[List[int]] = None,
    log: Callable[..., None] = print
) -> Path:
    """
    Build a unified matchup dataset across all requested years and save it
    to <ctx.data_directory>/matchup.parquet and .csv.

    The V2 fetchers store weekly matchup data under names like:
      - yahoo_matchups_<year>_week_<week>.parquet  (legacy)
      - matchup_data_week_<week>_year_<year>.parquet (current)

    This function:
      - Loads each weekly file for the specified years
      - Infers missing year/week columns from filenames when absent
      - Ensures all required matchup columns exist, with defaults if needed
      - Computes 'cumulative_week' as a sortable Int64 (year * 100 + week)
      - Computes 'manager_week' using canonical rule: manager(without spaces) + cumulative_week
      - Concatenates all parts, sorts deterministically by year/week/manager
      - Writes both Parquet and CSV outputs to the data directory
      - Returns the path to the Parquet file

    Args:
        context: Path to league_context.json
        years: List of years to include (default: all years in context)
        log: Logging function (default: print)

    Returns:
        Path to the output matchup.parquet file

    Raises:
        FileNotFoundError: If no matchup parts are found to merge
    """
    ctx = _load_ctx(context)

    # Save to canonical location (data_directory/matchup.parquet)
    mdir = Path(ctx.data_directory)
    mdir.mkdir(parents=True, exist_ok=True)

    # Weekly matchup files are read from matchup_data_directory
    weekly_dir = Path(ctx.matchup_data_directory)

    # Determine which seasons to include
    if years is None:
        start = int(ctx.start_year)
        end = int(ctx.end_year or pd.Timestamp.now().year)
        years = list(range(start, end + 1))

    parts: List[pd.DataFrame] = []

    # MANUAL FILE DETECTION: Scan for files outside the API fetch range
    # This allows users to manually add historical data (e.g., pre-API years)
    all_matchup_files = list(weekly_dir.glob("*.parquet"))
    manual_files = []

    for f in all_matchup_files:
        # Try to extract year from filename
        year_in_filename = None
        stem = f.stem

        # Pattern 1: yahoo_matchups_YYYY_week_N.parquet
        if stem.startswith("yahoo_matchups_") and "_week_" in stem:
            try:
                year_in_filename = int(stem.split("_")[2])  # yahoo_matchups_YYYY_week_N
            except (IndexError, ValueError):
                pass

        # Pattern 2: matchup_data_week_N_year_YYYY.parquet or matchup_data_week_all_year_YYYY.parquet
        elif "year_" in stem:
            try:
                year_in_filename = int(stem.split("year_")[-1])
            except ValueError:
                pass

        # If we found a year and it's OUTSIDE the normal range, it's a manual file
        if year_in_filename is not None and year_in_filename not in years:
            manual_files.append((f, year_in_filename))
            log(f"  [MANUAL] Detected manual file: {f.name} (year {year_in_filename})")

    # Add manual years to processing list
    manual_years = sorted(set(y for _, y in manual_files))
    all_years = sorted(set(years + manual_years))

    if manual_years:
        log(f"  [MANUAL] Including {len(manual_years)} manual year(s): {manual_years}")
        log(f"  [RANGE] Full year range: {min(all_years)} - {max(all_years)}")

    for season in all_years:
        # Gather files for this season from weekly_dir
        files = list(weekly_dir.glob(f"yahoo_matchups_{season}_week_*.parquet"))
        files += list(weekly_dir.glob(f"matchup_data_week_*_year_{season}.parquet"))

        for f in sorted(files):
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            if df is None or df.empty:
                continue

            # CRITICAL: Skip files without league_id column (old format)
            if "league_id" not in df.columns:
                log(f"      [SKIP] {f.name} missing league_id column (old format) - skipping")
                continue

            # Filter to any of the linked league IDs (multi-year leagues have different IDs per year)
            # Build set of all valid league IDs: from league_ids mapping + base league_id
            valid_league_ids = set(ctx.league_ids.values()) if ctx.league_ids else set()
            valid_league_ids.add(ctx.league_id)  # Always include the base league_id

            df = df[df["league_id"].isin(valid_league_ids)].copy()
            if df.empty:
                continue

            # Add season/year if missing
            if "year" not in df.columns:
                inferred_year = season
                # If filename has _year_ pattern, extract
                name = f.stem
                if "_year_" in name:
                    try:
                        inferred_year = int(name.split("_year_")[-1])
                    except Exception:
                        inferred_year = season
                df["year"] = inferred_year
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

            # Add week if missing
            if "week" not in df.columns:
                wk_val = pd.NA
                name = f.stem
                try:
                    # legacy yahoo_matchups_<year>_week_<wk>
                    if name.startswith("yahoo_matchups_") and "_week_" in name:
                        wk_str = name.split("_week_")[-1]
                        if wk_str.isdigit():
                            wk_val = int(wk_str)
                    # new matchup_data_week_<wk>_year_<year>
                    elif name.startswith("matchup_data_week_"):
                        wk_str = name.split("matchup_data_week_")[1].split("_year_")[0]
                        if wk_str == "all":
                            wk_val = pd.NA
                        elif wk_str.isdigit():
                            wk_val = int(wk_str)
                except Exception:
                    wk_val = pd.NA
                df["week"] = wk_val
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

            # Add opponent_team if missing (lookup from manager -> team_name mapping)
            if "opponent_team" not in df.columns or df["opponent_team"].isna().all():
                if "opponent" in df.columns and "team_name" in df.columns:
                    manager_to_team = df.groupby("manager")["team_name"].first().to_dict()
                    df["opponent_team"] = df["opponent"].map(manager_to_team)
                    if df["opponent_team"].notna().any():
                        log(f"      [ADD] {f.name}: Added opponent_team mapping")

            # Ensure required columns exist (add NULL if missing)
            for col in REQUIRED_MATCHUP_COLS:
                if col not in df.columns:
                    if col in OPTIONAL_MATCHUP_COLS_DEFAULTS:
                        df[col] = OPTIONAL_MATCHUP_COLS_DEFAULTS[col]
                    else:
                        df[col] = pd.NA

            # Add composite keys
            df = add_composite_keys(df)

            parts.append(df)

    if not parts:
        raise FileNotFoundError("No matchup parquet parts found to merge")

    merged = pd.concat(parts, ignore_index=True, sort=False)

    # CRITICAL: Remove empty rows (common in CSV conversions with trailing empty lines)
    # Filter out rows where manager is None/NA or year is NA
    before_count = len(merged)
    merged = merged[
        merged['manager'].notna() &
        (merged['manager'] != 'None') &
        (merged['manager'] != '') &
        merged['year'].notna()
    ].copy()
    after_count = len(merged)

    if before_count > after_count:
        log(f"  [CLEANUP] Removed {before_count - after_count:,} empty rows (manager=None or year=NA)")

    # Sort for deterministic output
    sort_cols = [c for c in ["year", "week", "manager"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # Normalize numeric columns before writing (prevents string->numeric conversion errors)
    merged = normalize_numeric_columns(merged)

    # Write outputs
    out_parquet = mdir / "matchup.parquet"
    out_csv = mdir / "matchup.csv"
    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)

    log(f"[POST] wrote matchup master: {out_parquet} ({len(merged):,} rows)")
    log(f"[POST] wrote matchup CSV:    {out_csv}")

    return out_parquet


def normalize_draft_parquet(
    context: str,
    log: Callable[..., None] = print
) -> Path:
    """
    Combine draft year files and normalize to canonical draft.parquet.

    Similar to merge_matchups_to_parquet, this function:
    - Loads individual year files from draft_data_directory
    - Combines them into a single DataFrame
    - Normalizes column types and names
    - Filters out deprecated columns
    - Writes to canonical location

    Normalizes:
    - yahoo_player_id -> string
    - year -> Int64
    - is_keeper_status -> Int64
    - league_id -> added if missing
    - Removes deprecated keeper SPAR columns

    Args:
        context: Path to league_context.json
        log: Logging function (default: print)

    Returns:
        Path to canonical draft.parquet file

    Raises:
        FileNotFoundError: If no draft year files found
        ValueError: If combined draft data is empty
    """
    ctx = _load_ctx(context)

    # Load and combine individual year files (like matchup merge)
    draft_dir = Path(ctx.draft_data_directory)
    year_files = sorted(draft_dir.glob("draft_data_*.parquet"))

    if not year_files:
        raise FileNotFoundError(f"No draft year files found in {draft_dir}")

    log(f"[normalize_draft] Combining {len(year_files)} draft year files...")
    parts = []
    for f in year_files:
        year_df = pd.read_parquet(f)
        parts.append(year_df)
        log(f"   [LOAD] {f.name} ({len(year_df):,} rows)")

    df = pd.concat(parts, ignore_index=True)
    log(f"[normalize_draft] Combined {len(df):,} total draft picks")
    if df.empty:
        raise ValueError("draft.parquet is empty")

    # Ensure yahoo_player_id exists
    if "yahoo_player_id" not in df.columns:
        for alt in ("yahoo_id", "yahoo_player_key", "player_id", "playerId", "player"):
            if alt in df.columns:
                df["yahoo_player_id"] = df[alt]
                break
        if "yahoo_player_id" not in df.columns:
            df["yahoo_player_id"] = pd.NA

    # Ensure year exists
    if "year" not in df.columns:
        for alt in ("season", "Season", "draft_year"):
            if alt in df.columns:
                df["year"] = df[alt]
                break
        if "year" not in df.columns:
            try:
                parent_year = int(found.stem.split("_")[-1])
                df["year"] = parent_year
            except Exception:
                df["year"] = getattr(ctx, "start_year", pd.NA)

    # Standardize column names (handle different naming conventions)
    column_renames = {}
    if "player_name" in df.columns and "player" not in df.columns:
        column_renames["player_name"] = "player"
    if "primary_position" in df.columns and "yahoo_position" not in df.columns:
        column_renames["primary_position"] = "yahoo_position"
    if "player_id" in df.columns and "yahoo_player_id" not in df.columns:
        column_renames["player_id"] = "yahoo_player_id"

    if column_renames:
        df = df.rename(columns=column_renames)
        log(f"[normalize_draft] Standardized column names: {column_renames}")

    # Normalize types
    if "yahoo_player_id" in df.columns:
        try:
            df["yahoo_player_id"] = df["yahoo_player_id"].astype("string")
        except Exception:
            df["yahoo_player_id"] = df["yahoo_player_id"].astype(str).astype("string")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Normalize is_keeper_status: convert empty strings to 0, "1" to 1
    if "is_keeper_status" in df.columns:
        # Replace empty strings with 0
        df["is_keeper_status"] = df["is_keeper_status"].replace("", "0")
        # Convert to numeric, then to Int64
        df["is_keeper_status"] = pd.to_numeric(df["is_keeper_status"], errors="coerce").fillna(0).astype("Int64")
        log(f"[normalize_draft] Normalized is_keeper_status to Int64 (keepers: {(df['is_keeper_status'] == 1).sum():,})")

    # Normalize is_keeper_cost similarly
    if "is_keeper_cost" in df.columns:
        df["is_keeper_cost"] = df["is_keeper_cost"].replace("", "0")
        df["is_keeper_cost"] = pd.to_numeric(df["is_keeper_cost"], errors="coerce").fillna(0).astype("Int64")

    # Ensure league_id exists
    if "league_id" not in df.columns:
        df["league_id"] = ctx.league_id
        log(f"[normalize_draft] Added league_id: {ctx.league_id}")

    # DEPRECATED columns that should be filtered out (old keeper SPAR metrics)
    deprecated_cols = [
        'keeper_spar_per_dollar', 'keeper_surplus_spar', 'keeper_roi_spar',
        'keeper_spar_per_dollar_rank', 'keeper_surplus_rank'
    ]

    # Canonical output path
    out_path = ctx.canonical_draft_file

    # Normalize numeric columns before writing (prevents string->numeric conversion errors)
    df = normalize_numeric_columns(df)

    # KEEP ALL PLAYERS (drafted and undrafted) for SPAR calculations
    # Undrafted players have manager=None/empty but still have avg_pick, percent_drafted, etc.
    # This allows us to calculate SPAR for players who SHOULD have been drafted
    initial_count = len(df)
    drafted_count = 0
    undrafted_count = 0
    if 'manager' in df.columns:
        drafted_count = df['manager'].notna().sum()
        undrafted_count = df['manager'].isna().sum()
        log(f"[normalize_draft] Keeping ALL draft-eligible players:")
        log(f"  - Drafted: {drafted_count:,} players")
        log(f"  - Undrafted (eligible): {undrafted_count:,} players")
        log(f"  - Total: {initial_count:,} players")
    else:
        log(f"[normalize_draft] WARNING: No 'manager' column found")

    # Final cleanup: ensure deprecated columns are not in output
    final_deprecated = [c for c in deprecated_cols if c in df.columns]
    if final_deprecated:
        df = df.drop(columns=final_deprecated)
        log(f"[normalize_draft] Removed {len(final_deprecated)} deprecated columns from output: {final_deprecated}")

    # Write to canonical location
    out_csv = out_path.parent / "draft.csv"
    write_parquet_robust(df, out_path, description="draft data")
    df.to_csv(out_csv, index=False)
    log(f"[POST] wrote draft CSV: {out_csv}")

    return out_path


def normalize_transactions_parquet(
    context: str,
    log: Callable[..., None] = print
) -> Path:
    """
    Ensure transactions.parquet exists at canonical location with proper normalization.

    Args:
        context: Path to league_context.json
        log: Logging function (default: print)

    Returns:
        Path to canonical transactions.parquet file

    Raises:
        FileNotFoundError: If transactions.parquet not found
        ValueError: If transactions.parquet is empty
    """
    ctx = _load_ctx(context)

    # Try likely locations (check main league directory first, then subdirectories)
    candidates = []
    if getattr(ctx, "data_directory", None):
        candidates.append(Path(ctx.data_directory) / "transactions.parquet")
        candidates.append(Path(ctx.data_directory) / "transaction.parquet")
    if getattr(ctx, "transaction_data_directory", None):
        candidates.append(Path(ctx.transaction_data_directory) / "transactions.parquet")
        candidates.append(Path(ctx.transaction_data_directory) / "transaction.parquet")

    found = None
    for c in candidates:
        if c and c.exists():
            found = c
            break
    if not found:
        raise FileNotFoundError("transactions.parquet not found in expected locations")

    df = pd.read_parquet(found)
    if df.empty:
        raise ValueError("transactions.parquet is empty")

    # Ensure league_id exists
    if "league_id" not in df.columns:
        df["league_id"] = ctx.league_id
        log(f"[normalize_transactions] Added league_id: {ctx.league_id}")

    # Normalize year/week
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # CRITICAL: Deduplicate source data before any merges
    # Use transaction_id + player_key to preserve multi-player transactions (add/drop combos, trades)
    # A single transaction_id can have multiple players (e.g., trade, add+drop combo)
    if "transaction_id" in df.columns and "player_key" in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["transaction_id", "player_key"], keep="first")
        if len(df) < initial_count:
            log(f"[normalize_transactions] Removed {initial_count - len(df):,} duplicate transaction records")
    elif "transaction_id" in df.columns and "yahoo_player_id" in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["transaction_id", "yahoo_player_id"], keep="first")
        if len(df) < initial_count:
            log(f"[normalize_transactions] Removed {initial_count - len(df):,} duplicate transaction records")
    else:
        # Fallback to composite key if transaction_id missing
        dedup_keys = [k for k in ["yahoo_player_id", "year", "cumulative_week", "timestamp"] if k in df.columns]
        if len(dedup_keys) >= 3:
            initial_count = len(df)
            df = df.drop_duplicates(subset=dedup_keys, keep="first")
            if len(df) < initial_count:
                log(f"[normalize_transactions] Removed {initial_count - len(df):,} duplicates using keys {dedup_keys}")

    # Preserve enrichment columns from existing canonical file if they exist
    # (These are added by transformation scripts like player_to_transactions_v2.py)
    enrichment_cols = [
        'ppg_before_transaction', 'ppg_after_transaction',
        'total_fantasy_points_before', 'total_fantasy_points_after',
        'weeks_owned_before', 'weeks_started_before',
        'points_per_faab_dollar', 'transaction_quality_score'
    ]

    out_path = ctx.canonical_transaction_file
    try:
        if out_path.exists():
            existing_canonical = pd.read_parquet(out_path)

            # Check which enrichment columns exist and have data in canonical
            cols_to_preserve = [c for c in enrichment_cols if c in existing_canonical.columns
                               and existing_canonical[c].notna().sum() > 0
                               and c not in df.columns]  # Only preserve if not already in source

            if cols_to_preserve:
                join_keys = ['yahoo_player_id', 'year', 'cumulative_week']
                # Ensure join keys exist in both dataframes
                if all(k in df.columns for k in join_keys) and all(k in existing_canonical.columns for k in join_keys):
                    enrichment_data = existing_canonical[join_keys + cols_to_preserve].copy()

                    # CRITICAL: Deduplicate enrichment data before merge to prevent Cartesian product
                    # Keep the first occurrence of each unique key combination
                    enrichment_data = enrichment_data.drop_duplicates(subset=join_keys, keep='first')

                    df = df.merge(
                        enrichment_data,
                        on=join_keys,
                        how='left',
                        suffixes=('', '_preserved')
                    )

                    # Clean up duplicate columns from merge
                    for col in cols_to_preserve:
                        if f'{col}_preserved' in df.columns:
                            df[col] = df[f'{col}_preserved']
                            df = df.drop(columns=[f'{col}_preserved'])

                    preserved_count = sum(df[c].notna().sum() for c in cols_to_preserve)
                    log(f"[normalize_transactions] Preserved {len(cols_to_preserve)} enrichment columns with {preserved_count:,} values")
    except Exception as e:
        log(f"[normalize_transactions][WARN] Failed to preserve enrichment columns: {e}")

    # Normalize numeric columns before writing (prevents string->numeric conversion errors)
    df = normalize_numeric_columns(df)

    # Write to canonical location
    out_csv = out_path.parent / "transactions.csv"
    write_parquet_robust(df, out_path, description="transactions data")
    df.to_csv(out_csv, index=False)
    log(f"[POST] wrote transactions CSV: {out_csv}")

    return out_path


def normalize_schedule_parquet(
    context: str,
    log: Callable[..., None] = print
) -> Path:
    """
    Ensure schedule.parquet exists at canonical location with proper normalization.

    Adds missing composite keys if not present:
    - cumulative_week: year * 100 + week (e.g., 202401 for 2024 week 1)
    - manager_week: manager (no spaces) + cumulative_week
    - manager_year: manager (no spaces) + year

    Args:
        context: Path to league_context.json
        log: Logging function (default: print)

    Returns:
        Path to canonical schedule.parquet file

    Raises:
        FileNotFoundError: If schedule.parquet not found
        ValueError: If schedule.parquet is empty
    """
    ctx = _load_ctx(context)

    # Try likely locations (check main league directory first, then subdirectories)
    candidates = []
    if getattr(ctx, "data_directory", None):
        candidates.append(Path(ctx.data_directory) / "schedule.parquet")
        # Legacy support for old naming convention
        candidates.append(Path(ctx.data_directory) / "schedule_data_all_years.parquet")
    if getattr(ctx, "schedule_data_directory", None):
        candidates.append(Path(ctx.schedule_data_directory) / "schedule.parquet")

    found = None
    for c in candidates:
        if c and c.exists():
            found = c
            break
    if not found:
        raise FileNotFoundError("schedule.parquet not found in expected locations")

    df = pd.read_parquet(found)
    if df.empty:
        raise ValueError("schedule.parquet is empty")

    # Normalize year/week to Int64
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Add cumulative_week if missing (YYYYWW format)
    if "cumulative_week" not in df.columns:
        if "year" in df.columns and "week" in df.columns:
            df["cumulative_week"] = (
                df["year"].fillna(0) * 100 +
                df["week"].fillna(0)
            ).astype("Int64")
            log("[normalize_schedule] Added cumulative_week column (year*100 + week)")
        else:
            df["cumulative_week"] = pd.NA

    # Fix manager_week if it's just an integer (old format)
    if "manager_week" in df.columns:
        # Check if manager_week is numeric (old format = just week number)
        try:
            numeric_check = pd.to_numeric(df["manager_week"], errors="coerce")
            if numeric_check.notna().sum() == len(df):
                # It's all numeric - needs fixing
                log("[normalize_schedule] Fixing manager_week (was just week number, updating to manager + cumulative_week)")
                if "manager" in df.columns and "cumulative_week" in df.columns:
                    manager_no_spaces = df["manager"].astype(str).str.replace(" ", "", regex=False)
                    cumulative_str = df["cumulative_week"].astype(str)
                    df["manager_week"] = manager_no_spaces + cumulative_str
                else:
                    log("[normalize_schedule] Warning: Cannot fix manager_week - missing manager or cumulative_week column")
        except Exception:
            # If conversion fails, assume it's already in the correct format
            pass
    elif "manager" in df.columns and "cumulative_week" in df.columns:
        # manager_week column doesn't exist - create it
        manager_no_spaces = df["manager"].astype(str).str.replace(" ", "", regex=False)
        cumulative_str = df["cumulative_week"].astype(str)
        df["manager_week"] = manager_no_spaces + cumulative_str
        log("[normalize_schedule] Added manager_week column (manager + cumulative_week)")

    # Fix manager_year if it's just an integer (old format)
    if "manager_year" in df.columns:
        # Check if manager_year is numeric (old format = just year)
        try:
            numeric_check = pd.to_numeric(df["manager_year"], errors="coerce")
            if numeric_check.notna().sum() == len(df):
                # It's all numeric - needs fixing
                log("[normalize_schedule] Fixing manager_year (was just year, updating to manager + year)")
                if "manager" in df.columns and "year" in df.columns:
                    manager_no_spaces = df["manager"].astype(str).str.replace(" ", "", regex=False)
                    year_str = df["year"].astype(str)
                    df["manager_year"] = manager_no_spaces + year_str
                else:
                    log("[normalize_schedule] Warning: Cannot fix manager_year - missing manager or year column")
        except Exception:
            # If conversion fails, assume it's already in the correct format
            pass
    elif "manager" in df.columns and "year" in df.columns:
        # manager_year column doesn't exist - create it
        manager_no_spaces = df["manager"].astype(str).str.replace(" ", "", regex=False)
        year_str = df["year"].astype(str)
        df["manager_year"] = manager_no_spaces + year_str
        log("[normalize_schedule] Added manager_year column (manager + year)")

    # Normalize numeric columns before writing (prevents string->numeric conversion errors)
    df = normalize_numeric_columns(df)

    # Write to canonical location
    out_path = Path(ctx.data_directory) / "schedule.parquet"
    out_csv = out_path.parent / "schedule.csv"
    write_parquet_robust(df, out_path, description="schedule data")
    df.to_csv(out_csv, index=False)
    log(f"[POST] wrote schedule CSV: {out_csv}")

    return out_path


def ensure_fantasy_points_alias(
    context: str,
    log: Callable[..., None] = print
) -> Path:
    """
    Ensure player.parquet has a 'fantasy_points' column as an alias
    to the actual points column (which varies by scoring type).

    Args:
        context: Path to league_context.json
        log: Logging function (default: print)

    Returns:
        Path to player.parquet file

    Raises:
        FileNotFoundError: If player.parquet not found
        ValueError: If player.parquet is empty
        KeyError: If no suitable points column found
    """
    ctx = _load_ctx(context)
    pfile = ctx.canonical_player_file

    if not pfile.exists():
        raise FileNotFoundError(f"player.parquet not found at {pfile}")

    df = pd.read_parquet(pfile)
    if df.empty:
        raise ValueError("player.parquet is empty")

    candidates = [
        "fantasy_points", "fantasy_points_total", "fantasy_points_ppr",
        "fpts", "points", "FPTS", "fantasy_points_std"
    ]
    chosen = None
    for c in candidates:
        if c in df.columns:
            chosen = c
            break

    if chosen is None:
        raise KeyError("No suitable points column found to alias to 'fantasy_points'")

    if "fantasy_points" not in df.columns:
        df["fantasy_points"] = df[chosen]
        write_parquet_robust(df, pfile)  # Removed log=log parameter
        log(f"[POST] Added fantasy_points alias from {chosen}")

    return pfile
