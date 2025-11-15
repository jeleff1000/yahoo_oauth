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
    "year", "week", "manager", "opponent", "manager_team", "opponent_team",
    "manager_score", "opponent_score", "is_playoffs", "is_consolation",
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

    # LEAGUE-SPECIFIC: Auto-inject ESPN 2013 data for KMFFL if CSV exists
    # This is a one-time migration quirk that doesn't affect scalability
    inject_espn_2013_matchups(context, log=log)

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

    for season in years:
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

            # Filter to current league_id
            df = df[df["league_id"] == ctx.league_id].copy()
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

            # STANDARDIZE COLUMN NAMES (Yahoo vs ESPN naming differences)
            # Yahoo fetcher creates: team_name, team_points, opponent_points
            # Expected schema: manager_team, manager_score, opponent_score, opponent_team
            column_renames = {}

            # Rename team_name -> manager_team (if manager_team doesn't exist)
            if "team_name" in df.columns and "manager_team" not in df.columns:
                column_renames["team_name"] = "manager_team"

            # Rename team_points -> manager_score (if manager_score doesn't exist)
            if "team_points" in df.columns and "manager_score" not in df.columns:
                column_renames["team_points"] = "manager_score"

            # Rename opponent_points -> opponent_score (if opponent_score doesn't exist)
            if "opponent_points" in df.columns and "opponent_score" not in df.columns:
                column_renames["opponent_points"] = "opponent_score"

            # Rename team_projected_points -> manager_proj_score (if manager_proj_score doesn't exist)
            if "team_projected_points" in df.columns and "manager_proj_score" not in df.columns:
                column_renames["team_projected_points"] = "manager_proj_score"

            # Rename opponent_projected_points -> opponent_proj_score (if opponent_proj_score doesn't exist)
            if "opponent_projected_points" in df.columns and "opponent_proj_score" not in df.columns:
                column_renames["opponent_projected_points"] = "opponent_proj_score"

            if column_renames:
                df = df.rename(columns=column_renames)
                log(f"      [RENAME] {f.name}: {list(column_renames.keys())} -> {list(column_renames.values())}")

            # Add opponent_team if missing (lookup from manager -> manager_team mapping)
            if "opponent_team" not in df.columns or df["opponent_team"].isna().all():
                if "opponent" in df.columns and "manager_team" in df.columns:
                    manager_to_team = df.groupby("manager")["manager_team"].first().to_dict()
                    df["opponent_team"] = df["opponent"].map(manager_to_team)
                    if df["opponent_team"].notna().any():
                        log(f"      [ADD] {f.name}: Added opponent_team mapping")

            # Ensure required columns exist (only add NULL if still missing after renames)
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

    # Sort for deterministic output
    sort_cols = [c for c in ["year", "week", "manager"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # Add backward compatibility aliases for transformation scripts
    # Many scripts still reference team_points/opponent_points
    if "manager_score" in merged.columns and "team_points" not in merged.columns:
        merged["team_points"] = merged["manager_score"]
        log(f"[COMPAT] Added team_points as alias for manager_score")

    if "opponent_score" in merged.columns and "opponent_points" not in merged.columns:
        merged["opponent_points"] = merged["opponent_score"]
        log(f"[COMPAT] Added opponent_points as alias for opponent_score")

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
    Ensure draft.parquet exists at canonical location with proper normalization.

    Tries multiple candidate locations:
    - draft_data_directory/draft.parquet
    - draft_data_directory/drafts.parquet
    - data_directory/draft.parquet

    Normalizes:
    - yahoo_player_id -> string
    - year -> Int64
    - league_id -> added if missing

    Args:
        context: Path to league_context.json
        log: Logging function (default: print)

    Returns:
        Path to canonical draft.parquet file

    Raises:
        FileNotFoundError: If draft.parquet not found
        ValueError: If draft.parquet is empty
    """
    ctx = _load_ctx(context)

    # Try likely locations (check main league directory first, then subdirectories)
    candidates = []
    if getattr(ctx, "data_directory", None):
        candidates.append(Path(ctx.data_directory) / "draft.parquet")
        candidates.append(Path(ctx.data_directory) / "drafts.parquet")
        candidates.append(Path(ctx.data_directory) / "draft_data_all_years.parquet")
    if getattr(ctx, "draft_data_directory", None):
        candidates.append(Path(ctx.draft_data_directory) / "draft.parquet")
        candidates.append(Path(ctx.draft_data_directory) / "drafts.parquet")
        candidates.append(Path(ctx.draft_data_directory) / "draft_data_all_years.parquet")

    found = None
    for c in candidates:
        if c and c.exists():
            found = c
            break
    if not found:
        raise FileNotFoundError("draft.parquet not found in expected locations")

    df = pd.read_parquet(found)
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

    # Ensure league_id exists
    if "league_id" not in df.columns:
        df["league_id"] = ctx.league_id
        log(f"[normalize_draft] Added league_id: {ctx.league_id}")

    # Preserve enrichment columns from existing canonical file if they exist
    # (These are added by transformation scripts like player_to_draft_v2.py)
    enrichment_cols = [
        'total_fantasy_points', 'season_ppg', 'games_played', 'games_with_points',
        'season_std', 'best_game', 'worst_game', 'weeks_rostered', 'weeks_started',
        'season_overall_rank', 'season_position_rank', 'total_position_players',
        'price_rank_within_position', 'pick_rank_within_position',
        'price_rank_vs_finish_rank', 'pick_rank_vs_finish_rank',
        'points_per_dollar', 'points_per_pick', 'value_over_replacement', 'draft_position_delta'
    ]

    out_path = ctx.canonical_draft_file
    try:
        if out_path.exists():
            existing_canonical = pd.read_parquet(out_path)

            # Check which enrichment columns exist and have data in canonical
            cols_to_preserve = [c for c in enrichment_cols if c in existing_canonical.columns
                               and existing_canonical[c].notna().sum() > 0
                               and c not in df.columns]  # Only preserve if not already in source

            if cols_to_preserve:
                join_keys = ['yahoo_player_id', 'year']
                enrichment_data = existing_canonical[join_keys + cols_to_preserve].copy()

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
                log(f"[normalize_draft] Preserved {len(cols_to_preserve)} enrichment columns with {preserved_count:,} values")
    except Exception as e:
        log(f"[normalize_draft][WARN] Failed to preserve enrichment columns: {e}")

    # Normalize numeric columns before writing (prevents string->numeric conversion errors)
    df = normalize_numeric_columns(df)

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
    # Use transaction_id if available, otherwise use composite key
    if "transaction_id" in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        if len(df) < initial_count:
            log(f"[normalize_transactions] Removed {initial_count - len(df):,} duplicate transaction_ids")
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
        candidates.append(Path(ctx.data_directory) / "schedule_data_all_years.parquet")
    if getattr(ctx, "schedule_data_directory", None):
        candidates.append(Path(ctx.schedule_data_directory) / "schedule_data_all_years.parquet")
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


def inject_espn_2013_matchups(
    context: str,
    csv_path: Optional[str] = None,
    log: Callable[..., None] = print
) -> None:
    """
    Inject 2013 ESPN matchup data from CSV for KMFFL league only.

    *** LEAGUE-SPECIFIC QUIRK - DOES NOT AFFECT SCALABILITY ***

    This function handles a one-time migration for KMFFL (league_id "449.l.198278")
    which started on ESPN in 2013 before moving to Yahoo. This year ONLY has:
    - Matchup data (manually reconstructed from historical records)
    - NO player stats, draft data, or transaction data (treated as pre-league year)

    The 2013 season was an 8-team league with 4/8 playoff teams, no bye week.
    Championship: Gavi beat Yaacov. Playoff scores (weeks 13-14) were lost to history.

    This function:
    1. Only runs for KMFFL (league_id "449.l.198278")
    2. Automatically looks for CSV in standard location: matchup_data/matchup_data_week_all_year_2013.csv
    3. Converts CSV to weekly parquet files that merge_matchups_to_parquet() can consume
    4. Silently skips if league is not KMFFL or CSV is not found

    This quirk does NOT affect:
    - Other leagues (completely ignored for non-KMFFL leagues)
    - System scalability (one-time 8-team, 14-week dataset)
    - Data pipeline (treated as normal matchup data after injection)

    Args:
        context: Path to league_context.json
        csv_path: Optional path to CSV file. If None, auto-detects from matchup_data_directory
        log: Logging function (default: print)
    """
    ctx = _load_ctx(context)

    # LEAGUE-SPECIFIC: Only run for KMFFL league
    if ctx.league_id != "449.l.198278":
        return  # Silently skip for all other leagues

    # Auto-detect CSV path if not provided
    if csv_path is None:
        matchup_dir = Path(ctx.matchup_data_directory)
        csv_path = matchup_dir / "matchup_data_week_all_year_2013.csv"
    else:
        csv_path = Path(csv_path)

    # Silently skip if CSV doesn't exist (already processed or not needed)
    if not csv_path.exists():
        return

    log(f"[ESPN 2013][KMFFL-SPECIFIC] Injecting 2013 ESPN matchup data from {csv_path.name}")
    log(f"[ESPN 2013] This is a one-time migration quirk for KMFFL's pre-Yahoo season")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter out empty rows (some rows at the end have no data)
    df = df[df["manager"].notna()].copy()

    log(f"[ESPN 2013] Loaded {len(df)} matchup records from CSV")

    # Column mapping from CSV to expected schema
    # The CSV now comes from matchup_data_week_all_year_2013.csv which already has most columns
    column_mapping = {
        "team_name": "manager_team",
        "team_points": "manager_score",
        "team_projected_points": "manager_proj_score",
        "opponent_points": "opponent_score",
        "opponent_projected_points": "opponent_proj_score",
    }

    # Rename columns (only rename if they exist and target doesn't already exist)
    renames_to_apply = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            renames_to_apply[old_col] = new_col

    if renames_to_apply:
        df = df.rename(columns=renames_to_apply)
        log(f"[ESPN 2013] Renamed columns: {renames_to_apply}")

    # Add league_id
    df["league_id"] = ctx.league_id

    # Ensure year/week are Int64
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Add opponent_team if not already present
    # The cleaned CSV should already have this, but add it as fallback
    if "opponent_team" not in df.columns or df["opponent_team"].isna().all():
        # Create a lookup: manager -> team_name
        manager_to_team = df.groupby("manager")["manager_team"].first().to_dict()
        df["opponent_team"] = df["opponent"].map(manager_to_team)
        log(f"[ESPN 2013] Added opponent_team mapping")

    # Ensure required columns exist with defaults
    for col, default in OPTIONAL_MATCHUP_COLS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Convert is_playoffs to boolean (currently 1/0 in CSV)
    if "is_playoffs" in df.columns:
        df["is_playoffs"] = df["is_playoffs"].fillna(0).astype(bool)
    if "is_consolation" in df.columns:
        df["is_consolation"] = df["is_consolation"].fillna(0).astype(bool)

    # KMFFL 2013 SPECIAL HANDLING: Playoff games (weeks 13-14) have winners/losers but NO SCORES
    # This is BY DESIGN - playoff scores were lost to history
    # Fill in score-based derived columns with appropriate defaults for these games
    null_score_mask = df["manager_score"].isna() & df["opponent_score"].isna()

    if null_score_mask.any():
        log(f"[ESPN 2013] Found {null_score_mask.sum()} playoff rows with no scores (weeks 13-14)")
        log(f"[ESPN 2013] Setting score-based metrics to null/zero for these games")

        # For games with null scores, set derived score columns to appropriate values
        df.loc[null_score_mask, "margin"] = pd.NA
        df.loc[null_score_mask, "total_matchup_score"] = pd.NA
        df.loc[null_score_mask, "close_margin"] = 0  # Not close if no score
        df.loc[null_score_mask, "weekly_mean"] = pd.NA
        df.loc[null_score_mask, "weekly_median"] = pd.NA
        df.loc[null_score_mask, "teams_beat_this_week"] = 0  # Can't compare null scores
        df.loc[null_score_mask, "opponent_teams_beat_this_week"] = 0

        # Projection-based columns should also be null
        for col in ["proj_score_error", "abs_proj_score_error", "above_proj_score", "below_proj_score",
                    "expected_spread", "expected_odds", "win_vs_spread", "lose_vs_spread",
                    "underdog_wins", "favorite_losses", "proj_wins", "proj_losses"]:
            if col in df.columns:
                df.loc[null_score_mask, col] = pd.NA

        # CRITICAL: win/loss columns should REMAIN intact (these are known!)
        # Don't overwrite these - they're the only data we have

    # Add composite keys
    df = add_composite_keys(df)

    # Get matchup data directory from context
    matchup_dir = Path(ctx.matchup_data_directory)
    matchup_dir.mkdir(parents=True, exist_ok=True)

    # Save as single yearly file (matching the format of other years)
    out_file = matchup_dir / "matchup_data_week_all_year_2013.parquet"
    df.to_parquet(out_file, index=False)

    log(f"[ESPN 2013] Created matchup file: {out_file.name}")
    log(f"[ESPN 2013] Total matchups: {len(df)}")
    log(f"[ESPN 2013] Weeks: {sorted(df['week'].dropna().unique())}")

