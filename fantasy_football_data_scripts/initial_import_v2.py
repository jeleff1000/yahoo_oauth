#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INITIAL IMPORT V2 - Unified Multi-League Pipeline

What this does:
1) Fetchers: Yahoo, NFL offense, DST (per-week as needed), matchups, schedule, draft (all years), transactions (all years)
2) Weekly merges: Yahoo+NFL (+DST when present) across all seasons -> unified player.parquet
   - Normalizes Yahoo & NFL multi-year/all-weeks files into per-week parquet on-the-fly
   - DST missing files are warnings (offense-only still merges)
3) Transformations: cumulative stats, expected record, playoff odds, player<->matchup, player->transactions, draft->player, keeper economics
4) Aggregations & Canonical checks: writes matchup.parquet, transactions.parquet, draft.parquet; validates outputs

Usage:
  python initial_import_v2.py --context path/to/league_context.json
  python initial_import_v2.py --context path/to/league_context.json --dry-run
  python initial_import_v2.py --context path/to/league_context.json --skip-fetchers
  python initial_import_v2.py --context path/to/league_context.json --skip-transformations
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import pandas as pd
import duckdb  # <-- needed by verify_unified_outputs()

# Import new modular utilities
from multi_league.core.data_normalization import (
    normalize_numeric_columns,
    write_parquet_robust,
    validate_league_isolation,
    add_composite_keys,
    ensure_league_id
)
from multi_league.core.script_runner import log, run_script, setup_oauth_environment
from multi_league.core.yahoo_league_settings import fetch_league_settings
from multi_league.data_fetchers.aggregators import (
    merge_matchups_to_parquet,
    normalize_draft_parquet,
    normalize_transactions_parquet,
    normalize_schedule_parquet,
    ensure_fantasy_points_alias
)
from multi_league.core.league_context import LeagueContext

# canonical script dir
SCRIPT_DIR = Path(__file__).parent

TERMINAL_ERROR_PATTERNS = [
    r"no (league|data) found",
    r"no data returned",
    r"404",
    r"league.*not found",
]
RECOVERABLE_ERROR_PATTERNS = [
    r"RecoverableAPIError",
    r"Request denied",
    r"temporarily unavailable",
]

# Helper to load context (compatibility wrapper)
def _load_ctx(context_path: str):
    return LeagueContext.load(context_path)

# --------------------------------------------------------------------------------------
# Child script configuration
# --------------------------------------------------------------------------------------

DATA_FETCHERS = [
    ("multi_league/data_fetchers/weekly_matchup_data_v2.py", "Matchup data"),  # Run FIRST so player data can use max week
    ("multi_league/data_fetchers/yahoo_fantasy_data.py", "Yahoo player data"),
    ("multi_league/data_fetchers/nfl_offense_stats.py", "NFL offense stats"),
    ("multi_league/data_fetchers/defense_stats.py", "NFL defense stats"),
    ("multi_league/data_fetchers/draft_data_v2.py", "Draft data (all years)"),
    ("multi_league/data_fetchers/transactions_v2.py", "Transactions (all years)"),
    ("multi_league/data_fetchers/season_schedules.py", "Schedule data (league week windows)"),
]

DST_FETCHER = "multi_league/data_fetchers/defense_stats.py"
# Use the modular scripts from multi_league/data_fetchers/
YAHOO_NFL_MERGE = "multi_league/data_fetchers/yahoo_nfl_merge.py"
COMBINE_DST = "multi_league/data_fetchers/combine_dst_to_nfl.py"
NFL_OFFENSE_FETCHER = "multi_league/data_fetchers/nfl_offense_stats.py"
YAHOO_FETCHER = "multi_league/data_fetchers/yahoo_fantasy_data.py"

# Transformation scripts are executed in sequence.  The order here is important:
# CRITICAL: Split into multiple passes to handle dependencies
# Pass 1: Base calculations (no dependencies)
TRANSFORMATIONS_PASS_1 = [
    ("multi_league/transformations/base/cumulative_stats_v2.py", "Cumulative Stats", 600),  # FIX matchup playoff flags FIRST using seed-based detection
    ("multi_league/transformations/base/enrich_schedule_with_playoff_flags.py", "Enrich Schedule w/ Playoff Flags", 120),  # Merge playoff flags from matchup into schedule (needed by playoff_odds_import)
]

# Pass 2: Joins that need cumulative stats
TRANSFORMATIONS_PASS_2 = [
    ("multi_league/transformations/player_enrichment/matchup_to_player_v2.py", "Matchup -> Player", 600),  # Join matchup columns INTO player (now with fixed playoff flags)
    ("multi_league/transformations/player_enrichment/player_stats_v2.py", "Player Stats", 900),  # MUST run before player_to_matchup (adds optimal_points to player)
    ("multi_league/transformations/player_enrichment/replacement_level_v2.py", "Replacement Levels", 600),  # Calculate position replacement baselines for SPAR (needs fantasy_points from player_stats)
]

# Pass 3: Everything else that depends on above
TRANSFORMATIONS_PASS_3 = [
    ("multi_league/transformations/matchup_enrichment/player_to_matchup_v2.py", "Player -> Matchup", 600),  # Join player aggregates INTO matchup
    # CRITICAL ENRICHMENT ORDER: Draft enrichment MUST run BEFORE transaction enrichment
    # This ensures transactions_to_player preserves the draft columns added by draft_to_player
    # CRITICAL KEEPER ECONOMICS ORDER: Must run in this exact sequence for keeper columns to reach player table
    ("multi_league/transformations/draft_enrichment/player_to_draft_v2.py", "Player -> Draft", 600),  # [1] Add player stats TO draft (needed for SPAR calculations)
    ("multi_league/transformations/draft_enrichment/draft_value_metrics_v3.py", "Draft SPAR Metrics", 600),  # [2] Calculate SPAR + keeper economics, add TO draft (creates kept_next_year, spar, pgvor, etc.)
    ("multi_league/transformations/player_enrichment/draft_to_player_v2.py", "Draft -> Player", 600),  # [3] Import keeper/draft columns FROM draft TO player (now they exist!)
    # Transaction enrichment runs AFTER draft so it preserves draft columns when it writes player.parquet
    ("multi_league/transformations/transaction_enrichment/fix_unknown_managers.py", "Fix Unknown Managers", 120),  # Fix manager="Unknown" by backfilling from most recent add - MUST run before any transaction joins
    ("multi_league/transformations/transaction_enrichment/player_to_transactions_v2.py", "Player <-> Transactions", 600),  # Add ROS performance TO transactions - MUST run before transaction_value_metrics
    ("multi_league/transformations/transaction_enrichment/transaction_value_metrics_v3.py", "Transaction SPAR Metrics", 600),  # SPAR-based transaction value (replaces old VOR metrics in player_to_transactions)
    ("multi_league/transformations/player_enrichment/transactions_to_player_v2.py", "Transactions -> Player", 600),  # Add FAAB data TO player (preserves draft columns added above)
    ("multi_league/transformations/draft_enrichment/keeper_economics_v2.py", "Keeper Economics", 600),  # Calculate keeper_price for next year planning (needs draft cost + FAAB from transactions)
    ("multi_league/transformations/matchup_enrichment/expected_record_v2.py", "Expected Record (V2)", 900),  # Needs wins_to_date and playoff_seed_to_date from cumulative_stats
    ("multi_league/transformations/matchup_enrichment/playoff_odds_import.py", "Playoff Odds", None),  # No timeout - Monte Carlo simulations can take a while
    ("multi_league/transformations/aggregation/aggregate_player_season_v2.py", "Aggregate Player Season", 600),  # Create players_by_year
    ("multi_league/transformations/finalize/normalize_canonical_types.py", "Normalize Join Key Types", 120),  # MUST BE LAST - ensures all join keys have consistent Int64 types for cross-table joins
    # ("multi_league/transformations/validation/validate_outputs.py", "Validate Outputs", 600),  # TODO: Create this script
]

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def verify_unified_outputs(ctx):
    summary = {}
    for name, path in {
        "player": ctx.canonical_player_file,
        "matchup": ctx.canonical_matchup_file,
        "draft": ctx.canonical_draft_file,
        "transactions": ctx.canonical_transaction_file,
    }.items():
        if path.exists():
            res = duckdb.sql(f"SELECT count(*), min(year), max(year) FROM '{path}'").fetchone()
            summary[name] = {"rows": res[0], "min_year": res[1], "max_year": res[2]}
        else:
            summary[name] = "missing"
    log(f"[SUMMARY] Unified outputs:\n{json.dumps(summary, indent=2)}")

def ensure_nfl_week_materialized(ctx, year: int, week: int, log):
    """
    Make sure we have an NFL offense/defense parquet for (year, week) that the merger can read.

    We consider a week "materialized" if we can point to or create:
        player_stats_{YEAR}_week_{WEEK}.parquet

    We will try, in order:
      1. Does a per-week file for (year, week) already exist under any known naming pattern?
      2. Can we slice it out of a season-all-weeks file we created ourselves
         (player_stats_{YEAR}_allweeks.parquet / player_stats_{YEAR}_all_weeks.parquet)?
      3. Can we slice it out of nflverse's seasonal dump
         stats_player_week_{YEAR}.parquet (columns: season, week, etc.)?

    Returns:
        True  -> per-week parquet now exists / was created
        False -> we really do not have that week
    """
    import pandas as pd
    base = Path(ctx.player_data_directory)

    # ---- 1. If a usable per-week file already exists, we're done.
    weekly_candidates = [
        base / f"player_stats_{year}_week_{week}.parquet",      # new per-week output (preferred)
        base / f"player_stats_{year}_w{week}.parquet",          # alt per-week naming
        base / f"nfl_offense_stats_{year}_week_{week}.parquet", # legacy naming (back-compat)
        base / f"nfl_combined_{year}_week_{week}.parquet",      # combined offense+dst if already built
    ]
    for wc in weekly_candidates:
        if wc.exists() and wc.stat().st_size > 0:
            return True

    # Helper to slice a "season file" down to a single (year, week) and write it
    def _try_slice(source_path: Path, season_col: str, week_col: str) -> bool:
        if not (source_path.exists() and source_path.stat().st_size > 0):
            return False
        try:
            df = pd.read_parquet(source_path)
        except Exception as e:
            log(f"[materialize] Failed reading {source_path.name}: {e}")
            return False

        # Normalize numeric just in case
        yvals = pd.to_numeric(df.get(season_col), errors="coerce")
        wvals = pd.to_numeric(df.get(week_col),   errors="coerce")

        if yvals is None or wvals is None:
            return False

        sub = df[(yvals == year) & (wvals == week)].copy()
        if sub.empty:
            return False

        out_path = base / f"player_stats_{year}_week_{week}.parquet"
        try:
            sub.to_parquet(out_path, index=False)
            log(f"[materialize] Created {out_path.name} with {len(sub)} rows from {source_path.name}")
            return True
        except Exception as e:
            log(f"[materialize] Failed to write {out_path.name} from {source_path.name}: {e}")
            return False

    # ---- 2. Try our own season-wide "allweeks" parquet(s)
    our_season_candidates = [
        base / f"player_stats_{year}_allweeks.parquet",
        base / f"player_stats_{year}_all_weeks.parquet",
        base / f"nfl_offense_stats_{year}_all_weeks.parquet",  # very old naming
    ]
    for src in our_season_candidates:
        # our season parquet is usually {season: <year>, week: <wk>} OR {year, week}
        # First try ('season','week'), then ('year','week')
        if _try_slice(src, "season", "week"):
            return True
        if _try_slice(src, "year", "week"):
            return True

    # ---- 3. Try nflverse season dump (stats_player_week_<YEAR>.parquet)
    # This is exactly what you just showed me from github.com/nflverse/... .
    # It's one parquet per season, all weeks, with columns like: season, week, player_id, player_name, etc.
    nflverse_candidate = base / f"stats_player_week_{year}.parquet"
    if _try_slice(nflverse_candidate, "season", "week"):
        return True

    # If we got here, we couldn't find or build that week.
    log(f"[materialize] NFL {year} W{week} not found in any source")
    return False

def main():
    parser = argparse.ArgumentParser(description="Complete historical data import for a fantasy league (Unified)")
    parser.add_argument("--context", required=True, help="Path to league_context.json")
    parser.add_argument("--dry-run", action="store_true", help="Run all scripts in dry-run mode when supported")
    parser.add_argument("--skip-fetchers", action="store_true", help="Skip data fetchers (use existing files)")
    parser.add_argument("--skip-transformations", action="store_true", help="Skip transformations")
    parser.add_argument("--start-phase", type=int, choices=[1, 2, 3], help="Start at specific phase (1=fetchers, 2=merges, 3=transformations)")
    parser.add_argument("--utility", action="store_true", help="Run utility functions (matchups, draft, transactions, schedule, fantasy points)")
    parser.add_argument("--util-action", choices=["merge_matchups", "normalize_draft", "normalize_transactions", "normalize_schedule", "points_alias", "all"], help="Which utility to run when --utility is set")
    parser.add_argument("--years", nargs="*", type=int, help="Optional years for merge_matchups (e.g. --years 2021 2022)")
    args = parser.parse_args()

    context_path = Path(args.context).resolve()
    if not context_path.exists():
        log(f"[FAIL] League context not found: {context_path}")
        sys.exit(1)

    ctx = _load_ctx(str(context_path))

    # If utilities requested, run them and exit
    if args.utility:
        action = args.util_action or "all"
        log(f"[UTILITY] Running utility action: {action}")
        try:
            if action in ("merge_matchups", "all"):
                yrs = args.years if args.years else None
                path = merge_matchups_to_parquet(str(context_path), years=yrs)
                log(f"[UTILITY] merge_matchups wrote: {path}")
            if action in ("normalize_draft", "all"):
                path = normalize_draft_parquet(str(context_path))
                log(f"[UTILITY] normalize_draft wrote: {path}")
            if action in ("normalize_transactions", "all"):
                path = normalize_transactions_parquet(str(context_path))
                log(f"[UTILITY] normalize_transactions wrote: {path}")
            if action in ("normalize_schedule", "all"):
                path = normalize_schedule_parquet(str(context_path))
                log(f"[UTILITY] normalize_schedule wrote: {path}")
            if action in ("points_alias", "all"):
                path = ensure_fantasy_points_alias(str(context_path))
                log(f"[UTILITY] points_alias updated: {path}")
        except Exception as e:
            log(f"[UTILITY][ERROR] {e}")
            sys.exit(2)
        log("[UTILITY] Completed.")
        return

    # Determine starting phase (handles both new --start-phase and legacy --skip-fetchers)
    if args.start_phase:
        start_phase = args.start_phase
    elif args.skip_fetchers:
        start_phase = 2  # Legacy: --skip-fetchers meant start at merges
    else:
        start_phase = 1  # Default: start at beginning

    log("=" * 96)
    log("INITIAL IMPORT V2 - Unified Pipeline")
    log("=" * 96)
    log(f"League: {ctx.league_name} | League ID: {ctx.league_id}")
    log(f"Years:  {ctx.start_year} - {ctx.end_year or 'current'}")
    log(f"Root:   {ctx.data_directory}")
    log(f"DryRun: {'Yes' if args.dry_run else 'No'}")
    log(f"Start Phase: {start_phase} ({'Settings/Fetchers' if start_phase == 1 else 'Merges' if start_phase == 2 else 'Transformations'})")
    log("=" * 96)

    results: Dict[str, List[Tuple[str, bool]]] = {"settings": [], "fetchers": [], "merges": [], "transformations": []}
    extra_args = ["--dry-run"] if args.dry_run else []

    # -------------------------------------------------------------------------
    # PHASE 0: LEAGUE SETTINGS (fetch all league configuration first)
    # -------------------------------------------------------------------------
    if start_phase <= 1:
        log("\n" + "=" * 96)
        log("PHASE 0: League Settings Discovery")
        log("=" * 96)
        log("[INFO] Fetching league settings for all years from Yahoo API")
        log("[INFO] All other scripts will READ these settings (no additional API calls)")

        # Determine settings directory (league-wide, not player-specific)
        settings_dir = Path(ctx.data_directory) / "league_settings"
        settings_dir.mkdir(parents=True, exist_ok=True)

        # Fetch settings for each year IN PARALLEL (10-12x faster than sequential)
        years = list(range(ctx.start_year, (ctx.end_year or datetime.now().year) + 1))
        log(f"[SETTINGS] Fetching settings for {len(years)} years in parallel (max 5 concurrent requests)...")

        all_settings_ok = True
        successful_years = []
        failed_years = []

        def fetch_year_settings(year):
            """Fetch settings for a single year (runs in parallel)"""
            try:
                settings = fetch_league_settings(
                    year=year,
                    league_key=None,  # Let auto-discovery find correct league key for each year
                    settings_dir=settings_dir,
                    context=str(context_path)
                )
                if settings:
                    return (year, True, settings)
                else:
                    return (year, False, None)
            except Exception as e:
                return (year, False, str(e))

        # Use ThreadPoolExecutor for parallel fetching (max 5 workers to avoid API rate limits)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all years for parallel processing
            future_to_year = {executor.submit(fetch_year_settings, year): year for year in years}

            # Process results as they complete
            for future in as_completed(future_to_year):
                year, success, result = future.result()

                if success:
                    num_teams = result.get("metadata", {}).get("num_teams", "?")
                    playoff_teams = result.get("metadata", {}).get("playoff_teams", "?")
                    bye_teams = result.get("metadata", {}).get("bye_teams", "?")
                    log(f"[SETTINGS] [OK] {year}: {num_teams} teams, {playoff_teams} playoff spots, {bye_teams} byes")
                    successful_years.append(year)
                else:
                    error_msg = result if isinstance(result, str) else "Failed to fetch settings"
                    log(f"[SETTINGS] [FAIL] {year}: {error_msg}")
                    failed_years.append(year)
                    all_settings_ok = False

        # Summary
        log(f"\n[SETTINGS] Parallel fetch complete: {len(successful_years)} succeeded, {len(failed_years)} failed")
        if failed_years:
            log(f"[SETTINGS] Failed years: {sorted(failed_years)}")

        results["settings"].append(("League Settings (all years)", all_settings_ok))
        log("\n[SETTINGS] Settings fetch complete. All data fetchers will use these saved settings.")
        log("=" * 96)

    # -------------------------------------------------------------------------
    # PHASE 1: FETCHERS
    # -------------------------------------------------------------------------
    if not args.skip_fetchers:
        log("\n" + "=" * 96)
        log("PHASE 1: Data Fetchers")
        log("=" * 96)

        for script_path, label in DATA_FETCHERS:
            # Handle original nfl_offense_stats.py (not v2)
            if "nfl_offense_stats.py" in script_path and "v2" not in script_path:
                # Original script: fetch NFL stats back to 1999
                nfl_start_year = 1999
                end_year = ctx.end_year or datetime.now().year
                all_ok = True
                for y in range(nfl_start_year, end_year + 1):
                    ok = run_script(
                        script_path,
                        f"{label} ({y})",
                        str(context_path),
                        additional_args=extra_args + ["--year", str(y), "--week", "0"],
                    )
                    all_ok = all_ok and ok
                results["fetchers"].append((label, all_ok))

            elif "defense_stats.py" in script_path:
                # Defense stats: fetch year by year like NFL offense stats
                nfl_start_year = 1999
                end_year = ctx.end_year or datetime.now().year
                all_ok = True
                for y in range(nfl_start_year, end_year + 1):
                    ok = run_script(
                        script_path,
                        f"{label} ({y})",
                        str(context_path),
                        additional_args=extra_args + ["--year", str(y)],
                    )
                    all_ok = all_ok and ok
                results["fetchers"].append((label, all_ok))

            elif "weekly_matchup_data_v2.py" in script_path:
                # Fetch matchups from Yahoo API for all years in league range
                end_year = ctx.end_year or datetime.now().year
                yahoo_years = list(range(ctx.start_year, end_year + 1))

                log(f"[INFO] Fetching Yahoo matchups for years: {yahoo_years}")

                all_ok = True
                for y in yahoo_years:
                    ok = run_script(
                        script_path, f"{label} ({y})", str(context_path),
                        additional_args=extra_args + ["--year", str(y)],
                        timeout=1200  # 20 minutes for matchup data
                    )
                    all_ok = all_ok and ok
                results["fetchers"].append((label, all_ok))

            # Handle original yahoo_fantasy_data.py (not v2)
            elif "yahoo_fantasy_data.py" in script_path and "v2" not in script_path:
                # Fetch ALL years from Yahoo API for player roster data
                # Yahoo API will naturally return empty results for years before the league existed
                end_year = ctx.end_year or datetime.now().year

                log(f"[INFO] Fetching Yahoo player data for years: {ctx.start_year}-{end_year}")

                ok = run_script(
                    script_path,
                    label,
                    str(context_path),
                    additional_args=extra_args + ["--year", "0", "--week", "0"],
                    timeout=3600  # Allow up to 1 hour for all years
                )
                results["fetchers"].append((label, ok))

            elif "draft_data_v2.py" in script_path:
                # Fetch draft data from Yahoo API for all years in league range
                end_year = ctx.end_year or datetime.now().year
                yahoo_years = list(range(ctx.start_year, end_year + 1))

                log(f"[INFO] Fetching Yahoo draft data for years: {yahoo_years}")

                all_ok = True
                for y in yahoo_years:
                    ok = run_script(
                        script_path, f"{label} ({y})", str(context_path),
                        additional_args=extra_args + ["--year", str(y)]
                    )
                    all_ok = all_ok and ok

                # Note: Draft aggregation now happens in PRE-TRANSFORMATION phase
                # (before transformations run) to ensure draft.parquet exists even when using --skip-fetchers

                results["fetchers"].append((label, all_ok))

            elif "transactions_v2.py" in script_path:
                # Fetch transactions from Yahoo API for all years in league range
                log(f"[INFO] Fetching Yahoo transactions for all years: {ctx.start_year}-{ctx.end_year or datetime.now().year}")

                ok = run_script(script_path, label, str(context_path),
                                additional_args=extra_args + ["--all-years"],
                                timeout=1800)  # 30 minutes for all transactions

                # Aggregate all individual year files into one combined transactions.parquet
                if ok:
                    try:
                        trans_dir = Path(ctx.transaction_data_directory)
                        year_files = sorted(trans_dir.glob("transactions_year_*.parquet"))
                        if year_files:
                            log(f"[AGGREGATE] Combining {len(year_files)} transaction year files...")
                            trans_dfs = []
                            for f in year_files:
                                df = pd.read_parquet(f)
                                trans_dfs.append(df)
                                log(f"      [LOAD] {f.name} ({len(df):,} rows)")

                            combined = pd.concat(trans_dfs, ignore_index=True)

                            # Deduplicate by transaction_id + player (preserves add+drop combos and trades)
                            # A single transaction can involve multiple players (add+drop, trades)
                            if "transaction_id" in combined.columns:
                                initial_count = len(combined)
                                # Try player_key first, then yahoo_player_id
                                if "player_key" in combined.columns:
                                    combined = combined.drop_duplicates(subset=["transaction_id", "player_key"], keep="first")
                                    dedup_key = "transaction_id + player_key"
                                elif "yahoo_player_id" in combined.columns:
                                    combined = combined.drop_duplicates(subset=["transaction_id", "yahoo_player_id"], keep="first")
                                    dedup_key = "transaction_id + yahoo_player_id"
                                else:
                                    # Fallback to transaction_id only if no player column exists
                                    combined = combined.drop_duplicates(subset=["transaction_id"], keep="first")
                                    dedup_key = "transaction_id (WARNING: may lose add/drop combos)"

                                if len(combined) < initial_count:
                                    log(f"      [DEDUP] Removed {initial_count - len(combined):,} duplicate records ({dedup_key})")

                            # Save to main league directory (not subdirectory)
                            combined_path = Path(ctx.data_directory) / "transactions.parquet"
                            combined.to_parquet(combined_path, index=False)
                            log(f"[AGGREGATE] Wrote combined transactions file: {combined_path} ({len(combined):,} rows)")
                        else:
                            log("[WARN] No transaction year files found to aggregate")
                            ok = False
                    except Exception as e:
                        log(f"[ERROR] Transactions aggregation failed: {e}")
                        ok = False

                results["fetchers"].append((label, ok))

            elif "season_schedules.py" in script_path:
                # Fetch schedule for all years in league range
                log(f"[INFO] Fetching schedule for ALL years: {ctx.start_year}-{ctx.end_year or datetime.now().year}")

                ok = run_script(script_path, label, str(context_path),
                                additional_args=extra_args + ["--all-years"],
                                timeout=1800)  # 30 minutes for all schedules

                # Aggregate all individual year files into one combined schedule.parquet
                if ok:
                    try:
                        sched_dir = Path(ctx.schedule_data_directory)
                        year_files = sorted(sched_dir.glob("schedule_data_year_*.parquet"))
                        if year_files:
                            log(f"[AGGREGATE] Combining {len(year_files)} schedule year files...")
                            sched_dfs = []
                            for f in year_files:
                                df = pd.read_parquet(f)
                                sched_dfs.append(df)
                                log(f"      [LOAD] {f.name} ({len(df):,} rows)")

                            combined = pd.concat(sched_dfs, ignore_index=True)
                            # Save to main league directory (not subdirectory)
                            combined_path = Path(ctx.data_directory) / "schedule.parquet"
                            combined.to_parquet(combined_path, index=False)
                            log(f"[AGGREGATE] Wrote combined schedule file: {combined_path} ({len(combined):,} rows)")
                        else:
                            log("[WARN] No schedule year files found to aggregate")
                            ok = False
                    except Exception as e:
                        log(f"[ERROR] Schedule aggregation failed: {e}")
                        ok = False

                results["fetchers"].append((label, ok))

            else:
                ok = run_script(script_path, label, str(context_path), additional_args=extra_args)
                results["fetchers"].append((label, ok))
    else:
        log(f"\n[SKIP] Skipping PHASE 0 & 1 (Settings/Fetchers) - starting at phase {start_phase}")

    # -------------------------------------------------------------------------
    # POST-FETCH: Combine NFL Offense + DST for ALL years
    # -------------------------------------------------------------------------
    if start_phase <= 1:
        log("\n" + "=" * 96)
        log("POST-FETCH: Combining NFL Offense + DST")
        log("=" * 96)

        # Combine for ALL NFL years (1999 - end_year), not just Yahoo years
        nfl_start_year = 1999
        end_year = ctx.end_year or datetime.now().year

        log(f"[INFO] Combining NFL offense + DST for years {nfl_start_year}-{end_year}")

        for y in range(nfl_start_year, end_year + 1):
            try:
                ok = run_script(
                    COMBINE_DST,
                    f"Combine NFL + DST ({y})",
                    str(context_path),
                    additional_args=["--year", str(y), "--week", "0"],
                    timeout=600  # 10 minutes per year
                )
                if ok:
                    log(f"[OK] Combined NFL + DST for {y}")
                else:
                    log(f"[WARN] Failed to combine NFL + DST for {y}")
            except Exception as e:
                log(f"[ERROR] Combine failed for {y}: {e}")

    # -------------------------------------------------------------------------
    # PHASE 2: WEEKLY MERGES -> player.parquet
    # -------------------------------------------------------------------------
    if start_phase <= 2:
        log("\n" + "=" * 96)
        log("PHASE 2: Weekly Merges -> player.parquet")
        log("=" * 96)

        all_weeks: List[pd.DataFrame] = []
        merge_failures: List[Tuple[int, int]] = []

        # Process ALL NFL years (1999+), not just Yahoo league years
        # For years before ctx.start_year, we'll load NFL-only data (no Yahoo manager/roster data)
        nfl_start_year = 1999
        yahoo_start_year = ctx.start_year
        end_year = ctx.end_year or datetime.now().year

        years_to_process = list(range(nfl_start_year, end_year + 1))
        log(f"[DEBUG] Processing years {nfl_start_year}-{end_year} (Yahoo league starts at {yahoo_start_year})")

        for year in years_to_process:
            # Determine end_week; default to 17 if unknown
            end_week = None
            try:
                ew_attr = getattr(ctx, "end_week", None)
                if isinstance(ew_attr, dict):
                    end_week = int(ew_attr.get(str(year)) or ew_attr.get(year) or 17)
                elif ew_attr:
                    end_week = int(ew_attr)
            except Exception:
                end_week = None
            if not end_week:
                end_week = 17

            player_data_dir = Path(ctx.player_data_directory)

            # For years before Yahoo league started, load NFL data only (no manager/roster data)
            if year < yahoo_start_year:
                log(f"\n[NFL-ONLY] {year} (before Yahoo league started)")
                # Try to load yearly NFL file - FIXED: Prioritize new "all_weeks" naming
                nfl_year_candidates = [
                    player_data_dir / f"nfl_stats_merged_{year}_all_weeks.parquet",  # NEW: combine_dst_to_nfl.py output
                    player_data_dir / f"player_stats_{year}_all_weeks.parquet",  # LEGACY: old naming
                    player_data_dir / f"player_stats_{year}_allweeks.parquet",  # LEGACY: even older naming
                ]

                nfl_df = None
                for nfl_year_file in nfl_year_candidates:
                    if nfl_year_file.exists():
                        try:
                            nfl_df = pd.read_parquet(nfl_year_file)

                            # CRITICAL: Filter to fantasy-relevant positions only
                            # Load roster settings to determine which positions are rosterable
                            if "nfl_position" in nfl_df.columns:
                                try:
                                    # Get rosterable positions from league settings
                                    settings_dir = Path(ctx.data_directory) / "league_settings"
                                    rosterable_positions = None

                                    if settings_dir.exists():
                                        # Try to find any settings file for this league (assume roster structure is consistent)
                                        normalized_league_id = ctx.league_id.replace(".", "_")
                                        settings_files = sorted(settings_dir.glob(f"*{normalized_league_id}.json"))

                                        if settings_files:
                                            with open(settings_files[0], 'r') as f:
                                                settings_data = json.load(f)

                                            roster_positions = settings_data.get("roster_positions", [])
                                            rosterable_positions = set()
                                            for pos_info in roster_positions:
                                                position = pos_info.get("position", "")
                                                # Exclude bench and IR
                                                if position and position not in ("BN", "IR"):
                                                    rosterable_positions.add(position.upper())

                                    # Fallback to standard fantasy positions
                                    if not rosterable_positions:
                                        rosterable_positions = {"QB", "RB", "WR", "TE", "K", "DEF"}
                                        log(f"      [filter] Using default fantasy positions: {sorted(rosterable_positions)}")
                                    else:
                                        log(f"      [filter] Using roster positions from league settings: {sorted(rosterable_positions)}")

                                    # Expand flex positions (e.g., "W/R/T" -> ["WR", "RB", "TE"])
                                    expanded_positions = set()
                                    for pos in rosterable_positions:
                                        if "/" in pos:
                                            expanded_positions.update(p.strip() for p in pos.split("/"))
                                        else:
                                            expanded_positions.add(pos)

                                    # Filter NFL data to only fantasy-relevant positions
                                    initial_count = len(nfl_df)
                                    nfl_pos = nfl_df["nfl_position"].astype(str).str.upper()
                                    keep_mask = nfl_df["nfl_position"].isna() | nfl_pos.isin(expanded_positions)
                                    nfl_df = nfl_df[keep_mask]

                                    filtered_count = initial_count - len(nfl_df)
                                    if filtered_count > 0:
                                        log(f"      [filter] Removed {filtered_count:,} non-fantasy players (kept positions: {sorted(expanded_positions)})")

                                except Exception as e:
                                    log(f"      [WARN] Position filtering failed: {e} - keeping all positions")

                            # Create unified 'position' column from nfl_position (for optimal lineup calculation)
                            if "position" not in nfl_df.columns and "nfl_position" in nfl_df.columns:
                                nfl_df["position"] = nfl_df["nfl_position"]
                                log(f"      [position] Created 'position' column from 'nfl_position'")

                            # Add empty Yahoo columns for consistency with merged data
                            for col in ["manager", "opponent", "yahoo_player_id", "yahoo_position", "lineup_position", "started"]:
                                if col not in nfl_df.columns:
                                    nfl_df[col] = pd.NA

                            # Ensure year column exists
                            if "year" not in nfl_df.columns:
                                nfl_df["year"] = year

                            all_weeks.append(nfl_df)
                            log(f"      [OK] Loaded NFL-only data: {nfl_year_file.name} ({len(nfl_df):,} rows after filtering)")
                            break
                        except Exception as e:
                            log(f"      [WARN] Failed to load NFL data from {nfl_year_file.name}: {e}")

                if nfl_df is None:
                    log(f"      [WARN] NFL data not found for {year}")
                    merge_failures.append((year, 0))
                continue

            # For Yahoo league years, run the full Yahoo + NFL merge (by year, not by week)
            log(f"\n[MERGE] {year} (all weeks)")

            # Call the modular yahoo_nfl_merge.py script which now keeps ALL NFL players
            # (position filtering has been disabled to include all NFL players)
            # The script handles:
            # - Fuzzy matching between Yahoo roster data and NFL player stats
            # - OUTER JOIN: ALL NFL players kept, with Yahoo data merged where available
            # - Proper 1:1 validation at each matching layer
            try:
                merge_ok = run_script(
                    YAHOO_NFL_MERGE,
                    f"Yahoo + NFL merge ({year})",
                    str(context_path),
                    additional_args=["--year", str(year), "--week", "0"],
                    timeout=1800  # 30 minutes for merge
                )

                if merge_ok:
                    # Load the merged output
                    merged_candidates = [
                        player_data_dir / f"yahoo_nfl_merged_{year}_all_weeks.parquet",
                        player_data_dir / f"yahoo_nfl_merged_{year}_week_0.parquet",
                    ]

                    merged_df = None
                    for mf in merged_candidates:
                        if mf.exists():
                            try:
                                merged_df = pd.read_parquet(mf)
                                if not merged_df.empty:
                                    log(f"      [MERGED] Loaded {mf.name} ({len(merged_df):,} rows)")

                                    # Ensure league_id is populated
                                    if "league_id" not in merged_df.columns:
                                        merged_df["league_id"] = ctx.league_id

                                    all_weeks.append(merged_df)
                                    results["merges"].append((f"Yahoo + NFL merge ({year})", True))
                                    break
                            except Exception as e:
                                log(f"      [WARN] Failed to load merged data from {mf.name}: {e}")

                    if merged_df is None:
                        log(f"      [WARN] Merge script succeeded but no output file found for {year}")
                        merge_failures.append((year, 0))
                        results["merges"].append((f"Yahoo + NFL merge ({year})", False))
                else:
                    log(f"      [WARN] Merge script failed for {year}")
                    merge_failures.append((year, 0))
                    results["merges"].append((f"Yahoo + NFL merge ({year})", False))

            except Exception as e:
                log(f"      [ERROR] Merge failed for {year}: {e}")
                merge_failures.append((year, 0))
                results["merges"].append((f"Yahoo + NFL merge ({year})", False))

        # Write unified player.parquet
        try:
            # --- Begin league-safe aggregation block ---
            from functools import reduce

            if all_weeks:
                log("\n[AGGREGATE] Concatenating weekly DataFrames -> player.parquet")

                # 1) Build unified set of columns so no columns vanish during concat
                try:
                    all_cols = sorted(reduce(lambda acc, df: acc.union(df.columns), all_weeks, set()))
                    all_weeks = [df.reindex(columns=all_cols) for df in all_weeks]
                except Exception:
                    # Fallback: best-effort concat if the above fails
                    log("      [WARN] Column-unification failed; falling back to naive concat")

                # 1.5) Standardize critical numeric columns to Int64 BEFORE concat
                for df in all_weeks:
                    if "year" in df.columns:
                        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
                    if "week" in df.columns:
                        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
                    # CRITICAL: Convert yahoo_player_id to string BEFORE concat to prevent 'nan' strings
                    if "yahoo_player_id" in df.columns:
                        # Convert numeric IDs to string properly (29235.0 → "29235", NaN → pd.NA)
                        df["yahoo_player_id"] = df["yahoo_player_id"].apply(
                            lambda x: str(int(x)) if pd.notna(x) else pd.NA
                        ).astype("string")

                player_df = pd.concat(all_weeks, ignore_index=True, sort=False)

                # 1.6) SAFE Deduplicate player_df after concat (safety net for duplicate weekly files)
                # CRITICAL: Only deduplicate if rows represent the SAME person
                # Example: Adrian Peterson (Bears) vs Adrian Peterson (Vikings) are DIFFERENT people - don't deduplicate
                # Use player IDs to distinguish different people with same name
                if "player" in player_df.columns and "year" in player_df.columns and "week" in player_df.columns:
                    initial_count = len(player_df)

                    # Group by (player, year, week) to find potential duplicates
                    dup_key_cols = ['player', 'year', 'week']
                    dup_groups = player_df.groupby(dup_key_cols)

                    rows_to_keep = []
                    rows_to_remove = []

                    for (player_name, year, week), group in dup_groups:
                        if len(group) == 1:
                            # No duplicates, keep as-is
                            rows_to_keep.append(group.index[0])
                        else:
                            # Multiple rows - check if they're the SAME person or DIFFERENT people
                            yahoo_ids = group['yahoo_player_id'].dropna().astype(str) if 'yahoo_player_id' in group.columns else pd.Series([], dtype=str)
                            nfl_ids = group['NFL_player_id'].dropna().astype(str) if 'NFL_player_id' in group.columns else pd.Series([], dtype=str)

                            # Check if there are CONFLICTING IDs (different people with same name)
                            unique_yahoo_ids = set(yahoo_ids.unique()) - {'nan', 'None', ''}
                            unique_nfl_ids = set(nfl_ids.unique()) - {'nan', 'None', ''}

                            if len(unique_yahoo_ids) > 1 or len(unique_nfl_ids) > 1:
                                # DIFFERENT PEOPLE (e.g., two Adrian Petersons)
                                # Keep all rows - they're different players
                                log(f"[DEDUP][WARN] Multiple players with same name: {player_name} (year {year}, week {week})")
                                log(f"  Yahoo IDs: {unique_yahoo_ids}, NFL IDs: {unique_nfl_ids}")
                                log(f"  Keeping all {len(group)} rows (different people)")
                                rows_to_keep.extend(group.index.tolist())
                            else:
                                # SAME PERSON - deduplicate
                                # Prefer: rostered > unrostered, then more complete data
                                group_sorted = group.copy()

                                if 'manager' in group_sorted.columns:
                                    group_sorted['_is_rostered'] = (
                                        group_sorted['manager'].notna() &
                                        (group_sorted['manager'].astype(str).str.strip() != 'Unrostered')
                                    )
                                else:
                                    group_sorted['_is_rostered'] = False

                                group_sorted['_non_null_count'] = group_sorted.notna().sum(axis=1)

                                group_sorted = group_sorted.sort_values(
                                    ['_is_rostered', '_non_null_count'],
                                    ascending=[False, False]
                                )

                                # Keep first (best quality row)
                                rows_to_keep.append(group_sorted.index[0])
                                rows_to_remove.extend(group_sorted.index[1:].tolist())

                    if rows_to_remove:
                        player_df = player_df.loc[rows_to_keep].copy()
                        removed = len(rows_to_remove)
                        log(f"[DEDUP] Removed {removed} duplicate rows (same person, kept best quality)")
                        log(f"[DEDUP] Final row count: {len(player_df):,}")
                    else:
                        log(f"[DEDUP] No duplicates found after concat")

                # 1.5) Preserve enrichment columns from existing player.parquet
                # These columns are added by transformation scripts and should not be lost
                enrichment_cols = [
                    # Draft enrichments (from draft_to_player_v2.py)
                    'round', 'pick', 'cost', 'is_keeper_status',
                    # Transaction enrichments (from transactions_to_player_v2.py)
                    'faab_bid', 'max_faab_bid_to_date', 'first_acquisition_week', 'total_acquisitions',
                ]

                try:
                    existing_player_file = ctx.canonical_player_file
                    if existing_player_file.exists():
                        log("[MERGE] Loading existing player.parquet to preserve enrichment columns...")
                        existing_player = pd.read_parquet(existing_player_file)

                        # CRITICAL: SAFELY Deduplicate existing_player BEFORE merge to prevent duplicate amplification
                        # Issue: If existing_player has duplicates, merging on join keys creates cartesian product
                        # Example: 1 new row + 2 existing duplicates = 2 output rows (duplicate amplification)
                        # SAFETY: Only deduplicate if rows represent SAME person (use IDs to distinguish)
                        if "yahoo_player_id" in existing_player.columns and "year" in existing_player.columns and "week" in existing_player.columns:
                            initial_count = len(existing_player)

                            # Group by (yahoo_player_id, year, week) - more specific than player name
                            dup_key_cols = ['yahoo_player_id', 'year', 'week']
                            dup_groups = existing_player.groupby(dup_key_cols)

                            rows_to_keep_indices = []
                            rows_to_remove_count = 0

                            for (yahoo_id, year, week), group in dup_groups:
                                if len(group) == 1:
                                    # No duplicates
                                    rows_to_keep_indices.append(group.index[0])
                                else:
                                    # Multiple rows with SAME yahoo_player_id - definitely same person
                                    # (or all None/NA - check NFL IDs as backup)
                                    if pd.isna(yahoo_id) or str(yahoo_id) in ['nan', 'None', '']:
                                        # No yahoo ID - check NFL IDs to see if same person
                                        nfl_ids = group['NFL_player_id'].dropna().astype(str) if 'NFL_player_id' in group.columns else pd.Series([], dtype=str)
                                        unique_nfl_ids = set(nfl_ids.unique()) - {'nan', 'None', ''}

                                        if len(unique_nfl_ids) > 1:
                                            # DIFFERENT PEOPLE (different NFL IDs)
                                            log(f"[MERGE][DEDUP][WARN] Multiple unrostered players at year {year} week {week} with different NFL IDs: {unique_nfl_ids}")
                                            log(f"  Keeping all {len(group)} rows (different people)")
                                            rows_to_keep_indices.extend(group.index.tolist())
                                            continue

                                    # SAME PERSON - deduplicate, prefer rows with enrichment data
                                    enrichment_check_cols = ['round', 'pick', 'cost', 'faab_bid', 'manager']
                                    group_sorted = group.copy()
                                    group_sorted['_enrichment_score'] = sum(
                                        group_sorted[col].notna().astype(int)
                                        for col in enrichment_check_cols if col in group_sorted.columns
                                    )

                                    group_sorted = group_sorted.sort_values('_enrichment_score', ascending=False)
                                    rows_to_keep_indices.append(group_sorted.index[0])
                                    rows_to_remove_count += len(group) - 1

                            if rows_to_remove_count > 0:
                                existing_player = existing_player.loc[rows_to_keep_indices].copy()
                                log(f"[MERGE][DEDUP] Removed {rows_to_remove_count} duplicates from existing player.parquet (same person, kept best enrichment)")
                            else:
                                log(f"[MERGE][DEDUP] No duplicates found in existing player.parquet")

                        # Check which enrichment columns exist and have data
                        cols_to_preserve = [c for c in enrichment_cols if c in existing_player.columns
                                           and existing_player[c].notna().sum() > 0]

                        if cols_to_preserve:
                            # Create join keys
                            join_keys = ['yahoo_player_id', 'year', 'week']
                            # Add cumulative_week if it exists in both
                            if 'cumulative_week' in existing_player.columns and 'cumulative_week' in player_df.columns:
                                join_keys.append('cumulative_week')

                            # Filter to only the keys + enrichment columns we want to preserve
                            enrichment_data = existing_player[join_keys + cols_to_preserve].copy()

                            # Merge enrichment columns back into new player_df
                            player_df = player_df.merge(
                                enrichment_data,
                                on=join_keys,
                                how='left',
                                suffixes=('', '_enrichment')
                            )

                            # If merge created duplicate columns, use enrichment value
                            for col in cols_to_preserve:
                                if f'{col}_enrichment' in player_df.columns:
                                    player_df[col] = player_df[f'{col}_enrichment'].fillna(player_df.get(col, pd.NA))
                                    player_df = player_df.drop(columns=[f'{col}_enrichment'])

                            preserved_count = sum(player_df[c].notna().sum() for c in cols_to_preserve)
                            log(f"[MERGE] Preserved {len(cols_to_preserve)} enrichment columns with {preserved_count:,} total values")
                            log(f"[MERGE] Preserved columns: {', '.join(cols_to_preserve)}")
                        else:
                            log("[MERGE] No enrichment columns found in existing player.parquet")
                except Exception as e:
                    log(f"[MERGE][WARN] Failed to preserve enrichment columns: {e}")

                # 2) Ensure league_id exists
                if "league_id" not in player_df.columns:
                    player_df["league_id"] = pd.NA

                # 3) Restore league_id for rostered rows if all values are NA
                # Use existing ctx (LeagueContext loaded earlier) when available
                try:
                    if "manager" in player_df.columns and player_df["league_id"].isna().all():
                        try:
                            # Prefer using the already-loaded ctx if present
                            league_id_val = getattr(ctx, "league_id", None)
                            if league_id_val:
                                player_df.loc[player_df["manager"].notna(), "league_id"] = league_id_val
                            else:
                                # As a fallback, try to load context from context_path
                                from multi_league.core.league_context import LeagueContext
                                ctx_temp = LeagueContext.load(context_path)
                                player_df.loc[player_df["manager"].notna(), "league_id"] = ctx_temp.league_id
                        except Exception as e:
                            log(f"[WARN] Could not restore league_id: {e}")
                except Exception:
                    # Defensive: if something about columns fails, continue
                    pass

                # Ensure league_id is stable string dtype
                try:
                    player_df["league_id"] = player_df["league_id"].astype("string")
                except Exception:
                    # If conversion fails, coerce via str then to string dtype
                    player_df["league_id"] = player_df["league_id"].astype(str).astype("string")

                # 4) Deduplicate using a league‑safe key
                dedupe_candidate_keys = ("league_id", "year", "week", "yahoo_player_id", "NFL_player_id", "player")
                dedupe_keys = [k for k in dedupe_candidate_keys if k in player_df.columns]
                if dedupe_keys and not player_df.empty:
                    player_df = player_df.drop_duplicates(subset=dedupe_keys, keep="last")

                # 5) Add unified 'position' column (required by player_stats_v2.py)
                # CRITICAL: yahoo_position = actual position from Yahoo (QB, RB, WR, etc.)
                #           nfl_position = actual position from NFLverse
                #           fantasy_position = roster SLOT from Yahoo (QB, RB1, FLEX, BN, IR, etc.)
                #           position = unified column (prefers yahoo_position, fallbacks to nfl_position)

                def is_valid_position_value(val):
                    """Check if position value is valid (not null, empty, 'nan', '0', etc.)"""
                    if pd.isna(val):
                        return False
                    val_str = str(val).strip().upper()
                    return val_str not in ("", "NAN", "NONE", "0", "NULL", "N/A")

                # Ensure position columns are strings before processing
                if "yahoo_position" in player_df.columns:
                    player_df["yahoo_position"] = player_df["yahoo_position"].astype("string")
                if "nfl_position" in player_df.columns:
                    player_df["nfl_position"] = player_df["nfl_position"].astype("string")

                # Create or update position column
                if "position" not in player_df.columns:
                    # Get yahoo_position and nfl_position
                    yahoo_pos = player_df["yahoo_position"] if "yahoo_position" in player_df.columns else pd.Series([pd.NA] * len(player_df), index=player_df.index)
                    nfl_pos = player_df["nfl_position"] if "nfl_position" in player_df.columns else pd.Series([pd.NA] * len(player_df), index=player_df.index)

                    # Prefer yahoo_position when valid, otherwise use nfl_position
                    player_df["position"] = yahoo_pos.where(yahoo_pos.apply(is_valid_position_value), nfl_pos)

                    # Clean up position column: convert invalid values to NA
                    player_df["position"] = player_df["position"].apply(lambda x: x if is_valid_position_value(x) else pd.NA)

                    log("[position] Added unified 'position' column (yahoo_position preferred, nfl_position fallback)")

                # Ensure position is always string type (prevent Polars type errors)
                player_df["position"] = player_df["position"].astype("string")

                # =========================================
                # CALCULATE POINTS FOR ALL PLAYERS
                # =========================================
                log("\n[POINTS] Calculating fantasy points for ALL players (including unrostered and pre-league)...")
                try:
                    from multi_league.transformations.modules.scoring_calculator import (
                        load_scoring_rules,
                        calculate_fantasy_points
                    )

                    # Load scoring rules from settings directory
                    settings_dir = Path(ctx.data_directory) / "league_settings"
                    if settings_dir.exists():
                        scoring_rules_by_year = load_scoring_rules(settings_dir)

                        if scoring_rules_by_year:
                            # Convert to polars for calculation
                            player_df_pl = pl.from_pandas(player_df)

                            # Calculate points for ALL players (no filtering by manager/roster)
                            # CRITICAL: Pass league_start_year from context so pre-league years use earliest rules
                            player_df_pl = calculate_fantasy_points(
                                player_df_pl,
                                scoring_rules_by_year,
                                year_col="year",
                                league_start_year=ctx.start_year  # From league_context.json
                            )

                            # Convert back to pandas
                            player_df = player_df_pl.to_pandas()

                            # Log statistics
                            total_rows = len(player_df)
                            has_points = (player_df['fantasy_points'] > 0).sum() if 'fantasy_points' in player_df.columns else 0
                            unrostered = player_df['manager'].isna().sum() if 'manager' in player_df.columns else 0
                            pre_league = (player_df['year'] < ctx.start_year).sum() if 'year' in player_df.columns else 0

                            log(f"[POINTS] Calculated points for {has_points:,}/{total_rows:,} players")
                            log(f"[POINTS]   Unrostered: {unrostered:,} players")
                            log(f"[POINTS]   Pre-{ctx.start_year}: {pre_league:,} players")
                        else:
                            log("[POINTS] No scoring rules found in settings directory")
                    else:
                        log(f"[POINTS] Settings directory not found: {settings_dir}")
                except ImportError as e:
                    log(f"[POINTS] scoring_calculator module not available: {e}")
                except Exception as e:
                    log(f"[POINTS] Error calculating points: {e}")

                # 6) Verify composite keys exist (should be created by yahoo_nfl_merge.py)
                # NOTE: These keys are now created in yahoo_nfl_merge.py, not here
                # This orchestrator should only validate, not create
                required_keys = ["cumulative_week", "manager_week"]
                missing_keys = [k for k in required_keys if k not in player_df.columns or player_df[k].isna().all()]
                if missing_keys:
                    log(f"[WARN] Missing composite keys from merge: {missing_keys}")
                    log(f"[WARN] These should be created by yahoo_nfl_merge.py")
                    # Defensive: create them if missing (but this indicates a problem with the merge script)
                    if "cumulative_week" in missing_keys and "year" in player_df.columns and "week" in player_df.columns:
                        player_df["cumulative_week"] = (
                            player_df["year"].fillna(-1).astype("Int64") * 100 +
                            player_df["week"].fillna(-1).astype("Int64")
                        ).astype("Int64")
                        log("[FALLBACK] Created cumulative_week (should come from merge)")
                    if "manager_week" in missing_keys and "manager" in player_df.columns and "cumulative_week" in player_df.columns:
                        mgr = player_df["manager"].astype("string")
                        cw_str = player_df["cumulative_week"].astype("Int64").astype("string")
                        player_df["manager_week"] = (
                            mgr.fillna("").str.replace(" ", "", regex=False) + cw_str.fillna("")
                        ).astype("string")
                        log("[FALLBACK] Created manager_week (should come from merge)")

                # 7) Ensure Year and Week are proper nullable integers
                if "year" in player_df.columns:
                    player_df["year"] = pd.to_numeric(player_df["year"], errors='coerce').astype("Int64")
                if "week" in player_df.columns:
                    player_df["week"] = pd.to_numeric(player_df["week"], errors='coerce').astype("Int64")

                # 8) Data type normalization: coerce stat columns to numeric to avoid mixed-type Parquet writes
                try:
                    player_df = normalize_numeric_columns(player_df)
                except Exception as e:
                    log(f"      [WARN] normalize_numeric_columns failed: {e}")

                # 8.5) Backfill missing headshot_url values based on yahoo_player_id
                if "headshot_url" in player_df.columns and "yahoo_player_id" in player_df.columns:
                    # Find players with missing headshot_url
                    missing_headshots = player_df["headshot_url"].isna() | (player_df["headshot_url"] == "")
                    missing_count_before = missing_headshots.sum()

                    if missing_count_before > 0:
                        # Create a mapping of yahoo_player_id to first non-null headshot_url
                        valid_headshots = player_df[player_df["headshot_url"].notna() & (player_df["headshot_url"] != "")]
                        headshot_mapping = valid_headshots.groupby("yahoo_player_id")["headshot_url"].first().to_dict()

                        # Fill missing headshot_url values using vectorized map operation
                        # Only update rows where headshot_url is missing/empty
                        mask = missing_headshots
                        player_df.loc[mask, "headshot_url"] = player_df.loc[mask, "yahoo_player_id"].map(headshot_mapping)

                        missing_count_after = (player_df["headshot_url"].isna() | (player_df["headshot_url"] == "")).sum()
                        filled_count = missing_count_before - missing_count_after

                        if filled_count > 0:
                            log(f"[headshot_url] Backfilled {filled_count:,} missing headshot URLs based on yahoo_player_id")

                # 9) Calculate fantasy points from league settings
                log("[POINTS] Calculating fantasy points from league settings...")
                try:
                    from calculate_fantasy_points import calculate_points_for_dataframe
                    settings_dir = Path(ctx.data_directory) / "league_settings"
                    if settings_dir.exists():
                        player_df = calculate_points_for_dataframe(player_df, settings_dir)
                        has_yahoo = player_df["yahoo_points"].notna().sum() if "yahoo_points" in player_df.columns else 0
                        has_calc = player_df["calculated_points"].notna().sum()
                        log(f"[POINTS] Yahoo points: {has_yahoo:,} rows, Calculated: {has_calc:,} rows")
                    else:
                        log(f"[POINTS] Settings directory not found: {settings_dir}, skipping calculation")
                except Exception as e:
                    log(f"[WARN] Points calculation failed: {e}")

                # 9.5) Cleanup: Drop any duplicate "_right" columns from joins
                right_cols = [c for c in player_df.columns if c.endswith('_right')]
                if right_cols:
                    log(f"[cleanup] Dropping {len(right_cols)} duplicate '_right' columns: {right_cols[:5]}{'...' if len(right_cols) > 5 else ''}")
                    player_df = player_df.drop(columns=right_cols)

                # 9.6) Filter out invalid player names
                if 'player' in player_df.columns:
                    before_count = len(player_df)
                    # Remove rows where player name is None, blank, 'None', or whitespace
                    player_df = player_df[
                        player_df['player'].notna() &
                        (player_df['player'].astype(str).str.strip() != '') &
                        (player_df['player'].astype(str).str.strip() != 'None')
                    ]
                    removed_count = before_count - len(player_df)
                    if removed_count > 0:
                        log(f"[cleanup] Removed {removed_count:,} rows with invalid player names")

                # 9.7) Reorder columns in logical order
                # Define column order groups (columns that exist will be placed in this order)
                column_order = [
                    # Key identification
                    'league_id', 'year', 'week', 'cumulative_week',
                    'player', 'yahoo_player_id', 'NFL_player_id', 'player_id',

                    # Context
                    'manager', 'manager_week', 'manager_year',
                    'opponent', 'team_abbr', 'recent_team',
                    'position', 'yahoo_position', 'nfl_position', 'fantasy_position', 'lineup_position',

                    # Status flags
                    'started', 'is_optimal', 'is_starter', 'is_bench',
                    'is_keeper_status', 'kept_next_year',

                    # Points (most important metrics)
                    'fantasy_points', 'calculated_points', 'yahoo_points', 'points',
                    'optimal_points', 'bench_points', 'points_above_bench',

                    # Game context
                    'game_id', 'game_date', 'home_team', 'away_team',

                    # Passing stats
                    'completions', 'attempts', 'passing_yards', 'passing_tds',
                    'interceptions', 'passing_2pt_conversions',

                    # Rushing stats
                    'carries', 'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions',

                    # Receiving stats
                    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                    'receiving_2pt_conversions',

                    # Other offensive
                    'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost',

                    # Defense stats
                    'def_interceptions', 'def_fumbles', 'def_sacks', 'def_safeties',
                    'def_tds', 'def_2pt_returns', 'def_points_allowed', 'def_yards_allowed',

                    # Rankings & percentiles
                    'player_personal_all_time_rank', 'player_personal_all_time_percentile',
                    'player_position_all_time_rank', 'player_position_all_time_percentile',
                    'manager_all_player_all_time_rank', 'manager_all_player_all_time_percentile',

                    # PPG metrics
                    'season_ppg', 'alltime_ppg', 'rolling_3_avg', 'rolling_5_avg',
                    'weighted_ppg', 'ppg_trend', 'consistency_score',

                    # Draft data
                    'draft_round', 'pick', 'cost', 'avg_pick', 'avg_cost',
                    'keeper_price', 'savings',

                    # Transaction data
                    'faab_bid', 'max_faab_bid_to_date', 'total_acquisitions',

                    # Advanced metrics
                    'air_yards_share', 'target_share', 'wopr', 'racr',
                    'snap_pct', 'route_participation',

                    # Image/metadata
                    'headshot_url', 'player_key', 'player_week', 'player_year',
                ]

                # Reorder: put defined columns first, then append any remaining columns
                existing_ordered = [col for col in column_order if col in player_df.columns]
                remaining_cols = [col for col in player_df.columns if col not in existing_ordered]
                final_column_order = existing_ordered + sorted(remaining_cols)

                player_df = player_df[final_column_order]
                log(f"[cleanup] Reordered {len(player_df.columns)} columns in logical order")

                # 9.5) Filter to only league-relevant positions (post-merge position filtering)
                # Keep ALL rows during merge for better matching, but filter final output
                # to only include positions that exist in this league's Yahoo roster data
                try:
                    if "yahoo_position" in player_df.columns and "position" in player_df.columns:
                        rows_before = len(player_df)

                        # Get set of valid positions from Yahoo roster data (non-null yahoo_position values)
                        yahoo_positions = set(
                            player_df["yahoo_position"]
                            .dropna()
                            .astype(str)
                            .str.upper()
                            .str.strip()
                            .unique()
                        )
                        yahoo_positions.discard("")  # Remove empty strings

                        if yahoo_positions:
                            # Normalize position column for comparison
                            position_normalized = (
                                player_df["position"]
                                .fillna("")
                                .astype(str)
                                .str.upper()
                                .str.strip()
                            )

                            # Keep rows where position matches a Yahoo position OR position is empty/null
                            # (empty positions might be added by transformations later)
                            keep_mask = (
                                position_normalized.isin(yahoo_positions) |
                                (position_normalized == "") |
                                player_df["position"].isna()
                            )

                            player_df = player_df[keep_mask].copy()
                            rows_after = len(player_df)
                            filtered_out = rows_before - rows_after

                            if filtered_out > 0:
                                log(f"[position_filter] Filtered out {filtered_out:,} rows with positions not in Yahoo league")
                                log(f"[position_filter] Yahoo league positions: {sorted(yahoo_positions)}")
                                log(f"[position_filter] Kept {rows_after:,} rows with league-relevant positions")
                        else:
                            log("[position_filter] No yahoo_position values found, skipping position filtering")
                    else:
                        log("[position_filter] Missing yahoo_position or position column, skipping filtering")
                except Exception as e:
                    log(f"[position_filter][WARN] Position filtering failed: {e}")

                # 10) Final write to canonical player file
                try:
                    # Final normalization pass to ensure all stat columns are numeric
                    log("[PRE-WRITE] Final data type normalization before parquet write...")

                    # CRITICAL: Fix cumulative_week to ensure it's Int64 and not string "nan"
                    # This prevents "Could not convert string 'nan' to DECIMAL" errors in MotherDuck
                    if "cumulative_week" in player_df.columns:
                        if player_df["cumulative_week"].dtype == "object":
                            log("[PRE-WRITE] Converting cumulative_week from object to Int64")
                            player_df["cumulative_week"] = player_df["cumulative_week"].replace("nan", pd.NA)
                            player_df["cumulative_week"] = pd.to_numeric(player_df["cumulative_week"], errors="coerce").astype("Int64")

                    # CRITICAL: Convert all defensive and stat columns from string to numeric
                    # This prevents "Could not convert '54' with type str: tried to convert to double" errors
                    stat_columns = [
                        # Defensive stats (these often have string values)
                        'def_interceptions', 'def_fumbles', 'def_sacks', 'def_safeties',
                        'def_tds', 'def_2pt_returns', 'def_points_allowed', 'def_yards_allowed',
                        'dst_points_allowed', 'dst_yards_allowed', 'dst_sacks', 'dst_interceptions',
                        'dst_fumbles', 'dst_safeties', 'dst_tds', 'dst_blocked_kicks', 'dst_ret_tds',
                        'points_allowed', 'pts_allow', 'pass_yds_allowed', 'passing_yds_allowed',
                        'rush_yds_allowed', 'rushing_yds_allowed',

                        # Points allowed buckets
                        'pts_allow_0', 'pts_allow_1_6', 'pts_allow_7_13', 'pts_allow_14_20',
                        'pts_allow_21_27', 'pts_allow_28_34', 'pts_allow_35_plus',

                        # Yards allowed buckets
                        'yds_allow_0_99', 'yds_allow_100_199', 'yds_allow_200_299',
                        'yds_allow_300_399', 'yds_allow_400_499', 'yds_allow_500_plus',

                        # Offensive stats
                        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
                        'carries', 'rushing_yards', 'rushing_tds',
                        'receptions', 'targets', 'receiving_yards', 'receiving_tds',

                        # All fantasy points columns
                        'fantasy_points', 'calculated_points', 'yahoo_points', 'points',
                        'optimal_points', 'bench_points',

                        # Transaction data
                        'faab_bid', 'max_faab_bid_to_date', 'cost', 'pick',
                    ]

                    converted_count = 0
                    for col in stat_columns:
                        if col in player_df.columns:
                            # Check if column needs conversion (not numeric type)
                            # Handle both 'object' and 'string' dtypes (pandas StringDtype)
                            if not pd.api.types.is_numeric_dtype(player_df[col]):
                                # Convert string values to numeric, coercing errors to NaN
                                player_df[col] = pd.to_numeric(player_df[col], errors='coerce')
                                converted_count += 1

                    if converted_count > 0:
                        log(f"[PRE-WRITE] Converted {converted_count} stat columns from string to numeric")

                    # Now run the general normalization
                    try:
                        player_df = normalize_numeric_columns(player_df)
                    except Exception as e:
                        log(f"      [WARN] Final normalization failed: {e}")

                    out_player = ctx.canonical_player_file
                    write_parquet_robust(player_df, out_player)
                    # Optional: also emit a small CSV sample for quick inspection
                    try:
                        (player_df.head(1000)).to_csv(out_player.with_suffix(".csv"), index=False)
                    except Exception:
                        # Non-fatal: CSV emission is optional
                        pass
                except Exception as e:
                    log(f"      [WARN] Failed to write player.parquet: {e}")

            else:
                # --- Fallback path retained as-is (no weekly DataFrames)
                log("[WARN] No weekly DataFrames collected from merge loop")
                # ...existing fallback code continues...
                log("[FALLBACK] Searching for existing yahoo_nfl_merged files...")
                player_data_dir = Path(ctx.player_data_directory)
                existing_merged = sorted(player_data_dir.glob("yahoo_nfl_merged_*_week_*.parquet"))
                if existing_merged:
                    log(f"[FALLBACK] Found {len(existing_merged)} existing merged files")
                    fallback_frames = []
                    for f in existing_merged:
                        try:
                            df = pd.read_parquet(f)
                            if not df.empty:
                                fallback_frames.append(df)
                                log(f"      [LOAD] {f.name} ({len(df):,} rows)")
                        except Exception as e:
                            log(f"      [SKIP] Failed to read {f.name}: {e}")

                    if fallback_frames:
                        player_df = pd.concat(fallback_frames, ignore_index=True, sort=False)
                        log(f"[FALLBACK] Loaded {len(player_df):,} total rows from existing files")

                        # Preserve enrichment columns from existing player.parquet (fallback path)
                        enrichment_cols = [
                            'round', 'pick', 'cost', 'is_keeper_status',
                            'faab_bid', 'max_faab_bid_to_date', 'first_acquisition_week', 'total_acquisitions',
                        ]

                        try:
                            existing_player_file = ctx.canonical_player_file
                            if existing_player_file.exists():
                                log("[FALLBACK] Loading existing player.parquet to preserve enrichment columns...")
                                existing_player = pd.read_parquet(existing_player_file)

                                cols_to_preserve = [c for c in enrichment_cols if c in existing_player.columns
                                                   and existing_player[c].notna().sum() > 0]

                                if cols_to_preserve:
                                    join_keys = ['yahoo_player_id', 'year', 'week']
                                    if 'cumulative_week' in existing_player.columns and 'cumulative_week' in player_df.columns:
                                        join_keys.append('cumulative_week')

                                    enrichment_data = existing_player[join_keys + cols_to_preserve].copy()

                                    player_df = player_df.merge(
                                        enrichment_data,
                                        on=join_keys,
                                        how='left',
                                        suffixes=('', '_enrichment')
                                    )

                                    for col in cols_to_preserve:
                                        if f'{col}_enrichment' in player_df.columns:
                                            player_df[col] = player_df[f'{col}_enrichment'].fillna(player_df.get(col, pd.NA))
                                            player_df = player_df.drop(columns=[f'{col}_enrichment'])

                                    preserved_count = sum(player_df[c].notna().sum() for c in cols_to_preserve)
                                    log(f"[FALLBACK] Preserved {len(cols_to_preserve)} enrichment columns with {preserved_count:,} total values")
                                else:
                                    log("[FALLBACK] No enrichment columns found in existing player.parquet")
                        except Exception as e:
                            log(f"[FALLBACK][WARN] Failed to preserve enrichment columns: {e}")

                        # Apply same normalization as above
                        dedupe_candidate_keys = ("league_id", "year", "week", "yahoo_player_id", "NFL_player_id", "player")
                        dedupe_keys = [k for k in dedupe_candidate_keys if k in player_df.columns]
                        if dedupe_keys:
                            player_df = player_df.drop_duplicates(subset=dedupe_keys, keep="last")

                        if "position" not in player_df.columns:
                            if "yahoo_position" in player_df.columns:
                                player_df["position"] = player_df["yahoo_position"]
                            elif "nfl_position" in player_df.columns:
                                player_df["position"] = player_df["nfl_position"]
                            else:
                                player_df["position"] = pd.NA

                            if "yahoo_position" in player_df.columns and "nfl_position" in player_df.columns:
                                player_df["position"] = player_df["position"].fillna(player_df["nfl_position"])
                            log("[position] Added unified 'position' column (fallback path)")

                        # NOTE: Composite keys should come from yahoo_nfl_merge.py
                        # This is fallback-only for old merged files that predate the composite key creation
                        if "cumulative_week" not in player_df.columns or player_df["cumulative_week"].isna().all():
                            if "year" in player_df.columns and "week" in player_df.columns:
                                player_df["cumulative_week"] = (
                                    player_df["year"].fillna(-1).astype("Int64") * 100 +
                                    player_df["week"].fillna(-1).astype("Int64")
                                ).astype("Int64")
                                log("[FALLBACK] Created cumulative_week from old merged files")
                            else:
                                player_df["cumulative_week"] = pd.NA

                        if "manager_week" not in player_df.columns or player_df["manager_week"].isna().all():
                            if "manager" in player_df.columns and "cumulative_week" in player_df.columns:
                                mgr = player_df["manager"].astype("string")
                                cw_str = player_df["cumulative_week"].astype("Int64").astype("string")
                                player_df["manager_week"] = (
                                    mgr.fillna("").str.replace(" ", "", regex=False) + cw_str.fillna("")
                                ).astype("string")
                                log("[FALLBACK] Created manager_week from old merged files")
                            else:
                                player_df["manager_week"] = pd.NA

                        if "year" in player_df.columns:
                            player_df["year"] = pd.to_numeric(player_df["year"], errors='coerce').astype("Int64")
                        if "week" in player_df.columns:
                            player_df["week"] = pd.to_numeric(player_df["week"], errors='coerce').astype("Int64")

                        try:
                            player_df = normalize_numeric_columns(player_df)
                        except Exception as e:
                            log(f"      [WARN] normalize_numeric_columns failed: {e}")

                        # Backfill missing headshot_url values based on yahoo_player_id (fallback path)
                        if "headshot_url" in player_df.columns and "yahoo_player_id" in player_df.columns:
                            missing_headshots = player_df["headshot_url"].isna() | (player_df["headshot_url"] == "")
                            missing_count_before = missing_headshots.sum()

                            if missing_count_before > 0:
                                valid_headshots = player_df[player_df["headshot_url"].notna() & (player_df["headshot_url"] != "")]
                                headshot_mapping = valid_headshots.groupby("yahoo_player_id")["headshot_url"].first().to_dict()

                                mask = missing_headshots
                                player_df.loc[mask, "headshot_url"] = player_df.loc[mask, "yahoo_player_id"].map(headshot_mapping)

                                missing_count_after = (player_df["headshot_url"].isna() | (player_df["headshot_url"] == "")).sum()
                                filled_count = missing_count_before - missing_count_after

                                if filled_count > 0:
                                    log(f"[headshot_url] Backfilled {filled_count:,} missing headshot URLs (fallback path)")

                        # Cleanup: Drop any duplicate "_right" columns from joins (fallback path)
                        right_cols = [c for c in player_df.columns if c.endswith('_right')]
                        if right_cols:
                            log(f"[cleanup] Dropping {len(right_cols)} duplicate '_right' columns (fallback path): {right_cols[:5]}{'...' if len(right_cols) > 5 else ''}")
                            player_df = player_df.drop(columns=right_cols)

                        # Filter out invalid player names (fallback path)
                        if 'player' in player_df.columns:
                            before_count = len(player_df)
                            player_df = player_df[
                                player_df['player'].notna() &
                                (player_df['player'].astype(str).str.strip() != '') &
                                (player_df['player'].astype(str).str.strip() != 'None')
                            ]
                            removed_count = before_count - len(player_df)
                            if removed_count > 0:
                                log(f"[cleanup] Removed {removed_count:,} rows with invalid player names (fallback path)")

                        # Reorder columns in logical order (fallback path)
                        column_order = [
                            'league_id', 'year', 'week', 'cumulative_week',
                            'player', 'yahoo_player_id', 'NFL_player_id', 'player_id',
                            'manager', 'manager_week', 'manager_year',
                            'opponent', 'team_abbr', 'recent_team',
                            'position', 'yahoo_position', 'nfl_position', 'fantasy_position', 'lineup_position',
                            'started', 'is_optimal', 'is_starter', 'is_bench',
                            'is_keeper_status', 'kept_next_year',
                            'fantasy_points', 'calculated_points', 'yahoo_points', 'points',
                            'optimal_points', 'bench_points', 'points_above_bench',
                            'game_id', 'game_date', 'home_team', 'away_team',
                            'completions', 'attempts', 'passing_yards', 'passing_tds',
                            'interceptions', 'passing_2pt_conversions',
                            'carries', 'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions',
                            'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                            'receiving_2pt_conversions',
                            'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost',
                            'def_interceptions', 'def_fumbles', 'def_sacks', 'def_safeties',
                            'def_tds', 'def_2pt_returns', 'def_points_allowed', 'def_yards_allowed',
                            'player_personal_all_time_rank', 'player_personal_all_time_percentile',
                            'player_position_all_time_rank', 'player_position_all_time_percentile',
                            'manager_all_player_all_time_rank', 'manager_all_player_all_time_percentile',
                            'season_ppg', 'alltime_ppg', 'rolling_3_avg', 'rolling_5_avg',
                            'weighted_ppg', 'ppg_trend', 'consistency_score',
                            'draft_round', 'pick', 'cost', 'avg_pick', 'avg_cost',
                            'keeper_price', 'savings',
                            'faab_bid', 'max_faab_bid_to_date', 'total_acquisitions',
                            'air_yards_share', 'target_share', 'wopr', 'racr',
                            'snap_pct', 'route_participation',
                            'headshot_url', 'player_key', 'player_week', 'player_year',
                        ]
                        existing_ordered = [col for col in column_order if col in player_df.columns]
                        remaining_cols = [col for col in player_df.columns if col not in existing_ordered]
                        final_column_order = existing_ordered + sorted(remaining_cols)
                        player_df = player_df[final_column_order]
                        log(f"[cleanup] Reordered {len(player_df.columns)} columns in logical order (fallback path)")

                        try:
                            out_player = Path(getattr(ctx, "player_data_directory", getattr(ctx, "data_directory", "."))) / "player.parquet"
                            write_parquet_robust(player_df, out_player)
                            # Optional: also emit a small CSV sample for quick inspection
                            try:
                                (player_df.head(1000)).to_csv(out_player.with_suffix(".csv"), index=False)
                            except Exception:
                                pass
                            log(f"[FALLBACK SUCCESS] Wrote player.parquet from existing merged files")
                        except Exception as e:
                            log(f"      [WARN] Failed to write player.parquet: {e}")
                    else:
                        log("[FALLBACK FAILED] No valid data in existing merged files")
                else:
                    log("[FALLBACK FAILED] No existing yahoo_nfl_merged files found")
            # --- End league-safe aggregation block ---
        except Exception as e:
            log(f"[ERROR] Aggregation failed: {e}")
    else:
        log(f"\n[SKIP] Skipping PHASE 2 (Merges) - starting at phase {start_phase}")


    # -------------------------------------------------------------------------
    # Pre-Transformation Setup: Create canonical matchup.parquet & draft.parquet
    # -------------------------------------------------------------------------
    log("\n" + "=" * 96)
    log("PRE-TRANSFORMATION: Creating canonical parquet files")
    log("=" * 96)

    # Create matchup.parquet BEFORE transformations (they need it)
    try:
        path = merge_matchups_to_parquet(str(context_path), years=args.years if args.years else None)
        log(f"[PRE] matchup.parquet ready: {path}")
    except Exception as e:
        log(f"[PRE][ERROR] matchups merge failed: {e}")

    # Normalize draft.parquet BEFORE transformations (combines year files and normalizes)
    try:
        path = normalize_draft_parquet(str(context_path))
        log(f"[PRE] draft.parquet ready: {path}")
    except Exception as e:
        log(f"[PRE][WARN] draft normalize failed: {e}")

    # Normalize transactions.parquet BEFORE transformations (player_to_transactions needs it)
    try:
        path = normalize_transactions_parquet(str(context_path))
        log(f"[PRE] transactions.parquet ready: {path}")
    except Exception as e:
        log(f"[PRE][WARN] transactions normalize failed: {e}")

    # Normalize schedule.parquet BEFORE transformations (playoff_odds needs it)
    try:
        path = normalize_schedule_parquet(str(context_path))
        log(f"[PRE] schedule.parquet ready: {path}")
    except Exception as e:
        log(f"[PRE][WARN] schedule normalize failed: {e}")

    # -------------------------------------------------------------------------
    # PHASE 3: Transformations / Cross-Imports / Aggregations
    # -------------------------------------------------------------------------
    if not args.skip_transformations:
        log("\n" + "=" * 96)
        log("PHASE 3: Transformations / Cross-Imports / Aggregations")
        log("=" * 96)

        # Run transformations in multiple passes to handle dependencies
        for pass_num, transformations in enumerate([TRANSFORMATIONS_PASS_1, TRANSFORMATIONS_PASS_2, TRANSFORMATIONS_PASS_3], start=1):
            log(f"\n[TRANSFORM] Running pass {pass_num} / 3 transformations")
            log(f"  [INFO] This pass contains {len(transformations)} transformation(s)")
            for transformation in transformations:
                # Unpack transformation tuple (supports 2 or 3 elements)
                if len(transformation) == 2:
                    script_path, label = transformation
                    custom_timeout = None
                elif len(transformation) == 3:
                    script_path, label, custom_timeout = transformation
                else:
                    log(f"[ERROR] Invalid transformation format: {transformation}")
                    continue

                # Reduce simulations for expected_record during initial imports (10x speedup)
                extra_args = ["--dry-run"] if args.dry_run else []
                if "expected_record_v2.py" in script_path and not args.dry_run:
                    extra_args.extend(["--n-sims", "10000"])
                    log(f"  [INFO] Reducing expected_record simulations to 10,000 for faster initial import")

                # Use custom timeout - can be a specific value, None (no timeout), or default to 1200
                # Only use default 1200 if not explicitly set in the tuple (i.e., when tuple length is 2)
                if custom_timeout is None:
                    log(f"  [INFO] No timeout limit for {label} - script can run indefinitely")
                    timeout = None
                elif custom_timeout != 900:  # 900 is the default in run_script, so only log if different
                    log(f"  [INFO] Using extended timeout for {label}: {custom_timeout} seconds ({custom_timeout//60} minutes)")
                    timeout = custom_timeout
                else:
                    timeout = custom_timeout

                ok = run_script(script_path, label, str(context_path), additional_args=extra_args, timeout=timeout)
                results.setdefault("transformations", []).append((label, ok))
    else:
        log("\n[SKIP] Skipping transformations (--skip-transformations)")

    # -------------------------------------------------------------------------
    # Post-fixers: points alias
    # -------------------------------------------------------------------------
    try:
        path = ensure_fantasy_points_alias(str(context_path))
        log(f"[POST] fantasy_points alias ensured in: {path}")
    except Exception as e:
        log(f"[POST][WARN] points alias step skipped: {e}")

    # -------------------------------------------------------------------------
    # Validation: Ensure league_id isolation
    # -------------------------------------------------------------------------
    log("\n" + "=" * 96)
    log("VALIDATION: League Isolation Check")
    log("=" * 96)

    validation_results = {}
    expected_league_id = ctx.league_id

    # Define files to validate - use canonical paths (all in broader directory, not subdirectories)
    files_to_validate = {
        "player.parquet": ctx.canonical_player_file,
        "matchup.parquet": ctx.canonical_matchup_file,
        "draft.parquet": ctx.canonical_draft_file,
        "transactions.parquet": ctx.canonical_transaction_file,
    }

    for file_name, file_path in files_to_validate.items():
        try:
            if file_path.exists():
                df = pd.read_parquet(file_path)
                is_valid = validate_league_isolation(df, expected_league_id, file_name, log=log)
                validation_results[file_name] = is_valid
            else:
                log(f"[VALIDATION SKIP] {file_name}: File not found")
                validation_results[file_name] = None
        except Exception as e:
            log(f"[VALIDATION ERROR] {file_name}: Failed to validate - {e}")
            validation_results[file_name] = False

    # Check if all validations passed
    failed_validations = [name for name, result in validation_results.items() if result is False]
    if failed_validations:
        log("\n" + "!" * 96)
        log("WARNING: Some files FAILED league isolation validation!")
        log("These files may contain data from multiple leagues or are missing league_id:")
        for name in failed_validations:
            log(f"  - {name}")
        log("!" * 96)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("\n" + "=" * 96)
    log("INITIAL IMPORT SUMMARY")
    log("=" * 96)
    def _summarize(section: str):
        items = results.get(section, [])
        succeeded = [n for n, ok in items if ok]
        failed = [n for n, ok in items if not ok]
        log(f"{section.upper()}: {len(succeeded)} succeeded, {len(failed)} failed")
        if failed:
            for f in failed:
                log(f"      FAILED: {f}")

    _summarize("fetchers")
    _summarize("merges")
    _summarize("transformations")

    if merge_failures:
        log("\nMerge failures (year, week):")
        for y, w in merge_failures:
            log(f"      - {y} W{w}")

    verify_unified_outputs(ctx)

    log("\nInitial import completed.")

if __name__ == "__main__":
    main()
