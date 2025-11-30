#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WEEKLY UPDATE V2 - Incremental Data Pipeline

Designed to run at 4 AM every Tuesday to update league data after each NFL week.

What this does:
1) Detects the most recently completed NFL week
2) Fetches ONLY new data (current week matchups, player stats, transactions)
3) Appends new data to existing parquet files (with deduplication)
4) Runs transformations to update cumulative/career stats

Usage:
  python weekly_update_v2.py --context path/to/league_context.json
  python weekly_update_v2.py --context path/to/league_context.json --week 5  # Force specific week
  python weekly_update_v2.py --context path/to/league_context.json --dry-run

Schedule (cron example for 4 AM Tuesday):
  0 4 * * 2 cd /path/to/scripts && python weekly_update_v2.py --context /path/to/league_context.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import os

import pandas as pd

# Import from initial_import_v2 for shared utilities
from multi_league.core.data_normalization import (
    normalize_numeric_columns,
    write_parquet_robust,
    add_composite_keys,
    ensure_league_id
)
from multi_league.core.script_runner import log, run_script, setup_oauth_environment
from multi_league.core.league_context import LeagueContext
from multi_league.data_fetchers.aggregators import (
    merge_matchups_to_parquet,
    normalize_transactions_parquet,
    ensure_fantasy_points_alias
)

# Script directory
SCRIPT_DIR = Path(__file__).parent

# Transformations to run after data update (subset of full pipeline)
# Only include transformations that update cumulative/career stats
WEEKLY_TRANSFORMATIONS = [
    # Pass 1: Base calculations
    ("multi_league/transformations/base/cumulative_stats_v2.py", "Cumulative Stats", 600),
    ("multi_league/transformations/base/enrich_schedule_with_playoff_flags.py", "Enrich Schedule", 120),

    # Pass 2: Player stats updates
    ("multi_league/transformations/player_enrichment/matchup_to_player_v2.py", "Matchup -> Player", 600),
    ("multi_league/transformations/player_enrichment/player_stats_v2.py", "Player Stats", 900),
    ("multi_league/transformations/player_enrichment/replacement_level_v2.py", "Replacement Levels", 600),

    # Pass 3: Cross-table updates
    ("multi_league/transformations/matchup_enrichment/player_to_matchup_v2.py", "Player -> Matchup", 600),
    ("multi_league/transformations/transaction_enrichment/fix_unknown_managers.py", "Fix Unknown Managers", 120),
    ("multi_league/transformations/transaction_enrichment/player_to_transactions_v2.py", "Player <-> Transactions", 600),
    ("multi_league/transformations/transaction_enrichment/transaction_value_metrics_v3.py", "Transaction Metrics", 600),
    ("multi_league/transformations/player_enrichment/transactions_to_player_v2.py", "Transactions -> Player", 600),
    ("multi_league/transformations/matchup_enrichment/expected_record_v2.py", "Expected Record", 900),
    ("multi_league/transformations/matchup_enrichment/playoff_odds_import.py", "Playoff Odds", 1800),
    ("multi_league/transformations/aggregation/aggregate_player_season_v2.py", "Aggregate Player Season", 600),
    ("multi_league/transformations/finalize/normalize_canonical_types.py", "Normalize Types", 120),
]


def get_current_week_from_yahoo(ctx: LeagueContext, year: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Get current week and last completed week from Yahoo API.

    Uses the same logic as weekly_matchup_data_v2.py for consistency.

    Returns:
        Tuple of (current_week, last_completed_week)
        current_week: The week Yahoo says we're in
        last_completed_week: current_week - 1 (the week whose games are finished)
    """
    try:
        from yahoo_oauth import OAuth2
        from yfpy.query import YahooFantasySportsQuery

        oauth_path = ctx.oauth_file_path or os.environ.get("OAUTH_PATH")
        if not oauth_path or not Path(oauth_path).exists():
            log(f"[WARN] OAuth file not found: {oauth_path}")
            return None, None

        # Get league key for this year
        league_key = ctx.get_league_id_for_year(year) or ctx.league_id
        game_code = ctx.game_code or "nfl"

        log(f"[INFO] Connecting to Yahoo API for league {league_key}...")

        # Use yfpy to get league object (same as weekly_matchup_data_v2.py)
        query = YahooFantasySportsQuery(
            league_id=league_key.split(".l.")[-1] if ".l." in league_key else league_key,
            game_code=game_code,
            game_id=league_key.split(".")[0] if "." in league_key else None,
            yahoo_consumer_key=None,
            yahoo_consumer_secret=None,
            env_file_location=Path(oauth_path).parent,
            save_token_data_to_env_file=False
        )

        # Get current week from league object
        league = query.get_league_info()
        current_week = None

        if hasattr(league, 'current_week'):
            cw = league.current_week
            current_week = int(cw() if callable(cw) else cw)

        if current_week:
            # Last completed week is current - 1 (games still in progress this week)
            last_completed = max(0, current_week - 1)
            log(f"[INFO] Yahoo current_week: {current_week}, last completed: {last_completed}")
            return current_week, last_completed
        else:
            log("[WARN] Could not get current_week from Yahoo league object")
            return None, None

    except ImportError as e:
        log(f"[WARN] yfpy not available, falling back to direct API: {e}")
        return _get_current_week_direct_api(ctx, year)
    except Exception as e:
        log(f"[WARN] Could not get current week from Yahoo: {e}")
        return _get_current_week_direct_api(ctx, year)


def _get_current_week_direct_api(ctx: LeagueContext, year: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Fallback: Get current week directly from Yahoo API without yfpy.
    """
    try:
        from yahoo_oauth import OAuth2

        oauth_path = ctx.oauth_file_path or os.environ.get("OAUTH_PATH")
        if not oauth_path or not Path(oauth_path).exists():
            return None, None

        oauth = OAuth2(None, None, from_file=oauth_path)
        if not oauth.token_is_valid():
            oauth.refresh_access_token()

        league_key = ctx.get_league_id_for_year(year) or ctx.league_id
        url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/metadata"
        response = oauth.session.get(url, params={"format": "json"})

        if response.status_code == 200:
            data = response.json()
            league_data = data.get("fantasy_content", {}).get("league", [{}])[0]
            current_week = league_data.get("current_week")
            if current_week:
                cw = int(current_week)
                return cw, max(0, cw - 1)
    except Exception as e:
        log(f"[WARN] Direct API call failed: {e}")

    return None, None


def estimate_current_week_from_date() -> Tuple[int, int]:
    """
    Fallback: Estimate current NFL week based on date.
    Only used if Yahoo API is unavailable.

    Returns:
        Tuple of (current_week, last_completed_week)
    """
    today = datetime.now()
    year = today.year

    # NFL season typically starts first Thursday after Labor Day
    sept_1 = datetime(year, 9, 1)
    days_to_monday = (7 - sept_1.weekday()) % 7
    if sept_1.weekday() == 0:
        days_to_monday = 0
    labor_day = sept_1 + timedelta(days=days_to_monday)
    season_start = labor_day + timedelta(days=3)

    if today < season_start:
        return 0, 0

    days_since_start = (today - season_start).days
    current_week = min((days_since_start // 7) + 1, 18)

    # On Tuesday+, previous week is complete
    # On Sunday/Monday, games still in progress
    if today.weekday() >= 1:  # Tuesday-Saturday
        last_completed = current_week
    else:
        last_completed = max(0, current_week - 1)

    return current_week, last_completed


def get_last_fetched_week(ctx: LeagueContext, year: int) -> int:
    """
    Determine the last week we have data for.
    """
    matchup_file = ctx.canonical_matchup_file
    if not matchup_file.exists():
        return 0

    try:
        df = pd.read_parquet(matchup_file)
        df_year = df[df["year"] == year]
        if df_year.empty:
            return 0
        return int(df_year["week"].max())
    except Exception as e:
        log(f"[WARN] Could not read matchup file: {e}")
        return 0


def fetch_week_data(
    ctx: LeagueContext,
    year: int,
    week: int,
    dry_run: bool = False
) -> Dict[str, bool]:
    """
    Fetch data for a specific week.
    Returns dict of {script_name: success} results.
    """
    results = {}
    context_path = str(ctx._source_path) if hasattr(ctx, '_source_path') else None

    if not context_path:
        # Find context file
        for candidate in [
            Path(ctx.data_directory) / "league_context.json",
            Path(ctx.data_directory).parent / "league_context.json",
        ]:
            if candidate.exists():
                context_path = str(candidate)
                break

    if not context_path:
        log("[ERROR] Cannot find league_context.json")
        return {"error": False}

    setup_oauth_environment(ctx)

    # 1. Fetch matchup data for this week
    log(f"\n[FETCH] Matchup data for week {week}...")
    if not dry_run:
        ok, _ = run_script(
            SCRIPT_DIR / "multi_league/data_fetchers/weekly_matchup_data_v2.py",
            "Matchup data",
            context_path,
            additional_args=["--year", str(year), "--week", str(week)],
            timeout=300
        )
        results["matchup"] = ok
    else:
        log("  [DRY-RUN] Would fetch matchup data")
        results["matchup"] = True

    # 2. Fetch player data for this week
    log(f"\n[FETCH] Player data for week {week}...")
    if not dry_run:
        ok, _ = run_script(
            SCRIPT_DIR / "multi_league/data_fetchers/yahoo_fantasy_data.py",
            "Yahoo player data",
            context_path,
            additional_args=["--year", str(year), "--week", str(week)],
            timeout=600
        )
        results["player"] = ok
    else:
        log("  [DRY-RUN] Would fetch player data")
        results["player"] = True

    # 3. Fetch NFL offense stats for this week
    log(f"\n[FETCH] NFL offense stats for week {week}...")
    if not dry_run:
        ok, _ = run_script(
            SCRIPT_DIR / "multi_league/data_fetchers/nfl_offense_stats.py",
            "NFL offense stats",
            context_path,
            additional_args=["--year", str(year), "--week", str(week)],
            timeout=300
        )
        results["nfl_offense"] = ok
    else:
        log("  [DRY-RUN] Would fetch NFL offense stats")
        results["nfl_offense"] = True

    # 4. Fetch defense stats for this week
    log(f"\n[FETCH] Defense stats for week {week}...")
    if not dry_run:
        ok, _ = run_script(
            SCRIPT_DIR / "multi_league/data_fetchers/defense_stats.py",
            "Defense stats",
            context_path,
            additional_args=["--year", str(year), "--week", str(week)],
            timeout=300
        )
        results["defense"] = ok
    else:
        log("  [DRY-RUN] Would fetch defense stats")
        results["defense"] = True

    # 5. Fetch transactions (all new ones since last fetch)
    log(f"\n[FETCH] Transactions...")
    if not dry_run:
        ok, _ = run_script(
            SCRIPT_DIR / "multi_league/data_fetchers/transactions_v2.py",
            "Transactions",
            context_path,
            additional_args=["--year", str(year)],
            timeout=300
        )
        results["transactions"] = ok
    else:
        log("  [DRY-RUN] Would fetch transactions")
        results["transactions"] = True

    return results


def merge_new_data(
    ctx: LeagueContext,
    year: int,
    week: int,
    dry_run: bool = False
) -> bool:
    """
    Merge newly fetched data into canonical parquet files.
    Handles deduplication if week data already exists.
    """
    log("\n[MERGE] Combining data into canonical files...")

    context_path = str(ctx._source_path) if hasattr(ctx, '_source_path') else None

    if not context_path:
        for candidate in [
            Path(ctx.data_directory) / "league_context.json",
            Path(ctx.data_directory).parent / "league_context.json",
        ]:
            if candidate.exists():
                context_path = str(candidate)
                break

    if dry_run:
        log("  [DRY-RUN] Would merge data")
        return True

    try:
        # 1. Merge matchups
        log("  [MERGE] Matchup data...")
        merge_matchups_to_parquet(context_path, years=[year], log=log)

        # 2. Merge Yahoo + NFL player data
        log("  [MERGE] Player data (Yahoo + NFL)...")
        ok, _ = run_script(
            SCRIPT_DIR / "multi_league/data_fetchers/yahoo_nfl_merge.py",
            "Yahoo + NFL merge",
            context_path,
            additional_args=["--year", str(year), "--week", str(week)],
            timeout=600
        )

        if ok:
            # Combine DST data
            ok2, _ = run_script(
                SCRIPT_DIR / "multi_league/data_fetchers/combine_dst_to_nfl.py",
                "Combine DST",
                context_path,
                additional_args=["--year", str(year), "--week", str(week)],
                timeout=300
            )

        # 3. Normalize transactions
        log("  [MERGE] Transaction data...")
        normalize_transactions_parquet(context_path, log=log)

        return True

    except Exception as e:
        log(f"  [ERROR] Merge failed: {e}")
        return False


def run_transformations(
    ctx: LeagueContext,
    dry_run: bool = False
) -> Dict[str, bool]:
    """
    Run all transformations to update cumulative stats.
    """
    results = {}

    context_path = str(ctx._source_path) if hasattr(ctx, '_source_path') else None

    if not context_path:
        for candidate in [
            Path(ctx.data_directory) / "league_context.json",
            Path(ctx.data_directory).parent / "league_context.json",
        ]:
            if candidate.exists():
                context_path = str(candidate)
                break

    if not context_path:
        log("[ERROR] Cannot find league_context.json for transformations")
        return {"error": False}

    log("\n" + "=" * 80)
    log("RUNNING TRANSFORMATIONS")
    log("=" * 80)

    setup_oauth_environment(ctx)

    for script_path, label, timeout in WEEKLY_TRANSFORMATIONS:
        log(f"\n[TRANSFORM] {label}...")

        if dry_run:
            log(f"  [DRY-RUN] Would run {script_path}")
            results[label] = True
            continue

        ok, _ = run_script(
            SCRIPT_DIR / script_path,
            label,
            context_path,
            timeout=timeout
        )
        results[label] = ok

        if not ok:
            log(f"  [WARN] {label} failed, continuing...")

    return results


def verify_update(ctx: LeagueContext, year: int, week: int) -> Dict:
    """
    Verify the update was successful by checking data.
    """
    summary = {}

    # Check matchup data
    matchup_file = ctx.canonical_matchup_file
    if matchup_file.exists():
        df = pd.read_parquet(matchup_file)
        df_week = df[(df["year"] == year) & (df["week"] == week)]
        summary["matchup"] = {
            "total_rows": len(df),
            "week_rows": len(df_week),
            "max_week": int(df[df["year"] == year]["week"].max()) if not df[df["year"] == year].empty else 0
        }

    # Check player data
    player_file = ctx.canonical_player_file
    if player_file.exists():
        df = pd.read_parquet(player_file)
        df_week = df[(df["year"] == year) & (df["week"] == week)]
        summary["player"] = {
            "total_rows": len(df),
            "week_rows": len(df_week),
            "max_week": int(df[df["year"] == year]["week"].max()) if not df[df["year"] == year].empty else 0
        }

    # Check transactions
    trans_file = ctx.canonical_transaction_file
    if trans_file.exists():
        df = pd.read_parquet(trans_file)
        df_year = df[df["year"] == year]
        summary["transactions"] = {
            "total_rows": len(df),
            "year_rows": len(df_year)
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Weekly Update V2 - Incremental data pipeline")
    parser.add_argument("--context", required=True, help="Path to league_context.json")
    parser.add_argument("--week", type=int, default=None, help="Force specific week (default: auto-detect)")
    parser.add_argument("--year", type=int, default=None, help="Force specific year (default: current)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching, only run transforms")
    parser.add_argument("--skip-transforms", action="store_true", help="Skip transformations, only fetch data")

    args = parser.parse_args()

    # Load context
    ctx = LeagueContext.load(args.context)
    ctx._source_path = Path(args.context)

    # Determine year
    year = args.year or datetime.now().year

    # Determine week to fetch
    if args.week:
        target_week = args.week
        log(f"[INFO] Using specified week: {target_week}")
    else:
        # Try Yahoo API first (uses same logic as weekly_matchup_data_v2.py)
        current_week, last_completed = get_current_week_from_yahoo(ctx, year)

        if last_completed is not None:
            target_week = last_completed
            log(f"[INFO] Yahoo API: current_week={current_week}, last completed={last_completed}")
        else:
            # Fall back to date-based estimation
            current_week, target_week = estimate_current_week_from_date()
            log(f"[INFO] Date-based estimate: current_week={current_week}, targeting={target_week}")

    if target_week == 0:
        log("[INFO] NFL season hasn't started yet. Nothing to update.")
        return 0

    # Check what we already have
    last_week = get_last_fetched_week(ctx, year)
    log(f"[INFO] Last fetched week: {last_week}")

    if last_week >= target_week and not args.week:
        log(f"[INFO] Already have data through week {last_week}. Nothing new to fetch.")
        # Still run transforms in case something was updated
        if not args.skip_transforms:
            run_transformations(ctx, dry_run=args.dry_run)
        return 0

    # Determine weeks to fetch
    weeks_to_fetch = list(range(last_week + 1, target_week + 1)) if not args.week else [target_week]

    log("\n" + "=" * 80)
    log(f"WEEKLY UPDATE V2 - {ctx.league_name}")
    log("=" * 80)
    log(f"Year:           {year}")
    log(f"Weeks to fetch: {weeks_to_fetch}")
    log(f"Dry run:        {args.dry_run}")
    log(f"Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    all_results = {"fetch": {}, "merge": {}, "transform": {}}

    # Fetch data for each week
    if not args.skip_fetch:
        for week in weeks_to_fetch:
            log(f"\n{'=' * 40}")
            log(f"FETCHING WEEK {week}")
            log(f"{'=' * 40}")

            fetch_results = fetch_week_data(ctx, year, week, dry_run=args.dry_run)
            all_results["fetch"][week] = fetch_results

            # Merge after each week
            merge_ok = merge_new_data(ctx, year, week, dry_run=args.dry_run)
            all_results["merge"][week] = merge_ok

    # Run transformations
    if not args.skip_transforms:
        transform_results = run_transformations(ctx, dry_run=args.dry_run)
        all_results["transform"] = transform_results

    # Verify and summarize
    log("\n" + "=" * 80)
    log("UPDATE SUMMARY")
    log("=" * 80)

    if not args.dry_run:
        verification = verify_update(ctx, year, target_week)
        log(f"Verification: {json.dumps(verification, indent=2)}")

    # Count successes/failures
    fetch_success = sum(1 for w in all_results["fetch"].values() for v in w.values() if v)
    fetch_total = sum(len(w) for w in all_results["fetch"].values())
    transform_success = sum(1 for v in all_results["transform"].values() if v)
    transform_total = len(all_results["transform"])

    log(f"\nFetch:      {fetch_success}/{fetch_total} succeeded")
    log(f"Transform:  {transform_success}/{transform_total} succeeded")

    if fetch_success == fetch_total and transform_success == transform_total:
        log("\n[SUCCESS] Weekly update completed successfully!")
        return 0
    else:
        log("\n[WARN] Weekly update completed with some failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())
