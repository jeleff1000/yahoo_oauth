#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WEEKLY IMPORT SCRIPT

Purpose: Run every Tuesday morning (after Monday Night Football) to update weekly data.
This should be automated via Windows Task Scheduler.

Usage:
    python weekly_import.py

What it does:
    1. Detects the latest completed NFL week (in NYC timezone)
    2. Fetches that week's matchup data
    3. Fetches that week's transaction data
    4. Fetches that week's player stats (Yahoo + NFL merged)
    5. Checks data completeness (prevents Thursday-only partial data)
    6. Upserts into canonical parquet files
    7. Runs post-processing scripts (cumulative stats, expected records, etc.)
    8. Uploads to MotherDuck
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import duckdb
from md.md_utils import df_from_md_or_parquet

# =============================================================================
# Paths
# =============================================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parent
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "fantasy_football_data"

# Producer scripts
MATCHUP_SCRIPT = ROOT_DIR / "matchup_scripts" / "weekly_matchup_data.py"
TRANSACTION_SCRIPT = ROOT_DIR / "transaction_scripts" / "transactions.py"
MERGE_SCRIPT = ROOT_DIR / "player_stats" / "yahoo_nfl_merge.py"

# Canonical targets
CANONICAL = {
    "matchup": DATA_DIR / "matchup.parquet",
    "transactions": DATA_DIR / "transactions.parquet",
    "player": DATA_DIR / "player.parquet",
}

# Source directories
SOURCE_DIRS = {
    "matchup": DATA_DIR / "matchup_data",
    "transactions": DATA_DIR / "transaction_data",
    "player": DATA_DIR / "player_data",
}

# Dedup keys
DEDUP_KEYS = {
    "matchup": ["manager", "opponent", "year", "week"],
    "transactions": ["transaction_key", "year", "week"],
    "player": ["yahoo_player_id", "nfl_player_id", "year", "week"],
}

# Column signatures (used to identify files)
SIGNATURES = {
    "matchup": ["manager", "opponent", "week", "year", "team_points", "opponent_points"],
    "transactions": ["transaction_key", "timestamp", "week", "year", "status"],
    "player": ["player", "yahoo_player_id", "nfl_player_id", "year", "week", "points"],
}

# Post-processing scripts (run AFTER canonical tables updated)
RUNS_POST = [
    (ROOT_DIR / "matchup_scripts" / "cumulative_stats.py", "Matchup cumulative stats"),
    (ROOT_DIR / "player_stats" / "cumulative_player_stats.py", "Player cumulative stats"),
    (ROOT_DIR / "matchup_scripts" / "expected_record_import.py", "Expected record calculation"),
    (ROOT_DIR / "matchup_scripts" / "opponent_expected_record.py", "Opponent expected record"),
    (ROOT_DIR / "matchup_scripts" / "playoff_odds_import.py", "Playoff odds calculation"),
    (ROOT_DIR / "player_stats" / "keeper_import.py", "Keeper analysis"),
    (ROOT_DIR / "player_stats" / "matchup_stats_import.py", "Matchup stats import"),
    (ROOT_DIR / "matchup_scripts" / "add_optimal.py", "Add optimal lineup"),
    (ROOT_DIR / "player_stats" / "aggregate_on_season.py", "Season aggregation"),
    (ROOT_DIR / "draft_scripts" / "ppg_draft_join.py", "Draft PPG join"),
    (DATA_DIR / "motherduck_upload.py", "MotherDuck upload"),
]

# =============================================================================
# Logging
# =============================================================================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# Week/season inference (latest COMPLETED week in NYC)
# =============================================================================
def _season_for_today(dt: datetime) -> int:
    return dt.year if dt.month >= 8 else dt.year - 1

def _first_thu_after_labor_day(season_year: int, tz) -> datetime:
    sept1 = datetime(season_year, 9, 1, tzinfo=tz)
    labor_mon = sept1 + timedelta(days=(7 - sept1.weekday()) % 7)
    first_thu = labor_mon + timedelta(days=((3 - labor_mon.weekday()) % 7))
    return first_thu

def infer_latest_completed_week(tz_name: str = "America/New_York") -> Tuple[int, int]:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone(timedelta(hours=-5))  # EST fallback

    now = datetime.now(tz)
    season = _season_for_today(now)
    cutoff = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    first_thu = _first_thu_after_labor_day(season, tz)
    week = 1 + max(0, (cutoff - first_thu).days // 7)
    week = max(1, min(22, week))
    return season, week

# =============================================================================
# Data completeness check
# =============================================================================
def is_week_complete(df: pd.DataFrame, year: int, week: int, kind: str) -> bool:
    """
    Check if a week's data appears complete (not just Thursday night football).
    Returns True if the week looks complete, False if it's partial.
    """
    if df.empty:
        return False

    # For player data: check if we have meaningful points data
    if kind == "player":
        if "points" not in df.columns:
            return True  # Can't verify, assume it's ok

        # Filter to the specific week
        week_data = df
        if "year" in df.columns and "week" in df.columns:
            week_data = df[
                (pd.to_numeric(df["year"], errors="coerce") == year) &
                (pd.to_numeric(df["week"], errors="coerce") == week)
            ]

        if week_data.empty:
            return False

        # Filter to only offensive players + kickers (not IDP who legitimately have 0 points)
        if "position" in week_data.columns:
            offensive_positions = ["QB", "RB", "WR", "TE", "K"]
            position_col = week_data["position"].astype(str).str.upper().str.strip()
            week_data = week_data[position_col.isin(offensive_positions)]

            if week_data.empty:
                log(f"  [Check] No offensive players found in week {week} data, cannot verify completion")
                return True  # Can't verify, assume it's ok

        # Count non-zero points among offensive players only
        points = pd.to_numeric(week_data["points"], errors="coerce").fillna(0)
        non_zero_count = (points > 0).sum()
        total_rows = len(week_data)

        # If less than 10% of OFFENSIVE players have points, it's likely incomplete
        if total_rows > 0:
            completion_rate = non_zero_count / total_rows
            log(f"  [Check] Week {week} completion rate: {completion_rate:.1%} ({non_zero_count}/{total_rows} offensive players with points)")

            if completion_rate < 0.10:
                log(f"  [WARN] Week {week} appears INCOMPLETE (only {completion_rate:.1%} offensive players have points)")
                return False

    # For matchup data: check if team_points look realistic
    elif kind == "matchup":
        if "team_points" not in df.columns:
            return True

        week_data = df
        if "year" in df.columns and "week" in df.columns:
            week_data = df[
                (pd.to_numeric(df["year"], errors="coerce") == year) &
                (pd.to_numeric(df["week"], errors="coerce") == week)
            ]

        if week_data.empty:
            return False

        # Check if team_points are non-zero
        points = pd.to_numeric(week_data["team_points"], errors="coerce").fillna(0)
        non_zero_count = (points > 0).sum()
        total_rows = len(week_data)

        if total_rows > 0:
            completion_rate = non_zero_count / total_rows
            log(f"  [Check] Week {week} matchup completion rate: {completion_rate:.1%} ({non_zero_count}/{total_rows} teams with points)")

            if completion_rate < 0.50:
                log(f"  [WARN] Week {week} matchups appear INCOMPLETE")
                return False

    log(f"  [OK] Week {week} appears COMPLETE")
    return True

# =============================================================================
# Script runners
# =============================================================================
def run_script(script: Path, label: str, year: int, week: int) -> None:
    """Run a producer script with specified year and week"""
    if not script.exists():
        log(f"SKIP (missing): {script}")
        return

    env = dict(os.environ)
    env["KMFFL_YEAR"] = str(year)
    env["KMFFL_WEEK"] = str(week)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, str(script), "--year", str(year), "--week", str(week)]
    log(f"RUN: {label} -> {script.name} --year {year} --week {week}")

    try:
        rc = subprocess.call(cmd, cwd=script.parent, env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")

def run_post_script(script: Path, label: str, year: int, week: int) -> None:
    """Run post-processing script"""
    if not script.exists():
        log(f"SKIP (missing): {label} -> {script}")
        return

    env = dict(os.environ)
    env["KMFFL_YEAR"] = str(year)
    env["KMFFL_WEEK"] = str(week)
    env["PYTHONUNBUFFERED"] = "1"

    # Some post-processing scripts take args, some don't
    scripts_with_args = {"ppg_draft_join.py", "motherduck_upload.py"}

    if script.name in scripts_with_args:
        cmd = [sys.executable, str(script), "--year", str(year), "--week", str(week)]
    else:
        cmd = [sys.executable, str(script)]

    log(f"RUN POST: {label} -> {script.name}")

    try:
        rc = subprocess.call(cmd, cwd=script.parent, env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")

# =============================================================================
# Parquet discovery
# =============================================================================
def locate_latest_parquet(kind: str, year: int, week: int) -> Optional[Path]:
    base = SOURCE_DIRS[kind]
    if not base.exists():
        return None

    preferred = {
        "matchup": [f"matchup_data_week_{week:02d}_year_{year}.parquet", "matchup.parquet"],
        "transactions": ["transactions.parquet"],
        "player": [
            f"yahoo_nfl_merged_{year}_week_{week}.parquet",
            f"yahoo_player_stats_{year}_week_{week}.parquet",
        ],
    }.get(kind, [])

    for name in preferred:
        p = base / name
        if p.exists():
            return p

    # Fallback: find by signature
    sig = SIGNATURES[kind]
    best = None
    best_score = (0, 0.0)

    for p in base.glob("*.parquet"):
        try:
            df = df_from_md_or_parquet(p.stem, p)
        except:
            continue

        cols = set(c.lower() for c in df.columns)
        if not any(c.lower() in cols for c in sig):
            continue

        # Filter to year/week
        if "year" in df.columns and "week" in df.columns:
            df = df[
                (pd.to_numeric(df["year"], errors="coerce") == year) &
                (pd.to_numeric(df["week"], errors="coerce") == week)
            ]

        rows = len(df)
        mtime = p.stat().st_mtime
        score = (rows, mtime)

        if score > best_score:
            best = p
            best_score = score

    return best

# =============================================================================
# DuckDB upsert
# =============================================================================
def upsert_parquet_via_duckdb(out_path: Path, new_df: pd.DataFrame, keys: list[str], kind: str) -> int:
    """Upsert new data into canonical parquet file using DuckDB"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Sanitize column names
    new_df = new_df.copy()
    new_df.columns = [str(c).strip() for c in new_df.columns]

    con.register("new_df", new_df)
    con.execute("CREATE TEMP TABLE _new AS SELECT * FROM new_df")

    out_str = str(out_path).replace("\\", "/")

    if out_path.exists():
        con.execute(f"CREATE TEMP TABLE _old AS SELECT * FROM read_parquet('{out_str}')")

        # Get all columns
        cols_new = [r[1] for r in con.execute("PRAGMA table_info('_new')").fetchall()]
        cols_old = [r[1] for r in con.execute("PRAGMA table_info('_old')").fetchall()]
        all_cols = list(dict.fromkeys(cols_old + cols_new))

        # Build select statements
        sel_old = ", ".join([f'"{c}"' if c in cols_old else f'NULL AS "{c}"' for c in all_cols])
        sel_new = ", ".join([f'"{c}"' if c in cols_new else f'NULL AS "{c}"' for c in all_cols])

        # Partition clause for deduplication
        partition_cols = [c for c in keys if c in all_cols]
        if not partition_cols:
            partition_cols = ["year", "week"] if "year" in all_cols and "week" in all_cols else all_cols[:1]

        partition_by = ", ".join([f'"{c}"' for c in partition_cols])

        # Merge with deduplication (new rows win)
        con.execute(f"""
            CREATE TEMP TABLE _merged AS
            SELECT *
            FROM (
                SELECT {sel_old}, 0 AS is_new FROM _old
                UNION ALL
                SELECT {sel_new}, 1 AS is_new FROM _new
            )
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_by} ORDER BY is_new DESC) = 1
        """)
    else:
        con.execute("CREATE TEMP TABLE _merged AS SELECT *, 1 AS is_new FROM _new")

    # Write to parquet
    con.execute(f"""
        COPY (SELECT * EXCLUDE(is_new) FROM _merged)
        TO '{out_str}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)

    total = con.execute("SELECT COUNT(*) FROM _merged").fetchone()[0]
    con.close()
    return int(total)

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Detect latest completed week
    year, week = infer_latest_completed_week()

    log("=" * 80)
    log(f"WEEKLY IMPORT: {year} Week {week}")
    log("=" * 80)
    log(f"Detected latest completed week: {year} Week {week}")
    log("")

    # ========================================================================
    # PHASE 1: Fetch weekly data
    # ========================================================================
    log("=== PHASE 1: Fetching weekly data ===")

    log(f"Fetching Week {week} matchup data...")
    run_script(MATCHUP_SCRIPT, "Weekly matchup data", year, week)

    log(f"Fetching Week {week} transaction data...")
    run_script(TRANSACTION_SCRIPT, "Transactions", year, week)

    log(f"Fetching Week {week} player stats...")
    run_script(MERGE_SCRIPT, "Yahoo/NFL merge", year, week)

    # ========================================================================
    # PHASE 2: Locate and validate data, then upsert
    # ========================================================================
    log("")
    log("=== PHASE 2: Validating and updating canonical files ===")

    for kind in ["matchup", "transactions", "player"]:
        src = locate_latest_parquet(kind, year, week)
        if not src or not src.exists():
            log(f"WARN: No {kind} source found for Week {week}")
            continue

        try:
            df_src = df_from_md_or_parquet(src.stem, src)
        except Exception as e:
            log(f"WARN: Could not read {kind} source {src}: {e}")
            continue

        # Filter to target week
        if kind in ("matchup", "player") and all(c in df_src.columns for c in ("year", "week")):
            df_src = df_src[
                (pd.to_numeric(df_src["year"], errors="coerce") == year) &
                (pd.to_numeric(df_src["week"], errors="coerce") == week)
            ]

        if df_src.empty:
            log(f"WARN: {kind} source is empty after filtering to Week {week}")
            continue

        # Check completeness for matchup and player data
        if kind in ("matchup", "player"):
            if not is_week_complete(df_src, year, week, kind):
                log(f"SKIP: {kind} data for Week {week} appears INCOMPLETE (likely only Thursday night)")
                log(f"      Will NOT overwrite existing complete data. Run again after Sunday games.")
                continue

        rows_used = len(df_src)
        log(f"Processing {kind}: {src.name} ({rows_used:,} rows)")

        # Upsert
        total_rows = upsert_parquet_via_duckdb(CANONICAL[kind], df_src, DEDUP_KEYS[kind], kind)
        log(f"✓ Updated {CANONICAL[kind].name}: {rows_used:,} new rows merged; total {total_rows:,}")

    # ========================================================================
    # PHASE 3: Run post-processing scripts
    # ========================================================================
    log("")
    log("=== PHASE 3: Post-processing and upload ===")

    for script, label in RUNS_POST:
        run_post_script(script, label, year, week)

    log("")
    log("=" * 80)
    log("WEEKLY IMPORT COMPLETED!")
    log("=" * 80)
    log(f"Successfully updated data for {year} Week {week}")

if __name__ == "__main__":
    main()
