#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRESEASON IMPORT SCRIPT

Purpose: Run AFTER draft but BEFORE Week 1 to capture preseason data.
Run this once per season after your draft completes.

Usage:
    python preseason_import.py --year 2025

What it does:
    1. Fetches draft data for the specified year
    2. Fetches Week 0 rosters (post-draft, pre-Week 1)
    3. Fetches season schedule
    4. Runs keeper analysis (if applicable)
    5. Updates canonical files
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import duckdb

# =============================================================================
# Paths
# =============================================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parent
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "fantasy_football_data"

# Producer scripts
SCHEDULE_SCRIPT = ROOT_DIR / "schedule_script" / "season_schedules.py"
DRAFT_SCRIPT = ROOT_DIR / "draft_scripts" / "draft_data.py"
MERGE_SCRIPT = ROOT_DIR / "player_stats" / "yahoo_nfl_merge.py"

# Post-processing scripts
KEEPER_SCRIPT = ROOT_DIR / "player_stats" / "keeper_import.py"

# Canonical targets
CANONICAL = {
    "schedule": DATA_DIR / "schedule.parquet",
    "player": DATA_DIR / "player.parquet",
}

# Source directories
SOURCE_DIRS = {
    "schedule": DATA_DIR / "schedule_data",
    "player": DATA_DIR / "player_data",
    "draft": DATA_DIR / "draft_data",
}

# Dedup keys
DEDUP_KEYS = {
    "schedule": ["year", "week", "manager", "opponent"],
    "player": ["yahoo_player_id", "nfl_player_id", "year", "week"],
}

# =============================================================================
# Logging
# =============================================================================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# Script runners
# =============================================================================
def run_script(script: Path, label: str, year: int, week: int = 0) -> None:
    """Run a script with specified year and week"""
    if not script.exists():
        log(f"SKIP (missing): {script}")
        return

    env = dict(os.environ)
    env["KMFFL_YEAR"] = str(year)
    env["KMFFL_WEEK"] = str(week)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, str(script), "--year", str(year)]
    if week is not None:
        cmd.extend(["--week", str(week)])

    log(f"RUN: {label} -> {script.name} --year {year} --week {week}")

    try:
        rc = subprocess.call(cmd, cwd=script.parent, env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")

def run_post_script(script: Path, label: str) -> None:
    """Run post-processing script (no args)"""
    if not script.exists():
        log(f"SKIP (missing): {label} -> {script}")
        return

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, str(script)]
    log(f"RUN POST: {label} -> {script.name}")

    try:
        rc = subprocess.call(cmd, cwd=script.parent, env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")

# =============================================================================
# Parquet helpers
# =============================================================================
def locate_latest_parquet(kind: str, year: int) -> Optional[Path]:
    """Find the most recent parquet file for a given kind and year"""
    base = SOURCE_DIRS[kind]
    if not base.exists():
        return None

    # Preferred filenames
    if kind == "schedule":
        candidates = [
            base / f"schedule_data_year_{year}.parquet",
            base / "schedule.parquet",
        ]
    elif kind == "player":
        candidates = [
            base / f"yahoo_nfl_merged_{year}_week_0.parquet",
            base / f"yahoo_player_stats_{year}_week_0.parquet",
        ]
    elif kind == "draft":
        candidates = [
            base / f"draft_data_{year}.csv",
            base / "draft_data.csv",
        ]
    else:
        candidates = []

    for p in candidates:
        if p.exists():
            return p

    # Fallback: find any matching file
    parquets = list(base.glob(f"*{year}*.parquet"))
    if parquets:
        return max(parquets, key=lambda p: p.stat().st_mtime)

    return None

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
    parser = argparse.ArgumentParser(description="Preseason data import (post-draft, pre-Week 1)")
    parser.add_argument('--year', type=int, required=True, help='Season year (e.g., 2025)')
    args = parser.parse_args()

    year = args.year

    log("=" * 80)
    log(f"PRESEASON IMPORT: Year {year}")
    log("=" * 80)
    log("")
    log("This will fetch:")
    log(f"  - Draft data for {year}")
    log(f"  - Week 0 rosters (post-draft)")
    log(f"  - Season {year} schedule")
    log("")

    # ========================================================================
    # PHASE 1: Fetch preseason data
    # ========================================================================
    log("=== PHASE 1: Fetching preseason data ===")

    log("Fetching draft data...")
    run_script(DRAFT_SCRIPT, "Draft data", year=year)

    log("Fetching season schedule...")
    run_script(SCHEDULE_SCRIPT, "Season schedule", year=year)

    log("Fetching Week 0 rosters (post-draft)...")
    run_script(MERGE_SCRIPT, "Week 0 player stats", year=year, week=0)

    # ========================================================================
    # PHASE 2: Update canonical files
    # ========================================================================
    log("")
    log("=== PHASE 2: Updating canonical files ===")

    # Schedule
    src = locate_latest_parquet("schedule", year)
    if src and src.exists():
        try:
            df_src = pd.read_parquet(src)
            if not df_src.empty:
                total = upsert_parquet_via_duckdb(CANONICAL["schedule"], df_src, DEDUP_KEYS["schedule"], "schedule")
                log(f"✓ Updated schedule.parquet: {total:,} total rows")
        except Exception as e:
            log(f"WARN: Could not process schedule: {e}")

    # Player (Week 0)
    src = locate_latest_parquet("player", year)
    if src and src.exists():
        try:
            df_src = pd.read_parquet(src)
            if not df_src.empty:
                total = upsert_parquet_via_duckdb(CANONICAL["player"], df_src, DEDUP_KEYS["player"], "player")
                log(f"✓ Updated player.parquet: {total:,} total rows")
        except Exception as e:
            log(f"WARN: Could not process player data: {e}")

    # ========================================================================
    # PHASE 3: Run keeper analysis (if applicable)
    # ========================================================================
    log("")
    log("=== PHASE 3: Post-processing ===")

    run_post_script(KEEPER_SCRIPT, "Keeper analysis")

    log("")
    log("=" * 80)
    log("PRESEASON IMPORT COMPLETED!")
    log("=" * 80)
    log("")
    log("Data captured:")

    # Check draft data
    draft_file = SOURCE_DIRS["draft"] / f"draft_data_{year}.csv"
    if draft_file.exists():
        try:
            draft_df = pd.read_csv(draft_file)
            log(f"  Draft data: {len(draft_df):,} picks")
        except:
            pass

    # Check schedule
    if CANONICAL["schedule"].exists():
        try:
            sched_df = pd.read_parquet(CANONICAL["schedule"])
            year_sched = sched_df[sched_df["year"] == year]
            log(f"  Schedule:   {len(year_sched):,} matchups for {year}")
        except:
            pass

    # Check Week 0 rosters
    if CANONICAL["player"].exists():
        try:
            player_df = pd.read_parquet(CANONICAL["player"])
            week0 = player_df[(player_df["year"] == year) & (player_df["week"] == 0)]
            log(f"  Week 0:     {len(week0):,} rostered players")
        except:
            pass

    log("")
    log("Next steps:")
    log("  1. Verify draft data is complete")
    log("  2. Wait for Week 1 to complete (Tuesday after MNF)")
    log("  3. Run weekly_import.py to start regular season updates")

if __name__ == "__main__":
    main()

