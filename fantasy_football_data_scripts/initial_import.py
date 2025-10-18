#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INITIAL IMPORT SCRIPT (wrapper)

Purpose: One-time historical data import (hitting year 0) to build complete league history.
Run by the Streamlit app after OAuth is saved.

Flow:
    1) Run your four producers (schedule, matchup, transactions, player) with year=0/week=0
    2) Upsert into canonical parquet files in fantasy_football_data/
    3) Run post-processing scripts that read/write those canonical files
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import duckdb

# =============================================================================
# Paths
# =============================================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parent
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "fantasy_football_data"

# Producer scripts (adjust names/paths if yours differ)
SCHEDULE_SCRIPT    = ROOT_DIR / "schedule_script"     / "season_schedules.py"
MATCHUP_SCRIPT     = ROOT_DIR / "matchup_scripts"     / "weekly_matchup_data.py"
TRANSACTION_SCRIPT = ROOT_DIR / "transaction_scripts" / "transactions.py"
MERGE_SCRIPT       = ROOT_DIR / "player_stats"        / "yahoo_nfl_merge.py"

# Canonical targets
CANONICAL = {
    "schedule":     DATA_DIR / "schedule.parquet",
    "matchup":      DATA_DIR / "matchup.parquet",
    "transactions": DATA_DIR / "transactions.parquet",
    "player":       DATA_DIR / "player.parquet",
}

# Source directories (where producers drop their outputs)
SOURCE_DIRS = {
    "schedule":     DATA_DIR / "schedule_data",
    "matchup":      DATA_DIR / "matchup_data",
    "transactions": DATA_DIR / "transaction_data",
    "player":       DATA_DIR / "player_data",
}

# Dedup keys (tweak to match your model)
DEDUP_KEYS = {
    "schedule":     ["year", "week", "manager", "opponent"],
    "matchup":      ["manager", "opponent", "year", "week"],
    "transactions": ["transaction_key", "year", "week"],
    "player":       ["yahoo_player_id", "nfl_player_id", "year", "week"],
}

# Post-processing scripts (run AFTER canonical tables are built)
RUNS_POST: List[tuple[Path, str]] = [
    (ROOT_DIR / "matchup_scripts" / "cumulative_stats.py",           "Matchup cumulative stats"),
    (ROOT_DIR / "player_stats"  / "cumulative_player_stats.py",      "Player cumulative stats"),
    (ROOT_DIR / "matchup_scripts" / "expected_record_import.py",     "Expected record calculation"),
    (ROOT_DIR / "matchup_scripts" / "opponent_expected_record.py",   "Opponent expected record"),
    (ROOT_DIR / "matchup_scripts" / "playoff_odds_import.py",        "Playoff odds calculation"),
    (ROOT_DIR / "player_stats"  / "keeper_import.py",                "Keeper analysis"),
    (ROOT_DIR / "player_stats"  / "matchup_stats_import.py",         "Matchup stats import"),
    (ROOT_DIR / "matchup_scripts" / "add_optimal.py",                "Add optimal lineup"),
    (ROOT_DIR / "player_stats"  / "aggregate_on_season.py",          "Season aggregation"),
    # DRAFT STEP REMOVED FOR NOW:
    # (ROOT_DIR / "draft_scripts" / "ppg_draft_join.py",               "Draft PPG join"),
]

# =============================================================================
# Logging
# =============================================================================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# Script runners
# =============================================================================
def run_script(script: Path, label: str, year: int = 0, week: int = 0) -> None:
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
        rc = subprocess.call(cmd, cwd=str(script.parent), env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
        else:
            log(f"✓ Completed: {label}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")

def run_post_script(script: Path, label: str) -> None:
    if not script.exists():
        log(f"SKIP (missing): {label} -> {script}")
        return
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(script)]
    log(f"RUN POST: {label} -> {script.name}")
    try:
        rc = subprocess.call(cmd, cwd=str(script.parent), env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
        else:
            log(f"✓ Completed POST: {label}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")

# =============================================================================
# Parquet discovery
# =============================================================================
def locate_latest_parquet(kind: str) -> Optional[Path]:
    base = SOURCE_DIRS[kind]
    if not base.exists():
        return None
    preferred = {
        "schedule":     ["schedule_data_all_years.parquet", "schedule.parquet"],
        "matchup":      ["matchup.parquet"],
        "transactions": ["transactions.parquet"],
        "player":       ["yahoo_player_stats_multi_year_all_weeks.parquet", "player.parquet"],
    }.get(kind, [])
    for name in preferred:
        p = base / name
        if p.exists():
            return p
    parquets = list(base.glob("*.parquet"))
    if parquets:
        return max(parquets, key=lambda p: p.stat().st_mtime)
    return None

# =============================================================================
# DuckDB upsert
# =============================================================================
def upsert_parquet_via_duckdb(out_path: Path, new_df: pd.DataFrame, keys: list[str], kind: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    new_df = new_df.copy()
    new_df.columns = [str(c).strip() for c in new_df.columns]
    con.register("new_df", new_df)
    con.execute("CREATE TEMP TABLE _new AS SELECT * FROM new_df")
    out_str = str(out_path).replace("\\", "/")
    if out_path.exists():
        con.execute(f"CREATE TEMP TABLE _old AS SELECT * FROM read_parquet('{out_str}')")
        cols_new = [r[1] for r in con.execute("PRAGMA table_info('_new')").fetchall()]
        cols_old = [r[1] for r in con.execute("PRAGMA table_info('_old')").fetchall()]
        all_cols = list(dict.fromkeys(cols_old + cols_new))
        sel_old = ", ".join([f'"{c}"' if c in cols_old else f'NULL AS "{c}"' for c in all_cols])
        sel_new = ", ".join([f'"{c}"' if c in cols_new else f'NULL AS "{c}"' for c in all_cols])
        partition_cols = [c for c in keys if c in all_cols]
        if not partition_cols:
            partition_cols = ["year", "week"] if "year" in all_cols and "week" in all_cols else all_cols[:1]
        partition_by = ", ".join([f'"{c}"' for c in partition_cols])
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
    log("=" * 80)
    log("INITIAL IMPORT: Building complete league history from year 0")
    log("=" * 80)
    log("")
    auto_confirm = os.environ.get("AUTO_CONFIRM", "").lower() in ("1", "true", "yes")
    if not auto_confirm:
        try:
            if sys.stdin and sys.stdin.isatty():
                response = input("Continue? (yes/no): ").strip().lower()
                if response not in ("y", "yes"):
                    log("Aborted by user")
                    return
            else:
                log("Non-interactive environment detected — auto-confirming import.")
        except Exception:
            log("Auto-confirming import due to non-interactive environment.")
    else:
        log("AUTO_CONFIRM set — proceeding without interactive prompt.")

    # PHASE 1
    log("")
    log("=== PHASE 1: Historical data collection (year=0) ===")
    log("Fetching ALL schedule data (all years)...")
    run_script(SCHEDULE_SCRIPT, "Season schedules", year=0, week=0)

    log("Fetching ALL matchup data (all years, all weeks)...")
    run_script(MATCHUP_SCRIPT, "Weekly matchup data", year=0, week=0)

    log("Fetching ALL transaction data (all years)...")
    run_script(TRANSACTION_SCRIPT, "Transactions", year=0, week=0)

    log("Fetching ALL player stats (all years, all weeks)...")
    run_script(MERGE_SCRIPT, "Yahoo/NFL merge", year=0, week=0)

    # PHASE 2
    log("")
    log("=== PHASE 2: Building canonical parquet files ===")
    for kind in ["schedule", "matchup", "transactions", "player"]:
        src = locate_latest_parquet(kind)
        if not src or not src.exists():
            log(f"WARN: No {kind} source found in {SOURCE_DIRS[kind]}")
            continue
        try:
            df_src = pd.read_parquet(src)
        except Exception as e:
            log(f"WARN: Could not read {kind} source {src}: {e}")
            continue
        if df_src.empty:
            log(f"WARN: {kind} source is empty: {src.name}")
            continue
        rows_used = len(df_src)
        log(f"Processing {kind}: {src.name} ({rows_used:,} rows)")
        total_rows = upsert_parquet_via_duckdb(CANONICAL[kind], df_src, DEDUP_KEYS[kind], kind)
        log(f"✓ Updated {CANONICAL[kind].name}: {total_rows:,} total rows")

    # PHASE 3
    log("")
    log("=== PHASE 3: Post-processing ===")
    for script, label in RUNS_POST:
        run_post_script(script, label)

    log("")
    log("=" * 80)
    log("INITIAL IMPORT COMPLETED!")
    log("=" * 80)
    log("")
    log("Your canonical files are ready:")
    for kind, path in CANONICAL.items():
        if path.exists():
            try:
                df = pd.read_parquet(path)
                log(f"  {kind:15s}: {len(df):,} rows -> {path}")
            except Exception:
                log(f"  {kind:15s}: -> {path}")

if __name__ == "__main__":
    main()
