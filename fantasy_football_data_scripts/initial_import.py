#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INITIAL IMPORT SCRIPT - Simplified for reliability

Fetches all historical data and consolidates into canonical parquet files.
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import duckdb

# =============================================================================
# Paths
# =============================================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parent
REPO_DIR = ROOT_DIR.parent

# OAuth utilities (safe import, fall back to adding script dir to sys.path)
try:
    from oauth_utils import find_oauth_file, create_oauth2
except Exception:
    # If importing directly fails (running from different CWD), add script dir to path
    sys.path.insert(0, str(ROOT_DIR))
    from oauth_utils import find_oauth_file, create_oauth2

# Use EXPORT_DATA_DIR if set, otherwise default
env_data_dir = os.environ.get("EXPORT_DATA_DIR") or os.environ.get("DATA_DIR")
if env_data_dir:
    DATA_DIR = Path(env_data_dir).resolve()
else:
    DATA_DIR = REPO_DIR / "fantasy_football_data"

# Producer scripts
SCHEDULE_SCRIPT = ROOT_DIR / "schedule_script" / "season_schedules.py"
MATCHUP_SCRIPT = ROOT_DIR / "matchup_scripts" / "weekly_matchup_data.py"
TRANSACTION_SCRIPT = ROOT_DIR / "transaction_scripts" / "transactions.py"
MERGE_SCRIPT = ROOT_DIR / "player_stats" / "yahoo_nfl_merge.py"

# Canonical output files
CANONICAL = {
    "schedule": DATA_DIR / "schedule.parquet",
    "matchup": DATA_DIR / "matchup.parquet",
    "transactions": DATA_DIR / "transactions.parquet",
    "player": DATA_DIR / "player.parquet",
}

# Source directories (where producers write their outputs)
SOURCE_DIRS = {
    "schedule": DATA_DIR / "schedule_data",
    "matchup": DATA_DIR / "matchup_data",
    "transactions": DATA_DIR / "transaction_data",
    "player": DATA_DIR / "player_data",
}

# Deduplication keys
DEDUP_KEYS = {
    "schedule": ["year", "week", "manager", "opponent"],
    "matchup": ["manager", "opponent", "year", "week"],
    "transactions": ["transaction_key", "year", "week"],
    "player": ["yahoo_player_id", "nfl_player_id", "year", "week"],
}

# Post-processing scripts (run AFTER canonical tables are built)
POST_SCRIPTS = [
    (ROOT_DIR / "matchup_scripts" / "cumulative_stats.py", "Matchup cumulative stats"),
    (ROOT_DIR / "player_stats" / "cumulative_player_stats.py", "Player cumulative stats"),
    (ROOT_DIR / "matchup_scripts" / "expected_record_import.py", "Expected record calculation"),
    (ROOT_DIR / "matchup_scripts" / "opponent_expected_record.py", "Opponent expected record"),
    (ROOT_DIR / "matchup_scripts" / "playoff_odds_import.py", "Playoff odds calculation"),
    (ROOT_DIR / "player_stats" / "keeper_import.py", "Keeper analysis"),
    (ROOT_DIR / "player_stats" / "matchup_stats_import.py", "Matchup stats import"),
    (ROOT_DIR / "matchup_scripts" / "add_optimal.py", "Add optimal lineup"),
    (ROOT_DIR / "player_stats" / "aggregate_on_season.py", "Season aggregation"),
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
def run_script(script: Path, label: str, year: int = 0, week: int = 0) -> bool:
    """Run a producer script and return success status"""
    if not script.exists():
        log(f"⚠️  SKIP (missing): {script}")
        return False

    env = dict(os.environ)
    env["KMFFL_YEAR"] = str(year)
    env["KMFFL_WEEK"] = str(week)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, str(script), "--year", str(year), "--week", str(week)]
    log(f"▶️  RUN: {label} -> {script.name} --year {year} --week {week}")

    try:
        # Capture both stdout and stderr
        result = subprocess.run(
            cmd,
            cwd=str(script.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Log output
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-10:]:  # Last 10 lines
                log(f"     {line}")

        if result.returncode != 0:
            log(f"⚠️  {label} exited with code {result.returncode}")
            if result.stderr:
                log(f"  ❌ ERROR OUTPUT:")
                for line in result.stderr.strip().split('\n')[-20:]:  # Last 20 error lines
                    log(f"     {line}")
            return False
        else:
            log(f"✅ Completed: {label}")
            return True
    except subprocess.TimeoutExpired:
        log(f"❌ TIMEOUT running {label} (exceeded 5 minutes)")
        return False
    except Exception as e:
        log(f"❌ ERROR running {label}: {e}")
        return False


def run_post_script(script: Path, label: str) -> bool:
    """Run a post-processing script and return success status"""
    if not script.exists():
        log(f"⚠️  SKIP (missing): {label} -> {script}")
        return False

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(script)]
    log(f"▶️  RUN POST: {label} -> {script.name}")

    try:
        rc = subprocess.call(cmd, cwd=str(script.parent), env=env)
        if rc != 0:
            log(f"⚠️  {label} exited with code {rc}")
            return False
        else:
            log(f"✅ Completed POST: {label}")
            return True
    except Exception as e:
        log(f"❌ ERROR running {label}: {e}")
        return False


# =============================================================================
# Parquet discovery
# =============================================================================
def find_latest_parquet(kind: str) -> Path | None:
    """Find the most recent parquet file for a given data type"""
    base = SOURCE_DIRS[kind]
    if not base.exists():
        log(f"⚠️  Source directory doesn't exist: {base}")
        return None

    # Preferred filenames (in priority order)
    preferred = {
        "schedule": ["schedule_data_all_years.parquet", "schedule.parquet"],
        "matchup": ["matchup.parquet", "matchup_data_week_all_year_0.parquet"],
        "transactions": ["transactions.parquet", "transaction.parquet"],
        "player": ["yahoo_player_stats_multi_year_all_weeks.parquet",
                   "player.parquet", "players_by_year.parquet"],
    }.get(kind, [])

    # Try preferred names first
    for name in preferred:
        p = base / name
        if p.exists():
            log(f"✅ Found preferred file for {kind}: {p.name}")
            return p

    # Fallback: find most recent parquet file
    parquets = list(base.glob("*.parquet"))
    if parquets:
        latest = max(parquets, key=lambda p: p.stat().st_mtime)
        log(f"✅ Found latest file for {kind}: {latest.name}")
        return latest

    log(f"❌ No parquet files found for {kind} in {base}")
    return None


# =============================================================================
# DuckDB upsert (consolidation logic)
# =============================================================================
def consolidate_to_canonical(out_path: Path, new_df: pd.DataFrame, keys: list[str]) -> int:
    """
    Merge new data into canonical parquet file using DuckDB.
    Returns: total row count after merge
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    new_df = new_df.copy()
    new_df.columns = [str(c).strip() for c in new_df.columns]

    con.register("new_df", new_df)
    con.execute("CREATE TEMP TABLE _new AS SELECT * FROM new_df")

    out_str = str(out_path).replace("\\", "/")

    if out_path.exists():
        # Merge with existing data
        log(f"  📊 Merging with existing {out_path.name}...")
        con.execute(f"CREATE TEMP TABLE _old AS SELECT * FROM read_parquet('{out_str}')")

        # Get columns from both tables
        cols_new = [r[1] for r in con.execute("PRAGMA table_info('_new')").fetchall()]
        cols_old = [r[1] for r in con.execute("PRAGMA table_info('_old')").fetchall()]
        all_cols = list(dict.fromkeys(cols_old + cols_new))

        # Build SELECT statements with NULL padding for missing columns
        sel_old = ", ".join([f'"{c}"' if c in cols_old else f'NULL AS "{c}"' for c in all_cols])
        sel_new = ", ".join([f'"{c}"' if c in cols_new else f'NULL AS "{c}"' for c in all_cols])

        # Determine partition columns for deduplication
        partition_cols = [c for c in keys if c in all_cols]
        if not partition_cols:
            partition_cols = ["year", "week"] if "year" in all_cols and "week" in all_cols else all_cols[:1]

        partition_by = ", ".join([f'"{c}"' for c in partition_cols])

        # Merge: prefer new data over old
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
        # No existing file, just use new data
        log(f"  📝 Creating new {out_path.name}...")
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
    log("=" * 80)
    log("INITIAL IMPORT: Building complete league history from year 0")
    log("=" * 80)
    log(f"Data directory: {DATA_DIR}")

    # Create all necessary directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in SOURCE_DIRS.values():
        subdir.mkdir(parents=True, exist_ok=True)
        log(f"  📁 Created/verified: {subdir}")

    # Debug: Check OAuth file
    # =============================================================================
    # Safe OAuth path detection - respects OAUTH_PATH env var and falls back to searching
    # =============================================================================
    try:
        THIS_FILE_VAR = __file__
        BASE_DIR = os.path.dirname(os.path.abspath(THIS_FILE_VAR))
    except NameError:
        BASE_DIR = os.getcwd()

    # Preferred: OAUTH_PATH environment variable
    oauth_candidate = None
    if os.environ.get("OAUTH_PATH"):
        oauth_candidate = Path(os.environ.get("OAUTH_PATH")).resolve()
        log(f"Using OAUTH_PATH from environment: {oauth_candidate}")
    else:
        # Default relative location in repo
        oauth_candidate = (REPO_DIR / "oauth" / "Oauth.json").resolve()
        log(f"Using default candidate OAUTH_PATH: {oauth_candidate}")

    # If candidate doesn't exist, try searching via oauth_utils.find_oauth_file()
    if not oauth_candidate.exists():
        found = find_oauth_file()
        if found:
            oauth_candidate = found
            log(f"Found OAuth file by search: {oauth_candidate}")
        else:
            log("⚠️  OAuth.json not found in default locations; some operations may fail if authentication is required.")

    # Export to environment so child scripts pick it up when we spawn subprocesses
    try:
        if oauth_candidate and oauth_candidate.exists():
            os.environ["OAUTH_PATH"] = str(oauth_candidate)
            log(f"Exported OAUTH_PATH -> {os.environ.get('OAUTH_PATH')}")
            # Attempt to validate by creating an oauth session (non-fatal)
            try:
                create_oauth2(oauth_candidate)
                log("✅ OAuth.json appears valid (create_oauth2 succeeded)")
            except Exception as e:
                log(f"⚠️  create_oauth2 validation failed: {e}")
        else:
            log("⚠️  No OAuth.json available to export to environment.")
    except Exception as e:
        log(f"⚠️  Error setting OAUTH_PATH: {e}")

    log("")

    # Auto-confirm check
    auto_confirm = os.environ.get("AUTO_CONFIRM", "").lower() in ("1", "true", "yes")
    if not auto_confirm:
        try:
            if sys.stdin and sys.stdin.isatty():
                response = input("Continue? (yes/no): ").strip().lower()
                if response not in ("y", "yes"):
                    log("Aborted by user")
                    return
            else:
                log("Non-interactive environment detected – auto-confirming import.")
        except Exception:
            log("Auto-confirming import due to non-interactive environment.")
    else:
        log("AUTO_CONFIRM set – proceeding without interactive prompt.")

    # =============================================================================
    # PHASE 1: Historical data collection (year=0, week=0)
    # =============================================================================
    log("")
    log("=" * 80)
    log("PHASE 1: Historical data collection (year=0, week=0)")
    log("=" * 80)

    results = {}

    log("\n📅 Fetching ALL schedule data (all years)...")
    results['schedule'] = run_script(SCHEDULE_SCRIPT, "Season schedules", year=0, week=0)

    log("\n🏈 Fetching ALL matchup data (all years, all weeks)...")
    results['matchup'] = run_script(MATCHUP_SCRIPT, "Weekly matchup data", year=0, week=0)

    log("\n💰 Fetching ALL transaction data (all years)...")
    results['transactions'] = run_script(TRANSACTION_SCRIPT, "Transactions", year=0, week=0)

    log("\n👤 Fetching ALL player stats (all years, all weeks)...")
    results['player'] = run_script(MERGE_SCRIPT, "Yahoo/NFL merge", year=0, week=0)

    # Summary
    log("")
    log("📊 Phase 1 Summary:")
    for kind, success in results.items():
        status = "✅" if success else "⚠️"
        log(f"  {status} {kind}: {'success' if success else 'completed with warnings'}")

    # =============================================================================
    # PHASE 2: Building canonical parquet files
    # =============================================================================
    log("")
    log("=" * 80)
    log("PHASE 2: Building canonical parquet files")
    log("=" * 80)

    created_canonicals = {}

    for kind in ["schedule", "matchup", "transactions", "player"]:
        log(f"\n--- Processing {kind.upper()} ---")

        # Debug: Check if source directory exists and what's in it
        src_dir = SOURCE_DIRS[kind]
        log(f"  🔍 Checking source directory: {src_dir}")
        if src_dir.exists():
            try:
                items = list(src_dir.iterdir())
                log(f"  📂 Directory exists with {len(items)} items:")
                for item in items[:5]:
                    log(f"     - {item.name}")
                if len(items) > 5:
                    log(f"     ... and {len(items) - 5} more")
            except Exception as e:
                log(f"  ❌ Error listing directory: {e}")
        else:
            log(f"  ❌ Source directory doesn't exist!")

        # Find source file
        src = find_latest_parquet(kind)

        if not src or not src.exists():
            log(f"⚠️  Skipping {kind} - no source file found")
            continue

        # Verify file is readable
        try:
            file_size = src.stat().st_size
            log(f"  📁 Source: {src.name} ({file_size / 1024:.1f} KB)")
        except Exception as e:
            log(f"❌ Can't access file: {e}")
            continue

        # Read parquet
        try:
            log(f"  📖 Reading parquet...")
            df_src = pd.read_parquet(src)
            log(f"  ✅ Read {len(df_src):,} rows, {len(df_src.columns)} columns")
        except Exception as e:
            log(f"❌ Failed to read {src.name}: {e}")
            continue

        if df_src.empty:
            log(f"⚠️  Source dataframe is empty, skipping")
            continue

        # Consolidate to canonical file
        try:
            log(f"  💾 Writing to canonical: {CANONICAL[kind].name}")
            total_rows = consolidate_to_canonical(CANONICAL[kind], df_src, DEDUP_KEYS[kind])
            log(f"✅ Created {CANONICAL[kind].name}: {total_rows:,} total rows")

            # Verify file was written
            if CANONICAL[kind].exists():
                actual_size = CANONICAL[kind].stat().st_size
                log(f"  ✓ Verified on disk: {actual_size / 1024:.1f} KB")
                created_canonicals[kind] = str(CANONICAL[kind].resolve())
            else:
                log(f"⚠️  WARNING: File doesn't exist after write: {CANONICAL[kind]}")

        except Exception as e:
            log(f"❌ Failed to create canonical {kind} file: {e}")
            import traceback
            log(traceback.format_exc())
            continue

    log(f"\n✅ Phase 2 complete. Created {len(created_canonicals)}/{len(CANONICAL)} canonical files.")

    # =============================================================================
    # PHASE 3: Post-processing
    # =============================================================================
    log("")
    log("=" * 80)
    log("PHASE 3: Post-processing")
    log("=" * 80)

    post_results = []
    for script, label in POST_SCRIPTS:
        success = run_post_script(script, label)
        post_results.append((label, success))

    # Summary
    log("")
    log("📊 Phase 3 Summary:")
    for label, success in post_results:
        status = "✅" if success else "⚠️"
        log(f"  {status} {label}")

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    log("")
    log("=" * 80)
    log("INITIAL IMPORT COMPLETED!")
    log("=" * 80)
    log("")
    log("Your canonical files:")

    for kind, path_str in CANONICAL.items():
        path = Path(path_str)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                size = path.stat().st_size / 1024
                log(f"  ✅ {kind:15s}: {len(df):,} rows, {size:.1f} KB -> {path}")
            except Exception:
                log(f"  ⚠️  {kind:15s}: exists but couldn't read -> {path}")
        else:
            log(f"  ❌ {kind:15s}: NOT CREATED")

    # Output canonical paths for parent process to parse
    log("")
    log("Canonical file paths:")
    for kind, path in created_canonicals.items():
        log(f"  {kind}: {path}")


if __name__ == "__main__":
    main()