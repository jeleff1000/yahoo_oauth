#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import re
import math
import glob
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Tuple

import pandas as pd
import duckdb

# =============================================================================
# Relative layout (no hard-coded absolute paths)
# =============================================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parent                                   # .../fantasy_football_data_scripts
REPO_DIR  = ROOT_DIR.parent                                    # .../fantasy_football_data_downloads
DATA_DIR  = REPO_DIR / "fantasy_football_data"                 # .../fantasy_football_data

# Producer scripts (match your repo layout)
SCHEDULE_SCRIPT    = ROOT_DIR / "schedule_script"     / "season_schedules.py"
MATCHUP_SCRIPT     = ROOT_DIR / "matchup_scripts"     / "weekly_matchup_data.py"
TRANSACTION_SCRIPT = ROOT_DIR / "transaction_scripts" / "transactions.py"
MERGE_SCRIPT       = ROOT_DIR / "player_stats"        / "yahoo_nfl_merge.py"

# Canonical targets (rolling)
CANONICAL = {
    "schedule":     DATA_DIR / "schedule.parquet",
    "matchup":      DATA_DIR / "matchup.parquet",
    "transactions": DATA_DIR / "transactions.parquet",
    "player":       DATA_DIR / "player.parquet",
}

# Where each producer tends to drop fresh files (but we will also glob)
SOURCE_DIRS = {
    "schedule":     DATA_DIR / "schedule_data",
    "matchup":      DATA_DIR / "matchup_data",
    "transactions": DATA_DIR / "transaction_data",
    "player":       DATA_DIR / "player_data",
}

# Dedup keys (used by Pandas path; DuckDB path computes per-kind partitions below)
DEDUP_KEYS = {
    "schedule":     ["year", "week", "manager", "opponent"],
    "matchup":      ["manager", "opponent", "year", "week"],
    "transactions": ["transaction_key", "year", "week"],
    # for players we override at runtime to use COALESCE(nfl_player_id, yahoo_player_id)
    "player":       ["yahoo_player_id", "nfl_player_id", "year", "week"],
}

# Column signatures (used to positively identify files even if names vary)
SIGNATURES: Dict[str, List[str]] = {
    "schedule":     ["is_playoffs","is_consolation","manager","opponent","week","year","team_points","opponent_points"],
    "matchup":      ["manager","opponent","week","year","team_points","opponent_points","win","loss"],
    "transactions": ["transaction_key","timestamp","week","year","status"],
    "player":       ["player","yahoo_player_id","nfl_player_id","year","week","points"],
}

# =============================================================================
# Phase 2: IN-HOUSE CALCS & UPLOAD (run AFTER canonical tables updated)
# =============================================================================
RUNS_POST = [
    (ROOT_DIR / "matchup_scripts" / "cumulative_stats.py",          "post: matchup cumulative_stats"),
    (ROOT_DIR / "player_stats"   / "cumulative_player_stats.py",    "post: player cumulative_player_stats"),
    (ROOT_DIR / "matchup_scripts" / "expected_record_import.py",    "post: matchup expected_record_import"),
    (ROOT_DIR / "matchup_scripts" / "opponent_expected_record.py",  "post: matchup opponent_expected_record"),
    (ROOT_DIR / "matchup_scripts" / "playoff_odds_import.py",       "post: matchup playoff_odds_import"),
    (ROOT_DIR / "player_stats"   / "keeper_import.py",              "post: player keeper_import"),
    (ROOT_DIR / "player_stats"   / "matchup_stats_import.py",       "post: player matchup_stats_import"),
    (ROOT_DIR / "matchup_scripts" / "add_optimal.py",               "post: matchup add_optimal"),
    (ROOT_DIR / "player_stats"   / "aggregate_on_season.py",        "post: player aggregate_on_season"),
    (ROOT_DIR / "draft_scripts"  / "ppg_draft_join.py",             "post: draft ppg_draft_join"),
    (DATA_DIR / "motherduck_upload.py",                             "post: motherduck_upload"),
]

PROMPT_DETECT_TIMEOUT_SEC = 2


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
        tz = timezone(timedelta(hours=-4))
    now = datetime.now(tz)
    season = _season_for_today(now)
    cutoff = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    first_thu = _first_thu_after_labor_day(season, tz)
    week = 1 + max(0, (cutoff - first_thu).days // 7)
    week = max(1, min(22, week))
    return season, week


# =============================================================================
# Process running (CLI first, then stdin if script prompts)
# =============================================================================
def _bundle_year_only(year: int) -> bytes:
    lines = [str(year)]
    return ("\n".join(lines) + "\n").encode("utf-8")

def _bundle_year_week(year: int, week: int) -> bytes:
    lines = [str(year), str(week)]
    return ("\n".join(lines) + "\n").encode("utf-8")

def run_script_interactive(script: Path, label: str, mode: str, year: int, week: int) -> None:
    if not script.exists():
        log(f"SKIP (missing): {script}")
        return
    env = dict(os.environ)
    env["KMFFL_YEAR"] = str(year)
    env["KMFFL_WEEK"] = str(week)
    env["PYTHONUNBUFFERED"] = "1"

    cli = [sys.executable, str(script), "--year", str(year), "--week", str(week)]
    log(f"RUN (CLI, {PROMPT_DETECT_TIMEOUT_SEC}s): {script.name} --year {year} --week {week}")
    try:
        res = subprocess.run(cli, cwd=script.parent, env=env, timeout=PROMPT_DETECT_TIMEOUT_SEC, check=False)
        if res.returncode == 0:
            log(f"OK (CLI): {label}")
            return
        log(f"CLI rc={res.returncode}; switching to stdin mode.")
    except subprocess.TimeoutExpired:
        log("Detected interactive prompt; switching to stdin mode.")

    payload = _bundle_year_only(year) if mode == "year_only" else _bundle_year_week(year, week)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
    )
    out, _ = proc.communicate(payload)
    rc = proc.returncode
    tail = ""
    try:
        tail = "\n".join(out.decode(errors="ignore").splitlines()[-20:])
    except Exception:
        pass
    log(f"STDIN exit code {rc} for {label}" + (f"\n…tail:\n{tail}" if tail else ""))
    if rc != 0:
        raise SystemExit(f"{label} failed (rc={rc})")

def run_script_simple(script: Path, label: str, year: int, week: int) -> None:
    if not script.exists():
        log(f"SKIP (missing): {label} -> {script}")
        return
    env = dict(os.environ)
    env["KMFFL_YEAR"] = str(year)
    env["KMFFL_WEEK"] = str(week)
    env["PYTHONUNBUFFERED"] = "1"

    # Most post-processing scripts don't take CLI args; they process canonical files directly
    # Only pass args if the script explicitly supports them
    scripts_with_args = {
        "ppg_draft_join.py",
        "motherduck_upload.py",
    }

    if script.name in scripts_with_args:
        cmd = [sys.executable, str(script), "--year", str(year), "--week", str(week)]
        log(f"RUN: {label} -> {script.name} --year {year} --week {week}")
    else:
        cmd = [sys.executable, str(script)]
        log(f"RUN: {label} -> {script.name} (no args, processes canonical files)")

    try:
        rc = subprocess.call(cmd, cwd=script.parent, env=env)
        if rc != 0:
            log(f"WARN: {label} exited with code {rc}")
    except Exception as e:
        log(f"ERROR running {label}: {e}")


# =============================================================================
# Parquet discovery helpers
# =============================================================================
def _read_parquet_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception as e:
        log(f"WARN: Could not read {path}: {e}")
    return None

def _has_signature(df: pd.DataFrame, cols_any: List[str]) -> bool:
    cols = set(c.lower() for c in df.columns)
    return any(c.lower() in cols for c in cols_any)

def _filter_year_week(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    d = df
    if "year" in d.columns and year > 0:
        d = d[pd.to_numeric(d["year"], errors="coerce") == year]
    if "week" in d.columns and week > 0:
        d = d[pd.to_numeric(d["week"], errors="coerce") == week]
    return d

def _is_week_complete(df: pd.DataFrame, year: int, week: int, kind: str) -> bool:
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

        # **CRITICAL FIX: Filter to only offensive players + kickers**
        # Defensive players (IDP) legitimately have 0 points most weeks
        # Only count QB, RB, WR, TE, K positions (offensive skill positions that score points)
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
        # (Thursday night only has ~20-30 offensive players active out of ~200+ total offensive players)
        if total_rows > 0:
            completion_rate = non_zero_count / total_rows
            log(f"  [Check] Week {week} completion rate: {completion_rate:.1%} ({non_zero_count}/{total_rows} offensive players with points)")

            # Threshold: if less than 10% have points, it's probably just Thursday night
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

            # For matchups, we need at least 50% to have points (some might be 0 legitimately)
            if completion_rate < 0.50:
                log(f"  [WARN] Week {week} matchups appear INCOMPLETE")
                return False

    log(f"  [OK] Week {week} appears COMPLETE")
    return True

def _locate_latest_parquet(kind: str, year: int, week: int) -> Optional[Path]:
    base = SOURCE_DIRS[kind]
    base.mkdir(parents=True, exist_ok=True)
    preferred = {
        "schedule":     ["schedule.parquet", "schedule_data.parquet", f"schedule_data_year_{year}.parquet"],
        "matchup":      [f"matchup_data_week_{week:02d}_year_{year}.parquet", "matchup.parquet"],
        "transactions": ["transactions.parquet"],
        "player":       [
            f"yahoo_nfl_merged_{year}_week_{week}.parquet",
            f"yahoo_player_stats_{year}_week_{week}.parquet",
            f"yahoo_player_stats_{year}_all_weeks.parquet",
            "yahoo_player_stats_multi_year_all_weeks.parquet",
            f"yahoo_player_stats_multi_year_week_{week}.parquet",
            "player.parquet",
        ],
    }.get(kind, [])

    for name in preferred:
        p = base / name
        if p.exists():
            return p

    best: Tuple[int, float, Path] | None = None
    sig = SIGNATURES[kind]
    for p in base.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if not _has_signature(df, sig):
            continue
        df_tw = _filter_year_week(df, year, week)
        rows = len(df_tw) if not df_tw.empty else len(df)
        score = (rows, p.stat().st_mtime)
        if (best is None) or (score > (best[0], best[1])):
            best = (rows, p.stat().st_mtime, p)
    if best:
        return best[2]
    return None


# =============================================================================
# DuckDB upsert helpers
# =============================================================================
def _sanitize_for_duckdb(df: pd.DataFrame) -> pd.DataFrame:
    # DuckDB handles mixed dtypes well; ensure column names are strings and strip BOM/whitespace
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _duckdb_literal(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _duckdb_cast(col: str, kind: str) -> str:
    # Very light casting; you can expand if specific columns need coercion.
    c = _duckdb_literal(col)
    if kind in ("schedule", "matchup", "transactions", "player"):
        if col.lower() in ("year", "week"):
            return f"TRY_CAST({c} AS BIGINT) AS {c}"
    return f"{c} AS {c}"

def _duckdb_cast_list(df: pd.DataFrame, kind: str) -> str:
    return ", ".join(_duckdb_cast(c, kind) for c in df.columns)

def _duckdb_cols(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    # PRAGMA table_info returns: (column_id, column_name, column_type, ...)
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]

def _partition_clause_for_kind(kind: str, available_cols: List[str]) -> str:
    cols = set(available_cols)

    def present(sequence: List[str]) -> List[str]:
        return [c for c in sequence if c in cols]

    if kind == "player":
        base = present(["__key_player", "year", "week"])
        if not base:
            base = present(["yahoo_player_id", "nfl_player_id", "year", "week"])
        if not base:
            base = present(["player", "year", "week"])
        if not base:
            base = present(["year", "week"])
        return ", ".join(_duckdb_literal(c) for c in base)

    if kind == "matchup":
        base = present(["manager", "opponent", "year", "week"])
        if not base:
            base = present(["year", "week"])
        return ", ".join(_duckdb_literal(c) for c in base)

    if kind == "transactions":
        base = present(["transaction_key", "year", "week"])
        if not base:
            base = present(["year", "week"])
        return ", ".join(_duckdb_literal(c) for c in base)

    if kind == "schedule":
        base = present(["year", "week", "manager", "opponent"])
        if not base:
            base = present(["year", "week"])
        return ", ".join(_duckdb_literal(c) for c in base)

    # Fallback
    base = present(["year", "week"]) or available_cols[:1]
    return ", ".join(_duckdb_literal(c) for c in base)


def _upsert_parquet_via_duckdb(out_path: Path, new_df: pd.DataFrame, keys: list[str], kind: str) -> int:
    """
    Upsert Parquet using DuckDB:
      - Align columns between old/new
      - For 'player', create __key_player = COALESCE(TRIM(nfl_player_id), TRIM(yahoo_player_id))
      - UNION ALL (old + new) with is_new flag
      - Window-dedup so new rows overwrite old on the chosen partition keys
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    new_df = _sanitize_for_duckdb(new_df)
    con.register("new_df", new_df)

    select_new = _duckdb_cast_list(new_df, kind)
    con.execute(f"CREATE TEMP TABLE _new_raw AS SELECT {select_new} FROM new_df")

    if kind == "player":
        con.execute("""
            CREATE TEMP TABLE _new AS
            SELECT
                *,
                COALESCE(NULLIF(TRIM("nfl_player_id"), ''), NULLIF(TRIM("yahoo_player_id"), '')) AS "__key_player"
            FROM _new_raw
        """)
    else:
        con.execute('CREATE TEMP TABLE _new AS SELECT * FROM _new_raw')

    out_str = str(out_path).replace("\\", "/")

    if out_path.exists():
        con.execute(f"CREATE TEMP TABLE _old_raw AS SELECT * FROM read_parquet('{out_str}')")
        if kind == "player":
            cols_old_chk = _duckdb_cols(con, "_old_raw")
            if "__key_player" in cols_old_chk:
                con.execute('CREATE TEMP TABLE _old AS SELECT * FROM _old_raw')
            else:
                con.execute("""
                    CREATE TEMP TABLE _old AS
                    SELECT
                        *,
                        COALESCE(NULLIF(TRIM("nfl_player_id"), ''), NULLIF(TRIM("yahoo_player_id"), '')) AS "__key_player"
                    FROM _old_raw
                """)
        else:
            con.execute('CREATE TEMP TABLE _old AS SELECT * FROM _old_raw')

        cols_new = _duckdb_cols(con, "_new")
        cols_old = _duckdb_cols(con, "_old")
        all_cols = list(dict.fromkeys(cols_old + cols_new))

        sel_old = ", ".join([f'"{c}"' if c in cols_old else f'NULL AS "{c}"' for c in all_cols])
        sel_new = ", ".join([f'"{c}"' if c in cols_new else f'NULL AS "{c}"' for c in all_cols])

        partition_by = _partition_clause_for_kind(kind, all_cols)

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
        con.execute("""
            CREATE TEMP TABLE _merged AS
            SELECT *, 1 AS is_new FROM _new
        """)

    exclude_cols = "is_new"
    if kind == "player":
        exclude_cols += ", __key_player"

    con.execute(f"""
        COPY (SELECT * EXCLUDE({exclude_cols}) FROM _merged)
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
    # 1) Figure out latest completed NFL week (NYC)
    year, week = infer_latest_completed_week()
    log(f"Resolved season/week -> {year} / {week}")

    # ========================================================================
    # PHASE 1: Run the four base producers (CLI first, fallback to stdin if they prompt)
    # ========================================================================
    log("=== PHASE 1: Base data collection ===")
    runs = [
        (SCHEDULE_SCRIPT,    "Season schedules",     "year_only"),
        (MATCHUP_SCRIPT,     "Weekly matchup data",  "year_week"),
        (TRANSACTION_SCRIPT, "Transactions",         "year_week"),
        (MERGE_SCRIPT,       "Yahoo/NFL merge",      "year_week"),
    ]
    for script, label, mode in runs:
        run_script_interactive(script, label, mode, year, week)

    # 2) Locate freshest sources (no matter the filename)
    log("=== PHASE 1: Append into canonical tables ===")
    loc_map = {
        "schedule":     _locate_latest_parquet("schedule", year, 0),
        "matchup":      _locate_latest_parquet("matchup", year, week),
        "transactions": _locate_latest_parquet("transactions", year, 0),
        "player":       _locate_latest_parquet("player", year, week),
    }

    for kind, src in loc_map.items():
        if not src or not src.exists():
            log(f"WARN: No fresh {kind} source located in {SOURCE_DIRS[kind]}")
            continue

        try:
            df_src = pd.read_parquet(src)
        except Exception as e:
            log(f"WARN: Could not read {kind} source {src}: {e}")
            continue

        # Narrow to target scope where it makes sense
        if kind in ("matchup","player") and all(c in df_src.columns for c in ("year","week")):
            df_src = _filter_year_week(df_src, year, week)

        if df_src.empty:
            log(f"WARN: {kind} source found but empty after filter: {src.name}")
            continue

        # **CRITICAL: Check if week data is complete before upserting**
        # This prevents overwriting complete data with partial Thursday-night-only data
        if kind in ("matchup", "player"):
            if not _is_week_complete(df_src, year, week, kind):
                log(f"SKIP: {kind} data for week {week} appears INCOMPLETE (likely only Thursday night)")
                log(f"      Will NOT overwrite existing complete data. Run again after Sunday games.")
                continue

        rows_used = len(df_src)
        log(f"Selected {kind} -> {src.name} | rows_used={rows_used} (filtered from {rows_used})")

        # DuckDB upsert (overwrites-on-keys using windowed dedup)
        total_rows = _upsert_parquet_via_duckdb(CANONICAL[kind], df_src, DEDUP_KEYS[kind], kind)
        log(f"Updated {CANONICAL[kind].name}: {rows_used} new rows merged; total {total_rows}")

    # ========================================================================
    # PHASE 2: Run post-processing scripts and MotherDuck upload
    # ========================================================================
    log("=== PHASE 2: Post-processing and upload ===")
    for script, label in RUNS_POST:
        run_script_simple(script, label, year, week)

    log("=== ALL TASKS COMPLETED ===")


if __name__ == "__main__":
    main()
