#!/usr/bin/env python3
"""
MotherDuck Database Operations

This module handles all MotherDuck-related operations including:
- Database discovery and validation
- Import job management
- Progress tracking
- File collection and upload
"""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

# Get MotherDuck token
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")

# Try to get from streamlit secrets as fallback
try:
    import streamlit as st
    if not MOTHERDUCK_TOKEN:
        MOTHERDUCK_TOKEN = st.secrets.get("MOTHERDUCK_TOKEN", "")
except (ImportError, AttributeError):
    pass


def format_league_display_name(db_name: str) -> str:
    """
    Format league database name for display.
    Strips 'l_' prefix that was added for digit-starting names.

    Example: 'l_5townsfootball' -> '5townsfootball'
    """
    if not db_name:
        return db_name
    # Strip the 'l_' prefix if it was added because name started with a digit
    if db_name.startswith("l_") and len(db_name) > 2 and db_name[2].isdigit():
        return db_name[2:]
    return db_name


def _get_motherduck_connection(token: str = None):
    """
    Get a MotherDuck connection using token in connection string (not env var).
    This avoids global state mutation that could affect concurrent requests.
    """
    token = token or MOTHERDUCK_TOKEN
    if not token:
        raise ValueError("No MotherDuck token available")
    # Use connection string with token to avoid setting os.environ
    return duckdb.connect(f"md:?motherduck_token={token}")


def get_existing_league_databases() -> list[str]:
    """
    Discover existing league databases in MotherDuck.
    Returns a sorted list of database names (excluding system databases).
    """
    if not MOTHERDUCK_TOKEN:
        return []

    try:
        con = _get_motherduck_connection()

        # Query all databases
        result = con.execute("SHOW DATABASES").fetchall()
        con.close()

        # Filter out system databases and return sorted list
        system_dbs = {"my_db", "sample_data", "secrets", "ops", "information_schema", "md_information_schema"}
        league_dbs = [
            row[0] for row in result
            if row[0].lower() not in system_dbs
            and not row[0].startswith("_")
            and not row[0].startswith("md_")  # Exclude MotherDuck system databases
        ]

        return sorted(league_dbs, key=str.lower)
    except Exception as e:
        return []


def validate_league_database(db_name: str) -> bool:
    """
    Validate that a database has the expected league tables (matchup, player, etc.).
    """
    if not MOTHERDUCK_TOKEN or not db_name:
        return False

    try:
        con = _get_motherduck_connection()

        # Check if required tables exist
        result = con.execute(f"SHOW TABLES IN {db_name}").fetchall()
        con.close()

        tables = {row[0].lower() for row in result}
        required_tables = {"matchup", "player"}

        return required_tables.issubset(tables)
    except Exception:
        return False


def create_import_job_in_motherduck(league_info: dict) -> Optional[str]:
    """
    Create an import job record in MotherDuck ops.import_status table.
    Returns the job_id if successful, None otherwise.
    """
    if not MOTHERDUCK_TOKEN:
        return None

    try:
        con = _get_motherduck_connection()

        # Create table if it doesn't exist
        con.execute("""
            CREATE TABLE IF NOT EXISTS ops.import_status (
                job_id TEXT, league_key TEXT, league_name TEXT, season TEXT,
                status TEXT, created_at TIMESTAMP, updated_at TIMESTAMP
            )
        """)

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        con.execute("INSERT INTO ops.import_status VALUES (?,?,?,?,?,?,?)",
                    [job_id, league_info.get("league_key"), league_info.get("name"),
                     str(league_info.get("season", "")), "queued", now, now])
        con.close()
        return job_id
    except Exception:
        return None


def get_job_status(job_id: str) -> dict:
    """Get the status of an import job from MotherDuck."""
    if not MOTHERDUCK_TOKEN or not job_id:
        return {"status": "error"}

    try:
        con = _get_motherduck_connection()
        result = con.execute(
            "SELECT status, updated_at FROM ops.import_status WHERE job_id = ?",
            [job_id]
        ).fetchone()
        con.close()
        if result:
            return {"status": result[0], "updated_at": result[1]}
        return {"status": "not_found"}
    except Exception:
        return {"status": "error"}


def get_motherduck_progress(job_id: str) -> Optional[dict]:
    """
    Query MotherDuck for import progress.
    Returns progress dict or None if not available.
    """
    if not MOTHERDUCK_TOKEN or not job_id:
        return None

    try:
        con = _get_motherduck_connection()

        result = con.execute("""
            SELECT job_id, league_name, phase, stage, stage_detail,
                   current_step, total_steps, overall_pct, status,
                   error_message, started_at, updated_at
            FROM ops.import_progress
            WHERE job_id = ?
        """, [job_id]).fetchone()

        con.close()

        if result:
            return {
                "job_id": result[0],
                "league_name": result[1],
                "phase": result[2],
                "stage": result[3],
                "stage_detail": result[4],
                "current_step": result[5],
                "total_steps": result[6],
                "overall_pct": result[7] or 0,
                "status": result[8],
                "error_message": result[9],
                "started_at": result[10],
                "updated_at": result[11],
            }
        return None
    except Exception:
        # Table might not exist yet
        return None


def _slug(s: str, lead_prefix: str) -> str:
    """Create a valid database/table name from a string"""
    x = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip().lower()).strip("_")
    if not x:
        x = "db"
    if re.match(r"^\d", x):
        x = f"{lead_prefix}_{x}"
    return x[:63]


def collect_parquet_files(base_dir: Optional[Path] = None, data_dir: Optional[Path] = None) -> list[Path]:
    """
    Collect parquet files in priority order:
    1. Canonical files at base_dir root (schedule.parquet, matchup.parquet, etc.)
    2. Any other parquet files in base_dir subdirectories

    Args:
        base_dir: Directory to collect from.
        data_dir: Fallback directory if base_dir is None.
    """
    # Determine which directory to use
    if base_dir is None:
        base_dir = data_dir

    if base_dir is None:
        return []

    base_dir = Path(base_dir)
    files = []
    seen = set()

    # Priority 1: Canonical files at root
    canonical_names = ["schedule.parquet", "matchup.parquet", "transactions.parquet",
                       "player.parquet", "players_by_year.parquet", "draft.parquet"]

    for name in canonical_names:
        p = base_dir / name
        if p.exists() and p.is_file():
            files.append(p)
            seen.add(p.resolve())

    # Priority 2: Subdirectories (schedule_data, matchup_data, etc.)
    if base_dir.exists():
        for subdir in ["schedule_data", "matchup_data", "transaction_data", "player_data", "draft_data"]:
            sub_path = base_dir / subdir
            if sub_path.exists() and sub_path.is_dir():
                for p in sub_path.glob("*.parquet"):
                    resolved = p.resolve()
                    if resolved not in seen:
                        files.append(p)
                        seen.add(resolved)

    # Priority 3: Any other parquet files in base_dir (non-recursive, to avoid noise)
    if base_dir.exists():
        for p in base_dir.glob("*.parquet"):
            resolved = p.resolve()
            if resolved not in seen:
                files.append(p)
                seen.add(resolved)

    return files


def upload_to_motherduck(files: list[Path], db_name: str, token: str = None) -> list[tuple[str, int]]:
    """
    Upload parquet files directly to MotherDuck.

    Args:
        files: List of parquet file paths to upload.
        db_name: Database name to create/use.
        token: MotherDuck token (uses env var if not provided).

    Returns:
        List of tuples (table_name, row_count) for successful uploads.
    """
    if not files:
        return []

    token = token or MOTHERDUCK_TOKEN
    if not token:
        print("No MotherDuck token available")
        return []

    db = _slug(db_name, "l")

    con = _get_motherduck_connection(token)
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    con.execute(f"USE {db}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS public")

    # Table name mapping (handle common aliases)
    aliases = {
        "players_by_year": "player",
        "yahoo_player_stats_multi_year_all_weeks": "player",
        "matchups": "matchup",
        "schedules": "schedule",
        "transaction": "transactions",
    }

    results = []
    for pf in files:
        stem = pf.stem.lower()
        stem = aliases.get(stem, stem)
        tbl = _slug(stem, "t")

        try:
            con.execute(f"CREATE OR REPLACE TABLE public.{tbl} AS SELECT * FROM read_parquet(?)", [str(pf)])
            cnt = con.execute(f"SELECT COUNT(*) FROM public.{tbl}").fetchone()[0]
            results.append((tbl, int(cnt)))
        except Exception as e:
            print(f"Failed to upload {pf.name}: {e}")

    con.close()
    return results


def upload_external_data_to_staging(
    external_data: dict,
    db_name: str,
    token: str = None
) -> dict:
    """
    Upload external data (from Streamlit file uploads) directly to MotherDuck staging tables.

    This bypasses GitHub Actions size limits by uploading data directly to MotherDuck
    instead of passing it through the workflow dispatch API.

    Args:
        external_data: Dict from render_external_data_ui() with structure:
            {
                "matchup": [{"filename": "...", "data": [...], "columns": [...], "row_count": N}, ...],
                "player": [...],
                ...
            }
        db_name: Database name (league name).
        token: MotherDuck token (uses env var if not provided).

    Returns:
        Dict with upload results:
            {
                "success": True/False,
                "tables_uploaded": [("staging_matchup", 500), ...],
                "error": "..." (if failed)
            }
    """
    import pandas as pd

    if not external_data:
        return {"success": True, "tables_uploaded": [], "message": "No external data to upload"}

    token = token or MOTHERDUCK_TOKEN
    if not token:
        return {"success": False, "error": "MotherDuck token not configured"}

    db = _slug(db_name, "l")

    try:
        con = _get_motherduck_connection(token)
        con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        con.execute(f"USE {db}")
        con.execute(f"CREATE SCHEMA IF NOT EXISTS staging")

        results = []

        for data_type, files in external_data.items():
            if not files:
                continue

            # Combine all files of this type into one DataFrame
            all_records = []
            for file_info in files:
                records = file_info.get("data", [])
                all_records.extend(records)

            if not all_records:
                continue

            # Create DataFrame
            df = pd.DataFrame(all_records)

            # Upload to staging table
            staging_table = f"staging_{_slug(data_type, 't')}"

            # Register DataFrame and create table
            con.register("temp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE staging.{staging_table} AS SELECT * FROM temp_df")
            con.unregister("temp_df")

            # Get row count
            cnt = con.execute(f"SELECT COUNT(*) FROM staging.{staging_table}").fetchone()[0]
            results.append((staging_table, int(cnt)))
            print(f"Uploaded {cnt} rows to {db}.staging.{staging_table}")

        con.close()

        return {
            "success": True,
            "tables_uploaded": results,
            "database": db,
            "message": f"Uploaded {len(results)} staging tables to {db}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def check_staging_tables(db_name: str, token: str = None) -> list[str]:
    """
    Check which staging tables exist for a database.

    Returns list of staging table names.
    """
    token = token or MOTHERDUCK_TOKEN
    if not token:
        return []

    try:
        db = _slug(db_name, "l")
        con = _get_motherduck_connection(token)

        # Check if staging schema exists
        result = con.execute(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_catalog = '{db}'
            AND table_schema = 'staging'
        """).fetchall()

        con.close()
        return [row[0] for row in result]

    except Exception:
        return []
