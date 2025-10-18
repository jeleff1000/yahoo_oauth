#!/usr/bin/env python3
"""
MotherDuck Upload Script

Uploads all parquet files to MotherDuck, creating a database named after the league.

Usage:
    python motherduck_upload.py

Environment Variables:
    MOTHERDUCK_TOKEN - Your MotherDuck API token
    LEAGUE_NAME - Name of the league (used for database name)
    LEAGUE_KEY - Yahoo league key
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import duckdb
import re
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR
OAUTH_DIR = ROOT_DIR.parent / "fantasy_football_data_scripts" / "player_stats" / "oauth"

# Load MotherDuck token from environment or Streamlit secrets
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")

# Try to load from Streamlit secrets if available
if not MOTHERDUCK_TOKEN:
    try:
        import streamlit as st
        MOTHERDUCK_TOKEN = st.secrets.get("MOTHERDUCK_TOKEN")
    except:
        pass

# Prepare a local log file for upload runs
LOG_DIR = ROOT_DIR / "upload_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"motherduck_upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str) -> None:
    """Simple logging with timestamp (prints and appends to a log file)"""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as lf:
            lf.write(line + "\n")
    except Exception:
        # Best-effort logging; don't fail the upload because of log write issues
        pass


def sanitize_db_name(name: str) -> str:
    """
    Sanitize league name to be a valid DuckDB database name.

    Rules:
    - Lowercase
    - Replace spaces and special chars with underscores
    - Remove consecutive underscores
    - Max 63 chars (PostgreSQL limit)
    """
    # Convert to lowercase
    name = name.lower()

    # Replace spaces and special chars with underscores
    name = re.sub(r'[^a-z0-9_]', '_', name)

    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)

    # Remove leading/trailing underscores
    name = name.strip('_')

    # Truncate to 63 chars
    if len(name) > 63:
        name = name[:63].rstrip('_')

    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = 'league_' + name

    return name or 'fantasy_league'


def get_league_info() -> tuple[str, str]:
    """
    Get league name and key from environment or OAuth file.

    Returns:
        (league_name, league_key)
    """
    # First try environment variables (set by main.py)
    league_name = os.environ.get("LEAGUE_NAME")
    league_key = os.environ.get("LEAGUE_KEY")

    # If a league name is provided in the environment, prefer it even if LEAGUE_KEY is missing
    if league_name:
        return league_name, league_key or "unknown"

    # Fallback: try to read from OAuth file
    oauth_file = OAUTH_DIR / "Oauth.json"
    if oauth_file.exists():
        import json
        try:
            with open(oauth_file, 'r') as f:
                data = json.load(f)

            league_info = data.get("league_info", {})
            league_name = league_info.get("name", "Unknown League")
            league_key = league_info.get("league_key", "unknown")

            return league_name, league_key
        except Exception as e:
            log(f"Warning: Could not read league info from OAuth file: {e}")

    # Last resort defaults
    return "Unknown League", "unknown"


def upload_to_motherduck():
    """Upload all parquet files to MotherDuck"""

    log(f"Looking for parquet files under: {DATA_DIR} (and subdirectories)")

    # Search recursively for parquet files inside DATA_DIR
    parquet_files = list(DATA_DIR.rglob("*.parquet"))

    # If none found in DATA_DIR, also attempt to search one level up (repo root) and the entire repo as a last resort
    if not parquet_files:
        repo_root = ROOT_DIR.parent
        log(f"No parquet files found in {DATA_DIR}. Searching repo root: {repo_root} (recursive)")
        parquet_files = list(repo_root.rglob("*.parquet"))

    if not parquet_files:
        log("No parquet files found in repository. Nothing to upload.")
        if not MOTHERDUCK_TOKEN:
            log("ERROR: MOTHERDUCK_TOKEN not set. Cannot upload to MotherDuck.")
            log("Set MOTHERDUCK_TOKEN environment variable or add to Streamlit secrets.")
            return False
        else:
            # Nothing to upload but token is set — return True to indicate upload step didn't fail, but nothing happened
            return True

    log(f"Found {len(parquet_files)} parquet file(s) to consider for upload:")
    for p in parquet_files:
        log(f"  - {p}")

    if not MOTHERDUCK_TOKEN:
        log("ERROR: MOTHERDUCK_TOKEN not set. Cannot upload to MotherDuck.")
        log("Set MOTHERDUCK_TOKEN environment variable or add to Streamlit secrets.")
        return False

    # Get league info
    league_name, league_key = get_league_info()
    db_name = sanitize_db_name(league_name)

    log("=" * 80)
    log("MOTHERDUCK UPLOAD")
    log("=" * 80)
    log(f"League Name: {league_name}")
    log(f"League Key: {league_key}")
    log(f"Database Name: {db_name}")
    log("")

    try:
        # Connect to MotherDuck
        connection_string = f"md:?motherduck_token={MOTHERDUCK_TOKEN}"
        log("Connecting to MotherDuck...")
        con = duckdb.connect(connection_string)

        # Create database if it doesn't exist
        log(f"Creating/using database: {db_name}")
        con.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        con.execute(f"USE {db_name}")

        log(f"Preparing to upload {len(parquet_files)} file(s)")
        log("")

        # Upload each parquet file as a table
        for parquet_file in sorted(parquet_files):
            table_name = parquet_file.stem  # filename without extension
            parquet_path = str(parquet_file).replace("\\", "/")

            log(f"Uploading {parquet_file.name} -> {table_name}")

            try:
                # Drop existing table if it exists
                con.execute(f"DROP TABLE IF EXISTS {table_name}")

                # Create table from parquet
                con.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_parquet('{parquet_path}')
                """)

                # Get row count
                count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                log(f"  ✓ Uploaded {count:,} rows to {table_name}")

            except Exception as e:
                log(f"  ✗ Error uploading {parquet_file.name}: {e}")

        log("")
        log("=" * 80)
        log("UPLOAD COMPLETE")
        log("=" * 80)
        log(f"Database: {db_name}")
        log(f"Access at: https://app.motherduck.com/")
        log("")

        # Show tables
        tables = con.execute("SHOW TABLES").fetchall()
        if tables:
            log("Tables created:")
            for table in tables:
                table_name = table[0]
                count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                log(f"  - {table_name}: {count:,} rows")

        con.close()
        return True

    except Exception as e:
        log(f"ERROR: Failed to upload to MotherDuck: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    success = upload_to_motherduck()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
