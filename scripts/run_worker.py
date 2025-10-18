#!/usr/bin/env python3
"""Worker entrypoint: read OAuth token from MotherDuck, write local oauth/Oauth.json, run initial import,
then upload canonical parquet files to MotherDuck and update ops.import_status.

This script is designed to be run from CI (GitHub Actions) and expects MOTHERDUCK_TOKEN to be set as an env var.
Optionally pass JOB_ID via env to tie work to an ops.import_status row.
"""
import os
import sys
import json
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

try:
    import duckdb
except Exception as e:
    print("Missing dependency: duckdb", file=sys.stderr)
    raise

ROOT = Path(__file__).parent.parent
OAUTH_DIR = ROOT / "oauth"
DATA_DIR = ROOT / "fantasy_football_data"
SCRIPTS_DIR = ROOT / "fantasy_football_data_scripts"
INITIAL_IMPORT = SCRIPTS_DIR / "initial_import.py"


def _slug(s: str, lead_prefix: str) -> str:
    import re
    x = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip().lower()).strip("_")
    if not x:
        x = "db"
    if re.match(r"^\d", x):
        x = f"{lead_prefix}_{x}"
    return x[:63]


def write_oauth_file(token_json: dict):
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    oauth_path = OAUTH_DIR / "Oauth.json"
    with open(oauth_path, 'w', encoding='utf-8') as f:
        json.dump(token_json, f, indent=2, default=str)
    print(f"Wrote OAuth token to: {oauth_path}")
    return oauth_path


def pick_token(con, league_key=None):
    # Try to find a token matching league_key, otherwise use latest
    q = "SELECT id, league_key, token_json, updated_at FROM secrets.yahoo_oauth_tokens"
    if league_key:
        q += " WHERE league_key = ?"
        rows = con.execute(q, [league_key]).fetchall()
    else:
        rows = con.execute(q).fetchall()

    if not rows:
        return None
    # rows: pick last by updated_at
    rows_sorted = sorted(rows, key=lambda r: r[3] or datetime.min)
    return rows_sorted[-1][2]  # token_json


def upload_parquets_to_md(parquet_files, db_name, token):
    if not parquet_files:
        return []
    if token:
        os.environ['MOTHERDUCK_TOKEN'] = token
    db = _slug(db_name, 'l')
    con = duckdb.connect('md:')
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    con.execute(f"USE {db}")
    con.execute("CREATE SCHEMA IF NOT EXISTS public")

    aliases = {
        'players_by_year': 'player',
        'yahoo_player_stats_multi_year_all_weeks': 'player',
        'matchups': 'matchup',
        'schedules': 'schedule',
        'transaction': 'transactions',
    }

    results = []
    for pf in parquet_files:
        stem = pf.stem.lower()
        stem = aliases.get(stem, stem)
        tbl = _slug(stem, 't')
        try:
            print(f"Uploading {pf} to {db}.public.{tbl}...")
            con.execute(f"CREATE OR REPLACE TABLE public.{tbl} AS SELECT * FROM read_parquet(?)", [str(pf)])
            cnt = con.execute(f"SELECT COUNT(*) FROM public.{tbl}").fetchone()[0]
            results.append((tbl, int(cnt)))
            print(f"Uploaded {tbl}: {cnt} rows")
        except Exception as e:
            print(f"Failed to upload {pf}: {e}", file=sys.stderr)
    con.close()
    return results


def main():
    MOTHERDUCK_TOKEN = os.getenv('MOTHERDUCK_TOKEN')
    if not MOTHERDUCK_TOKEN:
        print('MOTHERDUCK_TOKEN is required in env', file=sys.stderr)
        sys.exit(2)

    JOB_ID = os.getenv('JOB_ID')

    con = duckdb.connect('md:')

    league_key = None
    league_name = None
    season = None

    if JOB_ID:
        try:
            row = con.execute('SELECT league_key, league_name, season FROM ops.import_status WHERE job_id = ?', [JOB_ID]).fetchone()
            if row:
                league_key, league_name, season = row
        except Exception:
            pass

    token_json = None
    try:
        token_json_text = pick_token(con, league_key)
        if token_json_text:
            try:
                token_json = json.loads(token_json_text) if isinstance(token_json_text, str) else token_json_text
            except Exception:
                token_json = token_json_text
    except Exception as e:
        print('Error selecting token from MD:', e, file=sys.stderr)

    if not token_json:
        print('No OAuth token found in MotherDuck. Exiting.', file=sys.stderr)
        if JOB_ID:
            con.execute('UPDATE ops.import_status SET status = ?, updated_at = ? WHERE job_id = ?', ['failed', datetime.now(), JOB_ID])
        sys.exit(3)

    # write local oauth file for existing scripts
    write_oauth_file(token_json)

    # Run initial_import
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['AUTO_CONFIRM'] = '1'
    env['EXPORT_DATA_DIR'] = str(DATA_DIR.resolve())
    oauth_path = OAUTH_DIR / 'Oauth.json'
    if oauth_path.exists():
        env['OAUTH_PATH'] = str(oauth_path.resolve())

    if JOB_ID:
        try:
            con.execute('UPDATE ops.import_status SET status = ?, updated_at = ? WHERE job_id = ?', ['running', datetime.now(), JOB_ID])
        except Exception:
            pass

    rc = 1
    try:
        print('Starting initial import...')
        cmd = [sys.executable, str(INITIAL_IMPORT)]
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f'initial_import failed: {proc.returncode}')
        rc = 0
    except Exception as e:
        print('Import failed:', e, file=sys.stderr)
        traceback.print_exc()
        rc = 2

    # After import, upload parquet files to MD
    parquet_files = []
    if DATA_DIR.exists():
        for p in list(DATA_DIR.glob('*.parquet')) + list((DATA_DIR / 'matchup_data').glob('*.parquet') if (DATA_DIR / 'matchup_data').exists() else []):
            if p.exists():
                parquet_files.append(p)

    # determine db name
    db_name = None
    if league_name and season:
        db_name = f"{league_name}_{season}"
    elif league_name:
        db_name = league_name
    else:
        db_name = 'league'

    try:
        uploaded = upload_parquets_to_md(parquet_files, db_name, MOTHERDUCK_TOKEN)
        print('Upload summary:', uploaded)
    except Exception as e:
        print('Upload failed:', e, file=sys.stderr)

    # Update job status
    if JOB_ID:
        try:
            status = 'success' if rc == 0 else 'failed'
            con.execute('UPDATE ops.import_status SET status = ?, updated_at = ? WHERE job_id = ?', [status, datetime.now(), JOB_ID])
        except Exception:
            pass

    con.close()
    sys.exit(rc)

if __name__ == '__main__':
    main()

