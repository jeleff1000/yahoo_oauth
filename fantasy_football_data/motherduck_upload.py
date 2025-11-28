#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable

import duckdb

def _slug(s: str, prefix_if_digit: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    if re.match(r"^\d", x):
        x = f"{prefix_if_digit}_{x}"
    return x[:63]

def upload_parquets_to_motherduck(
    data_dir: Path,
    db_name: str,
    schema: str = "public",
    token: str | None = None,
    status_cb: Callable[[str], None] | None = None
) -> list[tuple[str, int]]:
    if token:
        os.environ["MOTHERDUCK_TOKEN"] = token
    db_slug = _slug(db_name, "l")

    # Connect to MotherDuck first (without specific database)
    con = duckdb.connect("md:")
    # Create the database if it doesn't exist
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db_slug}")
    # Switch to the database
    con.execute(f"USE {db_slug}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    results: list[tuple[str, int]] = []
    for pf in sorted(Path(data_dir).glob("*.parquet")):
        tbl = _slug(pf.stem, "t")
        if status_cb:
            status_cb(f"Uploading {pf.name} → {db_slug}.{schema}.{tbl}")
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{tbl} AS SELECT * FROM read_parquet(?)", [str(pf)])
        cnt = con.execute(f"SELECT COUNT(*) FROM {schema}.{tbl}").fetchone()[0]
        results.append((tbl, int(cnt)))
        if status_cb:
            status_cb(f"✓ {tbl}: {cnt} rows")
    return results

if __name__ == "__main__":
    # Minimal CLI: python motherduck_upload.py <db_name> [data_dir]
    if len(sys.argv) < 2:
        print("Usage: python motherduck_upload.py <db_name> [data_dir]", file=sys.stderr)
        sys.exit(2)
    db = sys.argv[1]
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parents[1] / "fantasy_football_data"
    tok = os.getenv("MOTHERDUCK_TOKEN", "")
    def echo(m): print(m, flush=True)
    uploaded = upload_parquets_to_motherduck(data, db_name=db, token=tok, status_cb=echo)
    print(f"Uploaded {len(uploaded)} tables.")
