#!/usr/bin/env python3
"""
CLI wrapper for MotherDuck upload.

This is a thin wrapper around streamlit_helpers.database.upload_to_motherduck
for use by GitHub Actions workflows.

Usage: python motherduck_upload.py <db_name> [data_dir]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from streamlit_helpers
sys.path.insert(0, str(Path(__file__).parents[1]))

from streamlit_helpers.database import upload_to_motherduck, collect_parquet_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python motherduck_upload.py <db_name> [data_dir]", file=sys.stderr)
        sys.exit(2)

    db_name = sys.argv[1]
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    token = os.getenv("MOTHERDUCK_TOKEN", "")

    # Collect parquet files from the data directory
    files = collect_parquet_files(base_dir=data_dir)

    if not files:
        print(f"No parquet files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} parquet files to upload")
    for f in files:
        print(f"  - {f.name}")

    # Upload to MotherDuck
    uploaded = upload_to_motherduck(files, db_name=db_name, token=token)

    if uploaded:
        print(f"\nUploaded {len(uploaded)} tables:")
        for tbl, cnt in uploaded:
            print(f"  - {tbl}: {cnt:,} rows")
    else:
        print("No tables uploaded", file=sys.stderr)
        sys.exit(1)
