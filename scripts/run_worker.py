"""
Background Worker for Fantasy Football Imports using Modal.com

Modal is a serverless Python platform perfect for this use case.
It handles:
- Auto-scaling
- Job queuing
- Long-running tasks
- No infrastructure management

Setup:
1. pip install modal
2. modal token new
3. modal deploy modal_worker.py
4. Set JOB_QUEUE_URL in Streamlit secrets to your Modal endpoint
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("fantasy-football-importer")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "duckdb",
        "yahoo-fantasy-api",
        "yahoo-oauth",
        "requests",
        "pyarrow",
        "numpy"
    )
)

# Persistent volume for temporary data
volume = modal.Volume.from_name("fantasy-data-cache", create_if_missing=True)


@app.function(
    image=image,
    timeout=3600,  # 1 hour timeout
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("motherduck-token")]
)
def process_import_job(job_data: dict) -> dict:
    """
    Process a league import job.
    This runs in an isolated Modal container with plenty of resources.
    """
    import json
    import tempfile
    import subprocess
    import sys
    from datetime import datetime

    job_id = job_data['job_id']
    league_key = job_data['league_key']
    league_name = job_data['league_name']
    season = job_data['season']
    oauth_token = job_data['oauth_token']
    motherduck_token = job_data.get('motherduck_token')

    print(f"[{job_id}] Starting import for {league_name} ({season})")

    # Create temp directory
    temp_dir = Path(f"/cache/{job_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    oauth_dir = temp_dir / "oauth"
    oauth_dir.mkdir(exist_ok=True)
    data_dir = temp_dir / "fantasy_football_data"
    data_dir.mkdir(exist_ok=True)

    # Save OAuth credentials
    oauth_file = oauth_dir / "Oauth.json"
    with open(oauth_file, "w") as f:
        json.dump({"token_data": oauth_token}, f)

    # Set environment variables
    env = os.environ.copy()
    env["OAUTH_PATH"] = str(oauth_file)
    env["EXPORT_DATA_DIR"] = str(data_dir)
    env["MOTHERDUCK_TOKEN"] = motherduck_token or ""
    env["AUTO_CONFIRM"] = "1"
    env["LEAGUE_NAME"] = league_name
    env["LEAGUE_KEY"] = league_key
    env["LEAGUE_SEASON"] = season

    try:
        # Here you would run your producer scripts
        # For now, this is a placeholder showing the structure

        # Example: Run schedule script
        print(f"[{job_id}] Fetching schedule data...")
        # result = subprocess.run(
        #     [sys.executable, "path/to/season_schedules.py", "--year", "0", "--week", "0"],
        #     env=env,
        #     capture_output=True,
        #     text=True,
        #     timeout=600
        # )

        # Similarly for matchup, transactions, player scripts...

        # Upload to MotherDuck
        if motherduck_token:
            print(f"[{job_id}] Uploading to MotherDuck...")
            import duckdb

            os.environ["MOTHERDUCK_TOKEN"] = motherduck_token
            con = duckdb.connect("md:")

            # Create database per league
            db_name = f"{league_name}_{season}".lower().replace(' ', '_').replace('-', '_')
            con.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            con.execute(f"USE {db_name}")

            # Upload parquet files
            parquet_files = list(data_dir.glob("*.parquet"))
            for pf in parquet_files:
                table_name = pf.stem.lower()
                print(f"[{job_id}] Uploading {table_name}...")
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{pf}')")

            con.close()

        print(f"[{job_id}] Import complete!")

        return {
            "status": "complete",
            "job_id": job_id,
            "message": "Import successful",
            "completed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        }

    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.function()
@modal.web_endpoint(method="POST")
def submit_job(job_data: dict):
    """
    Web endpoint to receive job submissions from Streamlit app.
    This queues the job and returns immediately.
    """
    # Spawn the import job asynchronously
    process_import_job.spawn(job_data)

    return {
        "status": "queued",
        "job_id": job_data['job_id'],
        "message": "Import job queued successfully"
    }


@app.function()
@modal.web_endpoint(method="GET")
def check_status(job_id: str):
    """
    Check the status of a job.
    In production, you'd query a database/cache here.
    """
    # Placeholder - you'd implement actual status tracking
    return {
        "status": "running",
        "job_id": job_id,
        "progress": 50,
        "message": "Processing matchup data..."
    }