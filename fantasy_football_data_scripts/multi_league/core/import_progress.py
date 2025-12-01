#!/usr/bin/env python3
"""
Import Progress Tracking for MotherDuck

This module provides functions to update and query import progress in MotherDuck.
Used by initial_import_v2.py to report progress that Streamlit can display.

Schema:
    ops.import_progress (
        job_id TEXT,           -- Unique job identifier (from workflow trigger)
        league_name TEXT,      -- League name for display
        phase TEXT,            -- Current phase: settings, fetchers, merges, transformations
        stage TEXT,            -- Current stage within phase (e.g., "yahoo_player_data")
        stage_detail TEXT,     -- Additional detail (e.g., "Fetching 2020...")
        current_step INT,      -- Current step number
        total_steps INT,       -- Total steps in current phase
        overall_pct FLOAT,     -- Overall progress percentage (0-100)
        status TEXT,           -- running, completed, failed
        error_message TEXT,    -- Error message if failed
        started_at TIMESTAMP,  -- When import started
        updated_at TIMESTAMP   -- Last update time
    )
"""

import os
from datetime import datetime, timezone
from typing import Optional
import json


def get_motherduck_token() -> Optional[str]:
    """Get MotherDuck token from environment."""
    return os.environ.get("MOTHERDUCK_TOKEN")


def init_progress_table():
    """
    Initialize the progress table in MotherDuck if it doesn't exist.
    Called once at the start of an import.
    """
    token = get_motherduck_token()
    if not token:
        return False

    try:
        import duckdb
        con = duckdb.connect("md:")

        # Create ops schema if needed
        con.execute("CREATE SCHEMA IF NOT EXISTS ops")

        # Create progress table
        con.execute("""
            CREATE TABLE IF NOT EXISTS ops.import_progress (
                job_id TEXT,
                league_name TEXT,
                phase TEXT,
                stage TEXT,
                stage_detail TEXT,
                current_step INTEGER,
                total_steps INTEGER,
                overall_pct DOUBLE,
                status TEXT,
                error_message TEXT,
                started_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (job_id)
            )
        """)
        con.close()
        return True
    except Exception as e:
        print(f"[PROGRESS] Warning: Could not init progress table: {e}")
        return False


def update_progress(
    job_id: str,
    league_name: str,
    phase: str,
    stage: str,
    stage_detail: str = "",
    current_step: int = 0,
    total_steps: int = 0,
    overall_pct: float = 0.0,
    status: str = "running",
    error_message: str = None
):
    """
    Update import progress in MotherDuck.

    Args:
        job_id: Unique job identifier
        league_name: League name for display
        phase: Current phase (settings, fetchers, merges, transformations)
        stage: Current stage within phase
        stage_detail: Additional detail (e.g., year being processed)
        current_step: Current step number within phase
        total_steps: Total steps in current phase
        overall_pct: Overall progress percentage (0-100)
        status: running, completed, failed
        error_message: Error message if failed
    """
    token = get_motherduck_token()
    if not token or not job_id:
        return False

    try:
        import duckdb
        con = duckdb.connect("md:")

        now = datetime.now(timezone.utc)

        # Upsert progress record
        con.execute("""
            INSERT OR REPLACE INTO ops.import_progress
            (job_id, league_name, phase, stage, stage_detail, current_step, total_steps,
             overall_pct, status, error_message, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT started_at FROM ops.import_progress WHERE job_id = ?), ?),
                    ?)
        """, [
            job_id, league_name, phase, stage, stage_detail, current_step, total_steps,
            overall_pct, status, error_message, job_id, now, now
        ])

        con.close()
        print(f"[PROGRESS] {phase}/{stage}: {stage_detail} ({overall_pct:.0f}%)")
        return True
    except Exception as e:
        print(f"[PROGRESS] Warning: Could not update progress: {e}")
        return False


def get_progress(job_id: str) -> Optional[dict]:
    """
    Get current progress for a job from MotherDuck.

    Returns dict with progress info or None if not found.
    """
    token = get_motherduck_token()
    if not token or not job_id:
        return None

    try:
        import duckdb
        con = duckdb.connect("md:")

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
                "overall_pct": result[7],
                "status": result[8],
                "error_message": result[9],
                "started_at": result[10].isoformat() if result[10] else None,
                "updated_at": result[11].isoformat() if result[11] else None,
            }
        return None
    except Exception as e:
        print(f"[PROGRESS] Warning: Could not get progress: {e}")
        return None


def mark_completed(job_id: str, league_name: str):
    """Mark an import job as completed successfully."""
    return update_progress(
        job_id=job_id,
        league_name=league_name,
        phase="complete",
        stage="all",
        stage_detail="Import completed successfully",
        overall_pct=100.0,
        status="completed"
    )


def mark_failed(job_id: str, league_name: str, error_message: str):
    """Mark an import job as failed."""
    return update_progress(
        job_id=job_id,
        league_name=league_name,
        phase="error",
        stage="failed",
        stage_detail=error_message[:500],  # Truncate long errors
        status="failed",
        error_message=error_message[:1000]
    )


class ProgressTracker:
    """
    Context manager / helper class for tracking import progress.

    Usage:
        tracker = ProgressTracker(job_id, league_name, total_phases=4)

        with tracker.phase("fetchers", total_steps=7):
            for i, fetcher in enumerate(fetchers):
                tracker.step(i, fetcher_name, f"Fetching {year}...")
                run_fetcher(...)
    """

    # Phase weights for overall progress calculation
    PHASE_WEIGHTS = {
        "settings": 5,      # 5% - Quick API calls
        "fetchers": 40,     # 40% - Most time consuming
        "merges": 15,       # 15% - Data processing
        "transformations": 40,  # 40% - Complex calculations
    }

    def __init__(self, job_id: str, league_name: str):
        self.job_id = job_id
        self.league_name = league_name
        self.current_phase = None
        self.current_step = 0
        self.total_steps = 0
        self.phases_completed = []

        # Initialize table
        init_progress_table()

        # Mark as started
        update_progress(
            job_id=job_id,
            league_name=league_name,
            phase="starting",
            stage="initializing",
            stage_detail="Import starting...",
            overall_pct=0.0,
            status="running"
        )

    def _calculate_overall_pct(self) -> float:
        """Calculate overall progress percentage based on phase weights."""
        completed_pct = sum(self.PHASE_WEIGHTS.get(p, 0) for p in self.phases_completed)

        if self.current_phase and self.total_steps > 0:
            phase_weight = self.PHASE_WEIGHTS.get(self.current_phase, 10)
            phase_progress = (self.current_step / self.total_steps) * phase_weight
            return completed_pct + phase_progress

        return completed_pct

    def start_phase(self, phase: str, total_steps: int):
        """Start a new phase."""
        self.current_phase = phase
        self.current_step = 0
        self.total_steps = total_steps

        update_progress(
            job_id=self.job_id,
            league_name=self.league_name,
            phase=phase,
            stage="starting",
            stage_detail=f"Starting {phase}...",
            current_step=0,
            total_steps=total_steps,
            overall_pct=self._calculate_overall_pct(),
            status="running"
        )

    def step(self, step_num: int, stage: str, detail: str = ""):
        """Update progress within current phase."""
        self.current_step = step_num + 1  # 1-indexed for display

        update_progress(
            job_id=self.job_id,
            league_name=self.league_name,
            phase=self.current_phase,
            stage=stage,
            stage_detail=detail,
            current_step=self.current_step,
            total_steps=self.total_steps,
            overall_pct=self._calculate_overall_pct(),
            status="running"
        )

    def end_phase(self):
        """Mark current phase as completed."""
        if self.current_phase:
            self.phases_completed.append(self.current_phase)

            update_progress(
                job_id=self.job_id,
                league_name=self.league_name,
                phase=self.current_phase,
                stage="complete",
                stage_detail=f"{self.current_phase.title()} phase completed",
                current_step=self.total_steps,
                total_steps=self.total_steps,
                overall_pct=self._calculate_overall_pct(),
                status="running"
            )

    def complete(self):
        """Mark entire import as completed."""
        mark_completed(self.job_id, self.league_name)

    def fail(self, error_message: str):
        """Mark import as failed."""
        mark_failed(self.job_id, self.league_name, error_message)
