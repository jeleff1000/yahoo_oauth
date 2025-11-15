"""
Centralized run metadata and logging system.

Provides JSON-based structured logging for all data pipeline runs with:
- Start/end timestamps
- Rows read/written per step
- Retries, timeouts, errors
- Peak memory usage
- Performance metrics

Makes SRE-style debugging trivial by having one JSON log per run.

Usage:
    from run_metadata import RunLogger

    with RunLogger("yahoo_nfl_merge", year=2024, week=5) as logger:
        logger.start_step("fetch_yahoo_data")
        # ... do work ...
        logger.complete_step(rows_read=1000, rows_written=950)

        logger.start_step("merge_with_nfl")
        # ... do work ...
        logger.complete_step(rows_read=2000, rows_written=1900)
"""
from __future__ import annotations

import json
import os
import psutil
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional, Union


class RunLogger:
    """
    Structured logger for data pipeline runs.

    Tracks detailed metadata about each run including timing, row counts,
    errors, retries, and memory usage.
    """

    def __init__(
        self,
        script_name: str,
        log_dir: Optional[Path] = None,
        year: Optional[int] = None,
        week: Optional[int] = None,
        **metadata
    ):
        """
        Initialize run logger.

        Args:
            script_name: Name of the script/process
            log_dir: Directory for log files (default: data/logs)
            year: Season year (optional)
            week: Week number (optional)
            **metadata: Additional metadata to include
        """
        self.script_name = script_name
        self.year = year
        self.week = week

        # Setup log directory
        if log_dir is None:
            # Default to data/logs in the player_stats directory
            base_dir = Path(__file__).parent.parent.parent
            data_dir = os.environ.get('DATA_DIRECTORY', str(base_dir / 'fantasy_football_data'))
            log_dir = Path(data_dir) / 'logs'

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize run metadata
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.process = psutil.Process()
        self.start_time = datetime.now()
        self.start_monotonic = monotonic()

        self.metadata = {
            'run_id': self.run_id,
            'script_name': script_name,
            'start_time': self.start_time.isoformat(),
            'year': year,
            'week': week,
            'hostname': os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown')),
            'pid': os.getpid(),
            **metadata
        }

        # Track steps
        self.steps: List[Dict[str, Any]] = []
        self.current_step: Optional[Dict[str, Any]] = None
        self.current_step_start: Optional[float] = None

        # Track errors and retries
        self.errors: List[Dict[str, Any]] = []
        self.retries: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

        # Memory tracking
        self.peak_memory_mb = 0.0
        self.initial_memory_mb = self._get_memory_mb()

        # Thread safety
        self._lock = threading.Lock()

        # Log file path
        suffix = f"_y{year}" if year else ""
        suffix += f"_w{week}" if week else ""
        self.log_file = self.log_dir / f"{script_name}_{self.run_id}{suffix}.json"

        # Write initial metadata
        self._write_log()

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _update_peak_memory(self) -> None:
        """Update peak memory if current is higher."""
        current = self._get_memory_mb()
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current

    def start_step(self, step_name: str, **step_metadata) -> None:
        """
        Start a new step in the pipeline.

        Args:
            step_name: Name of the step
            **step_metadata: Additional metadata for this step
        """
        with self._lock:
            if self.current_step is not None:
                self.warning(f"Step '{self.current_step['name']}' was not completed before starting '{step_name}'")

            self._update_peak_memory()

            self.current_step = {
                'name': step_name,
                'start_time': datetime.now().isoformat(),
                'start_memory_mb': self._get_memory_mb(),
                **step_metadata
            }
            self.current_step_start = monotonic()

    def complete_step(
        self,
        rows_read: Optional[int] = None,
        rows_written: Optional[int] = None,
        files_read: Optional[int] = None,
        files_written: Optional[int] = None,
        **step_results
    ) -> None:
        """
        Complete the current step.

        Args:
            rows_read: Number of rows read
            rows_written: Number of rows written
            files_read: Number of files read
            files_written: Number of files written
            **step_results: Additional results/metrics
        """
        with self._lock:
            if self.current_step is None:
                self.warning("complete_step called but no step is active")
                return

            self._update_peak_memory()

            duration_sec = monotonic() - self.current_step_start if self.current_step_start else 0

            self.current_step.update({
                'end_time': datetime.now().isoformat(),
                'duration_sec': round(duration_sec, 3),
                'rows_read': rows_read,
                'rows_written': rows_written,
                'files_read': files_read,
                'files_written': files_written,
                'end_memory_mb': self._get_memory_mb(),
                'status': 'completed',
                **step_results
            })

            self.steps.append(self.current_step)
            self.current_step = None
            self.current_step_start = None

            self._write_log()

    def fail_step(self, error: Union[str, Exception]) -> None:
        """
        Mark the current step as failed.

        Args:
            error: Error message or exception
        """
        with self._lock:
            if self.current_step is None:
                self.warning("fail_step called but no step is active")
                return

            duration_sec = monotonic() - self.current_step_start if self.current_step_start else 0

            error_info = {
                'message': str(error),
                'type': type(error).__name__ if isinstance(error, Exception) else 'error'
            }

            if isinstance(error, Exception):
                error_info['traceback'] = traceback.format_exception(type(error), error, error.__traceback__)

            self.current_step.update({
                'end_time': datetime.now().isoformat(),
                'duration_sec': round(duration_sec, 3),
                'status': 'failed',
                'error': error_info
            })

            self.steps.append(self.current_step)
            self.errors.append({
                'step': self.current_step['name'],
                'time': datetime.now().isoformat(),
                **error_info
            })

            self.current_step = None
            self.current_step_start = None

            self._write_log()

    def log_retry(self, operation: str, attempt: int, max_attempts: int, error: str) -> None:
        """
        Log a retry attempt.

        Args:
            operation: Name of the operation being retried
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            error: Error that caused the retry
        """
        with self._lock:
            self.retries.append({
                'operation': operation,
                'attempt': attempt,
                'max_attempts': max_attempts,
                'error': str(error),
                'time': datetime.now().isoformat()
            })
            self._write_log()

    def log_timeout(self, operation: str, timeout_sec: float) -> None:
        """
        Log a timeout.

        Args:
            operation: Name of the operation that timed out
            timeout_sec: Timeout duration in seconds
        """
        with self._lock:
            self.errors.append({
                'type': 'timeout',
                'operation': operation,
                'timeout_sec': timeout_sec,
                'time': datetime.now().isoformat()
            })
            self._write_log()

    def warning(self, message: str) -> None:
        """
        Log a warning.

        Args:
            message: Warning message
        """
        with self._lock:
            self.warnings.append({
                'message': message,
                'time': datetime.now().isoformat()
            })
            self._write_log()

    def set_metadata(self, **metadata) -> None:
        """
        Add or update metadata fields.

        Args:
            **metadata: Metadata key-value pairs
        """
        with self._lock:
            self.metadata.update(metadata)
            self._write_log()

    def _write_log(self) -> None:
        """Write current state to log file."""
        try:
            self._update_peak_memory()

            log_data = {
                **self.metadata,
                'steps': self.steps,
                'errors': self.errors,
                'retries': self.retries,
                'warnings': self.warnings,
                'memory': {
                    'initial_mb': round(self.initial_memory_mb, 2),
                    'peak_mb': round(self.peak_memory_mb, 2),
                    'current_mb': round(self._get_memory_mb(), 2)
                }
            }

            # Add end time if run is complete
            if self.current_step is None and len(self.steps) > 0:
                total_duration = monotonic() - self.start_monotonic
                log_data['end_time'] = datetime.now().isoformat()
                log_data['total_duration_sec'] = round(total_duration, 3)

            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            # Don't let logging errors break the pipeline
            print(f"Warning: Failed to write log: {e}")

    def finalize(self, status: str = 'completed', **final_metadata) -> None:
        """
        Finalize the run and write final log.

        Args:
            status: Final status (completed, failed, partial)
            **final_metadata: Additional metadata to include
        """
        with self._lock:
            # If there's an active step, mark it as interrupted
            if self.current_step is not None:
                self.current_step['status'] = 'interrupted'
                self.steps.append(self.current_step)
                self.current_step = None

            self.metadata['status'] = status
            self.metadata.update(final_metadata)

            self._write_log()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Exception occurred
            self.finalize(status='failed', exception=str(exc_val))
        else:
            self.finalize(status='completed')
        return False  # Don't suppress exceptions


@contextmanager
def log_step(logger: RunLogger, step_name: str, **metadata):
    """
    Context manager for logging a step.

    Usage:
        with log_step(logger, "merge_data", source="yahoo"):
            # ... do work ...
            pass
    """
    logger.start_step(step_name, **metadata)
    try:
        yield logger
        logger.complete_step()
    except Exception as e:
        logger.fail_step(e)
        raise


def get_recent_logs(
    script_name: Optional[str] = None,
    limit: int = 10,
    log_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Get recent log files.

    Args:
        script_name: Filter by script name
        limit: Maximum number of logs to return
        log_dir: Log directory (uses default if not provided)

    Returns:
        List of log metadata dictionaries
    """
    if log_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        data_dir = os.environ.get('DATA_DIRECTORY', str(base_dir / 'fantasy_football_data'))
        log_dir = Path(data_dir) / 'logs'

    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    # Find log files
    pattern = f"{script_name}_*.json" if script_name else "*.json"
    log_files = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    logs = []
    for log_file in log_files[:limit]:
        try:
            with open(log_file) as f:
                log_data = json.load(f)
                logs.append({
                    'file': str(log_file),
                    'run_id': log_data.get('run_id'),
                    'script_name': log_data.get('script_name'),
                    'start_time': log_data.get('start_time'),
                    'status': log_data.get('status', 'unknown'),
                    'steps_count': len(log_data.get('steps', [])),
                    'errors_count': len(log_data.get('errors', [])),
                    'retries_count': len(log_data.get('retries', []))
                })
        except Exception:
            continue

    return logs
