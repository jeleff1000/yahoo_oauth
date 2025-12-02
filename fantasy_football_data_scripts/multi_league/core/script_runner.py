#!/usr/bin/env python3
"""
Script Runner Utilities

Provides utilities for orchestrating multiple child scripts with retry logic,
rate limiting, OAuth environment setup, and comprehensive logging.

Extracted from initial_import_v2.py for reusability across multiple orchestration scripts.
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Try to import LeagueContext; gracefully handle if not available
try:
    from multi_league.core.league_context import LeagueContext
except ImportError:
    LeagueContext = None


def log(*values: object, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    """
    Print with a timestamp prefix. Matches print() signature so it can be passed as a log callable.

    Usage:
        log("message", var)
        log("Starting process", process_name)

    Args:
        values: Values to print (same as print())
        sep: String inserted between values (default: " ")
        end: String appended after the last value (default: "\n")
        file: File object to write to (default: sys.stdout)
        flush: Whether to forcibly flush the stream (default: False)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Join values similar to print()
    try:
        msg = sep.join(str(v) for v in values)
    except Exception:
        # Fallback if some object doesn't stringify nicely
        msg = " ".join(map(str, values))
    print(f"[{ts}] {msg}", end=end, file=file, flush=flush)


def script_supports_flag(script: Path, flag: str, timeout: int = 10) -> bool:
    """
    Return True if running `script --help` mentions `flag` (defensive).

    This is used to avoid passing unsupported flags to legacy scripts.

    Args:
        script: Path to the script
        flag: Flag to check for (e.g., "--year")
        timeout: Timeout in seconds for the help command

    Returns:
        True if the flag is mentioned in the help text, False otherwise
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        help_text = (result.stdout or "") + "\n" + (result.stderr or "")
        return flag in help_text
    except Exception:
        return False


def setup_oauth_environment(context_path: str) -> dict:
    """
    Extract OAuth credentials from league context and set up environment variables.

    This centralizes OAuth setup logic that was previously duplicated in run_script().

    Args:
        context_path: Path to league_context.json

    Returns:
        Dictionary of environment variables to use for child processes
    """
    env = dict(os.environ)

    if LeagueContext is None:
        log("[ENV] LeagueContext not available; skipping OAuth setup")
        return env

    try:
        ctx = LeagueContext.load(context_path)
        oauth_file = getattr(ctx, "oauth_file_path", None)

        if oauth_file:
            oauth_path = Path(oauth_file)
            if oauth_path.exists():
                env["OAUTH_PATH"] = str(oauth_path)
                log(f"[ENV] OAUTH_PATH={oauth_path}")

                try:
                    token_json = json.loads(oauth_path.read_text(encoding="utf-8"))
                    if token_json.get("access_token"):
                        env["YAHOO_ACCESS_TOKEN"] = token_json["access_token"]
                    if token_json.get("refresh_token"):
                        env["YAHOO_REFRESH_TOKEN"] = token_json["refresh_token"]
                    if token_json.get("consumer_key"):
                        env["YAHOO_CONSUMER_KEY"] = token_json["consumer_key"]
                    if token_json.get("consumer_secret"):
                        env["YAHOO_CONSUMER_SECRET"] = token_json["consumer_secret"]
                    if token_json.get("token_type"):
                        env["YAHOO_TOKEN_TYPE"] = token_json["token_type"]
                    env.setdefault("YAHOO_OAUTH_ACCESS_TOKEN", env.get("YAHOO_ACCESS_TOKEN", ""))
                    env.setdefault("OAUTH_TOKEN", env.get("YAHOO_ACCESS_TOKEN", ""))
                except Exception as e:
                    log(f"[ENV] OAuth JSON load failed: {e}")
    except Exception as e:
        log(f"[ENV] Context load failed (continuing without OAuth env): {e}")

    return env


def run_script(
    script_path: str,
    label: str,
    context_path: str,
    additional_args: Optional[List[str]] = None,
    timeout: Optional[int] = 900,
    oauth_env: Optional[dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Execute a child script with retry logic for rate limits.

    Features:
    - Automatic retry with exponential backoff for rate limit errors
    - OAuth environment setup (if oauth_env provided)
    - Defensive flag filtering for legacy scripts
    - Comprehensive logging

    Args:
        script_path: Relative path to script from SCRIPT_DIR
        label: Human-readable label for logging
        context_path: Path to league_context.json
        additional_args: Extra CLI arguments to pass
        timeout: Timeout in seconds (default: 900), or None for no timeout
        oauth_env: Pre-configured OAuth environment (if None, will be set up)

    Returns:
        Tuple of (success: bool, stderr: Optional[str])

    Example:
        >>> ok, err = run_script(
        ...     "multi_league/data_fetchers/yahoo_fantasy_data.py",
        ...     "Yahoo player data",
        ...     "/path/to/context.json",
        ...     additional_args=["--year", "2024"]
        ... )
    """
    # Get script directory (assume this module is in multi_league/core/)
    SCRIPT_DIR = Path(__file__).parent.parent.parent

    def _run_once() -> Tuple[bool, Optional[str]]:
        script = SCRIPT_DIR / script_path
        if not script.exists():
            log(f"[SKIP] Script not found: {script}")
            return False, None

        # For our owned scripts, do NOT filter flags. They all support --year/--week/etc.
        KNOWN_SAFE = {
            "yahoo_fantasy_data_v2.py",
            "yahoo_fantasy_data.py",
            "nfl_offense_stats_v2.py",
            "nfl_offense_stats.py",
            "defense_stats_v2.py",
            "defense_stats.py",
            "yahoo_nfl_merge_v2.py",
            "yahoo_nfl_merge.py",
            "weekly_matchup_data_v2.py",
            "draft_data_v2.py",
            "transactions_v2.py",
            "combine_dst_to_nfl.py",
        }

        filtered_args: List[str] = []
        if additional_args:
            if script.name in KNOWN_SAFE:
                filtered_args = list(additional_args)
            else:
                # Defensive filtering: if a flag is skipped, also skip its following value.
                idx = 0
                while idx < len(additional_args):
                    token = additional_args[idx]
                    if token.startswith("--"):
                        if script_supports_flag(script, token.split("=")[0]):
                            filtered_args.append(token)
                        else:
                            log(f"      [INFO] Skipping unsupported flag {token} for {script.name}")
                            # also skip next token if it is a value (not another flag)
                            nxt = (additional_args[idx + 1] if idx + 1 < len(additional_args) else None)
                            if nxt and not str(nxt).startswith("--"):
                                idx += 1
                    else:
                        filtered_args.append(token)
                    idx += 1

        cmd = [sys.executable, str(script), "--context", context_path]
        if filtered_args:
            cmd.extend(filtered_args)

        # Use provided OAuth env or set up new one
        env = oauth_env if oauth_env is not None else setup_oauth_environment(context_path)

        log(f"[RUN] {label}")
        log(f"      Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(script.parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.stdout:
                for line in result.stdout.strip().splitlines()[-12:]:
                    log(f"      {line}")
            if result.returncode != 0:
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[-25:]:
                        log(f"      {line}")
                return False, result.stderr
            log(f"[OK] Completed: {label}")
            return True, None

        except subprocess.TimeoutExpired:
            timeout_msg = f"{timeout}s" if timeout is not None else "unlimited"
            log(f"[TIMEOUT] {label} timed out after {timeout_msg}")
            return False, None
        except Exception as e:
            log(f"[FAIL] Error running {label}: {e}")
            return False, None

    # Retry logic for rate limits
    MAX_RETRIES = 3
    RATE_LIMIT_COOLDOWN = 600  # 10 minutes in seconds

    last_stderr = None
    for attempt in range(MAX_RETRIES):
        ok, stderr = _run_once()
        last_stderr = stderr

        if ok:
            return True, None

        # Check if it's a rate limit error or temporary access denial
        if stderr and ("Request denied" in stderr or
                      "Forbidden access" in stderr or
                      "APITimeoutError" in stderr or
                      "rate limit" in stderr.lower()):
            if attempt < MAX_RETRIES - 1:  # Don't sleep on last attempt
                log(f"[RATE LIMIT] Yahoo API rate limit/access denial detected. Waiting {RATE_LIMIT_COOLDOWN // 60} minutes before retry {attempt + 2}/{MAX_RETRIES}...")
                time.sleep(RATE_LIMIT_COOLDOWN)
                log(f"[RETRY] Retrying {label} (attempt {attempt + 2}/{MAX_RETRIES})")
                continue
            else:
                log(f"[FAIL] Max retries reached for {label} after rate limiting")
                return False, stderr
        else:
            # Not a rate limit error, don't retry
            return False, stderr

    return False, last_stderr


def run_scripts_parallel(
    scripts: List[Tuple[str, str]],
    context_path: str,
    additional_args: Optional[List[str]] = None,
    timeout: Optional[int] = 900
) -> dict:
    """
    Run multiple scripts in parallel (future enhancement).

    Currently runs sequentially; can be enhanced with ThreadPoolExecutor.

    Args:
        scripts: List of (script_path, label) tuples
        context_path: Path to league_context.json
        additional_args: Extra CLI arguments for all scripts
        timeout: Timeout per script (or None for no timeout)

    Returns:
        Dictionary mapping labels to success/failure status
    """
    # Set up OAuth environment once for all scripts
    oauth_env = setup_oauth_environment(context_path)

    results = {}
    for script_path, label in scripts:
        ok, _ = run_script(script_path, label, context_path, additional_args, timeout, oauth_env)
        results[label] = ok

    return results

