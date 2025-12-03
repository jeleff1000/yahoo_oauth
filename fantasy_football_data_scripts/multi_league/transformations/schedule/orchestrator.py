#!/usr/bin/env python3
"""
Schedule Table Orchestrator

Runs all transformations that enrich the schedule.parquet file.
Called by initial_import_v2.py or can be run standalone for debugging.

Transformation Order:
1. enrich_schedule_with_playoff_flags.py - Merge playoff flags from matchup into schedule

Usage:
    python orchestrator.py --context /path/to/league_context.json
"""
import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))

from multi_league.core.script_runner import run_script, log
from multi_league.core.league_context import LeagueContext

# Schedule transformation scripts
SCHEDULE_SCRIPTS = [
    ("multi_league/transformations/schedule/enrich_schedule_with_playoff_flags.py", "Enrich Schedule w/ Playoff Flags", 120),
]


def run_schedule_enrichment(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run all schedule table transformations.

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[SCHEDULE] Starting schedule transformations...")

    all_success = True
    for script, description, timeout in SCHEDULE_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Schedule enrichment complete")
    return all_success


def main():
    parser = argparse.ArgumentParser(description="Schedule Table Orchestrator")
    parser.add_argument("--context", type=Path, required=True, help="Path to league_context.json")
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)
    context_path = str(args.context)

    success = run_schedule_enrichment(ctx, context_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
