#!/usr/bin/env python3
"""
Draft Table Orchestrator

Runs all transformations that enrich the draft.parquet file.
Called by initial_import_v2.py or can be run standalone for debugging.

Transformation Order:
1. player_to_draft_v2.py - Add player stats TO draft (needed for SPAR calculations)
2. draft_value_metrics_v3.py - Calculate SPAR + keeper economics (creates kept_next_year, spar, pgvor, etc.)
3. keeper_economics_v2.py - Calculate keeper_price for next year planning (needs draft cost + FAAB)

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

# Draft transformation scripts in dependency order
# Early Pass 3: Before draft -> player
EARLY_SCRIPTS = [
    ("multi_league/transformations/draft/player_to_draft_v2.py", "Player -> Draft", 600),
    ("multi_league/transformations/draft/draft_value_metrics_v3.py", "Draft SPAR Metrics", 600),
]

# Late Pass 3: After transactions -> player (needs FAAB data)
LATE_SCRIPTS = [
    ("multi_league/transformations/draft/keeper_economics_v2.py", "Keeper Economics", 600),
]


def run_draft_early(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run early draft transformations (SPAR calculations).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[DRAFT] Starting early transformations (player stats + SPAR)...")

    all_success = True
    for script, description, timeout in EARLY_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Draft early transformations complete")
    return all_success


def run_draft_late(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run late draft transformations (keeper economics).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[DRAFT] Starting late transformations (keeper economics)...")

    all_success = True
    for script, description, timeout in LATE_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Draft late transformations complete")
    return all_success


def run_all_draft_transformations(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run all draft transformations (for standalone debugging).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[DRAFT] Running ALL draft transformations...")

    success = run_draft_early(ctx, context_path)
    if not success:
        log("[WARN] Early transformations had failures, continuing...")

    success2 = run_draft_late(ctx, context_path)

    return success and success2


def main():
    parser = argparse.ArgumentParser(description="Draft Table Orchestrator")
    parser.add_argument("--context", type=Path, required=True, help="Path to league_context.json")
    parser.add_argument("--stage", choices=["early", "late"], help="Run only specific stage")
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)
    context_path = str(args.context)

    if args.stage == "early":
        success = run_draft_early(ctx, context_path)
    elif args.stage == "late":
        success = run_draft_late(ctx, context_path)
    else:
        success = run_all_draft_transformations(ctx, context_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
