#!/usr/bin/env python3
"""
Matchup Table Orchestrator

Runs all transformations that enrich the matchup.parquet file.
Called by initial_import_v2.py or can be run standalone for debugging.

Transformation Order:
1. resolve_hidden_managers.py - Unify --hidden-- manager names by GUID
2. cumulative_stats.py - Core matchup enrichment (playoff flags, records, rankings)
3. player_to_matchup_v2.py - Add player aggregates to matchup
4. expected_record_v2.py - Calculate expected records from schedule simulations
5. playoff_odds_import.py - Monte Carlo playoff odds simulation

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

# Matchup transformation scripts in dependency order
# Scripts in Pass 1: Base calculations (no dependencies on other tables)
PASS_1_SCRIPTS = [
    ("multi_league/transformations/matchup/resolve_hidden_managers.py", "Resolve Hidden Managers", 120),
    ("multi_league/transformations/matchup/cumulative_stats.py", "Cumulative Stats", 600),
]

# Scripts in Pass 3: After player table is enriched (need player data)
PASS_3_SCRIPTS = [
    ("multi_league/transformations/matchup/player_to_matchup_v2.py", "Player -> Matchup", 600),
    ("multi_league/transformations/matchup/expected_record_v2.py", "Expected Record", 900),
    ("multi_league/transformations/matchup/playoff_odds_import.py", "Playoff Odds", 1800),
]


def run_matchup_pass_1(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run Pass 1 matchup transformations (no dependencies on other tables).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[MATCHUP] Starting Pass 1 transformations (base calculations)...")

    all_success = True
    for script, description, timeout in PASS_1_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Matchup Pass 1 complete")
    return all_success


def run_matchup_pass_3(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run Pass 3 matchup transformations (after player enrichment).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[MATCHUP] Starting Pass 3 transformations (player aggregates + simulations)...")

    all_success = True
    for script, description, timeout in PASS_3_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Matchup Pass 3 complete")
    return all_success


def run_all_matchup_transformations(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run all matchup transformations (for standalone debugging).

    Note: In normal pipeline flow, Pass 1 and Pass 3 are run separately
    with player enrichment in between.

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[MATCHUP] Running ALL matchup transformations...")

    success = run_matchup_pass_1(ctx, context_path)
    if not success:
        log("[WARN] Pass 1 had failures, continuing to Pass 3...")

    success2 = run_matchup_pass_3(ctx, context_path)

    return success and success2


def main():
    parser = argparse.ArgumentParser(description="Matchup Table Orchestrator")
    parser.add_argument("--context", type=Path, required=True, help="Path to league_context.json")
    parser.add_argument("--pass", dest="pass_num", type=int, choices=[1, 3], help="Run only specific pass (1 or 3)")
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)
    context_path = str(args.context)

    if args.pass_num == 1:
        success = run_matchup_pass_1(ctx, context_path)
    elif args.pass_num == 3:
        success = run_matchup_pass_3(ctx, context_path)
    else:
        success = run_all_matchup_transformations(ctx, context_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
