#!/usr/bin/env python3
"""
Player Table Orchestrator

Runs all transformations that enrich the player.parquet file.
Called by initial_import_v2.py or can be run standalone for debugging.

Transformation Order (Pass 2 - after matchup Pass 1):
1. matchup_to_player_v2.py - Join matchup columns INTO player
2. player_stats_v2.py - Calculate player stats (optimal_points, etc.)
3. replacement_level_v2.py - Calculate position replacement baselines for SPAR

After Draft/Transaction enrichment (late in Pass 3):
4. draft_to_player_v2.py - Import keeper/draft columns FROM draft TO player
5. transactions_to_player_v2.py - Add FAAB data TO player

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

# Pass 2: Core player enrichment (after matchup Pass 1)
PASS_2_SCRIPTS = [
    ("multi_league/transformations/player/matchup_to_player_v2.py", "Matchup -> Player", 600),
    ("multi_league/transformations/player/player_stats_v2.py", "Player Stats", 900),
    ("multi_league/transformations/player/replacement_level_v2.py", "Replacement Levels", 600),
]

# Pass 3 (late): After draft/transaction enrichment
PASS_3_LATE_SCRIPTS = [
    ("multi_league/transformations/player/draft_to_player_v2.py", "Draft -> Player", 600),
    ("multi_league/transformations/player/transactions_to_player_v2.py", "Transactions -> Player", 600),
    ("multi_league/transformations/player/keeper_economics.py", "Keeper Economics", 600),
]


def run_player_pass_2(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run Pass 2 player transformations (core stats).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[PLAYER] Starting Pass 2 transformations (core stats)...")

    all_success = True
    for script, description, timeout in PASS_2_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Player Pass 2 complete")
    return all_success


def run_player_pass_3_late(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run Pass 3 late player transformations (after draft/transaction enrichment).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[PLAYER] Starting Pass 3 late transformations (draft/transaction imports)...")

    all_success = True
    for script, description, timeout in PASS_3_LATE_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Player Pass 3 late complete")
    return all_success


def run_all_player_transformations(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run all player transformations (for standalone debugging).

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[PLAYER] Running ALL player transformations...")

    success = run_player_pass_2(ctx, context_path)
    if not success:
        log("[WARN] Pass 2 had failures, continuing...")

    success2 = run_player_pass_3_late(ctx, context_path)

    return success and success2


def main():
    parser = argparse.ArgumentParser(description="Player Table Orchestrator")
    parser.add_argument("--context", type=Path, required=True, help="Path to league_context.json")
    parser.add_argument("--pass", dest="pass_num", choices=["2", "3late"], help="Run only specific pass")
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)
    context_path = str(args.context)

    if args.pass_num == "2":
        success = run_player_pass_2(ctx, context_path)
    elif args.pass_num == "3late":
        success = run_player_pass_3_late(ctx, context_path)
    else:
        success = run_all_player_transformations(ctx, context_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
