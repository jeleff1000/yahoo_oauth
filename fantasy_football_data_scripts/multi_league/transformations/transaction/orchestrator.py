#!/usr/bin/env python3
"""
Transaction Table Orchestrator

Runs all transformations that enrich the transactions.parquet file.
Called by initial_import_v2.py or can be run standalone for debugging.

Transformation Order:
1. fix_unknown_managers.py - Fix manager="Unknown" by backfilling from most recent add
2. player_to_transactions_v2.py - Add ROS performance TO transactions
3. transaction_value_metrics_v3.py - SPAR-based transaction value metrics

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

# Transaction transformation scripts in dependency order
TRANSACTION_SCRIPTS = [
    ("multi_league/transformations/transaction/fix_unknown_managers.py", "Fix Unknown Managers", 120),
    ("multi_league/transformations/transaction/player_to_transactions_v2.py", "Player <-> Transactions", 600),
    ("multi_league/transformations/transaction/transaction_value_metrics_v3.py", "Transaction SPAR Metrics", 600),
]


def run_transaction_enrichment(ctx: LeagueContext, context_path: str) -> bool:
    """
    Run all transaction table transformations.

    Args:
        ctx: LeagueContext instance
        context_path: Path to league_context.json

    Returns:
        True if all transformations succeeded, False if any failed
    """
    log("[TRANSACTION] Starting transaction transformations...")

    all_success = True
    for script, description, timeout in TRANSACTION_SCRIPTS:
        ok, err = run_script(script, description, context_path, timeout=timeout)
        if not ok:
            log(f"[FAIL] {description} failed")
            all_success = False

    if all_success:
        log("[OK] Transaction enrichment complete")
    return all_success


def main():
    parser = argparse.ArgumentParser(description="Transaction Table Orchestrator")
    parser.add_argument("--context", type=Path, required=True, help="Path to league_context.json")
    args = parser.parse_args()

    ctx = LeagueContext.load(args.context)
    context_path = str(args.context)

    success = run_transaction_enrichment(ctx, context_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
