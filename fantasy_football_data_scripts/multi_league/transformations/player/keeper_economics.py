"""
Keeper Economics - Player Table Transformation

Calculates keeper prices directly on the player table.

This is the simplified keeper economics that:
1. Uses player table as source of truth (has FAAB data, draft cost, keeper status)
2. Calculates consecutive keeper years per (player, manager)
3. Applies keeper cost formulas based on league rules
4. Writes keeper_price, keeper_year directly to player table

No need for draft → player copy step anymore.

Usage:
    python keeper_economics.py --context path/to/league_context.json
    python keeper_economics.py --context path/to/league_context.json --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np

# Path setup
_script_file = Path(__file__).resolve()
_player_dir = _script_file.parent
_transformations_dir = _player_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))

from multi_league.core.league_context import LeagueContext


# =========================================================
# Keeper Price Calculator
# =========================================================

class KeeperPriceCalculator:
    """Calculate keeper prices using league rules."""

    def __init__(self, rules: dict):
        self.rules = rules or {}
        self.enabled = self.rules.get('enabled', False)
        self.min_price = self.rules.get('min_price', 1)
        self.max_price = self.rules.get('max_price')
        self.budget = self.rules.get('budget', 200)
        self.base_cost_rules = self.rules.get('base_cost_rules', {})
        self.formulas = self.rules.get('formulas_by_keeper_year', {})

    def calculate_base_cost(self, cost: float, faab_bid: float, is_drafted: bool) -> float:
        """Calculate base keeper cost from acquisition."""
        if is_drafted and cost > 0:
            rule = self.base_cost_rules.get('auction', {})
            mult = rule.get('multiplier', 1.0)
            flat = rule.get('flat', 0.0)
            return cost * mult + flat
        elif faab_bid > 0:
            rule = self.base_cost_rules.get('faab_only', {})
            mult = rule.get('multiplier', 1.0)
            flat = rule.get('flat', 10.0)
            return faab_bid * mult + flat
        else:
            rule = self.base_cost_rules.get('free_agent', {})
            return rule.get('value', self.min_price)

    def calculate_keeper_price(
        self,
        base_cost: float,
        keeper_year: int,
    ) -> float:
        """
        Calculate keeper price for a given year.

        Args:
            base_cost: Original acquisition cost (draft price or FAAB-based)
            keeper_year: How many years player has been kept (1 = first time)

        Returns:
            Calculated keeper price
        """
        if not self.enabled:
            return max(base_cost, self.min_price)

        # Get formula for this keeper year
        formula = self._get_formula(keeper_year)

        if formula is None:
            price = base_cost
        else:
            esc_type = formula.get('type', 'from_base')

            if esc_type == 'from_base':
                # base_cost + flat_per_year * (keeper_year - 1)
                flat_per_year = formula.get('flat_per_year', 5.0)
                price = base_cost + flat_per_year * (keeper_year - 1)

            elif esc_type == 'compounding':
                # Recursively apply: prev_cost * mult + flat
                mult = formula.get('multiplier', 1.0)
                flat = formula.get('flat_add', 0.0)
                price = base_cost
                for _ in range(1, keeper_year):
                    price = price * mult + flat

            else:
                # No escalation or unknown type
                price = base_cost

        # Apply constraints
        price = max(price, self.min_price)
        if self.max_price is not None:
            price = min(price, self.max_price)

        return round(price)

    def _get_formula(self, keeper_year: int) -> Optional[dict]:
        """Get formula for specific keeper year."""
        # Check exact match
        if str(keeper_year) in self.formulas:
            return self.formulas[str(keeper_year)]

        # Check wildcard (e.g., "2+")
        for key, formula in self.formulas.items():
            if '+' in key:
                base_year = int(key.replace('+', ''))
                if keeper_year >= base_year:
                    return formula

        return None


# =========================================================
# Consecutive Keeper Year Calculator
# =========================================================

def calculate_consecutive_keeper_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consecutive years each player has been kept by the same manager.

    Uses end-of-season roster (max week per year) to determine keeper streaks.

    Returns DataFrame with:
        - keeper_year: consecutive years kept (1 = first time kept)
        - base_keeper_cost: original acquisition cost for price calculation
    """
    result = df.copy()
    result['keeper_year'] = 0
    result['base_keeper_cost'] = 0.0

    # Get end-of-season rows (max week per player/manager/year)
    if 'week' not in result.columns:
        # Already season-level data
        season_df = result
    else:
        # Get max week per player/manager/year
        idx = result.groupby(['yahoo_player_id', 'manager', 'year'])['week'].idxmax()
        season_df = result.loc[idx].copy()

    # Sort by player, manager, year
    season_df = season_df.sort_values(['yahoo_player_id', 'manager', 'year'])

    # Track keeper streaks: {(player_id, manager): {'streak': n, 'base_cost': x}}
    keeper_history = {}

    for idx, row in season_df.iterrows():
        player_id = row.get('yahoo_player_id')
        manager = row.get('manager', '')
        year = row.get('year')
        is_keeper = row.get('is_keeper_status', 0) == 1
        cost = float(row.get('cost', 0) or 0)
        faab = float(row.get('max_faab_bid_to_date', 0) or 0)

        if pd.isna(player_id) or pd.isna(year):
            continue

        key = (str(player_id), str(manager))

        if is_keeper:
            # Check if continuing a streak
            if key in keeper_history:
                prev = keeper_history[key]
                if prev['last_year'] == year - 1:
                    # Continuing streak
                    streak = prev['streak'] + 1
                    base_cost = prev['base_cost']
                else:
                    # New streak (gap in years)
                    streak = 1
                    base_cost = cost if cost > 0 else faab
            else:
                # First time kept by this manager
                streak = 1
                base_cost = cost if cost > 0 else faab

            keeper_history[key] = {
                'streak': streak,
                'last_year': year,
                'base_cost': base_cost
            }

            result.loc[idx, 'keeper_year'] = streak
            result.loc[idx, 'base_keeper_cost'] = base_cost
        else:
            # Not a keeper - reset streak
            if key in keeper_history:
                del keeper_history[key]

    return result


# =========================================================
# Main Transformation
# =========================================================

def calculate_keeper_economics(
    ctx: LeagueContext,
    player_path: Path,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Calculate keeper economics on player table.

    Args:
        ctx: LeagueContext with keeper_rules
        player_path: Path to player.parquet
        dry_run: If True, don't write changes

    Returns:
        Dict with statistics
    """
    print("=" * 60)
    print("KEEPER ECONOMICS (Player Table)")
    print("=" * 60)

    if not ctx.keepers_enabled:
        print(f"\n[SKIP] Keepers not enabled for {ctx.league_name}")
        return {'status': 'skipped', 'reason': 'keepers_not_enabled'}

    # Load player data
    print(f"\nLoading: {player_path}")
    if not player_path.exists():
        print(f"[ERROR] Player file not found: {player_path}")
        return {'status': 'error', 'reason': 'file_not_found'}

    player = pd.read_parquet(player_path)
    print(f"  Loaded {len(player):,} rows")

    # Check required columns
    required = ['yahoo_player_id', 'year', 'manager']
    missing = [c for c in required if c not in player.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return {'status': 'error', 'reason': f'missing_columns: {missing}'}

    # Calculate consecutive keeper years
    print("\nCalculating consecutive keeper years...")
    player = calculate_consecutive_keeper_years(player)

    keepers = player[player['keeper_year'] > 0]
    print(f"  Found {len(keepers):,} keeper records")

    # Initialize calculator
    calculator = KeeperPriceCalculator(ctx.keeper_rules)

    # Calculate keeper prices
    print("\nCalculating keeper prices...")
    player['keeper_price'] = 0

    for idx in keepers.index:
        row = player.loc[idx]
        base_cost = row['base_keeper_cost']
        keeper_year = int(row['keeper_year'])

        if base_cost <= 0:
            # Use draft cost or FAAB
            cost = float(row.get('cost', 0) or 0)
            faab = float(row.get('max_faab_bid_to_date', 0) or 0)
            base_cost = calculator.calculate_base_cost(cost, faab, cost > 0)

        price = calculator.calculate_keeper_price(base_cost, keeper_year)
        player.loc[idx, 'keeper_price'] = price

    # Stats
    prices_set = (player['keeper_price'] > 0).sum()
    print(f"  Calculated prices for {prices_set:,} records")

    # Write results
    if dry_run:
        print("\n[DRY RUN] Would write to:", player_path)
    else:
        print(f"\nWriting: {player_path}")
        player.to_parquet(player_path, index=False)
        print("  Done!")

    return {
        'status': 'success',
        'total_rows': len(player),
        'keeper_records': len(keepers),
        'prices_calculated': prices_set,
    }


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Calculate keeper economics on player table")
    parser.add_argument("--context", required=True, help="Path to league_context.json")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    args = parser.parse_args()

    ctx = LeagueContext.load(Path(args.context))
    player_path = ctx.data_dir / "player.parquet"

    result = calculate_keeper_economics(ctx, player_path, dry_run=args.dry_run)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
