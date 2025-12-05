"""
Keeper Economics - Player Table Transformation

Calculates PROJECTED NEXT YEAR keeper prices for ALL rostered players.

This transformation:
1. Uses player table as source of truth (has FAAB data, draft cost, keeper status)
2. Calculates consecutive keeper years per (player, manager)
3. Calculates base_cost for ALL players (not just keepers)
4. Projects what keeper_price would be NEXT YEAR if kept
5. Writes keeper_price, keeper_year, base_keeper_cost to player table

The keeper_price shows what it would cost to keep each player NEXT YEAR,
helping managers make keeper decisions.

Usage:
    python keeper_economics.py --context path/to/league_context.json
    python keeper_economics.py --context path/to/league_context.json --dry-run
"""

import argparse
import math
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
        self.round_up = self.rules.get('round_up', False)
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

        # Apply rounding
        if self.round_up:
            price = math.ceil(price)
        else:
            price = round(price)

        # Apply constraints
        price = max(price, self.min_price)
        if self.max_price is not None:
            price = min(price, self.max_price)

        return int(price)

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

    Uses end-of-season roster (last week with rostered players) to determine keeper streaks.
    This dynamically finds the championship week for any playoff format.

    Args:
        df: Player DataFrame with weekly data

    Returns DataFrame with:
        - keeper_year: consecutive years kept (1 = first time kept, 0 = never kept)
        - base_keeper_cost: original acquisition cost for price calculation
    """
    result = df.copy()
    result['keeper_year'] = 0
    result['base_keeper_cost'] = 0.0

    # Get end-of-season rows using championship week
    # (not max week which includes post-season roster clearing)
    if 'week' not in result.columns:
        # Already season-level data
        season_df = result
    else:
        # Determine championship week dynamically per year
        # Method: Find the last week where ANY players have real managers
        # This works for any playoff format (2-week, 3-week, 4-week, etc.)
        season_end_weeks = {}

        for year in result['year'].unique():
            year_data = result[result['year'] == year]

            # Find max week where any player has a real manager (not Unrostered/blank/null)
            rostered = year_data[
                ~year_data['manager'].isin(['Unrostered', 'No Manager', ''])
                & year_data['manager'].notna()
            ]
            if len(rostered) > 0:
                season_end_weeks[year] = int(rostered['week'].max())
            else:
                # Fallback to max week if no rostered players found
                season_end_weeks[year] = int(year_data['week'].max())

        # Log detected championship weeks for debugging
        print(f"  Championship weeks detected: {dict(sorted(season_end_weeks.items()))}")

        # Filter to only rows at the season end week for each year
        filtered_rows = []
        for year, end_week in season_end_weeks.items():
            year_end_data = result[(result['year'] == year) & (result['week'] == end_week)]
            filtered_rows.append(year_end_data)

        if filtered_rows:
            season_end_df = pd.concat(filtered_rows, ignore_index=False)
            # Get one row per player/manager/year (in case of duplicates)
            idx = season_end_df.groupby(['yahoo_player_id', 'manager', 'year'])['week'].idxmax()
            season_df = season_end_df.loc[idx].copy()
        else:
            # Fallback to old behavior if something went wrong
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
        # Handle NA values explicitly - pd.NA == 1 returns pd.NA, not False
        is_keeper_val = row.get('is_keeper_status')
        is_keeper = pd.notna(is_keeper_val) and is_keeper_val == 1
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
    Calculate PROJECTED NEXT YEAR keeper prices for ALL rostered players.

    This calculates what each player would cost if kept NEXT year, helping
    managers make keeper decisions. The keeper_price reflects the projected
    cost, not the current year's cost.

    Args:
        ctx: LeagueContext with keeper_rules
        player_path: Path to player.parquet
        dry_run: If True, don't write changes

    Returns:
        Dict with statistics
    """
    print("=" * 60)
    print("KEEPER ECONOMICS (Projected Next Year Prices)")
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

    # Calculate consecutive keeper years for historical tracking
    print("\nCalculating consecutive keeper years...")
    player = calculate_consecutive_keeper_years(player)

    keepers = player[player['keeper_year'] > 0]
    print(f"  Found {len(keepers):,} historical keeper records")

    # Initialize calculator
    calculator = KeeperPriceCalculator(ctx.keeper_rules)

    # Calculate PROJECTED NEXT YEAR keeper prices for ALL rostered players
    print("\nCalculating projected next year keeper prices for ALL players...")
    player['keeper_price'] = 0

    # Get end-of-season rows for each player/manager/year
    # Filter to rostered players only
    rostered_mask = (
        player['manager'].notna() &
        ~player['manager'].isin(['Unrostered', 'No Manager', ''])
    )

    # Get the last week for each player/manager/year combo
    rostered = player[rostered_mask].copy()
    if len(rostered) == 0:
        print("  [WARNING] No rostered players found")
        return {'status': 'error', 'reason': 'no_rostered_players'}

    # Group by player/manager/year and get max week
    end_of_season_idx = rostered.groupby(['yahoo_player_id', 'manager', 'year'])['week'].idxmax()

    prices_calculated = 0

    for idx in end_of_season_idx:
        row = player.loc[idx]

        # Get current keeper history
        current_keeper_year = int(row.get('keeper_year', 0) or 0)
        base_cost = float(row.get('base_keeper_cost', 0) or 0)
        cost = float(row.get('cost', 0) or 0)
        faab = float(row.get('max_faab_bid_to_date', 0) or 0)

        # Calculate base cost if not already set
        if base_cost <= 0:
            base_cost = calculator.calculate_base_cost(cost, faab, cost > 0)

        # Project NEXT YEAR's keeper price
        # If never kept (keeper_year=0), next year would be year 1
        # If kept 1 year, next year would be year 2
        # etc.
        projected_keeper_year = current_keeper_year + 1

        projected_price = calculator.calculate_keeper_price(base_cost, projected_keeper_year)

        # Set the projected price for ALL weeks of this player/manager/year
        player_mask = (
            (player['yahoo_player_id'] == row['yahoo_player_id']) &
            (player['manager'] == row['manager']) &
            (player['year'] == row['year'])
        )
        player.loc[player_mask, 'keeper_price'] = projected_price
        player.loc[player_mask, 'base_keeper_cost'] = base_cost

        prices_calculated += 1

    print(f"  Calculated projected prices for {prices_calculated:,} player/manager/year combinations")

    # Stats
    prices_set = (player['keeper_price'] > 0).sum()
    print(f"  Total rows with keeper_price: {prices_set:,}")

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
    player_path = ctx.data_directory / "player.parquet"

    result = calculate_keeper_economics(ctx, player_path, dry_run=args.dry_run)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
