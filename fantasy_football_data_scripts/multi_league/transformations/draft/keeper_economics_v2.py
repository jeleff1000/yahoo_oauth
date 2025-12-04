"""
Keeper Economics Transformation (Multi-League V2)

Calculates keeper prices using configurable rules from league_context.json.

This transformation:
- Reads keeper rules from league context (formulas, limits, draft type handling)
- Calculates keeper price based on acquisition type and keeper year
- Adds keeper economics to draft data for value analysis

Key Features:
- Fully configurable via league_context.json keeper_rules
- Supports auction and snake draft types automatically
- Handles FAAB acquisitions, free agents, and drafted players
- Year-over-year keeper price escalation (year 1, year 2+, etc.)

Usage:
    python keeper_economics_v2.py --context path/to/league_context.json
    python keeper_economics_v2.py --context path/to/league_context.json --dry-run
"""

import argparse
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import sys

import pandas as pd
import numpy as np


# Add parent directories to path for imports
_script_file = Path(__file__).resolve()

if _script_file.parent.name == 'draft_enrichment':
    _draft_enrichment_dir = _script_file.parent
    _transformations_dir = _draft_enrichment_dir.parent
    _multi_league_dir = _transformations_dir.parent
elif _script_file.parent.name == 'transformations':
    _transformations_dir = _script_file.parent
    _multi_league_dir = _transformations_dir.parent
else:
    _current = _script_file.parent
    while _current.name != 'multi_league' and _current.parent != _current:
        _current = _current.parent
    _multi_league_dir = _current

_scripts_dir = _multi_league_dir.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))

from multi_league.core.league_context import LeagueContext

# Import consecutive keeper calculator
try:
    from multi_league.transformations.draft.modules.consecutive_keeper_calculator import (
        calculate_consecutive_keeper_years
    )
except ImportError:
    # Fallback if consecutive_keeper_calculator not available
    def calculate_consecutive_keeper_years(df, player_id_col='yahoo_player_id', keeper_col='is_keeper_status',
                                           year_col='year', manager_col=None):
        """Fallback: simple keeper year calculation without module."""
        result = df.copy()
        result['consecutive_years_kept'] = 0
        result['first_kept_year'] = pd.NA
        result['keeper_streak_id'] = pd.NA

        if keeper_col not in result.columns:
            return result

        # Simple calculation: just mark keepers as year 1
        result.loc[result[keeper_col] == 1, 'consecutive_years_kept'] = 1
        return result

# Fallback draft type detection if module not available
try:
    from multi_league.transformations.draft.modules.draft_type_utils import detect_draft_type_for_year
except ImportError:
    def detect_draft_type_for_year(df, year, year_column='year', draft_type_column='draft_type', cost_column='cost'):
        """Fallback draft type detection."""
        year_df = df[df[year_column] == year]
        if year_df.empty:
            return 'snake'
        if draft_type_column in year_df.columns:
            draft_types = year_df[draft_type_column].dropna().str.lower().unique()
            for dtype in draft_types:
                if dtype in ['live', 'auction', 'offline']:
                    return 'auction'
                if dtype in ['self', 'snake', 'autopick']:
                    return 'snake'
        if cost_column in year_df.columns:
            cost = pd.to_numeric(year_df[cost_column], errors='coerce')
            nonzero_cost = cost.notna() & (cost > 0)
            if nonzero_cost.sum() >= max(1, int(len(year_df) * 0.25)):
                return 'auction'
        return 'snake'


# =========================================================
# Formula Evaluation Engine
# =========================================================

class KeeperFormulaEvaluator:
    """
    Safe expression evaluator for keeper price formulas.

    Supports variables:
    - base_cost: Base acquisition cost (draft cost, FAAB, etc.)
    - cost: Alias for base_cost
    - faab_bid: Maximum FAAB bid on player
    - previous_keeper_price: Last year's keeper price
    - prev_cost: Alias for previous_keeper_price (for escalation formulas)
    - pick: Draft pick number (snake)
    - round: Draft round number
    - budget: League auction budget
    - keeper_year: How many years player has been kept

    Supports functions:
    - max(a, b, ...), min(a, b, ...)
    - exp(x), log(x), sqrt(x)
    - floor(x), ceil(x), round(x)
    """

    # Safe math functions
    SAFE_FUNCTIONS = {
        'max': max,
        'min': min,
        'exp': math.exp,
        'log': math.log,
        'sqrt': math.sqrt,
        'floor': math.floor,
        'ceil': math.ceil,
        'round': round,
        'abs': abs,
    }

    def __init__(self, budget: int = 200):
        self.budget = budget

    def evaluate(
        self,
        expression: str,
        base_cost: float = 0.0,
        faab_bid: float = 0.0,
        previous_keeper_price: float = 0.0,
        pick: int = 0,
        round_num: int = 0,
        keeper_year: int = 1,
    ) -> float:
        """
        Evaluate a keeper price formula expression.

        Args:
            expression: Formula string (e.g., "max(base_cost, faab_bid * 0.5)")
            base_cost: Base acquisition cost
            faab_bid: Maximum FAAB bid
            previous_keeper_price: Previous year's keeper price
            pick: Draft pick number
            round_num: Draft round number
            keeper_year: Number of years player has been kept

        Returns:
            Calculated keeper price (float)
        """
        # Build variable context
        variables = {
            'base_cost': float(base_cost) if not pd.isna(base_cost) else 0.0,
            'cost': float(base_cost) if not pd.isna(base_cost) else 0.0,
            'faab_bid': float(faab_bid) if not pd.isna(faab_bid) else 0.0,
            'previous_keeper_price': float(previous_keeper_price) if not pd.isna(previous_keeper_price) else 0.0,
            'prev_cost': float(previous_keeper_price) if not pd.isna(previous_keeper_price) else 0.0,  # Alias for formulas
            'pick': int(pick) if not pd.isna(pick) else 0,
            'round': int(round_num) if not pd.isna(round_num) else 0,
            'budget': self.budget,
            'keeper_year': int(keeper_year) if not pd.isna(keeper_year) else 1,
        }

        # Create safe evaluation context
        eval_context = {**self.SAFE_FUNCTIONS, **variables}

        try:
            # Evaluate expression safely
            result = eval(expression, {"__builtins__": {}}, eval_context)
            return float(result)
        except Exception as e:
            print(f"[WARNING] Formula evaluation failed: {expression}")
            print(f"          Error: {e}")
            print(f"          Variables: {variables}")
            return 0.0


# =========================================================
# Keeper Price Calculator
# =========================================================

class KeeperPriceCalculator:
    """
    Calculates keeper prices using rules from league context or raw rules dict.
    """

    def __init__(self, ctx_or_rules):
        """
        Initialize calculator.

        Args:
            ctx_or_rules: Either a LeagueContext or a raw keeper_rules dict
        """
        # Accept either LeagueContext or raw rules dict
        if hasattr(ctx_or_rules, 'keeper_rules'):
            # It's a LeagueContext
            self.ctx = ctx_or_rules
            self.rules = ctx_or_rules.keeper_rules or {}
            budget = ctx_or_rules.keeper_budget
        else:
            # It's a raw rules dict
            self.ctx = None
            self.rules = ctx_or_rules or {}
            budget = self.rules.get('budget', 200)

        self.evaluator = KeeperFormulaEvaluator(budget)

        # Extract settings
        self.enabled = self.rules.get('enabled', False)
        self.max_keepers = self.rules.get('max_keepers', 0)
        self.min_price = self.rules.get('min_price', 1)
        self.max_price = self.rules.get('max_price')
        self.round_to_integer = self.rules.get('round_to_integer', True)
        self.formulas = self.rules.get('formulas_by_keeper_year', {})
        self.base_cost_rules = self.rules.get('base_cost_rules', {})

    def get_formula_for_keeper_year(self, keeper_year: int) -> Optional[str]:
        """Get the formula expression for a specific keeper year."""
        # Check exact match first
        if str(keeper_year) in self.formulas:
            return self.formulas[str(keeper_year)].get('expression')

        # Check wildcard patterns (e.g., "2+")
        for key, formula_dict in self.formulas.items():
            if '+' in key:
                base_year = int(key.replace('+', ''))
                if keeper_year >= base_year:
                    return formula_dict.get('expression')

        return None

    def calculate_base_cost(
        self,
        draft_type: str,
        cost: float,
        pick: int,
        faab_bid: float,
        is_drafted: bool,
    ) -> float:
        """
        Calculate base cost based on acquisition type.

        Args:
            draft_type: 'auction' or 'snake'
            cost: Draft cost (auction) or 0
            pick: Pick number (snake) or 0
            faab_bid: Maximum FAAB bid
            is_drafted: Whether player was drafted (vs FA pickup)

        Returns:
            Base cost for keeper calculation
        """
        if is_drafted:
            # Player was drafted
            if draft_type == 'auction':
                rule = self.base_cost_rules.get('auction', {})
                source = rule.get('source', 'draft_price')

                if source == 'max_of_draft_faab':
                    # MAX(draft_adjusted, faab_adjusted)
                    draft_mult = rule.get('draft_multiplier', 1.0)
                    draft_flat = rule.get('draft_flat', 0.0)
                    faab_mult = rule.get('faab_multiplier', 0.5)
                    faab_flat = rule.get('faab_flat', 0.0)

                    draft_cost = float(cost) if not pd.isna(cost) else 0.0
                    faab_cost = float(faab_bid) if not pd.isna(faab_bid) else 0.0

                    draft_value = draft_cost * draft_mult + draft_flat
                    faab_value = faab_cost * faab_mult + faab_flat

                    return max(draft_value, faab_value)

                elif source in ('draft_price', 'cost'):
                    # Draft price with optional multiplier/flat adjustment
                    mult = rule.get('multiplier', 1.0)
                    flat = rule.get('flat', 0.0)
                    draft_cost = float(cost) if not pd.isna(cost) else 0.0
                    return draft_cost * mult + flat

                else:
                    # Fallback to raw cost
                    return float(cost) if not pd.isna(cost) else 0.0
            else:
                # Snake draft - convert pick to cost
                rule = self.base_cost_rules.get('snake', {})
                source = rule.get('source', 'pick_to_cost')
                if source == 'pick_to_cost' and 'formula' in rule:
                    formula = rule['formula']
                    return self.evaluator.evaluate(
                        formula,
                        pick=pick,
                        round_num=0,
                    )
                # Default: exponential decay
                pick_val = int(pick) if not pd.isna(pick) else 1
                return self.ctx.keeper_budget * math.exp(-0.035 * (pick_val - 1))

        elif faab_bid and faab_bid > 0:
            # FAAB acquisition (player not drafted, picked up via FAAB)
            rule = self.base_cost_rules.get('faab_only', {})
            source = rule.get('source', 'faab_bid')

            if source == 'max_of_draft_faab':
                # For FAAB pickups, draft cost is 0, so just use FAAB calculation
                faab_mult = rule.get('faab_multiplier', 0.5)
                faab_flat = rule.get('faab_flat', 0.0)
                faab_cost = float(faab_bid) if not pd.isna(faab_bid) else 0.0
                return faab_cost * faab_mult + faab_flat

            else:
                # Simple multiplier + flat
                mult = rule.get('multiplier', 0.5)
                flat = rule.get('flat', 0.0)
                return float(faab_bid) * mult + flat

        else:
            # Free agent
            rule = self.base_cost_rules.get('free_agent', {})
            return rule.get('value', self.min_price)

    def calculate_keeper_price(
        self,
        draft_type: str,
        cost: float,
        pick: int,
        round_num: int,
        faab_bid: float,
        is_keeper: bool,
        previous_keeper_price: float,
        keeper_year: int,
    ) -> float:
        """
        Calculate keeper price using configured rules.

        Args:
            draft_type: 'auction' or 'snake'
            cost: Draft/keeper cost
            pick: Pick number
            round_num: Round number
            faab_bid: Maximum FAAB bid
            is_keeper: Whether this is a keeper (not fresh draft)
            previous_keeper_price: Last year's keeper price (if kept before)
            keeper_year: How many years player has been kept (1 = first time)

        Returns:
            Calculated keeper price
        """
        if not self.enabled:
            # Fallback to simple calculation
            return max(float(cost) if not pd.isna(cost) else 0, 1)

        # Determine if player was drafted or acquired via FA/FAAB
        is_drafted = (
            (draft_type == 'auction' and cost and cost > 0) or
            (draft_type == 'snake' and pick and pick > 0)
        )

        # Calculate base cost
        base_cost = self.calculate_base_cost(
            draft_type=draft_type,
            cost=cost,
            pick=pick,
            faab_bid=faab_bid,
            is_drafted=is_drafted,
        )

        # Get formula for keeper year
        formula = self.get_formula_for_keeper_year(keeper_year)

        if formula is None:
            # No formula configured - use base cost
            price = base_cost
        else:
            # Evaluate formula
            price = self.evaluator.evaluate(
                expression=formula,
                base_cost=base_cost,
                faab_bid=faab_bid,
                previous_keeper_price=previous_keeper_price,
                pick=pick,
                round_num=round_num,
                keeper_year=keeper_year,
            )

        # Apply min/max constraints
        price = max(price, self.min_price)
        if self.max_price is not None:
            price = min(price, self.max_price)

        # Round if configured
        if self.round_to_integer:
            price = int(round(price))

        return price


# =========================================================
# Helper Functions
# =========================================================

def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    dest.write_bytes(path.read_bytes())
    return dest


# =========================================================
# Main Transformation Function
# =========================================================

def calculate_keeper_economics(
    ctx: LeagueContext,
    draft_path: Path,
    dry_run: bool = False,
    make_backup: bool = False,
) -> Dict[str, int]:
    """
    Calculate keeper economics using rules from league context.

    Args:
        ctx: LeagueContext with keeper_rules configured
        draft_path: Path to draft.parquet
        dry_run: If True, don't write changes
        make_backup: If True, create backup before writing

    Returns:
        Dict with statistics
    """
    print("=" * 70)
    print("KEEPER ECONOMICS CALCULATION (V2 - Context-Driven)")
    print("=" * 70)

    # Check if keepers are enabled
    if not ctx.keepers_enabled:
        print(f"\n[SKIP] Keepers not enabled for {ctx.league_name}")
        print("       Add keeper_rules to league_context.json to enable")
        return {"skipped": True, "reason": "keepers_not_enabled"}

    print(f"\n[League] {ctx.league_name}")
    print(f"[Keepers] Max: {ctx.max_keepers}, Budget: ${ctx.keeper_budget}")

    # Initialize calculator
    calculator = KeeperPriceCalculator(ctx)

    # Load draft data
    print(f"\n[Loading] Draft data from: {draft_path}")
    draft = pd.read_parquet(draft_path)
    print(f"   Loaded {len(draft):,} draft records")

    # Get draft type per year
    years = draft['year'].dropna().unique()
    draft_types = {}
    for year in sorted(years):
        draft_types[year] = detect_draft_type_for_year(draft, year)

    print(f"\n[Draft Types]")
    for year, dtype in sorted(draft_types.items()):
        print(f"   {year}: {dtype}")

    # Ensure required columns exist
    if 'cost' not in draft.columns:
        draft['cost'] = 0.0
    if 'pick' not in draft.columns:
        draft['pick'] = pd.NA
    if 'round' not in draft.columns:
        draft['round'] = pd.NA
    if 'is_keeper_status' not in draft.columns:
        draft['is_keeper_status'] = 0

    # Get max FAAB bid per player-year if not already present
    if 'max_faab_bid' not in draft.columns:
        draft['max_faab_bid'] = 0.0

    # Calculate consecutive keeper years using the dedicated module
    # This properly tracks per (player, manager) streaks
    print(f"\n[Calculating] Consecutive keeper years...")

    # Use the consecutive keeper calculator module
    draft = calculate_consecutive_keeper_years(
        draft,
        player_id_col='yahoo_player_id',
        keeper_col='is_keeper_status',
        year_col='year',
        manager_col='manager' if 'manager' in draft.columns else None
    )

    # Rename for consistency with keeper_year expected by calculator
    draft['keeper_year'] = draft['consecutive_years_kept']

    print(f"   Found {(draft['keeper_year'] > 0).sum():,} keeper instances")
    print(f"   Max consecutive years: {draft['keeper_year'].max()}")

    # Add draft_type column based on year
    draft['detected_draft_type'] = draft['year'].map(draft_types)

    # Initialize output columns
    draft['keeper_price'] = pd.NA
    draft['previous_keeper_price'] = pd.NA

    # Build keeper price history for escalation calculations
    # We need to track prices per (player_id, manager) to get previous year's price
    print(f"\n[Calculating] Keeper prices...")

    # Sort by player, manager, year for proper sequential processing
    sort_cols = ['yahoo_player_id', 'year']
    if 'manager' in draft.columns:
        sort_cols = ['yahoo_player_id', 'manager', 'year']
    draft = draft.sort_values(sort_cols).copy()

    # Track keeper prices: {(player_id, manager): {year: price}}
    keeper_price_history = {}
    keeper_prices_calculated = 0

    for idx, row in draft.iterrows():
        player_id = str(row.get('yahoo_player_id', ''))
        year = row.get('year')
        manager = str(row.get('manager', '')) if 'manager' in draft.columns else ''
        keeper_year = int(row.get('keeper_year', 0))

        if pd.isna(year) or not player_id:
            continue

        year = int(year)
        draft_type = draft_types.get(year, 'snake')

        # Create key for tracking
        key = (player_id, manager)

        if key not in keeper_price_history:
            keeper_price_history[key] = {}

        # Get previous keeper price from history
        prev_year = year - 1
        previous_keeper_price = keeper_price_history[key].get(prev_year, 0.0)

        # Calculate keeper price
        cost = float(row.get('cost', 0)) if not pd.isna(row.get('cost')) else 0.0
        pick = row.get('pick')
        round_num = row.get('round')
        faab_bid = float(row.get('max_faab_bid', 0)) if not pd.isna(row.get('max_faab_bid')) else 0.0

        # Only calculate keeper_price for drafted players (not undrafted pool)
        is_drafted = pd.notna(row.get('pick')) or (cost > 0)
        is_keeper = keeper_year > 0

        if is_drafted:
            keeper_price = calculator.calculate_keeper_price(
                draft_type=draft_type,
                cost=cost,
                pick=int(pick) if pd.notna(pick) else 0,
                round_num=int(round_num) if pd.notna(round_num) else 0,
                faab_bid=faab_bid,
                is_keeper=is_keeper,
                previous_keeper_price=previous_keeper_price,
                keeper_year=keeper_year if keeper_year > 0 else 1,  # Default to year 1 formula
            )

            # Store in history for next year's calculation
            keeper_price_history[key][year] = keeper_price

            # Update dataframe
            draft.at[idx, 'keeper_price'] = keeper_price
            draft.at[idx, 'previous_keeper_price'] = previous_keeper_price if previous_keeper_price > 0 else pd.NA

            keeper_prices_calculated += 1

    print(f"   Calculated keeper_price for {keeper_prices_calculated:,} drafted players")

    # Calculate statistics
    drafted_mask = draft['keeper_price'].notna()
    keeper_mask = draft['is_keeper_status'] == 1

    stats = {
        "total_records": len(draft),
        "drafted_players": drafted_mask.sum(),
        "keepers": keeper_mask.sum(),
        "with_keeper_price": (draft['keeper_price'].notna()).sum(),
    }

    # Show sample output
    print(f"\n[Sample] Keeper prices calculated:")
    sample_cols = ['year', 'player', 'detected_draft_type', 'cost', 'pick', 'is_keeper_status',
                   'keeper_year', 'previous_keeper_price', 'keeper_price']
    sample_cols = [c for c in sample_cols if c in draft.columns]

    sample_df = draft[draft['keeper_price'].notna()][sample_cols].head(15)
    print(sample_df.to_string(index=False))

    # Show keeper escalation examples
    keepers = draft[(draft['is_keeper_status'] == 1) & (draft['keeper_price'].notna())]
    if len(keepers) > 0:
        print(f"\n[Sample] Keeper price escalation (is_keeper_status=1):")
        print(keepers[sample_cols].head(10).to_string(index=False))

    # Save results
    if dry_run:
        print("\n" + "=" * 70)
        print("[DRY RUN] No files were written.")
        print("=" * 70)
        return stats

    if make_backup and draft_path.exists():
        bpath = backup_file(draft_path)
        print(f"\n[Backup Created] {bpath}")

    # Clean up temporary column
    if 'detected_draft_type' in draft.columns:
        draft = draft.drop(columns=['detected_draft_type'])

    # Write back
    draft.to_parquet(draft_path, index=False)
    print(f"\n[SAVED] Updated draft data written to: {draft_path}")

    csv_path = draft_path.with_suffix('.csv')
    draft.to_csv(csv_path, index=False)
    print(f"[SAVED] CSV version written to: {csv_path}")

    print("\n" + "=" * 70)
    print("KEEPER ECONOMICS COMPLETE")
    print("=" * 70)
    print(f"Total records:      {stats['total_records']:,}")
    print(f"Drafted players:    {stats['drafted_players']:,}")
    print(f"Keepers:            {stats['keepers']:,}")
    print(f"With keeper_price:  {stats['with_keeper_price']:,}")

    return stats


# =========================================================
# CLI Interface
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate keeper economics using rules from league context"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Path to league_context.json"
    )
    parser.add_argument(
        "--draft",
        type=str,
        help="Override draft data path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create timestamped backup before saving"
    )

    args = parser.parse_args()

    # Load context
    ctx = LeagueContext.load(args.context)
    print(f"Loaded league context: {ctx.league_name}")

    # Determine paths
    draft_path = Path(args.draft) if args.draft else ctx.canonical_draft_file

    if not draft_path.exists():
        raise FileNotFoundError(f"Draft data not found: {draft_path}")

    # Run calculation
    calculate_keeper_economics(
        ctx=ctx,
        draft_path=draft_path,
        dry_run=args.dry_run,
        make_backup=args.backup
    )


if __name__ == "__main__":
    main()
