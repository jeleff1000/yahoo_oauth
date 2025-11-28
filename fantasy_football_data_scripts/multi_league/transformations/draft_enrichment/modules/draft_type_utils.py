"""
Draft Type Utilities Module

Provides shared functions for detecting and handling different draft types
(auction vs snake) dynamically from the data itself.

Key Features:
- Per-year draft type detection
- Handles mixed datasets (some years auction, some snake)
- Auto-selects appropriate columns based on draft type
- Budget detection for auction drafts

Draft Type Detection Logic:
1. If 'draft_type' column exists:
   - 'live', 'auction', 'offline' = auction draft
   - 'self', 'snake' = snake draft
2. Fallback: Check if cost column has non-null/non-zero values
   - 25%+ picks with cost > 0 = auction
   - Otherwise = snake
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List, Union

# Draft type constants
AUCTION_TYPES = ['live', 'auction', 'offline']
SNAKE_TYPES = ['self', 'snake', 'autopick']


def detect_draft_type_for_year(
    df: pd.DataFrame,
    year: int,
    year_column: str = 'year',
    draft_type_column: str = 'draft_type',
    cost_column: str = 'cost'
) -> str:
    """
    Detect draft type for a specific year.

    Args:
        df: Draft DataFrame
        year: Year to detect draft type for
        year_column: Column containing year
        draft_type_column: Column containing explicit draft type
        cost_column: Column containing auction cost

    Returns:
        'auction' or 'snake'
    """
    year_df = df[df[year_column] == year]

    if year_df.empty:
        return 'snake'  # Default fallback

    # Method 1: Check draft_type column
    if draft_type_column in year_df.columns:
        draft_types = year_df[draft_type_column].dropna().str.lower().unique()
        for dtype in draft_types:
            if dtype in AUCTION_TYPES:
                return 'auction'
            if dtype in SNAKE_TYPES:
                return 'snake'

    # Method 2: Heuristic based on cost column
    if cost_column in year_df.columns:
        cost = pd.to_numeric(year_df[cost_column], errors='coerce')
        nonzero_cost = cost.notna() & (cost > 0)
        nonzero_count = nonzero_cost.sum()
        threshold = max(1, int(len(year_df) * 0.25))
        if nonzero_count >= threshold:
            return 'auction'

    return 'snake'


def detect_draft_type_per_year(
    df: pd.DataFrame,
    year_column: str = 'year',
    draft_type_column: str = 'draft_type',
    cost_column: str = 'cost'
) -> pd.Series:
    """
    Detect draft type for each year and return a mapping.

    Args:
        df: Draft DataFrame
        year_column: Column containing year
        draft_type_column: Column containing explicit draft type
        cost_column: Column containing auction cost

    Returns:
        Series with year as index and 'auction'/'snake' as values
    """
    years = df[year_column].dropna().unique()
    draft_types = {}

    for year in years:
        draft_types[year] = detect_draft_type_for_year(
            df, year, year_column, draft_type_column, cost_column
        )

    return pd.Series(draft_types)


def add_draft_type_column(
    df: pd.DataFrame,
    output_column: str = 'detected_draft_type',
    year_column: str = 'year',
    draft_type_column: str = 'draft_type',
    cost_column: str = 'cost'
) -> pd.DataFrame:
    """
    Add a column with detected draft type per row (based on year).

    Args:
        df: Draft DataFrame
        output_column: Name for the output column
        year_column: Column containing year
        draft_type_column: Column containing explicit draft type
        cost_column: Column containing auction cost

    Returns:
        DataFrame with detected_draft_type column added
    """
    df = df.copy()

    # Get draft type per year
    year_types = detect_draft_type_per_year(df, year_column, draft_type_column, cost_column)

    # Map to each row
    df[output_column] = df[year_column].map(year_types)

    return df


def get_cost_column_for_year(
    df: pd.DataFrame,
    year: int,
    year_column: str = 'year',
    cost_column: str = 'cost',
    pick_column: str = 'pick'
) -> str:
    """
    Get the appropriate 'cost' column for a given year.

    For auction: use actual cost
    For snake: use pick number as proxy for draft capital

    Args:
        df: Draft DataFrame
        year: Year to check
        year_column: Column containing year
        cost_column: Column containing auction cost
        pick_column: Column containing pick number

    Returns:
        Column name to use for draft capital
    """
    draft_type = detect_draft_type_for_year(df, year, year_column)

    if draft_type == 'auction' and cost_column in df.columns:
        return cost_column
    elif pick_column in df.columns:
        return pick_column
    else:
        return cost_column  # Fallback


def get_peer_group_column(
    df: pd.DataFrame,
    year: Optional[int] = None,
    year_column: str = 'year',
    cost_bucket_column: str = 'cost_bucket',
    round_column: str = 'round'
) -> str:
    """
    Get the appropriate peer group column for draft evaluation.

    For auction: use cost_bucket (groups by similar spending)
    For snake: use round (groups by similar draft capital)

    Args:
        df: Draft DataFrame
        year: Specific year (if None, uses most common draft type)
        year_column: Column containing year
        cost_bucket_column: Column containing cost buckets
        round_column: Column containing round number

    Returns:
        Column name to use for peer grouping
    """
    if year is not None:
        draft_type = detect_draft_type_for_year(df, year, year_column)
    else:
        # Get most common draft type across all years
        year_types = detect_draft_type_per_year(df, year_column)
        draft_type = year_types.mode().iloc[0] if not year_types.empty else 'snake'

    if draft_type == 'auction' and cost_bucket_column in df.columns:
        return cost_bucket_column
    elif round_column in df.columns:
        return round_column
    elif cost_bucket_column in df.columns:
        return cost_bucket_column
    else:
        return round_column


def get_rank_delta_column_for_year(
    df: pd.DataFrame,
    year: int,
    year_column: str = 'year'
) -> str:
    """
    Get the appropriate rank delta column for a given year.

    For auction: use price_rank_vs_finish_rank
    For snake: use pick_rank_vs_finish_rank

    Args:
        df: Draft DataFrame
        year: Year to check
        year_column: Column containing year

    Returns:
        Column name for rank delta
    """
    draft_type = detect_draft_type_for_year(df, year, year_column)

    if draft_type == 'auction':
        return 'price_rank_vs_finish_rank'
    else:
        return 'pick_rank_vs_finish_rank'


def detect_budget_for_year(
    df: pd.DataFrame,
    year: int,
    year_column: str = 'year',
    cost_column: str = 'cost',
    default_budget: int = 200
) -> int:
    """
    Detect the auction budget for a specific year.

    Uses the maximum total cost spent by any manager as a proxy.

    Args:
        df: Draft DataFrame
        year: Year to detect budget for
        year_column: Column containing year
        cost_column: Column containing auction cost
        default_budget: Default budget if unable to detect

    Returns:
        Detected budget amount
    """
    year_df = df[df[year_column] == year]

    if year_df.empty or cost_column not in year_df.columns:
        return default_budget

    cost = pd.to_numeric(year_df[cost_column], errors='coerce').fillna(0)

    if cost.sum() == 0:
        return default_budget

    # If we have manager column, sum by manager to find total spent
    if 'manager' in year_df.columns:
        manager_totals = year_df.groupby('manager')[cost_column].apply(
            lambda x: pd.to_numeric(x, errors='coerce').sum()
        )
        max_spent = manager_totals.max()
    else:
        # Assume ~10 teams, divide total cost
        max_spent = cost.sum() / 10

    # Round to common budget amounts
    if max_spent > 0:
        # Budgets are typically round numbers: 100, 150, 200, 250, etc.
        for budget in [100, 150, 200, 250, 300]:
            if abs(max_spent - budget) <= budget * 0.15:  # Within 15%
                return budget

        # If no match, round to nearest 50
        return int(round(max_spent / 50) * 50)

    return default_budget


def get_normalized_cost(
    df: pd.DataFrame,
    year_column: str = 'year',
    cost_column: str = 'cost',
    pick_column: str = 'pick',
    round_column: str = 'round',
    default_budget: int = 200
) -> pd.Series:
    """
    Get normalized draft cost that works for both auction and snake.

    For auction: cost_norm = actual cost
    For snake: cost_norm = budget * exp(-decay * (pick - 1))
              where decay makes last pick ≈ $1

    Args:
        df: Draft DataFrame
        year_column: Column containing year
        cost_column: Column containing auction cost
        pick_column: Column containing pick number
        round_column: Column containing round number
        default_budget: Default auction budget

    Returns:
        Series with normalized cost values
    """
    cost_norm = pd.Series(index=df.index, dtype=float)

    for year in df[year_column].dropna().unique():
        year_mask = df[year_column] == year
        year_df = df[year_mask]

        draft_type = detect_draft_type_for_year(df, year, year_column)

        if draft_type == 'auction':
            # Auction: use actual cost
            cost_norm.loc[year_mask] = pd.to_numeric(
                year_df[cost_column], errors='coerce'
            ).fillna(0)
        else:
            # Snake: exponential decay from pick number
            budget = default_budget

            if pick_column in year_df.columns:
                pick = pd.to_numeric(year_df[pick_column], errors='coerce').fillna(1)
                max_pick = pick.max() if pick.max() > 0 else 150

                # decay = -ln(1/budget) / max_pick (so last pick ≈ $1)
                decay_rate = -np.log(1.0 / budget) / max_pick if max_pick > 0 else 0.05

                cost_norm.loc[year_mask] = budget * np.exp(-decay_rate * (pick - 1))
            else:
                # Fallback to round-based
                if round_column in year_df.columns:
                    round_num = pd.to_numeric(year_df[round_column], errors='coerce').fillna(1)
                    max_round = round_num.max() if round_num.max() > 0 else 15
                    decay_rate = -np.log(1.0 / budget) / max_round
                    cost_norm.loc[year_mask] = budget * np.exp(-decay_rate * (round_num - 1))
                else:
                    cost_norm.loc[year_mask] = budget / 2  # Default middle value

    # Ensure minimum cost
    return cost_norm.clip(lower=0.1)


def get_weight_method_for_year(
    df: pd.DataFrame,
    year: int,
    year_column: str = 'year'
) -> str:
    """
    Get the appropriate weight method for a given year.

    For auction: 'cost_weighted' (weight by normalized cost)
    For snake: 'exponential' (weight by round with decay)

    Args:
        df: Draft DataFrame
        year: Year to check
        year_column: Column containing year

    Returns:
        Weight method string
    """
    draft_type = detect_draft_type_for_year(df, year, year_column)

    if draft_type == 'auction':
        return 'cost_weighted'
    else:
        return 'exponential'


def summarize_draft_types(
    df: pd.DataFrame,
    year_column: str = 'year',
    draft_type_column: str = 'draft_type',
    cost_column: str = 'cost'
) -> pd.DataFrame:
    """
    Generate a summary of detected draft types per year.

    Useful for debugging and validation.

    Args:
        df: Draft DataFrame
        year_column: Column containing year
        draft_type_column: Column containing explicit draft type
        cost_column: Column containing auction cost

    Returns:
        DataFrame with year, detected_type, explicit_type, has_cost, budget
    """
    years = sorted(df[year_column].dropna().unique())
    summary = []

    for year in years:
        year_df = df[df[year_column] == year]

        detected = detect_draft_type_for_year(df, year, year_column, draft_type_column, cost_column)

        explicit = None
        if draft_type_column in year_df.columns:
            explicit = year_df[draft_type_column].dropna().iloc[0] if len(year_df[draft_type_column].dropna()) > 0 else None

        cost = pd.to_numeric(year_df[cost_column], errors='coerce') if cost_column in year_df.columns else pd.Series()
        has_cost = cost.notna().any() and (cost > 0).any()
        cost_mean = cost.mean() if has_cost else 0

        budget = detect_budget_for_year(df, year, year_column, cost_column) if detected == 'auction' else None

        summary.append({
            'year': year,
            'detected_type': detected,
            'explicit_type': explicit,
            'has_cost': has_cost,
            'avg_cost': round(cost_mean, 2) if has_cost else None,
            'budget': budget,
            'picks': len(year_df)
        })

    return pd.DataFrame(summary)
