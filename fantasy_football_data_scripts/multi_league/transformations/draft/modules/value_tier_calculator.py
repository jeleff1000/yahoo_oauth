"""
Value Tier Calculator Module

Calculates draft value tiers (Steal/Good/Fair/Reach/Bust) based on the difference
between draft position and actual season performance rank.

Works dynamically for both auction and snake drafts:
- Auction: uses price_rank_vs_finish_rank (cost rank - season position rank)
- Snake: uses pick_rank_vs_finish_rank (pick rank - season position rank)

Key Concepts:
- Positive delta = outperformed draft position (Steal/Good Value)
- Negative delta = underperformed draft position (Reach/Bust)
- Zero/small delta = performed as expected (Fair)

Draft Type Handling:
- Automatically detects auction vs snake PER YEAR from data
- Handles mixed datasets (some years auction, some snake)
- Uses appropriate delta column based on each row's year

Configuration:
- Tier bins and labels are fully configurable
- Default: Bust (<-5), Reach (-5 to -2), Fair (-2 to +2), Good (+2 to +5), Steal (>+5)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple

from .draft_type_utils import (
    detect_draft_type_for_year,
    detect_draft_type_per_year,
    get_rank_delta_column_for_year,
    AUCTION_TYPES,
    SNAKE_TYPES
)


# Default tier configuration
# Bins represent: [lower_bound, upper_bound) for each tier
# Positive values = outperformed, negative = underperformed
#
# IMPORTANT: These thresholds are calibrated for fantasy football where:
# - A -5 delta (drafted WR1, finished WR6) is still a GREAT pick
# - A -15 delta (drafted WR1, finished WR16) is actually a bust
# - We also protect top finishers: top-12 position finish = never a Bust
#
DEFAULT_VALUE_BINS = [-100, -15, -5, 5, 15, 100]
DEFAULT_VALUE_LABELS = ['Bust', 'Overpay', 'Fair', 'Good Value', 'Steal']

# Position finish rank threshold - if you finish this rank or better,
# you can't be called a "Bust" regardless of what you paid
DEFAULT_FINISH_RANK_BUST_PROTECTION = 12


def detect_draft_type(
    df: pd.DataFrame,
    year: Optional[int] = None,
    year_column: str = 'year',
    cost_column: str = 'cost',
    draft_type_column: str = 'draft_type'
) -> str:
    """
    Dynamically detect whether the draft is auction or snake.

    If year is provided, detects for that specific year.
    Otherwise, returns the most common type across all years.

    Detection logic:
    1. If draft_type column exists, use it ('live'/'offline' = auction, 'self' = snake)
    2. Otherwise, check if 25%+ of picks have non-zero cost (auction)

    Args:
        df: Draft DataFrame
        year: Optional specific year to detect for
        year_column: Column containing year
        cost_column: Column containing auction cost
        draft_type_column: Column containing explicit draft type

    Returns:
        'auction' or 'snake'
    """
    if year is not None:
        return detect_draft_type_for_year(df, year, year_column, draft_type_column, cost_column)

    # For backward compatibility: return mode across all years
    year_types = detect_draft_type_per_year(df, year_column, draft_type_column, cost_column)
    if not year_types.empty:
        return year_types.mode().iloc[0]
    return 'snake'


def get_rank_delta_column(
    df: pd.DataFrame,
    draft_type: Optional[str] = None,
    year: Optional[int] = None,
    year_column: str = 'year'
) -> str:
    """
    Get the appropriate rank delta column based on draft type.

    Args:
        df: Draft DataFrame
        draft_type: 'auction' or 'snake' (auto-detected if None)
        year: Optional specific year to check
        year_column: Column containing year

    Returns:
        Column name for rank vs finish delta
    """
    if draft_type is None:
        draft_type = detect_draft_type(df, year=year, year_column=year_column)

    if draft_type == 'auction':
        return 'price_rank_vs_finish_rank'
    else:
        return 'pick_rank_vs_finish_rank'


def calculate_rank_delta(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    performance_column: str = 'total_fantasy_points',
    draft_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate rank delta (draft position rank - season finish rank) within position/year.

    This is useful if the rank delta columns don't already exist.

    Args:
        df: Draft DataFrame with draft and performance data
        group_columns: Columns to group by (default: ['year', 'position'])
        performance_column: Column to rank by for season performance
        draft_type: 'auction' or 'snake' (auto-detected if None)

    Returns:
        DataFrame with rank delta columns added
    """
    df = df.copy()

    # Detect draft type if not specified
    if draft_type is None:
        draft_type = detect_draft_type(df)

    # Determine position column
    pos_col = 'position' if 'position' in df.columns else 'yahoo_position'

    # Set default group columns
    if group_columns is None:
        group_columns = ['year', pos_col]

    # Validate required columns
    if performance_column not in df.columns:
        raise ValueError(f"Performance column '{performance_column}' not found")

    # Calculate season position rank within groups
    df['_season_position_rank'] = df.groupby(group_columns, dropna=False)[performance_column].rank(
        method='min',
        ascending=False,  # Higher points = lower rank (rank 1 = best)
        na_option='keep'
    )

    # Calculate draft position rank based on draft type
    if draft_type == 'auction':
        if 'cost' not in df.columns:
            raise ValueError("Auction draft requires 'cost' column")

        # Price rank: higher cost = lower rank number (rank 1 = most expensive)
        df['_draft_position_rank'] = df.groupby(group_columns, dropna=False)['cost'].rank(
            method='min',
            ascending=False,
            na_option='keep'
        )
        delta_column = 'price_rank_vs_finish_rank'
    else:
        if 'pick' not in df.columns:
            raise ValueError("Snake draft requires 'pick' column")

        # Pick rank: earlier pick = lower rank number (rank 1 = first picked)
        df['_pick_rank'] = df.groupby(group_columns, dropna=False)['pick'].rank(
            method='min',
            ascending=True,
            na_option='keep'
        )
        df['_draft_position_rank'] = df['_pick_rank']
        delta_column = 'pick_rank_vs_finish_rank'

    # Calculate delta: draft_rank - finish_rank
    # Positive = outperformed (drafted lower but finished higher)
    # Negative = underperformed (drafted higher but finished lower)
    df[delta_column] = df['_draft_position_rank'] - df['_season_position_rank']

    # Clean up temp columns
    temp_cols = ['_season_position_rank', '_draft_position_rank', '_pick_rank']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns])

    return df


def assign_value_tier(
    df: pd.DataFrame,
    delta_column: Optional[str] = None,
    value_bins: Optional[List[float]] = None,
    value_labels: Optional[List[str]] = None,
    draft_type: Optional[str] = None,
    year_column: str = 'year',
    per_year: bool = True,
    pick_column: str = 'pick',
    finish_rank_column: str = 'season_position_rank',
    bust_protection_threshold: int = DEFAULT_FINISH_RANK_BUST_PROTECTION
) -> pd.DataFrame:
    """
    Assign value tiers based on rank delta.

    Only assigns tiers to DRAFTED players (pick is not null).
    Undrafted players get value_tier = NA.

    Supports per-year delta column selection for mixed datasets where
    some years are auction and some are snake.

    BUST PROTECTION: Players who finish in the top N at their position
    (default: top-12) cannot be labeled "Bust" regardless of their delta.
    They get upgraded to the next tier ("Overpay").

    Args:
        df: DataFrame with rank delta column
        delta_column: Column containing rank delta (auto-detected based on draft type if None)
        value_bins: List of bin edges for tier assignment
        value_labels: List of tier labels
        draft_type: 'auction' or 'snake' (auto-detected if None)
        year_column: Column containing year
        per_year: If True, use appropriate delta column per year (handles mixed datasets)
        pick_column: Column indicating drafted status (non-null = drafted)
        finish_rank_column: Column with position finish rank (for bust protection)
        bust_protection_threshold: Top N finishers protected from "Bust" label (default: 12)

    Returns:
        DataFrame with value_tier column added
    """
    df = df.copy()

    # Use defaults if not provided
    if value_bins is None:
        value_bins = DEFAULT_VALUE_BINS
    if value_labels is None:
        value_labels = DEFAULT_VALUE_LABELS

    # Validate configuration
    if len(value_labels) != len(value_bins) - 1:
        raise ValueError(
            f"Number of labels ({len(value_labels)}) must be one less than "
            f"number of bin edges ({len(value_bins)})"
        )

    # Initialize value_tier as NA for all rows
    df['value_tier'] = pd.NA

    # Only process drafted players
    drafted_mask = df[pick_column].notna() if pick_column in df.columns else pd.Series(True, index=df.index)

    if not drafted_mask.any():
        return df

    # If delta_column is explicitly provided, use it
    if delta_column is not None:
        if delta_column not in df.columns:
            raise ValueError(
                f"Delta column '{delta_column}' not found. "
                f"Run calculate_rank_delta() first or check that player_to_draft_v2.py has run."
            )
        # Only assign to drafted players, keep NA for missing delta values
        delta_values = df.loc[drafted_mask, delta_column]
        df.loc[drafted_mask, 'value_tier'] = pd.cut(
            delta_values,
            bins=value_bins,
            labels=value_labels,
            include_lowest=True
        )
        # Apply bust protection
        df = _apply_bust_protection(df, drafted_mask, finish_rank_column, bust_protection_threshold, value_labels)
        return df

    # Per-year handling for mixed datasets
    if per_year and year_column in df.columns:
        # Get delta values using appropriate column per year
        df['_unified_delta'] = pd.NA

        for year in df[year_column].dropna().unique():
            year_mask = (df[year_column] == year) & drafted_mask
            year_draft_type = detect_draft_type(df, year=year, year_column=year_column)
            year_delta_col = get_rank_delta_column(df, draft_type=year_draft_type)

            if year_delta_col in df.columns:
                df.loc[year_mask, '_unified_delta'] = df.loc[year_mask, year_delta_col]
            else:
                # Try alternate column
                alt_col = 'pick_rank_vs_finish_rank' if year_draft_type == 'auction' else 'price_rank_vs_finish_rank'
                if alt_col in df.columns:
                    df.loc[year_mask, '_unified_delta'] = df.loc[year_mask, alt_col]

        # Assign tiers using unified delta (only for drafted players with valid delta)
        valid_delta_mask = drafted_mask & df['_unified_delta'].notna()
        if valid_delta_mask.any():
            df.loc[valid_delta_mask, 'value_tier'] = pd.cut(
                df.loc[valid_delta_mask, '_unified_delta'],
                bins=value_bins,
                labels=value_labels,
                include_lowest=True
            )
        df = df.drop(columns=['_unified_delta'])
    else:
        # Single draft type - use auto-detection
        detected_delta_column = get_rank_delta_column(df, draft_type)

        if detected_delta_column not in df.columns:
            raise ValueError(
                f"Delta column '{detected_delta_column}' not found. "
                f"Run calculate_rank_delta() first or check that player_to_draft_v2.py has run."
            )

        # Only assign to drafted players with valid delta
        valid_mask = drafted_mask & df[detected_delta_column].notna()
        if valid_mask.any():
            df.loc[valid_mask, 'value_tier'] = pd.cut(
                df.loc[valid_mask, detected_delta_column],
                bins=value_bins,
                labels=value_labels,
                include_lowest=True
            )

    # Apply bust protection after tier assignment
    df = _apply_bust_protection(df, drafted_mask, finish_rank_column, bust_protection_threshold, value_labels)

    return df


def _apply_bust_protection(
    df: pd.DataFrame,
    drafted_mask: pd.Series,
    finish_rank_column: str,
    threshold: int,
    value_labels: List[str]
) -> pd.DataFrame:
    """
    Apply bust protection: top finishers cannot be labeled "Bust".

    If a player finishes top-N at their position but would be labeled "Bust"
    based on their delta, upgrade them to the next tier (typically "Overpay").

    This handles cases like: Drafted WR1 ($69), finished WR6
    - Delta of -5 might trigger "Bust" with strict bins
    - But finishing WR6 is actually a great season, not a bust
    - So we upgrade them to "Overpay" instead

    Args:
        df: DataFrame with value_tier already assigned
        drafted_mask: Boolean mask for drafted players
        finish_rank_column: Column with position finish rank
        threshold: Top N finishers get protection (default: 12)
        value_labels: List of tier labels (first is "Bust")

    Returns:
        DataFrame with bust protection applied
    """
    if finish_rank_column not in df.columns:
        # Can't apply protection without finish rank data
        return df

    if len(value_labels) < 2:
        # Need at least 2 tiers to upgrade
        return df

    bust_label = value_labels[0]  # First label is "Bust"
    upgrade_label = value_labels[1]  # Second label is "Overpay"

    # Find players who:
    # 1. Are drafted
    # 2. Have a Bust tier assigned
    # 3. Finished in top-N at their position
    bust_mask = (
        drafted_mask &
        (df['value_tier'] == bust_label) &
        (df[finish_rank_column].notna()) &
        (df[finish_rank_column] <= threshold)
    )

    upgraded_count = bust_mask.sum()
    if upgraded_count > 0:
        df.loc[bust_mask, 'value_tier'] = upgrade_label
        print(f"  [BUST PROTECTION] Upgraded {upgraded_count} top-{threshold} finishers from '{bust_label}' to '{upgrade_label}'")

    return df


def calculate_value_tiers(
    df: pd.DataFrame,
    draft_type: Optional[str] = None,
    recalculate_delta: bool = False,
    group_columns: Optional[List[str]] = None,
    performance_column: str = 'total_fantasy_points',
    value_bins: Optional[List[float]] = None,
    value_labels: Optional[List[str]] = None,
    year_column: str = 'year',
    per_year: bool = True,
    finish_rank_column: str = 'season_position_rank',
    bust_protection_threshold: int = DEFAULT_FINISH_RANK_BUST_PROTECTION
) -> pd.DataFrame:
    """
    Calculate value tiers for draft picks in one step.

    This is the main entry point for value tier calculation.

    Supports mixed datasets where some years are auction and some are snake.
    When per_year=True (default), automatically uses the appropriate delta
    column for each year based on detected draft type.

    BUST PROTECTION: Players who finish in the top N at their position
    (default: top-12) cannot be labeled "Bust" regardless of their delta.
    A WR6 finish is not a bust, even if they were drafted as WR1.

    Args:
        df: Draft DataFrame with performance metrics
        draft_type: 'auction' or 'snake' (auto-detected if None)
        recalculate_delta: If True, recalculate rank delta even if it exists
        group_columns: Columns to group by for ranking (default: ['year', 'position'])
        performance_column: Column to rank by for season performance
        value_bins: List of bin edges for tier assignment
        value_labels: List of tier labels
        year_column: Column containing year
        per_year: If True, handle draft type per year (for mixed datasets)
        finish_rank_column: Column with position finish rank (for bust protection)
        bust_protection_threshold: Top N finishers protected from "Bust" label (default: 12)

    Returns:
        DataFrame with value_tier column added

    Example:
        # Basic usage (auto-detect draft type per year)
        df = calculate_value_tiers(df)

        # Force auction mode for all years
        df = calculate_value_tiers(df, draft_type='auction', per_year=False)

        # Custom tier thresholds
        df = calculate_value_tiers(
            df,
            value_bins=[-100, -10, -3, 3, 10, 100],
            value_labels=['Major Bust', 'Bust', 'Fair', 'Good', 'Major Steal']
        )
    """
    # Show detected draft types per year
    if per_year and year_column in df.columns:
        year_types = detect_draft_type_per_year(df, year_column)
        auction_years = [y for y, t in year_types.items() if t == 'auction']
        snake_years = [y for y, t in year_types.items() if t == 'snake']
        print(f"  [INFO] Draft types detected:")
        if auction_years:
            print(f"         Auction years: {sorted(auction_years)}")
        if snake_years:
            print(f"         Snake years: {sorted(snake_years)}")

    # Calculate rank delta if needed (per year)
    if recalculate_delta:
        if per_year and year_column in df.columns:
            # Recalculate for each year with appropriate draft type
            for year in df[year_column].dropna().unique():
                year_mask = df[year_column] == year
                year_draft_type = detect_draft_type(df, year=year, year_column=year_column)
                delta_column = get_rank_delta_column(df, draft_type=year_draft_type)

                print(f"  [CALC] Calculating {delta_column} for {year}...")
                year_df = calculate_rank_delta(
                    df[year_mask].copy(),
                    group_columns=group_columns,
                    performance_column=performance_column,
                    draft_type=year_draft_type
                )
                df.loc[year_mask, delta_column] = year_df[delta_column]
        else:
            if draft_type is None:
                draft_type = detect_draft_type(df)
            delta_column = get_rank_delta_column(df, draft_type)
            print(f"  [CALC] Calculating {delta_column}...")
            df = calculate_rank_delta(
                df,
                group_columns=group_columns,
                performance_column=performance_column,
                draft_type=draft_type
            )

    # Assign value tiers (handles per-year internally)
    df = assign_value_tier(
        df,
        delta_column=None,  # Let it auto-detect per year
        value_bins=value_bins,
        value_labels=value_labels,
        draft_type=draft_type,
        year_column=year_column,
        per_year=per_year,
        finish_rank_column=finish_rank_column,
        bust_protection_threshold=bust_protection_threshold
    )

    return df


def get_value_tier_distribution(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get distribution of value tiers, optionally by group.

    Args:
        df: DataFrame with value_tier column
        group_columns: Optional columns to group by (e.g., ['year', 'position'])

    Returns:
        DataFrame with tier counts and percentages
    """
    if 'value_tier' not in df.columns:
        raise ValueError("value_tier column not found. Run calculate_value_tiers first.")

    if group_columns:
        # Group by specified columns + tier
        tier_counts = df.groupby(group_columns + ['value_tier'], dropna=False).size().reset_index(name='count')

        # Calculate percentage within each group
        totals = df.groupby(group_columns, dropna=False).size().reset_index(name='total')
        tier_counts = tier_counts.merge(totals, on=group_columns)
        tier_counts['percentage'] = (tier_counts['count'] / tier_counts['total'] * 100).round(1)
    else:
        # Overall distribution
        tier_counts = df['value_tier'].value_counts().reset_index()
        tier_counts.columns = ['value_tier', 'count']
        tier_counts['percentage'] = (tier_counts['count'] / len(df) * 100).round(1)

    return tier_counts


def get_value_tier_summary_stats(
    df: pd.DataFrame,
    delta_column: Optional[str] = None,
    draft_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Get summary statistics for each value tier.

    Args:
        df: DataFrame with value_tier and rank delta columns
        delta_column: Column containing rank delta (auto-detected if None)
        draft_type: 'auction' or 'snake' (auto-detected if None)

    Returns:
        DataFrame with mean, median, min, max delta for each tier
    """
    if 'value_tier' not in df.columns:
        raise ValueError("value_tier column not found. Run calculate_value_tiers first.")

    if delta_column is None:
        delta_column = get_rank_delta_column(df, draft_type)

    if delta_column not in df.columns:
        raise ValueError(f"Delta column '{delta_column}' not found in DataFrame")

    summary = df.groupby('value_tier', dropna=False)[delta_column].agg([
        ('count', 'count'),
        ('mean_delta', 'mean'),
        ('median_delta', 'median'),
        ('min_delta', 'min'),
        ('max_delta', 'max'),
        ('std_delta', 'std')
    ]).reset_index()

    return summary


def get_unified_rank_delta(
    df: pd.DataFrame,
    draft_type: Optional[str] = None
) -> pd.Series:
    """
    Get unified rank delta column that works for both draft types.

    Creates a 'rank_vs_finish_delta' column that uses:
    - price_rank_vs_finish_rank for auction
    - pick_rank_vs_finish_rank for snake

    Args:
        df: Draft DataFrame
        draft_type: 'auction' or 'snake' (auto-detected if None)

    Returns:
        Series with unified rank delta values
    """
    delta_column = get_rank_delta_column(df, draft_type)

    if delta_column not in df.columns:
        raise ValueError(f"Delta column '{delta_column}' not found")

    return df[delta_column]
