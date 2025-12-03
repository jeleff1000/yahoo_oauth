#!/usr/bin/env python3
"""
Inflation Rate Module

Calculates year-over-year scoring inflation for cross-season comparisons.

Extracted from cumulative_stats.py for modularity.
"""
from __future__ import annotations

import pandas as pd
from typing import Callable, Optional


def calculate_inflation_rate(
    df: pd.DataFrame,
    log_fn: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Calculate year-over-year scoring inflation rate.

    Uses the earliest year with valid data as the base year (inflation_rate = 1.0).
    Each subsequent year's inflation rate is calculated as:
        inflation_rate = year_avg_points / base_year_avg_points

    Args:
        df: DataFrame with 'team_points' and 'year' columns
        log_fn: Optional logging function (defaults to print)

    Returns:
        DataFrame with 'inflation_rate' column added
    """
    if log_fn is None:
        log_fn = print

    df = df.copy()

    if "team_points" not in df.columns or "year" not in df.columns:
        df["inflation_rate"] = 1.0
        log_fn("[inflation] Missing required columns (team_points/year), defaulting inflation_rate to 1.0")
        return df

    # Calculate average team_points per year (exclude nulls/zeros)
    valid_data = df[df["team_points"].notna() & (df["team_points"] > 0)]
    year_means = valid_data.groupby("year")["team_points"].mean()

    if year_means.empty or len(year_means) == 0:
        df["inflation_rate"] = 1.0
        log_fn("[inflation] No year data available, defaulting inflation_rate to 1.0")
        return df

    # Use earliest year WITH DATA as base (inflation_rate = 1.0)
    valid_years = year_means[year_means > 0]
    if len(valid_years) == 0:
        df["inflation_rate"] = 1.0
        log_fn("[inflation] No valid year averages found, defaulting all inflation_rate to 1.0")
        return df

    base_year = int(valid_years.index.min())
    base_mean = float(valid_years.loc[base_year])

    # Calculate inflation_rate for each year relative to base year
    infl_map = {int(y): float(m) / base_mean for y, m in valid_years.items()}
    df["inflation_rate"] = df["year"].map(infl_map).fillna(1.0).astype(float)

    log_fn(f"[inflation] Base year: {base_year}, base avg: {base_mean:.2f} pts/game")
    log_fn(f"[inflation] Calculated inflation rates for {len(infl_map)} years")

    # Log inflation rates for each year for debugging
    for y in sorted(infl_map.keys()):
        log_fn(f"[inflation]   {y}: {infl_map[y]:.3f} ({year_means.loc[y]:.2f} pts/game)")

    return df
