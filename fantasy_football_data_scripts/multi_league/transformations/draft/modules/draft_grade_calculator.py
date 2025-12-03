"""
Draft Grade Calculator Module

Calculates draft grades (A-F) based on SPAR percentile within position/year,
limited to only drafted players for accurate peer comparison.

Key Concepts:
- spar_percentile: SPAR percentile rank among drafted players at same position/year
- draft_grade: Letter grade (A-F) based on percentile thresholds

Configuration:
- Percentile bins and grade labels are configurable
- Default: A (top 20%), B (20-40%), C (40-60%), D (60-80%), F (bottom 20%)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict


# Default grade configuration - uses test score style grading with +/-
# 97-100: A+, 93-97: A, 90-93: A-, 87-90: B+, 83-87: B, 80-83: B-
# 77-80: C+, 73-77: C, 70-73: C-, 67-70: D+, 63-67: D, 60-63: D-, 0-60: F
DEFAULT_PERCENTILE_BINS = [0, 60, 63, 67, 70, 73, 77, 80, 83, 87, 90, 93, 97, 100]
DEFAULT_GRADE_LABELS = ['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']


def get_base_grade(grade: str) -> str:
    """Get the base letter (A, B, C, D, F) from a grade like A+ or B-."""
    if not grade or grade == 'N/A' or pd.isna(grade):
        return ''
    return str(grade)[0]


def calculate_spar_percentile(
    df: pd.DataFrame,
    spar_column: str = 'manager_spar',
    group_columns: Optional[List[str]] = None,
    drafted_only: bool = True,
    draft_indicator_column: str = 'pick'
) -> pd.DataFrame:
    """
    Calculate SPAR percentile within groups, optionally limited to drafted players.

    Args:
        df: DataFrame with SPAR metrics
        spar_column: Column name containing SPAR values (default: 'manager_spar')
        group_columns: Columns to group by for percentile calculation
                      (default: ['year', 'position'] or ['year', 'yahoo_position'])
        drafted_only: If True, only include drafted players in percentile calculation
        draft_indicator_column: Column that indicates a player was drafted
                               (non-null values mean drafted)

    Returns:
        DataFrame with spar_percentile column added
    """
    df = df.copy()

    # Determine position column dynamically
    pos_col = 'position' if 'position' in df.columns else 'yahoo_position'

    # Set default group columns if not provided
    if group_columns is None:
        group_columns = ['year', pos_col]

    # Validate required columns exist
    missing_cols = [c for c in group_columns + [spar_column] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to drafted players if requested
    if drafted_only:
        if draft_indicator_column not in df.columns:
            raise ValueError(
                f"Draft indicator column '{draft_indicator_column}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Drafted = has a non-null draft pick
        drafted_mask = df[draft_indicator_column].notna()
        df_drafted = df[drafted_mask].copy()

        if df_drafted.empty:
            df['spar_percentile'] = np.nan
            return df
    else:
        df_drafted = df.copy()

    # Calculate percentile rank within groups
    df_drafted['spar_percentile'] = df_drafted.groupby(group_columns, dropna=False)[spar_column].transform(
        lambda x: x.rank(pct=True, method='average') * 100
    )

    # Merge percentile back to original DataFrame if we filtered
    if drafted_only:
        # Only update the spar_percentile for drafted players
        df['spar_percentile'] = np.nan
        df.loc[drafted_mask, 'spar_percentile'] = df_drafted['spar_percentile'].values
    else:
        df['spar_percentile'] = df_drafted['spar_percentile']

    return df


def assign_draft_grade(
    df: pd.DataFrame,
    percentile_column: str = 'spar_percentile',
    percentile_bins: Optional[List[float]] = None,
    grade_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Assign letter grades based on percentile thresholds.

    Args:
        df: DataFrame with percentile column
        percentile_column: Column containing percentile values
        percentile_bins: List of bin edges (default: [0, 20, 40, 60, 80, 100])
        grade_labels: List of grade labels for each bin (default: ['F', 'D', 'C', 'B', 'A'])

    Returns:
        DataFrame with draft_grade column added
    """
    df = df.copy()

    # Use defaults if not provided
    if percentile_bins is None:
        percentile_bins = DEFAULT_PERCENTILE_BINS
    if grade_labels is None:
        grade_labels = DEFAULT_GRADE_LABELS

    # Validate configuration
    if len(grade_labels) != len(percentile_bins) - 1:
        raise ValueError(
            f"Number of grade labels ({len(grade_labels)}) must be one less than "
            f"number of bin edges ({len(percentile_bins)})"
        )

    if percentile_column not in df.columns:
        raise ValueError(f"Percentile column '{percentile_column}' not found in DataFrame")

    # Assign grades using pd.cut
    df['draft_grade'] = pd.cut(
        df[percentile_column],
        bins=percentile_bins,
        labels=grade_labels,
        include_lowest=True
    )

    return df


def calculate_draft_grades(
    df: pd.DataFrame,
    spar_column: str = 'manager_spar',
    group_columns: Optional[List[str]] = None,
    drafted_only: bool = True,
    draft_indicator_column: str = 'pick',
    percentile_bins: Optional[List[float]] = None,
    grade_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate SPAR percentile and assign draft grades in one step.

    This is the main entry point for draft grade calculation.

    Args:
        df: DataFrame with SPAR metrics
        spar_column: Column name containing SPAR values (default: 'manager_spar')
        group_columns: Columns to group by for percentile calculation
                      (default: ['year', 'position'])
        drafted_only: If True, only include drafted players in percentile calculation
        draft_indicator_column: Column that indicates a player was drafted
        percentile_bins: List of bin edges for grade assignment
        grade_labels: List of grade labels for each bin

    Returns:
        DataFrame with spar_percentile and draft_grade columns added

    Example:
        # Basic usage with defaults (A-F grades, drafted players only)
        df = calculate_draft_grades(df)

        # Custom grade scale
        df = calculate_draft_grades(
            df,
            percentile_bins=[0, 10, 30, 70, 90, 100],
            grade_labels=['F', 'D', 'C', 'B', 'A']  # Stricter grading
        )

        # Include all players (not just drafted)
        df = calculate_draft_grades(df, drafted_only=False)
    """
    # Step 1: Calculate SPAR percentile
    df = calculate_spar_percentile(
        df,
        spar_column=spar_column,
        group_columns=group_columns,
        drafted_only=drafted_only,
        draft_indicator_column=draft_indicator_column
    )

    # Step 2: Assign draft grades
    df = assign_draft_grade(
        df,
        percentile_column='spar_percentile',
        percentile_bins=percentile_bins,
        grade_labels=grade_labels
    )

    return df


def get_grade_distribution(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get distribution of draft grades, optionally by group.

    Args:
        df: DataFrame with draft_grade column
        group_columns: Optional columns to group by (e.g., ['year', 'position'])

    Returns:
        DataFrame with grade counts and percentages
    """
    if 'draft_grade' not in df.columns:
        raise ValueError("draft_grade column not found. Run calculate_draft_grades first.")

    if group_columns:
        # Group by specified columns + grade
        grade_counts = df.groupby(group_columns + ['draft_grade'], dropna=False).size().reset_index(name='count')

        # Calculate percentage within each group
        totals = df.groupby(group_columns, dropna=False).size().reset_index(name='total')
        grade_counts = grade_counts.merge(totals, on=group_columns)
        grade_counts['percentage'] = (grade_counts['count'] / grade_counts['total'] * 100).round(1)
    else:
        # Overall distribution
        grade_counts = df['draft_grade'].value_counts().reset_index()
        grade_counts.columns = ['draft_grade', 'count']
        grade_counts['percentage'] = (grade_counts['count'] / len(df) * 100).round(1)
        grade_counts = grade_counts.sort_values('draft_grade')

    return grade_counts


def get_grade_summary_stats(
    df: pd.DataFrame,
    spar_column: str = 'manager_spar'
) -> pd.DataFrame:
    """
    Get summary statistics for each draft grade.

    Args:
        df: DataFrame with draft_grade and SPAR columns
        spar_column: Column containing SPAR values

    Returns:
        DataFrame with mean, median, min, max SPAR for each grade
    """
    if 'draft_grade' not in df.columns:
        raise ValueError("draft_grade column not found. Run calculate_draft_grades first.")

    if spar_column not in df.columns:
        raise ValueError(f"SPAR column '{spar_column}' not found in DataFrame")

    summary = df.groupby('draft_grade', dropna=False)[spar_column].agg([
        ('count', 'count'),
        ('mean_spar', 'mean'),
        ('median_spar', 'median'),
        ('min_spar', 'min'),
        ('max_spar', 'max'),
        ('std_spar', 'std')
    ]).reset_index()

    return summary
