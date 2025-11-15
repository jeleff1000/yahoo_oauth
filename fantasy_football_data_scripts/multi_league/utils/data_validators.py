"""
Data validation tests for fantasy football data pipeline.

Provides automated checks to ensure data quality and catch common errors:
- Sum of starters equals roster counts
- manager_week unique keys
- No null league_key after discovery
- No backwards cumulative_week moves
- Referential integrity checks
- Data consistency checks

Usage:
    from data_validators import validate_matchup_data, validate_player_data

    # Validate matchup data
    errors = validate_matchup_data(matchup_df)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")

    # Validate all data
    from data_validators import validate_all_data
    results = validate_all_data(
        matchup_df=matchup_df,
        player_df=player_df,
        transactions_df=transactions_df
    )
"""
from __future__ import annotations

import pandas as pd
import polars as pl
from typing import Any, Dict, List, Optional, Union


class ValidationError:
    """Represents a data validation error."""

    def __init__(
        self,
        check: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error.

        Args:
            check: Name of the check that failed
            severity: 'error', 'warning', or 'info'
            message: Human-readable error message
            details: Additional details about the error
        """
        self.check = check
        self.severity = severity
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.check}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check': self.check,
            'severity': self.severity,
            'message': self.message,
            'details': self.details
        }


def _to_pandas(df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
    """Convert Polars to Pandas if needed."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


# =============================================================================
# Matchup Data Validation
# =============================================================================

def validate_matchup_data(df: Union[pd.DataFrame, pl.DataFrame]) -> List[ValidationError]:
    """
    Validate matchup data.

    Checks:
    - Required columns exist
    - No null values in key columns
    - manager_week is unique
    - cumulative_week is monotonic (no backwards moves)
    - Points are non-negative
    - Roster counts are reasonable

    Args:
        df: Matchup DataFrame

    Returns:
        List of ValidationError objects
    """
    errors = []
    df = _to_pandas(df)

    # Check required columns
    required_cols = ['manager', 'year', 'week', 'cumulative_week', 'points']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(ValidationError(
            'required_columns',
            'error',
            f"Missing required columns: {missing_cols}",
            {'missing_columns': missing_cols}
        ))
        return errors  # Can't proceed without required columns

    # Check for null values in key columns
    key_cols = ['manager', 'year', 'week', 'cumulative_week']
    for col in key_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            errors.append(ValidationError(
                'null_key_column',
                'error',
                f"Found {null_count} null values in '{col}'",
                {'column': col, 'null_count': int(null_count)}
            ))

    # Check manager_week uniqueness
    if 'manager' in df.columns and 'week' in df.columns and 'cumulative_week' in df.columns:
        dup_manager_week = df.duplicated(subset=['manager', 'cumulative_week']).sum()
        if dup_manager_week > 0:
            errors.append(ValidationError(
                'manager_week_unique',
                'error',
                f"Found {dup_manager_week} duplicate manager-cumulative_week combinations",
                {'duplicate_count': int(dup_manager_week)}
            ))

    # Check cumulative_week is monotonic per manager-year
    if 'manager' in df.columns and 'year' in df.columns and 'cumulative_week' in df.columns:
        for (manager, year), group in df.groupby(['manager', 'year']):
            sorted_weeks = group.sort_values('week')['cumulative_week'].values
            if not all(sorted_weeks[i] <= sorted_weeks[i+1] for i in range(len(sorted_weeks)-1)):
                errors.append(ValidationError(
                    'cumulative_week_monotonic',
                    'error',
                    f"cumulative_week is not monotonic for {manager} in {year}",
                    {'manager': manager, 'year': year}
                ))

    # Check points are non-negative
    if 'points' in df.columns:
        negative_points = (df['points'] < 0).sum()
        if negative_points > 0:
            errors.append(ValidationError(
                'non_negative_points',
                'warning',
                f"Found {negative_points} rows with negative points",
                {'negative_count': int(negative_points)}
            ))

    # Check roster counts if present
    if 'roster_count' in df.columns:
        invalid_rosters = ((df['roster_count'] < 5) | (df['roster_count'] > 30)).sum()
        if invalid_rosters > 0:
            errors.append(ValidationError(
                'roster_count_reasonable',
                'warning',
                f"Found {invalid_rosters} rows with unusual roster counts (< 5 or > 30)",
                {'invalid_count': int(invalid_rosters)}
            ))

    return errors


def validate_roster_consistency(df: Union[pd.DataFrame, pl.DataFrame]) -> List[ValidationError]:
    """
    Validate that sum of starters equals total roster counts.

    Checks that for each matchup:
    - starters + bench = roster_count
    - All counts are non-negative

    Args:
        df: Matchup DataFrame with roster columns

    Returns:
        List of ValidationError objects
    """
    errors = []
    df = _to_pandas(df)

    # Check if roster columns exist
    roster_cols = ['starters', 'bench', 'roster_count']
    missing = [col for col in roster_cols if col not in df.columns]
    if missing:
        errors.append(ValidationError(
            'roster_columns_exist',
            'info',
            f"Roster columns not found: {missing}. Skipping roster validation.",
            {'missing_columns': missing}
        ))
        return errors

    # Check sum consistency
    df_copy = df[roster_cols].copy()
    df_copy['computed_total'] = df_copy['starters'] + df_copy['bench']
    mismatch = (df_copy['computed_total'] != df_copy['roster_count']).sum()

    if mismatch > 0:
        errors.append(ValidationError(
            'roster_sum_matches',
            'error',
            f"Found {mismatch} rows where starters + bench != roster_count",
            {
                'mismatch_count': int(mismatch),
                'example_mismatches': df_copy[df_copy['computed_total'] != df_copy['roster_count']].head(5).to_dict('records')
            }
        ))

    # Check non-negative
    for col in roster_cols:
        negative = (df[col] < 0).sum()
        if negative > 0:
            errors.append(ValidationError(
                f'{col}_non_negative',
                'error',
                f"Found {negative} rows with negative {col}",
                {'column': col, 'negative_count': int(negative)}
            ))

    return errors


# =============================================================================
# Player Data Validation
# =============================================================================

def validate_player_data(df: Union[pd.DataFrame, pl.DataFrame]) -> List[ValidationError]:
    """
    Validate player stats data.

    Checks:
    - Required columns exist
    - No null player_id or player_name
    - player_id uniqueness per week
    - Stats are reasonable ranges

    Args:
        df: Player stats DataFrame

    Returns:
        List of ValidationError objects
    """
    errors = []
    df = _to_pandas(df)

    # Check required columns
    required_cols = ['player_id', 'player_name', 'year', 'week']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(ValidationError(
            'required_columns',
            'error',
            f"Missing required columns: {missing_cols}",
            {'missing_columns': missing_cols}
        ))
        return errors

    # Check for null player identifiers
    null_player_id = df['player_id'].isna().sum()
    if null_player_id > 0:
        errors.append(ValidationError(
            'null_player_id',
            'error',
            f"Found {null_player_id} rows with null player_id",
            {'null_count': int(null_player_id)}
        ))

    null_player_name = df['player_name'].isna().sum()
    if null_player_name > 0:
        errors.append(ValidationError(
            'null_player_name',
            'warning',
            f"Found {null_player_name} rows with null player_name",
            {'null_count': int(null_player_name)}
        ))

    # Check player uniqueness per week
    if all(col in df.columns for col in ['player_id', 'year', 'week']):
        dup_player_week = df.duplicated(subset=['player_id', 'year', 'week']).sum()
        if dup_player_week > 0:
            errors.append(ValidationError(
                'player_week_unique',
                'error',
                f"Found {dup_player_week} duplicate player-year-week combinations",
                {'duplicate_count': int(dup_player_week)}
            ))

    # Check reasonable stat ranges
    stat_checks = {
        'points': (0, 100),  # Fantasy points usually 0-100
        'passing_yards': (0, 600),
        'rushing_yards': (0, 300),
        'receiving_yards': (0, 300),
        'touchdowns': (0, 10)
    }

    for col, (min_val, max_val) in stat_checks.items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                errors.append(ValidationError(
                    f'{col}_reasonable_range',
                    'warning',
                    f"Found {out_of_range} rows with {col} outside typical range [{min_val}, {max_val}]",
                    {'column': col, 'out_of_range_count': int(out_of_range)}
                ))

    return errors


# =============================================================================
# Transaction Data Validation
# =============================================================================

def validate_transaction_data(df: Union[pd.DataFrame, pl.DataFrame]) -> List[ValidationError]:
    """
    Validate transaction data.

    Checks:
    - Required columns exist
    - No null transaction_id
    - Transaction types are valid
    - Dates are reasonable

    Args:
        df: Transaction DataFrame

    Returns:
        List of ValidationError objects
    """
    errors = []
    df = _to_pandas(df)

    # Check required columns
    required_cols = ['transaction_id', 'type', 'timestamp', 'player_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(ValidationError(
            'required_columns',
            'error',
            f"Missing required columns: {missing_cols}",
            {'missing_columns': missing_cols}
        ))
        return errors

    # Check for null transaction_id
    null_txn_id = df['transaction_id'].isna().sum()
    if null_txn_id > 0:
        errors.append(ValidationError(
            'null_transaction_id',
            'error',
            f"Found {null_txn_id} rows with null transaction_id",
            {'null_count': int(null_txn_id)}
        ))

    # Check valid transaction types
    if 'type' in df.columns:
        valid_types = {'add', 'drop', 'trade', 'add/drop'}
        invalid_types = set(df['type'].unique()) - valid_types
        if invalid_types:
            errors.append(ValidationError(
                'valid_transaction_type',
                'warning',
                f"Found invalid transaction types: {invalid_types}",
                {'invalid_types': list(invalid_types)}
            ))

    # Check timestamps are reasonable (within last 20 years)
    if 'timestamp' in df.columns:
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        min_date = pd.Timestamp('2000-01-01')
        max_date = pd.Timestamp.now() + pd.Timedelta(days=365)

        invalid_dates = ((df_copy['timestamp'] < min_date) | (df_copy['timestamp'] > max_date)).sum()
        if invalid_dates > 0:
            errors.append(ValidationError(
                'reasonable_timestamp',
                'warning',
                f"Found {invalid_dates} rows with unreasonable timestamps",
                {'invalid_count': int(invalid_dates)}
            ))

    return errors


# =============================================================================
# League Discovery Validation
# =============================================================================

def validate_league_discovery(df: Union[pd.DataFrame, pl.DataFrame]) -> List[ValidationError]:
    """
    Validate that league discovery was successful.

    Checks:
    - No null league_key after discovery
    - League_key format is valid
    - Consistent league_key per year

    Args:
        df: DataFrame with league_key column

    Returns:
        List of ValidationError objects
    """
    errors = []
    df = _to_pandas(df)

    if 'league_key' not in df.columns:
        errors.append(ValidationError(
            'league_key_column_exists',
            'info',
            "league_key column not found. Skipping league validation.",
            {}
        ))
        return errors

    # Check for null league_key
    null_league_key = df['league_key'].isna().sum()
    if null_league_key > 0:
        errors.append(ValidationError(
            'null_league_key',
            'error',
            f"Found {null_league_key} rows with null league_key (discovery may have failed)",
            {'null_count': int(null_league_key)}
        ))

    # Check league_key format (should be like "nfl.l.12345")
    if df['league_key'].notna().any():
        invalid_format = df[df['league_key'].notna()]['league_key'].str.match(r'^[a-z]+\.l\.\d+$', na=False)
        invalid_count = (~invalid_format).sum()
        if invalid_count > 0:
            errors.append(ValidationError(
                'league_key_format',
                'warning',
                f"Found {invalid_count} rows with invalid league_key format",
                {'invalid_count': int(invalid_count)}
            ))

    # Check consistency per year
    if 'year' in df.columns:
        inconsistent_years = []
        for year, group in df.groupby('year'):
            unique_keys = group['league_key'].dropna().unique()
            if len(unique_keys) > 1:
                inconsistent_years.append((year, list(unique_keys)))

        if inconsistent_years:
            errors.append(ValidationError(
                'league_key_consistent_per_year',
                'warning',
                f"Found {len(inconsistent_years)} years with multiple league_keys",
                {'inconsistent_years': inconsistent_years}
            ))

    return errors


# =============================================================================
# Comprehensive Validation
# =============================================================================

def validate_all_data(
    matchup_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    player_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    transactions_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    schedule_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
) -> Dict[str, List[ValidationError]]:
    """
    Run all validation checks across all datasets.

    Args:
        matchup_df: Matchup DataFrame
        player_df: Player stats DataFrame
        transactions_df: Transaction DataFrame
        schedule_df: Schedule DataFrame

    Returns:
        Dictionary mapping dataset name to list of validation errors

    Example:
        >>> results = validate_all_data(
        ...     matchup_df=matchup_df,
        ...     player_df=player_df
        ... )
        >>>
        >>> for dataset, errors in results.items():
        ...     if errors:
        ...         print(f"{dataset}:")
        ...         for error in errors:
        ...             print(f"  {error}")
    """
    results = {}

    if matchup_df is not None:
        errors = []
        errors.extend(validate_matchup_data(matchup_df))
        errors.extend(validate_roster_consistency(matchup_df))
        errors.extend(validate_league_discovery(matchup_df))
        results['matchup'] = errors

    if player_df is not None:
        errors = validate_player_data(player_df)
        results['player'] = errors

    if transactions_df is not None:
        errors = validate_transaction_data(transactions_df)
        results['transactions'] = errors

    if schedule_df is not None:
        errors = validate_league_discovery(schedule_df)
        results['schedule'] = errors

    return results


def print_validation_summary(results: Dict[str, List[ValidationError]]) -> None:
    """
    Print a summary of validation results.

    Args:
        results: Results from validate_all_data()
    """
    total_errors = sum(len(errors) for errors in results.values())
    total_error_level = sum(
        sum(1 for e in errors if e.severity == 'error')
        for errors in results.values()
    )
    total_warnings = sum(
        sum(1 for e in errors if e.severity == 'warning')
        for errors in results.values()
    )

    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total checks: {total_errors}")
    print(f"  Errors: {total_error_level}")
    print(f"  Warnings: {total_warnings}")
    print()

    for dataset, errors in results.items():
        if errors:
            print(f"{dataset.upper()}:")
            for error in errors:
                print(f"  {error}")
            print()

    if total_error_level == 0:
        print("[OK] All critical validations passed!")
    else:
        print(f"[FAIL] {total_error_level} critical validation errors found!")


def save_validation_report(
    results: Dict[str, List[ValidationError]],
    output_path: Union[str, Path]
) -> None:
    """
    Save validation results to JSON file.

    Args:
        results: Results from validate_all_data()
        output_path: Path to output JSON file
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)

    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {
            dataset: [error.to_dict() for error in errors]
            for dataset, errors in results.items()
        },
        'summary': {
            'total_checks': sum(len(errors) for errors in results.values()),
            'total_errors': sum(
                sum(1 for e in errors if e.severity == 'error')
                for errors in results.values()
            ),
            'total_warnings': sum(
                sum(1 for e in errors if e.severity == 'warning')
                for errors in results.values()
            )
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Validation report saved to {output_path}")
