"""
Core data normalization utilities for transformation modules.
Ensures consistent data types across all transformations.
"""

import pandas as pd
from pathlib import Path
from typing import Any, List, Optional


def find_league_settings_directory(league_id: Optional[str] = None, df: Optional[pd.DataFrame] = None, data_directory: Optional[Path] = None) -> Optional[Path]:
    """
    Find the yahoo_league_settings directory for a given league.

    This function searches for league settings in the standard LeagueContext directory structure:
    - {data_directory}/player_data/yahoo_league_settings/

    Args:
        league_id: League ID (e.g., "nfl.l.123456"). If None, will try to extract from df.
        df: DataFrame with league_id column (optional, used if league_id not provided)
        data_directory: Direct path to league data directory (optional, highest priority)

    Returns:
        Path to yahoo_league_settings directory if found, None otherwise
    """
    # PRIORITY 1: Use data_directory if provided (most reliable)
    if data_directory is not None:
        # NEW location (as of 2025): league_settings/ (league-wide config, not player-specific)
        settings_path_new = Path(data_directory) / "league_settings"
        if settings_path_new.exists() and settings_path_new.is_dir():
            return settings_path_new

        # OLD location (backwards compatibility): player_data/yahoo_league_settings/
        settings_path_old = Path(data_directory) / "player_data" / "yahoo_league_settings"
        if settings_path_old.exists() and settings_path_old.is_dir():
            return settings_path_old

    # Try to get league_id from df if not provided
    if league_id is None and df is not None:
        if 'league_id' in df.columns:
            league_id = df['league_id'].iloc[0] if len(df) > 0 else None

    if league_id is None:
        return None

    # Sanitize league_id for filesystem (replace dots with underscores)
    sanitized_id = str(league_id).replace(".", "_")

    # Start from this file's location and navigate to common data storage locations
    current = Path(__file__).resolve().parent  # multi_league/core/
    multi_league_dir = current.parent  # multi_league/
    scripts_dir = multi_league_dir.parent  # fantasy_football_data_scripts/

    # Search in standard locations based on LeagueContext structure
    possible_base_paths = [
        Path.home() / "fantasy_football_data",  # Default LeagueContext location
        scripts_dir / "data",
        scripts_dir.parent / "fantasy_football_data",
        multi_league_dir / "data",
    ]

    # PRIORITY 2: Look for {base}/{sanitized_league_id}/league_settings/ (NEW structure)
    for base_path in possible_base_paths:
        # Try NEW location first
        settings_path_new = base_path / sanitized_id / "league_settings"
        if settings_path_new.exists() and settings_path_new.is_dir():
            return settings_path_new

        # Try OLD location for backwards compatibility
        settings_path_old = base_path / sanitized_id / "player_data" / "yahoo_league_settings"
        if settings_path_old.exists() and settings_path_old.is_dir():
            return settings_path_old

    # PRIORITY 3: Broader search - look for ANY subdirectory containing the settings
    # This handles cases where the directory is named by league name instead of ID
    for base_path in possible_base_paths:
        if not base_path.exists():
            continue
        # Search for any subdirectory with league_settings or player_data/yahoo_league_settings
        try:
            for subdir in base_path.iterdir():
                if subdir.is_dir():
                    # Try NEW location first
                    settings_path_new = subdir / "league_settings"
                    if settings_path_new.exists() and settings_path_new.is_dir():
                        # Verify this is the right league by checking for settings files with matching league_id
                        settings_files = list(settings_path_new.glob(f"league_settings_*_{sanitized_id}.json"))
                        if settings_files:
                            return settings_path_new

                    # Try OLD location for backwards compatibility
                    settings_path_old = subdir / "player_data" / "yahoo_league_settings"
                    if settings_path_old.exists() and settings_path_old.is_dir():
                        # Verify this is the right league by checking for settings files with matching league_id
                        settings_files = list(settings_path_old.glob(f"league_settings_*_{sanitized_id}.json"))
                        if settings_files:
                            return settings_path_old
        except (PermissionError, OSError):
            # Skip directories we can't read
            pass

    return None


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric columns to ensure consistent data types.
    Converts string numbers to proper numeric types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized numeric columns
    """
    df = df.copy()

    for col in df.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Try to convert to numeric if it looks like numbers
        try:
            # Check if column is object or string type (handles both 'object' and pandas 'string' dtypes)
            if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only convert if we didn't lose too much data (>50% valid conversions)
                if converted.notna().sum() > len(df) * 0.5 and converted.notna().sum() > 0:
                    df[col] = converted
        except (ValueError, TypeError, AttributeError):
            # Keep as-is if conversion fails
            pass

    return df


def ensure_league_id(df: pd.DataFrame, league_id: Any) -> pd.DataFrame:
    """
    Ensure league_id column exists and is populated.

    Args:
        df: Input DataFrame
        league_id: League ID to ensure

    Returns:
        DataFrame with league_id column
    """
    df = df.copy()

    if 'league_id' not in df.columns:
        df['league_id'] = league_id
    else:
        # Fill missing league_id values
        df['league_id'] = df['league_id'].fillna(league_id)

    return df


def add_composite_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite key columns for common lookups.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with composite key columns
    """
    df = df.copy()

    # Add common composite keys
    if 'year' in df.columns and 'week' in df.columns:
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str)

    if 'player' in df.columns and 'year' in df.columns:
        df['player_year'] = df['player'].astype(str) + '_' + df['year'].astype(str)

    # --- canonical manager_week/opponent_week using cumulative_week ---
    # Ensure cumulative_week if year/week exist
    if 'cumulative_week' not in df.columns and {'year','week'}.issubset(df.columns):
        df['cumulative_week'] = (
            pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int) * 100
            + pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
        )

    # Canonical manager_week/opponent_week using cumulative_week
    if {'manager','cumulative_week'}.issubset(df.columns):
        df['manager_week'] = (
            df['manager'].astype(str).str.replace(' ', '', regex=False) +
            df['cumulative_week'].astype(str)
        )
    if {'opponent','cumulative_week'}.issubset(df.columns):
        df['opponent_week'] = (
            df['opponent'].astype(str).str.replace(' ', '', regex=False) +
            df['cumulative_week'].astype(str)
        )

    # Ensure BOTH manager_week and manager_year_week exist (bidirectional alias)
    # This fixes cloud/local sync issues where one might exist without the other
    if 'manager_week' in df.columns and 'manager_year_week' not in df.columns:
        df['manager_year_week'] = df['manager_week']
    elif 'manager_year_week' in df.columns and 'manager_week' not in df.columns:
        df['manager_week'] = df['manager_year_week']

    return df


def validate_league_isolation(
    df: pd.DataFrame,
    expected_league_id: Any,
    file_name: str = "unknown",
    log: Any = None,
    valid_league_ids: Any = None
) -> bool:
    """
    Validate that all rows in DataFrame belong to expected league(s).

    Args:
        df: Input DataFrame
        expected_league_id: Primary expected league ID (for single-year leagues)
        file_name: Name of file being validated (for logging)
        log: Optional logging function (defaults to print)
        valid_league_ids: Collection of valid league IDs (for multi-year leagues)
                         If provided, allows multiple league_ids as long as all are valid

    Returns:
        True if all rows match expected league(s), False otherwise
    """
    # Use provided log function or fall back to print
    _log = log if log is not None else print

    if 'league_id' not in df.columns:
        _log(f"  [{file_name}] WARNING: league_id column not found, cannot validate isolation")
        return False

    unique_leagues = df['league_id'].dropna().unique()

    if len(unique_leagues) == 0:
        _log(f"  [{file_name}] WARNING: No league_id values found")
        return False

    # Build set of valid league IDs
    valid_set = set()
    if valid_league_ids:
        valid_set = {str(lid) for lid in valid_league_ids}
    if expected_league_id:
        valid_set.add(str(expected_league_id))

    # Check if all league_ids in data are valid
    actual_set = {str(lid) for lid in unique_leagues}
    invalid_leagues = actual_set - valid_set

    if invalid_leagues:
        _log(f"  [{file_name}] ERROR: Invalid league IDs found: {invalid_leagues}")
        _log(f"  [{file_name}]        Valid league IDs: {valid_set}")
        return False

    _log(f"  [{file_name}] OK: League isolation validated ({len(actual_set)} league IDs across years)")
    return True


def write_parquet_robust(
    df: pd.DataFrame,
    file_path: Path,
    description: str = "data"
) -> None:
    """
    Robustly write DataFrame to parquet with error handling.

    Args:
        df: DataFrame to write
        file_path: Path to write to
        description: Description for logging
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet
        df.to_parquet(file_path, index=False, engine='pyarrow')

        print(f"  [OK] Wrote {len(df):,} rows of {description} to {file_path.name}")

    except Exception as e:
        print(f"  ERROR: Failed to write {description} to {file_path}: {e}")
        raise
