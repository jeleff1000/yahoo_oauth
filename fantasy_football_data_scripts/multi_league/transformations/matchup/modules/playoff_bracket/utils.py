"""
Playoff Bracket Utilities

Shared utilities for playoff bracket simulation.
Contains settings loading, bracket validation, and helper functions.
"""

import sys
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import pandas as pd

# Add parent directories to path - RELATIVE NAVIGATION
_script_file = Path(__file__).resolve()
_playoff_bracket_dir = _script_file.parent
_modules_dir = _playoff_bracket_dir.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))
sys.path.insert(0, str(_multi_league_dir / "core"))

from core.data_normalization import find_league_settings_directory


def validate_bracket_structure(num_playoff_teams: int, bye_teams: int, num_teams: int) -> Tuple[bool, str]:
    """
    Validate that playoff settings produce a valid bracket structure.

    Args:
        num_playoff_teams: Number of teams making playoffs
        bye_teams: Number of first-round byes
        num_teams: Total teams in league

    Returns:
        (is_valid, error_message) - error_message is empty string if valid
    """
    # Basic sanity checks
    if num_playoff_teams <= 0:
        return False, f"num_playoff_teams ({num_playoff_teams}) must be positive"

    if num_playoff_teams > num_teams:
        return False, f"num_playoff_teams ({num_playoff_teams}) exceeds total teams ({num_teams})"

    if bye_teams < 0:
        return False, f"bye_teams ({bye_teams}) cannot be negative"

    if bye_teams >= num_playoff_teams:
        return False, f"bye_teams ({bye_teams}) must be less than num_playoff_teams ({num_playoff_teams})"

    # Check if first round is valid
    teams_playing_round1 = num_playoff_teams - bye_teams

    if teams_playing_round1 < 0:
        return False, f"Invalid: {teams_playing_round1} teams would play in round 1 (negative)"

    # Teams playing round 1 must be even (pairs of matchups)
    if teams_playing_round1 % 2 != 0:
        return False, f"Invalid: {teams_playing_round1} teams in round 1 (must be even for matchups)"

    # Check if round 2 makes sense
    winners_round1 = teams_playing_round1 // 2
    teams_round2 = winners_round1 + bye_teams

    if teams_round2 < 2:
        return False, f"Invalid: Only {teams_round2} teams would reach round 2 (need at least 2 for championship)"

    # Ideally round 2 should also be even (unless it's exactly the finals)
    if teams_round2 > 2 and teams_round2 % 2 != 0:
        # This is a warning, not an error - some leagues have 3-way finals or other structures
        print(f"  [WARN] Round 2 has {teams_round2} teams (odd number, bracket may be unusual)")

    return True, ""


def _get_default_settings() -> Dict:
    """Return default league settings."""
    return {
        'playoff_start_week': 15,
        'num_playoff_teams': 6,
        'bye_teams': 2,
        'has_multiweek_championship': 0,
        'uses_playoff_reseeding': 0,
        'num_teams': 10
    }


def _load_from_parquet(data_directory: Path, year: int) -> Optional[Dict]:
    """
    Try to load settings from league_settings.parquet file.

    Args:
        data_directory: Directory containing the parquet file
        year: Year to load settings for

    Returns:
        Settings dict or None if not found
    """
    parquet_path = data_directory / 'league_settings.parquet'
    if not parquet_path.exists():
        return None

    try:
        settings_df = pd.read_parquet(parquet_path)
        year_settings = settings_df[settings_df['year'] == year]

        if year_settings.empty:
            return None

        row = year_settings.iloc[0]

        config = {
            'playoff_start_week': int(row.get('playoff_start_week', 15)),
            'num_playoff_teams': int(row.get('num_playoff_teams', row.get('playoff_teams', 6))),
            'bye_teams': int(row.get('bye_teams', 2) if pd.notna(row.get('bye_teams')) else 2),
            'has_multiweek_championship': int(row.get('has_multiweek_championship', 0) if pd.notna(row.get('has_multiweek_championship')) else 0),
            'uses_playoff_reseeding': int(row.get('uses_playoff_reseeding', 0) if pd.notna(row.get('uses_playoff_reseeding')) else 0),
            'num_teams': int(row.get('num_teams', 10))
        }

        return config
    except Exception as e:
        print(f"  [WARN] Failed to load from parquet: {e}")
        return None


def _infer_settings_from_data(df: pd.DataFrame, year: int) -> Optional[Dict]:
    """
    Infer playoff settings from data structure when no settings file exists.

    Detects playoffs by finding weeks with fewer games than regular season.

    Args:
        df: DataFrame with matchup data
        year: Year to infer settings for

    Returns:
        Settings dict or None if unable to infer
    """
    year_df = df[df['year'] == year]
    if year_df.empty:
        return None

    weeks = sorted(year_df['week'].dropna().unique())
    if len(weeks) < 2:
        return None

    # Count games per week
    games_per_week = {}
    for week in weeks:
        week_games = len(year_df[year_df['week'] == week]) // 2
        games_per_week[week] = week_games

    # Find regular season game count (mode of first 10 weeks)
    early_weeks = [w for w in weeks if w <= 10]
    if early_weeks:
        game_counts = [games_per_week[w] for w in early_weeks]
        regular_game_count = max(set(game_counts), key=game_counts.count)
    else:
        regular_game_count = max(games_per_week.values())

    # Detect playoff start: first week with fewer games than regular season
    playoff_start = None
    for week in weeks:
        if games_per_week[week] < regular_game_count:
            playoff_start = int(week)
            break

    if playoff_start is None:
        return None

    # Infer num_playoff_teams from first playoff week games
    first_playoff_games = games_per_week.get(playoff_start, 2)
    num_playoff_teams = first_playoff_games * 2

    # Add 2 if there are 3+ playoff weeks (suggests byes)
    playoff_weeks = [w for w in weeks if w >= playoff_start]
    if len(playoff_weeks) >= 3:
        num_playoff_teams = max(num_playoff_teams, 4)

    # Count unique teams in league for num_teams
    num_teams = len(year_df['manager'].unique())

    config = {
        'playoff_start_week': playoff_start,
        'num_playoff_teams': num_playoff_teams,
        'bye_teams': 0,  # Can't reliably infer byes
        'has_multiweek_championship': 0,
        'uses_playoff_reseeding': 0,
        'num_teams': num_teams
    }

    print(f"  [SETTINGS] {year} (inferred): playoff_start={playoff_start}, "
          f"playoff_teams={num_playoff_teams} (detected from {regular_game_count} regular -> {first_playoff_games} playoff games)")

    return config


def load_league_settings(
    year: int,
    settings_dir: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    data_directory: Optional[str] = None
) -> Dict:
    """
    Load league settings for a specific year.

    Tries multiple sources in order:
    1. JSON files in settings_dir (league_settings_{year}_*.json)
    2. Parquet file in data_directory (league_settings.parquet)
    3. Default values

    Args:
        year: Season year
        settings_dir: Directory containing league_settings JSON files (auto-detected if None)
        df: DataFrame with league_id (used for auto-detection if settings_dir is None)
        data_directory: Path to league data directory (for finding league settings)

    Returns:
        Dictionary with playoff_start_week, num_playoff_teams, bye_teams, has_multiweek_championship, uses_playoff_reseeding
    """
    # Try parquet first if data_directory is provided (most reliable for imports)
    if data_directory:
        parquet_config = _load_from_parquet(Path(data_directory), year)
        if parquet_config:
            # Validate bracket structure
            is_valid, error_msg = validate_bracket_structure(
                parquet_config['num_playoff_teams'],
                parquet_config['bye_teams'],
                parquet_config['num_teams']
            )

            if is_valid:
                print(f"  [SETTINGS] {year}: playoff_start={parquet_config['playoff_start_week']}, "
                      f"playoff_teams={parquet_config['num_playoff_teams']}, bye_teams={parquet_config['bye_teams']}")
                return parquet_config
            else:
                print(f"  [WARN] Invalid parquet settings for {year}: {error_msg}")

    # Helper function to load and validate JSON settings
    def _try_load_json_settings(search_path: Path) -> Optional[Dict]:
        """Try to load JSON settings from a directory."""
        settings_files = list(search_path.glob(f"league_settings_{year}_*.json"))
        if not settings_files:
            return None
        try:
            with open(settings_files[0], 'r') as f:
                settings = json.load(f)
                metadata = settings.get('metadata', settings)

                config = {
                    'playoff_start_week': int(metadata.get('playoff_start_week', 15)),
                    'num_playoff_teams': int(metadata.get('num_playoff_teams', metadata.get('playoff_teams', 6))),
                    'bye_teams': int(metadata.get('bye_teams', 2)),
                    'has_multiweek_championship': int(metadata.get('has_multiweek_championship', 0)),
                    'uses_playoff_reseeding': int(metadata.get('uses_playoff_reseeding', 0)),
                    'num_teams': int(metadata.get('num_teams', 10))
                }

                # Validate bracket structure
                is_valid, error_msg = validate_bracket_structure(
                    config['num_playoff_teams'],
                    config['bye_teams'],
                    config['num_teams']
                )

                if is_valid:
                    print(f"  [SETTINGS] {year}: playoff_start={config['playoff_start_week']}, "
                          f"playoff_teams={config['num_playoff_teams']}, bye_teams={config['bye_teams']}")
                    return config
                else:
                    print(f"  [WARN] Invalid JSON settings for {year}: {error_msg}")
                    return None
        except Exception as e:
            print(f"  [WARN] Failed to load JSON settings: {e}")
            return None

    # Try JSON files from settings_dir (explicit or auto-detected)
    if settings_dir is None:
        # Use centralized league-agnostic discovery function
        if data_directory:
            settings_path = find_league_settings_directory(data_directory=Path(data_directory), df=df)
        else:
            settings_path = find_league_settings_directory(df=df)
        if settings_path:
            settings_dir = str(settings_path)

    if settings_dir:
        config = _try_load_json_settings(Path(settings_dir))
        if config:
            return config

    # FALLBACK: Try JSON files directly in data_directory (for league imports)
    if data_directory:
        config = _try_load_json_settings(Path(data_directory))
        if config:
            return config

    # INFERENCE: Try to infer playoff settings from data structure
    if df is not None and not df.empty:
        inferred = _infer_settings_from_data(df, year)
        if inferred:
            return inferred

    # Fall back to defaults
    print(f"  [WARN] No settings found for year {year}, using defaults")
    return _get_default_settings()
