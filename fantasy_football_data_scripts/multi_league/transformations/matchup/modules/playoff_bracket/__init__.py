"""
Playoff Bracket Package

Simulates playoff brackets using actual game results and league settings.

MODULAR STRUCTURE:
- utils: Settings loading, bracket validation
- championship_bracket: Champion detection
- consolation_bracket: Sacko detection
- placement_games: Placement game detection and labeling (3rd, 5th, 7th, etc.)

All configuration comes from league_settings JSON files.
NO HARDCODED VALUES. Fully generic and scalable.
"""

from functools import wraps
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

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

from core.data_normalization import normalize_numeric_columns, ensure_league_id

# Import submodules
from . import utils
from . import championship_bracket
from . import consolation_bracket
from . import placement_games


def ensure_normalized(func):
    """Decorator to ensure data normalization"""
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        df = normalize_numeric_columns(df)
        result = func(df, *args, **kwargs)
        result = normalize_numeric_columns(result)
        if 'league_id' in df.columns:
            league_id = df['league_id'].iloc[0] if len(df) > 0 else None
            if league_id:
                result = ensure_league_id(result, league_id)
        return result
    return wrapper


@ensure_normalized
def simulate_playoff_brackets(
    df: pd.DataFrame,
    settings_dir: Optional[str] = None,
    data_directory: Optional[str] = None
) -> pd.DataFrame:
    """
    Simulate playoff brackets using actual game results.

    Properly determines:
    - Champion: Winner of the championship game
    - Sacko: Loser of the worst placement game
    - Placement ranks: 3rd, 5th, 7th, 9th, etc.

    Uses league settings for all configuration (no hardcoded values).

    Args:
        df: DataFrame with matchup data including final_playoff_seed
        settings_dir: Directory with league_settings JSON files (auto-detected if None)
        data_directory: Path to league data directory (for finding league settings)

    Returns:
        DataFrame with corrected champion, sacko, and placement_rank flags
    """
    df = df.copy()

    # Initialize flags
    for col in ['champion', 'sacko', 'placement_rank', 'placement_game']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = 0  # Reset all flags

    # Ensure required columns
    required = ['manager', 'opponent', 'year', 'week', 'win', 'loss', 'final_playoff_seed']
    for col in required:
        if col not in df.columns:
            print(f"  [ERROR] Missing required column: {col}")
            return df

    # Process each year separately
    years = sorted(df['year'].dropna().unique())

    for year in years:
        year = int(year)

        # Load settings for this year
        settings = utils.load_league_settings(year, settings_dir, df=df, data_directory=data_directory)
        playoff_start_config = settings['playoff_start_week']
        num_playoff_teams = settings['num_playoff_teams']
        bye_teams = settings['bye_teams']
        has_multiweek_championship = settings.get('has_multiweek_championship', 0)

        year_df = df[df['year'] == year].copy()

        # Get final playoff seeds
        seeds = {}
        for manager in year_df['manager'].unique():
            mgr_data = year_df[year_df['manager'] == manager]
            seed_values = mgr_data['final_playoff_seed'].dropna()
            if not seed_values.empty:
                seeds[manager] = int(seed_values.iloc[0])
            else:
                # Fallback: use 999 for teams without seeds
                seeds[manager] = 999

        if not seeds:
            print(f"  [WARN] No seeds found for year {year}, skipping bracket simulation")
            continue

        # Get playoff weeks
        playoff_weeks = sorted(year_df[year_df['week'] >= playoff_start_config]['week'].dropna().unique())

        if not playoff_weeks:
            continue

        # Determine championship week (last playoff week)
        championship_week = max(playoff_weeks) if playoff_weeks else None

        print(f"\n  [YEAR {year}] Processing playoff brackets...")
        print(f"    Playoff weeks: {playoff_weeks}")
        print(f"    Championship week: {championship_week}")

        # 1. Detect and label placement games (MUST be done first!)
        df = placement_games.detect_and_label_placement_games(
            df, year, championship_week, num_playoff_teams, seeds
        )

        # 2. Detect champion
        df = championship_bracket.detect_champion(
            df, year, championship_week, has_multiweek_championship
        )

        # 3. Detect sacko
        df = consolation_bracket.detect_sacko(
            df, year, championship_week, seeds
        )

    # Verify we have exactly 1 champion and 1 sacko per year
    for year in years:
        year = int(year)
        year_df = df[df['year'] == year]

        num_champs = year_df['champion'].sum()
        num_sackos = year_df['sacko'].sum()

        if num_champs != 1:
            print(f"  [ERROR] Year {year} has {num_champs} champions (should be 1)")
        if num_sackos != 1:
            print(f"  [ERROR] Year {year} has {num_sackos} sackos (should be 1)")

    return df


# Import load_league_settings for backward compatibility
from .utils import load_league_settings

# Export public API
__all__ = ['simulate_playoff_brackets', 'load_league_settings', 'utils', 'championship_bracket', 'consolation_bracket', 'placement_games']
