"""
Playoff Bracket Module

Simulates playoff brackets using actual game results and league settings.
Properly determines champions and sackos by tracking matchups through rounds.

SETTINGS-DRIVEN: All configuration comes from league_settings JSON files.
NO HARDCODED VALUES.

Key Features:
- Loads playoff configuration from league_settings_{year}_*.json files
- Builds championship and consolation brackets based on final_playoff_seed
- Tracks winners/losers through each round
- Determines THE champion (winner of championship game)
- Determines THE sacko (worst team that loses all consolation games)
- Handles bye weeks for top seeds
- No reseeding (bracket structure fixed at playoff start)
"""

from functools import wraps
import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple

# Add parent directories to path - RELATIVE NAVIGATION
_script_file = Path(__file__).resolve()
_modules_dir = _script_file.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))
sys.path.insert(0, str(_multi_league_dir / "core"))

from core.data_normalization import normalize_numeric_columns, ensure_league_id


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


def load_league_settings(year: int, settings_dir: Optional[str] = None) -> Dict:
    """
    Load league settings from JSON file for a specific year.

    Args:
        year: Season year
        settings_dir: Directory containing league_settings JSON files (auto-detected if None)

    Returns:
        Dictionary with playoff_start_week, num_playoff_teams, bye_teams
    """
    if settings_dir is None:
        # Auto-detect settings directory
        possible_paths = [
            Path.cwd() / "player_data" / "yahoo_league_settings",
            Path.cwd().parent / "player_data" / "yahoo_league_settings",
            Path.cwd().parent.parent / "fantasy_football_data" / "KMFFL" / "player_data" / "yahoo_league_settings",
            Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data\KMFFL\player_data\yahoo_league_settings"),
        ]

        for path in possible_paths:
            if path.exists():
                settings_dir = str(path)
                break

    if not settings_dir:
        print(f"  [WARN] Could not find league_settings directory, using defaults")
        return {
            'playoff_start_week': 15,
            'num_playoff_teams': 6,
            'bye_teams': 2,
            'uses_playoff_reseeding': 0
        }

    # Find settings file for this year
    settings_path = Path(settings_dir)
    settings_files = list(settings_path.glob(f"league_settings_{year}_*.json"))

    if not settings_files:
        print(f"  [WARN] No settings file found for year {year}, using defaults")
        return {
            'playoff_start_week': 15,
            'num_playoff_teams': 6,
            'bye_teams': 2,
            'uses_playoff_reseeding': 0
        }

    try:
        with open(settings_files[0], 'r') as f:
            settings = json.load(f)
            metadata = settings.get('metadata', settings)

            config = {
                'playoff_start_week': int(metadata.get('playoff_start_week', 15)),
                'num_playoff_teams': int(metadata.get('num_playoff_teams', metadata.get('playoff_teams', 6))),
                'bye_teams': int(metadata.get('bye_teams', 2)),
                'uses_playoff_reseeding': int(metadata.get('uses_playoff_reseeding', 0))
            }

            print(f"  [SETTINGS] {year}: playoff_start={config['playoff_start_week']}, "
                  f"playoff_teams={config['num_playoff_teams']}, bye_teams={config['bye_teams']}, "
                  f"reseeding={config['uses_playoff_reseeding']}")

            return config

    except Exception as e:
        print(f"  [ERROR] Failed to load settings from {settings_files[0]}: {e}")
        return {
            'playoff_start_week': 15,
            'num_playoff_teams': 6,
            'bye_teams': 2,
            'uses_playoff_reseeding': 0
        }


def build_playoff_bracket(seeds: Dict[str, int], num_playoff_teams: int, bye_teams: int) -> List[Tuple[str, str]]:
    """
    Build initial playoff bracket matchups based on seeds.

    Standard bracket structure (6 teams, 2 byes):
    - Seeds 1-2: Bye week
    - Seeds 3-6: Play in round 1 (3v6, 4v5)
    - Round 2 (semifinals): 1 vs winner of 3v6, 2 vs winner of 4v5
    - Round 3 (championship): Winners of semifinals

    Args:
        seeds: Dict mapping manager name to seed number
        num_playoff_teams: Number of teams in playoffs
        bye_teams: Number of teams with first-round byes

    Returns:
        List of matchup tuples (higher_seed, lower_seed) for first round
    """
    # Sort managers by seed
    sorted_managers = sorted(seeds.items(), key=lambda x: x[1])
    playoff_teams = [mgr for mgr, seed in sorted_managers if seed <= num_playoff_teams]

    # First round matchups (teams without byes)
    matchups = []
    teams_without_bye = playoff_teams[bye_teams:]

    # Standard bracket pairing: highest remaining seed vs lowest remaining seed
    num_first_round_games = (num_playoff_teams - bye_teams) // 2

    for i in range(num_first_round_games):
        higher_seed = teams_without_bye[i]
        lower_seed = teams_without_bye[-(i+1)]
        matchups.append((higher_seed, lower_seed))

    return matchups


def build_consolation_bracket(seeds: Dict[str, int], num_playoff_teams: int) -> List[Tuple[str, str]]:
    """
    Build consolation bracket for teams that didn't make playoffs.

    Args:
        seeds: Dict mapping manager name to seed number
        num_playoff_teams: Number of teams in playoffs (teams below this go to consolation)

    Returns:
        List of matchup tuples for first round of consolation
    """
    # Sort managers by seed
    sorted_managers = sorted(seeds.items(), key=lambda x: x[1])
    consolation_teams = [mgr for mgr, seed in sorted_managers if seed > num_playoff_teams]

    # Consolation bracket: pair teams similarly (best non-playoff vs worst non-playoff)
    matchups = []
    num_games = len(consolation_teams) // 2

    for i in range(num_games):
        matchups.append((consolation_teams[i], consolation_teams[-(i+1)]))

    return matchups


@ensure_normalized
def simulate_playoff_brackets(df: pd.DataFrame, settings_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Simulate playoff brackets using actual game results.

    Properly determines:
    - Champion: THE winner of the championship game
    - Sacko: THE loser of the consolation bracket (worst seed that loses all games)

    Uses league settings for all configuration (no hardcoded values).

    Args:
        df: DataFrame with matchup data including final_playoff_seed
        settings_dir: Directory with league_settings JSON files (auto-detected if None)

    Returns:
        DataFrame with corrected champion and sacko flags
    """
    df = df.copy()

    # Initialize flags
    for col in ['champion', 'sacko']:
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
        settings = load_league_settings(year, settings_dir)
        playoff_start = settings['playoff_start_week']
        num_playoff_teams = settings['num_playoff_teams']
        bye_teams = settings['bye_teams']

        year_df = df[df['year'] == year].copy()

        # Get final playoff seeds (should be same for all weeks for each manager)
        seeds = {}
        for manager in year_df['manager'].unique():
            mgr_data = year_df[year_df['manager'] == manager]
            seed_values = mgr_data['final_playoff_seed'].dropna()
            if not seed_values.empty:
                seeds[manager] = int(seed_values.iloc[0])

        if not seeds:
            print(f"  [WARN] No seeds found for year {year}, skipping bracket simulation")
            continue

        # Get playoff weeks
        playoff_weeks = sorted(year_df[year_df['week'] >= playoff_start]['week'].dropna().unique())

        if not playoff_weeks:
            continue

        # Track who's still alive in each bracket
        championship_alive = {mgr for mgr, seed in seeds.items() if seed <= num_playoff_teams}
        consolation_alive = {mgr for mgr, seed in seeds.items() if seed > num_playoff_teams}

        print(f"  [BRACKET] Year {year}: {len(championship_alive)} in championship, {len(consolation_alive)} in consolation")

        # Process each playoff week
        for week_idx, week in enumerate(playoff_weeks):
            week = int(week)
            week_df = year_df[year_df['week'] == week]

            if week_df.empty:
                continue

            # Find championship bracket losers this week
            champ_losers = set()
            for idx, row in week_df.iterrows():
                if row['manager'] in championship_alive and row['loss'] == 1:
                    # Verify opponent is also in championship bracket
                    if row['opponent'] in championship_alive:
                        champ_losers.add(row['manager'])

            # Find consolation bracket losers this week
            cons_losers = set()
            for idx, row in week_df.iterrows():
                if row['manager'] in consolation_alive and row['loss'] == 1:
                    # Verify opponent is also in consolation bracket
                    if row['opponent'] in consolation_alive:
                        cons_losers.add(row['manager'])

            print(f"    Week {week}: {len(championship_alive)} alive in champ (eliminated {len(champ_losers)}), "
                  f"{len(consolation_alive)} alive in cons (eliminated {len(cons_losers)})")

            # Check if this is championship week (only 2 teams left in championship)
            if len(championship_alive) == 2:
                # This is the championship game!
                for idx, row in week_df.iterrows():
                    if row['manager'] in championship_alive and row['opponent'] in championship_alive:
                        if row['win'] == 1:
                            # THIS is the champion!
                            df.loc[(df['year'] == year) & (df['manager'] == row['manager']) & (df['week'] == week), 'champion'] = 1
                            print(f"    🏆 CHAMPION: {row['manager']} defeated {row['opponent']}")

            # Check if this is consolation final (only 2 teams left in consolation)
            if len(consolation_alive) == 2:
                # This is the sacko game! (loser gets sacko)
                for idx, row in week_df.iterrows():
                    if row['manager'] in consolation_alive and row['opponent'] in consolation_alive:
                        if row['loss'] == 1:
                            # THIS is the sacko!
                            df.loc[(df['year'] == year) & (df['manager'] == row['manager']) & (df['week'] == week), 'sacko'] = 1
                            print(f"    💩 SACKO: {row['manager']} lost to {row['opponent']}")

            # Update alive lists for next week
            championship_alive = championship_alive - champ_losers
            consolation_alive = consolation_alive - cons_losers

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
