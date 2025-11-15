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

from core.data_normalization import normalize_numeric_columns, ensure_league_id, find_league_settings_directory


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


def load_league_settings(year: int, settings_dir: Optional[str] = None, df: Optional[pd.DataFrame] = None, data_directory: Optional[str] = None) -> Dict:
    """
    Load league settings from JSON file for a specific year.

    Args:
        year: Season year
        settings_dir: Directory containing league_settings JSON files (auto-detected if None)
        df: DataFrame with league_id (used for auto-detection if settings_dir is None)
        data_directory: Path to league data directory (for finding league settings)

    Returns:
        Dictionary with playoff_start_week, num_playoff_teams, bye_teams, has_multiweek_championship, uses_playoff_reseeding
    """
    if settings_dir is None:
        # Use centralized league-agnostic discovery function
        # If data_directory provided, use it for more reliable lookup
        if data_directory:
            settings_path = find_league_settings_directory(data_directory=Path(data_directory), df=df)
        else:
            settings_path = find_league_settings_directory(df=df)
        if settings_path:
            settings_dir = str(settings_path)

    if not settings_dir:
        print(f"  [WARN] Could not find league_settings directory, using defaults")
        return {
            'playoff_start_week': 15,
            'num_playoff_teams': 6,
            'bye_teams': 2,
            'has_multiweek_championship': 0,
            'uses_playoff_reseeding': 0,
            'num_teams': 10
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
            'has_multiweek_championship': 0,
            'uses_playoff_reseeding': 0,
            'num_teams': 10
        }

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

            if not is_valid:
                print(f"  [ERROR] Invalid bracket configuration for {year}: {error_msg}")
                print(f"          Using defaults instead")
                return {
                    'playoff_start_week': 15,
                    'num_playoff_teams': 6,
                    'bye_teams': 2,
                    'has_multiweek_championship': 0,
                    'uses_playoff_reseeding': 0,
                    'num_teams': 10
                }

            print(f"  [SETTINGS] {year}: playoff_start={config['playoff_start_week']}, "
                  f"playoff_teams={config['num_playoff_teams']}, bye_teams={config['bye_teams']}, "
                  f"multiweek_champ={config['has_multiweek_championship']}, "
                  f"reseeding={config['uses_playoff_reseeding']}")

            return config

    except Exception as e:
        print(f"  [ERROR] Failed to load settings from {settings_files[0]}: {e}")
        return {
            'playoff_start_week': 15,
            'num_playoff_teams': 6,
            'bye_teams': 2,
            'has_multiweek_championship': 0,
            'uses_playoff_reseeding': 0,
            'num_teams': 10
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
def simulate_playoff_brackets(df: pd.DataFrame, settings_dir: Optional[str] = None, data_directory: Optional[str] = None) -> pd.DataFrame:
    """
    Simulate playoff brackets using actual game results.

    Properly determines:
    - Champion: THE winner of the championship game
    - Sacko: THE loser of the consolation bracket (worst seed that loses all games)

    Uses league settings for all configuration (no hardcoded values).

    Args:
        df: DataFrame with matchup data including final_playoff_seed
        settings_dir: Directory with league_settings JSON files (auto-detected if None)
        data_directory: Path to league data directory (for finding league settings)

    Returns:
        DataFrame with corrected champion and sacko flags
    """
    df = df.copy()

    # Initialize flags
    for col in ['champion', 'sacko', 'placement_rank']:
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

        # Load settings for this year (pass df for league_id extraction and data_directory for reliable path lookup)
        settings = load_league_settings(year, settings_dir, df=df, data_directory=data_directory)
        playoff_start_config = settings['playoff_start_week']
        num_playoff_teams = settings['num_playoff_teams']
        bye_teams = settings['bye_teams']
        has_multiweek_championship = settings.get('has_multiweek_championship', 0)
        uses_playoff_reseeding = settings.get('uses_playoff_reseeding', 0)

        year_df = df[df['year'] == year].copy()

        # SETTINGS-DRIVEN: Trust playoff_start_week from settings
        # playoff_start_week means: the first week ANY playoff games occur
        #
        # Example (6-team playoffs, 2 byes):
        #   - playoff_start_week: 15
        #   - bye_teams: 2
        #
        # This means:
        #   - Week 15: Seeds 3-6 play (first playoff games, round 1)
        #   - Week 16: Seeds 1-2 play winners (round 2 / semifinals)
        #   - Week 17: Championship
        #
        # NO ADJUSTMENT NEEDED - trust the settings file directly.

        playoff_start = playoff_start_config

        # Log playoff configuration
        config_msg = f"  [SETTINGS] Year {year}: playoff_start_week={playoff_start}, "
        config_msg += f"playoff_teams={num_playoff_teams}, bye_teams={bye_teams}"

        if has_multiweek_championship == 1:
            config_msg += ", multiweek_championship=YES"

        if uses_playoff_reseeding == 1:
            config_msg += ", reseeding=YES (matchups determined each round)"
        else:
            config_msg += ", reseeding=NO (fixed bracket)"

        if bye_teams > 0:
            config_msg += f" -> Byes enter week {playoff_start + 1}"

        print(config_msg)

        # Get final playoff seeds (should be same for all weeks for each manager)
        # For teams without final_playoff_seed, use regular season standing as fallback
        seeds = {}
        for manager in year_df['manager'].unique():
            mgr_data = year_df[year_df['manager'] == manager]
            seed_values = mgr_data['final_playoff_seed'].dropna()
            if not seed_values.empty:
                seeds[manager] = int(seed_values.iloc[0])
            else:
                # Fallback: estimate seed based on wins_to_date at end of regular season
                # (Teams with more wins get better seeds)
                reg_season_data = year_df[(year_df['manager'] == manager) & (year_df['week'] < playoff_start)]
                if not reg_season_data.empty:
                    # Use negative wins so higher wins = lower seed number
                    wins = reg_season_data['wins_to_date'].max()
                    seeds[manager] = 999 - int(wins * 10) if pd.notna(wins) else 999
                else:
                    seeds[manager] = 999

        if not seeds:
            print(f"  [WARN] No seeds found for year {year}, skipping bracket simulation")
            continue

        # Get playoff weeks
        playoff_weeks = sorted(year_df[year_df['week'] >= playoff_start]['week'].dropna().unique())

        if not playoff_weeks:
            continue

        # Track who's still alive in each bracket
        # Use is_playoffs and is_consolation flags to determine initial bracket membership
        playoff_df = year_df[year_df['is_playoffs'] == 1]
        cons_df = year_df[year_df['is_consolation'] == 1]

        championship_alive = set(playoff_df['manager'].unique()) if not playoff_df.empty else set()
        consolation_alive = set(cons_df['manager'].unique()) if not cons_df.empty else set()

        # Track win-loss records in consolation bracket (for sacko detection)
        # The sacko is the team with all losses (no wins) in consolation games
        consolation_wins = {}  # Track wins per team in consolation
        consolation_losses = {}  # Track losses per team in consolation

        # Initialize counters for all consolation teams
        for team in consolation_alive:
            consolation_wins[team] = 0
            consolation_losses[team] = 0

        print(f"  [BRACKET] Year {year}: {len(championship_alive)} in championship, {len(consolation_alive)} in consolation")

        # Process each playoff week
        for week_idx, week in enumerate(playoff_weeks):
            week = int(week)
            week_df = year_df[year_df['week'] == week]

            if week_df.empty:
                continue

            # Find championship bracket losers this week
            champ_losers = set()
            champ_losers_list = []  # Track who lost and to whom
            for idx, row in week_df.iterrows():
                if row['manager'] in championship_alive and row['loss'] == 1:
                    # Verify opponent is also in championship bracket
                    if row['opponent'] in championship_alive:
                        champ_losers.add(row['manager'])
                        champ_losers_list.append(row['manager'])

            # Track wins/losses for ALL consolation games (regardless of alive status)
            # Then find consolation bracket losers for elimination tracking
            for idx, row in week_df.iterrows():
                # Track all consolation games
                if row['is_consolation'] == 1:
                    if row['loss'] == 1:
                        consolation_losses[row['manager']] = consolation_losses.get(row['manager'], 0) + 1
                    elif row['win'] == 1:
                        consolation_wins[row['manager']] = consolation_wins.get(row['manager'], 0) + 1

            # Find consolation bracket losers this week (for elimination tracking)
            cons_losers = set()
            for idx, row in week_df.iterrows():
                if row['manager'] in consolation_alive and row['opponent'] in consolation_alive:
                    if row['loss'] == 1:
                        cons_losers.add(row['manager'])

            print(f"    Week {week}: {len(championship_alive)} alive in champ (eliminated {len(champ_losers)}), "
                  f"{len(consolation_alive)} alive in cons (eliminated {len(cons_losers)})")

            # Champion detection will be done AFTER all weeks are processed
            # We'll look for the team that WON a championship game
            # (championship column == 1, is_playoffs == 1, win == 1)

            # Sacko detection will be done AFTER all weeks are processed
            # (see below - we need to check win/loss records across ALL consolation games)

            # ASSIGN PLACEMENT_RANK for consolation games
            # Calculate the best possible placement for teams still alive this week
            if len(consolation_alive) > 0:
                # Total teams in league (use all managers who played this year)
                total_teams = len(seeds)

                # Base placement rank = total teams - consolation teams still alive + 1
                # This represents the BEST placement achievable by teams in these games
                base_placement = total_teams - len(consolation_alive) + 1

                # Find all consolation games this week
                cons_games = []
                processed_pairs = set()

                for idx, row in week_df.iterrows():
                    mgr = row['manager']
                    opp = row['opponent']

                    if mgr in consolation_alive and opp in consolation_alive:
                        # Create a unique pair identifier (sorted to avoid duplicates)
                        pair = tuple(sorted([mgr, opp]))
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            # Get seeds for both participants
                            mgr_seed = seeds.get(mgr, 999)
                            opp_seed = seeds.get(opp, 999)
                            cons_games.append((mgr, opp, min(mgr_seed, opp_seed)))

                # Sort games by best seed in each game (higher seeded games get better placements)
                cons_games.sort(key=lambda x: x[2])

                # Assign placement_rank to each game
                for game_idx, (mgr, opp, _) in enumerate(cons_games):
                    # Each game determines 2 placements (winner and loser)
                    # For multiple games, space them by 2
                    placement = base_placement + (game_idx * 2)

                    # Assign to both participants in this game
                    game_mask = (df['year'] == year) & (df['week'] == week) & \
                               ((df['manager'] == mgr) | (df['manager'] == opp))
                    df.loc[game_mask, 'placement_rank'] = placement

                    # Update consolation_round with specific placement game name
                    placement_names = {
                        1: 'championship',
                        3: 'third_place_game',
                        5: 'fifth_place_game',
                        7: 'seventh_place_game',
                        9: 'ninth_place_game',
                        11: 'eleventh_place_game',
                        13: 'thirteenth_place_game'
                    }
                    if placement in placement_names:
                        if placement == 1:
                            # Championship game only
                            df.loc[game_mask, 'playoff_round'] = placement_names[placement]
                        else:
                            # All placement games (3rd, 5th, 7th, 9th, etc.) are consolation
                            df.loc[game_mask, 'consolation_round'] = placement_names[placement]

                    print(f"      {mgr} vs {opp}: placement_rank={placement}")

            # Update alive lists for next week
            championship_alive = championship_alive - champ_losers
            consolation_alive = consolation_alive - cons_losers

        # CHAMPION DETECTION: Support both single-week and multiweek championships
        champ_games = year_df[year_df['championship'] == 1]

        if not champ_games.empty:
            if has_multiweek_championship == 1:
                # MULTIWEEK CHAMPIONSHIP: Accumulate scores across all championship weeks
                print(f"  [MULTIWEEK] Championship spans multiple weeks")

                # Get all teams who played in championship games
                champ_participants = set(champ_games['manager'].unique())

                # Accumulate total scores across all championship weeks
                total_scores = {}
                for participant in champ_participants:
                    participant_champ_games = champ_games[champ_games['manager'] == participant]
                    total_score = participant_champ_games['team_points'].sum()
                    total_scores[participant] = total_score
                    print(f"    {participant}: {total_score:.2f} total points across {len(participant_champ_games)} weeks")

                # Determine champion (highest total score)
                if total_scores:
                    champion_manager = max(total_scores, key=total_scores.get)
                    champion_score = total_scores[champion_manager]

                    # Mark champion flag on the LAST championship week for this manager
                    last_champ_week = champ_games[champ_games['manager'] == champion_manager]['week'].max()
                    df.loc[(df['year'] == year) & (df['manager'] == champion_manager) &
                           (df['week'] == last_champ_week), 'champion'] = 1

                    try:
                        print(f"  [CHAMPION] {champion_manager} won with {champion_score:.2f} total points")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        print(f"  [CHAMPION] Champion detected in year {year}")
            else:
                # SINGLE-WEEK CHAMPIONSHIP: Traditional winner-takes-all
                # Find the LAST week with championship games (the actual finals)
                last_champ_week = champ_games['week'].max()
                final_champ_games = champ_games[champ_games['week'] == last_champ_week]

                # The champion is the team that WON the final championship game
                champ_winners = final_champ_games[final_champ_games['win'] == 1]

                if not champ_winners.empty:
                    # Should only be one winner in the finals
                    for idx, row in champ_winners.iterrows():
                        # Mark this team as champion ONLY on this final week
                        df.loc[(df['year'] == year) & (df['manager'] == row['manager']) &
                               (df['week'] == last_champ_week), 'champion'] = 1
                        try:
                            print(f"  [CHAMPION] {row['manager']} defeated {row['opponent']} in week {last_champ_week}")
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            print(f"  [CHAMPION] Champion detected in year {year} week {last_champ_week}")

        # SACKO DETECTION: Find the team that lost ALL their consolation games
        # User requirement: "if you come in last place in all your consolation brackets you get a 1 in sacko"
        # The sacko is the team with:
        # - 0 wins in consolation
        # - Maximum number of losses (they played the most games and lost all of them)
        sacko_team = None
        max_losses = 0

        for team, wins in consolation_wins.items():
            losses = consolation_losses.get(team, 0)
            if wins == 0 and losses > max_losses:
                # This team has more losses than previous candidates
                sacko_team = team
                max_losses = losses

        if sacko_team and max_losses > 0:
            try:
                print(f"  [SACKO] {sacko_team}: 0 wins, {max_losses} losses in consolation")
            except (UnicodeEncodeError, UnicodeDecodeError):
                print(f"  [SACKO] Sacko detected in year {year}")
            # Award sacko flag on their last consolation game (highest playoff week)
            team_playoff_games = year_df[(year_df['manager'] == sacko_team) &
                                          (year_df['week'] >= playoff_start) &
                                          (year_df['is_consolation'] == 1)]
            if not team_playoff_games.empty:
                last_week = team_playoff_games['week'].max()
                df.loc[(df['year'] == year) & (df['manager'] == sacko_team) &
                       (df['week'] == last_week), 'sacko'] = 1

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
