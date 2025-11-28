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

        # Determine championship week (last playoff week)
        championship_week = max(playoff_weeks) if playoff_weeks else None

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

            # CRITICAL: Add championship losers to consolation_alive
            # These teams will play consolation games in future weeks (3rd place, 5th place, etc.)
            consolation_alive = consolation_alive | champ_losers

            # IMPORTANT: Save consolation records BEFORE processing this week's games
            # (needed for placement_rank calculation below)
            consolation_wins_before_week = consolation_wins.copy()
            consolation_losses_before_week = consolation_losses.copy()

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
            # CRITICAL: Track winners' vs losers' brackets properly
            # - Winners from previous round compete for BETTER placement
            # - Losers from previous round compete for WORSE placement
            week_cons_games = week_df[(week_df['is_consolation'] == 1)]

            if not week_cons_games.empty:
                # Total teams in league (use all managers who played this year)
                total_teams = len(seeds)

                # Find all unique consolation games this week
                cons_games = []
                processed_pairs = set()

                for _, row in week_cons_games.iterrows():
                    mgr = row['manager']
                    opp = row['opponent']

                    # Create a unique pair identifier (sorted to avoid duplicates)
                    pair = tuple(sorted([mgr, opp]))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        # Get seeds for both participants
                        mgr_seed = seeds.get(mgr, 999)
                        opp_seed = seeds.get(opp, 999)
                        cons_games.append((mgr, opp, min(mgr_seed, opp_seed), max(mgr_seed, opp_seed)))

                # Identify 3rd place game (best-seeded consolation game with playoff teams)
                # ONLY in the championship week (final playoff week)
                third_place_game = None
                remaining_games = []

                is_championship_week = (championship_week is not None and week == championship_week)

                # Sort games by best seed to identify 3rd place game first
                cons_games_sorted = sorted(cons_games, key=lambda x: x[2])

                for game_data in cons_games_sorted:
                    mgr, opp, min_seed, max_seed = game_data
                    # 3rd place game: both teams were in playoffs (seed <= num_playoff_teams)
                    # AND this is the championship week
                    if is_championship_week and min_seed <= num_playoff_teams and max_seed <= num_playoff_teams:
                        if third_place_game is None:
                            # This is the best-seeded playoff team matchup = 3rd place game
                            third_place_game = (mgr, opp)
                        else:
                            # Additional playoff team consolation games go to remaining
                            remaining_games.append(game_data)
                    else:
                        # All other consolation games
                        remaining_games.append(game_data)

                # Assign 3rd place game (only in championship week)
                if third_place_game is not None:
                    mgr, opp = third_place_game
                    game_mask = (df['year'] == year) & (df['week'] == week) & \
                               ((df['manager'] == mgr) | (df['manager'] == opp))
                    df.loc[game_mask, 'placement_rank'] = 3
                    df.loc[game_mask, 'consolation_round'] = 'third_place_game'
                    print(f"      {mgr} vs {opp}: placement_rank=3 (third_place_game)")

                # Assign remaining consolation games based on bracket tracking
                # CRITICAL: Differentiate winners' bracket (playing for better placement)
                #           from losers' bracket (playing for worse placement)

                placement_names = {
                    5: 'fifth_place_game',
                    7: 'seventh_place_game',
                    9: 'ninth_place_game',
                    11: 'eleventh_place_game',
                    13: 'thirteenth_place_game'
                }

                # DYNAMIC PLACEMENT: Calculate based on consolation W/L records
                # This handles ANY bracket structure (byes, uneven brackets, multiple tiers)
                #
                # Strategy:
                # 1. Calculate each team's consolation W/L record up to this week
                # 2. Group games by W/L record (teams with same record compete for adjacent placements)
                # 3. Teams with BETTER records (more wins) get BETTER placements
                # 4. Placement = num_playoff_teams + 1 + (# of teams with better consolation records)

                # Calculate consolation W/L records for all teams playing this week
                # Use the records BEFORE this week (saved earlier) to determine bracket tier
                team_cons_records = {}  # {manager: (wins, losses)}

                for game_data in remaining_games:
                    mgr, opp, _, _ = game_data

                    for team in [mgr, opp]:
                        if team not in team_cons_records:
                            # Get consolation W/L record BEFORE this week
                            wins = consolation_wins_before_week.get(team, 0)
                            losses = consolation_losses_before_week.get(team, 0)
                            team_cons_records[team] = (wins, losses)

                # Group games by W/L record (for placement calculation)
                # All teams with same W/L record compete for adjacent placements
                games_by_record = {}  # {(wins, losses): [game_data, ...]}

                for game_data in remaining_games:
                    mgr, opp, _, _ = game_data
                    mgr_record = team_cons_records[mgr]
                    opp_record = team_cons_records[opp]

                    # Both teams should have same record (or they wouldn't be matched)
                    # Use better record if they differ (edge case for byes)
                    record_key = max(mgr_record, opp_record, key=lambda r: (r[0], -r[1]))

                    if record_key not in games_by_record:
                        games_by_record[record_key] = []
                    games_by_record[record_key].append(game_data)

                # Sort record groups by performance (best records first)
                # Best record = most wins, then fewest losses
                sorted_records = sorted(games_by_record.keys(),
                                      key=lambda r: (r[0], -r[1]),
                                      reverse=True)

                # Assign placements: better records get better (lower) placements
                # CRITICAL FIX: Use championship_alive count to determine starting placement
                # Teams still alive in championship are competing for places 1 through len(championship_alive)
                # So first consolation placement should be len(championship_alive) + 1
                # Example: If 4 teams alive in championship (competing for 1st-4th),
                #          first consolation game is for 5th place, not (num_playoff_teams + 1)
                current_placement = len(championship_alive) + 1

                for record in sorted_records:
                    wins, losses = record
                    games_at_this_level = games_by_record[record]

                    # CRITICAL: Multiple games at same W/L level need DIFFERENT placements
                    # Example: 2 games with (1-1) record:
                    #   - Game 1 (better seeds): placement = 9 → 9/10 place game
                    #   - Game 2 (worse seeds): placement = 11 → 11/12 place game
                    #
                    # Sort games by seed quality (better seeds get better placement)
                    games_at_this_level.sort(key=lambda g: g[2])  # g[2] = min_seed (lower is better)

                    for game_data in games_at_this_level:
                        mgr, opp, min_seed, max_seed = game_data

                        # Assign this placement to both participants
                        game_mask = (df['year'] == year) & (df['week'] == week) & \
                                   ((df['manager'] == mgr) | (df['manager'] == opp))
                        df.loc[game_mask, 'placement_rank'] = current_placement

                        # Update consolation_round with specific placement game name
                        if current_placement in placement_names:
                            df.loc[game_mask, 'consolation_round'] = placement_names[current_placement]

                        record_str = f"{wins}-{losses}"
                        print(f"      {mgr} vs {opp}: placement_rank={current_placement} (record: {record_str}, seeds: {min_seed}-{max_seed})")

                        # Move to next placement within this tier
                        # Each game determines 2 places (winner and loser)
                        current_placement += 2

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

        # SACKO DETECTION: Find who lost the worst placement game in the final week
        # The consolation bracket has placement games (7th place, 9th place, 11th place, etc.)
        # The sacko is whoever LOST the game with the HIGHEST placement_rank in the championship week
        sacko_team = None

        # Find consolation games in the championship week (last playoff week)
        if championship_week is not None:
            # Need to get updated year_df with placement_rank from main df
            year_df_updated = df[df['year'] == year].copy()
            final_week_cons = year_df_updated[(year_df_updated['week'] == championship_week) &
                                               (year_df_updated['is_consolation'] == 1)]

            if not final_week_cons.empty and 'placement_rank' in final_week_cons.columns:
                # Find the worst (highest) placement_rank game
                max_placement = final_week_cons['placement_rank'].max()

                if pd.notna(max_placement) and max_placement > 0:
                    # Find who lost this worst placement game
                    worst_game = final_week_cons[final_week_cons['placement_rank'] == max_placement]
                    losers = worst_game[worst_game['loss'] == 1]

                    if not losers.empty:
                        # Should only be one loser per game
                        sacko_team = losers.iloc[0]['manager']
                        placement_name = losers.iloc[0].get('consolation_round', f'{int(max_placement)}th_place_game')
                        try:
                            print(f"  [SACKO] {sacko_team} lost {placement_name} (placement_rank={int(max_placement)})")
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            print(f"  [SACKO] Sacko detected in year {year} (lost placement_rank={int(max_placement)})")

        if sacko_team:
            # Mark sacko flag on the championship week game
            df.loc[(df['year'] == year) & (df['manager'] == sacko_team) &
                   (df['week'] == championship_week) & (df['is_consolation'] == 1), 'sacko'] = 1

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
