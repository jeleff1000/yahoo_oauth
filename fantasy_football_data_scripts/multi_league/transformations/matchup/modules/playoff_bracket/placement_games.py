"""
Placement Games Module

Dynamically detects and labels placement games in consolation bracket.
Handles ANY bracket structure (3rd, 5th, 7th, 9th, 11th place, etc.)

SCALABLE ALGORITHM:
1. Identify championship finalists (they don't play placement games)
2. Collect all valid consolation games (excluding exhibition games)
3. Sort games by tier (championship bracket dropouts first, then consolation by seed)
4. Assign SEQUENTIAL placement ranks (3rd, 5th, 7th, 9th...) - no duplicates

Generic, scalable, no hardcoding.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional

# Add parent directories to path - RELATIVE NAVIGATION
_script_file = Path(__file__).resolve()
_playoff_bracket_dir = _script_file.parent
_modules_dir = _playoff_bracket_dir.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))


# Mapping of round names to tier priority (lower = better placement)
ROUND_PRIORITY = {
    'semifinal': 1,      # 3rd place
    'quarterfinal': 2,   # 5th place
    'first_round': 3,    # 7th place (for 8+ team brackets)
}


def get_placement_name(rank: int) -> str:
    """Convert numeric rank to placement game name."""
    names = {
        3: 'third_place_game',
        5: 'fifth_place_game',
        7: 'seventh_place_game',
        9: 'ninth_place_game',
        11: 'eleventh_place_game',
        13: 'thirteenth_place_game',
    }
    return names.get(rank, f'{rank}th_place_game')


def detect_and_label_placement_games(
    df: pd.DataFrame,
    year: int,
    championship_week: int,
    num_playoff_teams: int,
    seeds: Dict[str, int]
) -> pd.DataFrame:
    """
    Detect and label all placement games for a specific year.

    SCALABLE ALGORITHM:
    1. Find championship finalists (winner=1st, loser=2nd) - exclude from placements
    2. Track team paths through postseason
    3. Collect valid consolation games (exclude exhibition games)
    4. Sort by tier and assign SEQUENTIAL placement ranks (3rd, 5th, 7th...)

    Args:
        df: Full matchup DataFrame
        year: Year to process
        championship_week: Final playoff week
        num_playoff_teams: Number of teams in playoff bracket
        seeds: Dict mapping manager to final_playoff_seed

    Returns:
        DataFrame with placement_rank and consolation_round labels updated
    """
    year_mask = df['year'] == year
    postseason_mask = year_mask & ((df['is_playoffs'] == 1) | (df['is_consolation'] == 1))
    postseason_weeks = sorted(df.loc[postseason_mask, 'week'].dropna().unique().astype(int))

    if not postseason_weeks:
        return df

    if championship_week is None:
        return df

    # Step 1: Find championship finalists (exclude from placement games)
    finalists = _find_championship_finalists(df, year, championship_week)
    if finalists:
        print(f"    Championship finalists (excluded from placements): {finalists}")

    # Step 2: Track team paths through postseason
    team_paths = _build_team_paths(df, year, postseason_weeks, seeds, num_playoff_teams)

    # Step 3: Collect valid consolation games (exclude exhibition games)
    champ_week_mask = year_mask & (df['week'] == championship_week)
    champ_week_cons = df[champ_week_mask & (df['is_consolation'] == 1)]

    if champ_week_cons.empty:
        return df

    # Collect unique matchups with their tier scores
    # Structure: [(mgr, opp, tier_score, best_seed), ...]
    game_tiers = []
    processed_pairs = set()
    teams_in_games = set()

    for _, row in champ_week_cons.iterrows():
        mgr = row['manager']
        opp = row['opponent']

        # Create unique pair identifier
        pair = tuple(sorted([mgr, opp]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        # Skip if either team is a championship finalist (exhibition game)
        if mgr in finalists or opp in finalists:
            print(f"    Skipping exhibition game: {mgr} vs {opp} (involves finalist)")
            continue

        # Skip if team already in another game this week (data corruption)
        if mgr in teams_in_games or opp in teams_in_games:
            print(f"    Skipping duplicate game: {mgr} vs {opp} (team already in game)")
            continue

        if mgr not in team_paths or opp not in team_paths:
            continue

        teams_in_games.add(mgr)
        teams_in_games.add(opp)

        mgr_path = team_paths[mgr]
        opp_path = team_paths[opp]

        # Calculate tier score for this game
        tier_score = _calculate_tier_score(mgr_path, opp_path, seeds, mgr, opp, num_playoff_teams)
        best_seed = min(seeds.get(mgr, 999), seeds.get(opp, 999))

        game_tiers.append((mgr, opp, tier_score, best_seed))

    # Step 4: Sort games by tier (lower tier = better placement) then by seed
    game_tiers.sort(key=lambda x: (x[2], x[3]))

    # Step 5: Assign SEQUENTIAL placement ranks starting at 3rd
    # Each game determines 2 consecutive placements (winner gets rank, loser gets rank+1)
    # So placement games are for: 3rd/4th, 5th/6th, 7th/8th, 9th/10th, etc.
    current_rank = 3  # Start at 3rd place (1st and 2nd are championship)

    for mgr, opp, tier_score, best_seed in game_tiers:
        placement_name = get_placement_name(current_rank)

        # Apply to dataframe
        game_mask = champ_week_mask & (
            ((df['manager'] == mgr) & (df['opponent'] == opp)) |
            ((df['manager'] == opp) & (df['opponent'] == mgr))
        )

        df.loc[game_mask, 'placement_rank'] = current_rank
        df.loc[game_mask, 'placement_game'] = 1
        df.loc[game_mask, 'consolation_round'] = placement_name

        print(f"      {mgr} vs {opp}: {placement_name} (rank {current_rank}, tier={tier_score})")

        # Next placement game is 2 ranks lower (3rd→5th, 5th→7th, etc.)
        current_rank += 2

    return df


def _find_championship_finalists(df: pd.DataFrame, year: int, championship_week: int) -> Set[str]:
    """
    Find the teams that played in the championship game.
    These teams should NOT be assigned placement ranks (they are 1st/2nd).
    """
    finalists = set()

    # Find championship game
    champ_games = df[(df['year'] == year) & (df['championship'] == 1)]
    if not champ_games.empty:
        # Get the final championship week (in case of multi-week championships)
        final_champ_week = champ_games['week'].max()
        final_champ = champ_games[champ_games['week'] == final_champ_week]

        for _, row in final_champ.iterrows():
            finalists.add(row['manager'])
            finalists.add(row['opponent'])

    return finalists


def _build_team_paths(
    df: pd.DataFrame,
    year: int,
    postseason_weeks: List[int],
    seeds: Dict[str, int],
    num_playoff_teams: int
) -> Dict[str, Dict]:
    """
    Track each team's path through the postseason bracket.

    Returns dict mapping manager to their path info:
    {
        'bracket': 'championship' | 'consolation',
        'eliminated_week': week number or None,
        'eliminated_round': round name or None,
        'cons_wins': count,
        'cons_losses': count,
        'cons_entry_week': first week playing consolation
    }
    """
    year_mask = df['year'] == year
    team_paths = {}

    # Initialize team paths
    for manager in seeds.keys():
        seed = seeds[manager]
        team_paths[manager] = {
            'bracket': 'championship' if seed <= num_playoff_teams else 'consolation',
            'eliminated_week': None,
            'eliminated_round': None,
            'cons_wins': 0,
            'cons_losses': 0,
            'cons_entry_week': None if seed <= num_playoff_teams else min(postseason_weeks) if postseason_weeks else None
        }

    # Process each postseason week
    for week in postseason_weeks:
        week_mask = year_mask & (df['week'] == week)
        week_df = df[week_mask]

        for _, row in week_df.iterrows():
            mgr = row['manager']

            if mgr not in team_paths:
                continue

            # Track consolation entry for championship bracket dropouts
            if row['is_consolation'] == 1:
                if team_paths[mgr]['cons_entry_week'] is None:
                    team_paths[mgr]['cons_entry_week'] = week

                if row['win'] == 1:
                    team_paths[mgr]['cons_wins'] += 1
                elif row['loss'] == 1:
                    team_paths[mgr]['cons_losses'] += 1

            # Track championship bracket eliminations
            if row['is_playoffs'] == 1 and row['loss'] == 1:
                if team_paths[mgr]['eliminated_week'] is None:
                    team_paths[mgr]['eliminated_week'] = week
                    team_paths[mgr]['eliminated_round'] = row.get('playoff_round', '')
                    team_paths[mgr]['bracket'] = 'consolation'

    return team_paths


def _calculate_tier_score(
    mgr_path: Dict,
    opp_path: Dict,
    seeds: Dict[str, int],
    mgr: str,
    opp: str,
    num_playoff_teams: int
) -> float:
    """
    Calculate a tier score for a consolation game.
    Lower score = better placement (closer to 3rd place).

    Tier scoring:
    - Championship semifinal losers: tier 1 (3rd place)
    - Championship quarterfinal losers: tier 2 (5th place)
    - Consolation winners (best records): tier 3 (7th place)
    - Consolation losers (worst records): tier 4+ (9th+ place)
    """
    mgr_seed = seeds.get(mgr, 999)
    opp_seed = seeds.get(opp, 999)

    # Check if teams came from championship bracket
    mgr_round = mgr_path.get('eliminated_round', '')
    opp_round = opp_path.get('eliminated_round', '')

    mgr_from_champ = mgr_round in ROUND_PRIORITY
    opp_from_champ = opp_round in ROUND_PRIORITY

    # Case 1: Both from championship bracket - use their elimination round
    if mgr_from_champ and opp_from_champ:
        # Average the round priorities (should be same for matched teams)
        mgr_priority = ROUND_PRIORITY.get(mgr_round, 10)
        opp_priority = ROUND_PRIORITY.get(opp_round, 10)
        return (mgr_priority + opp_priority) / 2

    # Case 2: One from championship, one from consolation
    # Championship dropout gets priority
    if mgr_from_champ or opp_from_champ:
        champ_round = mgr_round if mgr_from_champ else opp_round
        return ROUND_PRIORITY.get(champ_round, 10) + 0.5

    # Case 3: Both from pure consolation bracket
    # Score based on consolation record (more wins = better tier)
    mgr_cons_score = mgr_path['cons_wins'] - mgr_path['cons_losses']
    opp_cons_score = opp_path['cons_wins'] - opp_path['cons_losses']
    avg_cons_score = (mgr_cons_score + opp_cons_score) / 2

    # Better consolation records get lower (better) tier scores
    # Base tier for consolation is 3 (7th place), worse records go higher
    base_tier = 3.0
    # Each net loss adds 1 to tier
    tier = base_tier - avg_cons_score

    return tier
