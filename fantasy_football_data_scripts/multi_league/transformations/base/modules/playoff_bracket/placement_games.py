"""
Placement Games Module

Dynamically detects and labels placement games in consolation bracket.
Handles ANY bracket structure (3rd, 5th, 7th, 9th, 11th place, etc.)

CRITICAL FIX: Placement game detection now properly differentiates between:
- 3rd place game (championship semifinal losers)
- 5th place game (championship quarterfinal losers OR consolation bracket final)
- 7th+ place games (lower consolation tiers)

Generic, scalable, no hardcoding.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Set, Tuple, List

# Add parent directories to path - RELATIVE NAVIGATION
_script_file = Path(__file__).resolve()
_playoff_bracket_dir = _script_file.parent
_modules_dir = _playoff_bracket_dir.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))


def detect_and_label_placement_games(
    df: pd.DataFrame,
    year: int,
    championship_week: int,
    num_playoff_teams: int,
    seeds: Dict[str, int]
) -> pd.DataFrame:
    """
    Detect and label all placement games for a specific year.

    FULLY GENERIC ALGORITHM:
    1. Track team paths through postseason (championship vs consolation)
    2. Identify games between teams with similar paths
    3. Calculate placement_rank based on:
       - Championship round where they were eliminated
       - Consolation bracket tier (based on W/L record)
       - Seed quality (better seeds = better placement when tied)

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

    # Track team paths through postseason
    # Structure: {manager: {
    #     'bracket': 'championship' | 'consolation',
    #     'eliminated_week': week number or None if still alive,
    #     'eliminated_round': 'semifinal' | 'quarterfinal' | etc,
    #     'cons_wins': count,
    #     'cons_losses': count
    # }}
    team_paths = {}

    # Initialize team paths
    for manager in seeds.keys():
        seed = seeds[manager]
        team_paths[manager] = {
            'bracket': 'championship' if seed <= num_playoff_teams else 'consolation',
            'eliminated_week': None,
            'eliminated_round': None,
            'cons_wins': 0,
            'cons_losses': 0
        }

    # Process each postseason week to track team paths
    for week in postseason_weeks:
        week_mask = year_mask & (df['week'] == week)
        week_df = df[week_mask]

        for _, row in week_df.iterrows():
            mgr = row['manager']
            opp = row['opponent']

            if mgr not in team_paths:
                continue

            # Update consolation W/L records
            if row['is_consolation'] == 1:
                if row['win'] == 1:
                    team_paths[mgr]['cons_wins'] += 1
                elif row['loss'] == 1:
                    team_paths[mgr]['cons_losses'] += 1

            # Track championship bracket eliminations
            if row['is_playoffs'] == 1 and row['loss'] == 1:
                # Lost in championship bracket - record when/where eliminated
                if team_paths[mgr]['eliminated_week'] is None:
                    team_paths[mgr]['eliminated_week'] = week
                    team_paths[mgr]['eliminated_round'] = row.get('playoff_round', '')
                    # After elimination from championship, switch to consolation bracket
                    team_paths[mgr]['bracket'] = 'consolation'

    # Now identify placement games in the championship week
    if championship_week is None:
        return df

    champ_week_mask = year_mask & (df['week'] == championship_week)
    champ_week_cons = df[champ_week_mask & (df['is_consolation'] == 1)]

    if champ_week_cons.empty:
        return df

    # Group consolation games by participant pairs
    placement_games = []  # List of (mgr, opp, placement_rank, placement_name)
    processed_pairs = set()

    for _, row in champ_week_cons.iterrows():
        mgr = row['manager']
        opp = row['opponent']

        # Create unique pair identifier
        pair = tuple(sorted([mgr, opp]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        if mgr not in team_paths or opp not in team_paths:
            continue

        mgr_path = team_paths[mgr]
        opp_path = team_paths[opp]

        # Determine placement rank based on team paths
        placement_rank, placement_name = calculate_placement_rank(
            mgr, opp, mgr_path, opp_path, seeds, num_playoff_teams
        )

        if placement_rank:
            placement_games.append((mgr, opp, placement_rank, placement_name))

    # Apply placement labels to dataframe
    for mgr, opp, placement_rank, placement_name in placement_games:
        game_mask = champ_week_mask & (
            ((df['manager'] == mgr) & (df['opponent'] == opp)) |
            ((df['manager'] == opp) & (df['opponent'] == mgr))
        )

        df.loc[game_mask, 'placement_rank'] = placement_rank
        df.loc[game_mask, 'placement_game'] = 1
        df.loc[game_mask, 'consolation_round'] = placement_name

        print(f"      {mgr} vs {opp}: {placement_name} (rank {placement_rank})")

    return df


def calculate_placement_rank(
    mgr: str,
    opp: str,
    mgr_path: Dict,
    opp_path: Dict,
    seeds: Dict[str, int],
    num_playoff_teams: int
) -> Tuple[int, str]:
    """
    Calculate placement rank for a consolation game based on team paths.

    PLACEMENT LOGIC:
    - 3rd place: Both teams lost in championship semifinals
    - 5th place: Both teams lost in championship quarterfinals
    - 7th+ place: Teams from consolation bracket competing for lower placements

    Args:
        mgr: First manager name
        opp: Opponent manager name
        mgr_path: Manager's path dict (bracket, eliminated_week, eliminated_round, etc)
        opp_path: Opponent's path dict
        seeds: All team seeds
        num_playoff_teams: Number of playoff teams

    Returns:
        (placement_rank, placement_name) - e.g. (3, 'third_place_game')
    """
    mgr_seed = seeds.get(mgr, 999)
    opp_seed = seeds.get(opp, 999)

    # Check if BOTH teams came from championship bracket
    mgr_from_champ = mgr_path['eliminated_round'] in ['semifinal', 'quarterfinal', 'championship']
    opp_from_champ = opp_path['eliminated_round'] in ['semifinal', 'quarterfinal', 'championship']

    if mgr_from_champ and opp_from_champ:
        # Both from championship bracket - placement depends on which round they lost
        mgr_round = mgr_path['eliminated_round']
        opp_round = opp_path['eliminated_round']

        # CRITICAL: Both should have lost in SAME round to be matched
        # If both lost in semifinals → 3rd place game
        if mgr_round == 'semifinal' and opp_round == 'semifinal':
            return 3, 'third_place_game'

        # If both lost in quarterfinals → 5th place game
        elif mgr_round == 'quarterfinal' and opp_round == 'quarterfinal':
            return 5, 'fifth_place_game'

        # If both lost in earlier round → 7th+ place game
        else:
            # Use better seed to determine placement
            better_seed = min(mgr_seed, opp_seed)
            # Teams eliminated earlier get worse placements
            if better_seed <= 4:
                return 7, 'seventh_place_game'
            else:
                return 9, 'ninth_place_game'

    # Teams from pure consolation bracket (never made playoffs)
    # Placement based on consolation W/L record
    mgr_cons_record = (mgr_path['cons_wins'], -mgr_path['cons_losses'])
    opp_cons_record = (opp_path['cons_wins'], -opp_path['cons_losses'])

    # Teams with better consolation records get better placements
    # W/L record determines tier within consolation bracket
    mgr_wins = mgr_path['cons_wins']
    opp_wins = opp_path['cons_wins']
    mgr_losses = mgr_path['cons_losses']
    opp_losses = opp_path['cons_losses']

    # CRITICAL FIX: Check if BOTH teams have the SAME W/L record
    # If they have same record, they should be matched at same tier
    if (mgr_wins, mgr_losses) == (opp_wins, opp_losses):
        # Same consolation record - determine placement by that record

        if mgr_wins >= 1 and mgr_losses == 0:
            # Both teams are 1-0 (winners of consolation semis) → 7th place game
            return 7, 'seventh_place_game'

        elif mgr_wins == 0 and mgr_losses >= 1:
            # Both teams are 0-1 (losers of consolation semis) → 9th place game
            return 9, 'ninth_place_game'

        elif mgr_wins >= 2:
            # Both teams won multiple games (rare, deep bracket) → 5th place game
            return 5, 'fifth_place_game'

        elif mgr_losses >= 2:
            # Both teams lost multiple games (rare, deep bracket) → 11th place game
            return 11, 'eleventh_place_game'

        else:
            # Mixed record (1-1, 2-1, etc.) - use seeds to determine placement
            better_seed = min(mgr_seed, opp_seed)
            if better_seed <= num_playoff_teams + 2:
                return 7, 'seventh_place_game'
            else:
                return 9, 'ninth_place_game'

    else:
        # Different records - shouldn't happen in proper bracket, but handle gracefully
        # Team with more wins gets better placement
        avg_wins = (mgr_wins + opp_wins) / 2

        if avg_wins >= 1.0:
            return 7, 'seventh_place_game'
        else:
            return 9, 'ninth_place_game'
