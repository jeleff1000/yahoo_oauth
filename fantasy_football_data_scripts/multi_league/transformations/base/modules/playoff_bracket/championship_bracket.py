"""
Championship Bracket Module

Handles championship bracket simulation and champion detection.
Tracks teams competing for 1st place through playoff rounds.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Set

# Add parent directories to path - RELATIVE NAVIGATION
_script_file = Path(__file__).resolve()
_playoff_bracket_dir = _script_file.parent
_modules_dir = _playoff_bracket_dir.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))


def detect_champion(
    df: pd.DataFrame,
    year: int,
    championship_week: int,
    has_multiweek_championship: int
) -> pd.DataFrame:
    """
    Detect and mark the league champion for a specific year.

    Supports both single-week and multiweek championship formats.

    Args:
        df: Full matchup DataFrame
        year: Year to process
        championship_week: Final playoff week
        has_multiweek_championship: Whether championship spans multiple weeks

    Returns:
        DataFrame with champion flag set for the winning manager
    """
    year_df = df[df['year'] == year].copy()
    champ_games = year_df[year_df['championship'] == 1]

    if champ_games.empty:
        print(f"  [WARN] No championship games found for year {year}")
        return df

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

    return df
