"""
Consolation Bracket Module

Handles consolation bracket simulation and sacko detection.
Tracks teams competing for final standings outside of championship track.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Optional

# Add parent directories to path - RELATIVE NAVIGATION
_script_file = Path(__file__).resolve()
_playoff_bracket_dir = _script_file.parent
_modules_dir = _playoff_bracket_dir.parent
_transformations_dir = _modules_dir.parent
_multi_league_dir = _transformations_dir.parent
_scripts_dir = _multi_league_dir.parent

sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_multi_league_dir))


def detect_sacko(
    df: pd.DataFrame,
    year: int,
    championship_week: Optional[int],
    seeds: Dict[str, int]
) -> pd.DataFrame:
    """
    Detect and mark the sacko (last place) for a specific year.

    SACKO DEFINITION: The team that lost the WORST placement game in the championship week.

    Logic:
    1. Find all consolation games in the championship week
    2. Identify the game with the HIGHEST placement_rank
    3. The LOSER of that game is the sacko

    Args:
        df: Full matchup DataFrame
        year: Year to process
        championship_week: Final playoff week
        seeds: Dict mapping manager to final_playoff_seed

    Returns:
        DataFrame with sacko flag set for the worst-placing manager
    """
    if championship_week is None:
        print(f"  [WARN] No championship week found for year {year}, cannot detect sacko")
        return df

    # Find consolation games in the championship week (last playoff week)
    year_mask = df['year'] == year
    final_week_cons = df[(df['week'] == championship_week) & (df['is_consolation'] == 1) & year_mask]

    if final_week_cons.empty:
        print(f"  [WARN] No consolation games in championship week for year {year}")
        return df

    if 'placement_rank' not in final_week_cons.columns:
        print(f"  [WARN] No placement_rank column found for year {year}")
        return df

    # Find the worst (highest) placement_rank game
    max_placement = final_week_cons['placement_rank'].max()

    if pd.isna(max_placement) or max_placement == 0:
        print(f"  [WARN] No valid placement_rank found for year {year}")
        return df

    # Find who lost this worst placement game
    worst_game = final_week_cons[final_week_cons['placement_rank'] == max_placement]
    losers = worst_game[worst_game['loss'] == 1]

    if losers.empty:
        print(f"  [WARN] Could not find loser of worst placement game for year {year}")
        return df

    # Should only be one loser per game
    sacko_team = losers.iloc[0]['manager']
    placement_name = losers.iloc[0].get('consolation_round', f'{int(max_placement)}th_place_game')

    # Mark sacko flag on the championship week game
    df.loc[(df['year'] == year) & (df['manager'] == sacko_team) &
           (df['week'] == championship_week) & (df['is_consolation'] == 1), 'sacko'] = 1

    try:
        print(f"  [SACKO] {sacko_team} lost {placement_name} (placement_rank={int(max_placement)})")
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(f"  [SACKO] Sacko detected in year {year} (lost placement_rank={int(max_placement)})")

    return df
