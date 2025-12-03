"""
Homepage tab data access.

Optimized loaders for Homepage tab components:
- Overview stats
- Hall of Fame data
- Season standings
- Head-to-head matchups
- Recaps
"""

from .matchup_data import load_homepage_matchup_data
from .summary_stats import load_homepage_summary_stats
from .combined import load_optimized_homepage_data
from .recaps_matchup_data import load_recaps_matchup_data
from .recaps_player_data import load_player_two_week_slice

__all__ = [
    "load_homepage_matchup_data",
    "load_homepage_summary_stats",
    "load_optimized_homepage_data",
    "load_recaps_matchup_data",
    "load_player_two_week_slice",
]
