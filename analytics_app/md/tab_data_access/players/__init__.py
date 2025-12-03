"""
Player data access functions for optimized column loading.

This module provides optimized data loaders for the Player Stats tab:
- weekly_player_data: Loads 116 out of 270 columns for weekly player stats (57% reduction)
- h2h_player_data: Loads 16 out of 270 columns for H2H viewer (94% reduction)
- season_player_data: Loads ~105 out of 234 columns for season player stats (55% reduction)
- career_player_data: Loads ~103 out of 234 columns for career player stats (56% reduction)
"""

from .weekly_player_data import (
    load_weekly_player_data,
    load_filtered_weekly_player_data,
    load_player_week_data,
    load_optimal_week_data,
    load_h2h_week_data,
    load_h2h_optimal_week_data,
)

from .season_player_data import (
    load_season_player_data,
)

from .career_player_data import (
    load_career_player_data,
)

# Aliases for backward compatibility (old names -> new names)
load_player_week = load_h2h_week_data
load_optimal_week = load_h2h_optimal_week_data
load_players_weekly_data = load_weekly_player_data
load_filtered_weekly_data = load_filtered_weekly_player_data
load_players_season_data = load_season_player_data
load_players_career_data = load_career_player_data

__all__ = [
    # Primary exports
    "load_weekly_player_data",
    "load_filtered_weekly_player_data",
    "load_player_week_data",
    "load_optimal_week_data",
    "load_h2h_week_data",
    "load_h2h_optimal_week_data",
    "load_season_player_data",
    "load_career_player_data",
    # Aliases for backward compatibility
    "load_player_week",
    "load_optimal_week",
    "load_players_weekly_data",
    "load_filtered_weekly_data",
    "load_players_season_data",
    "load_players_career_data",
]
