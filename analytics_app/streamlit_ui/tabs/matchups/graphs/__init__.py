"""
Matchup Graphs Module

This module provides comprehensive visualization tools for analyzing manager performance,
scoring patterns, lineup efficiency, and competitive dynamics.
"""

# Import all graph display functions for easy access
from .scoring_trends import display_scoring_trends
from .win_percentage_graph import display_win_percentage_graph
from .power_rating import display_power_rating_graph
from .scoring_distribution import display_scoring_distribution_graph
from .margin_of_victory import display_margin_of_victory_graph
from .optimal_lineup_efficiency import display_optimal_lineup_efficiency_graph
from .playoff_vs_regular import display_playoff_vs_regular_graph
from .strength_of_schedule import display_strength_of_schedule_graph

__all__ = [
    'display_scoring_trends',
    'display_win_percentage_graph',
    'display_power_rating_graph',
    'display_scoring_distribution_graph',
    'display_margin_of_victory_graph',
    'display_optimal_lineup_efficiency_graph',
    'display_playoff_vs_regular_graph',
    'display_strength_of_schedule_graph',
]
