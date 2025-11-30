"""
Managers tab data access.

Optimized loaders for Managers/Matchups tab components:
- Weekly matchup overview
- Season matchup overview
- Career matchup overview
- Manager graphs
"""
from .matchup_data import load_managers_matchup_data
from .summary_data import load_managers_summary_data
from .combined import load_optimized_managers_data

__all__ = [
    "load_managers_matchup_data",
    "load_managers_summary_data",
    "load_optimized_managers_data",
]
