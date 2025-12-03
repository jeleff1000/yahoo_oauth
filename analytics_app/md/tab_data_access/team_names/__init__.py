"""
Team Names tab data access.

Optimized loaders for Team Names components.

Key Features:
    - Column selection: 5 of 276 columns (~98% reduction)
    - Row filtering: DISTINCT on manager/year (~95% reduction)
    - Database-level filters: Excludes unrostered managers
    - Cached with 10-minute TTL

Combined optimization: ~99.9% reduction in data transferred
"""
from .team_name_data import load_team_name_data
from .combined import load_optimized_team_names_data

__all__ = [
    "load_team_name_data",
    "load_optimized_team_names_data",
]
