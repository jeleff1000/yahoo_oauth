"""
Keepers tab data access.

Optimized loaders for Keepers components.

Key Features:
    - Column selection: 17 of 272 columns (~94% reduction)
    - Row filtering: Only max week per manager/year (~95% reduction)
    - Database-level filters: Excludes unrostered, DEF, K
    - Uses CTE for max week calculations
    - Includes player headshots for visual display
    - Cached with 10-minute TTL

Combined optimization: ~99.7% reduction in data transferred
"""

from .keeper_data import load_keeper_data
from .combined import load_optimized_keepers_data

__all__ = [
    "load_keeper_data",
    "load_optimized_keepers_data",
]
