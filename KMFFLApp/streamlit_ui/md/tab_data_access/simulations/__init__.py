"""
Simulations tab data access.

Optimized loaders for Simulations components.

Note: Currently uses SELECT * for matchup data because simulations rely on
many dynamically generated shuffle_* columns. Future optimization could
identify specific columns once all shuffle column names are documented.

Key Features:
    - Uses generic table mapping for scalability
    - Cached with 10-minute TTL
    - Ready for future column-level optimization
    - Specialized loaders for Playoff Machine, critical matchups, and clinch scenarios
"""
from .matchup_data import (
    load_simulation_matchup_data,
    load_playoff_machine_data,
    load_playoff_machine_schedule,
    load_critical_matchups_data,
    load_clinch_scenarios_data,
)
from .combined import load_optimized_simulations_data

__all__ = [
    "load_simulation_matchup_data",
    "load_playoff_machine_data",
    "load_playoff_machine_schedule",
    "load_critical_matchups_data",
    "load_clinch_scenarios_data",
    "load_optimized_simulations_data",
]
