"""
MotherDuck Data Access Package.

Provides database connectivity and data loading utilities for the KMFFL Analytics app.

Modules:
    - core: Core primitives (run_query, T, sql helpers, list functions)
    - motherduck_connection: Database connection management
    - data_access: Tab-specific loaders (deprecated - use tab_data_access/)
    - data_cache: Caching decorators and utilities
    - tab_data_access/: Tab-specific optimized data loaders

Usage:
    # Core primitives
    from md.core import run_query, T, sql_quote, list_seasons, list_managers

    # Tab-specific loaders (preferred)
    from md.tab_data_access.homepage import load_optimized_homepage_data
    from md.tab_data_access.players import load_weekly_player_data
"""

# Export core primitives for convenience
from .core import (
    # Database config
    get_current_league_db,
    get_table_dict,
    T,
    # Connection
    get_motherduck_connection,
    run_query,
    # SQL helpers
    sql_quote,
    sql_in_list,
    sql_upper,
    sql_upper_in_list,
    sql_manager_norm,
    SORT_MARKER,
    # Common queries
    latest_season_and_week,
    list_seasons,
    list_weeks,
    list_managers,
    list_player_seasons,
    list_player_weeks,
    list_player_positions,
    list_optimal_seasons,
    list_optimal_weeks,
    # Utilities
    detect_roster_structure,
    STARTER_POSITIONS,
    BENCH_POSITIONS,
)

__all__ = [
    # Database config
    "get_current_league_db",
    "get_table_dict",
    "T",
    # Connection
    "get_motherduck_connection",
    "run_query",
    # SQL helpers
    "sql_quote",
    "sql_in_list",
    "sql_upper",
    "sql_upper_in_list",
    "sql_manager_norm",
    "SORT_MARKER",
    # Common queries
    "latest_season_and_week",
    "list_seasons",
    "list_weeks",
    "list_managers",
    "list_player_seasons",
    "list_player_weeks",
    "list_player_positions",
    "list_optimal_seasons",
    "list_optimal_weeks",
    # Utilities
    "detect_roster_structure",
    "STARTER_POSITIONS",
    "BENCH_POSITIONS",
]
