#!/usr/bin/env python3
"""
DEPRECATED: This module has been refactored.

All functionality has been moved to:
    - md.core: Core primitives (run_query, T, sql_quote, list_*, etc.)
    - md.tab_data_access.*: Tab-specific data loaders

Import examples:
    from md.core import run_query, T, sql_quote, list_seasons
    from md.tab_data_access.players import load_season_player_data
    from md.tab_data_access.draft import load_draft_data

This file is kept for any remaining internal references.
See md/REFACTORING_PLAN.md for migration details.
"""
from __future__ import annotations

# Re-export core primitives for any remaining internal usage
from .core import (
    get_current_league_db,
    get_table_dict,
    T,
    _TableDict,
    get_motherduck_connection,
    _execute_query,
    run_query,
    sql_quote,
    sql_in_list,
    sql_upper,
    sql_upper_in_list,
    sql_manager_norm,
    SORT_MARKER,
    latest_season_and_week,
    list_seasons,
    list_weeks,
    list_managers,
    list_player_seasons,
    list_player_weeks,
    list_player_positions,
    list_optimal_seasons,
    list_optimal_weeks,
    detect_roster_structure,
    STARTER_POSITIONS,
    BENCH_POSITIONS,
)

# Re-export MotherDuckConnection for backward compatibility
from .motherduck_connection import MotherDuckConnection
