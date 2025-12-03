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

# Re-export MotherDuckConnection for backward compatibility
