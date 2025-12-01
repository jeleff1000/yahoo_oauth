#!/usr/bin/env python3
"""
Import Flow Management

This module handles import-related logic including:
- Import protection (prevent double-clicks)
- Data templates
- Import flow coordination

Note: Most import execution logic remains in main.py due to tight
integration with Streamlit session state and UI.
"""

import time
from typing import Optional


def can_start_import(session_state) -> tuple[bool, str]:
    """
    Check if a new import can be started.
    Returns (can_start, reason_if_blocked).

    Args:
        session_state: Streamlit session state object.
    """
    # Check if import is flagged as in progress
    if session_state.get("import_in_progress", False):
        return False, "Import already in progress"

    # Check cooldown - prevent rapid clicks (30 second cooldown)
    last_import_time = session_state.get("last_import_triggered_at", 0)
    elapsed = time.time() - last_import_time
    if elapsed < 30:
        remaining = int(30 - elapsed)
        return False, f"Please wait {remaining}s before starting another import"

    # Check if we already have a job for this league
    if session_state.get("import_job_id"):
        return False, "An import job was already started. Check the status below."

    return True, ""


def mark_import_started(session_state):
    """Mark that an import has been started (for cooldown tracking)."""
    session_state.import_in_progress = True
    session_state.last_import_triggered_at = time.time()


def get_data_templates() -> dict:
    """
    Return templates for external data file imports.

    These templates define the expected columns for different data types
    when importing from external sources (ESPN, older Yahoo data, etc.)
    """
    return {
        "matchup": {
            "description": "Weekly matchup results (scores, wins/losses)",
            "required_columns": ["week", "year", "manager", "team_name", "team_points", "opponent", "opponent_points", "win", "loss"],
            "optional_columns": ["team_projected_points", "opponent_projected_points", "margin", "division_id", "is_playoffs", "is_consolation"],
            "example_row": {
                "week": 1, "year": 2013, "manager": "John", "team_name": "Team Awesome",
                "team_points": 125.5, "opponent": "Jane", "opponent_points": 110.2,
                "win": 1, "loss": 0, "division_id": "", "is_playoffs": 0, "is_consolation": 0
            }
        },
        "player": {
            "description": "Weekly player stats (points scored per player per week)",
            "required_columns": ["year", "week", "manager", "player", "points"],
            "optional_columns": ["nfl_position", "lineup_position", "nfl_team", "projected_points", "percent_started", "percent_owned"],
            "example_row": {
                "year": 2013, "week": 1, "manager": "John", "player": "Patrick Mahomes",
                "points": 28.5, "nfl_position": "QB", "lineup_position": "QB", "nfl_team": "KC"
            }
        },
        "draft": {
            "description": "Draft results (picks and costs)",
            "required_columns": ["year", "round", "pick", "manager", "player"],
            "optional_columns": ["cost", "keeper", "draft_type"],
            "example_row": {
                "year": 2013, "round": 1, "pick": 1, "manager": "John",
                "player": "Adrian Peterson", "cost": 65, "keeper": 0, "draft_type": "auction"
            }
        },
        "transactions": {
            "description": "Waiver/FA pickups, drops, and trades",
            "required_columns": ["year", "week", "manager", "player", "transaction_type"],
            "optional_columns": ["faab_bid", "source_type", "destination_type", "trade_partner"],
            "example_row": {
                "year": 2013, "week": 3, "manager": "John", "player": "Tyreek Hill",
                "transaction_type": "add", "faab_bid": 15, "source_type": "waivers"
            }
        },
        "schedule": {
            "description": "Season schedule (who plays who each week)",
            "required_columns": ["year", "week", "manager", "opponent"],
            "optional_columns": ["is_playoffs", "is_consolation", "week_start", "week_end"],
            "example_row": {
                "year": 2013, "week": 1, "manager": "John", "opponent": "Jane",
                "is_playoffs": 0, "is_consolation": 0
            }
        }
    }
