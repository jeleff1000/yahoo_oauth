"""
Recap Helpers
=============
Utility functions for generating contextual messages and dialogue.
"""

from .contextual_helpers import (
    _safe_get,
    _safe_compare,
    get_optimal_lineup_message,
    get_streak_message,
    get_weekly_ranking_message,
    get_playoff_race_message,
    get_schedule_luck_message,
    get_alternate_schedule_message,
    get_grade_message,
    get_milestone_message,
    get_power_ranking_message,
    get_player_percentile_message,
    build_player_spotlight_lines,
    build_player_spotlight_paragraph,
    compose_weekly_paragraph,
    compose_season_paragraph,
    generate_all_contextual_messages,
)

from .recap_dialogue import (
    get_outcome_dialogue,
    format_dialogue,
    CONTEXTUAL_MESSAGES,
    PLAYER_PERFORMANCE,
    SEASON_ANALYSIS,
)

__all__ = [
    "_safe_get",
    "_safe_compare",
    "get_optimal_lineup_message",
    "get_streak_message",
    "get_weekly_ranking_message",
    "get_playoff_race_message",
    "get_schedule_luck_message",
    "get_alternate_schedule_message",
    "get_grade_message",
    "get_milestone_message",
    "get_power_ranking_message",
    "get_player_percentile_message",
    "build_player_spotlight_lines",
    "build_player_spotlight_paragraph",
    "compose_weekly_paragraph",
    "compose_season_paragraph",
    "generate_all_contextual_messages",
    "get_outcome_dialogue",
    "format_dialogue",
    "CONTEXTUAL_MESSAGES",
    "PLAYER_PERFORMANCE",
    "SEASON_ANALYSIS",
]
