"""
RECAP CONFIGURATION TABLES
==========================
Easy-to-edit tables that define what paragraphs appear and when.

Each table is a list of dicts with:
- "condition": A function that takes a row dict and returns True/False
- "text": Template string with {placeholders} for data values
- "priority": (optional) Lower number = appears first in paragraph

To add new criteria: Just add a new dict to the appropriate table.
To modify text: Edit the "text" field.
To change when something appears: Edit the "condition" function.
"""

# =============================================================================
# WEEKLY RECAP CRITERIA
# =============================================================================
# These build the weekly matchup recap paragraph

WEEKLY_RESULT_TEMPLATES = {
    # Core result sentence based on win/loss
    "win": "{manager} defeated {opponent} {team_points:.2f} to {opponent_points:.2f}",
    "loss": "{manager} fell to {opponent} {team_points:.2f} to {opponent_points:.2f}",
}

WEEKLY_CONTEXT_CRITERIA = [
    # Margin context
    {
        "condition": lambda r: r.get("win") == 1 and abs(r.get("margin", 0)) >= 30,
        "text": "a dominant {margin:.1f}-point blowout",
        "category": "margin",
    },
    {
        "condition": lambda r: r.get("win") == 1 and abs(r.get("margin", 0)) >= 15,
        "text": "a comfortable {margin:.1f}-point victory",
        "category": "margin",
    },
    {
        "condition": lambda r: r.get("win") == 1 and abs(r.get("margin", 0)) < 5,
        "text": "a nail-biter decided by just {margin:.1f} points",
        "category": "margin",
    },
    {
        "condition": lambda r: r.get("loss") == 1 and abs(r.get("margin", 0)) >= 30,
        "text": "a rough {margin:.1f}-point loss",
        "category": "margin",
    },
    {
        "condition": lambda r: r.get("loss") == 1 and abs(r.get("margin", 0)) < 5,
        "text": "a heartbreaker decided by just {margin:.1f} points",
        "category": "margin",
    },

    # Projection accuracy
    {
        "condition": lambda r: r.get("above_proj_score") == 1 and r.get("win") == 1,
        "text": "outperforming projections",
        "category": "projection",
    },
    {
        "condition": lambda r: r.get("below_proj_score") == 1 and r.get("win") == 1,
        "text": "despite underperforming projections",
        "category": "projection",
    },
    {
        "condition": lambda r: r.get("above_proj_score") == 1 and r.get("loss") == 1,
        "text": "even while outperforming projections",
        "category": "projection",
    },

    # Spread/odds context
    {
        "condition": lambda r: r.get("underdog_wins") == 1,
        "text": "pulling off the upset as an underdog",
        "category": "spread",
    },
    {
        "condition": lambda r: r.get("favorite_losses") == 1,
        "text": "in a surprising upset loss as the favorite",
        "category": "spread",
    },

    # Streak context
    {
        "condition": lambda r: (r.get("winning_streak") or r.get("win_streak") or 0) >= 3,
        "text": "extending their winning streak to {winning_streak} games",
        "category": "streak",
    },
    {
        "condition": lambda r: (r.get("losing_streak") or r.get("loss_streak") or 0) >= 3,
        "text": "dropping their {losing_streak}th straight",
        "category": "streak",
    },

    # League standing context
    {
        "condition": lambda r: r.get("teams_beat_this_week", 0) >= 9,
        "text": "posting the highest score in the league this week",
        "category": "league",
    },
    {
        "condition": lambda r: r.get("teams_beat_this_week", 0) == 0 and r.get("loss") == 1,
        "text": "posting the lowest score in the league",
        "category": "league",
    },
    {
        "condition": lambda r: r.get("above_league_median") == 1,
        "text": "finishing above the league median",
        "category": "league",
    },
]

WEEKLY_MILESTONE_CRITERIA = [
    # Milestones get their own sentence
    {
        "condition": lambda r: r.get("wins_to_date") == 1 and r.get("win") == 1,
        "text": "This marks their first win of the season.",
    },
    {
        "condition": lambda r: r.get("wins_to_date") == r.get("losses_to_date") and r.get("win") == 1 and r.get("wins_to_date", 0) > 1,
        "text": "They're now back to .500 on the season.",
    },
    {
        "condition": lambda r: r.get("playoff_seed_to_date", 99) <= 4 and r.get("win") == 1,
        "text": "They currently sit in playoff position at the #{playoff_seed_to_date} seed.",
    },
    {
        "condition": lambda r: r.get("champion") == 1,
        "text": "With this victory, they clinch the championship!",
    },
    {
        "condition": lambda r: r.get("sacko") == 1,
        "text": "Unfortunately, this loss earns them the Sacko.",
    },
]


# =============================================================================
# SEASON RECAP CRITERIA
# =============================================================================
# These build the season progress paragraph

SEASON_RECORD_TEMPLATE = "{manager} sits at {wins_to_date}-{losses_to_date}"

SEASON_CONTEXT_CRITERIA = [
    # Playoff positioning
    {
        "condition": lambda r: r.get("playoff_seed_to_date", 99) == 1,
        "text": "holding the #1 seed",
        "category": "seed",
    },
    {
        "condition": lambda r: 1 < r.get("playoff_seed_to_date", 99) <= 4,
        "text": "currently in playoff position at #{playoff_seed_to_date}",
        "category": "seed",
    },
    {
        "condition": lambda r: 4 < r.get("playoff_seed_to_date", 99) <= 6,
        "text": "on the playoff bubble at #{playoff_seed_to_date}",
        "category": "seed",
    },
    {
        "condition": lambda r: r.get("playoff_seed_to_date", 99) > 6,
        "text": "outside the playoff picture at #{playoff_seed_to_date}",
        "category": "seed",
    },

    # Playoff probability
    {
        "condition": lambda r: (r.get("p_playoffs") or 0) >= 0.9,
        "text": "with a {p_playoffs_pct}% chance to make playoffs",
        "category": "probability",
    },
    {
        "condition": lambda r: 0.5 <= (r.get("p_playoffs") or 0) < 0.9,
        "text": "with a {p_playoffs_pct}% playoff probability",
        "category": "probability",
    },
    {
        "condition": lambda r: (r.get("p_playoffs") or 0) < 0.5 and (r.get("p_playoffs") or 0) > 0,
        "text": "facing long odds at just {p_playoffs_pct}% playoff probability",
        "category": "probability",
    },

    # Championship probability
    {
        "condition": lambda r: (r.get("p_champ") or 0) >= 0.15,
        "text": "and a {p_champ_pct}% shot at the title",
        "category": "championship",
    },

    # Schedule luck
    {
        "condition": lambda r: (r.get("wins_vs_shuffle_wins") or 0) >= 2,
        "text": "They've benefited from a favorable schedule, winning {wins_vs_shuffle_wins:.1f} more games than expected.",
        "category": "luck",
        "standalone": True,
    },
    {
        "condition": lambda r: (r.get("wins_vs_shuffle_wins") or 0) <= -2,
        "text": "Bad luck has cost them - they've won {wins_vs_shuffle_wins_abs:.1f} fewer games than their points suggest.",
        "category": "luck",
        "standalone": True,
    },

    # Projected finish
    {
        "condition": lambda r: r.get("proj_wins") is not None,
        "text": "On pace to finish with {proj_wins:.0f} wins.",
        "category": "projection",
        "standalone": True,
    },
]


# =============================================================================
# PLAYER RECAP CRITERIA
# =============================================================================
# These build the player performance paragraph

PLAYER_INTRO_TEMPLATE = "Top performers this week:"

PLAYER_HIGHLIGHT_CRITERIA = [
    # Best performer
    {
        "condition": lambda r: r.get("is_top_scorer") == True,
        "text": "{player} led the way with {points:.2f} points",
        "category": "top",
    },

    # Position leaders
    {
        "condition": lambda r: r.get("is_position_leader") == True,
        "text": "{player} ({position}) contributed {points:.2f}",
        "category": "position",
    },

    # Bench disappointments
    {
        "condition": lambda r: r.get("was_benched") == True and r.get("points", 0) > r.get("starter_points", 0),
        "text": "{player} put up {points:.2f} on the bench - more than the starter",
        "category": "bench",
    },

    # Percentile-based highlights
    {
        "condition": lambda r: (r.get("weekly_percentile") or 0) >= 95,
        "text": "{player} had an elite week, ranking in the {weekly_percentile:.0f}th percentile",
        "category": "percentile",
    },
    {
        "condition": lambda r: (r.get("weekly_percentile") or 0) <= 5 and r.get("started") == True,
        "text": "{player} struggled mightily, ranking in the bottom {weekly_percentile:.0f}%",
        "category": "percentile",
    },
]

PLAYER_SUMMARY_CRITERIA = [
    # Position group summaries
    {
        "condition": lambda r: r.get("position") == "QB",
        "text": "QB: {total_points:.1f} pts",
        "category": "position_total",
    },
    {
        "condition": lambda r: r.get("position") == "RB",
        "text": "RB: {total_points:.1f} pts",
        "category": "position_total",
    },
    {
        "condition": lambda r: r.get("position") == "WR",
        "text": "WR: {total_points:.1f} pts",
        "category": "position_total",
    },
    {
        "condition": lambda r: r.get("position") == "TE",
        "text": "TE: {total_points:.1f} pts",
        "category": "position_total",
    },
]


# =============================================================================
# HELPER: Format a value for display
# =============================================================================

def format_value(value, decimals=2):
    """Format numeric values for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)
