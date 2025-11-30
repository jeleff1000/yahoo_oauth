"""
Configuration and constants for matchups tab.
Centralizes tab names, cache settings, and other shared config.
"""

# Tab names used throughout matchups views
# Format: (display_name, internal_key)
MATCHUP_TABS = {
    'matchup_stats': "📊 Matchup Stats",
    'advanced_stats': "🔬 Advanced Stats",
    'projected_stats': "📈 Projected Stats",
    'team_ratings': "⭐ Team Ratings",
    'optimal_lineups': "🎯 Optimal Lineups",
    'head_to_head': "🤝 Head-to-Head",
    'about': "ℹ️ About",
}

# Get ordered list of tab names
MATCHUP_TAB_NAMES = list(MATCHUP_TABS.values())

# Cache settings
CACHE_TTL_SECONDS = 600  # 10 minutes
SHOW_SPINNER_DEFAULT = True

# UI Messages
FILTER_TIP = """
💡 <strong>Tip:</strong> Use filters below to customize your view. Leave filters empty to view all data.
Filters maintain your tab position during calculations.
"""

NO_DATA_MESSAGE = "No data available"

EMPTY_FILTER_RESULT_MESSAGE = "No matchups found with the selected filters. Try adjusting your filter criteria."

# View-specific descriptions
VIEW_DESCRIPTIONS = {
    'weekly': {
        'title': "⚡ Weekly Matchup Viewer",
        'subtitle': "Explore Your Fantasy League's Weekly Performance",
        'features': {
            MATCHUP_TABS['matchup_stats']: "See basic weekly results, scores, and win/loss outcomes",
            MATCHUP_TABS['advanced_stats']: "Dive into margins, medians, streaks, grades, and analytics",
            MATCHUP_TABS['projected_stats']: "Compare actual vs. projected points and accuracy",
            MATCHUP_TABS['team_ratings']: "View power ratings, playoff odds, and team metrics",
            MATCHUP_TABS['optimal_lineups']: "See points left on bench and optimal outcomes",
            MATCHUP_TABS['head_to_head']: "Compare your record against each opponent",
        }
    },
    'season': {
        'title': "📅 Season Matchup Viewer",
        'subtitle': "Explore Your Fantasy League's Season Performance",
        'features': {
            MATCHUP_TABS['matchup_stats']: "See season-long results, scores, and win/loss outcomes",
            MATCHUP_TABS['advanced_stats']: "Dive into margins, medians, streaks, grades, and analytics",
            MATCHUP_TABS['projected_stats']: "Compare actual vs. projected points and accuracy",
            MATCHUP_TABS['team_ratings']: "View power ratings, playoff odds, and team metrics",
            MATCHUP_TABS['optimal_lineups']: "See points left on bench and optimal outcomes",
            MATCHUP_TABS['head_to_head']: "Compare your season record against each opponent",
        }
    },
    'career': {
        'title': "🏆 Career Matchup Viewer",
        'subtitle': "Explore Your Fantasy League's All-Time Performance",
        'features': {
            MATCHUP_TABS['matchup_stats']: "See all-time results, scores, and win/loss outcomes",
            MATCHUP_TABS['advanced_stats']: "Dive into career margins, medians, streaks, grades, and analytics",
            MATCHUP_TABS['projected_stats']: "Compare actual vs. projected points across all seasons",
            MATCHUP_TABS['optimal_lineups']: "See cumulative points left on bench and optimal outcomes",
            MATCHUP_TABS['team_ratings']: "View career power ratings and metrics",
            MATCHUP_TABS['head_to_head']: "Compare your all-time record against each opponent",
        }
    }
}
