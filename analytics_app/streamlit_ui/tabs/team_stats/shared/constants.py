"""
Constants for team stats module.

This module centralizes all configuration constants including:
- Color palettes for positions, themes, and charts
- Column name mappings for display
- Position-specific column configurations
- Navigation and tab configurations
"""

from typing import Dict, List

# ============================================================================
# POSITION COLOR PALETTE
# ============================================================================

POSITION_COLORS = {
    'QB': '#3B82F6',    # Blue - strategic/cerebral
    'RB': '#10B981',    # Green - ground & pound
    'WR': '#F59E0B',    # Orange - explosive
    'TE': '#EF4444',    # Red - versatile
    'K': '#8B5CF6',     # Purple - precision
    'DEF': '#6B7280',   # Gray - defensive
    'FLEX': '#EC4899',  # Pink - flexibility
    'All': '#3B82F6',   # Default blue
}

# Chart color sequences (for multi-series)
CHART_COLOR_SEQUENCE = [
    '#3B82F6',  # Blue
    '#10B981',  # Green
    '#F59E0B',  # Orange
    '#EF4444',  # Red
    '#8B5CF6',  # Purple
    '#EC4899',  # Pink
    '#14B8A6',  # Teal
    '#F97316',  # Dark Orange
]

# Stat type colors (for visualizations)
STAT_COLORS = {
    'passing_td': '#3B82F6',     # Blue
    'rushing_td': '#10B981',     # Green
    'receiving_td': '#F59E0B',   # Orange
    'defensive_td': '#EF4444',   # Red
    'passing_yds': '#60A5FA',    # Light blue
    'rushing_yds': '#34D399',    # Light green
    'receiving_yds': '#FBBF24',  # Light orange
}

# Performance gradient colors
PERFORMANCE_GRADIENT = {
    'elite': '#10B981',      # Green
    'good': '#84CC16',       # Lime
    'average': '#FBBF24',    # Yellow
    'below': '#FB923C',      # Orange
    'poor': '#EF4444',       # Red
}

# ============================================================================
# THEME COLORS
# ============================================================================

THEME_COLORS_LIGHT = {
    'bg': '#FFFFFF',
    'text': '#1F2937',
    'primary': '#3B82F6',
    'secondary': '#6B7280',
    'success': '#10B981',
    'warning': '#F59E0B',
    'error': '#EF4444',
    'info': '#3B82F6',
    'card_bg': '#F9FAFB',
    'border': '#E5E7EB',
}

THEME_COLORS_DARK = {
    'bg': '#0E1117',
    'text': '#FAFAFA',
    'primary': '#60A5FA',
    'secondary': '#9CA3AF',
    'success': '#34D399',
    'warning': '#FBBF24',
    'error': '#F87171',
    'info': '#60A5FA',
    'card_bg': '#1E1F22',
    'border': '#3A3C41',
}

# ============================================================================
# COLUMN NAME MAPPINGS (Database → Display)
# ============================================================================

BASIC_COLUMN_RENAMES = {
    # Core
    'manager': 'Manager',
    'fantasy_position': 'Position',
    'year': 'Year',
    'week': 'Week',
    'points': 'Points',
    'spar': 'SPAR',
    'player_spar': 'Player SPAR',
    'manager_spar': 'Manager SPAR',

    # Passing
    'pass_yds': 'Pass Yds',
    'pass_td': 'Pass TD',
    'pass_int': 'INT',
    'pass_cmp': 'Comp',
    'pass_att': 'Pass Att',
    'pass_air_yds': 'Air Yds',
    'pass_yac': 'Pass YAC',
    'pass_epa': 'Pass EPA',
    'cpoe': 'CPOE',
    'pacr': 'PACR',

    # Rushing
    'rush_yds': 'Rush Yds',
    'rush_td': 'Rush TD',
    'rush_att': 'Rush Att',
    'rush_fumbles': 'Rush Fum',
    'rush_fumbles_lost': 'Rush Fum Lost',
    'rush_first_downs': 'Rush 1st',
    'rush_epa': 'Rush EPA',

    # Receiving
    'rec': 'Rec',
    'rec_yds': 'Rec Yds',
    'rec_td': 'Rec TD',
    'rec_tgt': 'Targets',
    'rec_yac': 'Rec YAC',
    'rec_air_yds': 'Rec Air Yds',
    'rec_epa': 'Rec EPA',
    'target_share': 'Target Share',
    'wopr': 'WOPR',
    'racr': 'RACR',

    # Kicking
    'fg_made': 'FGM',
    'fg_att': 'FGA',
    'fg_pct': 'FG%',
    'fg_long': 'FG Long',
    'fg_made_0_19': 'FG 0-19',
    'fg_made_20_29': 'FG 20-29',
    'fg_made_30_39': 'FG 30-39',
    'fg_made_40_49': 'FG 40-49',
    'fg_made_50_plus': 'FG 50+',
    'pat_made': 'PAT Made',
    'pat_att': 'PAT Att',

    # Defense
    'sacks': 'Sacks',
    'def_int': 'Def INT',
    'def_fumbles_rec': 'Fum Rec',
    'def_td': 'Def TD',
    'tackles_combined': 'Tackles',
    'tackles_for_loss': 'TFL',
    'qb_hits': 'QB Hits',
    'points_allowed': 'Pts Allow',
    'yds_allowed': 'Yds Allow',
}

# ============================================================================
# POSITION-SPECIFIC COLUMN CONFIGURATIONS
# ============================================================================

POSITION_COLUMNS = {
    'QB': [
        'Manager', 'Year', 'Week', 'Points', 'SPAR',
        'Pass Yds', 'Pass TD', 'INT', 'Comp', 'Pass Att',
        'CPOE', 'Pass EPA', 'Air Yds', 'Pass YAC',
        'Rush Yds', 'Rush TD', 'Rush Att',
    ],
    'RB': [
        'Manager', 'Year', 'Week', 'Points', 'SPAR',
        'Rush Yds', 'Rush TD', 'Rush Att', 'Rush EPA',
        'Rec', 'Rec Yds', 'Rec TD', 'Targets',
        'Target Share', 'WOPR',
    ],
    'WR': [
        'Manager', 'Year', 'Week', 'Points', 'SPAR',
        'Rec', 'Rec Yds', 'Rec TD', 'Targets',
        'Rec Air Yds', 'Rec YAC', 'Rec EPA',
        'Target Share', 'WOPR', 'RACR',
    ],
    'TE': [
        'Manager', 'Year', 'Week', 'Points', 'SPAR',
        'Rec', 'Rec Yds', 'Rec TD', 'Targets',
        'Rec Air Yds', 'Rec YAC', 'Target Share',
    ],
    'K': [
        'Manager', 'Year', 'Week', 'Points',
        'FGM', 'FGA', 'FG%', 'FG Long',
        'FG 0-19', 'FG 20-29', 'FG 30-39', 'FG 40-49', 'FG 50+',
        'PAT Made', 'PAT Att',
    ],
    'DEF': [
        'Manager', 'Year', 'Week', 'Points',
        'Sacks', 'Def INT', 'Fum Rec', 'Def TD',
        'Tackles', 'TFL', 'QB Hits',
        'Pts Allow', 'Yds Allow',
    ],
    'All': [
        'Manager', 'Year', 'Week', 'Position', 'Points', 'SPAR',
        'Pass Yds', 'Pass TD', 'Rush Yds', 'Rush TD',
        'Rec', 'Rec Yds', 'Rec TD',
    ],
}

# Advanced stats columns (by position)
ADVANCED_COLUMNS = {
    'QB': ['Comp%', 'YPA', 'TD%', 'INT%', 'Pass EPA/Play', 'CPOE', 'PACR'],
    'RB': ['YPC', 'Catch%', 'YPR', 'Rush EPA/Att', 'Target Share', 'WOPR'],
    'WR': ['Catch%', 'YPR', 'YPRT', 'RACR', 'Target Share', 'Rec EPA/Target'],
    'TE': ['Catch%', 'YPR', 'YPRT', 'Target Share'],
    'K': ['FG%', 'FG 40+%', 'Long'],
    'DEF': ['Sacks/G', 'TO/G', 'Pts Allow/G', 'Yds Allow/G'],
    'All': ['Efficiency', 'Consistency', 'Peak Score', 'Floor Score'],
}

# ============================================================================
# NUMERIC COLUMNS (for type coercion)
# ============================================================================

NUMERIC_COLUMNS = [
    'points', 'spar', 'player_spar', 'manager_spar',
    'pass_yds', 'pass_td', 'pass_int', 'pass_cmp', 'pass_att',
    'pass_air_yds', 'pass_yac', 'pass_epa', 'cpoe', 'pacr',
    'rush_yds', 'rush_td', 'rush_att', 'rush_fumbles',
    'rush_fumbles_lost', 'rush_first_downs', 'rush_epa',
    'rec', 'rec_yds', 'rec_td', 'rec_tgt', 'rec_yac',
    'rec_air_yds', 'rec_epa', 'target_share', 'wopr', 'racr',
    'fg_made', 'fg_att', 'fg_pct', 'fg_long',
    'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39',
    'fg_made_40_49', 'fg_made_50_plus',
    'pat_made', 'pat_att',
    'sacks', 'def_int', 'def_fumbles_rec', 'def_td',
    'tackles_combined', 'tackles_for_loss', 'qb_hits',
    'points_allowed', 'yds_allowed',
]

# ============================================================================
# TAB CONFIGURATIONS
# ============================================================================

TAB_ICONS = {
    'basic_stats': '📊',
    'advanced_stats': '📈',
    'visualizations': '📉',
    'matchup_stats': '⚔️',
    'optimal_lineups': '🎯',
}

TAB_LABELS = {
    'basic_stats': f"{TAB_ICONS['basic_stats']} Basic Stats",
    'advanced_stats': f"{TAB_ICONS['advanced_stats']} Advanced Stats",
    'visualizations': f"{TAB_ICONS['visualizations']} Visualizations",
}

VIEW_DESCRIPTIONS = {
    'weekly': 'Analyze weekly performance by position and manager',
    'season': 'Explore season-long trends and aggregated statistics',
    'career': 'View all-time records and career achievements',
}

# ============================================================================
# FILTER CONFIGURATIONS
# ============================================================================

POSITION_OPTIONS = ['All', 'QB', 'RB', 'WR', 'TE', 'K', 'DEF']

# Game type filters
GAME_TYPE_OPTIONS = {
    'all': 'All Games',
    'regular': 'Regular Season',
    'playoffs': 'Playoffs',
    'consolation': 'Consolation',
}

# ============================================================================
# CHART CONFIGURATIONS
# ============================================================================

DEFAULT_CHART_HEIGHT = 500
MIN_CHART_HEIGHT = 400
CHART_HEIGHT_PER_ROW = 40

CHART_LAYOUT_DEFAULTS = {
    'hovermode': 'x unified',
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1,
    },
    'margin': {'l': 50, 'r': 30, 't': 50, 'b': 50},
}

# ============================================================================
# METRIC FORMATTING
# ============================================================================

METRIC_FORMATS = {
    'points': '{:.1f}',
    'percentage': '{:.1f}%',
    'yards': '{:.0f}',
    'count': '{:.0f}',
    'ratio': '{:.2f}',
    'epa': '{:.2f}',
}

# ============================================================================
# PERFORMANCE THRESHOLDS (for color coding)
# ============================================================================

# Weekly points thresholds
WEEKLY_POINTS_THRESHOLDS = {
    'elite': 120,
    'good': 100,
    'average': 80,
    'below': 60,
}

# SPAR thresholds
SPAR_THRESHOLDS = {
    'elite': 10,
    'good': 5,
    'average': 0,
    'below': -5,
}

# Consistency thresholds (coefficient of variation)
CONSISTENCY_THRESHOLDS = {
    'very_consistent': 0.15,   # < 15% CV
    'consistent': 0.25,        # 15-25% CV
    'moderate': 0.35,          # 25-35% CV
    'volatile': 0.50,          # 35-50% CV
    # > 50% CV = very volatile
}

# ============================================================================
# EXPORT CONFIGURATIONS
# ============================================================================

EXPORT_FILENAME_TEMPLATES = {
    'weekly_basic': 'weekly_team_basic_stats_{year}_{week}.csv',
    'weekly_advanced': 'weekly_team_advanced_stats_{year}_{week}.csv',
    'season_basic': 'season_team_basic_stats_{year}.csv',
    'season_advanced': 'season_team_advanced_stats_{year}.csv',
    'career_basic': 'career_team_basic_stats.csv',
    'career_advanced': 'career_team_advanced_stats.csv',
}

MAX_EXCEL_ROWS = 10000  # Excel export only for datasets under this size

# ============================================================================
# CACHE CONFIGURATIONS
# ============================================================================

CACHE_TTL_SECONDS = 600  # 10 minutes
CACHE_SHOW_SPINNER = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_position_color(position: str) -> str:
    """Get color for a given position."""
    return POSITION_COLORS.get(position, POSITION_COLORS['All'])


def get_performance_color(value: float, thresholds: Dict[str, float]) -> str:
    """Get color based on performance value and thresholds."""
    if value >= thresholds['elite']:
        return PERFORMANCE_GRADIENT['elite']
    elif value >= thresholds['good']:
        return PERFORMANCE_GRADIENT['good']
    elif value >= thresholds['average']:
        return PERFORMANCE_GRADIENT['average']
    elif value >= thresholds['below']:
        return PERFORMANCE_GRADIENT['below']
    else:
        return PERFORMANCE_GRADIENT['poor']


def get_columns_for_position(position: str, view_type: str = 'basic') -> List[str]:
    """Get relevant columns for a position and view type."""
    if view_type == 'basic':
        return POSITION_COLUMNS.get(position, POSITION_COLUMNS['All'])
    elif view_type == 'advanced':
        base = ['Manager', 'Year', 'Week', 'Position', 'Points', 'SPAR']
        advanced = ADVANCED_COLUMNS.get(position, [])
        return base + advanced
    else:
        return POSITION_COLUMNS.get(position, POSITION_COLUMNS['All'])


def format_metric(value: float, metric_type: str) -> str:
    """Format a metric value based on its type."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return '-'

    format_str = METRIC_FORMATS.get(metric_type, '{:.1f}')
    return format_str.format(value)
