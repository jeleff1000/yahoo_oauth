"""
Shared utilities for the Streamlit UI.

Modules:
- themes: Light/dark mode theme system
- chart_themes: Plotly chart theming for light/dark mode
- dataframe_utils: DataFrame cleaning and transformation utilities
- responsive: Mobile detection and responsive layout utilities
"""

from .themes import (
    get_theme_colors,
    get_gradient,
    inject_theme_css,
    detect_theme
)

from .chart_themes import (
    get_chart_theme,
    get_chart_colors,
    apply_chart_theme,
    get_plotly_template,
    create_grade_bar_chart,
    create_regret_bar_chart,
    create_faab_tier_chart,
    create_horizontal_bar_chart,
    get_grade_colors_list,
    get_regret_colors_list,
    get_faab_tier_colors_list,
    GRADE_COLORS,
    REGRET_COLORS,
    FAAB_TIER_COLORS,
    CATEGORICAL_COLORS,
)

from .dataframe_utils import (
    clean_dataframe,
    ensure_numeric,
    apply_common_renames
)

__all__ = [
    # Theme functions
    'get_theme_colors',
    'get_gradient',
    'inject_theme_css',
    'detect_theme',

    # Chart theming
    'get_chart_theme',
    'get_chart_colors',
    'apply_chart_theme',
    'get_plotly_template',
    'create_grade_bar_chart',
    'create_regret_bar_chart',
    'create_faab_tier_chart',
    'create_horizontal_bar_chart',
    'get_grade_colors_list',
    'get_regret_colors_list',
    'get_faab_tier_colors_list',
    'GRADE_COLORS',
    'REGRET_COLORS',
    'FAAB_TIER_COLORS',
    'CATEGORICAL_COLORS',

    # DataFrame utilities
    'clean_dataframe',
    'ensure_numeric',
    'apply_common_renames',
]
