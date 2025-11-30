"""
Theme system for light/dark mode support across the application.

This module provides:
- Theme detection from Streamlit config
- CSS variable injection using design tokens
- Theme-aware utility functions

Usage:
    from streamlit_ui.shared.themes import inject_theme_css, detect_theme
"""

import streamlit as st
from typing import Dict, Literal

from .design_tokens import (
    COLORS,
    SPACING,
    RADIUS,
    SHADOWS,
    SHADOWS_DARK,
    TYPOGRAPHY,
    TRANSITIONS,
    get_theme_tokens,
    get_css_variables,
    ThemeType,
)

# Re-export ThemeType for backwards compatibility
__all__ = ['detect_theme', 'get_theme_colors', 'inject_theme_css', 'ThemeType']


def detect_theme() -> ThemeType:
    """
    Detect the current theme from Streamlit's theme configuration.

    Returns:
        'light' or 'dark' based on Streamlit's theme settings
    """
    try:
        theme_base = st.get_option("theme.base")
        if theme_base:
            return 'dark' if theme_base == 'dark' else 'light'
    except:
        pass

    if 'theme' in st.session_state:
        return st.session_state['theme']

    return 'light'


def get_theme_colors(theme: ThemeType = None) -> Dict[str, str]:
    """
    Get color palette for the specified theme.

    Args:
        theme: 'light' or 'dark'. If None, auto-detects current theme.

    Returns:
        Dictionary of color values for the theme
    """
    if theme is None:
        theme = detect_theme()
    return get_theme_tokens(theme)


def inject_theme_css():
    """
    Inject theme-aware CSS variables into the page.

    This should be called once at the top of each page to enable theme support.
    Uses the design tokens for consistent styling.
    """
    theme = detect_theme()
    colors = get_theme_tokens(theme)
    shadows = SHADOWS_DARK if theme == 'dark' else SHADOWS

    css = f"""
    <style>
    /* ===========================================
       CSS Variables from Design Tokens
       Theme: {theme}
       =========================================== */
    :root {{
        /* Backgrounds */
        --bg-primary: {colors['bg_primary']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-tertiary: {colors['bg_tertiary']};

        /* Text */
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --text-muted: {colors['text_muted']};

        /* Borders */
        --border: {colors['border']};
        --border-subtle: {colors['border_subtle']};

        /* Accent */
        --accent: {colors['accent']};
        --accent-hover: {colors['accent_hover']};
        --accent-subtle: {colors['accent_subtle']};
        --accent-light: {colors['accent_light']};

        /* Gradients */
        --gradient-start: {colors['gradient_start']};
        --gradient-end: {colors['gradient_end']};

        /* Status */
        --success: {colors['success']};
        --success-bg: {colors['success_bg']};
        --warning: {colors['warning']};
        --warning-bg: {colors['warning_bg']};
        --error: {colors['error']};
        --error-bg: {colors['error_bg']};
        --info: {colors['info']};
        --info-bg: {colors['info_bg']};

        /* Interactive */
        --hover: {colors['hover']};
        --active: {colors['active']};
        --focus: {colors['focus']};

        /* Charts */
        --chart-1: {colors['chart_1']};
        --chart-2: {colors['chart_2']};
        --chart-3: {colors['chart_3']};
        --chart-4: {colors['chart_4']};
        --chart-5: {colors['chart_5']};

        /* Spacing */
        --space-xs: {SPACING['xs']};
        --space-sm: {SPACING['sm']};
        --space-md: {SPACING['md']};
        --space-lg: {SPACING['lg']};
        --space-xl: {SPACING['xl']};
        --space-xxl: {SPACING['xxl']};

        /* Radius */
        --radius-sm: {RADIUS['sm']};
        --radius-md: {RADIUS['md']};
        --radius-lg: {RADIUS['lg']};
        --radius-full: {RADIUS['full']};

        /* Shadows */
        --shadow-none: {shadows['none']};
        --shadow-sm: {shadows['sm']};
        --shadow-md: {shadows['md']};
        --shadow-lg: {shadows['lg']};
        --shadow-xl: {shadows['xl']};

        /* Transitions */
        --transition-fast: {TRANSITIONS['fast']};
        --transition-normal: {TRANSITIONS['normal']};
        --transition-slow: {TRANSITIONS['slow']};
    }}

    /* ===========================================
       BASE ELEMENT THEMING
       =========================================== */

    /* Streamlit dataframes */
    .stDataFrame {{
        border-radius: var(--radius-md);
    }}

    .stDataFrame table {{
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }}

    .stDataFrame th {{
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        font-weight: 600;
        border-bottom: 2px solid var(--accent);
    }}

    .stDataFrame td {{
        border-color: var(--border-subtle);
    }}

    .stDataFrame tr:hover {{
        background-color: var(--hover);
    }}

    /* Position badges */
    .pos-badge {{
        background: linear-gradient(135deg, var(--chart-1), var(--chart-2));
        color: white;
        padding: 4px 10px;
        border-radius: var(--radius-sm);
        font-weight: 600;
        font-size: 0.85em;
    }}

    /* Stats highlights */
    .points-highlight,
    .metric-highlight {{
        color: var(--success);
        font-weight: 600;
    }}

    /* ===========================================
       RESPONSIVE ADJUSTMENTS
       =========================================== */
    @media (max-width: 768px) {{
        .stDataFrame table {{
            font-size: 0.85em;
        }}
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def create_theme_toggle():
    """
    Create a theme toggle widget in the sidebar.

    Returns:
        The selected theme ('light' or 'dark')
    """
    with st.sidebar:
        st.markdown("---")
        theme = st.radio(
            "Theme",
            options=["light", "dark"],
            index=0 if detect_theme() == 'light' else 1,
            horizontal=True,
            key='theme_toggle'
        )
        st.session_state['theme'] = theme
        return theme
