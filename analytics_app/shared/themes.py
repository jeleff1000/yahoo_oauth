"""
Theme system for light/dark mode support across the application.

This module provides:
- Theme detection from Streamlit config
- CSS variable injection using design tokens
- Theme-aware utility functions

Usage:
    from shared.themes import inject_theme_css, detect_theme
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
    Uses CSS media queries to automatically respond to light/dark mode changes.
    """
    light = get_theme_tokens('light')
    dark = get_theme_tokens('dark')

    css = f"""
    <style>
    /* ===========================================
       CSS Variables from Design Tokens
       Light Mode (default)
       =========================================== */
    :root {{
        /* Backgrounds */
        --bg-primary: {light['bg_primary']};
        --bg-secondary: {light['bg_secondary']};
        --bg-tertiary: {light['bg_tertiary']};

        /* Text */
        --text-primary: {light['text_primary']};
        --text-secondary: {light['text_secondary']};
        --text-muted: {light['text_muted']};

        /* Borders */
        --border: {light['border']};
        --border-subtle: {light['border_subtle']};

        /* Accent */
        --accent: {light['accent']};
        --accent-hover: {light['accent_hover']};
        --accent-subtle: {light['accent_subtle']};
        --accent-light: {light['accent_light']};

        /* Gradients */
        --gradient-start: {light['gradient_start']};
        --gradient-end: {light['gradient_end']};

        /* Status */
        --success: {light['success']};
        --success-bg: {light['success_bg']};
        --warning: {light['warning']};
        --warning-bg: {light['warning_bg']};
        --error: {light['error']};
        --error-bg: {light['error_bg']};
        --info: {light['info']};
        --info-bg: {light['info_bg']};

        /* Interactive */
        --hover: {light['hover']};
        --active: {light['active']};
        --focus: {light['focus']};

        /* Charts */
        --chart-1: {light['chart_1']};
        --chart-2: {light['chart_2']};
        --chart-3: {light['chart_3']};
        --chart-4: {light['chart_4']};
        --chart-5: {light['chart_5']};

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
        --shadow-none: {SHADOWS['none']};
        --shadow-sm: {SHADOWS['sm']};
        --shadow-md: {SHADOWS['md']};
        --shadow-lg: {SHADOWS['lg']};
        --shadow-xl: {SHADOWS['xl']};

        /* Transitions */
        --transition-fast: {TRANSITIONS['fast']};
        --transition-normal: {TRANSITIONS['normal']};
        --transition-slow: {TRANSITIONS['slow']};
    }}

    /* ===========================================
       Dark Mode Variables
       Responds to system preference AND Streamlit's theme
       =========================================== */
    @media (prefers-color-scheme: dark) {{
        :root {{
            --bg-primary: {dark['bg_primary']};
            --bg-secondary: {dark['bg_secondary']};
            --bg-tertiary: {dark['bg_tertiary']};
            --text-primary: {dark['text_primary']};
            --text-secondary: {dark['text_secondary']};
            --text-muted: {dark['text_muted']};
            --border: {dark['border']};
            --border-subtle: {dark['border_subtle']};
            --accent: {dark['accent']};
            --accent-hover: {dark['accent_hover']};
            --accent-subtle: {dark['accent_subtle']};
            --accent-light: {dark['accent_light']};
            --gradient-start: {dark['gradient_start']};
            --gradient-end: {dark['gradient_end']};
            --success: {dark['success']};
            --success-bg: {dark['success_bg']};
            --warning: {dark['warning']};
            --warning-bg: {dark['warning_bg']};
            --error: {dark['error']};
            --error-bg: {dark['error_bg']};
            --info: {dark['info']};
            --info-bg: {dark['info_bg']};
            --hover: {dark['hover']};
            --active: {dark['active']};
            --focus: {dark['focus']};
            --chart-1: {dark['chart_1']};
            --chart-2: {dark['chart_2']};
            --chart-3: {dark['chart_3']};
            --chart-4: {dark['chart_4']};
            --chart-5: {dark['chart_5']};
            --shadow-sm: {SHADOWS_DARK['sm']};
            --shadow-md: {SHADOWS_DARK['md']};
            --shadow-lg: {SHADOWS_DARK['lg']};
            --shadow-xl: {SHADOWS_DARK['xl']};
        }}
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
