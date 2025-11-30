"""
Theme system for light/dark mode support across the application.

This module provides:
- Theme detection from Streamlit config
- Color palettes for light and dark modes
- Theme-aware gradients
- CSS injection for consistent styling
"""

import streamlit as st
from typing import Dict, Literal

ThemeType = Literal['light', 'dark']


def detect_theme() -> ThemeType:
    """
    Detect the current theme from Streamlit's theme configuration.

    Returns:
        'light' or 'dark' based on Streamlit's theme settings
    """
    # Try to detect from Streamlit's theme config
    try:
        # Check if theme.base is set
        theme_base = st.get_option("theme.base")
        if theme_base:
            return 'dark' if theme_base == 'dark' else 'light'
    except:
        pass

    # Check session state for manual override
    if 'theme' in st.session_state:
        return st.session_state['theme']

    # Default to light theme
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

    if theme == 'dark':
        return {
            # Backgrounds
            'bg': '#0E1117',
            'bg_secondary': '#262730',
            'bg_tertiary': '#1E1E1E',

            # Text
            'text': '#FAFAFA',
            'text_secondary': '#BDBDBD',
            'text_muted': '#808080',

            # Borders & dividers
            'border': '#404040',
            'border_light': '#505050',

            # Brand colors
            'primary': '#FF4B4B',
            'primary_light': '#FF6B6B',
            'primary_dark': '#CC3A3A',

            # Status colors
            'success': '#00D26A',
            'warning': '#FFB400',
            'error': '#FF4B4B',
            'info': '#0099FF',

            # Interactive elements
            'hover': '#2E2E2E',
            'active': '#3E3E3E',
            'focus': '#4E4E4E',

            # Data visualization
            'chart_1': '#667eea',
            'chart_2': '#764ba2',
            'chart_3': '#f093fb',
            'chart_4': '#4facfe',
        }
    else:  # light theme
        return {
            # Backgrounds
            'bg': '#FFFFFF',
            'bg_secondary': '#F0F2F6',
            'bg_tertiary': '#FAFAFA',

            # Text
            'text': '#262730',
            'text_secondary': '#555555',
            'text_muted': '#888888',

            # Borders & dividers
            'border': '#E0E0E0',
            'border_light': '#F0F0F0',

            # Brand colors
            'primary': '#FF4B4B',
            'primary_light': '#FF6B6B',
            'primary_dark': '#CC3A3A',

            # Status colors
            'success': '#00C07F',
            'warning': '#FFA500',
            'error': '#FF4B4B',
            'info': '#0066CC',

            # Interactive elements
            'hover': '#F5F5F5',
            'active': '#EBEBEB',
            'focus': '#E0E0E0',

            # Data visualization
            'chart_1': '#667eea',
            'chart_2': '#764ba2',
            'chart_3': '#f093fb',
            'chart_4': '#4facfe',
        }


def get_gradient(style: str = 'primary', theme: ThemeType = None) -> str:
    """
    Get a theme-aware CSS gradient.

    Args:
        style: Gradient style ('primary', 'secondary', 'success', 'purple', 'blue')
        theme: 'light' or 'dark'. If None, auto-detects current theme.

    Returns:
        CSS gradient string
    """
    colors = get_theme_colors(theme)

    gradients = {
        'primary': f"linear-gradient(135deg, {colors['primary']} 0%, {colors['primary_dark']} 100%)",
        'secondary': "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        'success': f"linear-gradient(135deg, {colors['success']} 0%, #00A854 100%)",
        'purple': "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        'blue': "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        'chart': f"linear-gradient(135deg, {colors['chart_1']} 0%, {colors['chart_2']} 100%)",
    }

    return gradients.get(style, gradients['primary'])


def inject_theme_css():
    """
    Inject theme-aware CSS variables into the page.

    This should be called once at the top of each page to enable theme support.
    All subsequent CSS can use the CSS variables defined here.

    Example:
        ```python
        from streamlit_ui.shared.themes import inject_theme_css

        def my_page():
            inject_theme_css()
            # ... rest of page code
        ```
    """
    colors = get_theme_colors()
    theme = detect_theme()

    # Build CSS with all theme variables
    css = f"""
    <style>
    /* ============================================
       THEME VARIABLES
       Auto-generated based on current theme: {theme}
       ============================================ */
    :root {{
        /* Background colors */
        --bg-color: {colors['bg']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-tertiary: {colors['bg_tertiary']};

        /* Text colors */
        --text-color: {colors['text']};
        --text-secondary: {colors['text_secondary']};
        --text-muted: {colors['text_muted']};

        /* Border colors */
        --border-color: {colors['border']};
        --border-light: {colors['border_light']};

        /* Brand colors */
        --primary-color: {colors['primary']};
        --primary-light: {colors['primary_light']};
        --primary-dark: {colors['primary_dark']};

        /* Status colors */
        --success-color: {colors['success']};
        --warning-color: {colors['warning']};
        --error-color: {colors['error']};
        --info-color: {colors['info']};

        /* Interactive states */
        --hover-color: {colors['hover']};
        --active-color: {colors['active']};
        --focus-color: {colors['focus']};

        /* Gradients */
        --gradient-primary: {get_gradient('primary', theme)};
        --gradient-secondary: {get_gradient('secondary', theme)};
        --gradient-success: {get_gradient('success', theme)};
        --gradient-purple: {get_gradient('purple', theme)};
        --gradient-blue: {get_gradient('blue', theme)};
        --gradient-chart: {get_gradient('chart', theme)};
    }}

    /* ============================================
       STREAMLIT DATAFRAME THEMING
       ============================================ */
    .dataframe {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}

    .dataframe th {{
        background: var(--gradient-primary) !important;
        color: white !important;
        font-weight: 600 !important;
    }}

    .dataframe td {{
        border-color: var(--border-color) !important;
    }}

    .dataframe tr:hover {{
        background-color: var(--hover-color) !important;
    }}

    .dataframe tr:nth-child(even) {{
        background-color: var(--bg-secondary) !important;
    }}

    /* ============================================
       OPTIMAL LINEUP TABLES
       ============================================ */
    table.optimal-visual,
    table.optimal {{
        background-color: var(--bg-color);
        color: var(--text-color);
        border-color: var(--border-color);
    }}

    table.optimal-visual th,
    table.optimal th {{
        background: var(--gradient-primary);
        color: white;
    }}

    table.optimal-visual td,
    table.optimal td {{
        border-color: var(--border-color);
    }}

    table.optimal-visual tr:nth-child(even),
    table.optimal tr:nth-child(even) {{
        background-color: var(--bg-secondary);
    }}

    table.optimal-visual tr:hover,
    table.optimal tr:hover {{
        background-color: var(--hover-color);
    }}

    .total-row {{
        background: var(--gradient-purple) !important;
        color: white !important;
    }}

    /* ============================================
       POSITION BADGES
       ============================================ */
    .pos-badge,
    .opt-pos-badge {{
        background: var(--gradient-chart);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85em;
    }}

    /* ============================================
       STATS HIGHLIGHTS
       ============================================ */
    .points-highlight,
    .opt-stat-cell,
    .metric-highlight {{
        color: {colors['success']};
        font-weight: bold;
    }}

    /* ============================================
       HERO SECTIONS
       ============================================ */
    .hero-section {{
        background: var(--gradient-primary);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }}

    /* ============================================
       CARDS & CONTAINERS
       ============================================ */
    .card {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
    }}

    .card:hover {{
        background-color: var(--hover-color);
        border-color: var(--primary-color);
    }}

    /* ============================================
       RESPONSIVE MOBILE ADJUSTMENTS
       ============================================ */
    @media (max-width: 768px) {{
        table.optimal-visual,
        table.optimal {{
            font-size: 0.85em;
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }}

        .opt-player-img {{
            width: 35px !important;
            height: 35px !important;
        }}

        .hide-mobile {{
            display: none;
        }}

        .hero-section {{
            padding: 1rem;
        }}
    }}

    /* ============================================
       DESKTOP OPTIMIZATIONS
       ============================================ */
    @media (min-width: 769px) {{
        table.optimal-visual {{
            max-width: 1400px;
            margin: 0 auto;
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
            "🎨 Theme",
            options=["light", "dark"],
            index=0 if detect_theme() == 'light' else 1,
            horizontal=True,
            key='theme_toggle'
        )
        st.session_state['theme'] = theme
        return theme
