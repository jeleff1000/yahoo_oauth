"""
Design Tokens - Single source of truth for all UI styling.

This module provides standardized values for colors, spacing, typography,
and other design elements used throughout the KMFFL application.

Usage:
    from streamlit_ui.shared.design_tokens import COLORS, SPACING, get_theme_tokens
"""

from typing import Dict, Literal

ThemeType = Literal['light', 'dark']


# =============================================================================
# COLOR PALETTES
# =============================================================================

COLORS = {
    'light': {
        # Backgrounds
        'bg_primary': '#FFFFFF',
        'bg_secondary': '#F8F9FA',
        'bg_tertiary': '#F0F2F6',

        # Text
        'text_primary': '#1F2937',
        'text_secondary': '#6B7280',
        'text_muted': '#9CA3AF',

        # Borders
        'border': '#E5E7EB',
        'border_subtle': '#F0F0F0',

        # Accent (purple theme)
        'accent': '#667eea',
        'accent_hover': '#5a6fd6',
        'accent_subtle': 'rgba(102, 126, 234, 0.08)',
        'accent_light': 'rgba(102, 126, 234, 0.15)',

        # Gradients (subtle)
        'gradient_start': 'rgba(102, 126, 234, 0.08)',
        'gradient_end': 'rgba(118, 75, 162, 0.05)',

        # Status colors
        'success': '#10B981',
        'success_bg': 'rgba(16, 185, 129, 0.1)',
        'warning': '#F59E0B',
        'warning_bg': 'rgba(245, 158, 11, 0.1)',
        'error': '#EF4444',
        'error_bg': 'rgba(239, 68, 68, 0.1)',
        'info': '#3B82F6',
        'info_bg': 'rgba(59, 130, 246, 0.1)',

        # Interactive states
        'hover': '#F5F5F5',
        'active': '#EBEBEB',
        'focus': '#E0E0E0',

        # Chart colors
        'chart_1': '#667eea',
        'chart_2': '#764ba2',
        'chart_3': '#f093fb',
        'chart_4': '#4facfe',
        'chart_5': '#00f2fe',
    },
    'dark': {
        # Backgrounds
        'bg_primary': '#0E1117',
        'bg_secondary': '#1F2937',
        'bg_tertiary': '#374151',

        # Text
        'text_primary': '#F9FAFB',
        'text_secondary': '#D1D5DB',
        'text_muted': '#9CA3AF',

        # Borders
        'border': '#374151',
        'border_subtle': '#2D3748',

        # Accent (slightly lighter for dark mode)
        'accent': '#818CF8',
        'accent_hover': '#6366F1',
        'accent_subtle': 'rgba(129, 140, 248, 0.12)',
        'accent_light': 'rgba(129, 140, 248, 0.2)',

        # Gradients (subtle)
        'gradient_start': 'rgba(102, 126, 234, 0.15)',
        'gradient_end': 'rgba(118, 75, 162, 0.1)',

        # Status colors
        'success': '#10B981',
        'success_bg': 'rgba(16, 185, 129, 0.15)',
        'warning': '#F59E0B',
        'warning_bg': 'rgba(245, 158, 11, 0.15)',
        'error': '#EF4444',
        'error_bg': 'rgba(239, 68, 68, 0.15)',
        'info': '#3B82F6',
        'info_bg': 'rgba(59, 130, 246, 0.15)',

        # Interactive states
        'hover': '#2D3748',
        'active': '#374151',
        'focus': '#4A5568',

        # Chart colors
        'chart_1': '#818CF8',
        'chart_2': '#A78BFA',
        'chart_3': '#F472B6',
        'chart_4': '#60A5FA',
        'chart_5': '#34D399',
    }
}


# =============================================================================
# SPACING SCALE - REDUCED FOR TIGHTER LAYOUT
# =============================================================================

SPACING = {
    'xs': '0.25rem',    # 4px
    'sm': '0.375rem',   # 6px (reduced from 8px)
    'md': '0.625rem',   # 10px (reduced from 16px)
    'lg': '1rem',       # 16px (reduced from 24px)
    'xl': '1.25rem',    # 20px (reduced from 32px)
    'xxl': '1.5rem',    # 24px (reduced from 48px)
}


# =============================================================================
# BORDER RADIUS
# =============================================================================

RADIUS = {
    'sm': '4px',
    'md': '8px',
    'lg': '12px',
    'full': '9999px',
}


# =============================================================================
# SHADOWS
# =============================================================================

SHADOWS = {
    'none': 'none',
    'sm': '0 1px 2px rgba(0, 0, 0, 0.05)',
    'md': '0 2px 4px rgba(0, 0, 0, 0.08)',
    'lg': '0 4px 8px rgba(0, 0, 0, 0.1)',
    'xl': '0 8px 16px rgba(0, 0, 0, 0.12)',
}

# For dark mode, slightly more visible shadows
SHADOWS_DARK = {
    'none': 'none',
    'sm': '0 1px 2px rgba(0, 0, 0, 0.2)',
    'md': '0 2px 4px rgba(0, 0, 0, 0.25)',
    'lg': '0 4px 8px rgba(0, 0, 0, 0.3)',
    'xl': '0 8px 16px rgba(0, 0, 0, 0.35)',
}


# =============================================================================
# TYPOGRAPHY
# =============================================================================

TYPOGRAPHY = {
    'h1': {'size': '1.75rem', 'weight': '600', 'line_height': '1.2'},
    'h2': {'size': '1.5rem', 'weight': '600', 'line_height': '1.3'},
    'h3': {'size': '1.25rem', 'weight': '500', 'line_height': '1.4'},
    'h4': {'size': '1.1rem', 'weight': '500', 'line_height': '1.4'},
    'body': {'size': '1rem', 'weight': '400', 'line_height': '1.5'},
    'small': {'size': '0.875rem', 'weight': '400', 'line_height': '1.5'},
    'caption': {'size': '0.75rem', 'weight': '400', 'line_height': '1.4'},
}


# =============================================================================
# BREAKPOINTS
# =============================================================================

BREAKPOINTS = {
    'mobile': '480px',
    'tablet': '768px',
    'desktop': '1024px',
    'wide': '1280px',
}


# =============================================================================
# TRANSITIONS
# =============================================================================

TRANSITIONS = {
    'fast': '0.1s ease',
    'normal': '0.2s ease',
    'slow': '0.3s ease',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_theme_tokens(theme: ThemeType) -> Dict[str, str]:
    """
    Get all design tokens for a specific theme.

    Args:
        theme: 'light' or 'dark'

    Returns:
        Dictionary with all color tokens for the theme
    """
    return COLORS.get(theme, COLORS['light'])


def get_shadows(theme: ThemeType) -> Dict[str, str]:
    """
    Get shadow tokens appropriate for the theme.

    Args:
        theme: 'light' or 'dark'

    Returns:
        Dictionary with shadow values
    """
    return SHADOWS_DARK if theme == 'dark' else SHADOWS


def get_css_variables(theme: ThemeType) -> str:
    """
    Generate CSS custom properties (variables) for a theme.

    Args:
        theme: 'light' or 'dark'

    Returns:
        CSS string with all variables defined in :root
    """
    colors = get_theme_tokens(theme)
    shadows = get_shadows(theme)

    css_vars = []

    # Color variables
    for key, value in colors.items():
        css_vars.append(f"--{key.replace('_', '-')}: {value};")

    # Spacing variables
    for key, value in SPACING.items():
        css_vars.append(f"--space-{key}: {value};")

    # Radius variables
    for key, value in RADIUS.items():
        css_vars.append(f"--radius-{key}: {value};")

    # Shadow variables
    for key, value in shadows.items():
        css_vars.append(f"--shadow-{key}: {value};")

    # Transition variables
    for key, value in TRANSITIONS.items():
        css_vars.append(f"--transition-{key}: {value};")

    return ":root {\n    " + "\n    ".join(css_vars) + "\n}"
