"""
Theme support for matchups tab.

This module re-exports all styling utilities from the centralized modern_styles module.
Import from here for backwards compatibility, but all styles are now centralized.
"""

# Re-export from the centralized styles module
from ...shared.modern_styles import render_info_box, render_legend_box  # noqa: F401


def apply_theme_styles():
    """
    Apply enhanced theme-aware styles for matchups tab.

    NOTE: This function is now a no-op since all styles are included
    in apply_modern_styles(). Kept for backwards compatibility.
    """
    # All styles are now centralized in apply_modern_styles()
    # This function is kept for backwards compatibility with existing code
    pass
