"""
Theme support for matchups tab.

This module re-exports all styling utilities from the centralized modern_styles module.
Import from here for backwards compatibility, but all styles are now centralized.
"""

# Re-export everything from the centralized styles module
from ...shared.modern_styles import (
    apply_modern_styles,
    render_info_box,
    render_success_box,
    render_warning_box,
    render_error_box,
    render_loading_indicator,
    render_stats_count,
    render_gradient_header,
    render_section_card,
    render_metric_card,
    render_empty_state,
    render_filter_count,
    render_legend_box,
    format_value_with_color,
)


def apply_theme_styles():
    """
    Apply enhanced theme-aware styles for matchups tab.

    NOTE: This function is now a no-op since all styles are included
    in apply_modern_styles(). Kept for backwards compatibility.
    """
    # All styles are now centralized in apply_modern_styles()
    # This function is kept for backwards compatibility with existing code
    pass
