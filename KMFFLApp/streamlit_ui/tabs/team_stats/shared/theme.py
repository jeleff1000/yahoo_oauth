"""
Theming system and styled UI components for team stats.

This module re-exports all styling utilities from the centralized modern_styles module.
Import from here for backwards compatibility, but all styles are now centralized.

Provides:
- Light/dark mode support via CSS variables
- Styled info/success/warning/error boxes
- Gradient headers and section cards
- Metric cards for displaying statistics
- Theme-aware table styling
- Loading states and empty states
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

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
    Apply comprehensive theme styles including light/dark mode support.

    NOTE: This function is now a no-op since all styles are included
    in apply_modern_styles(). Kept for backwards compatibility.
    """
    # All styles are now centralized in apply_modern_styles()
    # This function is kept for backwards compatibility with existing code
    pass


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS SPECIFIC TO TEAM STATS
# ============================================================================

def render_metric_grid(metrics: List[Dict[str, Any]], columns: int = 4):
    """
    Render a grid of metric cards.

    Args:
        metrics: List of dicts with 'label', 'value', 'delta' (optional)
        columns: Number of columns in grid
    """
    cols = st.columns(columns)
    for idx, metric in enumerate(metrics):
        with cols[idx % columns]:
            render_metric_card(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta'),
                delta_label=metric.get('delta_label')
            )


def style_dataframe(
    df: pd.DataFrame,
    highlight_columns: Optional[List[str]] = None,
    format_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Apply styling to a dataframe.

    Args:
        df: Input dataframe
        highlight_columns: Columns to highlight
        format_dict: Dictionary of column: format_string

    Returns:
        Styled dataframe
    """
    # Apply formatting if specified
    if format_dict:
        for col, fmt in format_dict.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '-')

    return df


def get_trend_indicator(current: float, previous: float) -> Tuple[str, str]:
    """
    Get trend indicator (arrow and color) based on current vs previous value.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Tuple of (arrow_emoji, color_class)
    """
    if current > previous:
        return ("↑", "positive")
    elif current < previous:
        return ("↓", "negative")
    else:
        return ("→", "neutral")


def format_value_with_trend(
    value: float,
    previous_value: Optional[float] = None,
    format_str: str = "{:.1f}"
) -> str:
    """
    Format a value with optional trend indicator.

    Args:
        value: Current value
        previous_value: Previous value for comparison
        format_str: Format string

    Returns:
        Formatted string with trend indicator
    """
    formatted = format_str.format(value)

    if previous_value is not None:
        arrow, _ = get_trend_indicator(value, previous_value)
        diff = value - previous_value
        return f"{formatted} {arrow} ({diff:+.1f})"

    return formatted
