"""
Theming system and styled UI components for team stats.

Provides:
- Light/dark mode support via CSS variables
- Styled info/success/warning/error boxes
- Gradient headers and section cards
- Metric cards for displaying statistics
- Theme-aware table styling
- Loading states and empty states

Updated: Force cache clear
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

# Try to import from centralized module, fallback to local definition
try:
    from tabs.shared.modern_styles import render_metric_card
except ImportError:
    def render_metric_card(label, value, delta=None, delta_label=None):
        """Fallback metric card renderer."""
        st.metric(label=label, value=value, delta=delta)


def render_empty_state(
    title: str = "No Data Available",
    message: str = "Try adjusting your filters or selection.",
    icon: str = "",
) -> None:
    """Render a styled empty state."""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            color: var(--text-muted, #9CA3AF);
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.25rem;">{title}</div>
            <div style="font-size: 0.9rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta"),
                delta_label=metric.get("delta_label"),
            )


def style_dataframe(
    df: pd.DataFrame,
    highlight_columns: Optional[List[str]] = None,
    format_dict: Optional[Dict[str, str]] = None,
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
                df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "-")

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
    value: float, previous_value: Optional[float] = None, format_str: str = "{:.1f}"
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
