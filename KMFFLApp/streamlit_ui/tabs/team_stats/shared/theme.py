"""
Theming system and styled UI components for team stats.

Provides:
- Light/dark mode support via CSS media queries
- Styled info/success/warning/error boxes
- Gradient headers and section cards
- Metric cards for displaying statistics
- Theme-aware table styling
- Loading states and empty states
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd


def apply_theme_styles():
    """
    Apply comprehensive theme styles including light/dark mode support.

    This function injects CSS that:
    - Provides themed message boxes (info, success, warning, error)
    - Styles tables for both light and dark modes
    - Adds gradient headers and section cards
    - Ensures proper contrast and readability
    """
    st.markdown("""
    <style>
    /* ============================================================
       THEME-AWARE MESSAGE BOXES
       ============================================================ */

    /* Info Box - Blue */
    .theme-info-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
        border-left: 4px solid #2196F3;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1565C0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    @media (prefers-color-scheme: dark) {
        .theme-info-box {
            background: linear-gradient(135deg, #1a2332 0%, #1e2a3a 100%);
            border-left: 4px solid #4d9eff;
            color: #90caf9;
        }
    }

    /* Success Box - Green */
    .theme-success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #22c55e;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #166534;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    @media (prefers-color-scheme: dark) {
        .theme-success-box {
            background: linear-gradient(135deg, #1a2e23 0%, #1e3329 100%);
            border-left: 4px solid #4ade80;
            color: #86efac;
        }
    }

    /* Warning Box - Orange */
    .theme-warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #92400e;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    @media (prefers-color-scheme: dark) {
        .theme-warning-box {
            background: linear-gradient(135deg, #2e2416 0%, #332815 100%);
            border-left: 4px solid #fbbf24;
            color: #fde68a;
        }
    }

    /* Error Box - Red */
    .theme-error-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #991b1b;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    @media (prefers-color-scheme: dark) {
        .theme-error-box {
            background: linear-gradient(135deg, #2e1a1a 0%, #331e1e 100%);
            border-left: 4px solid #f87171;
            color: #fca5a5;
        }
    }

    /* ============================================================
       GRADIENT HEADERS
       ============================================================ */

    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .gradient-header h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .gradient-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1rem;
    }

    @media (max-width: 768px) {
        .gradient-header {
            padding: 1rem 1.5rem;
        }
        .gradient-header h2 {
            font-size: 1.5rem;
        }
        .gradient-header p {
            font-size: 0.9rem;
        }
    }

    /* ============================================================
       SECTION CARDS
       ============================================================ */

    .section-card {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    @media (prefers-color-scheme: dark) {
        .section-card {
            background: #2b2d31;
            border: 2px solid #3a3c41;
        }
    }

    /* ============================================================
       METRIC CARDS
       ============================================================ */

    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f9fafb 100%);
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: linear-gradient(145deg, #2b2d31 0%, #1e1f22 100%);
            border: 1px solid #3a3c41;
            color: #f0f0f0;
        }
    }

    .metric-card-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    @media (prefers-color-scheme: dark) {
        .metric-card-label {
            color: #9ca3af;
        }
    }

    .metric-card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.25rem 0;
    }

    @media (prefers-color-scheme: dark) {
        .metric-card-value {
            color: #f9fafb;
        }
    }

    .metric-card-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .metric-card-delta.positive {
        color: #10b981;
    }

    .metric-card-delta.negative {
        color: #ef4444;
    }

    /* ============================================================
       TABLE STYLING
       ============================================================ */

    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .stDataFrame table {
        font-size: 0.9rem;
    }

    .stDataFrame thead tr th {
        background-color: #f3f4f6 !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        border-bottom: 2px solid #d1d5db !important;
    }

    @media (prefers-color-scheme: dark) {
        .stDataFrame thead tr th {
            background-color: #374151 !important;
            color: #f9fafb !important;
            border-bottom: 2px solid #4b5563 !important;
        }
    }

    .stDataFrame tbody tr {
        background-color: #ffffff !important;
        transition: background-color 0.2s;
    }

    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f9fafb !important;
    }

    .stDataFrame tbody tr:hover {
        background-color: #f3f4f6 !important;
    }

    @media (prefers-color-scheme: dark) {
        .stDataFrame tbody tr {
            background-color: #1e1f22 !important;
            color: #e0e0e0 !important;
        }
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #252629 !important;
        }
        .stDataFrame tbody tr:hover {
            background-color: #2b2d31 !important;
        }
    }

    /* ============================================================
       EMPTY STATE
       ============================================================ */

    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: #f9fafb;
        border: 2px dashed #d1d5db;
        border-radius: 0.75rem;
        margin: 2rem 0;
    }

    @media (prefers-color-scheme: dark) {
        .empty-state {
            background: #1e1f22;
            border-color: #3a3c41;
        }
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    .empty-state-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }

    @media (prefers-color-scheme: dark) {
        .empty-state-title {
            color: #d1d5db;
        }
    }

    .empty-state-message {
        font-size: 0.95rem;
        color: #6b7280;
    }

    @media (prefers-color-scheme: dark) {
        .empty-state-message {
            color: #9ca3af;
        }
    }

    /* ============================================================
       FILTER COUNT BADGE
       ============================================================ */

    .filter-count {
        display: inline-block;
        background: #3b82f6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    @media (prefers-color-scheme: dark) {
        .filter-count {
            background: #60a5fa;
        }
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# STYLED COMPONENT FUNCTIONS
# ============================================================================

def render_info_box(message: str, icon: str = "💡"):
    """Render a themed info box."""
    st.markdown(f"""
    <div class="theme-info-box">
        <strong>{icon} Info:</strong> {message}
    </div>
    """, unsafe_allow_html=True)


def render_success_box(message: str, icon: str = "✅"):
    """Render a themed success box."""
    st.markdown(f"""
    <div class="theme-success-box">
        <strong>{icon} Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)


def render_warning_box(message: str, icon: str = "⚠️"):
    """Render a themed warning box."""
    st.markdown(f"""
    <div class="theme-warning-box">
        <strong>{icon} Warning:</strong> {message}
    </div>
    """, unsafe_allow_html=True)


def render_error_box(message: str, icon: str = "❌"):
    """Render a themed error box."""
    st.markdown(f"""
    <div class="theme-error-box">
        <strong>{icon} Error:</strong> {message}
    </div>
    """, unsafe_allow_html=True)


def render_gradient_header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None):
    """
    Render a gradient header with optional subtitle and icon.

    Args:
        title: Main header text
        subtitle: Optional subtitle text
        icon: Optional emoji icon
    """
    icon_html = f"{icon} " if icon else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""

    st.markdown(f"""
    <div class="gradient-header">
        <h2>{icon_html}{title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_card(content: str):
    """Render a section card with themed background."""
    st.markdown(f"""
    <div class="section-card">
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: Any,
    delta: Optional[float] = None,
    delta_label: Optional[str] = None
):
    """
    Render a styled metric card.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional change value
        delta_label: Optional description of delta
    """
    delta_class = "positive" if delta and delta > 0 else "negative" if delta and delta < 0 else ""
    delta_icon = "↑" if delta and delta > 0 else "↓" if delta and delta < 0 else ""
    delta_text = delta_label if delta_label else f"{delta:+.1f}" if delta else ""

    delta_html = f"""
    <div class="metric-card-delta {delta_class}">
        {delta_icon} {delta_text}
    </div>
    """ if delta is not None else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


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


def render_empty_state(
    title: str = "No Data Available",
    message: str = "Try adjusting your filters or selection.",
    icon: str = "📭"
):
    """Render a styled empty state."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-message">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_filter_count(filtered_count: int, total_count: int):
    """Render a badge showing filter results count."""
    percentage = (filtered_count / total_count * 100) if total_count > 0 else 0

    st.markdown(f"""
    Showing **{filtered_count:,}** of **{total_count:,}** records
    <span class="filter-count">{percentage:.0f}%</span>
    """, unsafe_allow_html=True)


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
