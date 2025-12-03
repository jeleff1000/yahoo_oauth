"""
Reusable UI Components for the KMFFL application.

This module provides standardized, theme-aware components that maintain
visual consistency across the application.

Key principles:
- Static components (info displays) have NO shadows or hover effects
- Interactive components (buttons, clickable items) have shadows and hover
- All components use CSS variables from design tokens

Usage:
    from streamlit_ui.shared.components import hero_section, static_card, section_header
"""

import streamlit as st
from typing import Optional, List, Dict, Any


def hero_section(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None
) -> None:
    """
    Render a hero section with subtle gradient background.

    Args:
        title: Main heading text
        subtitle: Optional description text
        icon: Optional emoji/icon to display before title
    """
    icon_html = f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ''

    html = f'''
    <div class="hero-section">
        <h2 style="margin: 0; display: flex; align-items: center;">
            {icon_html}{title}
        </h2>
        {f'<p style="margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, icon: Optional[str] = None) -> None:
    """
    Render a section header with accent underline.

    Args:
        title: Section heading text
        icon: Optional emoji/icon
    """
    icon_html = f'{icon} ' if icon else ''
    st.markdown(
        f'<div class="section-header-title">{icon_html}{title}</div>',
        unsafe_allow_html=True
    )


def static_card(
    content: str,
    title: Optional[str] = None,
    icon: Optional[str] = None
) -> None:
    """
    Render a static info card - NO hover effects, NOT clickable.

    Use for displaying information, stats, or data that users read but don't interact with.

    Args:
        content: Main content/body text
        title: Optional card title
        icon: Optional emoji/icon
    """
    title_html = ''
    if title:
        icon_html = f'{icon} ' if icon else ''
        title_html = f'<div class="feature-title">{icon_html}{title}</div>'

    html = f'''
    <div class="static-card">
        {title_html}
        <div class="feature-desc">{content}</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def feature_card(
    title: str,
    description: str,
    icon: Optional[str] = None
) -> None:
    """
    Render a feature/info card - static, no hover.

    Args:
        title: Feature title
        description: Feature description
        icon: Optional emoji/icon
    """
    icon_html = f'<div class="feature-icon">{icon}</div>' if icon else ''

    html = f'''
    <div class="feature-card">
        {icon_html}
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def stat_display(
    value: Any,
    label: str,
    trend: Optional[str] = None,
    trend_positive: bool = True
) -> None:
    """
    Render a clean stat/metric display.

    Args:
        value: The stat value to display
        label: Label describing the stat
        trend: Optional trend indicator (e.g., "+5%")
        trend_positive: Whether trend is positive (green) or negative (red)
    """
    trend_html = ''
    if trend:
        color = 'var(--success)' if trend_positive else 'var(--error)'
        trend_html = f'<span style="color: {color}; font-size: 0.875rem; margin-left: 0.5rem;">{trend}</span>'

    html = f'''
    <div class="stat-display">
        <div class="stat-value">{value}{trend_html}</div>
        <div class="stat-label">{label}</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def info_box(content: str, title: Optional[str] = None) -> None:
    """
    Render an info message box with blue theme.

    Args:
        content: Message content
        title: Optional title
    """
    title_html = f'<strong>{title}</strong><br>' if title else ''
    st.markdown(
        f'<div class="info-box">{title_html}{content}</div>',
        unsafe_allow_html=True
    )


def success_box(content: str, title: Optional[str] = None) -> None:
    """
    Render a success message box with green theme.

    Args:
        content: Message content
        title: Optional title
    """
    title_html = f'<strong>{title}</strong><br>' if title else ''
    st.markdown(
        f'<div class="success-box">{title_html}{content}</div>',
        unsafe_allow_html=True
    )


def warning_box(content: str, title: Optional[str] = None) -> None:
    """
    Render a warning message box with orange theme.

    Args:
        content: Message content
        title: Optional title
    """
    title_html = f'<strong>{title}</strong><br>' if title else ''
    st.markdown(
        f'<div class="warning-box">{title_html}{content}</div>',
        unsafe_allow_html=True
    )


def error_box(content: str, title: Optional[str] = None) -> None:
    """
    Render an error message box with red theme.

    Args:
        content: Message content
        title: Optional title
    """
    title_html = f'<strong>{title}</strong><br>' if title else ''
    st.markdown(
        f'<div class="error-box">{title_html}{content}</div>',
        unsafe_allow_html=True
    )


def interactive_card(
    title: str,
    description: str,
    icon: Optional[str] = None,
    key: Optional[str] = None
) -> bool:
    """
    Render an interactive/clickable card with hover effects.

    Use for navigation or actions where users click to do something.

    Args:
        title: Card title
        description: Card description
        icon: Optional emoji/icon
        key: Unique key for the button

    Returns:
        True if card was clicked
    """
    icon_html = f'<div class="feature-icon">{icon}</div>' if icon else ''

    # Use columns to create a full-width clickable area
    col = st.container()
    with col:
        st.markdown(
            f'''
            <div class="interactive-card">
                {icon_html}
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{description}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        # Hidden button for click handling
        return st.button(f"Select {title}", key=key, type="secondary", use_container_width=True)


def stat_grid(stats: List[Dict[str, Any]], columns: int = 4) -> None:
    """
    Render a grid of stats in a clean layout.

    Args:
        stats: List of dicts with 'value', 'label', and optional 'trend', 'trend_positive'
        columns: Number of columns (2, 3, or 4)
    """
    cols = st.columns(columns)
    for i, stat in enumerate(stats):
        with cols[i % columns]:
            stat_display(
                value=stat.get('value', ''),
                label=stat.get('label', ''),
                trend=stat.get('trend'),
                trend_positive=stat.get('trend_positive', True)
            )


def data_table_css() -> str:
    """
    Return CSS for styling data tables consistently.

    Returns:
        CSS string to be injected with st.markdown
    """
    return '''
    <style>
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .data-table th {
        background: var(--bg-secondary);
        color: var(--text-primary);
        font-weight: 600;
        padding: 0.75rem;
        text-align: left;
        border-bottom: 2px solid var(--accent);
    }
    .data-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .data-table tr:hover {
        background: var(--hover);
    }
    .data-table tr:nth-child(even) {
        background: var(--bg-secondary);
    }
    .data-table tr:nth-child(even):hover {
        background: var(--hover);
    }

    @media (max-width: 768px) {
        .data-table {
            display: block;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .data-table th,
        .data-table td {
            padding: 0.4rem;
            font-size: 0.8rem;
            white-space: nowrap;
        }
    }
    </style>
    '''


def position_badge(position: str) -> str:
    """
    Return HTML for a position badge.

    Args:
        position: Position abbreviation (QB, RB, WR, etc.)

    Returns:
        HTML string for the badge
    """
    return f'<span class="pos-badge">{position}</span>'


def apply_all_styles() -> None:
    """
    Apply all theme CSS and component styles.

    Call this once at the start of each page.
    """
    from streamlit_ui.tabs.shared.modern_styles import apply_modern_styles

    # apply_modern_styles now includes all CSS variables and theme support
    apply_modern_styles()
