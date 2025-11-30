"""
Theme support for matchups tab.
Provides utilities for light/dark mode and theme-aware styling.
"""
import streamlit as st


def apply_theme_styles():
    """
    Apply enhanced theme-aware styles for matchups tab.
    Builds on top of modern_styles to ensure proper light/dark mode support.
    """
    st.markdown("""
    <style>
    /* Enhanced info boxes with theme support */
    .theme-info-box {
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }

    /* Light mode - blue info box */
    .theme-info-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
        border-left: 4px solid #2196F3;
        color: #1565C0;
    }

    /* Dark mode - adjusted blue info box */
    @media (prefers-color-scheme: dark) {
        .theme-info-box {
            background: linear-gradient(135deg, #1a2332 0%, #1e2a3a 100%);
            border-left: 4px solid #4d9eff;
            color: #90caf9;
        }
    }

    /* Loading indicator with theme support */
    .theme-loading {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Light mode loading */
    .theme-loading {
        background: linear-gradient(135deg, #fff9e6 0%, #fff3cc 100%);
        border-left: 3px solid #ffc107;
        color: #f57c00;
    }

    /* Dark mode loading */
    @media (prefers-color-scheme: dark) {
        .theme-loading {
            background: linear-gradient(135deg, #2d2416 0%, #3d2f1a 100%);
            border-left: 3px solid #ffb300;
            color: #ffcc80;
        }
    }

    /* Success message with theme support */
    .theme-success {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Light mode success */
    .theme-success {
        background: linear-gradient(135deg, #f1f8f4 0%, #e8f5e9 100%);
        border-left: 4px solid #4CAF50;
        color: #2e7d32;
    }

    /* Dark mode success */
    @media (prefers-color-scheme: dark) {
        .theme-success {
            background: linear-gradient(135deg, #1a2e1f 0%, #1e3326 100%);
            border-left: 4px solid #66bb6a;
            color: #a5d6a7;
        }
    }

    /* Warning/alert message with theme support */
    .theme-warning {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Light mode warning */
    .theme-warning {
        background: linear-gradient(135deg, #fff8e1 0%, #fff9c4 100%);
        border-left: 4px solid #ff9800;
        color: #e65100;
    }

    /* Dark mode warning */
    @media (prefers-color-scheme: dark) {
        .theme-warning {
            background: linear-gradient(135deg, #2d2516 0%, #3d3020 100%);
            border-left: 4px solid #ffb74d;
            color: #ffcc80;
        }
    }

    /* Error message with theme support */
    .theme-error {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Light mode error */
    .theme-error {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        color: #c62828;
    }

    /* Dark mode error */
    @media (prefers-color-scheme: dark) {
        .theme-error {
            background: linear-gradient(135deg, #2d1a1a 0%, #3d2020 100%);
            border-left: 4px solid #ef5350;
            color: #ef9a9a;
        }
    }

    /* Data freshness indicator */
    .theme-freshness {
        font-size: 0.85rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Light mode freshness */
    .theme-freshness {
        background: #f5f5f5;
        color: #666;
        border: 1px solid #e0e0e0;
    }

    /* Dark mode freshness */
    @media (prefers-color-scheme: dark) {
        .theme-freshness {
            background: #2b2d31;
            color: #b0b0b0;
            border: 1px solid #3a3c41;
        }
    }

    /* Stats count indicator */
    .theme-stats-count {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }

    /* Light mode stats count */
    .theme-stats-count {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1565C0;
        border: 1px solid #90caf9;
    }

    /* Dark mode stats count */
    @media (prefers-color-scheme: dark) {
        .theme-stats-count {
            background: linear-gradient(135deg, #1e2a3a 0%, #2a3a4a 100%);
            color: #90caf9;
            border: 1px solid #4d9eff;
        }
    }

    /* Improve table visibility in dark mode */
    @media (prefers-color-scheme: dark) {
        .stDataFrame {
            border: 1px solid #3a3c41;
        }

        /* Table headers */
        .stDataFrame thead th {
            background-color: #2b2d31 !important;
            color: #f0f0f0 !important;
        }

        /* Table cells */
        .stDataFrame tbody td {
            background-color: #1e1f22 !important;
            color: #e0e0e0 !important;
            border-color: #3a3c41 !important;
        }

        /* Alternating rows */
        .stDataFrame tbody tr:nth-child(even) td {
            background-color: #252629 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def render_info_box(message: str, icon: str = "💡") -> None:
    """
    Render a theme-aware info box.

    Args:
        message: Message to display
        icon: Emoji icon to show (default: 💡)
    """
    st.markdown(f"""
    <div class="theme-info-box">
    <p style="margin: 0; font-size: 0.9rem;">
    {icon} {message}
    </p>
    </div>
    """, unsafe_allow_html=True)


def render_success_box(message: str, icon: str = "✅") -> None:
    """
    Render a theme-aware success box.

    Args:
        message: Message to display
        icon: Emoji icon to show (default: ✅)
    """
    st.markdown(f"""
    <div class="theme-success">
    <p style="margin: 0; font-size: 0.9rem;">
    {icon} {message}
    </p>
    </div>
    """, unsafe_allow_html=True)


def render_warning_box(message: str, icon: str = "⚠️") -> None:
    """
    Render a theme-aware warning box.

    Args:
        message: Message to display
        icon: Emoji icon to show (default: ⚠️)
    """
    st.markdown(f"""
    <div class="theme-warning">
    <p style="margin: 0; font-size: 0.9rem;">
    {icon} {message}
    </p>
    </div>
    """, unsafe_allow_html=True)


def render_error_box(message: str, icon: str = "❌") -> None:
    """
    Render a theme-aware error box.

    Args:
        message: Message to display
        icon: Emoji icon to show (default: ❌)
    """
    st.markdown(f"""
    <div class="theme-error">
    <p style="margin: 0; font-size: 0.9rem;">
    {icon} {message}
    </p>
    </div>
    """, unsafe_allow_html=True)


def render_loading_indicator(message: str = "Loading...") -> None:
    """
    Render a theme-aware loading indicator.

    Args:
        message: Loading message to display
    """
    st.markdown(f"""
    <div class="theme-loading">
    🔄 {message}
    </div>
    """, unsafe_allow_html=True)


def render_stats_count(filtered_count: int, total_count: int) -> None:
    """
    Render a theme-aware statistics count display.

    Args:
        filtered_count: Number of filtered items
        total_count: Total number of items
    """
    percentage = (filtered_count / total_count * 100) if total_count > 0 else 0

    if filtered_count < total_count:
        message = f"📊 Showing {filtered_count:,} of {total_count:,} matchups ({percentage:.1f}%)"
    else:
        message = f"📊 Showing all {total_count:,} matchups"

    st.markdown(f"""
    <div class="theme-stats-count">
    {message}
    </div>
    """, unsafe_allow_html=True)
