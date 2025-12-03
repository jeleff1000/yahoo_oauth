"""
Plotly chart theming for light/dark mode support.

Provides theme-aware chart configurations that automatically adapt to the current theme.
Uses the existing theme detection from themes.py.

Usage:
    from streamlit_ui.shared.chart_themes import get_chart_theme, apply_chart_theme

    # Option 1: Get theme dict and pass to Plotly
    fig = px.bar(data, template=get_chart_theme())

    # Option 2: Apply theme to existing figure
    fig = px.bar(data)
    fig = apply_chart_theme(fig)

    # Option 3: Get specific color palettes
    colors = get_chart_colors()
    fig = px.bar(data, color_discrete_sequence=colors['categorical'])
"""

import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, List, Any, Optional

# Import theme detection and design tokens
try:
    from .themes import detect_theme, get_theme_colors
    from .design_tokens import COLORS
except ImportError:
    # Fallback if running standalone
    def detect_theme():
        return 'dark'
    def get_theme_colors(theme=None):
        return {}
    COLORS = {'light': {}, 'dark': {}}


# ============================================
# COLOR PALETTES
# ============================================

# Categorical colors - work well in both light and dark modes
CATEGORICAL_COLORS = [
    '#667eea',  # Purple-blue
    '#f093fb',  # Pink
    '#4facfe',  # Light blue
    '#00f2fe',  # Cyan
    '#43e97b',  # Green
    '#fa709a',  # Rose
    '#fee140',  # Yellow
    '#fa8142',  # Orange
    '#6a11cb',  # Deep purple
    '#2575fc',  # Blue
]

# Sequential color scales
SEQUENTIAL_POSITIVE = ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20']
SEQUENTIAL_NEGATIVE = ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350', '#f44336', '#e53935', '#d32f2f', '#c62828', '#b71c1c']

# Diverging scale for positive/negative values (red-white-green)
DIVERGING_RWG = [
    [0.0, '#d62728'],   # Red (negative)
    [0.25, '#ff7f0e'],  # Orange
    [0.5, '#f0f0f0'],   # White/neutral
    [0.75, '#98df8a'],  # Light green
    [1.0, '#2ca02c'],   # Green (positive)
]

# Grade colors (A-F)
GRADE_COLORS = {
    'A': '#2ca02c',  # Green
    'B': '#98df8a',  # Light green
    'C': '#ffbb78',  # Orange-yellow
    'D': '#ff7f0e',  # Orange
    'F': '#d62728',  # Red
}

# Regret tier colors
REGRET_COLORS = {
    'No Regret': '#2ca02c',
    'Minor Regret': '#98df8a',
    'Some Regret': '#ffbb78',
    'Big Regret': '#ff7f0e',
    'Major Regret': '#d62728',
    'Disaster': '#8b0000',
}

# FAAB value tier colors
FAAB_TIER_COLORS = {
    'Steal': '#0d47a1',
    'Great Value': '#1f77b4',
    'Good Value': '#2ca02c',
    'Fair': '#ffbb78',
    'Overpay': '#d62728',
}


# ============================================
# THEME CONFIGURATIONS
# ============================================

def get_plotly_template(theme: str = None) -> str:
    """
    Get the appropriate Plotly template name for the theme.

    Args:
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        Plotly template name string
    """
    if theme is None:
        theme = detect_theme()

    return 'plotly_dark' if theme == 'dark' else 'plotly_white'


def get_chart_colors(theme: str = None) -> Dict[str, Any]:
    """
    Get theme-aware color configurations for charts.

    Args:
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        Dict with color configurations:
            - categorical: List of colors for categorical data
            - sequential_positive: Scale for positive values
            - sequential_negative: Scale for negative values
            - diverging: Scale for positive/negative diverging data
            - grades: Dict mapping grades to colors
            - regret: Dict mapping regret tiers to colors
            - faab_tiers: Dict mapping FAAB tiers to colors
            - text: Text color
            - grid: Grid line color
            - background: Background color
    """
    if theme is None:
        theme = detect_theme()

    if theme == 'dark':
        return {
            'categorical': CATEGORICAL_COLORS,
            'sequential_positive': SEQUENTIAL_POSITIVE,
            'sequential_negative': SEQUENTIAL_NEGATIVE,
            'diverging': DIVERGING_RWG,
            'grades': GRADE_COLORS,
            'regret': REGRET_COLORS,
            'faab_tiers': FAAB_TIER_COLORS,
            'text': '#FAFAFA',
            'text_secondary': '#BDBDBD',
            'grid': '#404040',
            'background': 'rgba(0,0,0,0)',  # Transparent
            'paper': '#0E1117',
            'positive': '#00D26A',
            'negative': '#FF4B4B',
            'neutral': '#808080',
        }
    else:  # light
        return {
            'categorical': CATEGORICAL_COLORS,
            'sequential_positive': SEQUENTIAL_POSITIVE,
            'sequential_negative': SEQUENTIAL_NEGATIVE,
            'diverging': DIVERGING_RWG,
            'grades': GRADE_COLORS,
            'regret': REGRET_COLORS,
            'faab_tiers': FAAB_TIER_COLORS,
            'text': '#262730',
            'text_secondary': '#555555',
            'grid': '#E0E0E0',
            'background': 'rgba(0,0,0,0)',  # Transparent
            'paper': '#FFFFFF',
            'positive': '#00C07F',
            'negative': '#FF4B4B',
            'neutral': '#888888',
        }


def get_chart_theme(theme: str = None) -> Dict[str, Any]:
    """
    Get complete chart theme configuration as a dict for Plotly.

    Args:
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        Dict with Plotly layout configuration
    """
    if theme is None:
        theme = detect_theme()

    colors = get_chart_colors(theme)

    return {
        'template': get_plotly_template(theme),
        'paper_bgcolor': colors['background'],
        'plot_bgcolor': colors['background'],
        'font': {
            'color': colors['text'],
            'family': 'Source Sans Pro, sans-serif',
        },
        'title': {
            'font': {
                'color': colors['text'],
                'size': 16,
            }
        },
        'xaxis': {
            'gridcolor': colors['grid'],
            'tickfont': {'color': colors['text_secondary']},
            'title': {'font': {'color': colors['text']}},
        },
        'yaxis': {
            'gridcolor': colors['grid'],
            'tickfont': {'color': colors['text_secondary']},
            'title': {'font': {'color': colors['text']}},
        },
        'legend': {
            'font': {'color': colors['text_secondary']},
            'bgcolor': 'rgba(0,0,0,0)',
        },
        'colorway': colors['categorical'],
    }


def apply_chart_theme(fig: go.Figure, theme: str = None) -> go.Figure:
    """
    Apply theme styling to an existing Plotly figure.

    Args:
        fig: Plotly figure to style
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        The styled figure (modified in place and returned)
    """
    if theme is None:
        theme = detect_theme()

    theme_config = get_chart_theme(theme)

    # Apply layout updates
    fig.update_layout(
        template=theme_config['template'],
        paper_bgcolor=theme_config['paper_bgcolor'],
        plot_bgcolor=theme_config['plot_bgcolor'],
        font=theme_config['font'],
        xaxis=theme_config['xaxis'],
        yaxis=theme_config['yaxis'],
        legend=theme_config['legend'],
        colorway=theme_config['colorway'],
    )

    return fig


def create_grade_bar_chart(grade_counts: Dict[str, int], title: str = "Grade Distribution",
                           theme: str = None) -> go.Figure:
    """
    Create a themed bar chart for grade distribution.

    Args:
        grade_counts: Dict mapping grades (A-F) to counts
        title: Chart title
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        Themed Plotly figure
    """
    colors = get_chart_colors(theme)
    grades = ['A', 'B', 'C', 'D', 'F']
    counts = [grade_counts.get(g, 0) for g in grades]
    bar_colors = [colors['grades'][g] for g in grades]

    fig = go.Figure(go.Bar(
        x=grades,
        y=counts,
        marker_color=bar_colors,
        text=counts,
        textposition='outside',
        textfont={'color': colors['text']},
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Grade",
        yaxis_title="Count",
        showlegend=False,
        height=350,
    )

    return apply_chart_theme(fig, theme)


def create_regret_bar_chart(regret_counts: Dict[str, int], title: str = "Drop Regret Distribution",
                            theme: str = None) -> go.Figure:
    """
    Create a themed bar chart for drop regret distribution.

    Args:
        regret_counts: Dict mapping regret tiers to counts
        title: Chart title
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        Themed Plotly figure
    """
    colors = get_chart_colors(theme)
    tier_order = ['No Regret', 'Minor Regret', 'Some Regret', 'Big Regret', 'Major Regret', 'Disaster']
    tiers = [t for t in tier_order if t in regret_counts]
    counts = [regret_counts.get(t, 0) for t in tiers]
    bar_colors = [colors['regret'].get(t, '#808080') for t in tiers]

    fig = go.Figure(go.Bar(
        x=tiers,
        y=counts,
        marker_color=bar_colors,
        text=counts,
        textposition='outside',
        textfont={'color': colors['text']},
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Regret Level",
        yaxis_title="Count",
        showlegend=False,
        height=350,
        xaxis_tickangle=-45,
    )

    return apply_chart_theme(fig, theme)


def create_faab_tier_chart(tier_counts: Dict[str, int], title: str = "FAAB Value Distribution",
                           theme: str = None) -> go.Figure:
    """
    Create a themed bar chart for FAAB value tier distribution.

    Args:
        tier_counts: Dict mapping FAAB tiers to counts
        title: Chart title
        theme: 'light' or 'dark'. If None, auto-detects.

    Returns:
        Themed Plotly figure
    """
    colors = get_chart_colors(theme)
    tier_order = ['Steal', 'Great Value', 'Good Value', 'Fair', 'Overpay']
    tiers = [t for t in tier_order if t in tier_counts]
    counts = [tier_counts.get(t, 0) for t in tiers]
    bar_colors = [colors['faab_tiers'].get(t, '#808080') for t in tiers]

    fig = go.Figure(go.Bar(
        x=tiers,
        y=counts,
        marker_color=bar_colors,
        text=counts,
        textposition='outside',
        textfont={'color': colors['text']},
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Value Tier",
        yaxis_title="Count",
        showlegend=False,
        height=350,
    )

    return apply_chart_theme(fig, theme)


def create_horizontal_bar_chart(labels: List[str], values: List[float],
                                title: str = "", theme: str = None,
                                color_by_value: bool = True) -> go.Figure:
    """
    Create a themed horizontal bar chart (good for rankings).

    Args:
        labels: Y-axis labels (e.g., manager names)
        values: Bar values
        title: Chart title
        theme: 'light' or 'dark'. If None, auto-detects.
        color_by_value: If True, color bars green (positive) or red (negative)

    Returns:
        Themed Plotly figure
    """
    colors = get_chart_colors(theme)

    if color_by_value:
        bar_colors = [colors['positive'] if v > 0 else colors['negative'] for v in values]
    else:
        bar_colors = colors['categorical'][0]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.1f}" for v in values],
        textposition='outside',
        textfont={'color': colors['text']},
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        height=max(350, len(labels) * 35),
    )

    # Add zero line if we have positive and negative values
    if color_by_value and any(v > 0 for v in values) and any(v < 0 for v in values):
        fig.add_vline(x=0, line_dash="solid", line_color=colors['text_secondary'], line_width=2)

    return apply_chart_theme(fig, theme)


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def get_grade_colors_list() -> List[str]:
    """Get grade colors as a list in A-F order."""
    return [GRADE_COLORS[g] for g in ['A', 'B', 'C', 'D', 'F']]


def get_regret_colors_list() -> List[str]:
    """Get regret tier colors as a list in severity order."""
    order = ['No Regret', 'Minor Regret', 'Some Regret', 'Big Regret', 'Major Regret', 'Disaster']
    return [REGRET_COLORS[t] for t in order]


def get_faab_tier_colors_list() -> List[str]:
    """Get FAAB tier colors as a list in value order."""
    order = ['Steal', 'Great Value', 'Good Value', 'Fair', 'Overpay']
    return [FAAB_TIER_COLORS[t] for t in order]
