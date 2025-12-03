"""
Global modern CSS styles for the KMFFL application.

Key principles:
- Reduced vertical spacing for tighter, more modern look
- Static elements (info displays, stat boxes) have NO shadows or hover effects
- Only interactive elements (buttons, clickable cards, links) have shadows and hover states
- Clean, minimal borders - 1px max, subtle colors
- Generous whitespace instead of heavy visual separators
- Consistent light/dark mode support via CSS variables and media queries

Usage:
    from streamlit_ui.tabs.shared.modern_styles import apply_modern_styles
    apply_modern_styles()
"""

import streamlit as st


def apply_modern_styles():
    """Apply modern, decluttered CSS styling to Streamlit pages."""
    st.markdown("""
    <style>
    /* ===========================================
       CSS VARIABLES - Light Mode (default)
       =========================================== */
    :root {
        /* Backgrounds */
        --bg-primary: #FFFFFF;
        --bg-secondary: #F8F9FA;
        --bg-tertiary: #F0F2F6;
        --bg-card: #FFFFFF;

        /* Text */
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --text-muted: #9CA3AF;

        /* Borders */
        --border: #E5E7EB;
        --border-subtle: #F0F0F0;
        --border-strong: #D1D5DB;

        /* Accent (purple theme) */
        --accent: #667eea;
        --accent-hover: #5a6fd6;
        --accent-subtle: rgba(102, 126, 234, 0.08);
        --accent-light: rgba(102, 126, 234, 0.15);

        /* Gradients */
        --gradient-start: rgba(102, 126, 234, 0.08);
        --gradient-end: rgba(118, 75, 162, 0.05);

        /* Status colors */
        --success: #10B981;
        --success-bg: rgba(16, 185, 129, 0.1);
        --warning: #F59E0B;
        --warning-bg: rgba(245, 158, 11, 0.1);
        --error: #EF4444;
        --error-bg: rgba(239, 68, 68, 0.1);
        --info: #3B82F6;
        --info-bg: rgba(59, 130, 246, 0.1);

        /* Table colors */
        --table-header-bg: #F3F4F6;
        --table-row-even: #F9FAFB;
        --table-row-hover: #F3F4F6;
        --table-border: #E5E7EB;

        /* Spacing - REDUCED for tighter layout */
        --space-xs: 0.25rem;
        --space-sm: 0.375rem;
        --space-md: 0.625rem;
        --space-lg: 1rem;
        --space-xl: 1.25rem;
        --space-xxl: 1.5rem;

        /* Radius */
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;

        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 2px 4px rgba(0,0,0,0.08);
        --shadow-lg: 0 4px 8px rgba(0,0,0,0.1);

        /* Transitions */
        --transition-fast: 0.1s ease;
        --transition-normal: 0.2s ease;
    }

    /* ===========================================
       CSS VARIABLES - Dark Mode
       =========================================== */
    @media (prefers-color-scheme: dark) {
        :root {
            /* Backgrounds */
            --bg-primary: #0E1117;
            --bg-secondary: #1F2937;
            --bg-tertiary: #374151;
            --bg-card: #1F2937;

            /* Text */
            --text-primary: #F9FAFB;
            --text-secondary: #D1D5DB;
            --text-muted: #9CA3AF;

            /* Borders */
            --border: #374151;
            --border-subtle: #2D3748;
            --border-strong: #4B5563;

            /* Accent (lighter for dark mode) */
            --accent: #818CF8;
            --accent-hover: #6366F1;
            --accent-subtle: rgba(129, 140, 248, 0.12);
            --accent-light: rgba(129, 140, 248, 0.2);

            /* Gradients */
            --gradient-start: rgba(102, 126, 234, 0.15);
            --gradient-end: rgba(118, 75, 162, 0.1);

            /* Status colors - slightly more visible in dark */
            --success-bg: rgba(16, 185, 129, 0.15);
            --warning-bg: rgba(245, 158, 11, 0.15);
            --error-bg: rgba(239, 68, 68, 0.15);
            --info-bg: rgba(59, 130, 246, 0.15);

            /* Table colors */
            --table-header-bg: #374151;
            --table-row-even: #252629;
            --table-row-hover: #2D3748;
            --table-border: #4B5563;

            /* Shadows - more visible in dark mode */
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.2);
            --shadow-md: 0 2px 4px rgba(0,0,0,0.25);
            --shadow-lg: 0 4px 8px rgba(0,0,0,0.3);
        }
    }

    /* ===========================================
       GLOBAL SPACING OVERRIDES - Tight, consistent layout
       =========================================== */
    /* FIX: Consistent left margin for ALL content */
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* CRITICAL: Force ALL elements to same left edge (0 offset from container) */
    .stTabs,
    .stTabs > div,
    [data-testid="stExpander"],
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"],
    .tab-header,
    [data-testid="stMarkdown"],
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-baseweb="tab-panel"],
    .element-container {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    /* Remove any nested indentation */
    [data-testid="stVerticalBlock"] > div {
        padding-left: 0 !important;
        margin-left: 0 !important;
    }

    /* Reduce spacing between ALL stacked elements */
    .element-container {
        margin-bottom: 0.125rem !important;
    }

    /* Very tight heading margins */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }

    /* Tighter paragraph spacing */
    p {
        margin-bottom: 0.25rem !important;
    }

    /* Reduce markdown divider spacing */
    hr {
        margin: 0.5rem 0 !important;
    }

    /* ===========================================
       CRITICAL: Reduce gaps between major sections
       =========================================== */
    /* Target Streamlit's vertical block gaps directly */
    [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }

    [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.25rem !important;
    }

    /* TOP TABS to FILTERS gap */
    .stTabs {
        margin-bottom: 0.5rem !important;
    }

    /* FILTERS (expander) spacing */
    [data-testid="stExpander"] {
        margin-top: 0.25rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* SUBTABS spacing (nested tabs) */
    .stTabs .stTabs {
        margin-top: 0.25rem !important;
    }

    /* Tab content area */
    [data-testid="stTabContent"] {
        padding-top: 0.25rem !important;
    }

    /* Reduce ALL gaps in the main content area */
    .main [data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 0.25rem !important;
    }

    /* Target the specific gap after tabs */
    [data-baseweb="tab-panel"] {
        padding-top: 0.25rem !important;
        margin-top: 0 !important;
    }

    /* Reduce space around markdown elements */
    [data-testid="stMarkdown"] {
        margin-bottom: 0.25rem !important;
    }

    /* Target column gaps */
    [data-testid="column"] {
        padding: 0 0.25rem !important;
    }

    /* ===========================================
       HERO SECTIONS - Subtle gradient
       =========================================== */
    .hero-section {
        background: linear-gradient(135deg,
            var(--gradient-start) 0%,
            var(--gradient-end) 100%);
        padding: var(--space-lg);
        border-radius: var(--radius-lg);
        border-bottom: 1px solid var(--border);
        margin-bottom: var(--space-lg);
    }
    .hero-section h1,
    .hero-section h2,
    .hero-section h3 {
        color: var(--text-primary) !important;
        margin-top: 0 !important;
        margin-bottom: var(--space-xs) !important;
    }
    .hero-section p {
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: 0 !important;
    }

    /* ===========================================
       TAB HEADERS - Section titles, pulled up tight
       =========================================== */
    .tab-header {
        margin-top: 0 !important;
        margin-bottom: 0.25rem;
        padding-top: 0 !important;
        padding-bottom: 0.125rem;
        border-bottom: 1px solid var(--border);
    }
    .tab-header h2 {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2;
    }
    .tab-header p {
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2;
    }

    /* Pull section content up after subtabs */
    [data-baseweb="tab-panel"] > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* ===========================================
       SECTION HEADERS - Clean accent underline
       =========================================== */
    .section-header-title {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        padding-bottom: var(--space-xs);
        border-bottom: 2px solid var(--accent);
        margin-bottom: var(--space-md);
        display: inline-block;
    }

    /* ===========================================
       STATIC CARDS - No shadows, no hover
       For displaying info, stats, data - NOT clickable
       =========================================== */
    .static-card,
    .info-card,
    .stat-card,
    .feature-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-xs) 0;
        /* NO shadow, NO hover transforms */
    }

    .feature-card .feature-icon {
        font-size: 1.25rem;
        margin-bottom: var(--space-xs);
    }
    .feature-card .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--space-xs);
    }
    .feature-card .feature-desc {
        color: var(--text-secondary);
        line-height: 1.4;
        font-size: 0.85rem;
    }

    /* ===========================================
       FILTER CARD - Styled container for filters
       =========================================== */
    .filter-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        margin: var(--space-md) 0;
    }
    .filter-card-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-md);
        padding-bottom: var(--space-sm);
        border-bottom: 1px solid var(--border);
    }
    .filter-card-title h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 !important;
    }

    /* Style Streamlit's native expander for filters - COMPACT */
    div[data-testid="stExpander"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        margin-bottom: 0.375rem !important;
        margin-top: 0.25rem !important;
    }

    /* Expander header - compact with title bar feel */
    div[data-testid="stExpander"] > div:first-child {
        padding: 0.375rem 0.75rem !important;
        background: linear-gradient(to bottom, var(--bg-tertiary), var(--bg-secondary));
        border-bottom: 1px solid var(--border);
        border-radius: var(--radius-md) var(--radius-md) 0 0;
    }

    /* Expander header text */
    div[data-testid="stExpander"] summary {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* Expander content - reduced padding */
    div[data-testid="stExpander"] > div:last-child {
        padding: 0.5rem 0.75rem !important;
    }

    /* Reduce internal spacing in expanders */
    div[data-testid="stExpander"] .stMultiSelect,
    div[data-testid="stExpander"] .stSelectbox,
    div[data-testid="stExpander"] .stCheckbox {
        margin-bottom: 0.25rem !important;
    }

    div[data-testid="stExpander"] [data-testid="column"] {
        padding: 0 0.25rem !important;
    }

    /* ===========================================
       INTERACTIVE CARDS - Has shadow and hover
       For navigation, clickable items ONLY
       =========================================== */
    .interactive-card,
    .nav-card,
    .clickable-card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-xs) 0;
        box-shadow: var(--shadow-sm);
        cursor: pointer;
        transition: box-shadow var(--transition-normal),
                    border-color var(--transition-normal);
    }
    .interactive-card:hover,
    .nav-card:hover,
    .clickable-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--accent);
    }

    /* ===========================================
       STATUS BOXES - Themed message boxes
       =========================================== */
    .info-box,
    .theme-info-box {
        background: var(--info-bg);
        border: 1px solid var(--info);
        border-left: 3px solid var(--info);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .success-box,
    .theme-success,
    .theme-success-box {
        background: var(--success-bg);
        border: 1px solid var(--success);
        border-left: 3px solid var(--success);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .warning-box,
    .theme-warning,
    .theme-warning-box {
        background: var(--warning-bg);
        border: 1px solid var(--warning);
        border-left: 3px solid var(--warning);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .error-box,
    .theme-error,
    .theme-error-box {
        background: var(--error-bg);
        border: 1px solid var(--error);
        border-left: 3px solid var(--error);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        color: var(--text-primary);
        font-size: 0.9rem;
    }

    /* Loading indicator */
    .theme-loading {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: var(--space-sm) var(--space-md);
        border-radius: var(--radius-md);
        font-weight: 500;
        background: var(--warning-bg);
        border-left: 3px solid var(--warning);
        color: var(--text-primary);
        font-size: 0.9rem;
    }

    /* Stats count badge */
    .theme-stats-count {
        padding: var(--space-sm) var(--space-md);
        border-radius: var(--radius-md);
        font-weight: 500;
        display: inline-block;
        margin: var(--space-sm) 0;
        background: var(--info-bg);
        color: var(--text-primary);
        border: 1px solid var(--info);
        font-size: 0.85rem;
    }

    /* ===========================================
       STREAMLIT NATIVE ALERTS (st.info, st.warning, etc.)
       Override for better dark mode support
       =========================================== */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border-width: 1px !important;
        border-style: solid !important;
        margin: var(--space-sm) 0 !important;
        padding: var(--space-md) !important;
    }

    /* Info alert */
    .stAlert[data-baseweb="notification"][kind="info"],
    div[data-testid="stNotificationContentInfo"] {
        background-color: var(--info-bg) !important;
        border-color: var(--info) !important;
    }
    .stAlert[data-baseweb="notification"][kind="info"] *,
    div[data-testid="stNotificationContentInfo"] * {
        color: var(--text-primary) !important;
    }

    /* Warning alert */
    .stAlert[data-baseweb="notification"][kind="warning"],
    div[data-testid="stNotificationContentWarning"] {
        background-color: var(--warning-bg) !important;
        border-color: var(--warning) !important;
    }
    .stAlert[data-baseweb="notification"][kind="warning"] *,
    div[data-testid="stNotificationContentWarning"] * {
        color: var(--text-primary) !important;
    }

    /* Success alert */
    .stAlert[data-baseweb="notification"][kind="positive"],
    div[data-testid="stNotificationContentSuccess"] {
        background-color: var(--success-bg) !important;
        border-color: var(--success) !important;
    }
    .stAlert[data-baseweb="notification"][kind="positive"] *,
    div[data-testid="stNotificationContentSuccess"] * {
        color: var(--text-primary) !important;
    }

    /* Error alert */
    .stAlert[data-baseweb="notification"][kind="negative"],
    div[data-testid="stNotificationContentError"] {
        background-color: var(--error-bg) !important;
        border-color: var(--error) !important;
    }
    .stAlert[data-baseweb="notification"][kind="negative"] *,
    div[data-testid="stNotificationContentError"] * {
        color: var(--text-primary) !important;
    }

    /* Generic Streamlit callout/alert overrides */
    [data-testid="stAlert"],
    .element-container .stAlert {
        border-radius: var(--radius-md) !important;
        padding: var(--space-md) !important;
        margin: var(--space-sm) 0 !important;
    }

    /* Ensure text is readable */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    .stAlert p,
    .stAlert span {
        color: var(--text-primary) !important;
    }

    /* Remove heavy left border accent from native alerts */
    [data-testid="stAlert"]::before,
    .stAlert > div:first-child {
        display: none !important;
    }

    /* ===========================================
       METRIC/STAT DISPLAY - Clean and minimal
       =========================================== */
    .stat-display {
        text-align: center;
        padding: var(--space-md);
    }
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: var(--space-xs);
    }

    /* Metric cards */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        text-align: center;
    }
    .metric-card-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-bottom: var(--space-xs);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-card-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: var(--space-xs);
    }
    .metric-card-delta.positive {
        color: var(--success);
    }
    .metric-card-delta.negative {
        color: var(--error);
    }

    /* Section cards */
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        margin: var(--space-md) 0;
    }

    /* Gradient header */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: var(--space-lg);
        border-radius: var(--radius-lg);
        margin: var(--space-md) 0;
    }
    .gradient-header h2 {
        margin: 0 !important;
        font-size: 1.4rem;
        font-weight: 600;
        color: white !important;
    }
    .gradient-header p {
        margin: var(--space-xs) 0 0 0 !important;
        opacity: 0.95;
        font-size: 0.9rem;
        color: white !important;
    }

    /* ===========================================
       STREAMLIT NATIVE TABS - Compact navigation style
       =========================================== */
    /* Container for tabs */
    .stTabs {
        margin-bottom: 0.375rem !important;
    }

    /* Tab list container - minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.375rem;
        border-bottom: 1px solid var(--border);
        padding: 0 0 0.375rem 0;
        margin-bottom: 0.25rem;
        flex-wrap: wrap;
        background: transparent;
        border-radius: 0;
        border-left: none;
        border-right: none;
        border-top: none;
    }

    /* Individual tabs - SMALL, navigation-like */
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-weight: 500;
        font-size: 0.75rem;
        background-color: var(--bg-secondary);
        color: var(--text-secondary);
        border: 1px solid var(--border-subtle);
        margin-bottom: 0;
        transition: all var(--transition-fast);
        white-space: nowrap;
    }

    /* Selected tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--accent);
        color: white;
        border-color: var(--accent);
        font-weight: 600;
    }

    /* Hover state */
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: var(--text-primary);
        background-color: var(--bg-tertiary);
        border-color: var(--border);
    }

    /* Force consistent text in tabs */
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    /* Tab panel - minimal top spacing */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.25rem !important;
        margin-top: 0 !important;
    }

    /* ===========================================
       DATAFRAMES & TABLES - IMPROVED
       =========================================== */
    .stDataFrame {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--table-border);
    }

    /* Table headers - BOLDER and larger */
    .stDataFrame thead tr th {
        background-color: var(--table-header-bg) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 0.625rem 0.5rem !important;
        border-bottom: 2px solid var(--border-strong) !important;
        text-align: left !important;
    }

    /* Table rows - increased height and alternating colors */
    .stDataFrame tbody tr {
        background-color: var(--bg-primary) !important;
        transition: background-color var(--transition-fast);
    }

    .stDataFrame tbody tr:nth-child(even) {
        background-color: var(--table-row-even) !important;
    }

    .stDataFrame tbody tr:hover {
        background-color: var(--table-row-hover) !important;
    }

    /* Table cells - better padding */
    .stDataFrame tbody td {
        padding: 0.5rem !important;
        font-size: 0.85rem !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* ===========================================
       BUTTONS - Interactive, has hover
       =========================================== */
    .stButton > button {
        border-radius: var(--radius-md);
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        transition: all var(--transition-normal);
    }
    .stButton > button:hover {
        box-shadow: var(--shadow-sm);
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
    }

    /* ===========================================
       RADIO BUTTONS & TOGGLES - Pill style
       =========================================== */
    .stRadio > div {
        gap: var(--space-sm);
        flex-wrap: wrap;
    }
    .stRadio label {
        padding: 0.375rem 0.75rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
        background: var(--bg-primary);
        cursor: pointer;
        transition: all var(--transition-fast);
        font-size: 0.85rem;
    }
    .stRadio label:hover {
        border-color: var(--accent);
        background: var(--accent-subtle);
    }
    .stRadio label[data-checked="true"] {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
    }

    /* ===========================================
       SELECT BOXES & DROPDOWNS
       =========================================== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: var(--radius-md) !important;
        border-color: var(--border) !important;
        background: var(--bg-primary) !important;
    }

    .stSelectbox label,
    .stMultiSelect label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        margin-bottom: var(--space-xs) !important;
    }

    /* ===========================================
       EXPANDERS
       =========================================== */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: var(--radius-md);
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        padding: var(--space-sm) var(--space-md) !important;
    }

    .streamlit-expanderContent {
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
        padding: var(--space-md) !important;
    }

    /* ===========================================
       LEGEND / ACCURACY TIERS BOX
       =========================================== */
    .legend-box,
    .accuracy-legend {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        display: inline-block;
    }
    .legend-box h4,
    .accuracy-legend h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-xs) 0 !important;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        margin-right: var(--space-md);
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    /* ===========================================
       POSITIVE/NEGATIVE VALUE INDICATORS
       =========================================== */
    .value-positive {
        color: var(--success) !important;
        font-weight: 600;
    }
    .value-negative {
        color: var(--error) !important;
        font-weight: 600;
    }
    .value-neutral {
        color: var(--text-muted);
    }

    /* Small indicator dots (replaces loud checkmarks) */
    .indicator-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }
    .indicator-dot.yes {
        background: var(--success);
    }
    .indicator-dot.no {
        background: var(--error);
    }

    /* ===========================================
       REMOVE CLUTTER - Override existing styles
       =========================================== */
    /* Remove left accent borders from old styles */
    .feature-card,
    .metric-card,
    .section-card {
        border-left: 1px solid var(--border) !important;
    }

    /* Remove hover transforms from static elements */
    .feature-card:hover,
    .stat-card:hover,
    .info-card:hover,
    .metric-card:hover {
        transform: none !important;
        box-shadow: none !important;
    }

    /* ===========================================
       EMPTY STATE
       =========================================== */
    .empty-state {
        text-align: center;
        padding: 2rem;
        background: var(--bg-secondary);
        border: 2px dashed var(--border);
        border-radius: var(--radius-lg);
        margin: var(--space-lg) 0;
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: var(--space-sm);
        opacity: 0.5;
    }
    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: var(--space-xs);
    }
    .empty-state-message {
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    /* ===========================================
       ABOUT PAGE - Centered card
       =========================================== */
    .about-content {
        max-width: 800px;
        margin: 0 auto;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
    }
    .about-content h1,
    .about-content h2,
    .about-content h3 {
        color: var(--text-primary);
    }
    .about-content p,
    .about-content li {
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* ===========================================
       RESPONSIVE - TABLET (768px)
       =========================================== */
    @media (max-width: 768px) {
        .hero-section {
            padding: var(--space-md);
            margin-bottom: var(--space-md);
        }
        .hero-section h1 { font-size: 1.3rem !important; }
        .hero-section h2 { font-size: 1.15rem !important; }
        .hero-section p { font-size: 0.85rem; }

        .static-card,
        .feature-card,
        .interactive-card,
        .filter-card {
            padding: var(--space-md);
            margin: var(--space-xs) 0;
        }

        .info-box, .success-box, .warning-box, .error-box,
        .theme-info-box, .theme-success, .theme-warning, .theme-error {
            padding: var(--space-sm);
            font-size: 0.85rem;
        }

        /* Full width buttons */
        .stButton > button {
            width: 100%;
        }

        /* Better touch targets */
        .stMultiSelect [data-baseweb="select"],
        .stSelectbox select,
        .stTextInput input,
        .stNumberInput input {
            min-height: 44px;
            font-size: 16px; /* Prevents iOS zoom */
        }

        /* Reduce font sizes */
        h1 { font-size: 1.3rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1rem !important; }
        h4 { font-size: 0.9rem !important; }

        .stat-value { font-size: 1.3rem; }
        .metric-card-value { font-size: 1.25rem; }

        .gradient-header {
            padding: var(--space-md);
        }
        .gradient-header h2 {
            font-size: 1.2rem;
        }
    }

    /* ===========================================
       RESPONSIVE - MOBILE (600px) - KEY MOBILE FIXES
       =========================================== */
    @media (max-width: 600px) {
        /* ===== FIX #1: TOP NAV BUTTONS AS 2x2 GRID ===== */
        /* Make the columns container wrap into 2x2 grid */
        [data-testid="stHorizontalBlock"]:first-of-type,
        .main [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:first-child {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 6px !important;
        }

        /* Each column takes 50% width minus gap */
        [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"],
        .main [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:first-child > [data-testid="column"] {
            flex: 0 0 calc(50% - 3px) !important;
            width: calc(50% - 3px) !important;
            min-width: calc(50% - 3px) !important;
            max-width: calc(50% - 3px) !important;
            padding: 0 !important;
        }

        /* Compact buttons inside */
        button[kind="primary"],
        button[kind="secondary"],
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-secondary"],
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"],
        .stButton button,
        .stButton > button,
        [data-testid="column"] button,
        [data-testid="stHorizontalBlock"] button {
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
            min-height: unset !important;
            height: auto !important;
            line-height: 1.3 !important;
        }

        /* Other horizontal blocks (not nav) - keep normal */
        [data-testid="stHorizontalBlock"]:not(:first-of-type) {
            gap: 4px !important;
        }
        [data-testid="stHorizontalBlock"]:not(:first-of-type) > [data-testid="column"] {
            padding: 0 2px !important;
        }

        /* ===== FIX #2: REDUCE FILTERS BLOCK - MINIMAL ===== */
        div[data-testid="stExpander"] {
            margin: 4px 0 !important;
        }
        /* Expander header - super compact */
        div[data-testid="stExpander"] > div:first-child,
        [data-testid="stExpanderToggleIcon"],
        .streamlit-expanderHeader {
            padding: 4px 8px !important;
            min-height: unset !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] span {
            font-size: 0.75rem !important;
        }
        /* Expander content - tight */
        div[data-testid="stExpander"] > div:last-child {
            padding: 6px 8px !important;
        }

        /* ===== FIX #3: HORIZONTAL SCROLLING SUBTABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            -ms-overflow-style: none;
            gap: 4px !important;
            padding: 0 0 4px 0 !important;
            margin-bottom: 4px !important;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            flex-shrink: 0 !important;
            white-space: nowrap !important;
            padding: 4px 8px !important;
            font-size: 0.7rem !important;
            min-height: unset !important;
            height: auto !important;
        }

        /* ===== FIX #4: TABLE HORIZONTAL SCROLL ===== */
        .stDataFrame,
        [data-testid="stDataFrame"],
        [data-testid="stDataFrameResizable"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
            max-width: 100% !important;
        }
        .stDataFrame table,
        [data-testid="stDataFrame"] table {
            display: block !important;
            overflow-x: auto !important;
            white-space: nowrap;
            font-size: 0.7rem !important;
        }
        .stDataFrame th,
        [data-testid="stDataFrame"] th {
            padding: 4px 6px !important;
            font-size: 0.65rem !important;
        }
        .stDataFrame td,
        [data-testid="stDataFrame"] td {
            padding: 3px 5px !important;
            font-size: 0.65rem !important;
        }

        /* ===== FIX #5: KILL ALL EXCESS SPACING ===== */
        .tab-header {
            margin: 4px 0 !important;
            padding: 2px 0 !important;
        }
        .tab-header h2 {
            font-size: 0.8rem !important;
            margin: 0 !important;
        }
        .tab-header p {
            font-size: 0.6rem !important;
            margin: 0 !important;
        }

        /* Container - minimal padding */
        .main .block-container {
            padding: 4px 8px 8px 8px !important;
        }

        /* Kill ALL vertical gaps */
        [data-testid="stVerticalBlock"] {
            gap: 2px !important;
        }
        [data-testid="stVerticalBlock"] > div {
            margin-bottom: 2px !important;
        }
        .element-container {
            margin-bottom: 1px !important;
        }

        /* Remove spacer divs */
        hr {
            margin: 4px 0 !important;
        }

        /* Tighten markdown spacing */
        [data-testid="stMarkdown"] {
            margin-bottom: 2px !important;
        }
        [data-testid="stMarkdown"] p {
            margin-bottom: 2px !important;
        }
    }

    /* ===========================================
       RESPONSIVE - SMALL MOBILE (480px)
       =========================================== */
    @media (max-width: 480px) {
        /* Even smaller buttons */
        button[kind="primary"],
        button[kind="secondary"],
        .stButton button {
            padding: 4px 6px !important;
            font-size: 0.7rem !important;
        }

        /* Tiny tabs */
        .stTabs [data-baseweb="tab"] {
            padding: 3px 6px !important;
            font-size: 0.65rem !important;
        }

        /* Super compact expander */
        div[data-testid="stExpander"] > div:first-child {
            padding: 3px 6px !important;
        }
        div[data-testid="stExpander"] summary {
            font-size: 0.7rem !important;
        }

        /* Minimal content padding */
        .main .block-container {
            padding: 2px 4px 4px 4px !important;
        }

        /* Tiny fonts */
        h1 { font-size: 1rem !important; }
        h2 { font-size: 0.9rem !important; }
        h3 { font-size: 0.8rem !important; }

        .tab-header h2 { font-size: 0.75rem !important; }
        .tab-header p { font-size: 0.55rem !important; }

        /* Tiny table */
        .stDataFrame table { font-size: 0.6rem !important; }
        .stDataFrame th { font-size: 0.55rem !important; padding: 2px 4px !important; }
        .stDataFrame td { padding: 2px 3px !important; font-size: 0.55rem !important; }

        .stat-value { font-size: 1rem; }
        .metric-card-value { font-size: 0.95rem; }

        .hero-section { padding: 4px; }

        /* Stack filter toggles vertically */
        .stRadio > div {
            flex-direction: column;
        }
    }

    /* ===========================================
       TOUCH DEVICES
       =========================================== */
    @media (hover: none) and (pointer: coarse) {
        button, a, input, select, .stRadio label {
            min-height: 44px !important;
        }

        /* No hover effects on touch */
        .interactive-card:hover,
        .nav-card:hover {
            box-shadow: var(--shadow-sm);
        }
    }

    /* ===========================================
       PRINT STYLES
       =========================================== */
    @media print {
        .hero-section { background: none; border: 1px solid #ccc; }
        .stTabs, .stButton { display: none; }
        * { box-shadow: none !important; }
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS - Consolidated from tab-specific theme files
# =============================================================================

def render_info_box(message: str, icon: str = "") -> None:
    """Render a theme-aware info box."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="theme-info-box">
        <p style="margin: 0; font-size: 0.9rem;">{icon_html}{message}</p>
    </div>
    """, unsafe_allow_html=True)


def render_success_box(message: str, icon: str = "") -> None:
    """Render a theme-aware success box."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="theme-success">
        <p style="margin: 0; font-size: 0.9rem;">{icon_html}{message}</p>
    </div>
    """, unsafe_allow_html=True)


def render_warning_box(message: str, icon: str = "") -> None:
    """Render a theme-aware warning box."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="theme-warning">
        <p style="margin: 0; font-size: 0.9rem;">{icon_html}{message}</p>
    </div>
    """, unsafe_allow_html=True)


def render_error_box(message: str, icon: str = "") -> None:
    """Render a theme-aware error box."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="theme-error">
        <p style="margin: 0; font-size: 0.9rem;">{icon_html}{message}</p>
    </div>
    """, unsafe_allow_html=True)


def render_loading_indicator(message: str = "Loading...") -> None:
    """Render a theme-aware loading indicator."""
    st.markdown(f"""
    <div class="theme-loading">
        {message}
    </div>
    """, unsafe_allow_html=True)


def render_stats_count(filtered_count: int, total_count: int) -> None:
    """Render a theme-aware statistics count display."""
    percentage = (filtered_count / total_count * 100) if total_count > 0 else 0

    if filtered_count < total_count:
        message = f"Showing {filtered_count:,} of {total_count:,} matchups ({percentage:.1f}%)"
    else:
        message = f"Showing all {total_count:,} matchups"

    st.markdown(f"""
    <div class="theme-stats-count">
        {message}
    </div>
    """, unsafe_allow_html=True)


def render_gradient_header(title: str, subtitle: str = None, icon: str = None) -> None:
    """Render a gradient header with optional subtitle."""
    icon_html = f"{icon} " if icon else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""

    st.markdown(f"""
    <div class="gradient-header">
        <h2>{icon_html}{title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_card(content: str) -> None:
    """Render a section card with themed background."""
    st.markdown(f"""
    <div class="section-card">
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value, delta: float = None, delta_label: str = None) -> None:
    """Render a styled metric card."""
    delta_class = "positive" if delta and delta > 0 else "negative" if delta and delta < 0 else ""
    delta_icon = "+" if delta and delta > 0 else "" if delta and delta < 0 else ""
    delta_text = delta_label if delta_label else f"{delta_icon}{delta:.1f}" if delta else ""

    delta_html = f"""
    <div class="metric-card-delta {delta_class}">
        {delta_text}
    </div>
    """ if delta is not None else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(
    title: str = "No Data Available",
    message: str = "Try adjusting your filters or selection.",
    icon: str = ""
) -> None:
    """Render a styled empty state."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-message">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_filter_count(filtered_count: int, total_count: int) -> None:
    """Render a badge showing filter results count."""
    percentage = (filtered_count / total_count * 100) if total_count > 0 else 0
    st.markdown(f"""
    <span style="font-size: 0.9rem;">
        Showing <strong>{filtered_count:,}</strong> of <strong>{total_count:,}</strong> records
        <span class="theme-stats-count" style="padding: 0.125rem 0.5rem; margin-left: 0.25rem;">{percentage:.0f}%</span>
    </span>
    """, unsafe_allow_html=True)


def render_legend_box(title: str, items: list) -> None:
    """
    Render a styled legend box.

    Args:
        title: Legend title
        items: List of (label, color) tuples or (label, emoji) tuples
    """
    items_html = ""
    for item in items:
        if len(item) == 2:
            label, indicator = item
            if indicator.startswith('#') or indicator.startswith('rgb'):
                # It's a color
                items_html += f'<span class="legend-item"><span style="display:inline-block;width:12px;height:12px;background:{indicator};border-radius:2px;"></span> {label}</span>'
            else:
                # It's an emoji or text
                items_html += f'<span class="legend-item">{indicator} {label}</span>'

    st.markdown(f"""
    <div class="legend-box">
        <h4>{title}</h4>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


def format_value_with_color(value: float, format_str: str = "{:.2f}") -> str:
    """
    Format a value with positive/negative color class.

    Returns HTML string with appropriate CSS class.
    """
    formatted = format_str.format(value)
    if value > 0:
        return f'<span class="value-positive">+{formatted}</span>'
    elif value < 0:
        return f'<span class="value-negative">{formatted}</span>'
    else:
        return f'<span class="value-neutral">{formatted}</span>'
