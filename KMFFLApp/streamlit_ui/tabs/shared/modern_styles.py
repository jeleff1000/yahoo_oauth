"""
Global modern CSS styles for the KMFFL application.

Key principles:
- Static elements (info displays, stat boxes) have NO shadows or hover effects
- Only interactive elements (buttons, clickable cards, links) have shadows and hover states
- Clean, minimal borders - 1px max, subtle colors
- Generous whitespace instead of heavy visual separators
- Subtle gradients only on main heroes, not everywhere

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
       HERO SECTIONS - Subtle gradient
       =========================================== */
    .hero-section {
        background: linear-gradient(135deg,
            var(--gradient-start, rgba(102, 126, 234, 0.08)) 0%,
            var(--gradient-end, rgba(118, 75, 162, 0.05)) 100%);
        padding: var(--space-lg, 1.5rem);
        border-radius: var(--radius-md, 8px);
        border-bottom: 1px solid var(--border, #E5E7EB);
        margin-bottom: var(--space-lg, 1.5rem);
    }
    .hero-section h1,
    .hero-section h2,
    .hero-section h3 {
        color: var(--text-primary, #1F2937) !important;
        margin-top: 0;
    }
    .hero-section p {
        color: var(--text-secondary, #6B7280);
        line-height: 1.6;
    }

    /* ===========================================
       SECTION HEADERS - Clean accent underline
       =========================================== */
    .section-header-title {
        color: var(--text-primary, #1F2937);
        font-size: 1.25rem;
        font-weight: 600;
        padding-bottom: var(--space-sm, 0.5rem);
        border-bottom: 2px solid var(--accent, #667eea);
        margin-bottom: var(--space-md, 1rem);
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
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
        /* NO shadow, NO hover transforms */
    }

    .feature-card .feature-icon {
        font-size: 1.5rem;
        margin-bottom: var(--space-xs, 0.25rem);
    }
    .feature-card .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary, #1F2937);
        margin-bottom: var(--space-xs, 0.25rem);
    }
    .feature-card .feature-desc {
        color: var(--text-secondary, #6B7280);
        line-height: 1.5;
        font-size: 0.9rem;
    }

    /* ===========================================
       INTERACTIVE CARDS - Has shadow and hover
       For navigation, clickable items ONLY
       =========================================== */
    .interactive-card,
    .nav-card,
    .clickable-card {
        background: var(--bg-primary, #FFFFFF);
        border: 1px solid var(--border, #E5E7EB);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
        box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,0.05));
        cursor: pointer;
        transition: box-shadow var(--transition-normal, 0.2s ease),
                    border-color var(--transition-normal, 0.2s ease);
    }
    .interactive-card:hover,
    .nav-card:hover,
    .clickable-card:hover {
        box-shadow: var(--shadow-md, 0 2px 4px rgba(0,0,0,0.08));
        border-color: var(--accent, #667eea);
    }

    /* ===========================================
       STATUS BOXES - Subtle background tint
       =========================================== */
    .info-box {
        background: var(--info-bg, rgba(59, 130, 246, 0.1));
        border: 1px solid var(--info, #3B82F6);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
        color: var(--text-primary, #1F2937);
    }
    .success-box {
        background: var(--success-bg, rgba(16, 185, 129, 0.1));
        border: 1px solid var(--success, #10B981);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
        color: var(--text-primary, #1F2937);
    }
    .warning-box {
        background: var(--warning-bg, rgba(245, 158, 11, 0.1));
        border: 1px solid var(--warning, #F59E0B);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
        color: var(--text-primary, #1F2937);
    }
    .error-box {
        background: var(--error-bg, rgba(239, 68, 68, 0.1));
        border: 1px solid var(--error, #EF4444);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
        color: var(--text-primary, #1F2937);
    }

    /* ===========================================
       STREAMLIT NATIVE ALERTS (st.info, st.warning, etc.)
       Override for better dark mode support
       =========================================== */
    /* Base alert styling */
    .stAlert {
        border-radius: var(--radius-md, 8px) !important;
        border-width: 1px !important;
        border-style: solid !important;
    }

    /* Info alert */
    .stAlert[data-baseweb="notification"][kind="info"],
    div[data-testid="stNotificationContentInfo"] {
        background-color: var(--info-bg, rgba(59, 130, 246, 0.1)) !important;
        border-color: var(--info, #3B82F6) !important;
    }
    .stAlert[data-baseweb="notification"][kind="info"] *,
    div[data-testid="stNotificationContentInfo"] * {
        color: var(--text-primary, #F9FAFB) !important;
    }

    /* Warning alert */
    .stAlert[data-baseweb="notification"][kind="warning"],
    div[data-testid="stNotificationContentWarning"] {
        background-color: var(--warning-bg, rgba(245, 158, 11, 0.1)) !important;
        border-color: var(--warning, #F59E0B) !important;
    }
    .stAlert[data-baseweb="notification"][kind="warning"] *,
    div[data-testid="stNotificationContentWarning"] * {
        color: var(--text-primary, #F9FAFB) !important;
    }

    /* Success alert */
    .stAlert[data-baseweb="notification"][kind="positive"],
    div[data-testid="stNotificationContentSuccess"] {
        background-color: var(--success-bg, rgba(16, 185, 129, 0.1)) !important;
        border-color: var(--success, #10B981) !important;
    }
    .stAlert[data-baseweb="notification"][kind="positive"] *,
    div[data-testid="stNotificationContentSuccess"] * {
        color: var(--text-primary, #F9FAFB) !important;
    }

    /* Error alert */
    .stAlert[data-baseweb="notification"][kind="negative"],
    div[data-testid="stNotificationContentError"] {
        background-color: var(--error-bg, rgba(239, 68, 68, 0.1)) !important;
        border-color: var(--error, #EF4444) !important;
    }
    .stAlert[data-baseweb="notification"][kind="negative"] *,
    div[data-testid="stNotificationContentError"] * {
        color: var(--text-primary, #F9FAFB) !important;
    }

    /* Generic Streamlit callout/alert overrides */
    [data-testid="stAlert"],
    .element-container .stAlert {
        border-radius: var(--radius-md, 8px) !important;
        padding: var(--space-md, 1rem) !important;
        margin: var(--space-sm, 0.5rem) 0 !important;
    }

    /* Ensure text is readable in dark mode */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    .stAlert p,
    .stAlert span {
        color: var(--text-primary, #F9FAFB) !important;
    }

    /* Remove heavy left border accent */
    [data-testid="stAlert"]::before,
    .stAlert > div:first-child {
        display: none !important;
    }

    /* ===========================================
       METRIC/STAT DISPLAY - Clean and minimal
       =========================================== */
    .stat-display {
        text-align: center;
        padding: var(--space-md, 1rem);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary, #1F2937);
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.875rem;
        color: var(--text-muted, #9CA3AF);
        margin-top: var(--space-xs, 0.25rem);
    }

    /* ===========================================
       STREAMLIT NATIVE TABS (section tabs like Matchup Stats, etc.)
       =========================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--border, rgba(255,255,255,0.1));
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.25rem;
        padding: 0 0.75rem;
        border-radius: 4px 4px 0 0;
        font-weight: 500;
        font-size: 0.85rem;
        background-color: transparent;
        color: rgba(255, 255, 255, 0.55);
        border: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: all 0.15s ease;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        color: #EF4444;
        border-bottom: 2px solid #EF4444;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: rgba(255, 255, 255, 0.85);
        background-color: rgba(255, 255, 255, 0.05);
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    }

    /* Force consistent text in tabs */
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    /* ===========================================
       BUTTONS - Interactive, has hover
       =========================================== */
    .stButton > button {
        border-radius: var(--radius-md, 8px);
        font-weight: 600;
        transition: all var(--transition-normal, 0.2s ease);
    }
    .stButton > button:hover {
        box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,0.05));
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: var(--accent, #667eea);
        border-color: var(--accent, #667eea);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover, #5a6fd6);
    }

    /* ===========================================
       DATAFRAMES & TABLES
       =========================================== */
    .stDataFrame {
        border-radius: var(--radius-md, 8px);
        overflow: hidden;
        border: 1px solid var(--border, #E5E7EB);
    }

    /* ===========================================
       RADIO BUTTONS - Cleaner look
       =========================================== */
    .stRadio > div {
        gap: var(--space-sm, 0.5rem);
    }
    .stRadio label {
        padding: var(--space-sm, 0.5rem) var(--space-md, 1rem);
        border-radius: var(--radius-sm, 4px);
        border: 1px solid var(--border, #E5E7EB);
        background: var(--bg-primary, #FFFFFF);
        cursor: pointer;
        transition: all var(--transition-fast, 0.1s ease);
    }
    .stRadio label:hover {
        border-color: var(--accent, #667eea);
    }
    .stRadio label[data-checked="true"] {
        background: var(--accent-subtle, rgba(102, 126, 234, 0.08));
        border-color: var(--accent, #667eea);
        color: var(--accent, #667eea);
    }

    /* ===========================================
       EXPANDERS
       =========================================== */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: var(--radius-md, 8px);
        background: var(--bg-secondary, #F8F9FA);
        border: 1px solid var(--border, #E5E7EB);
    }

    /* ===========================================
       REMOVE CLUTTER - Override existing styles
       =========================================== */
    /* Remove left accent borders from old styles */
    .feature-card,
    .info-box,
    .success-box,
    .warning-box,
    .metric-card,
    .section-card {
        border-left: 1px solid var(--border, #E5E7EB) !important;
    }

    /* Remove hover transforms from static elements */
    .feature-card:hover,
    .stat-card:hover,
    .info-card:hover {
        transform: none !important;
        box-shadow: none !important;
    }

    /* ===========================================
       RESPONSIVE - TABLET/MOBILE (768px)
       =========================================== */
    @media (max-width: 768px) {
        /* Tighter page margins */
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 0.75rem !important;
        }

        .hero-section {
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .hero-section h1 { font-size: 1.4rem !important; }
        .hero-section h2 { font-size: 1.2rem !important; }
        .hero-section p { font-size: 0.85rem; margin-top: 0.25rem; }

        .static-card,
        .feature-card,
        .interactive-card {
            padding: 0.5rem;
            margin: 0.25rem 0;
        }

        .info-box, .success-box, .warning-box, .error-box {
            padding: 0.5rem;
            font-size: 0.85rem;
        }

        /* Horizontal scrollable section tabs */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: nowrap;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            gap: 0.25rem;
            padding-bottom: 0.25rem;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            height: 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 1.75rem;
            padding: 0 0.5rem;
            font-size: 0.7rem;
            flex-shrink: 0;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(239, 68, 68, 0.15) !important;
        }

        /* Tighter expanders */
        .streamlit-expanderHeader {
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stExpander"] > div > div {
            padding: 0.5rem !important;
        }

        /* Touch-friendly inputs */
        .stMultiSelect [data-baseweb="select"],
        .stSelectbox select,
        .stTextInput input,
        .stNumberInput input {
            min-height: 40px;
            font-size: 16px;
        }

        /* Compact typography */
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        h4 { font-size: 0.9rem !important; }

        /* Scrollable tables */
        .stDataFrame {
            overflow-x: auto;
        }
        .stDataFrame table { font-size: 0.8rem; }
        .stDataFrame th { padding: 0.3rem !important; font-size: 0.75rem; }
        .stDataFrame td { padding: 0.25rem !important; }

        .stat-value { font-size: 1.4rem; }

        /* Reduce vertical gaps */
        .element-container { margin-bottom: 0.5rem !important; }
        .stMarkdown { margin-bottom: 0.25rem !important; }
    }

    /* ===========================================
       RESPONSIVE - SMALL MOBILE (480px)
       =========================================== */
    @media (max-width: 480px) {
        .main .block-container {
            padding-left: 0.35rem !important;
            padding-right: 0.35rem !important;
        }

        .hero-section {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .hero-section h1 { font-size: 1.2rem !important; }
        .hero-section h2 { font-size: 1rem !important; }

        .static-card,
        .feature-card,
        .interactive-card {
            padding: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 1.5rem;
            padding: 0 0.4rem;
            font-size: 0.65rem;
        }

        h1 { font-size: 1.2rem !important; }
        h2 { font-size: 1rem !important; }
        h3 { font-size: 0.9rem !important; }

        .stDataFrame table { font-size: 0.7rem; }
        .stDataFrame th { font-size: 0.65rem; padding: 0.2rem !important; }
        .stDataFrame td { padding: 0.15rem !important; }

        .stat-value { font-size: 1.2rem; }
    }

    /* ===========================================
       TOUCH DEVICES
       =========================================== */
    @media (hover: none) and (pointer: coarse) {
        button, a, input, select {
            min-height: 44px !important;
        }

        /* No hover effects on touch */
        .interactive-card:hover,
        .nav-card:hover {
            box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,0.05));
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
