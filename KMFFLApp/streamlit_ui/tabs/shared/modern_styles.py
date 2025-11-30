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
    }
    .success-box {
        background: var(--success-bg, rgba(16, 185, 129, 0.1));
        border: 1px solid var(--success, #10B981);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
    }
    .warning-box {
        background: var(--warning-bg, rgba(245, 158, 11, 0.1));
        border: 1px solid var(--warning, #F59E0B);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
    }
    .error-box {
        background: var(--error-bg, rgba(239, 68, 68, 0.1));
        border: 1px solid var(--error, #EF4444);
        border-radius: var(--radius-md, 8px);
        padding: var(--space-md, 1rem);
        margin: var(--space-sm, 0.5rem) 0;
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
       STREAMLIT NATIVE TABS
       =========================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--space-sm, 0.5rem);
        border-bottom: 1px solid var(--border, #E5E7EB);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.75rem;
        padding: 0 var(--space-md, 1rem);
        border-radius: var(--radius-sm, 4px) var(--radius-sm, 4px) 0 0;
        font-weight: 500;
        background-color: transparent;
        color: var(--text-secondary, #6B7280);
        border: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        color: var(--accent, #667eea);
        border-bottom: 2px solid var(--accent, #667eea);
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: var(--text-primary, #1F2937);
        background-color: var(--hover, #F5F5F5);
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
       RESPONSIVE - TABLET (768px)
       =========================================== */
    @media (max-width: 768px) {
        .hero-section {
            padding: var(--space-md, 1rem);
            margin-bottom: var(--space-md, 1rem);
        }
        .hero-section h1 { font-size: 1.5rem !important; }
        .hero-section h2 { font-size: 1.3rem !important; }
        .hero-section p { font-size: 0.9rem; }

        .static-card,
        .feature-card,
        .interactive-card {
            padding: var(--space-sm, 0.5rem);
            margin: var(--space-xs, 0.25rem) 0;
        }

        .info-box, .success-box, .warning-box, .error-box {
            padding: var(--space-sm, 0.5rem);
            font-size: 0.9rem;
        }

        /* Scrollable tabs */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: nowrap;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.5rem;
            padding: 0 var(--space-sm, 0.5rem);
            font-size: 0.85rem;
            flex-shrink: 0;
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
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        h4 { font-size: 1rem !important; }

        /* Scrollable tables */
        .stDataFrame {
            overflow-x: auto;
        }
        .stDataFrame table { font-size: 0.85rem; }
        .stDataFrame th { padding: 0.4rem !important; }
        .stDataFrame td { padding: 0.3rem !important; }

        .stat-value { font-size: 1.5rem; }
    }

    /* ===========================================
       RESPONSIVE - MOBILE (480px)
       =========================================== */
    @media (max-width: 480px) {
        .hero-section {
            padding: var(--space-sm, 0.5rem);
        }
        .hero-section h1 { font-size: 1.3rem !important; }
        .hero-section h2 { font-size: 1.1rem !important; }

        .static-card,
        .feature-card,
        .interactive-card {
            padding: var(--space-sm, 0.5rem);
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.25rem;
            padding: 0 var(--space-xs, 0.25rem);
            font-size: 0.8rem;
        }

        h1 { font-size: 1.3rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1rem !important; }

        .stDataFrame table { font-size: 0.75rem; }
        .stDataFrame th { font-size: 0.7rem; padding: 0.25rem !important; }
        .stDataFrame td { padding: 0.2rem !important; }

        .stat-value { font-size: 1.3rem; }
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
