"""
Modern responsive CSS framework for team stats.

Provides comprehensive mobile-first responsive design with:
- Multi-tier breakpoint system (desktop/tablet/phone/landscape/touch)
- Touch-friendly 44px minimum targets
- Horizontal scrolling optimizations
- Compact typography for small screens
- Smooth transitions and animations
"""

import streamlit as st


def apply_modern_styles():
    """
    Apply modern responsive styles for team stats components.

    Includes:
    - Responsive breakpoints for all screen sizes
    - Touch-optimized interface elements
    - Mobile-first table styling
    - Tab navigation optimizations
    - Button and input enhancements
    """
    st.markdown("""
    <style>
    /* ============================================================
       BASE RESPONSIVE FRAMEWORK
       ============================================================ */

    * {
        box-sizing: border-box;
    }

    .main .block-container {
        max-width: 100%;
        padding-left: 5%;
        padding-right: 5%;
    }

    /* ============================================================
       HERO/HEADER SECTIONS - Responsive
       ============================================================ */

    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    .hero-section h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .hero-section p {
        margin: 0.75rem 0 0 0;
        opacity: 0.95;
        font-size: 1rem;
    }

    @media (max-width: 768px) {
        .hero-section {
            padding: 1.5rem;
        }
        .hero-section h2 {
            font-size: 1.5rem;
        }
        .hero-section p {
            font-size: 0.9rem;
        }
    }

    @media (max-width: 480px) {
        .hero-section {
            padding: 1rem;
        }
        .hero-section h2 {
            font-size: 1.3rem;
        }
        .hero-section p {
            font-size: 0.85rem;
        }
    }

    /* ============================================================
       FEATURE CARDS - Responsive
       ============================================================ */

    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        margin: 1rem 0;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }

    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: linear-gradient(145deg, #2b2d31 0%, #1e1f22 100%);
            color: #f0f0f0;
        }
    }

    @media (max-width: 768px) {
        .feature-card {
            padding: 1.25rem;
        }
    }

    @media (max-width: 480px) {
        .feature-card {
            padding: 1rem;
        }
    }

    /* ============================================================
       TAB STYLING - Horizontal Scrolling Support
       ============================================================ */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #7c8df0 0%, #8a5fb0 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #5a67d8 0%, #6a3f94 100%);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Mobile: Horizontal scrolling for tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
            flex-wrap: nowrap !important;
        }

        .stTabs [data-baseweb="tab"] {
            flex-shrink: 0;
            padding: 0.6rem 1.2rem;
            font-size: 0.9rem;
        }
    }

    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
        }
    }

    /* ============================================================
       RADIO BUTTONS - Enhanced Styling
       ============================================================ */

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: rgb(70, 73, 80) !important;
        color: rgb(255, 255, 255) !important;
        border-radius: 0.5rem !important;
        padding: 0.6rem 1.2rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border: 2px solid transparent !important;
        font-weight: 500 !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background-color: rgb(85, 88, 95) !important;
        transform: translateY(-1px);
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: rgb(103, 108, 245) !important;
        border-color: rgb(129, 140, 248) !important;
        box-shadow: 0 0 0 3px rgba(103, 108, 245, 0.2);
    }

    /* Hide the radio circle */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    @media (max-width: 480px) {
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            padding: 0.5rem 0.9rem !important;
            font-size: 0.85rem !important;
        }
    }

    /* ============================================================
       BUTTONS - Mobile Optimization
       ============================================================ */

    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        padding: 0.5rem 1.5rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
            padding: 0.6rem 1rem;
        }
    }

    /* ============================================================
       DATAFRAME/TABLE - Mobile Responsive
       ============================================================ */

    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }

    /* Horizontal scrolling for tables on mobile */
    @media (max-width: 768px) {
        .stDataFrame {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        .stDataFrame table {
            font-size: 0.85rem;
            min-width: 600px; /* Ensures table doesn't squish */
        }

        .stDataFrame thead tr th {
            padding: 0.5rem 0.75rem !important;
            white-space: nowrap;
        }

        .stDataFrame tbody tr td {
            padding: 0.5rem 0.75rem !important;
        }
    }

    @media (max-width: 480px) {
        .stDataFrame table {
            font-size: 0.8rem;
        }

        .stDataFrame thead tr th,
        .stDataFrame tbody tr td {
            padding: 0.4rem 0.6rem !important;
        }
    }

    /* ============================================================
       COLUMNS - Stack on Mobile
       ============================================================ */

    @media (max-width: 768px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }

        .row-widget.stHorizontal > div {
            width: 100% !important;
        }
    }

    /* ============================================================
       SELECTBOX & MULTISELECT - Mobile Optimization
       ============================================================ */

    .stSelectbox, .stMultiSelect {
        font-size: 0.95rem;
    }

    @media (max-width: 480px) {
        .stSelectbox, .stMultiSelect {
            font-size: 0.85rem;
        }

        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            padding: 0.5rem;
        }
    }

    /* ============================================================
       NUMBER INPUT - Compact on Mobile
       ============================================================ */

    @media (max-width: 480px) {
        .stNumberInput > div > div > input {
            font-size: 0.9rem;
            padding: 0.5rem;
        }
    }

    /* ============================================================
       EXPANDER - Smooth Transitions
       ============================================================ */

    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 0.5rem;
        transition: background-color 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(0, 0, 0, 0.05);
    }

    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
    }

    @media (max-width: 480px) {
        .streamlit-expanderHeader {
            font-size: 0.9rem;
        }
    }

    /* ============================================================
       PLOTLY CHARTS - Responsive
       ============================================================ */

    .js-plotly-plot {
        border-radius: 0.5rem;
        overflow: hidden;
    }

    @media (max-width: 768px) {
        .js-plotly-plot .plotly {
            /* Plotly handles its own responsive behavior */
            min-height: 350px;
        }
    }

    @media (max-width: 480px) {
        .js-plotly-plot .plotly {
            min-height: 300px;
        }
    }

    /* ============================================================
       TOUCH DEVICE OPTIMIZATIONS
       ============================================================ */

    @media (hover: none) and (pointer: coarse) {
        /* Increase touch targets to minimum 44px */
        button, a, input, select,
        .stButton > button,
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            min-height: 44px !important;
            min-width: 44px !important;
        }

        /* Larger tap targets for checkboxes and radio */
        input[type="checkbox"],
        input[type="radio"] {
            width: 24px;
            height: 24px;
        }

        /* Prevent double-tap zoom on buttons */
        button, .stButton > button {
            touch-action: manipulation;
        }

        /* Smooth scrolling for touch */
        * {
            -webkit-overflow-scrolling: touch;
        }
    }

    /* ============================================================
       LANDSCAPE MODE (TABLETS)
       ============================================================ */

    @media (max-width: 768px) and (orientation: landscape) {
        .hero-section {
            padding: 1rem 2rem;
        }

        .hero-section h2 {
            font-size: 1.4rem;
        }

        .hero-section p {
            font-size: 0.9rem;
        }

        /* Compact vertical spacing */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    }

    /* ============================================================
       CUSTOM SCROLLBARS (WEBKIT)
       ============================================================ */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    @media (prefers-color-scheme: dark) {
        ::-webkit-scrollbar-track {
            background: #2b2d31;
        }

        ::-webkit-scrollbar-thumb {
            background: #555;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
    }

    /* ============================================================
       LOADING SKELETON
       ============================================================ */

    .loading-skeleton {
        background: linear-gradient(
            90deg,
            #f0f0f0 25%,
            #e0e0e0 50%,
            #f0f0f0 75%
        );
        background-size: 200% 100%;
        animation: loading 1.5s ease-in-out infinite;
        border-radius: 0.5rem;
        height: 20px;
        margin: 0.5rem 0;
    }

    @media (prefers-color-scheme: dark) {
        .loading-skeleton {
            background: linear-gradient(
                90deg,
                #2b2d31 25%,
                #3a3c41 50%,
                #2b2d31 75%
            );
            background-size: 200% 100%;
        }
    }

    @keyframes loading {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }

    /* ============================================================
       METRIC DISPLAY - Responsive Grid
       ============================================================ */

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }

    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 480px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }

    /* ============================================================
       TYPOGRAPHY - Responsive Scaling
       ============================================================ */

    @media (max-width: 768px) {
        h1 {
            font-size: 1.75rem !important;
        }

        h2 {
            font-size: 1.5rem !important;
        }

        h3 {
            font-size: 1.25rem !important;
        }

        p, div, span {
            font-size: 0.95rem;
        }
    }

    @media (max-width: 480px) {
        h1 {
            font-size: 1.5rem !important;
        }

        h2 {
            font-size: 1.3rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        p, div, span {
            font-size: 0.9rem;
        }
    }

    /* ============================================================
       PRINT STYLES
       ============================================================ */

    @media print {
        .stButton, .stDownloadButton {
            display: none;
        }

        .hero-section {
            background: white;
            color: black;
            border: 1px solid #ccc;
        }

        .feature-card {
            break-inside: avoid;
        }

        .stDataFrame {
            font-size: 10pt;
        }
    }

    </style>
    """, unsafe_allow_html=True)


def apply_compact_mode():
    """
    Apply extra compact styling for dense information display.
    Useful for tables with many columns or data-heavy views.
    """
    st.markdown("""
    <style>
    .compact-mode .stDataFrame table {
        font-size: 0.8rem;
    }

    .compact-mode .stDataFrame thead tr th,
    .compact-mode .stDataFrame tbody tr td {
        padding: 0.4rem 0.6rem !important;
    }

    .compact-mode .stSelectbox,
    .compact-mode .stMultiSelect {
        font-size: 0.85rem;
    }

    .compact-mode h2 {
        font-size: 1.4rem !important;
    }

    .compact-mode h3 {
        font-size: 1.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
