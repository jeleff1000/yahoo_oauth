"""
Shared modern CSS styles for consistent UI across all KMFFL pages.
Import this module and call apply_modern_styles() at the top of your page display function.
"""
import streamlit as st


def apply_modern_styles():
    """Apply modern, consistent CSS styling to Streamlit pages."""
    st.markdown("""
    <style>
    /* Hero/Header Sections */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero-section h1, .hero-section h2, .hero-section h3 {
        color: white !important;
        margin-top: 0;
    }
    .hero-section p {
        opacity: 0.95;
        line-height: 1.6;
    }

    /* Tab Headers */
    .tab-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tab-header h2, .tab-header h3 {
        margin: 0;
        color: white !important;
    }
    .tab-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Feature Cards */
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    .feature-icon {
        font-size: 2rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .feature-desc {
        color: #666;
        line-height: 1.5;
    }

    /* Dark mode for feature cards */
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: linear-gradient(145deg, #2b2d31 0%, #1e1f22 100%);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            border-left: 4px solid #7289da;
        }
        .feature-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        }
        .feature-title {
            color: #f0f0f0;
        }
        .feature-desc {
            color: #b0b0b0;
        }
    }

    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
        border-left: 4px solid #2196F3;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #f1f8f4 0%, #e8f5e9 100%);
        border-left: 4px solid #4CAF50;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #fff9c4 100%);
        border-left: 4px solid #ff9800;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-card h4 {
        margin-top: 0;
        color: #667eea;
    }

    /* Section Cards */
    .section-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s;
        color: #333;
    }
    .section-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    .section-card p {
        color: #666;
        line-height: 1.6;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .section-icon {
        font-size: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
    }

    /* Dark mode for section cards */
    @media (prefers-color-scheme: dark) {
        .section-card {
            background: #2b2d31;
            border: 2px solid #3a3c41;
            color: #f0f0f0;
        }
        .section-card:hover {
            border-color: #7289da;
            box-shadow: 0 4px 12px rgba(114, 137, 218, 0.25);
        }
        .section-card p {
            color: #b0b0b0;
        }
        .section-title {
            color: #f0f0f0;
        }
    }

    /* ============================================
       NATIVE STREAMLIT TABS (st.tabs)
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    /* Inactive tabs - light mode */
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        background-color: rgb(70, 73, 80);
        color: rgb(255, 255, 255);
        border: 2px solid rgb(100, 103, 110);
    }

    /* Active/selected tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }

    /* Force white text in tab content */
    .stTabs [data-baseweb="tab"] * {
        color: rgb(255, 255, 255) !important;
    }

    /* Hover state for inactive tabs */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgb(80, 83, 90);
        border-color: rgb(103, 108, 245);
    }

    /* ============================================
       RADIO BUTTON TABS (Subtabs throughout app)
       EXCLUDES popovers - they use clean list style
       ============================================ */

    /* Hide radio button circles - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* Reset radio button default styles - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        all: unset;
    }

    /* Style radio group container - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        gap: 0.5rem;
        background-color: transparent;
        padding: 0;
        margin: 0;
        flex-wrap: wrap;
        width: 100%;
    }

    /* Universal dark button style - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: rgb(70, 73, 80) !important;
        border: 2px solid rgb(100, 103, 110) !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.5rem !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        font-family: "Source Sans Pro", sans-serif !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: rgb(255, 255, 255) !important;
        line-height: 1.6 !important;
        white-space: nowrap !important;
        user-select: none !important;
        box-sizing: border-box !important;
    }

    /* Force white text in all children - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label * {
        color: rgb(255, 255, 255) !important;
    }

    /* Hover state - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: rgb(103, 108, 245) !important;
        background-color: rgb(80, 83, 90) !important;
    }

    /* Active/selected tab - bright purple - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: rgb(103, 108, 245) !important;
        border-color: rgb(103, 108, 245) !important;
        color: rgb(255, 255, 255) !important;
        font-weight: 600 !important;
    }

    /* Force active tab text color - except in popovers */
    :not([data-testid="stPopover"]) div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: rgb(255, 255, 255) !important;
    }

    /* Remove default radio container padding */
    div[data-testid="stRadio"] {
        margin-bottom: 0.5rem;
    }

    /* ============================================
       POPOVER MENU - Clean list style
       ============================================ */
    [data-testid="stPopover"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        flex-direction: column !important;
        gap: 0 !important;
    }
    [data-testid="stPopover"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: transparent !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 0.75rem !important;
        justify-content: flex-start !important;
        color: inherit !important;
        font-weight: normal !important;
    }
    [data-testid="stPopover"] div[data-testid="stRadio"] > div[role="radiogroup"] > label * {
        color: inherit !important;
    }
    [data-testid="stPopover"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(128, 128, 128, 0.15) !important;
    }
    [data-testid="stPopover"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(102, 126, 234, 0.2) !important;
        font-weight: 500 !important;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Remove extra padding from main container */
    .main > div {
        padding-top: 1rem;
    }

    /* Improve dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }

    /* ========================================
       MOBILE-RESPONSIVE STYLES
       ======================================== */

    /* Tablets and smaller (768px and below) */
    @media (max-width: 768px) {
        /* Radio button tabs - mobile optimized */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            padding: 0.375rem 0.625rem !important;
            font-size: 0.9rem !important;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.375rem !important;
        }

        /* Hero sections - reduce padding */
        .hero-section {
            padding: 1.5rem 1rem;
            margin-bottom: 1.5rem;
        }
        .hero-section h1 {
            font-size: 1.75rem !important;
        }
        .hero-section h2 {
            font-size: 1.5rem !important;
        }
        .hero-section p {
            font-size: 0.9rem;
        }

        /* Tab headers - reduce padding */
        .tab-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .tab-header h2 {
            font-size: 1.4rem !important;
        }
        .tab-header h3 {
            font-size: 1.2rem !important;
        }

        /* Feature cards - reduce padding and margins */
        .feature-card {
            padding: 1rem;
            margin: 0.75rem 0;
        }

        /* Info boxes - reduce padding */
        .info-box, .success-box, .warning-box {
            padding: 0.9rem;
            margin: 0.75rem 0;
            font-size: 0.9rem;
        }

        /* Metric cards - reduce padding */
        .metric-card {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }

        /* Section cards - reduce padding */
        .section-card {
            padding: 1rem;
            margin: 0.75rem 0;
        }

        /* Native tabs - make them scrollable horizontally */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            flex-wrap: nowrap;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.5rem !important;
            padding: 0 1rem !important;
            font-size: 0.85rem !important;
            min-width: fit-content;
            flex-shrink: 0;
        }

        /* Buttons - full width on mobile */
        .stButton > button {
            width: 100%;
            font-size: 0.9rem;
        }

        /* Columns - stack vertically */
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        .row-widget.stHorizontal > div {
            width: 100% !important;
            margin-bottom: 0.5rem;
        }

        /* Multiselect - better mobile touch targets */
        .stMultiSelect {
            font-size: 0.9rem;
        }
        .stMultiSelect [data-baseweb="select"] {
            min-height: 44px; /* Better touch target */
        }

        /* Selectbox - better mobile touch targets */
        .stSelectbox select {
            min-height: 44px;
            font-size: 0.9rem;
        }

        /* Text input - better mobile touch targets */
        .stTextInput input {
            min-height: 44px;
            font-size: 0.9rem;
        }

        /* Number input - better mobile touch targets */
        .stNumberInput input {
            min-height: 44px;
            font-size: 0.9rem;
        }

        /* Checkbox - larger touch targets */
        .stCheckbox {
            min-height: 44px;
            display: flex;
            align-items: center;
        }

        /* Dataframes - make them scrollable */
        .stDataFrame {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .stDataFrame table {
            font-size: 0.85rem;
        }
        .stDataFrame th {
            font-size: 0.8rem;
            padding: 0.5rem !important;
        }
        .stDataFrame td {
            padding: 0.4rem !important;
        }

        /* Metrics - reduce font sizes */
        .stMetric {
            font-size: 0.9rem;
        }
        .stMetric label {
            font-size: 0.85rem;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        /* Expanders - better mobile styling */
        .streamlit-expanderHeader {
            font-size: 0.95rem;
            padding: 0.75rem !important;
        }

        /* Reduce overall font sizes */
        h1 {
            font-size: 1.8rem !important;
        }
        h2 {
            font-size: 1.5rem !important;
        }
        h3 {
            font-size: 1.3rem !important;
        }
        h4 {
            font-size: 1.1rem !important;
        }
        p, div, span {
            font-size: 0.95rem;
        }
    }

    /* Mobile phones (480px and below) */
    @media (max-width: 480px) {
        /* Radio button tabs - compact for phones */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            padding: 0.3rem 0.5rem !important;
            font-size: 0.85rem !important;
            flex: 1 1 auto !important;
            min-width: fit-content !important;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.25rem !important;
        }

        /* Even more compact for phones */
        .hero-section {
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
        }
        .hero-section h1 {
            font-size: 1.5rem !important;
        }
        .hero-section h2 {
            font-size: 1.3rem !important;
        }

        .tab-header {
            padding: 0.75rem;
        }
        .tab-header h2 {
            font-size: 1.2rem !important;
        }
        .tab-header h3 {
            font-size: 1.1rem !important;
        }

        .feature-card, .section-card {
            padding: 0.75rem;
            margin: 0.5rem 0;
        }

        .info-box, .success-box, .warning-box {
            padding: 0.75rem;
            font-size: 0.85rem;
        }

        .metric-card {
            padding: 0.75rem;
        }

        /* Native tabs - even smaller for phones */
        .stTabs [data-baseweb="tab"] {
            height: 2.25rem !important;
            padding: 0 0.75rem !important;
            font-size: 0.8rem !important;
        }

        /* Further reduce font sizes */
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.15rem !important;
        }
        h4 {
            font-size: 1rem !important;
        }
        p, div, span {
            font-size: 0.9rem;
        }

        /* Dataframes - even smaller text */
        .stDataFrame table {
            font-size: 0.75rem;
        }
        .stDataFrame th {
            font-size: 0.7rem;
            padding: 0.3rem !important;
        }
        .stDataFrame td {
            padding: 0.25rem !important;
        }

        /* Metrics - compact */
        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
        .stMetric label {
            font-size: 0.8rem;
        }
    }

    /* Landscape mobile phones */
    @media (max-width: 768px) and (orientation: landscape) {
        .hero-section {
            padding: 1rem;
        }
        .hero-section h1, .hero-section h2 {
            font-size: 1.3rem !important;
        }
        .tab-header {
            padding: 0.75rem;
        }
    }

    /* Touch-friendly interactions for all mobile devices */
    @media (hover: none) and (pointer: coarse) {
        /* Larger touch targets */
        button, a, input, select {
            min-height: 44px !important;
            min-width: 44px !important;
        }

        /* Remove hover effects on touch devices */
        .feature-card:hover,
        .section-card:hover {
            transform: none;
        }

        /* Better spacing for touch */
        .stButton {
            margin: 0.5rem 0;
        }
    }

    /* Prevent text selection issues on mobile */
    @media (max-width: 768px) {
        * {
            -webkit-tap-highlight-color: transparent;
        }
    }

    /* Improve scrolling performance on mobile */
    @media (max-width: 768px) {
        .stDataFrame, .stTabs [data-baseweb="tab-list"] {
            -webkit-overflow-scrolling: touch;
            scroll-behavior: smooth;
        }
    }
    </style>
    """, unsafe_allow_html=True)
